"""Claude Agent SDK inference/agent backend for Hermes.

This adapter lets Hermes drive inference (and, optionally, whole agentic
turns) through Anthropic's **Claude Agent SDK** (`claude-agent-sdk`, import
name ``claude_agent_sdk``) instead of the native ``anthropic`` Messages
client.  The headline reason to pick this path is authentication: the SDK
spawns the bundled/installed ``claude`` CLI, which can authenticate with a
Claude Code / Claude Pro-Max **OAuth subscription token** (as well as a
plain API key).

The mode is selected via ``config.yaml``::

    model:
      claude_agent_sdk:
        mode: inference        # inference | delegate | hybrid
        permission_mode: bypassPermissions   # delegate/hybrid only
        max_turns: 20          # delegate/hybrid only

or, as a shorthand, ``model.claude_agent_sdk: inference``.

Three modes
-----------
``inference`` (default, safest)
    The SDK performs a single model call (``max_turns=1``, no tools) and
    returns the model's text.  Hermes keeps full ownership of its own agent
    loop, tools, soul, budget and safety wrapping — the SDK is purely an
    alternate *transport to Anthropic inference* (and its OAuth auth).  Note
    that because the SDK is a session-based agent rather than a stateless
    completion API, tool-calling turns are **not** routed through this path;
    it is for text turns and to unlock subscription billing.  Cross-turn
    context is preserved through the SDK's own session (``resume``).

``delegate``
    Hands the user's request to the SDK's own agent loop with the SDK's
    built-in tools (Read/Edit/Bash/Grep/…).  Effectively "run Claude Code
    from inside Hermes."  This is a distinct trust/permission surface — see
    ``permission_mode``.

``hybrid`` (experimental)
    The SDK drives the loop but calls **Hermes'** tools, exposed through an
    in-process MCP server built from the registry.  Agent-level tools that
    need live ``AIAgent`` state (todo/memory/clarify/delegate_task/…) are not
    exposed; results are wrapped with the same ``<untrusted_tool_result>``
    promptware defense the native path uses.

Design note
-----------
No new ``api_mode`` string is introduced.  When SDK mode is active the agent
keeps ``api_mode == "anthropic_messages"`` and sets ``_claude_agent_sdk_mode``.
``create_claude_agent_message`` returns an object shaped like a native
Anthropic ``Message`` (``.content`` blocks + ``.stop_reason`` + ``.usage``)
so the existing ``AnthropicTransport.normalize_response`` consumes it
unchanged — the hot agent loop needs no edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from agent.transports.hermes_tool_exposure import (
    looks_like_tool_error,
    normalize_tool_spec,
)

logger = logging.getLogger("run_agent")

# Valid sub-modes for model.claude_agent_sdk.mode
_VALID_MODES = ("inference", "delegate", "hybrid")

# In-process MCP server name used to expose Hermes tools in hybrid mode.
_HYBRID_SERVER = "hermes"

# Prefix on the display label of a tool run by a nested SDK subagent (spawned
# via the `Agent`/`Task` tool). Keeps subagent activity visible on UI/CLI
# consumers that only render the tool name; richer consumers also get a
# `subagent=True` kwarg.
_SUBAGENT_TOOL_MARKER = "↳ "

# Prefix on the display label of a background task / dynamic-workflow agent. The
# SDK reports these out-of-band from the message stream as Task* system messages
# (TaskStarted/Progress/Updated/Notification) rather than parent_tool_use_id
# stream events, so they get their own lifecycle surface (see _StreamBridge).
_TASK_MARKER = "⚙ "

# Task statuses that mean the task is finished (mirrors the SDK's
# TERMINAL_TASK_STATUSES; hard-coded so we don't depend on the constant existing
# on older SDKs).
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "killed", "stopped"})


# ---------------------------------------------------------------------------
# Lazy SDK import (mirrors agent/anthropic_adapter.py:_get_anthropic_sdk)
# ---------------------------------------------------------------------------
_claude_agent_sdk: Any = ...  # sentinel — None means "tried and missing"


def _get_claude_agent_sdk():
    """Return the ``claude_agent_sdk`` module, importing lazily. None if missing."""
    global _claude_agent_sdk
    if _claude_agent_sdk is ...:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("provider.claude_agent_sdk", prompt=False)
        except ImportError:
            pass
        except Exception:
            # FeatureUnavailable (lazy installs disabled) — fall through.
            pass
        try:
            import claude_agent_sdk as _sdk

            _claude_agent_sdk = _sdk
        except ImportError:
            _claude_agent_sdk = None
    return _claude_agent_sdk


def _require_sdk():
    sdk = _get_claude_agent_sdk()
    if sdk is None:
        raise RuntimeError(
            "claude-agent-sdk is not installed. Install it with:\n"
            "    uv pip install 'hermes-agent[claude-agent-sdk]'\n"
            "It also requires the `claude` CLI on PATH (npm i -g "
            "@anthropic-ai/claude-code, or the native installer)."
        )
    return sdk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def resolve_claude_agent_sdk_settings(provider: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read ``model.claude_agent_sdk`` from config; return normalized settings.

    Returns ``None`` (SDK mode disabled) when the key is unset/blank/"auto",
    when the provider is not native ``anthropic`` (the OAuth/subscription
    path only makes sense there), or when the value is malformed.

    Accepted config shapes::

        model.claude_agent_sdk: inference          # bare mode string
        model.claude_agent_sdk: {mode: hybrid, max_turns: 30}

    Returned dict keys: ``mode``, ``permission_mode``, ``max_turns``,
    ``allowed_tools``, ``cwd``, ``system_prompt_preset``, ``append_system_prompt``.
    """
    # Only the native Anthropic provider carries OAuth/subscription creds the
    # CLI understands; third-party Anthropic-compatible endpoints (MiniMax,
    # Kimi, …) use bearer/custom-header auth that the env-only CLI path cannot
    # express, so we deliberately do not enable SDK mode for them.
    if (provider or "").strip().lower() != "anthropic":
        return None

    try:
        from hermes_cli.config import load_config_readonly, cfg_get

        cfg = load_config_readonly()
    except Exception as exc:  # noqa: BLE001 — config is best-effort here
        logger.debug("claude_agent_sdk: config load failed (%s); mode disabled", exc)
        return None

    raw = cfg_get(cfg, "model", "claude_agent_sdk", default=None)
    if raw is None:
        return None

    mode: Optional[str] = None
    opts: Dict[str, Any] = {}
    if isinstance(raw, str):
        mode = raw.strip().lower()
    elif isinstance(raw, dict):
        opts = raw
        mode = str(raw.get("mode") or "inference").strip().lower()
        # An explicit `enabled: false` turns it off regardless of mode.
        if raw.get("enabled") is False:
            return None
    else:
        return None

    if mode in ("", "auto", "off", "none", "false", "disabled"):
        return None
    if mode not in _VALID_MODES:
        logger.warning(
            "claude_agent_sdk: unknown mode %r (expected one of %s); disabling.",
            mode, ", ".join(_VALID_MODES),
        )
        return None

    def _get(key, default):
        val = opts.get(key)
        return default if val is None else val

    # delegate/hybrid run autonomously (no human to answer prompts), so the
    # default permission mode auto-approves. The user opted into SDK mode
    # explicitly; document the trust surface. Override via config.
    default_perm = "bypassPermissions" if mode in ("delegate", "hybrid") else "dontAsk"

    max_budget = _get("max_budget_usd", None)
    settings = {
        "mode": mode,
        "permission_mode": str(_get("permission_mode", default_perm)),
        "max_turns": int(_get("max_turns", 1 if mode == "inference" else 24)),
        "allowed_tools": list(_get("allowed_tools", []) or []),
        "disallowed_tools": list(_get("disallowed_tools", []) or []),
        "max_budget_usd": float(max_budget) if max_budget is not None else None,
        "cwd": _get("cwd", None),
        "system_prompt_preset": _get("system_prompt_preset", None),
        "append_system_prompt": _get("append_system_prompt", None),
    }
    # Advanced ClaudeAgentOptions passthroughs — forwarded verbatim when set
    # so power users reach the rest of the SDK surface without new plumbing.
    # (_build_options drops any key the installed SDK doesn't know.)
    for passthrough in ("cli_path", "betas", "fallback_model", "sandbox", "add_dirs", "extra_args"):
        val = opts.get(passthrough)
        if val is not None:
            settings[passthrough] = val
    return settings


# ---------------------------------------------------------------------------
# Auth: map Hermes' resolved token onto the CLI's env-based auth.
# ---------------------------------------------------------------------------
def build_auth_env(agent) -> Dict[str, str]:
    """Build the ``options.env`` dict that authenticates the spawned CLI.

    Resolves the token via Hermes' existing 5-priority resolver, classifies
    it as OAuth-vs-API-key, and sets the matching CLI env var. Starts from a
    copy of ``os.environ`` (so PATH/HOME reach the subprocess) and overrides
    only the auth keys, explicitly clearing the non-chosen one so a stale
    ambient key can't contradict the Hermes-resolved token.
    """
    from agent.anthropic_adapter import resolve_anthropic_token, _is_oauth_token

    token = (
        getattr(agent, "_anthropic_api_key", None)
        or resolve_anthropic_token()
        or ""
    )
    token = token.strip() if isinstance(token, str) else ""

    env: Dict[str, str] = {k: v for k, v in os.environ.items()}
    # Clear both, then set exactly one, so precedence is unambiguous.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)

    if token:
        if _is_oauth_token(token):
            # Subscription / Claude Code OAuth token — Bearer auth via the CLI.
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        else:
            env["ANTHROPIC_API_KEY"] = token

    base_url = getattr(agent, "_anthropic_base_url", None)
    if isinstance(base_url, str) and base_url.strip() and "api.anthropic.com" not in base_url:
        # Route through a gateway / self-hosted Anthropic-compatible endpoint.
        env["ANTHROPIC_BASE_URL"] = base_url.strip().rstrip("/")

    return env


# ---------------------------------------------------------------------------
# Anthropic-Message-shaped return objects (consumed by AnthropicTransport).
# ---------------------------------------------------------------------------
class _SDKBlock:
    """A content block quacking like an Anthropic SDK block.

    Only public attributes are set so ``_to_plain_data`` (which reads
    ``vars()`` minus ``_``-prefixed keys) converts it to a clean dict.
    """

    __slots__ = ("type", "text", "thinking", "signature", "id", "name", "input")

    def __init__(self, **kw):
        self.type = kw.get("type")
        self.text = kw.get("text")
        self.thinking = kw.get("thinking")
        self.signature = kw.get("signature")
        self.id = kw.get("id")
        self.name = kw.get("name")
        self.input = kw.get("input")


class _SDKUsage:
    """Usage object exposing the attrs AnthropicTransport.extract_cache_stats reads."""

    def __init__(self, usage: Optional[dict], total_cost_usd: Optional[float], session_id: Optional[str]):
        u = usage or {}
        self.input_tokens = u.get("input_tokens", 0) or 0
        self.output_tokens = u.get("output_tokens", 0) or 0
        self.cache_read_input_tokens = u.get("cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens = u.get("cache_creation_input_tokens", 0) or 0
        self.total_cost_usd = total_cost_usd
        self.session_id = session_id


class _SDKMessage:
    """Duck-typed Anthropic ``Message`` returned to the agent loop."""

    def __init__(self, content: List[_SDKBlock], stop_reason: str, usage: _SDKUsage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        # Present for parity with anthropic.types.Message; unused by the loop.
        self.role = "assistant"
        self.type = "message"


# ---------------------------------------------------------------------------
# api_kwargs (Anthropic-shaped) -> SDK inputs
# ---------------------------------------------------------------------------
def _content_to_text(content: Any) -> str:
    """Flatten Anthropic-format message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text") or "")
                elif btype == "tool_result":
                    inner = block.get("content")
                    parts.append(inner if isinstance(inner, str) else _content_to_text(inner))
                # image / tool_use blocks are intentionally skipped
        return "\n".join(p for p in parts if p)
    return ""


def _coerce_system(system: Any) -> Optional[str]:
    """Anthropic ``system`` may be a string or a list of blocks with cache_control."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            (b.get("text") or "") if isinstance(b, dict) else str(b) for b in system
        ).strip() or None
    return str(system)


def _last_user_text(messages: List[dict]) -> str:
    """Return the flattened text of the last user-role message."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _content_to_text(msg.get("content"))
            if text:
                return text
    # Fallback: flatten everything (e.g. a lone system-primed turn).
    return _content_to_text([m.get("content") for m in (messages or []) if isinstance(m, dict)])


# normalize_tool_spec / looks_like_tool_error now live in the shared
# agent.transports.hermes_tool_exposure module (single source of truth with
# the codex_app_server backend). Imported at module top.


def _build_guardrail_hook(sdk, agent):
    """PreToolUse hook enforcing Hermes' tool guardrails inside the SDK loop.

    Fires before every tool the SDK is about to run — delegate's built-ins and
    hybrid's ``mcp__hermes__*`` tools — and denies calls Hermes' guardrail
    controller rejects. Never raises: a hook exception must not break the loop.
    Returns a ``hooks`` dict suitable for ``ClaudeAgentOptions.hooks``.
    """
    guardrails = getattr(agent, "_tool_guardrails", None)
    prefix = f"mcp__{_HYBRID_SERVER}__"

    async def _pre_tool_use(input_data, tool_use_id, context):
        try:
            if guardrails is None or not hasattr(guardrails, "before_call"):
                return {}
            name = (input_data or {}).get("tool_name") or ""
            args = (input_data or {}).get("tool_input") or {}
            reg_name = name[len(prefix):] if name.startswith(prefix) else name
            decision = guardrails.before_call(reg_name, args)
            if decision is not None and not getattr(decision, "allows_execution", True):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            getattr(decision, "message", None)
                            or "Blocked by Hermes tool guardrail policy"
                        ),
                    }
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("claude_agent_sdk guardrail hook error: %s", exc)
        return {}

    return {"PreToolUse": [sdk.HookMatcher(hooks=[_pre_tool_use])]}


def _build_hybrid_mcp_server(sdk, agent, tools: List[dict]):
    """Build an in-process MCP server exposing Hermes tools to the SDK loop.

    Each Hermes tool becomes an ``@tool`` whose handler routes through Hermes'
    unified single-tool entry ``invoke_tool`` — which runs the tool_request
    middleware + plugin block hooks and handles BOTH registry tools and
    agent-level tools (todo/memory/clarify/delegate_task/read_terminal/
    session_search) with live agent state — then wraps the result in the
    ``<untrusted_tool_result>`` promptware defense used by the native path.
    Returns ``(server, allowed_tool_names)``.
    """
    from agent.tool_dispatch_helpers import _maybe_wrap_untrusted
    from agent.agent_runtime_helpers import invoke_tool

    effective_task_id = (
        getattr(agent, "task_id", None) or getattr(agent, "_task_id", None) or ""
    )
    guardrails = getattr(agent, "_tool_guardrails", None)

    def _make_handler(tool_name: str):
        async def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
            call_args = args or {}
            # In-process guardrail check — belt-and-suspenders with the
            # PreToolUse hook, which may not fire in single-message query mode.
            if guardrails is not None and hasattr(guardrails, "before_call"):
                try:
                    decision = guardrails.before_call(tool_name, call_args)
                    if decision is not None and not getattr(decision, "allows_execution", True):
                        msg = getattr(decision, "message", None) or "Blocked by Hermes tool guardrail policy"
                        return {"content": [{"type": "text", "text": msg}], "is_error": True}
                except Exception as exc:  # noqa: BLE001
                    logger.debug("claude_agent_sdk hybrid guardrail error for %s: %s", tool_name, exc)
            # invoke_tool is the concurrent path's worker, so running it in a
            # thread executor matches its intended use and keeps the SDK's event
            # loop responsive during tool execution.
            try:
                raw = await asyncio.to_thread(
                    invoke_tool, agent, tool_name, call_args, effective_task_id,
                    tool_call_id=f"cas_{uuid.uuid4().hex[:12]}",
                )
            except Exception as exc:  # noqa: BLE001
                return {"content": [{"type": "text", "text": f"Tool failed: {exc}"}], "is_error": True}
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            is_error = looks_like_tool_error(text)
            text = _maybe_wrap_untrusted(tool_name, text)
            return {"content": [{"type": "text", "text": text}], "is_error": is_error}

        return _handler

    sdk_tools = []
    allowed: List[str] = []
    for spec in tools or []:
        normalized = normalize_tool_spec(spec)
        if normalized is None:
            continue
        name, description, input_schema = normalized
        try:
            decorated = sdk.tool(name, description, input_schema)(_make_handler(name))
        except Exception as exc:  # noqa: BLE001 — a single malformed schema shouldn't kill the run
            logger.debug("claude_agent_sdk hybrid: skipping tool %s (%s)", name, exc)
            continue
        sdk_tools.append(decorated)
        allowed.append(f"mcp__{_HYBRID_SERVER}__{name}")

    server = sdk.create_sdk_mcp_server(name=_HYBRID_SERVER, version="1.0.0", tools=sdk_tools)
    return server, allowed


# ---------------------------------------------------------------------------
# Async bridge — Hermes' loop is synchronous; the SDK is async.
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Drive an async coroutine to completion from Hermes' sync worker thread.

    Uses ``asyncio.run`` on a fresh loop. If (unexpectedly) a loop is already
    running on this thread, falls back to a dedicated thread with its own loop
    so we never raise ``asyncio.run() cannot be called from a running loop``.
    """
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
            box["error"] = exc
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="claude-agent-sdk", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


class _StreamBridge:
    """Forward Claude Agent SDK stream events to Hermes' live-progress callbacks.

    In delegate/hybrid the SDK loop runs many model turns and tool executions
    inside ONE Hermes API call, and SDK mode disables Hermes' own streaming
    path (``_disable_streaming``) — without this bridge the user stares at a
    silent screen until the whole turn finishes.  With
    ``include_partial_messages=True`` the SDK yields ``StreamEvent`` messages
    wrapping raw Anthropic stream events; this bridge mirrors the semantics of
    the native streaming paths (``_call_anthropic`` /
    ``stream_converse_with_callbacks``): text deltas → ``_fire_stream_delta``,
    tool starts → ``_fire_tool_gen_started`` +
    ``tool_progress_callback("tool.started", …)`` (same contract the codex
    app-server route bridges in #38835), thinking deltas →
    ``_fire_reasoning_delta``.
    """

    def __init__(self, agent):
        self._agent = agent
        has_stream = getattr(agent, "_has_stream_consumers", None)
        self._wants_text = bool(has_stream()) if callable(has_stream) else False
        self._wants_tools = (
            getattr(agent, "tool_progress_callback", None) is not None
            or getattr(agent, "tool_gen_callback", None) is not None
        )
        self._wants_reasoning = getattr(agent, "reasoning_callback", None) is not None
        # Tool_use blocks currently streaming, keyed by stream key — the
        # ``parent_tool_use_id`` (or "" for the top-level agent). Keying by
        # parent is essential now that subagents run in the BACKGROUND by
        # default (Claude Code v2.1.198) and the `Workflow` tool fans out many
        # agents at once: their StreamEvents interleave, so a single shared slot
        # would let one agent's tool input clobber another's. The start
        # callbacks fire at content_block_stop, once the input JSON is complete,
        # so the chip can show WHAT the tool ran (command, path, query…).
        self._pending: Dict[str, tuple] = {}          # key → (tool_use_id, display, is_sub)
        self._pending_json: Dict[str, List[str]] = {}  # key → [partial json chunks]
        # tool_use_id → (display_label, display_args) for completing the chip
        # when the SDK later reports the tool result.
        self._started_tools: Dict[str, tuple] = {}
        # task_id → display_label for background tasks / workflow agents, so a
        # terminal Task message can resolve the chip started by TaskStarted.
        self._started_tasks: Dict[str, str] = {}

    @property
    def active(self) -> bool:
        return self._wants_text or self._wants_tools or self._wants_reasoning

    def handle(self, event: Dict[str, Any], parent_tool_use_id: Optional[str] = None) -> None:
        """Dispatch one raw Anthropic stream event to the agent callbacks.

        ``parent_tool_use_id`` is set on events from a nested SDK subagent or a
        ``Workflow``-spawned agent (the id of the ``Agent``/``Task``/``Workflow``
        tool that spawned it). Such nested tool activity is surfaced (tagged as
        subagent work), but nested text/thinking deltas are dropped so they
        don't interleave with the top-level response bubble. Events are keyed by
        this id so concurrently-streaming agents don't clobber each other.
        """
        agent = self._agent
        # Keep the stale-call detector fed during long SDK turns — the
        # non-streaming path Hermes uses for SDK mode otherwise sees zero
        # activity while the SDK loop is busy running tools.
        touch = getattr(agent, "_touch_activity", None)
        if callable(touch):
            try:
                touch("receiving claude-agent-sdk stream")
            except Exception:
                pass
        key = parent_tool_use_id or ""
        is_sub = bool(parent_tool_use_id)
        etype = event.get("type")
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._on_tool_start(
                    str(block.get("name") or ""),
                    str(block.get("id") or ""),
                    key,
                    subagent=is_sub,
                )
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                # A subagent/workflow agent's response text is its own internal
                # output, not the top-level answer — never stream it into the
                # main bubble.
                if is_sub:
                    return
                text = delta.get("text") or ""
                fire = getattr(agent, "_fire_stream_delta", None)
                if text and self._wants_text and callable(fire):
                    fire(text)
            elif dtype == "thinking_delta":
                if is_sub:
                    return
                thinking = delta.get("thinking") or ""
                fire = getattr(agent, "_fire_reasoning_delta", None)
                if thinking and self._wants_reasoning and callable(fire):
                    fire(thinking)
            elif dtype == "input_json_delta" and key in self._pending:
                self._pending_json.setdefault(key, []).append(delta.get("partial_json") or "")
        elif etype == "content_block_stop":
            self._on_tool_stop(key)

    def _on_tool_start(self, tool_name: str, tool_use_id: str, key: str, subagent: bool = False) -> None:
        agent = self._agent
        # Hybrid tools arrive as "mcp__hermes__<name>" — show the bare name.
        display = tool_name.rsplit("__", 1)[-1] if tool_name else "tool"
        self._pending[key] = (
            tool_use_id or f"sdk-{uuid.uuid4().hex[:12]}",
            display,
            subagent,
        )
        self._pending_json[key] = []
        # Only a TOP-LEVEL tool interrupts the main text bubble; a nested
        # subagent/workflow tool runs off to the side and must not break it.
        if not subagent and self._wants_text:
            # Close the open streaming display segment so tool chrome doesn't
            # wrap into the text bubble. Same contract as the pre-tool flush in
            # conversation_loop: display callback gets None; the TTS callback
            # must NOT (it treats None as end-of-stream).
            cb = getattr(agent, "stream_delta_callback", None)
            if cb is not None:
                try:
                    cb(None)
                except Exception:
                    pass
            agent._stream_needs_break = True
        # Spinner/status while the model generates the tool's arguments.
        gen = getattr(agent, "_fire_tool_gen_started", None)
        if callable(gen):
            gen(f"{_SUBAGENT_TOOL_MARKER}{display}" if subagent else display)

    def _on_tool_stop(self, key: str) -> None:
        """Fire the start callbacks with the tool's real args once its input is complete."""
        pending = self._pending.pop(key, None)
        raw = "".join(self._pending_json.pop(key, []))
        if pending is None:
            return
        tool_use_id, name, subagent = pending
        try:
            args = json.loads(raw) if raw.strip() else {}
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        try:
            from agent.display import build_tool_preview, redact_tool_args_for_display

            # Build the preview/redaction against the REAL tool name so
            # tool-specific formatting (terminal→command, etc.) still applies;
            # only the user-visible label carries the subagent marker.
            display_args = redact_tool_args_for_display(name, args) or args
            preview = build_tool_preview(name, display_args)
        except Exception:
            display_args, preview = args, None
        label = f"{_SUBAGENT_TOOL_MARKER}{name}" if subagent else name
        self._started_tools[tool_use_id] = (label, display_args)
        # Desktop/TUI gateways render tool chips from tool_start_callback (id +
        # args → "Running command: <cmd>"); messaging gateways render bubbles
        # from tool_progress_callback. Fire both, like the native executor.
        # ``subagent=True`` rides along as a kwarg so consumers that want to nest
        # or badge subagent chips can, while the marker keeps it visible on
        # consumers that only render the name.
        start_cb = getattr(self._agent, "tool_start_callback", None)
        if start_cb is not None:
            try:
                start_cb(tool_use_id, label, display_args)
            except Exception:
                logger.debug("claude_agent_sdk: tool-start callback raised", exc_info=True)
        progress = getattr(self._agent, "tool_progress_callback", None)
        if progress is not None:
            try:
                progress("tool.started", label, preview, display_args, subagent=subagent)
            except TypeError:
                # Older consumer without the subagent kwarg — the marker in the
                # label already conveys it.
                progress("tool.started", label, preview, display_args)
            except Exception:
                logger.debug("claude_agent_sdk: tool-progress callback raised", exc_info=True)

    def handle_tool_result(self, tool_use_id: str, content: Any, is_error: Any = None) -> None:
        """Resolve a started tool chip when the SDK reports the tool's result."""
        name, display_args = self._started_tools.pop(tool_use_id, (None, None))
        if name is None:
            return
        if isinstance(content, str):
            result_text = content
        else:
            try:
                result_text = json.dumps(content, ensure_ascii=False, default=str)
            except Exception:
                result_text = str(content)
        # Chip channel (desktop/TUI inline diffs) resolves via tool_complete.
        complete_cb = getattr(self._agent, "tool_complete_callback", None)
        if complete_cb is not None:
            try:
                complete_cb(tool_use_id, name, display_args or {}, result_text)
            except Exception:
                logger.debug("claude_agent_sdk: tool-complete callback raised", exc_info=True)
        # Progress channel (CLI spinner/scrollback, messaging bubbles) closes via
        # a "tool.completed" event — same contract the native executor emits.
        self._fire_tool_completed(name, result_text, bool(is_error))

    def _fire_tool_completed(self, label: str, result_text: str, is_error: bool) -> None:
        progress = getattr(self._agent, "tool_progress_callback", None)
        if progress is None:
            return
        try:
            progress("tool.completed", label, None, None, is_error=is_error, result=result_text)
        except TypeError:
            try:
                progress("tool.completed", label, None, None)
            except Exception:
                logger.debug("claude_agent_sdk: tool.completed callback raised", exc_info=True)
        except Exception:
            logger.debug("claude_agent_sdk: tool.completed callback raised", exc_info=True)

    # -- Background tasks / dynamic workflows -------------------------------
    # The SDK reports background tasks (including dynamic-workflow agents) via
    # Task* system messages that flow through the query() loop OUT-OF-BAND from
    # the normal assistant stream — a workflow runs in its own runtime, so its
    # agents never appear as parent_tool_use_id stream events. We surface each
    # task as its own chip (started → resolved) through the same callbacks the
    # `/workflows` view is built on, so desktop and messaging gateways both see
    # workflow/background-task progress live.
    def handle_task(self, message: Any) -> None:
        """Dispatch a Task* system message (Started/Progress/Updated/Notification)."""
        agent = self._agent
        touch = getattr(agent, "_touch_activity", None)
        if callable(touch):
            try:
                touch("claude-agent-sdk task update")
            except Exception:
                pass
        name = type(message).__name__
        task_id = str(getattr(message, "task_id", "") or "")
        if not task_id:
            return
        if name == "TaskStartedMessage":
            desc = (getattr(message, "description", "") or "").strip()
            self._on_task_start(task_id, desc, getattr(message, "task_type", None))
        elif name == "TaskProgressMessage":
            # Keep the spinner/stale-detector fed; don't spawn a new chip per
            # progress tick (touch above already refreshed activity).
            last_tool = getattr(message, "last_tool_name", None)
            if last_tool:
                gen = getattr(agent, "_fire_tool_gen_started", None)
                if callable(gen):
                    label = self._started_tasks.get(task_id, f"{_TASK_MARKER}task")
                    try:
                        gen(f"{label} · {last_tool}")
                    except Exception:
                        pass
        elif name == "TaskUpdatedMessage":
            status = getattr(message, "status", None)
            if status in _TERMINAL_TASK_STATUSES:
                self._on_task_done(task_id, status, "")
        elif name == "TaskNotificationMessage":
            self._on_task_done(
                task_id,
                getattr(message, "status", None),
                (getattr(message, "summary", "") or "").strip(),
            )

    def _on_task_start(self, task_id: str, desc: str, task_type: Optional[str]) -> None:
        label = f"{_TASK_MARKER}{desc or task_type or 'task'}"
        self._started_tasks[task_id] = label
        args = {"task_type": task_type} if task_type else {}
        start_cb = getattr(self._agent, "tool_start_callback", None)
        if start_cb is not None:
            try:
                start_cb(task_id, label, args)
            except Exception:
                logger.debug("claude_agent_sdk: task-start callback raised", exc_info=True)
        progress = getattr(self._agent, "tool_progress_callback", None)
        if progress is not None:
            try:
                progress("tool.started", label, desc or None, args, subagent=True)
            except TypeError:
                progress("tool.started", label, desc or None, args)
            except Exception:
                logger.debug("claude_agent_sdk: task-progress callback raised", exc_info=True)

    def _on_task_done(self, task_id: str, status: Optional[str], summary: str) -> None:
        label = self._started_tasks.pop(task_id, None)
        if label is None:
            return
        result = summary or (f"task {status}" if status else "task finished")
        is_error = status in ("failed", "killed", "stopped")
        if is_error:
            result = f"[{status}] {result}"
        # Chip channel (desktop/TUI inline diffs).
        complete_cb = getattr(self._agent, "tool_complete_callback", None)
        if complete_cb is not None:
            try:
                complete_cb(task_id, label, {}, result)
            except Exception:
                logger.debug("claude_agent_sdk: task-complete callback raised", exc_info=True)
        # Progress channel (CLI spinner/scrollback, messaging bubbles).
        self._fire_tool_completed(label, result, is_error)


async def _collect_query(
    sdk,
    prompt: str,
    options,
    bridge: Optional[_StreamBridge] = None,
    interrupt_check=None,
) -> Dict[str, Any]:
    """Run ``query(prompt, options)`` and collect text, usage and session id.

    ``interrupt_check`` polls Hermes' /stop flag between SDK messages: one-shot
    ``query()`` has no native interrupt (that needs ``ClaudeSDKClient`` +
    streaming input), but raising out of the async-for closes the generator,
    which tears down the CLI transport — so a stop lands within one message.
    """
    AssistantMessage = sdk.AssistantMessage
    ResultMessage = sdk.ResultMessage
    TextBlock = getattr(sdk, "TextBlock", None)
    ThinkingBlock = getattr(sdk, "ThinkingBlock", None)
    StreamEvent = getattr(sdk, "StreamEvent", None)
    UserMessage = getattr(sdk, "UserMessage", None)
    ToolResultBlock = getattr(sdk, "ToolResultBlock", None)
    # Background-task / dynamic-workflow progress messages (absent on old SDKs).
    _task_types = tuple(
        t for t in (
            getattr(sdk, "TaskStartedMessage", None),
            getattr(sdk, "TaskProgressMessage", None),
            getattr(sdk, "TaskUpdatedMessage", None),
            getattr(sdk, "TaskNotificationMessage", None),
        ) if t is not None
    )

    assistant_text: List[str] = []
    thinking_text: List[str] = []
    result_text: Optional[str] = None
    usage: Optional[dict] = None
    total_cost: Optional[float] = None
    session_id: Optional[str] = None
    is_error = False
    subtype: Optional[str] = None
    stop_reason: Optional[str] = None
    errors: List[str] = []

    async for message in sdk.query(prompt=prompt, options=options):
        if interrupt_check is not None and interrupt_check():
            raise InterruptedError("Agent interrupted during Claude Agent SDK turn")
        if StreamEvent is not None and isinstance(message, StreamEvent):
            # Live progress only — never part of the collected result. Subagent
            # streams (parent_tool_use_id set) are still surfaced, but the bridge
            # shows only their TOOL activity (marked as subagent work); their
            # text/thinking is suppressed so it doesn't interleave with the
            # top-level response.
            if bridge is not None:
                try:
                    bridge.handle(
                        getattr(message, "event", None) or {},
                        parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                    )
                except Exception:
                    logger.debug("claude_agent_sdk: stream bridge error", exc_info=True)
            continue
        if _task_types and isinstance(message, _task_types):
            # Background task / workflow-agent progress — surface as its own chip
            # lifecycle, never part of the collected top-level result.
            if bridge is not None:
                try:
                    bridge.handle_task(message)
                except Exception:
                    logger.debug("claude_agent_sdk: task bridge error", exc_info=True)
            continue
        if UserMessage is not None and isinstance(message, UserMessage):
            # Tool results echoing back into the SDK loop — resolve the
            # matching tool chip (never part of the collected result).
            if bridge is not None:
                content = getattr(message, "content", None)
                for block in content if isinstance(content, list) else []:
                    if ToolResultBlock is not None and isinstance(block, ToolResultBlock):
                        try:
                            bridge.handle_tool_result(
                                getattr(block, "tool_use_id", "") or "",
                                getattr(block, "content", None),
                                getattr(block, "is_error", None),
                            )
                        except Exception:
                            logger.debug("claude_agent_sdk: tool-result bridge error", exc_info=True)
            continue
        if isinstance(message, AssistantMessage):
            # A subagent's own assistant turns (parent_tool_use_id set) are its
            # internal reasoning — not the top-level response. Its tool activity
            # is surfaced live via the stream bridge; keep its text/thinking out
            # of the collected result so it can't leak into the final message.
            if getattr(message, "parent_tool_use_id", None):
                continue
            for block in getattr(message, "content", []) or []:
                if TextBlock is not None and isinstance(block, TextBlock):
                    assistant_text.append(block.text or "")
                elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                    thinking_text.append(getattr(block, "thinking", "") or "")
                elif getattr(block, "type", None) == "text":
                    assistant_text.append(getattr(block, "text", "") or "")
                elif getattr(block, "type", None) == "thinking":
                    thinking_text.append(getattr(block, "thinking", "") or "")
        elif isinstance(message, ResultMessage):
            result_text = getattr(message, "result", None)
            usage = getattr(message, "usage", None)
            total_cost = getattr(message, "total_cost_usd", None)
            session_id = getattr(message, "session_id", None)
            is_error = bool(getattr(message, "is_error", False))
            subtype = getattr(message, "subtype", None)
            stop_reason = getattr(message, "stop_reason", None)
            errs = getattr(message, "errors", None)
            if isinstance(errs, list):
                errors = [str(e) for e in errs]

    # ResultMessage.result is only populated on the `success` subtype; on error
    # subtypes fall back to the collected assistant text so the user still sees
    # partial output rather than nothing.
    if subtype == "success" and result_text is not None:
        text = result_text
    else:
        text = result_text or "\n".join(t for t in assistant_text if t)

    return {
        "text": (text or "").strip(),
        "thinking": "\n\n".join(t for t in thinking_text if t).strip() or None,
        "usage": usage,
        "total_cost_usd": total_cost,
        "session_id": session_id,
        "is_error": is_error,
        "subtype": subtype,
        "stop_reason": stop_reason,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Public entry point — called from AIAgent._anthropic_messages_create.
# ---------------------------------------------------------------------------
def create_claude_agent_message(agent, api_kwargs: dict) -> _SDKMessage:
    """Run one Hermes "turn" through the Claude Agent SDK.

    Returns an object shaped like a native Anthropic ``Message`` so that
    ``AnthropicTransport.normalize_response`` (already selected because
    ``api_mode == "anthropic_messages"``) consumes it unchanged.
    """
    sdk = _require_sdk()

    mode = getattr(agent, "_claude_agent_sdk_mode", None) or "inference"
    settings = getattr(agent, "_claude_agent_sdk_settings", None) or {}

    model = api_kwargs.get("model") or getattr(agent, "model", None)
    system = _coerce_system(api_kwargs.get("system"))
    messages = api_kwargs.get("messages") or []
    tools = api_kwargs.get("tools") or []
    prompt = _last_user_text(messages)
    if not prompt:
        prompt = "Continue."

    env = build_auth_env(agent)

    # ── System prompt ──────────────────────────────────────────────────
    # Hermes owns its soul/persona → default to a custom string. Users may
    # opt into the claude_code preset (with optional append) via config,
    # useful for delegate mode acting as a coding agent.
    preset = settings.get("system_prompt_preset")
    append = settings.get("append_system_prompt")
    if preset:
        system_prompt: Any = {"type": "preset", "preset": str(preset)}
        if append or system:
            system_prompt["append"] = append or system
    else:
        system_prompt = system

    # ── Options per mode ───────────────────────────────────────────────
    opt_kwargs: Dict[str, Any] = {
        "model": model,
        "env": env,
        # Don't inherit local .claude/settings.json or CLAUDE.md — Hermes owns
        # the prompt and tool surface.
        "setting_sources": [],
    }
    if system_prompt is not None:
        opt_kwargs["system_prompt"] = system_prompt

    # Anchor the SDK session to the agent's logical working directory — the
    # per-session project dir on multi-session gateways (desktop Projects),
    # TERMINAL_CWD, or the launch dir — instead of wherever the backend
    # process happens to run. Applies to ALL modes so the CLI's terminal,
    # file tools, and session transcripts live in the user's project.
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = settings.get("cwd") or str(resolve_agent_cwd())
    except Exception:
        cwd = settings.get("cwd") or os.getcwd()
    opt_kwargs["cwd"] = cwd

    # Resume the SDK's own session for cross-turn context — but only within
    # the same working directory. The CLI keys session transcripts by cwd, so
    # resuming a session created under a different project dir would fail (or
    # continue in the wrong project); start fresh instead.
    resume = getattr(agent, "_claude_sdk_session_id", None)
    if resume and getattr(agent, "_claude_sdk_session_cwd", None) == cwd:
        opt_kwargs["resume"] = resume

    # Respect Hermes' reasoning settings instead of the CLI's own defaults —
    # AnthropicTransport.build_kwargs already mapped the user's reasoning
    # config (off/low/…/max) to `thinking` + `output_config.effort`; without
    # this passthrough the CLI picks its own thinking depth (users saw "low"
    # behave like "max").
    thinking_cfg = api_kwargs.get("thinking")
    opt_kwargs["thinking"] = thinking_cfg if isinstance(thinking_cfg, dict) else {"type": "disabled"}
    effort = (api_kwargs.get("output_config") or {}).get("effort")
    if effort:
        opt_kwargs["effort"] = effort

    mcp_note = ""
    if mode == "inference":
        # Single model call, no tools. Hermes drives its own agent loop.
        opt_kwargs["max_turns"] = 1
        opt_kwargs["tools"] = []          # strip built-in Read/Edit/Bash
        opt_kwargs["permission_mode"] = settings.get("permission_mode", "dontAsk")
    elif mode == "delegate":
        # SDK runs the whole task with its OWN built-in tools, governed by
        # Hermes' guardrails via a PreToolUse hook.
        opt_kwargs["permission_mode"] = settings.get("permission_mode", "bypassPermissions")
        opt_kwargs["max_turns"] = settings.get("max_turns", 24)
        extra_allowed = settings.get("allowed_tools") or []
        if extra_allowed:
            opt_kwargs["allowed_tools"] = list(extra_allowed)
        disallowed = settings.get("disallowed_tools") or []
        if disallowed:
            opt_kwargs["disallowed_tools"] = list(disallowed)
        opt_kwargs["hooks"] = _build_guardrail_hook(sdk, agent)
    elif mode == "hybrid":
        # SDK drives the loop but calls Hermes' tools via in-process MCP.
        # Prefer agent.tools (OpenAI-format, clean names) over api_kwargs["tools"]
        # (Anthropic-format, mcp__-prefixed on the OAuth wire).
        hybrid_tools = getattr(agent, "tools", None) or tools
        server, allowed = _build_hybrid_mcp_server(sdk, agent, hybrid_tools)
        opt_kwargs["mcp_servers"] = {_HYBRID_SERVER: server}
        # Pin the MCP surface to exactly this server — no user/project MCP
        # config can add tools behind Hermes' back.
        opt_kwargs["strict_mcp_config"] = True
        opt_kwargs["allowed_tools"] = allowed
        opt_kwargs["tools"] = []          # Hermes tools only; strip built-ins
        opt_kwargs["permission_mode"] = settings.get("permission_mode", "bypassPermissions")
        opt_kwargs["max_turns"] = settings.get("max_turns", 24)
        opt_kwargs["hooks"] = _build_guardrail_hook(sdk, agent)
        disallowed = settings.get("disallowed_tools") or []
        if disallowed:
            opt_kwargs["disallowed_tools"] = list(disallowed)
        mcp_note = f" ({len(allowed)} hermes tools)"

    # Spend cap (delegate/hybrid run multi-turn); the SDK ends the loop with
    # subtype error_max_budget_usd when exceeded.
    if mode in ("delegate", "hybrid") and settings.get("max_budget_usd") is not None:
        opt_kwargs["max_budget_usd"] = settings["max_budget_usd"]

    # Advanced passthroughs from config (cli_path, betas, fallback_model,
    # sandbox, add_dirs, extra_args) — see resolve_claude_agent_sdk_settings.
    for passthrough in ("cli_path", "betas", "fallback_model", "sandbox", "add_dirs", "extra_args"):
        if settings.get(passthrough) is not None:
            opt_kwargs[passthrough] = settings[passthrough]

    # Live progress: when a display/TTS/tool-progress consumer is attached,
    # ask the SDK for raw stream events so the user sees text/thinking/tool
    # activity in real time instead of a silent wait. _build_options drops the
    # flag on older SDKs that don't know it.
    bridge = _StreamBridge(agent)
    if bridge.active:
        opt_kwargs["include_partial_messages"] = True

    options = _build_options(sdk, opt_kwargs)

    logger.debug(
        "%sclaude_agent_sdk: mode=%s model=%s prompt_chars=%d%s",
        getattr(agent, "log_prefix", ""), mode, model, len(prompt), mcp_note,
    )

    try:
        collected = _run_async(
            _collect_query(
                sdk,
                prompt,
                options,
                bridge=bridge if bridge.active else None,
                # One-shot query() has no native interrupt; polling Hermes'
                # /stop flag between SDK messages and raising closes the
                # generator, which tears down the CLI transport.
                interrupt_check=lambda: bool(getattr(agent, "_interrupt_requested", False)),
            )
        )
    except InterruptedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _friendly = _classify_sdk_error(exc)
        if _friendly:
            raise RuntimeError(_friendly) from exc
        raise

    # Persist the session id so the next Hermes turn continues the SDK's
    # own conversation context.
    new_session = collected.get("session_id")
    if new_session:
        agent._claude_sdk_session_id = new_session
        # Remember which directory the session lives in — resume is only
        # valid from the same cwd (see the resume gate above).
        agent._claude_sdk_session_cwd = cwd

    text = collected.get("text") or ""
    if not text and collected.get("is_error"):
        detail = "; ".join(collected.get("errors") or []) or collected.get("subtype") or "unknown error"
        raise RuntimeError(f"Claude Agent SDK returned an error result: {detail}")

    blocks: List[_SDKBlock] = []
    if collected.get("thinking"):
        blocks.append(_SDKBlock(type="thinking", thinking=collected["thinking"]))
    blocks.append(_SDKBlock(type="text", text=text))

    # inference/delegate/hybrid all surface a completed assistant turn: no
    # tool_calls flow back to Hermes (the SDK either did no tools, or already
    # executed them internally). Map the SDK's terminal state to an
    # Anthropic-style stop_reason so AnthropicTransport derives the right
    # finish_reason. ResultMessage.stop_reason is the model's own stop reason
    # (end_turn/max_tokens/refusal); subtype is the loop's terminal state.
    sdk_stop = collected.get("stop_reason")
    subtype = collected.get("subtype")
    if sdk_stop == "refusal":
        stop_reason = "refusal"        # → finish_reason content_filter
    elif sdk_stop == "max_tokens" or subtype in ("error_max_turns", "error_max_budget_usd"):
        stop_reason = "max_tokens"     # → finish_reason length
    else:
        stop_reason = "end_turn"       # → finish_reason stop

    usage = _SDKUsage(collected.get("usage"), collected.get("total_cost_usd"), new_session)
    return _SDKMessage(content=blocks, stop_reason=stop_reason, usage=usage)


def _build_options(sdk, opt_kwargs: Dict[str, Any]):
    """Construct ClaudeAgentOptions, dropping keys unsupported by the installed SDK."""
    ClaudeAgentOptions = sdk.ClaudeAgentOptions
    kwargs = {k: v for k, v in opt_kwargs.items() if v is not None}
    while True:
        try:
            return ClaudeAgentOptions(**kwargs)
        except TypeError as exc:
            # An older SDK may not know a field (e.g. `tools`, `setting_sources`).
            msg = str(exc)
            dropped = None
            for key in list(kwargs.keys()):
                if key in msg:
                    dropped = key
                    break
            if dropped is None:
                raise
            logger.debug("claude_agent_sdk: dropping unsupported option %r (%s)", dropped, msg)
            kwargs.pop(dropped, None)


def _classify_sdk_error(exc: Exception) -> Optional[str]:
    """Return a friendly message for common SDK setup/startup failures, else None.

    Matches on the exception *class name* rather than isinstance so we never have
    to import the SDK's error classes (which live behind the lazy import) and
    stay resilient to version drift. Covers the documented ``claude_agent_sdk``
    errors (https://code.claude.com/docs/en/agent-sdk/python):

      - ``CLINotFoundError``  — the ``claude`` CLI isn't installed / not on PATH
      - ``ProcessError``      — the CLI was found but exited non-zero (usually an
                                unauthenticated CLI, a version mismatch, or bad
                                flags); carries ``exit_code`` + ``stderr``
      - ``CLIConnectionError`` / ``CLIJSONDecodeError`` — transport failures
    """
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()

    # Can't FIND the CLI (CLINotFoundError subclasses CLIConnectionError, so it
    # must be checked first).
    if "CLINotFound" in name or ("claude" in lowered and "not found" in lowered):
        return (
            "The Claude Agent SDK could not find the `claude` CLI. Install it "
            "(npm i -g @anthropic-ai/claude-code, or the native installer), make "
            "sure it is on PATH, then run `claude` once to log in. You can also "
            "pin the binary path with model.claude_agent_sdk.cli_path."
        )

    # Found the CLI but it FAILED TO START / exited non-zero.
    if "ProcessError" in name:
        exit_code = getattr(exc, "exit_code", None)
        stderr = (getattr(exc, "stderr", None) or "").strip()
        stderr_l = stderr.lower()
        # The most common startup failure for subscription/OAuth WebUI users is
        # an unauthenticated CLI — point them straight at the fix.
        auth_markers = (
            "login", "log in", "unauthorized", "authentication", "authenticate",
            "api key", "oauth", "credit balance", "invalid x-api-key",
        )
        if any(marker in stderr_l for marker in auth_markers):
            hint = (
                "This looks like an authentication problem: run `claude` (or "
                "`claude /login`) in this environment to sign in with your Claude "
                "subscription, or provide a valid ANTHROPIC_API_KEY / "
                "CLAUDE_CODE_OAUTH_TOKEN."
            )
        else:
            hint = (
                "The `claude` CLI was found but exited with an error. Confirm it "
                "is logged in and current (`claude` then /login; `claude update`)."
            )
        parts = ["The Claude Agent SDK could not start the `claude` CLI."]
        if exit_code is not None:
            parts.append(f"Exit code: {exit_code}.")
        if stderr:
            snippet = stderr if len(stderr) <= 500 else stderr[:500] + "…"
            parts.append(f"CLI output: {snippet}")
        parts.append(hint)
        return " ".join(parts)

    if "CLIConnection" in name or "CLIJSONDecode" in name:
        return f"Claude Agent SDK transport error ({name}): {text}"

    return None
