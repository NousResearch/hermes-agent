"""Session adapter for the claude-agent-sdk runtime.

Owns one Claude Agent SDK client per Hermes session — the structural twin of
``codex_app_server_session.py``, with the Codex JSON-RPC subprocess replaced
by Anthropic's official ``claude-agent-sdk`` (which manages the Claude Code
CLI subprocess, its agent loop, and — critically — **subscription OAuth**:
``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential store, never a
metered ``ANTHROPIC_API_KEY``). See GitHub issue #25267.

Lifecycle:
    session = ClaudeAgentSdkSession(cwd="/home/x/proj", model="claude-opus-4-8")
    session.ensure_started()                       # loop thread + SDK connect
    result = session.run_turn(user_input="hello")  # blocks until ResultMessage
    session.close()                                # disconnect + stop loop

Threading model: the SDK is async-first, but AIAgent.run_conversation() is
synchronous (the same constraint that made CodexAppServerClient thread-based).
The adapter owns a dedicated background thread running one asyncio event loop
for the whole session lifetime; every SDK coroutine is marshaled onto it with
``asyncio.run_coroutine_threadsafe`` and awaited with a timeout, so the SDK
client keeps stable loop affinity and ``run_turn`` stays blocking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import Any, Callable, Optional

# TurnResult is the shared contract with the runtime glue — reused verbatim
# from the codex session module (same fields, same semantics) so
# ``run_claude_agent_sdk_turn`` mirrors ``run_codex_app_server_turn`` 1:1.
from agent.transports.codex_app_server_session import TurnResult
from agent.transports.claude_sdk_event_projector import ClaudeSdkEventProjector

logger = logging.getLogger(__name__)


# Hermes' tools.terminal.security_mode → SDK permission_mode.
# "auto" parity note: codex's default profile is workspace-write; the closest
# SDK mode is acceptEdits (file edits auto-approved inside cwd, everything
# else goes through can_use_tool / fail-closed).
_HERMES_TO_SDK_PERMISSION_MODE = {
    "auto": "acceptEdits",
    "approval-required": "default",
    "unrestricted": "bypassPermissions",
    "yolo": "bypassPermissions",
}

# Substrings in SDK/CLI errors that signal broken subscription credentials.
# Conservative on purpose — mirrors codex's _classify_oauth_failure contract.
_AUTH_FAILURE_HINTS = (
    "not logged in",
    "please run /login",
    "invalid api key",
    "authentication_error",
    "401",
    "unauthorized",
    "oauth token",
    "token has expired",
    "expired token",
    "invalid bearer token",
    "setup-token",
    "credentials",
)


def classify_auth_failure(*parts: str) -> Optional[str]:
    """Return a user-friendly re-auth hint if the strings look like a Claude
    subscription auth failure; otherwise None."""
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _AUTH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Claude authentication failed — the subscription OAuth token "
                "looks expired or invalid. Refresh it with `claude setup-token` "
                "(or `claude login` on this machine) and update "
                "CLAUDE_CODE_OAUTH_TOKEN, then retry."
            )
    return None


def check_claude_sdk_available() -> tuple[bool, str]:
    """Preflight: the optional SDK extra must be importable, and it bundles /
    locates the Claude Code CLI itself. Mirrors check_codex_binary()."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return (
            False,
            "claude-agent-sdk is not installed. "
            "Install with: pip install 'hermes-agent[claude-agent-sdk]'",
        )
    return True, "ok"


def _hermes_repo_root() -> str:
    """Repo root for the hermes-tools MCP subprocess (PYTHONPATH)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# The SDK serializes the stdio MCP config — env INCLUDED — into the claude
# CLI's --mcp-config argument, i.e. onto the subprocess argv, which any local
# user can read via ps. Nothing secret may ever ride this dict: the env is a
# minimal ALLOWLIST, never a copy of the credentialed environment. Keyed
# Hermes tools inside the server degrade via their own check_fns — the
# subscription lane's fail-closed posture.
_MCP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONUTF8",
    "HERMES_HOME",
    "HERMES_KANBAN_TASK",
    "HERMES_MCP_STATE_DB",  # the shims' documented state-DB override — a path, not a secret
    "HERMES_QUIET",
    "HERMES_REDACT_SECRETS",
)


def _provider_config() -> dict:
    """The `agent.claude_agent_sdk` config block ({} when absent/unreadable)."""
    try:
        from hermes_cli.config import load_config_readonly

        block = ((load_config_readonly() or {}).get("agent", {}) or {}).get(
            "claude_agent_sdk", {}
        )
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _provider_flag(config_key: str, env_var: str, default: bool = False) -> bool:
    """Behavioral flag: config.yaml is the operator interface
    (`agent.claude_agent_sdk.<key>`); the env var remains as an explicit
    deployment override (systemd drop-ins) and wins when set."""
    env = os.environ.get(env_var, "").strip().lower()
    if env:
        return env in ("1", "true", "yes")
    value = _provider_config().get(config_key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _build_hermes_tools_mcp_config(
    hermes_session_id: Optional[str] = None,
) -> dict[str, Any]:
    """The stdio MCP server exposing Hermes tools into the SDK agent loop —
    the exact server the codex runtime uses (backend-agnostic), launched with
    this venv's interpreter. McpStdioServerConfig has no cwd field, so the
    repo root rides PYTHONPATH."""
    env = {
        key: os.environ[key]
        for key in _MCP_ENV_ALLOWLIST
        if os.environ.get(key)
    }
    env["PYTHONPATH"] = _hermes_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", "")
    if hermes_session_id:
        # Lets the stateless session_search shim exclude the calling
        # session's own lineage from recall results (#26567).
        env["HERMES_MCP_SESSION_ID"] = str(hermes_session_id)
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
        "env": env,
    }


class ClaudeAgentSdkSession:
    """One SDK client per Hermes session, lifetime owned by AIAgent.

    Not thread-safe from the caller's side — one caller drives it at a time,
    matching AIAgent.run_conversation(). Internally owns a loop thread."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_tool_started: Optional[Callable[[str, str, dict], None]] = None,
        max_budget_usd: Optional[float] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        include_hermes_tools: bool = True,
        hermes_session_id: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        self._permission_mode = (
            permission_mode
            or _HERMES_TO_SDK_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "acceptEdits",
            )
        )
        self._system_prompt_append = system_prompt_append
        self._approval_callback = approval_callback
        self._on_tool_started = on_tool_started
        self._max_budget_usd = max_budget_usd
        self._client_factory = client_factory  # test seam
        self._include_hermes_tools = include_hermes_tools
        # Hermes-side session id, exported to the hermes-tools MCP subprocess
        # so the stateless session_search shim can exclude its own lineage.
        self._hermes_session_id = hermes_session_id
        # SDK-side session id to resume (#25267 continuity). Verified live:
        # resume restores the model context and keeps the SAME session id; a
        # stale id fails the session start (the caller retires + retries
        # fresh).
        self._resume_session_id = resume_session_id
        # Display-only partial-text consumer (W4 streaming). Deltas never
        # enter the projected transcript; the gateway's stream consumer
        # handles rate limiting and the already_sent final-send dedup.
        self._on_stream_delta = on_stream_delta

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Start the loop thread, build the SDK client, connect. Idempotent —
        returns the session marker (SDK session ids arrive on first result)."""
        if self._client is not None:
            return self._session_id or "pending"
        # Hard rule, enforced fail-closed: this provider exists to bill the
        # Claude SUBSCRIPTION. If a metered ANTHROPIC_API_KEY is present the
        # underlying CLI would silently prefer it — refuse to start instead.
        allow_metered = _provider_flag(
            "allow_metered_key", "HERMES_CLAUDE_SDK_ALLOW_API_KEY"
        )
        for metered_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            if os.environ.get(metered_var) and not allow_metered:
                raise RuntimeError(
                    f"claude-agent-sdk runtime refuses to start: {metered_var} "
                    "is set, which would silently switch billing from the "
                    "Claude subscription to metered API usage. Unset it, or "
                    "set agent.claude_agent_sdk.allow_metered_key: true in "
                    "config.yaml (env override HERMES_CLAUDE_SDK_ALLOW_API_KEY=1) "
                    "to explicitly allow it."
                )
        if self._client_factory is None:
            ok, msg = check_claude_sdk_available()
            if not ok:
                raise RuntimeError(msg)

        self._start_loop_thread()
        client = self._build_client()
        # Assign BEFORE connect: a connect timeout/cancel leaves a
        # half-connected client whose CLI subprocess close() must still reap
        # — a None _client would skip disconnect and orphan it.
        self._client = client
        self._run_coro(client.connect(), timeout=60.0)
        logger.info(
            "claude-agent-sdk session started: model=%s mode=%s cwd=%s",
            self._model or "cli-default",
            self._permission_mode,
            self._cwd,
        )
        return self._session_id or "pending"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._loop is not None:
            try:
                self._run_coro(self._client.disconnect(), timeout=10.0)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._client = None
        self._stop_loop_thread()

    def __enter__(self) -> "ClaudeAgentSdkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def consume_interrupt(self) -> None:
        """Clear a pending interrupt signal — the caller honored it through
        another path (e.g. the runtime's cold-agent short-circuit)."""
        self._interrupt_event.clear()

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to interrupt and unwind."""
        self._interrupt_event.set()
        if self._client is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.interrupt(), self._loop
                )
            except Exception:  # pragma: no cover
                logger.debug("SDK interrupt scheduling failed", exc_info=True)

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
    ) -> TurnResult:
        """Send a user message and block until the SDK's ResultMessage,
        projecting the typed stream into Hermes' messages shape."""
        result = TurnResult()
        try:
            self.ensure_started()
        except Exception as exc:
            hint = classify_auth_failure(str(exc))
            result.error = hint or f"claude-agent-sdk startup failed: {exc}"
            result.should_retire = True
            return result

        # An interrupt that arrived between turns or during connect (up to
        # 60s) targets THIS turn — honor it instead of erasing it. (The old
        # unconditional clear() silently swallowed that window.)
        if self._interrupt_event.is_set():
            self._interrupt_event.clear()
            result.interrupted = True
            return result
        text = _coerce_turn_input_text(user_input)

        try:
            turn_data = self._run_coro(
                self._consume_turn(text), timeout=turn_timeout
            )
        except asyncio.TimeoutError:
            self.request_interrupt()
            result.interrupted = True
            result.error = f"turn timed out after {turn_timeout:.0f}s"
            result.should_retire = True
            return result
        except Exception as exc:
            hint = classify_auth_failure(str(exc))
            result.error = hint or f"claude-agent-sdk turn failed: {exc}"
            result.should_retire = True
            return result

        result.final_text = turn_data["final_text"]
        result.projected_messages = turn_data["messages"]
        result.tool_iterations = turn_data["tool_iterations"]
        result.token_usage_last = turn_data["usage"]
        result.token_usage_total = turn_data["usage"]
        result.thread_id = self._session_id
        result.turn_id = turn_data.get("result_uuid")
        result.interrupted = self._interrupt_event.is_set()
        if result.interrupted:
            # Consume the honored interrupt so it cannot bleed into the
            # next turn on this session object.
            self._interrupt_event.clear()
        if turn_data["error"]:
            hint = classify_auth_failure(turn_data["error"])
            result.error = hint or turn_data["error"]
            if hint is not None:
                result.should_retire = True
        return result

    # ---------- internals ----------

    async def _consume_turn(self, text: str) -> dict[str, Any]:
        """The async side of one turn: query + drain receive_response()."""
        projector = ClaudeSdkEventProjector()
        out: dict[str, Any] = {
            "final_text": "",
            "messages": [],
            "tool_iterations": 0,
            "usage": None,
            "error": None,
            "result_uuid": None,
        }
        await self._client.query(text)
        async for message in self._client.receive_response():
            # Capture the SDK session id from ANY message that carries it —
            # the init SystemMessage announces it first, so even a turn
            # interrupted before its ResultMessage keeps a resumable id.
            early_sid = getattr(message, "session_id", None)
            if early_sid:
                self._session_id = early_sid
            if self._interrupt_event.is_set():
                break
            if type(message).__name__ == "StreamEvent":
                self._forward_stream_delta(message)
                continue
            self._notify_tool_started(message)
            projection = projector.project(message)
            if projection.messages:
                out["messages"].extend(projection.messages)
            if projection.is_tool_iteration:
                out["tool_iterations"] += 1
            if projection.final_text is not None:
                out["final_text"] = projection.final_text
            if projection.is_result:
                usage = getattr(message, "usage", None)
                if isinstance(usage, dict):
                    out["usage"] = dict(usage)
                sid = getattr(message, "session_id", None)
                if sid:
                    self._session_id = sid
                out["result_uuid"] = getattr(message, "uuid", None)
                subtype = getattr(message, "subtype", "") or ""
                if getattr(message, "is_error", False):
                    errors = getattr(message, "errors", None) or []
                    out["error"] = (
                        f"SDK result error (subtype={subtype}): "
                        + ("; ".join(str(e) for e in errors) or subtype)
                    )
                elif subtype not in ("", "success"):
                    # e.g. error_max_turns / error_max_budget_usd — surface
                    # honestly; the partial transcript is still projected.
                    out["error"] = f"SDK turn ended: {subtype}"
        return out

    def _forward_stream_delta(self, message: Any) -> None:
        """Relay a top-level text delta to the display callback (never the
        transcript). Subagent streams (parent_tool_use_id set) stay quiet."""
        if self._on_stream_delta is None:
            return
        if getattr(message, "parent_tool_use_id", None):
            return
        event = getattr(message, "event", None) or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return
        text = delta.get("text")
        if not text:
            return
        try:
            self._on_stream_delta(text)
        except Exception:  # pragma: no cover - display callback
            logger.debug("stream delta callback raised", exc_info=True)

    def _notify_tool_started(self, message: Any) -> None:
        """Bridge ToolUseBlocks to Hermes tool-progress (gateway breadcrumbs),
        mirroring codex_runtime._codex_note_to_tool_progress (#38835)."""
        if self._on_tool_started is None:
            return
        if type(message).__name__ != "AssistantMessage":
            return
        for block in getattr(message, "content", None) or []:
            if type(block).__name__ != "ToolUseBlock":
                continue
            name = getattr(block, "name", "") or "unknown"
            args = getattr(block, "input", None) or {}
            if not isinstance(args, dict):
                args = {"input": args}
            preview = _tool_preview(name, args)
            try:
                self._on_tool_started(name, preview, args)
            except Exception:  # pragma: no cover - display callback
                logger.debug("tool-progress callback raised", exc_info=True)

    def build_option_fields(self) -> dict[str, Any]:
        """The ClaudeAgentOptions field dict — plain data so tests can assert
        on it without importing the SDK."""
        mcp_servers: dict[str, Any] = {}
        if self._include_hermes_tools:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config(
                hermes_session_id=self._hermes_session_id
            )

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if self._system_prompt_append:
            system_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt_append,
            }

        can_use_tool = None
        if (
            self._approval_callback is not None
            and self._permission_mode == "default"
        ):
            can_use_tool = self._make_can_use_tool()

        fields = {
            "model": self._model,
            "cwd": self._cwd,
            "permission_mode": self._permission_mode,
            "system_prompt": system_prompt,
            "mcp_servers": mcp_servers,
            "max_budget_usd": self._max_budget_usd,
            "can_use_tool": can_use_tool,
        }
        if self._resume_session_id:
            fields["resume"] = self._resume_session_id
        # Default OFF (upstream-conservative): partial messages only when the
        # operator opts in via agent.claude_agent_sdk.streaming in config.yaml
        # (or the HERMES_CLAUDE_SDK_STREAMING deployment override).
        if _provider_flag("streaming", "HERMES_CLAUDE_SDK_STREAMING"):
            fields["include_partial_messages"] = True
        return fields

    def _build_client(self) -> Any:
        fields = self.build_option_fields()
        if self._client_factory is not None:
            return self._client_factory(options=fields)
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        return ClaudeSDKClient(options=ClaudeAgentOptions(**fields))

    def _make_can_use_tool(self) -> Any:
        """Bridge SDK permission requests onto Hermes' approval callback.
        Fail-closed: any callback failure denies."""
        approval_callback = self._approval_callback

        async def _can_use_tool(tool_name: str, tool_input: dict, context: Any):
            from claude_agent_sdk import (
                PermissionResultAllow,
                PermissionResultDeny,
            )

            try:
                choice = await asyncio.to_thread(
                    approval_callback,
                    f"{tool_name}({_tool_preview(tool_name, tool_input)})",
                    f"Claude requests tool {tool_name}",
                    allow_permanent=False,
                )
            except Exception:
                logger.exception("approval_callback raised on SDK permission")
                return PermissionResultDeny(message="approval callback failed")
            if choice in ("once", "session", "always"):
                return PermissionResultAllow()
            return PermissionResultDeny(message="denied by user")

        return _can_use_tool

    # ---------- loop-thread plumbing ----------

    def _start_loop_thread(self) -> None:
        if self._loop_thread is not None:
            return
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name="claude-sdk-loop", daemon=True
        )
        thread.start()
        ready.wait(timeout=10)
        self._loop = loop
        self._loop_thread = thread

    def _stop_loop_thread(self) -> None:
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:  # pragma: no cover
                pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None

    def _run_coro(self, coro: Any, *, timeout: float) -> Any:
        import concurrent.futures

        assert self._loop is not None, "loop thread not started"
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            raise asyncio.TimeoutError(f"coroutine exceeded {timeout}s")


def _tool_preview(name: str, args: dict) -> str:
    """Short human preview of a tool call for progress breadcrumbs."""
    for key in ("command", "file_path", "path", "url", "query", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    return name


def _coerce_turn_input_text(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into plain text input (same
    contract as the codex session's _coerce_turn_input_text)."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)
