"""OpenAI-ChatCompletion-shaped client backed by a PERSISTENT `claude -p`.

This is the "live session" provider path for claude-cli. Unlike the per-turn
``ClaudeCLIClient`` (which spawns a fresh subprocess every call and describes
tools in the prompt), this client keeps ONE warm ``claude`` child per Hermes
conversation and exposes Hermes's real tools to it through a local stdio MCP
server. The model calls tools in-process, so the prompt cache stays hot across
tool cycles (the whole reason live mode exists).

IPC decision (the hard part):
  Claude drives the tool cycle itself and reaches the tools through a SEPARATE
  MCP bridge process (``agent/hermes_mcp_bridge.py``) that it spawns via
  ``--mcp-config``. That bridge cannot execute Hermes tools — it has no
  ``AIAgent`` (conversation/session/guardrail state). So it forwards every
  ``tools/call`` over an AF_UNIX socket to THIS parent process, where
  ``LiveToolServer`` runs the call through the injected executor
  (``agent._invoke_tool`` — the same synchronous dispatcher Hermes's own loop
  uses; falls back to ``model_tools.handle_function_call`` for headless/aux
  use). Result flows back through the socket → bridge → claude, all within the
  one warm process. A shared-secret token guards the socket.

Consequence for Hermes's loop: one ``create()`` call is one full user turn,
including every internal tool cycle. It returns the final assistant text with
``finish_reason="stop"`` (tool calls were already executed via the real
dispatcher, with their normal display/guardrails). Only the NEW user turn is
sent each call — the warm process remembers prior turns — so Hermes should keep
its system prompt stable (volatile per-turn context belongs in the user turn;
a changed system prompt is a fingerprint change and forces a respawn).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import tempfile
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from agent.claude_cli_client import (
    CLAUDE_CLI_MARKER_BASE_URL,
    _build_openai_tool_call,
    _normalize_effort,
    _normalize_model,
    _render_message_content,
    _resolve_command,
    _resolve_extra_args,
    _resolve_home_dir,
)
from agent.claude_live_session import (
    LiveSessionConfig,
    TurnResult,
    get_registry,
)
from tools.environments.local import hermes_subprocess_env

# Env credentials/routing that must NEVER reach the child: they would switch
# `claude` from the user's OAuth/subscription session to pay-per-token API
# billing, a cloud-provider gateway (Bedrock/Vertex — entirely separate billing),
# or a foreign endpoint. CLAUDE_CODE_OAUTH_TOKEN + HOME are kept (that IS the
# subscription auth). `hermes_subprocess_env` already blocks ANTHROPIC_API_KEY /
# ANTHROPIC_BASE_URL / ANTHROPIC_TOKEN; these are the additional live-mode keys.
_AUTH_SCRUB_KEYS = (
    # Direct API / OAuth credential overrides.
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",  # could inject an x-api-key / auth header.
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
    # Cloud-gateway routing — any of these diverts billing off the subscription.
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_VERTEX_PROJECT_ID",
)

_TOOL_NAMESPACE = "hermes"
_ALLOWED_TOOLS_GLOB = f"mcp__{_TOOL_NAMESPACE}__*"

# The last stable line of the system prompt file: an explicit cache boundary so
# everything above it is a single stable cache prefix.
_CACHE_BOUNDARY_MARKER = "\n<!-- hermes-cache-boundary -->"


# ---------------------------------------------------------------------------
# Overage guard
# ---------------------------------------------------------------------------


def live_mode_enabled() -> bool:
    """Live session mode is ON by default; documented off-switch reverts to the
    per-turn ``ClaudeCLIClient``. Set ``HERMES_CLAUDE_CLI_LIVE=0`` to disable."""
    raw = os.getenv("HERMES_CLAUDE_CLI_LIVE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def make_agent_tool_executor(agent: Any) -> ToolExecutor:
    """Bind a live-mode tool executor to a Hermes ``AIAgent`` instance.

    Uses ``agent._invoke_tool`` — the same synchronous dispatcher Hermes's own
    loop calls — so agent-stateful tools (todo/memory/clarify/delegate) and
    registry tools alike run with their real guardrails and session state."""

    def _execute(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        task_id = (
            getattr(agent, "_current_task_id", "")
            or getattr(agent, "session_id", "")
            or ""
        )
        try:
            result = agent._invoke_tool(name, arguments, task_id, messages=None)
            return _stringify_tool_result(result), False
        except Exception as exc:
            return f"tool '{name}' failed: {exc}", True

    return _execute


class ClaudeOverageError(RuntimeError):
    """Raised when a rate_limit_event reports the subscription is on overage."""


def _overage_mode() -> str:
    return (os.getenv("HERMES_CLAUDE_LIVE_OVERAGE", "abort").strip().lower() or "abort")


def check_overage(rate_limit_events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the rate_limit_info of the first overage event, else None."""
    for evt in rate_limit_events:
        info = evt.get("rate_limit_info") if isinstance(evt, dict) else None
        if isinstance(info, dict) and info.get("isUsingOverage") is True:
            return info
    return None


# ---------------------------------------------------------------------------
# Tool executor bridge (parent side of the socket)
# ---------------------------------------------------------------------------

# executor(name, arguments) -> (content_str, is_error)
ToolExecutor = Callable[[str, dict[str, Any]], "tuple[str, bool]"]


def _default_executor(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Headless fallback: dispatch straight to the registry (covers every tool
    except the few that need a live AIAgent — todo/memory/clarify/delegate)."""
    try:
        from model_tools import handle_function_call

        result = handle_function_call(name, arguments)
        return _stringify_tool_result(result), False
    except Exception as exc:
        return f"tool '{name}' failed: {exc}", True


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and result.get("_multimodal"):
        # The live model only consumes text; summarize multimodal payloads.
        return json.dumps({"multimodal": True, "blocks": len(result.get("content") or [])})
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


class LiveToolServer:
    """AF_UNIX server that executes bridge-forwarded tool calls in the parent.

    One per client, serving all of its warm sessions. Tool executions are
    serialized under a lock (claude may fire parallel tool_use blocks; Hermes's
    per-agent state is not built for concurrent invocation)."""

    def __init__(self, executor: ToolExecutor):
        self._executor = executor
        self._token = secrets.token_hex(16)
        self._exec_lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._path: str = ""
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False

    @property
    def token(self) -> str:
        return self._token

    @property
    def socket_path(self) -> str:
        return self._path

    def start(self) -> None:
        if self._started:
            return
        # Short base dir keeps the AF_UNIX path under the ~104-char macOS limit.
        base = os.getenv("TMPDIR", "/tmp").rstrip("/")
        self._path = os.path.join(base, f"hermes-live-{secrets.token_hex(6)}.sock")
        if os.path.exists(self._path):
            os.unlink(self._path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self._path)
        sock.listen(16)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._started = True

    def _serve(self) -> None:
        while not self._closed and self._sock is not None:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle_conn, args=(conn,), daemon=True
            ).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            payload = _recv_line(conn)
            response = self._process(payload)
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _process(self, payload: str) -> dict[str, Any]:
        try:
            request = json.loads(payload)
        except Exception:
            return {"content": "bad request", "is_error": True}
        if request.get("token") != self._token:
            return {"content": "unauthorized", "is_error": True}
        name = request.get("tool")
        arguments = request.get("arguments")
        if not isinstance(name, str) or not name.strip():
            return {"content": "missing tool name", "is_error": True}
        if not isinstance(arguments, dict):
            arguments = {}
        with self._exec_lock:
            try:
                content, is_error = self._executor(name.strip(), arguments)
            except Exception as exc:
                return {"content": f"tool '{name}' raised: {exc}", "is_error": True}
        return {"content": content, "is_error": bool(is_error)}

    def close(self) -> None:
        self._closed = True
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._path and os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass


def _recv_line(conn: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break
    return b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Stable per-fingerprint file materialization
# ---------------------------------------------------------------------------


def _live_state_dir() -> Path:
    base = Path(os.getenv("TMPDIR", "/tmp")) / "hermes-claude-live"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_stable(name_prefix: str, content: str, suffix: str) -> tuple[str, str]:
    """Write ``content`` to a content-addressed file (idempotent). Returns
    (path, sha16). Stable filename → byte-stable path across turns."""
    from agent.claude_live_session import _sha256_short

    digest = _sha256_short(content)
    path = _live_state_dir() / f"{name_prefix}-{digest}{suffix}"
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    return str(path), digest


def _system_prompt_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and str(message.get("role")).lower() == "system":
            rendered = _render_message_content(message.get("content"))
            if rendered:
                parts.append(rendered)
    body = "\n\n".join(parts).strip()
    return body + _CACHE_BOUNDARY_MARKER


def _render_turn_slice(messages: list[dict[str, Any]]) -> str:
    """Render a slice of the conversation (system messages dropped) into one
    stream-json user turn, labelling assistant/tool turns so a fresh process can
    read the prior exchange as a transcript."""
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "system":
            continue
        text = _render_message_content(message.get("content"))
        if text:
            label = {"user": "", "tool": "Tool result:\n"}.get(role, f"{role.title()}:\n")
            rendered.append(f"{label}{text}")
    return "\n\n".join(rendered).strip()


def _delta_user_text(messages: list[dict[str, Any]]) -> str:
    """The NEW turn to send to a WARM process: everything after the last
    assistant message, minus system messages (the process already holds prior
    turns in its own transcript, so resending them would duplicate context)."""
    last_assistant = -1
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and str(message.get("role")).lower() == "assistant":
            last_assistant = idx
    tail = messages[last_assistant + 1:] if last_assistant >= 0 else messages
    return _render_turn_slice(tail)


def _full_history_text(messages: list[dict[str, Any]]) -> str:
    """The seed turn for a FRESH process (first turn, or a fingerprint-drift
    respawn / eviction / orphan reseed): the entire conversation minus system
    messages. Without this, a respawn triggered by a live /model or /effort
    switch would send only the latest turn and silently lose all prior context."""
    return _render_turn_slice(messages)


# ---------------------------------------------------------------------------
# Environment / auth hardening
# ---------------------------------------------------------------------------


def build_live_subprocess_env() -> dict[str, str]:
    """OAuth-only env for the child: keep CLAUDE_CODE_OAUTH_TOKEN + HOME, scrub
    every API-key / gateway credential that would divert billing or identity."""
    env = hermes_subprocess_env(inherit_credentials=True)
    for key in _AUTH_SCRUB_KEYS:
        env.pop(key, None)
    env["HOME"] = _resolve_home_dir()
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    return env


def _auth_identity(env: dict[str, str]) -> str:
    """Stable identity of the auth the child will use (for the fingerprint)."""
    from agent.claude_live_session import _sha256_short

    token = env.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if token:
        return "oauth:" + _sha256_short(token)
    return "home:" + env.get("HOME", "")


# ---------------------------------------------------------------------------
# OpenAI-facade plumbing
# ---------------------------------------------------------------------------


class _LiveChatCompletions:
    def __init__(self, client: "ClaudeLiveClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _LiveChatNamespace:
    def __init__(self, client: "ClaudeLiveClient"):
        self.completions = _LiveChatCompletions(client)


class ClaudeLiveClient:
    """Persistent-session claude-cli client (OpenAI-client-compatible facade)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        cwd: str | None = None,
        acp_cwd: str | None = None,
        tool_executor: ToolExecutor | None = None,
        session_key: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._command = command or acp_command or _resolve_command()
        self._extra_args = list(args or acp_args or _resolve_extra_args())
        self._cwd = str(Path(cwd or acp_cwd or os.getcwd()).resolve())
        self._session_key_override = session_key
        self._fallback_session_key = "claude-live:" + secrets.token_hex(8)
        self._tool_server = LiveToolServer(tool_executor or _default_executor)
        self.chat = _LiveChatNamespace(self)
        self.is_closed = False
        # Hermes drops old clients with ``agent.client = None`` and never calls
        # close(). The tool server's accept thread strongly pins the server (and,
        # via its executor closure, the whole AIAgent), so without this the FD /
        # thread / agent leak forever on every client replacement. A finalizer
        # closes the server when this client is garbage-collected (and runs the
        # full set at interpreter exit), which unblocks accept() → the thread
        # exits → everything is released. Keyed on the client, so it never pins
        # the client itself.
        self._finalizer = weakref.finalize(self, self._tool_server.close)

    def close(self) -> None:
        self.is_closed = True
        self._finalizer()  # idempotent; also cancels the weakref finalizer

    # -- config materialization --------------------------------------------

    def _mcp_config_path(self) -> tuple[str, str]:
        """Write the --mcp-config that points claude's bridge at our socket."""
        bridge = str(Path(__file__).with_name("hermes_mcp_bridge.py"))
        config = {
            "mcpServers": {
                _TOOL_NAMESPACE: {
                    "command": os.getenv("HERMES_CLAUDE_LIVE_PYTHON", "python3"),
                    "args": [bridge],
                    "env": {
                        "HERMES_MCP_BRIDGE_SOCKET": self._tool_server.socket_path,
                        "HERMES_MCP_BRIDGE_TOKEN": self._tool_server.token,
                        "HERMES_MCP_BRIDGE_TOOLS": self._tools_path,
                    },
                }
            }
        }
        return _write_stable("mcp-config", json.dumps(config, sort_keys=True), ".json")

    def _build_argv(
        self, *, model: str, effort: str, mcp_config_path: str, system_prompt_path: str
    ) -> tuple[str, ...]:
        argv = [
            self._command,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--replay-user-messages",
            "--model", model,
            "--effort", effort,
            "--setting-sources", "user",
            "--strict-mcp-config",
            "--mcp-config", mcp_config_path,
            "--allowedTools", _ALLOWED_TOOLS_GLOB,
            "--permission-mode", "bypassPermissions",
            "--append-system-prompt-file", system_prompt_path,
        ]
        argv.extend(self._extra_args)
        return tuple(argv)

    def _build_config(
        self,
        *,
        model: str,
        effort: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LiveSessionConfig:
        from agent.hermes_mcp_bridge import translate_openai_tools_to_mcp

        self._tool_server.start()
        mcp_tools = translate_openai_tools_to_mcp(tools)
        self._tools_path, _ = _write_stable(
            "tools", json.dumps(mcp_tools, sort_keys=True), ".json"
        )
        mcp_config_path, mcp_hash = self._mcp_config_path()
        system_prompt = _system_prompt_text(messages)
        system_prompt_path, sys_hash = _write_stable("sysprompt", system_prompt, ".txt")
        env = build_live_subprocess_env()
        argv = self._build_argv(
            model=model,
            effort=effort,
            mcp_config_path=mcp_config_path,
            system_prompt_path=system_prompt_path,
        )
        return LiveSessionConfig(
            command=self._command,
            argv=argv,
            cwd=self._cwd,
            env=env,
            model=model,
            effort=effort,
            system_prompt_hash=sys_hash,
            mcp_config_hash=mcp_hash,
            auth_identity=_auth_identity(env),
        )

    def _session_key(self) -> str:
        return self._session_key_override or self._fallback_session_key

    # -- main entrypoint ----------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        resolved_model = _normalize_model(model)
        effort_hint = None
        session_hint = None
        if isinstance(extra_body, dict):
            effort_hint = extra_body.get("_hermes_claude_effort")
            session_hint = extra_body.get("_hermes_claude_session_id")
        resolved_effort = _normalize_effort(effort_hint)
        if isinstance(session_hint, str) and session_hint.strip():
            self._session_key_override = "claude-live:" + session_hint.strip()

        config = self._build_config(
            model=resolved_model,
            effort=resolved_effort,
            messages=messages or [],
            tools=tools,
        )
        result = self._run_turn(config, messages or [])

        info = check_overage(result.rate_limit_events)
        if info is not None:
            self._log_rate_limit(info)
            if _overage_mode() == "abort":
                raise ClaudeOverageError(
                    "claude-cli live session: subscription is on OVERAGE "
                    f"(rateLimitType={info.get('rateLimitType')}, "
                    f"resetsAt={info.get('resetsAt')}). Aborting turn "
                    "(set HERMES_CLAUDE_LIVE_OVERAGE=allow to proceed)."
                )

        completion = self._build_completion(result, resolved_model)
        if stream:
            from agent.claude_cli_client import _completion_to_stream_chunks

            return _completion_to_stream_chunks(completion)
        return completion

    @staticmethod
    def _turn_text(session: Any, messages: list[dict[str, Any]]) -> str:
        """Full prior history for a fresh process (seed), delta for a warm one.

        A warm process already holds the transcript, so resending history would
        duplicate context and evict the cache; a fresh process (first turn or a
        fingerprint-drift respawn from a /model or /effort switch) holds nothing
        and must be seeded with the whole conversation or it loses all context."""
        if session.has_prior_context:
            return _delta_user_text(messages)
        return _full_history_text(messages)

    def _run_turn(
        self, config: LiveSessionConfig, messages: list[dict[str, Any]]
    ) -> TurnResult:
        """Send one turn; on a dead/orphaned session, recover once and retry.

        The full-vs-delta choice is remade against whichever session actually
        handles the turn (a recovered/respawned session is fresh and reseeds)."""
        registry = get_registry()
        key = self._session_key()
        session = registry.get_or_create(key, config)
        try:
            result = session.send_turn(
                self._turn_text(session, messages),
                fresh=not session.has_prior_context,
            )
        except RuntimeError:
            recovered = registry.recover(key)
            if recovered is None:
                raise
            result = recovered.send_turn(
                self._turn_text(recovered, messages),
                fresh=not recovered.has_prior_context,
            )

        if (result.timed_out or not result.result_event) and not result.text:
            recovered = registry.recover(key)
            if recovered is not None:
                result = recovered.send_turn(
                    self._turn_text(recovered, messages),
                    fresh=not recovered.has_prior_context,
                )
        return result

    # -- result shaping -----------------------------------------------------

    def _build_completion(self, result: TurnResult, model: str) -> SimpleNamespace:
        usage = self._build_usage(result.usage)
        # In live mode tools are executed in-process, so a completed turn has no
        # pending tool_calls. If the turn ended WITHOUT a result (crash/timeout)
        # and left a native tool_use, surface it so Hermes doesn't hang.
        tool_calls: list[ChatCompletionMessageToolCall] = []
        finish_reason = "stop"
        if result.orphaned_tool_use and result.tool_uses:
            for tu in result.tool_uses:
                name = tu.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                call_id = tu.get("id") or f"claude_live_call_{len(tool_calls) + 1}"
                args = tu.get("input")
                tool_calls.append(
                    _build_openai_tool_call(
                        call_id=str(call_id),
                        name=name.strip(),
                        arguments=json.dumps(args or {}, ensure_ascii=False),
                    )
                )
            if tool_calls:
                finish_reason = "tool_calls"

        message = SimpleNamespace(
            content=result.text,
            tool_calls=tool_calls,
            reasoning=result.reasoning or None,
            reasoning_content=result.reasoning or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=model)

    @staticmethod
    def _build_usage(usage: dict[str, int]) -> SimpleNamespace:
        cached = int(usage.get("cache_read_input_tokens", 0))
        prompt = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + cached
        )
        completion = int(usage.get("output_tokens", 0))
        return SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )

    @staticmethod
    def _log_rate_limit(info: dict[str, Any]) -> None:
        import logging

        logging.getLogger(__name__).warning(
            "claude-cli live rate_limit_info: status=%s type=%s resetsAt=%s "
            "overageStatus=%s isUsingOverage=%s",
            info.get("status"),
            info.get("rateLimitType"),
            info.get("resetsAt"),
            info.get("overageStatus"),
            info.get("isUsingOverage"),
        )
