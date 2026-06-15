"""OpenAI-compatible shim that forwards Hermes requests to Claude Code over ACP.

Unlike :mod:`agent.copilot_acp_client`, which scrapes ``<tool_call>`` blocks
from flat text output, this client consumes the **native** ACP events emitted
by ``@agentclientprotocol/claude-agent-acp`` — Claude Code runs its own tool loop
autonomously, so tool-call facts arrive as first-class ``session/update``
messages of kind ``tool_call_start`` and ``tool_call_update``.

The client:

* Builds a per-session Claude Code sandbox (``CLAUDE.md``, skills,
  ``.mcp.json``) via :mod:`agent.claude_code_sandbox` so hermes's persona,
  memory, skills, and tools are available inside the subprocess.
* Keeps one ACP subprocess + one ``sessionId`` alive across turns, matching
  the Claude Code editor lifetime. Subsequent calls reuse the session.
* Routes streamed text → ``stream_delta_callback``, thought chunks →
  ``thinking_callback``, and tool events → ``tool_progress_callback``.
* Accumulates a :class:`ToolCallRecord` list and attaches it to the
  response as ``hermes_tool_trace`` so the AIAgent loop can feed it into
  auto-skill-creation and background review without re-running the tools.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Iterable, Iterator, List, Optional

from agent.file_safety import get_read_block_error, is_write_denied
from gateway.session_context import get_session_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (inlined from former _acp_client_base.py)
# ---------------------------------------------------------------------------

def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


class AcpCancelled(RuntimeError):
    """Raised when an in-flight ACP request is cancelled by the caller.

    Subclass of :class:`RuntimeError` on purpose: the run_agent retry loop
    catches broad ``Exception`` for transient transport errors, and
    cancellation should unwind the same way without being mistaken for a
    retryable failure. Callers that need to distinguish (e.g. to suppress
    the usual error surfacing) should catch ``AcpCancelled`` explicitly.
    """


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path '{resolved}' is outside the session cwd '{root}'."
        ) from exc
    return resolved


def _walk_text_blocks(obj: Any) -> Iterator[str]:
    """Yield text fragments from any nested OpenAI/ACP content shape.

    Handles the union of shapes we see in the wild:
      - ``str`` → yield as-is
      - ``{"text": "..."}`` → yield the text value
      - ``{"content": ...}`` → recurse into the inner content
      - ``list`` → recurse into each item
    Anything else (e.g. image/tool blocks) is skipped silently. Callers
    decide how to frame the output (strip/join separator, etc.).
    """
    if obj is None:
        return
    if isinstance(obj, str):
        if obj:
            yield obj
        return
    if isinstance(obj, dict):
        text = obj.get("text")
        if isinstance(text, str) and text:
            yield text
            return
        inner = obj.get("content")
        if inner is not None and inner is not obj:
            yield from _walk_text_blocks(inner)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _walk_text_blocks(item)


_NON_TEXT_BLOCK_TYPES = {"image", "image_url", "input_audio", "audio", "file"}
_MULTIMODAL_WARNED = False


def _warn_once_multimodal_stripped(block_types: set[str]) -> None:
    """Log once per process when non-text content (image/audio/…) is dropped."""
    global _MULTIMODAL_WARNED
    if _MULTIMODAL_WARNED or not block_types:
        return
    _MULTIMODAL_WARNED = True
    logger.warning(
        "ACP prompt path is text-only; dropped content block(s): %s. "
        "Attach the image via an on-disk path the model can read, or use "
        "a provider that supports multimodal prompts.",
        ", ".join(sorted(block_types)),
    )


def _render_message_content(content: Any) -> str:
    """Flatten a message ``content`` field to a single stripped string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        stripped_types: set[str] = set()
        for item in content:
            if isinstance(item, dict):
                block_type = str(item.get("type") or "")
                if block_type in _NON_TEXT_BLOCK_TYPES:
                    stripped_types.add(block_type)
        if stripped_types:
            _warn_once_multimodal_stripped(stripped_types)
        parts = [t.strip() for t in _walk_text_blocks(content) if t.strip()]
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    *,
    preamble: list[str] | None = None,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    tool_call_instructions: str | None = None,
    closing: str | None = "Continue the conversation from the latest user request.",
) -> str:
    """Flatten an OpenAI-style ``messages`` list into one ACP prompt string."""
    sections: list[str] = list(preamble or [])
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            header = tool_call_instructions or (
                "Available tools (OpenAI function schema)."
            )
            sections.append(f"{header}\n{json.dumps(tool_specs, ensure_ascii=False)}")

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    if closing:
        sections.append(closing)
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def resolve_effective_timeout(timeout: Any, *, default: float) -> float:
    """Translate run_agent's ``timeout`` kwarg (plain number or httpx.Timeout) to seconds."""
    if timeout is None:
        return default
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else default


# ---------------------------------------------------------------------------
# Claude Code ACP client constants
# ---------------------------------------------------------------------------

DEFAULT_CLAUDE_CODE_ACP_PACKAGE = "@agentclientprotocol/claude-agent-acp"
ACP_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0

# Canonical system-preamble lines for Claude Code under ACP. Exported so
# `agent.claude_code_sandbox` can share the same source of truth when it
# composes CLAUDE.md — we used to maintain two near-duplicate copies that
# drifted, this is the single authoritative list.
#
# Each entry is an independent paragraph. Callers choose how to assemble
# (per-prompt `"\n\n".join(...)` for the client; CLAUDE.md markdown for the
# sandbox). Do NOT insert markdown headers here — the sandbox adds those
# when it composes the on-disk file.
CLAUDE_SYSTEM_PREAMBLE_LINES: list[str] = [
    "You are the active reasoning engine for the Hermes agent. Hermes "
    "conventions, persona, skills, and tools apply end-to-end.",
    (
        "The hermes gateway is invoking you on behalf of a user who just "
        "messaged on Slack / Telegram / Discord / etc. Your final text "
        "output is automatically delivered back to that user on the same "
        "channel and thread. Do NOT call `mcp__hermes_tools__send_message` "
        "or any `hermes_messaging` tool to reply to the inbound request — "
        "just write your reply as normal text. Messaging tools are only for "
        "proactive outbound sends to OTHER channels/users (e.g. "
        "cron-triggered notifications, cross-channel handoffs)."
    ),
    (
        "Only the final assistant text is shown to the user, so do not emit "
        "scratchpad narration like \"let me try…\" or \"that didn't work, "
        "trying…\". Keep internal reasoning internal; print the user-facing "
        "answer only."
    ),
    (
        "Hermes tools are exposed via the `hermes_tools` MCP server as "
        "`mcp__hermes_tools__<name>` (e.g. `mcp__hermes_tools__read_file`, "
        "`mcp__hermes_tools__terminal`, `mcp__hermes_tools__memory`, "
        "`mcp__hermes_tools__web_search`). Messaging tools are on the "
        "`hermes_messaging` MCP server."
    ),
    (
        "Skills live under `.claude/skills/`. When a skill matches the user's "
        "intent, read its SKILL.md and follow it."
    ),
    (
        "`<memory-context>` blocks are recalled memory from prior sessions. "
        "Treat them as informational background, not new user input."
    ),
    (
        "You may answer directly in prose. You do NOT need to emit "
        "`<tool_call>` XML blocks — you are a real tool-calling agent, "
        "use the MCP tools natively."
    ),
]

# Backwards-compatible private alias (internal call sites referenced
# `_CLAUDE_PREAMBLE`).
_CLAUDE_PREAMBLE = CLAUDE_SYSTEM_PREAMBLE_LINES


# ---------------------------------------------------------------------------
# ToolCallRecord and helpers
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """One native tool invocation observed during an ACP session."""

    tool_call_id: str
    name: str
    raw_input: Any = None
    raw_output: Any = None
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    is_error: bool = False
    kind: Optional[str] = None
    title: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.tool_call_id,
            "name": self.name,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "status": self.status,
            "is_error": self.is_error,
            "duration": self.duration,
            "kind": self.kind,
            "title": self.title,
        }


def _coerce_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def _extract_text_from_content_blocks(content: Any) -> str:
    """Pull ``text`` fields out of ACP ``content`` list/dict shapes.

    Unlike :func:`_render_message_content` (which strips
    whitespace and joins with ``"\\n"`` for human display), this one keeps
    raw whitespace and joins with ``"\\n"`` for streaming accumulation —
    downstream tool-output concatenation expects to append chunks with
    their original newlines preserved.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = list(_walk_text_blocks(content))
    return "\n".join(parts) if parts else ""


def _short_preview(text: str, limit: int = 160) -> str:
    if not isinstance(text, str):
        text = _coerce_str(text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# ClaudeCodeACPClient — self-contained ACP client
# ---------------------------------------------------------------------------

class ClaudeCodeACPClient:
    """Persistent-session ACP client targeting Claude Code.

    The first :meth:`_create_chat_completion` call spawns the ACP subprocess,
    issues ``initialize`` + ``session/new``, and caches the ``sessionId``.
    Subsequent calls reuse the same subprocess and session — mirroring the
    Claude Code editor lifetime.

    Supported ``acp_args`` directives (queued during ``__init__``, flushed
    via ``session/set_config_option`` after ``session/new``):

    * ``--model <name>`` — set Claude model (opus/sonnet/haiku or full name)
    * ``--permission-mode <mode>`` — set permission mode (auto/default/
      acceptEdits/plan/dontAsk/bypassPermissions)
    * ``--effort <level>`` — set reasoning effort (default/low/medium/high/
      xhigh/max)

    Unrecognised flags in ``acp_args`` are dropped — they are NOT passed
    to the subprocess. The subprocess argv is the fixed
    ``_launch_command + _launch_args`` from ``_resolve_command()`` /
    ``_resolve_args()`` (default: ``npx -y @agentclientprotocol/claude-agent-acp``).
    """

    _provider_label = "claude-code-acp"
    _default_command = "npx"
    _default_args = ("-y", DEFAULT_CLAUDE_CODE_ACP_PACKAGE)
    _env_command_vars = ("HERMES_CLAUDE_CODE_ACP_COMMAND", "CLAUDE_ACP_PATH")
    _env_args_var = "HERMES_CLAUDE_CODE_ACP_ARGS"
    _marker_base_url = ACP_MARKER_BASE_URL
    _default_timeout_seconds = _DEFAULT_TIMEOUT_SECONDS
    _client_name = "hermes-agent"

    # Declarative dispatch table: maps recognized CLI flags from acp_args
    # to their ACP session config actions. Each entry defines:
    #   flag: the CLI flag to match (e.g. "--model")
    #   channel: "session_config" (via session/set_config_option) or "init"
    #   config_id: the ACP configId for session_config channel entries
    #   attr: the instance attribute to set for init channel entries
    #   takes_value: True if the flag consumes the next token as its value
    #   validate: optional method name (str) for value validation; None = trust caller
    #   repeatable: if False, only the first occurrence is honoured
    _ACP_ARG_DIRECTIVES: ClassVar[tuple[dict[str, Any], ...]] = (
        {
            "flag": "--model",
            "channel": "session_config",
            "config_id": "model",
            "takes_value": True,
            "validate": "_is_valid_claude_alias",
            "repeatable": False,
        },
        {
            "flag": "--permission-mode",
            "channel": "session_config",
            "config_id": "mode",
            "takes_value": True,
            "validate": None,  # ACP server validates: auto/default/acceptEdits/plan/dontAsk/bypassPermissions
            "repeatable": False,
        },
        {
            "flag": "--effort",
            "channel": "session_config",
            "config_id": "effort",
            "takes_value": True,
            "validate": "_is_valid_effort_level",
            "repeatable": False,
        },
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        agent: Any | None = None,
        hermes_home: Path | None = None,
        hermes_session_id: str | None = None,
        platform: str | None = None,
        stream_delta_callback: Callable[[str], Any] | None = None,
        thinking_callback: Callable[[str], Any] | None = None,
        tool_progress_callback: Callable[..., Any] | None = None,
        available_tools: Optional[set[str]] = None,
        available_toolsets: Optional[set[str]] = None,
        **_: Any,
    ) -> None:
        # ACP transport state (formerly from _AcpClientBase.__init__)
        self.api_key = api_key or self._provider_label
        self.base_url = base_url or self._marker_base_url
        self._default_headers = dict(default_headers or {})
        # Fixed subprocess launch argv — always npx + npm package, never user-configurable.
        # The acp_command parameter passed by delegate_task is a routing placeholder
        # (e.g. "claude-code-acp") to distinguish Claude vs Copilot providers — it is
        # NOT used for subprocess argv. The real launch command comes from
        # _resolve_command() (env var / default "npx"); the launch args come from
        # _resolve_args() (env var / default ["-y", "<npm package>"]).
        self._launch_command = self._resolve_command()
        self._launch_args = self._resolve_args()
        # Runtime ACP config directives (--model, --effort, --permission-mode).
        # Parsed by _apply_acp_arg_directives() into _pending_session_configs,
        # flushed to session/set_config_option after session/new.
        self._acp_args = list(acp_args or args or [])
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._next_request_id = 0
        self._stderr_log_path: Optional[Path] = None

        # ClaudeCode-specific state
        self._agent = agent
        self._hermes_home = Path(hermes_home) if hermes_home else None
        self._hermes_session_id = (
            hermes_session_id
            or get_session_env("HERMES_SESSION_ID", "").strip()
            or f"hermes-{int(time.time())}"
        )
        self._platform = platform or os.environ.get("HERMES_SESSION_PLATFORM") or None
        self._available_tools = set(available_tools) if available_tools else None
        self._available_toolsets = set(available_toolsets) if available_toolsets else None

        self.stream_delta_callback = stream_delta_callback
        self.thinking_callback = thinking_callback
        self.tool_progress_callback = tool_progress_callback

        self._session_lock = threading.Lock()
        self._session_proc: subprocess.Popen[str] | None = None
        self._session_inbox: "queue.Queue[dict[str, Any]]" | None = None
        self._session_stderr: deque[str] | None = None
        self._session_id: str | None = None
        self._sandbox_path: Path | None = None
        self._pending_model: str | None = None
        self._pending_session_configs: dict[str, str] = {}

        # Scan _acp_args (runtime directives from caller) for recognized
        # flags (--model, --permission-mode, --effort) and queue them into
        # _pending_session_configs. These are flushed via
        # session/set_config_option after session/new in _ensure_session().
        # Unrecognised flags are dropped — they are NOT passed to the subprocess
        # (the subprocess argv is the fixed _launch_command + _launch_args
        # from _resolve_command() / _resolve_args()).
        self._apply_acp_arg_directives()

        self._trace_lock = threading.Lock()
        self._tool_trace: list[ToolCallRecord] = []
        self._tool_records_by_id: dict[str, ToolCallRecord] = {}

    # ------------------------------------------------------------------
    # Env-driven resolution of the launcher path + args
    # ------------------------------------------------------------------

    def _resolve_command(self) -> str:
        for env_var in self._env_command_vars:
            val = os.getenv(env_var, "").strip()
            if val:
                return val
        return self._default_command

    def _resolve_args(self) -> list[str]:
        if self._env_args_var:
            raw = os.getenv(self._env_args_var, "").strip()
            if raw:
                return shlex.split(raw)
        return list(self._default_args)

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._session_lock:
            self._session_id = None
            self._session_inbox = None
            self._session_stderr = None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
        self._session_proc = None

    def _start_subprocess(self) -> tuple[subprocess.Popen[str], "queue.Queue[dict[str, Any]]", deque[str]]:
        """Launch the ACP subprocess and start reader threads.

        Returns ``(proc, inbox, stderr_tail)``. The caller owns the process
        lifetime: persistent clients keep the reference; one-shot clients
        terminate via :meth:`close` after the request completes.
        """
        try:
            proc = subprocess.Popen(
                [self._launch_command] + self._launch_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
            )
        except FileNotFoundError as exc:
            envs = "/".join(self._env_command_vars) if self._env_command_vars else "the launcher path"
            raise RuntimeError(
                f"Could not start {self._provider_label} command '{self._launch_command}'. "
                f"Install it or set {envs}."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"{self._provider_label} process did not expose stdin/stdout.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: "queue.Queue[dict[str, Any]]" = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        log_file = self._open_stderr_log_file(proc.pid)

        def _stdout_reader() -> None:
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            try:
                for line in proc.stderr:
                    stripped = line.rstrip("\n")
                    stderr_tail.append(stripped)
                    if log_file is not None:
                        try:
                            log_file.write(line if line.endswith("\n") else line + "\n")
                            log_file.flush()
                        except OSError:
                            pass
            finally:
                if log_file is not None:
                    try:
                        log_file.close()
                    except OSError:
                        pass

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()
        return proc, inbox, stderr_tail

    # ------------------------------------------------------------------
    # Stderr logging to disk
    # ------------------------------------------------------------------

    def _open_stderr_log_file(self, pid: int):
        """Open a per-subprocess stderr log under ``<hermes_home>/logs/acp/``."""
        try:
            try:
                from hermes_constants import get_hermes_home
                home = Path(get_hermes_home())
            except Exception:
                home = Path.home() / ".hermes"
            log_dir = home / "logs" / "acp"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            provider = self._provider_label.replace("/", "_")
            path = log_dir / f"{provider}-{pid}-{ts}.log"
            self._stderr_log_path = path
            return path.open("w", encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("Unable to open ACP stderr log: %s", exc)
            self._stderr_log_path = None
            return None

    # ------------------------------------------------------------------
    # JSON-RPC request/response
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._next_request_id += 1
        return self._next_request_id

    def _request(
        self,
        proc: subprocess.Popen[str],
        inbox: "queue.Queue[dict[str, Any]]",
        stderr_tail: deque[str],
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
        dispatch_ctx: Optional[dict[str, Any]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Any:
        """Send a JSON-RPC request and block for the matching response."""
        timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        )
        request_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if cancel_check is not None and cancel_check():
                raise AcpCancelled(
                    f"{self._provider_label} {method} cancelled by caller"
                )
            try:
                msg = inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._handle_server_message(msg, process=proc, dispatch_ctx=dispatch_ctx or {}):
                continue

            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(
                    f"{self._provider_label} {method} failed: {err.get('message') or err}"
                )
            return msg.get("result")

        stderr_text = "\n".join(stderr_tail).strip()
        if proc.poll() is not None and stderr_text:
            log_hint = (
                f"\nFull stderr: {self._stderr_log_path}"
                if self._stderr_log_path is not None
                else ""
            )
            raise RuntimeError(
                f"{self._provider_label} process exited early: {stderr_text}{log_hint}"
            )
        raise TimeoutError(
            f"Timed out waiting for {self._provider_label} response to {method}."
        )

    def _notify(
        self,
        proc: subprocess.Popen[str],
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception:
            logger.debug("Failed to write notification %s", method, exc_info=True)

    # ------------------------------------------------------------------
    # Server-initiated message dispatch
    # ------------------------------------------------------------------

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        dispatch_ctx: dict[str, Any],
    ) -> bool:
        """Return True if this message was a server-initiated notification /
        request and was handled here; False if it is a response the caller
        must correlate by id."""
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            self._handle_session_update(update, dispatch_ctx=dispatch_ctx, params=params)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = self._handle_permission_request(message_id, params)
        elif method == "fs/read_text_file":
            response = self._handle_fs_read(message_id, params)
        elif method == "fs/write_text_file":
            response = self._handle_fs_write(message_id, params)
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        try:
            process.stdin.write(json.dumps(response) + "\n")
            process.stdin.flush()
        except Exception:
            logger.debug("Failed to send response for %s", method, exc_info=True)
        return True

    # ------------------------------------------------------------------
    # Default fs / permission handlers
    # ------------------------------------------------------------------

    def _handle_permission_request(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {"outcome": {"outcome": "allow_once"}},
        }

    def _handle_fs_read(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = _ensure_path_within_cwd(str(params.get("path") or ""), self._acp_cwd)
            block_error = get_read_block_error(str(path))
            if block_error:
                raise PermissionError(block_error)
            content = path.read_text() if path.exists() else ""
            line = params.get("line")
            limit = params.get("limit")
            if isinstance(line, int) and line > 1:
                lines = content.splitlines(keepends=True)
                start = line - 1
                end = start + limit if isinstance(limit, int) and limit > 0 else None
                content = "".join(lines[start:end])
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"content": content},
            }
        except Exception as exc:
            return _jsonrpc_error(message_id, -32602, str(exc))

    def _handle_fs_write(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        try:
            path = _ensure_path_within_cwd(str(params.get("path") or ""), self._acp_cwd)
            if is_write_denied(str(path)):
                raise PermissionError(f"Write denied: '{path}' is protected by safety policy")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(params.get("content") or ""))
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": None,
            }
        except Exception as exc:
            return _jsonrpc_error(message_id, -32602, str(exc))

    # ------------------------------------------------------------------
    # Initialize / session/new params
    # ------------------------------------------------------------------

    def _build_initialize_params(self) -> dict[str, Any]:
        return {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
            },
            "clientInfo": {
                "name": self._client_name,
                "title": "Hermes Agent",
                "version": "0.0.0",
            },
        }

    def _build_session_new_params(self) -> dict[str, Any]:
        sandbox = self._ensure_sandbox()
        mcp_servers = self._load_mcp_servers(sandbox)
        params: dict[str, Any] = {
            "cwd": str(sandbox),
            "mcpServers": mcp_servers,
        }
        return params

    # ------------------------------------------------------------------
    # Sandbox + session lifecycle
    # ------------------------------------------------------------------

    def _resolve_hermes_home(self) -> Path:
        if self._hermes_home is not None:
            return self._hermes_home
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home()
        except Exception:
            return Path.home() / ".hermes"

    def _ensure_sandbox(self, *, model: str | None = None) -> Path:
        if self._sandbox_path is not None:
            return self._sandbox_path
        from agent.claude_code_sandbox import build_session_sandbox

        hermes_home = self._resolve_hermes_home()
        self._sandbox_path = build_session_sandbox(
            self._hermes_session_id,
            self._agent,
            hermes_home=hermes_home,
            platform=self._platform,
            available_tools=self._available_tools,
            available_toolsets=self._available_toolsets,
            model=model or self._pending_model,
        )
        # ACP subprocess must cd into the sandbox so .mcp.json and CLAUDE.md
        # are picked up.
        self._acp_cwd = str(self._sandbox_path)
        return self._sandbox_path

    def _load_mcp_servers(self, sandbox: Path) -> list[dict[str, Any]]:
        """Translate the sandbox's .mcp.json into ACP's mcpServers array shape."""
        mcp_path = sandbox / ".mcp.json"
        if not mcp_path.exists():
            return []
        try:
            cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        servers = cfg.get("mcpServers") or {}
        out: list[dict[str, Any]] = []
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            cmd = spec.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            entry: dict[str, Any] = {
                "name": name,
                "command": cmd,
                "args": list(spec.get("args") or []),
            }
            env = spec.get("env") or {}
            if isinstance(env, dict) and env:
                entry["env"] = [
                    {"name": str(k), "value": str(v)}
                    for k, v in env.items()
                ]
            out.append(entry)
        return out

    def _ensure_session(self) -> tuple[
        subprocess.Popen[str],
        "queue.Queue[dict[str, Any]]",
        deque[str],
        str,
    ]:
        with self._session_lock:
            if (
                self._session_proc is not None
                and self._session_proc.poll() is None
                and self._session_id is not None
                and self._session_inbox is not None
                and self._session_stderr is not None
            ):
                return (
                    self._session_proc,
                    self._session_inbox,
                    self._session_stderr,
                    self._session_id,
                )

            self._ensure_sandbox()
            proc, inbox, stderr_tail = self._start_subprocess()
            try:
                self._request(
                    proc, inbox, stderr_tail,
                    "initialize", self._build_initialize_params(),
                    timeout_seconds=self._default_timeout_seconds,
                )
                session = self._request(
                    proc, inbox, stderr_tail,
                    "session/new", self._build_session_new_params(),
                    timeout_seconds=self._default_timeout_seconds,
                ) or {}
                session_id = str(session.get("sessionId") or "").strip()
                if not session_id:
                    raise RuntimeError(
                        "claude-agent-acp did not return a sessionId from session/new."
                    )
                # Flush all pending session configs from directive scanner.
                # Order matters: model MUST be flushed first because switching
                # models can invalidate subsequent mode/effort settings
                # (e.g. Haiku does not support auto mode).
                _FLUSH_ORDER = ("model", "mode", "effort")
                for config_id in _FLUSH_ORDER:
                    if config_id not in self._pending_session_configs:
                        continue
                    try:
                        self._request(
                            proc, inbox, stderr_tail,
                            "session/set_config_option",
                            {
                                "sessionId": session_id,
                                "configId": config_id,
                                "value": self._pending_session_configs[config_id],
                            },
                            timeout_seconds=self._default_timeout_seconds,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to set %s via session/set_config_option "
                            "(%s: %r); continuing",
                            config_id, type(exc).__name__, exc,
                        )
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise

            self._session_proc = proc
            self._session_inbox = inbox
            self._session_stderr = stderr_tail
            self._session_id = session_id
            return proc, inbox, stderr_tail, session_id

    # ------------------------------------------------------------------
    # session/update routing (native tool events)
    # ------------------------------------------------------------------

    def _handle_session_update(
        self,
        update: dict[str, Any],
        *,
        dispatch_ctx: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        ctx = dispatch_ctx or {}
        kind = str(update.get("sessionUpdate") or "").strip()

        if kind == "agent_message_chunk":
            text = _extract_text_from_content_blocks(update.get("content"))
            if text:
                text_parts = ctx.get("text_parts")
                if isinstance(text_parts, list):
                    text_parts.append(text)
                cb = self.stream_delta_callback
                if callable(cb):
                    try:
                        cb(text)
                    except Exception as exc:
                        logger.debug("stream_delta_callback raised: %s", exc)
            return

        if kind == "agent_thought_chunk":
            text = _extract_text_from_content_blocks(update.get("content"))
            if text:
                reasoning_parts = ctx.get("reasoning_parts")
                if isinstance(reasoning_parts, list):
                    reasoning_parts.append(text)
                cb = self.thinking_callback
                if callable(cb):
                    try:
                        cb(text)
                    except Exception as exc:
                        logger.debug("thinking_callback raised: %s", exc)
            return

        if kind == "tool_call_start":
            self._handle_tool_call_start(update)
            return

        if kind == "tool_call_update":
            self._handle_tool_call_update(update)
            return

        if kind == "plan":
            self._handle_plan_update(update)
            return

        if kind == "available_commands_update":
            return

        logger.debug("Unhandled ACP session/update kind: %s", kind)

    def _handle_tool_call_start(self, update: dict[str, Any]) -> None:
        raw_input = update.get("rawInput")
        if raw_input is None:
            raw_input = update.get("input")
        kind = update.get("kind")
        tool_id_raw = str(update.get("toolCallId") or update.get("id") or "").strip()
        name = str(update.get("title") or update.get("name") or "").strip() or "unknown_tool"

        with self._trace_lock:
            tool_id = tool_id_raw or f"tool-{len(self._tool_trace) + 1}"
            record = ToolCallRecord(
                tool_call_id=tool_id,
                name=name,
                raw_input=raw_input,
                status="in_progress",
                kind=str(kind) if kind else None,
                title=update.get("title"),
            )
            self._tool_trace.append(record)
            self._tool_records_by_id[tool_id] = record

        cb = self.tool_progress_callback
        if callable(cb):
            preview = _short_preview(_coerce_str(raw_input))
            try:
                cb("tool.started", name, preview, raw_input)
            except Exception as exc:
                logger.debug("tool_progress_callback(started) raised: %s", exc)

    def _handle_tool_call_update(self, update: dict[str, Any]) -> None:
        tool_id = str(update.get("toolCallId") or update.get("id") or "").strip()
        raw_output = update.get("rawOutput")
        if raw_output is None:
            raw_output = update.get("output")
        if raw_output is None:
            raw_output = _extract_text_from_content_blocks(update.get("content"))
        status = update.get("status")
        is_error_flag = bool(update.get("isError"))

        with self._trace_lock:
            record = self._tool_records_by_id.get(tool_id)
            if record is None:
                record = ToolCallRecord(
                    tool_call_id=tool_id or f"tool-{len(self._tool_trace) + 1}",
                    name=str(update.get("title") or "unknown_tool"),
                    status="in_progress",
                )
                self._tool_trace.append(record)
                if tool_id:
                    self._tool_records_by_id[tool_id] = record

            if raw_output:
                if record.raw_output is None:
                    record.raw_output = raw_output
                elif isinstance(record.raw_output, str) and isinstance(raw_output, str):
                    record.raw_output = record.raw_output + raw_output
                else:
                    record.raw_output = raw_output

            if isinstance(status, str):
                record.status = status

            if is_error_flag:
                record.is_error = True

            is_terminal = (
                record.status in {"completed", "failed", "cancelled", "success"}
                or update.get("isComplete") is True
                or update.get("complete") is True
            )
            if is_terminal:
                record.completed_at = time.time()
                cb_snapshot = (
                    record.name,
                    _short_preview(_coerce_str(record.raw_output)),
                    record.raw_input,
                    record.duration or 0.0,
                    record.is_error,
                )
            else:
                cb_snapshot = None

        if cb_snapshot is not None:
            cb = self.tool_progress_callback
            if callable(cb):
                name, preview, raw_input_val, duration, is_error_val = cb_snapshot
                try:
                    cb(
                        "tool.completed",
                        name,
                        preview,
                        raw_input_val,
                        duration=duration,
                        is_error=is_error_val,
                    )
                except Exception as exc:
                    logger.debug(
                        "tool_progress_callback(completed) raised: %s", exc
                    )

    def _handle_plan_update(self, update: dict[str, Any]) -> None:
        cb = self.thinking_callback
        if not callable(cb):
            return
        entries = update.get("entries") or update.get("plan") or []
        if not isinstance(entries, list):
            return
        summary_parts: list[str] = []
        for item in entries:
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                if isinstance(content, str) and content.strip():
                    summary_parts.append(content.strip())
            elif isinstance(item, str) and item.strip():
                summary_parts.append(item.strip())
        if not summary_parts:
            return
        try:
            cb("\U0001f4dd plan: " + "; ".join(summary_parts[:3]))
        except Exception as exc:
            logger.debug("thinking_callback(plan) raised: %s", exc)

    # ------------------------------------------------------------------
    # create() entrypoint
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_model_name(raw: str) -> str:
        """Normalise Anthropic model names for Claude Code ACP.

        Strips optional context-window suffixes like ``[1m]``, ``[200k]``
        before alias resolution so that ``opus[1m]`` and ``sonnet[200k]``
        are recognised as valid Claude Code model identifiers.
        """
        if not raw:
            return raw
        import re
        lower = raw.lower()
        # Strip optional context-window suffix, e.g. "opus[1m]" → "opus"
        lower = re.sub(r"\[\w+\]$", "", lower)
        if lower in ("haiku", "sonnet", "opus", "claude", "default"):
            return lower
        # Full names like "claude-opus-4-7" are passed through verbatim — the
        # subprocess's session/set_config_option will receive the full string,
        # and the alias validator below checks the same regex.
        if re.match(r"^claude-(haiku|sonnet|opus)(?:-\d+(?:-\d+)*)?$", lower):
            return lower
        return raw

    def _is_valid_claude_alias(self, name: str) -> bool:
        """Return True if *name* is a Claude Code-recognizable model identifier.

        Accepts both short aliases (opus/sonnet/haiku) and full names
        (claude-opus-4-7, claude-sonnet-4-6, etc.) so that callers can pin
        a specific Claude Code version without losing the per-session
        ``set_config_option`` call that isolates parallel subagents.
        """
        normalized = self._normalize_model_name(name)
        if normalized in ("haiku", "sonnet", "opus", "claude", "default"):
            return True
        import re
        if re.match(r"^claude-(haiku|sonnet|opus)(?:-\d+(?:-\d+)*)?$", normalized):
            return True
        return False

    def _is_valid_effort_level(self, value: str) -> bool:
        """Return True if *value* is a valid ACP effort level.

        Valid values (from @agentclientprotocol/claude-agent-acp buildConfigOptions):
        default, low, medium, high, xhigh, max
        """
        return value.lower() in ("default", "low", "medium", "high", "xhigh", "max")

    def _apply_acp_arg_directives(self) -> None:
        """Scan ``self._acp_args`` for recognized flags and queue them.

        Each matched flag is queued into ``_pending_session_configs`` (for
        ``session_config`` channel directives).  Unrecognised flags are
        dropped — they are NOT passed to the ACP subprocess (the subprocess
        argv is the fixed ``_launch_command + _launch_args``).

        Only the first occurrence of each non-repeatable flag is honoured.
        """
        recognized_flags = {d["flag"]: d for d in self._ACP_ARG_DIRECTIVES}
        seen: set[str] = set()
        i = 0
        while i < len(self._acp_args):
            arg = self._acp_args[i]
            if arg not in recognized_flags:
                i += 1
                continue
            directive = recognized_flags[arg]

            # Skip duplicate non-repeatable flags
            if not directive.get("repeatable", False) and arg in seen:
                i += (2 if directive["takes_value"] else 1)
                continue
            seen.add(arg)

            if directive["takes_value"]:
                if i + 1 >= len(self._acp_args):
                    logger.warning(
                        "ClaudeCodeACPClient: %s flag has no value, skipping",
                        arg,
                    )
                    i += 1
                    continue
                value = self._acp_args[i + 1]
            else:
                value = "true"  # boolean flag

            # Validate if a validator is specified
            validator_name = directive.get("validate")
            if validator_name:
                validator = getattr(self, validator_name, None)
                if validator and not validator(value):
                    logger.warning(
                        "ClaudeCodeACPClient: ignoring %s=%r — validation failed",
                        arg,
                        value,
                    )
                    i += (2 if directive["takes_value"] else 1)
                    continue

            # Dispatch to the appropriate channel
            channel = directive["channel"]
            if channel == "session_config":
                config_id = directive["config_id"]
                self._pending_session_configs[config_id] = value
                # Backward compat: also set _pending_model for --model
                if config_id == "model":
                    self._pending_model = value
                    logger.info(
                        "ClaudeCodeACPClient: seeded _pending_model=%r from "
                        "acp_args --model flag",
                        value,
                    )
            elif channel == "init":
                attr = directive["attr"]
                setattr(self, attr, value)

            i += (2 if directive["takes_value"] else 1)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        raw_model = model
        model = self._normalize_model_name(model) if model else model
        prompt_text = _format_messages_as_prompt(
            messages or [],
            preamble=_CLAUDE_PREAMBLE,
            model=model,
            tools=None,
            tool_choice=None,
            tool_call_instructions=None,
            closing=(
                "Continue the conversation from the latest user request. "
                "Use the hermes MCP tools natively."
            ),
        )
        effective_timeout = resolve_effective_timeout(
            timeout, default=self._default_timeout_seconds
        )

        if raw_model and self._is_valid_claude_alias(raw_model):
            self._pending_model = raw_model
            self._pending_session_configs["model"] = raw_model

        proc, inbox, stderr_tail, session_id = self._ensure_session()

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        with self._trace_lock:
            pre_trace_len = len(self._tool_trace)

        def _cancel_check() -> bool:
            return bool(getattr(self._agent, "_interrupt_requested", False))

        try:
            result = self._request(
                proc, inbox, stderr_tail,
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
                timeout_seconds=effective_timeout,
                dispatch_ctx={
                    "text_parts": text_parts,
                    "reasoning_parts": reasoning_parts,
                },
                cancel_check=_cancel_check if self._agent is not None else None,
            )
        except Exception as exc:
            logger.warning(
                "Claude Code ACP prompt failed (%s: %r); closing session",
                type(exc).__name__,
                exc,
            )
            self.close()
            raise

        response_text = "".join(text_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()

        if isinstance(result, dict):
            tail = _extract_text_from_content_blocks(
                result.get("content") or result.get("message")
            )
            if tail and tail not in response_text:
                response_text = (response_text + "\n" + tail).strip() if response_text else tail

        with self._trace_lock:
            turn_trace_dicts = [r.to_dict() for r in self._tool_trace[pre_trace_len:]]

        usage = self._extract_usage(result if isinstance(result, dict) else None)

        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        response = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-code-acp",
            hermes_tool_trace=turn_trace_dicts,
        )
        return response

    def _extract_usage(self, result: Optional[dict[str, Any]]) -> Any:
        """Best-effort extraction of token usage from a session/prompt result."""
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(result, dict):
            usage = result.get("usage") or {}
            if isinstance(usage, dict):
                prompt_tokens = int(
                    usage.get("inputTokens")
                    or usage.get("prompt_tokens")
                    or 0
                )
                completion_tokens = int(
                    usage.get("outputTokens")
                    or usage.get("completion_tokens")
                    or 0
                )
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )

    # ------------------------------------------------------------------
    # Helpers exposed for tests / run_agent integration
    # ------------------------------------------------------------------

    @property
    def tool_trace(self) -> list[ToolCallRecord]:
        """Full list of tool records observed in this client's lifetime."""
        with self._trace_lock:
            return list(self._tool_trace)

    def reset_tool_trace(self) -> None:
        with self._trace_lock:
            self._tool_trace.clear()
            self._tool_records_by_id.clear()


# ---------------------------------------------------------------------------
# _ACPChatNamespace / _ACPChatCompletions — OpenAI-shaped .chat shim
# ---------------------------------------------------------------------------

class _ACPChatCompletions:
    def __init__(self, client: "ClaudeCodeACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ClaudeCodeACPClient"):
        self.completions = _ACPChatCompletions(client)


def trace_to_messages_snapshot(
    trace: Iterable[Any],
    *,
    fallback_name: str = "claude_code_tool",
) -> list[dict[str, Any]]:
    """Reconstruct a hermes-shape ``messages_snapshot`` from a tool trace.

    ``trace`` may be an iterable of :class:`ToolCallRecord` instances OR of
    dict records (as produced by :attr:`ToolCallRecord.to_dict`) — both are
    accepted so auto-skill-creation can consume the serialized form.
    Each record becomes an ``assistant`` message containing a ``tool_use``
    block, followed by a ``tool`` message with the ``tool_result`` block.
    """
    messages: list[dict[str, Any]] = []
    for record in trace:
        if hasattr(record, "to_dict"):
            data = record.to_dict()
        elif isinstance(record, dict):
            data = record
        else:
            continue
        call_id = str(data.get("id") or data.get("tool_call_id") or "").strip()
        if not call_id:
            call_id = f"tool-{len(messages) + 1}"
        name = str(data.get("name") or fallback_name).strip() or fallback_name
        raw_input = data.get("raw_input")
        raw_output = data.get("raw_output")

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": raw_input if raw_input is not None else {},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_use_id": call_id,
                "content": _coerce_str(raw_output) if raw_output is not None else "",
            }
        )
    return messages
