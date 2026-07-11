"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend.

Lifecycle (defaults on):
- **Process reuse** — keep the ACP subprocess warm across turns.
- **Session continuity** — when conversation history is a strict extension of
  the previous request, keep the ACP ``sessionId`` and send only the new
  messages (smaller prompts; tool-loop friendly).

Disable with env:
- ``HERMES_ACP_PROCESS_REUSE=0`` — spawn per request
- ``HERMES_ACP_SESSION_REUSE=0`` — new ``session/new`` every prompt (still
  reuses the process when process reuse is on)
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, is_write_denied
from agent.redact import redact_sensitive_text
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_REUSE_DISABLE_VALUES = frozenset({"0", "false", "no", "off"})


def _acp_process_reuse_enabled() -> bool:
    raw = os.getenv("HERMES_ACP_PROCESS_REUSE", "1").strip().lower()
    return raw not in _REUSE_DISABLE_VALUES


def _acp_session_reuse_enabled() -> bool:
    """Session continuity defaults to on whenever process reuse is on."""
    raw = os.getenv("HERMES_ACP_SESSION_REUSE", "").strip().lower()
    if raw in _REUSE_DISABLE_VALUES:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _acp_process_reuse_enabled()

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
# We require BOTH the literal product name ("gh-copilot") AND a deprecation
# marker, so generic stderr from the NEW `@github/copilot` CLI — whose repo
# is github.com/github/copilot-cli and which legitimately mentions "copilot-cli"
# in its own banners and error messages — doesn't get misclassified as the
# deprecated extension.
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    # Copilot ACP is a model-driving CLI executor: it legitimately needs LLM
    # provider credentials. Route through the central helper so Tier-1 secrets
    # (gateway bot tokens, GitHub auth, infra) are still stripped (#29157).
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _permission_auto_selected(message_id: Any, options: Any) -> dict[str, Any]:
    """Auto-select an allow option so ACP agent tools can run without a UI.

    Preference: allow_always → allow_once → first option. Falls back to
    cancelled when no options are present.
    """
    opts = options if isinstance(options, list) else []
    option_id = None
    for preferred in ("allow_always", "allow_once"):
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            kind = str(opt.get("kind") or "").strip().lower()
            if kind == preferred:
                option_id = opt.get("optionId") or opt.get("option_id")
                if option_id:
                    break
        if option_id:
            break
    if not option_id:
        for opt in opts:
            if not isinstance(opt, dict):
                continue
            kind = str(opt.get("kind") or "").strip().lower()
            if kind.startswith("allow"):
                option_id = opt.get("optionId") or opt.get("option_id")
                if option_id:
                    break
    if not option_id and opts and isinstance(opts[0], dict):
        option_id = opts[0].get("optionId") or opts[0].get("option_id")
    if not option_id:
        return _permission_denied(message_id)
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "optionId": str(option_id),
            }
        },
    }


def _acp_auto_approve_enabled() -> bool:
    """Whether ACP permission prompts should be auto-approved.

    Default on: ACP backends (Devin/Copilot) own their tool loop and Hermes
    has no interactive permission UI in the chat path. Set
    ``HERMES_ACP_AUTO_APPROVE=0`` to restore deny-all.
    """
    raw = os.getenv("HERMES_ACP_AUTO_APPROVE", "1").strip().lower()
    return raw not in _REUSE_DISABLE_VALUES


def _tool_update_text_preview(update: dict[str, Any], *, limit: int = 240) -> str:
    """Best-effort human preview from an ACP tool_call / tool_call_update."""
    title = str(update.get("title") or "").strip()
    chunks: list[str] = []
    content = update.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            # type: content → nested content.text; or plain text fields
            inner = block.get("content")
            if isinstance(inner, dict):
                text = str(inner.get("text") or "").strip()
                if text:
                    chunks.append(text)
            text = str(block.get("text") or "").strip()
            if text:
                chunks.append(text)
            if block.get("type") == "diff":
                path = str(block.get("path") or "").strip()
                if path:
                    chunks.append(f"diff {path}")
    elif isinstance(content, dict):
        text = str(content.get("text") or "").strip()
        if text:
            chunks.append(text)
    raw_in = update.get("rawInput")
    if isinstance(raw_in, dict) and not chunks:
        # compact path/command hints
        for key in ("path", "file", "command", "query", "url", "pattern"):
            if raw_in.get(key):
                chunks.append(f"{key}={raw_in.get(key)}")
                break
    body = " · ".join(chunks).strip()
    if title and body:
        preview = f"{title}: {body}"
    else:
        preview = title or body or "ACP tool"
    if len(preview) > limit:
        return preview[: limit - 1] + "…"
    return preview


def _tool_kind_name(update: dict[str, Any]) -> str:
    kind = str(update.get("kind") or "other").strip().lower() or "other"
    # Prefer a stable synthetic name the Desktop tool strip can show.
    return f"acp_{kind}"


def _message_continuity_key(message: dict[str, Any]) -> tuple[Any, ...]:
    """Stable identity for prefix-matching conversation history."""
    role = str(message.get("role") or "").strip().lower()
    content = _render_message_content(message.get("content"))
    tool_sig: tuple[Any, ...] = ()
    raw_tcs = message.get("tool_calls")
    if isinstance(raw_tcs, list) and raw_tcs:
        names: list[str] = []
        for tc in raw_tcs:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                names.append(str(fn.get("name") or tc.get("name") or ""))
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else getattr(tc, "name", None)
                names.append(str(name or ""))
        tool_sig = tuple(names)
    tool_call_id = str(message.get("tool_call_id") or "")
    return (role, content, tool_sig, tool_call_id)


def _common_message_prefix_len(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
) -> int:
    if not previous:
        return 0
    n = 0
    for left, right in zip(previous, current):
        if not isinstance(left, dict) or not isinstance(right, dict):
            break
        if _message_continuity_key(left) != _message_continuity_key(right):
            break
        n += 1
    return n


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    *,
    continuation: bool = False,
) -> str:
    sections: list[str] = []
    if continuation:
        sections.append(
            "Continue the same ACP session. The messages below are NEW since "
            "the previous prompt — do not restate earlier context unless needed."
        )
    else:
        sections.extend(
            [
                "You are being used as the active ACP agent backend for Hermes.",
                "Use ACP capabilities to complete tasks.",
                "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
                "If no tool is needed, answer normally.",
            ]
        )
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
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

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
        # Keep tool-call-only assistant turns in the transcript for continuity.
        if not rendered and role == "assistant" and message.get("tool_calls"):
            rendered = "[tool_calls]"
        if not rendered and role == "tool":
            rendered = "[tool result]"
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
        heading = "New messages:\n\n" if continuation else "Conversation transcript:\n\n"
        sections.append(heading + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
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
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


def _coalesce_acp_args(
    acp_args: list[str] | None,
    args: list[str] | None,
    default_args_fn,
) -> list[str]:
    """Resolve ACP CLI args without treating ``[]`` as intentional.

    Call sites sometimes pass ``args=[]`` when runtime wiring forgot to
    forward the provider defaults (e.g. oneshot). An empty argv is never a
    valid ACP launch, so fall back to the provider's defaults instead of
    spawning a bare binary — and never leak a sibling provider's defaults
    via truthiness fallbacks.
    """
    if acp_args is not None and len(acp_args) > 0:
        return list(acp_args)
    if args is not None and len(args) > 0:
        return list(args)
    return list(default_args_fn())


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

    # Subclasses (e.g. DevinACPClient) override these so error messages and
    # default model labels match the actual backend.
    _acp_display_name = "Copilot ACP"
    _default_model_name = "copilot-acp"
    _install_hint = (
        "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
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
        **_: Any,
    ):
        self.api_key = api_key or self._default_model_name
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = _coalesce_acp_args(acp_args, args, _resolve_args)
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._reuse_enabled = _acp_process_reuse_enabled()
        self._session_reuse_enabled = _acp_session_reuse_enabled()
        # Transport state for process reuse (guarded by _rpc_lock).
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any]] | None = None
        self._stderr_tail: deque[str] | None = None
        self._next_rpc_id = 0
        self._initialized = False
        self._spawn_count = 0  # test/metrics: how many times we Popen'd
        # Session continuity state (same lock as transport).
        self._session_id: str | None = None
        self._session_history: list[dict[str, Any]] = []
        self._session_count = 0  # test/metrics: session/new calls
        self._session_continues = 0  # test/metrics: prompts reusing sessionId
        # Optional AIAgent (or compatible) for tool_progress / status / activity.
        # Bound after construction via bind_agent_activity() so create_openai_client
        # can wire Desktop/TUI progress without coupling constructors.
        self._activity_agent: Any = None
        # toolCallId → last title (for completed events that omit title)
        self._tool_titles: dict[str, str] = {}

    def bind_agent_activity(self, agent: Any) -> None:
        """Attach an AIAgent so ACP tool/session updates surface in the UI."""
        self._activity_agent = agent

    def _emit_acp_activity(
        self,
        event_type: str,
        name: str,
        preview: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Push tool progress + status so the UI doesn't look frozen mid-turn."""
        agent = self._activity_agent
        if agent is None:
            return
        label = preview or name or "ACP activity"
        try:
            touch = getattr(agent, "_touch_activity", None)
            if callable(touch):
                touch(f"{self._acp_display_name}: {label}")
        except Exception:
            pass
        try:
            cb = getattr(agent, "tool_progress_callback", None)
            if callable(cb):
                cb(event_type, name, preview, args or {}, **kwargs)
        except Exception:
            pass
        # Lifecycle line for gateway/desktop status strip (spinner text).
        if event_type in {"tool.started", "tool.completed"}:
            try:
                emit = getattr(agent, "_emit_status", None)
                if callable(emit):
                    # ASCII-only markers — Windows CP* consoles choke on
                    # ellipsis/check glyphs in status paths.
                    verb = "... " if event_type == "tool.started" else "done "
                    emit(f"{self._acp_display_name} {verb}{label}")
            except Exception:
                pass

    def close(self) -> None:
        """Tear down the ACP subprocess and mark the client closed."""
        with self._rpc_lock:
            self._reset_transport(mark_closed=True)

    def interrupt(self) -> None:
        """Abort any in-flight ACP RPC by killing the warm subprocess.

        Does **not** wait on ``_rpc_lock`` so a blocked ``_rpc`` can observe
        ``poll() != None`` and raise. Leaves the client reusable
        (``is_closed`` stays False); the owning turn resets transport state.
        """
        with self._active_process_lock:
            proc = self._active_process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except Exception:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reset_session_state(self) -> None:
        self._session_id = None
        self._session_history = []

    def _reset_transport(self, *, mark_closed: bool = False) -> None:
        """Kill any live process and clear reuse state. Caller holds ``_rpc_lock``."""
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self._inbox = None
        self._stderr_tail = None
        self._next_rpc_id = 0
        self._initialized = False
        self._reset_session_state()
        if mark_closed:
            self.is_closed = True
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _process_alive(self) -> bool:
        proc = self._active_process
        return proc is not None and proc.poll() is None

    def _normalize_timeout(self, timeout: Any) -> float:
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS
        if isinstance(timeout, (int, float)):
            return float(timeout)
        # httpx.Timeout or similar — pick the largest component so the
        # subprocess has enough wall-clock time for the full response.
        _candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
        return max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        msg_list = [m for m in (messages or []) if isinstance(m, dict)]
        _effective_timeout = self._normalize_timeout(timeout)
        model_name = model or self._default_model_name

        if stream:
            return self._iter_stream_completion(
                msg_list,
                model=model_name,
                tools=tools,
                tool_choice=tool_choice,
                timeout_seconds=_effective_timeout,
            )

        response_text, reasoning_text = self._run_conversation_prompt(
            msg_list,
            model=model_name,
            tools=tools,
            tool_choice=tool_choice,
            timeout_seconds=_effective_timeout,
        )
        return self._build_completion(response_text, reasoning_text, model_name)

    def _build_completion(
        self,
        response_text: str,
        reasoning_text: str,
        model_name: str,
    ) -> SimpleNamespace:
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model_name,
        )

    def _iter_stream_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
    ):
        """Yield OpenAI-style stream chunks as ACP ``agent_message_chunk`` arrives."""
        events: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _on_text(chunk: str) -> None:
            events.put(("text", chunk))

        def _on_reasoning(chunk: str) -> None:
            events.put(("reasoning", chunk))

        def _worker() -> None:
            try:
                text, reasoning = self._run_conversation_prompt(
                    messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout_seconds=timeout_seconds,
                    on_text_chunk=_on_text,
                    on_reasoning_chunk=_on_reasoning,
                )
                events.put(("done", (text, reasoning)))
            except Exception as exc:
                events.put(("error", exc))

        worker = threading.Thread(target=_worker, daemon=True, name="acp-stream-worker")
        worker.start()

        role_sent = False
        while True:
            kind, payload = events.get()
            if kind == "text":
                delta = SimpleNamespace(
                    role="assistant" if not role_sent else None,
                    content=payload,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                )
                role_sent = True
                yield SimpleNamespace(
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                    model=model,
                    usage=None,
                )
            elif kind == "reasoning":
                delta = SimpleNamespace(
                    role=None,
                    content=None,
                    tool_calls=None,
                    reasoning=payload,
                    reasoning_content=payload,
                )
                yield SimpleNamespace(
                    choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)],
                    model=model,
                    usage=None,
                )
            elif kind == "error":
                worker.join(timeout=2)
                raise payload
            elif kind == "done":
                text, reasoning = payload
                completion = self._build_completion(text, reasoning or "", model)
                # Emit terminal tool-call / finish frames (tool calls are only
                # known after the full ACP text is assembled).
                for chunk in _completion_to_stream_chunks(completion):
                    # Skip the bulk content frame if we already streamed tokens
                    # — only keep tool_call + finish + usage frames.
                    choice0 = chunk.choices[0] if chunk.choices else None
                    delta = getattr(choice0, "delta", None) if choice0 else None
                    content = getattr(delta, "content", None) if delta else None
                    tool_calls = getattr(delta, "tool_calls", None) if delta else None
                    finish = getattr(choice0, "finish_reason", None) if choice0 else None
                    if content and not tool_calls and not finish and role_sent:
                        continue
                    if content and role_sent and not tool_calls:
                        # Strip already-streamed content from the finish frame.
                        if delta is not None:
                            delta.content = None
                    yield chunk
                break

        worker.join(timeout=5)

    def _subprocess_env(self) -> dict[str, str]:
        """Environment for the ACP child process. Subclasses may extend."""
        return _build_subprocess_env()

    def _spawn_argv(self) -> list[str]:
        """Argv for the ACP child process. Subclasses may extend."""
        return [self._acp_command] + list(self._acp_args)

    def _prepare_for_model(self, model: str | None) -> None:
        """Hook before initialize/prompt so backends can rebind model state.

        Copilot ACP ignores Hermes model ids (no process-level model switch).
        DevinACPClient overrides this to set DEVIN_MODEL and respawn when needed.
        """
        del model

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        """Hook after ``session/new`` to bind the Hermes-selected model.

        Default no-op. DevinACPClient uses ``session/set_config_option``.
        """
        del session_id, session, model, timeout_seconds

    def _spawn_process(self) -> subprocess.Popen[str]:
        label = self._acp_display_name
        try:
            # Force UTF-8 on the child pipes. On Windows the default console
            # encoding is often cp950/cp1252; Devin/Copilot ACP logs use UTF-8
            # (and occasionally raw bytes), so text=True alone raises
            # UnicodeDecodeError in the stderr reader and can leave the
            # transport half-dead while the UI shows no tokens.
            proc = subprocess.Popen(
                self._spawn_argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                env=self._subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start {label} command '{self._acp_command}'. "
                f"{self._install_hint}"
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"{label} process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)
        self._inbox = inbox
        self._stderr_tail = stderr_tail
        self._next_rpc_id = 0
        self._initialized = False
        self._spawn_count += 1

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()
        return proc

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> Any:
        """Send one JSON-RPC request on the live transport. Caller holds ``_rpc_lock``."""
        label = self._acp_display_name
        proc = self._active_process
        inbox = self._inbox
        stderr_tail = self._stderr_tail
        if proc is None or inbox is None or stderr_tail is None or proc.stdin is None:
            raise RuntimeError(f"{label} transport is not ready.")

        self._next_rpc_id += 1
        request_id = self._next_rpc_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                msg = inbox.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._handle_server_message(
                msg,
                process=proc,
                cwd=self._acp_cwd,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                on_text_chunk=on_text_chunk,
                on_reasoning_chunk=on_reasoning_chunk,
            ):
                continue

            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                raise RuntimeError(
                    f"{label} {method} failed: {err.get('message') or err}"
                )
            return msg.get("result")

        stderr_text = "\n".join(stderr_tail).strip()
        if proc.poll() is not None and stderr_text:
            if _is_gh_copilot_deprecation_message(stderr_text):
                raise RuntimeError(
                    "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                    "(github.com/github/copilot-cli), but the binary it just "
                    "spawned is the deprecated `gh copilot` extension.\n\n"
                    "Install the new CLI:\n"
                    "  npm install -g @github/copilot\n"
                    "  # then verify with: copilot --help\n\n"
                    "If `copilot` already resolves to the new CLI but you still see this,\n"
                    "point Hermes at it explicitly:\n"
                    "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                    "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                    "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                    f"Original error:\n{stderr_text}"
                )
            raise RuntimeError(f"{label} process exited early: {stderr_text}")
        raise TimeoutError(f"Timed out waiting for {label} response to {method}.")

    def _ensure_initialized(self, *, timeout_seconds: float) -> None:
        """Spawn (if needed) and run ACP ``initialize`` once per process."""
        if self._process_alive() and self._initialized:
            return
        if not self._process_alive():
            self._reset_transport(mark_closed=False)
            self._spawn_process()
        self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": True,
                        "writeTextFile": True,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        )
        self._initialized = True

    def _run_conversation_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> tuple[str, str]:
        """Run one completion, reusing process and optionally ACP session."""
        label = self._acp_display_name
        with self._rpc_lock:
            try:
                # Allow backends (Devin) to rebind process-level model before
                # initialize — may tear down a warm process when the model changes.
                self._prepare_for_model(model)
                self._ensure_initialized(timeout_seconds=timeout_seconds)

                prefix_len = 0
                continue_session = False
                if (
                    self._session_reuse_enabled
                    and self._session_id
                    and self._process_alive()
                    and self._initialized
                ):
                    prefix_len = _common_message_prefix_len(self._session_history, messages)
                    if 0 < prefix_len < len(messages):
                        continue_session = True

                if continue_session:
                    prompt_text = _format_messages_as_prompt(
                        messages[prefix_len:],
                        model=model,
                        tools=tools,
                        tool_choice=tool_choice,
                        continuation=True,
                    )
                    session_id = self._session_id
                    assert session_id is not None
                    try:
                        text, reasoning = self._session_prompt(
                            session_id,
                            prompt_text,
                            timeout_seconds=timeout_seconds,
                            on_text_chunk=on_text_chunk,
                            on_reasoning_chunk=on_reasoning_chunk,
                        )
                        self._session_continues += 1
                        self._session_history = list(messages)
                        return text, reasoning
                    except Exception:
                        # Session may have expired — fall through to a fresh
                        # session/new with the full transcript on the same process.
                        self._reset_session_state()

                prompt_text = _format_messages_as_prompt(
                    messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    continuation=False,
                )
                session = self._rpc(
                    "session/new",
                    {
                        "cwd": self._acp_cwd,
                        "mcpServers": [],
                    },
                    timeout_seconds=timeout_seconds,
                ) or {}
                session_id = str(session.get("sessionId") or "").strip()
                if not session_id:
                    raise RuntimeError(f"{label} did not return a sessionId.")
                self._session_id = session_id
                self._session_count += 1
                # Subclasses (Devin) bind model via ACP session/set_config_option;
                # CLI --model is ignored by some ACP agents.
                self._apply_session_model(
                    session_id,
                    session,
                    model,
                    timeout_seconds=timeout_seconds,
                )

                text, reasoning = self._session_prompt(
                    session_id,
                    prompt_text,
                    timeout_seconds=timeout_seconds,
                    on_text_chunk=on_text_chunk,
                    on_reasoning_chunk=on_reasoning_chunk,
                )
                self._session_history = list(messages)
                return text, reasoning
            except Exception:
                # Drop a possibly-poisoned transport so the next call gets a
                # clean process rather than fighting a half-dead inbox.
                self._reset_transport(mark_closed=False)
                raise
            finally:
                if not self._reuse_enabled:
                    self._reset_transport(mark_closed=True)

    def _session_prompt(
        self,
        session_id: str,
        prompt_text: str,
        *,
        timeout_seconds: float,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> tuple[str, str]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Immediate status so the UI shows activity before the first token/tool.
        # Devin often spends 10-60s connecting user MCP servers after
        # session/prompt with zero agent_message_chunk — without this the
        # chat looks frozen / "no opening reply".
        try:
            agent = self._activity_agent
            emit = getattr(agent, "_emit_status", None) if agent else None
            touch = getattr(agent, "_touch_activity", None) if agent else None
            msg = f"{self._acp_display_name} working (may connect MCP tools first)..."
            if callable(touch):
                touch(msg)
            if callable(emit):
                emit(msg)
        except Exception:
            pass
        session_deadline = time.monotonic() + timeout_seconds
        self._rpc(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
            },
            timeout_seconds=timeout_seconds,
            text_parts=text_parts,
            reasoning_parts=reasoning_parts,
            on_text_chunk=on_text_chunk,
            on_reasoning_chunk=on_reasoning_chunk,
        )
        # session/prompt may return stopReason before the final
        # session/update agent_message_chunk is flushed; drain trailing
        # updates until the response text is stable.
        self._drain_session_prompt_chunks(
            text_parts,
            reasoning_parts,
            on_text_chunk=on_text_chunk,
            on_reasoning_chunk=on_reasoning_chunk,
            timeout_seconds=max(0.0, session_deadline - time.monotonic()),
        )
        return "".join(text_parts), "".join(reasoning_parts)

    def _drain_session_prompt_chunks(
        self,
        text_parts: list[str],
        reasoning_parts: list[str],
        *,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
        timeout_seconds: float,
    ) -> None:
        """Drain trailing session/update chunks after session/prompt returns.

        ACP servers (Grok, Devin, Copilot) may send the JSON-RPC response for
        session/prompt before the final agent_message_chunk stream is flushed.
        Wait until no new chunk arrives for a short stable window.
        """
        proc = self._active_process
        inbox = self._inbox
        if proc is None or inbox is None:
            return

        deadline = time.monotonic() + timeout_seconds
        idle_seconds = 0.15
        stable_checks = 2
        stable_count = 0
        while time.monotonic() < deadline:
            remaining = min(idle_seconds, deadline - time.monotonic())
            if remaining <= 0:
                break
            try:
                msg = inbox.get(timeout=remaining)
            except queue.Empty:
                stable_count += 1
                if stable_count >= stable_checks:
                    break
                continue

            stable_count = 0
            if self._handle_server_message(
                msg,
                process=proc,
                cwd=self._acp_cwd,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                on_text_chunk=on_text_chunk,
                on_reasoning_chunk=on_reasoning_chunk,
            ):
                continue

            # Not a notification we can drain; put it back for the next RPC.
            try:
                inbox.put(msg)
            except Exception:
                pass
            break

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        """Low-level single-blob prompt (tests / callers that skip message lists)."""
        return self._run_conversation_prompt(
            [{"role": "user", "content": prompt_text}],
            model=None,
            tools=None,
            tool_choice=None,
            timeout_seconds=timeout_seconds,
        )

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        on_text_chunk: Any = None,
        on_reasoning_chunk: Any = None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            if not isinstance(update, dict):
                update = {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text:
                if text_parts is not None:
                    text_parts.append(chunk_text)
                if on_text_chunk is not None:
                    try:
                        on_text_chunk(chunk_text)
                    except Exception:
                        pass
            elif kind == "agent_thought_chunk" and chunk_text:
                if reasoning_parts is not None:
                    reasoning_parts.append(chunk_text)
                if on_reasoning_chunk is not None:
                    try:
                        on_reasoning_chunk(chunk_text)
                    except Exception:
                        pass
            elif kind in {"tool_call", "tool_call_update"}:
                self._handle_tool_session_update(kind, update)
            elif kind in {
                "available_commands_update",
                "current_mode_update",
                "config_option_update",
                "plan",
            }:
                # Heartbeat so long turns still touch activity even without tools.
                try:
                    agent = self._activity_agent
                    touch = getattr(agent, "_touch_activity", None) if agent else None
                    if callable(touch):
                        touch(f"{self._acp_display_name}: {kind.replace('_', ' ')}")
                except Exception:
                    pass
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            if _acp_auto_approve_enabled():
                response = _permission_auto_selected(
                    message_id, params.get("options")
                )
                # Surface the pending tool in the activity strip.
                tool_call = params.get("toolCall") or params.get("tool_call") or {}
                if isinstance(tool_call, dict):
                    preview = _tool_update_text_preview(tool_call)
                    self._emit_acp_activity(
                        "tool.started",
                        _tool_kind_name(tool_call),
                        preview or "awaiting permission (auto-approved)",
                        {"toolCallId": tool_call.get("toolCallId")},
                    )
            else:
                response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text()
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                if is_write_denied(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' is a protected system/credential file."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""))
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True

    def _handle_tool_session_update(self, kind: str, update: dict[str, Any]) -> None:
        """Map ACP tool_call / tool_call_update notifications onto Hermes progress."""
        tool_id = str(update.get("toolCallId") or update.get("tool_call_id") or "").strip()
        title = str(update.get("title") or "").strip()
        if title and tool_id:
            self._tool_titles[tool_id] = title
        elif tool_id and not title:
            title = self._tool_titles.get(tool_id, "")

        status = str(update.get("status") or "").strip().lower()
        preview = _tool_update_text_preview(
            {**update, "title": title or update.get("title")}
        )
        name = _tool_kind_name(update)
        args: dict[str, Any] = {
            "toolCallId": tool_id or None,
            "kind": update.get("kind"),
            "status": status or None,
        }
        locations = update.get("locations")
        if isinstance(locations, list) and locations:
            paths = []
            for loc in locations[:3]:
                if isinstance(loc, dict) and loc.get("path"):
                    paths.append(str(loc["path"]))
            if paths:
                args["paths"] = paths

        if kind == "tool_call" or status in {"", "pending", "in_progress"}:
            event = "tool.started"
            if status == "in_progress" and kind == "tool_call_update":
                # Keep as started so UIs that only listen for started still refresh
                # the preview; duration is unknown mid-flight.
                event = "tool.started"
            if status in {"completed", "failed"}:
                event = "tool.completed"
            self._emit_acp_activity(
                event,
                name,
                preview,
                args,
                is_error=(status == "failed"),
            )
            return

        if status in {"completed", "failed"}:
            self._emit_acp_activity(
                "tool.completed",
                name,
                preview,
                args,
                is_error=(status == "failed"),
            )
            if tool_id:
                self._tool_titles.pop(tool_id, None)
            return

        # Unknown status — still heartbeat so the UI doesn't freeze.
        self._emit_acp_activity("tool.started", name, preview, args)
