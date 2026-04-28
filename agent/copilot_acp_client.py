"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. Each request starts a short-lived ACP session, sends the formatted
conversation as a single prompt, collects text chunks, and converts the result
back into the minimal shape Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.file_safety import get_read_block_error, get_safe_write_root, is_write_denied
from agent.redact import redact_sensitive_text

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)
_ACP_SECRET_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])(sk-[A-Za-z0-9_-]{10,})(?![A-Za-z0-9_-])")
_ACP_SECRET_ENV_RE = re.compile(
    r"([A-Z0-9_]{0,50}(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]{0,50})\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args(command: str = "") -> list[str]:
    """Resolve default ACP args, with command-specific defaults."""
    # Command-specific defaults (no --acp flag needed for non-Copilot CLIs)
    if command:
        cmd_base = os.path.basename(command).lower()
        if cmd_base in _COMMAND_SIMPLE_PIPE_ARGS:
            return _COMMAND_SIMPLE_PIPE_ARGS[cmd_base]
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


# Commands that use simple stdin→stdout pipe (not JSON-RPC ACP).
#
# Permission flags:
#   - claude defaults to `plan` here so generic simple-pipe calls stay
#     read-only. Write-capable recipes must pass explicit acp_args such as
#     `--permission-mode acceptEdits`.
#   - cursor-agent in -p mode has all tools, so read-only defaults must add
#     `--mode plan`; write-capable recipes must opt into `--yolo`.
_COMMAND_SIMPLE_PIPE_ARGS: dict[str, list[str]] = {
    "claude": ["-p", "--output-format", "json", "--permission-mode", "plan"],
    "cursor-agent": ["-p", "--output-format", "text", "--mode", "plan"],
}
_ALLOWED_ACP_COMMANDS = frozenset({"copilot", *_COMMAND_SIMPLE_PIPE_ARGS.keys()})

_SUBPROCESS_ENV_ALLOWLIST = {
    "HOME",
    "PATH",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "USER",
    "LOGNAME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "XDG_DATA_HOME",
    "HERMES_WRITE_SAFE_ROOT",
}
_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|COOKIE|SESSION|AUTH)",
    re.IGNORECASE,
)
_ACP_PROTOCOL_ARGS = {"--acp", "--stdio"}
_CLAUDE_UNRESTRICTED_FLAGS = {"--dangerously-skip-permissions"}
_CLAUDE_UNRESTRICTED_PERMISSION_MODES = {"acceptedits", "bypasspermissions"}
_CURSOR_UNRESTRICTED_FLAGS = {"--yolo"}


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""

    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in _SUBPROCESS_ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value

    # Operators can opt in non-secret project-specific variables when a CLI
    # truly needs them. Secret-looking names remain blocked by default.
    for raw_name in os.environ.get("HERMES_ACP_ENV_ALLOWLIST", "").split(","):
        name = raw_name.strip()
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if _SENSITIVE_ENV_NAME_RE.search(name):
            continue
        value = os.environ.get(name)
        if value is not None:
            env[name] = value

    env["HOME"] = _resolve_home_dir()
    env.setdefault("PATH", os.defpath)
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


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    allow_text_tool_calls: bool = True,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "If no tool is needed, answer normally.",
    ]
    if allow_text_tool_calls:
        sections.insert(
            2,
            "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        )
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if allow_text_tool_calls and isinstance(tools, list) and tools:
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


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
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
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
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


def _redacted_tail(text: str, *, max_lines: int = 40, max_chars: int = 500) -> str:
    redacted = _redact_acp_text(text or "")
    lines = redacted.splitlines()[-max_lines:]
    tail = "\n".join(lines).strip()[:max_chars]
    return tail or "<no stderr>"


def _redact_acp_text(text: str) -> str:
    """Always redact ACP boundary text, independent of global log settings."""
    redacted = redact_sensitive_text(text or "")

    def _env_repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}={match.group(2)}***{match.group(2)}"

    redacted = _ACP_SECRET_ENV_RE.sub(_env_repl, redacted)
    return _ACP_SECRET_TOKEN_RE.sub("***", redacted)


def _command_basename(command: str | None) -> str:
    return os.path.basename(command or "").lower()


def _contains_arg(args: list[str], names: set[str]) -> bool:
    return any(str(arg).split("=", 1)[0] in names for arg in args)


def _permission_mode(args: list[str]) -> str | None:
    for idx, arg in enumerate(args):
        arg = str(arg)
        if arg == "--permission-mode" and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if arg.startswith("--permission-mode="):
            return arg.split("=", 1)[1].strip()
    return None


def _cursor_mode(args: list[str]) -> str | None:
    for idx, arg in enumerate(args):
        arg = str(arg)
        if arg == "--plan":
            return "plan"
        if arg == "--mode" and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1].strip()
    return None


def _validate_acp_args(command: str | None, args: list[str], *, allow_writes: bool) -> list[str]:
    """Reject protocol confusion and unrestricted write flags unless authorized."""
    command_base = _command_basename(command)
    normalized = [str(arg) for arg in args]
    if command_base not in _ALLOWED_ACP_COMMANDS:
        raise ValueError(
            f"ACP command '{command_base or command}' is not supported. "
            "Use one of: claude, cursor-agent, copilot."
        )
    if command_base in _COMMAND_SIMPLE_PIPE_ARGS and _contains_arg(normalized, _ACP_PROTOCOL_ARGS):
        raise ValueError(
            f"ACP command '{command_base}' is configured as simple-pipe but received "
            "--acp/--stdio JSON-RPC flags."
        )

    if command_base == "claude":
        mode = (_permission_mode(normalized) or "").lower()
        has_unrestricted = (
            mode in _CLAUDE_UNRESTRICTED_PERMISSION_MODES
            or _contains_arg(normalized, _CLAUDE_UNRESTRICTED_FLAGS)
        )
        if has_unrestricted and not allow_writes:
            raise PermissionError(
                "Claude ACP args request unrestricted write permission. "
                "Set unsafe_allow_writes=True on the delegated task to allow it."
            )
    elif command_base == "cursor-agent":
        if _contains_arg(normalized, _CURSOR_UNRESTRICTED_FLAGS) and not allow_writes:
            raise PermissionError(
                "Cursor ACP args request --yolo write permission. "
                "Set unsafe_allow_writes=True on the delegated task to allow it."
            )
        mode = (_cursor_mode(normalized) or "").lower()
        if mode not in {"plan", "ask"} and not allow_writes:
            raise PermissionError(
                "Cursor ACP print mode requires --mode plan/ask unless "
                "unsafe_allow_writes=True."
            )
    return normalized


def _terminate_process_tree(proc: subprocess.Popen[str], *, force: bool = False) -> None:
    if proc.poll() is not None:
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except Exception:
            pass
    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


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


def _ensure_cwd_within_safe_root(cwd: str) -> None:
    safe_root = get_safe_write_root()
    if not safe_root:
        return
    resolved = os.path.realpath(os.path.expanduser(cwd))
    root = os.path.realpath(os.path.expanduser(safe_root))
    if resolved != root and not resolved.startswith(root + os.sep):
        raise PermissionError(
            f"Write-capable ACP cwd '{resolved}' is outside HERMES_WRITE_SAFE_ROOT '{root}'."
        )


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

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
        allow_writes: bool = False,
        acp_allow_writes: bool = False,
        unsafe_allow_writes: bool = False,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._allow_writes = bool(allow_writes or acp_allow_writes or unsafe_allow_writes)
        self._acp_args = _validate_acp_args(
            self._acp_command,
            list(acp_args or args or _resolve_args(self._acp_command)),
            allow_writes=self._allow_writes,
        )
        self._is_simple_pipe = (
            self._acp_command
            and os.path.basename(self._acp_command).lower() in _COMMAND_SIMPLE_PIPE_ARGS
        )
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        if self._allow_writes and self._is_simple_pipe:
            _ensure_cwd_within_safe_root(self._acp_cwd)
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            _terminate_process_tree(proc)
            proc.wait(timeout=2)
        except Exception:
            _terminate_process_tree(proc, force=True)
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            allow_text_tool_calls=not self._is_simple_pipe,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
        )

        if self._is_simple_pipe:
            tool_calls, cleaned_text = [], response_text.strip()
        else:
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
            model=model or "copilot-acp",
        )

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        try:
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
                start_new_session=(os.name != "nt"),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Copilot ACP command '{self._acp_command}'. "
                "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Copilot ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        # ── Simple pipe mode (claude -p, cursor-agent -p) ──
        if self._is_simple_pipe:
            return self._run_simple_pipe(proc, prompt_text, timeout_seconds)

        # ── JSON-RPC ACP mode (copilot --acp --stdio) ──
        return self._run_acp_jsonrpc(proc, prompt_text, timeout_seconds)

    def _run_simple_pipe(
        self,
        proc: subprocess.Popen[str],
        prompt_text: str,
        timeout_seconds: float,
    ) -> tuple[str, str]:
        """Simple stdin→stdout pipe for claude -p and cursor-agent -p."""
        stdout_text = ""
        stderr_text = ""
        try:
            stdout_text, stderr_text = proc.communicate(
                input=prompt_text,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(proc, force=True)
            try:
                stdout_text, stderr_text = proc.communicate(timeout=2)
            except Exception:
                pass
            raise TimeoutError(
                f"ACP command '{self._acp_command}' timed out after "
                f"{timeout_seconds:.1f}s. stderr: {_redacted_tail(stderr_text)}"
            ) from exc
        except Exception:
            _terminate_process_tree(proc, force=True)
            raise
        finally:
            self.close()

        stdout_text = (stdout_text or "").strip()
        stderr_text = (stderr_text or "").strip()

        if proc.returncode is None:
            raise RuntimeError(
                f"ACP command '{self._acp_command}' did not exit after output collection."
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ACP command '{self._acp_command}' exited with code {proc.returncode}. "
                f"stderr: {_redacted_tail(stderr_text)}"
            )

        # Try parsing JSON output (claude --output-format json)
        try:
            data = json.loads(stdout_text)
            if not isinstance(data, dict):
                raise ValueError("simple-pipe JSON output was not an object")
            if data.get("is_error") is True or str(data.get("subtype") or "").startswith("error"):
                error_text = data.get("result") or data.get("error") or data.get("message") or stdout_text
                raise RuntimeError(
                    f"ACP command '{self._acp_command}' returned an error response: "
                    f"{_redacted_tail(str(error_text))}"
                )
            result_text = data.get("result", data.get("content", stdout_text))
            if isinstance(result_text, list):
                result_text = "\n".join(
                    item.get("text", str(item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in result_text
                )
            elif not isinstance(result_text, str):
                result_text = json.dumps(result_text)
            reasoning = data.get("reasoning", "")
            return str(result_text), str(reasoning) if reasoning else ""
        except (json.JSONDecodeError, ValueError):
            pass

        return stdout_text, ""

    def _run_acp_jsonrpc(
        self,
        proc: subprocess.Popen[str],
        prompt_text: str,
        timeout_seconds: float,
    ) -> tuple[str, str]:
        """JSON-RPC 2.0 ACP protocol (copilot --acp --stdio)."""

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)
        stderr_lock = threading.Lock()

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
                with stderr_lock:
                    stderr_tail.append(line.rstrip("\n"))

        def _stderr_text() -> str:
            with stderr_lock:
                return "\n".join(list(stderr_tail)).strip()

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
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
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"Copilot ACP {method} failed: "
                        f"{_redacted_tail(str(err.get('message') or err))}"
                    )
                return msg.get("result")

            stderr_text = _stderr_text()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(
                    f"Copilot ACP process exited early: {_redacted_tail(stderr_text)}"
                )
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": self._allow_writes,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Copilot ACP did not return a sessionId.")

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
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
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
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
                if content:
                    content = _redact_acp_text(content)
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
                if not self._allow_writes:
                    raise PermissionError(
                        "Write denied: unsafe_allow_writes is false for this ACP session."
                    )
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
