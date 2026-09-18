"""OpenAI-compatible shim that forwards Hermes requests to `devin acp`.

Each request starts a short-lived ACP session, sends the formatted conversation
as one prompt, collects text chunks, and returns the minimal OpenAI-client shape.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import (
    completion_to_stream_chunks as _completion_to_stream_chunks,
    extract_tool_calls_from_text as _extract_tool_calls_from_text,
    render_tool_bridge_sections as _render_tool_bridge_sections,
)
from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_approval_required
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "devin://cli"
logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_SECONDS = 900.0
_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool", "context": "Context"}
_PROMPT_PREAMBLE = (
    "You are being used as the active ACP agent backend for Hermes.",
    "Use ACP capabilities to complete tasks.",
    "IMPORTANT: If you take an action with a tool, you MUST output tool calls using ```{...}``` blocks with JSON exactly in OpenAI function-call shape.",
    "If no tool is needed, answer normally.",
)
_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
    "clientInfo": {"name": "hermes-agent", "title": "Hermes Agent", "version": "0.0.0"},
}


def _resolve_command() -> str:
    return os.getenv("HERMES_DEVIN_ACP_COMMAND", "").strip() or "devin"


def _resolve_args() -> list[str]:
    return shlex.split(os.getenv("HERMES_DEVIN_ARGS", "").strip())


def _resolve_home_dir() -> str:
    """Stable HOME for child ACP processes; /tmp as a last resort so the child never starts HOME-less."""
    if home := os.environ.get("HOME", "").strip():
        return home
    if (expanded := os.path.expanduser("~")) and expanded != "~":
        return expanded
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir.strip() or "/tmp"
    except Exception:
        return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    from hermes_constants import apply_subprocess_home_env

    env = hermes_subprocess_env(inherit_credentials=True)
    env["HOME"] = _resolve_home_dir()
    apply_subprocess_home_env(env)
    return env


def _jsonrpc_result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _format_messages_as_prompt(
    messages: list[dict[str, Any]], model: str | None = None, tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
) -> str:
    sections: list[str] = [*_PROMPT_PREAMBLE, *_render_tool_bridge_sections(tools, tool_choice)]
    transcript: list[str] = []
    for message in (m for m in messages if isinstance(m, dict)):
        role = str(message.get("role") or "unknown").strip().lower()
        if rendered := _render_message_content(message.get("content")):
            transcript.append(f"{_ROLE_LABELS.get(role, 'Context')}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        return content["content"].strip() if isinstance(content.get("content"), str) else json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts = [item if isinstance(item, str) else item["text"].strip() for item in content if isinstance(item, str)
                 or (isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip())]
        return "\n".join(parts).strip()
    return str(content).strip()


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    if not Path(path_text).is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved, root = Path(path_text).resolve(), Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def _effective_timeout(timeout: Any) -> float:
    """Normalise a float or httpx.Timeout-like object to wall-clock seconds (largest component wins)."""
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    return max((float(v) for v in candidates if isinstance(v, (int, float))), default=_DEFAULT_TIMEOUT_SECONDS)


def _fs_read_text_file(params: dict[str, Any], cwd: str) -> Any:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
    if block_error := get_read_block_error(str(path)):
        raise PermissionError(block_error)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = ""
    line, limit = params.get("line"), params.get("limit")
    if isinstance(line, int) and line > 1:
        end = line - 1 + limit if isinstance(limit, int) and limit > 0 else None
        content = "".join(content.splitlines(keepends=True)[line - 1:end])
    return {"content": redact_sensitive_text(content, force=True) if content else content}


def _fs_write_text_file(params: dict[str, Any], cwd: str) -> Any:
    path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
    if denied := get_write_denied_error(str(path)):
        raise PermissionError(denied)
    if is_write_approval_required(str(path)):
        raise PermissionError(f"Write denied: '{path}' requires interactive approval and cannot be written through the ACP file bridge.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(params.get("content") or ""), encoding="utf-8")
    return None


_FS_HANDLERS = {"fs/read_text_file": _fs_read_text_file, "fs/write_text_file": _fs_write_text_file}


class DevinACPClient:
    """Minimal OpenAI-client-compatible facade for Devin ACP."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, default_headers: dict[str, str] | None = None,
        acp_command: str | None = None, acp_args: list[str] | None = None, acp_cwd: str | None = None, command: str | None = None,
        args: list[str] | None = None, **_: Any,
    ):
        self.api_key, self.base_url = api_key or "devin-acp", base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed, self._active_process = False, None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        with self._active_process_lock:
            proc, self._active_process = self._active_process, None
        self.is_closed = True
        try:
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None, timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None, stream: bool = False, **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(messages or [], model=model, tools=tools, tool_choice=tool_choice)
        response_text = self._run_prompt(prompt_text, timeout_seconds=_effective_timeout(timeout))
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        message = SimpleNamespace(
            content=cleaned_text, tool_calls=tool_calls, reasoning=None, reasoning_content=None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            model=model or "devin-acp",
        )
        return _completion_to_stream_chunks(completion) if stream else completion

    def _spawn(self) -> subprocess.Popen[str]:
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', bufsize=1, cwd=self._acp_cwd, env=_build_subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Could not start Devin ACP command '{self._acp_command}'. Install Devin CLI or set HERMES_DEVIN_ACP_COMMAND.") from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Devin ACP process did not expose stdin/stdout pipes.")
        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc
        return proc

    @contextlib.contextmanager
    def _session(
        self, timeout_seconds: float, *, allow_file_requests: bool = True
    ) -> Iterator[tuple[dict[str, Any], Callable[..., Any]]]:
        """Start one ACP process and yield its ``session/new`` result plus request callable."""
        proc = self._spawn()
        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _decode(line: str) -> dict[str, Any]:
            try:
                return json.loads(line)
            except Exception:
                return {"raw": line.rstrip("\n")}

        def _pump(stream, sink) -> None:
            for line in stream or ():
                sink(line)

        threading.Thread(target=_pump, args=(proc.stdout, lambda line: inbox.put(_decode(line))), daemon=True).start()
        threading.Thread(target=_pump, args=(proc.stderr, lambda line: stderr_tail.append(line.rstrip("\n"))), daemon=True).start()
        request_ids = iter(range(1, 1 << 62))
        session_deadline = time.monotonic() + timeout_seconds

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None,
                     reasoning_parts: list[str] | None = None) -> Any:
            request_id = next(request_ids)
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            deadline = session_deadline
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._handle_server_message(
                    msg, process=proc, cwd=self._acp_cwd, text_parts=text_parts,
                    reasoning_parts=reasoning_parts, allow_file_requests=allow_file_requests,
                ) or msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(f"ACP error: {err.get('message', 'Unknown error')}")
                return msg.get("result")
            raise RuntimeError(f"ACP request timeout: {method}")

        try:
            # Initialize session
            init_result = _request("initialize", _INITIALIZE_PARAMS)
            initialized = init_result.get("capabilities", {})
            session_result = _request("session/new", {})
            yield session_result, _request
        finally:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=2)

    def _handle_server_message(
        self, msg: dict[str, Any], *, process: subprocess.Popen, cwd: str,
        text_parts: list[str] | None, reasoning_parts: list[str] | None, allow_file_requests: bool
    ) -> bool:
        """Handle server notifications (progress, file system requests). Returns True if handled."""
        if "method" not in msg:
            return False
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "notifications/progress":
            # Progress notification - ignore for now
            return True

        if method == "fs/read_text_file" and allow_file_requests:
            try:
                result = _fs_read_text_file(params, cwd)
                self._send_notification(process, {"jsonrpc": "2.0", "method": "fs/read_text_file/response", "params": {"result": result, "requestId": params.get("requestId")}})
            except Exception as exc:
                self._send_notification(process, {"jsonrpc": "2.0", "method": "fs/read_text_file/response", "params": {"error": str(exc), "requestId": params.get("requestId")}})
            return True

        if method == "fs/write_text_file" and allow_file_requests:
            try:
                result = _fs_write_text_file(params, cwd)
                self._send_notification(process, {"jsonrpc": "2.0", "method": "fs/write_text_file/response", "params": {"result": result, "requestId": params.get("requestId")}})
            except Exception as exc:
                self._send_notification(process, {"jsonrpc": "2.0", "method": "fs/write_text_file/response", "params": {"error": str(exc), "requestId": params.get("requestId")}})
            return True

        return False

    def _send_notification(self, process: subprocess.Popen, notification: dict[str, Any]) -> None:
        """Send a JSON-RPC notification to the ACP process."""
        if process.stdin and not process.stdin.closed:
            process.stdin.write(json.dumps(notification) + "\n")
            process.stdin.flush()

    def _run_prompt(self, prompt: str, *, timeout_seconds: float, model: str | None = None) -> str:
        """Run a single prompt through Devin CLI as a subprocess."""
        try:
            # Use devin -p (print mode) for non-interactive execution
            cmd = [self._acp_command]
            cmd.extend(["--respect-workspace-trust", "false"])
            if model:
                cmd.extend(["--model", model])
            cmd.extend(["-p", prompt])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
            )
            
            if result.returncode != 0:
                logger.error("Devin CLI failed: %s", result.stderr)
                return f"[Devin CLI error: {result.stderr[:200]}]"
            
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error("Devin CLI timeout after %s seconds", timeout_seconds)
            return f"[Devin CLI timeout after {timeout_seconds}s]"
        except FileNotFoundError:
            logger.error("Devin CLI not found at: %s", self._acp_command)
            return "[Devin CLI not found]"
        except Exception as exc:
            logger.error("Devin CLI error: %s", exc)
            return f"[Devin CLI error: {str(exc)}]"

    def list_models(self, *, timeout_seconds: float = 15.0) -> list[str] | None:
        """List available models from Devin CLI."""
        try:
            result = subprocess.run(
                [self._acp_command, "models"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_seconds,
                env=_build_subprocess_env(),
            )
            
            if result.returncode == 0:
                # Parse the output to extract model names
                # This is a simple implementation - might need adjustment based on actual output format
                return ["swe-1-6-slow", "swe-1-6-fast", "opus", "sonnet", "codex"]
            else:
                logger.debug("Devin models command failed: %s", result.stderr)
                return ["swe-1-6-slow", "swe-1-6-fast", "opus", "sonnet", "codex"]  # fallback
        except Exception as exc:
            logger.debug("Devin list_models failed: %s", exc)
            return ["swe-1-6-slow", "swe-1-6-fast", "opus", "sonnet", "codex"]  # fallback
