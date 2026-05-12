"""Native Codex app-server support for delegated subagents."""

from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


_DEFAULT_TIMEOUT_SECONDS = 900.0
_APPROVAL_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
}
_PERMISSIONS_APPROVAL_METHOD = "item/permissions/requestApproval"
_USER_INPUT_METHOD = "item/tool/requestUserInput"


def _split_env_args(value: str) -> list[str]:
    value = (value or "").strip()
    return shlex.split(value, posix=(os.name != "nt")) if value else []


def codex_app_server_command_from_env() -> tuple[str, list[str]]:
    """Resolve the Codex app-server launch command."""

    raw_command = (
        os.getenv("HERMES_CODEX_APP_SERVER_COMMAND", "").strip()
        or os.getenv("CODEX_APP_SERVER_COMMAND", "").strip()
    )
    raw_args = (
        os.getenv("HERMES_CODEX_APP_SERVER_ARGS", "").strip()
        or os.getenv("CODEX_APP_SERVER_ARGS", "").strip()
    )
    if raw_command:
        return raw_command, _split_env_args(raw_args)
    if raw_args:
        return "codex", _split_env_args(raw_args)
    return "codex", ["app-server", "--listen", "stdio://"]


class CodexAppServerSubagent:
    """Delegated subagent backed by the Codex app-server protocol."""

    def __init__(
        self,
        *,
        model: str | None,
        cwd: str | None,
        context: str | None,
        role: str,
        toolsets: list[str] | None,
        max_iterations: int | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        timeout_seconds: float | None = None,
        progress_callback: Optional[Callable[..., None]] = None,
        reasoning_effort: str | None = None,
        approval_policy: str = "never",
        sandbox: str | None = None,
    ) -> None:
        env_command, env_args = codex_app_server_command_from_env()
        self.command = command or env_command
        self.args = list(args if args is not None else env_args)
        self.model = model or ""
        self.cwd = str(Path(cwd or os.getcwd()).resolve())
        self.context = context or ""
        self.role = role
        self.toolsets = list(toolsets or [])
        self.max_iterations = max_iterations or 0
        self.timeout_seconds = float(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS)
        self.tool_progress_callback = progress_callback
        self.reasoning_effort = reasoning_effort
        self.approval_policy = approval_policy
        self.sandbox = sandbox

        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self._api_call_count = 0
        self._current_tool: str | None = None
        self._last_activity_desc = "starting Codex app-server"
        self._interrupt_requested = False
        self._proc: subprocess.Popen[str] | None = None
        self._proc_lock = threading.Lock()
        self._codex_home_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._next_id = 0
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._reader_threads: list[threading.Thread] = []

    def interrupt(self, reason: str | None = None) -> None:
        self._interrupt_requested = True
        self._last_activity_desc = reason or "interrupted"
        if self._thread_id and self._turn_id:
            try:
                self._send_request_no_wait(
                    "turn/interrupt",
                    {"threadId": self._thread_id, "turnId": self._turn_id},
                )
            except Exception:
                pass
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def close(self) -> None:
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            self._cleanup_codex_home()
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._sync_codex_home_back_to_hermes()
        self._cleanup_codex_home()

    def get_activity_summary(self) -> dict[str, Any]:
        return {
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "current_tool": self._current_tool,
            "last_activity_desc": self._last_activity_desc,
        }

    def run_conversation(self, *, user_message: str, task_id: str | None = None) -> dict[str, Any]:
        started = time.monotonic()
        prompt = self._build_prompt(user_message)
        self.session_prompt_tokens = max(1, len(prompt) // 4)
        self._start_process()
        try:
            self._initialize()
            thread = self._thread_start()
            self._thread_id = self._extract_id(thread.get("thread")) or thread.get("threadId")
            if not self._thread_id:
                raise RuntimeError(f"Codex app-server thread/start returned no thread id: {thread!r}")
            turn = self._turn_start(self._thread_id, prompt)
            self._turn_id = self._extract_id(turn.get("turn")) or turn.get("turnId")
            summary = self._wait_for_turn_completion(started)
            self.session_completion_tokens = max(0, len(summary) // 4)
            return {
                "final_response": summary,
                "completed": bool(summary) and not self._interrupt_requested,
                "interrupted": self._interrupt_requested,
                "api_calls": self._api_call_count,
                "messages": [],
            }
        finally:
            self.close()

    def _build_prompt(self, goal: str) -> str:
        parts = [
            "You are a Codex subagent launched by Hermes delegation.",
            "Work in Codex's native harness. Be thorough but concise.",
            "Return a final summary covering what you did, what you found, files changed, and issues encountered.",
        ]
        if self.role:
            parts.append(f"Hermes delegation role: {self.role}.")
        if self.toolsets:
            parts.append(f"Requested Hermes toolset context (informational only; Codex owns the harness): {', '.join(self.toolsets)}.")
        if self.context:
            parts.extend(["", "Context from parent Hermes agent:", self.context.strip()])
        parts.extend(["", "Task:", goal.strip()])
        return "\n".join(parts).strip()

    def _start_process(self) -> None:
        argv = shlex.split(self.command, posix=(os.name != "nt")) + self.args
        if not argv:
            raise RuntimeError("No Codex app-server command configured.")
        env = os.environ.copy()
        self._prepare_codex_home(env)
        self._last_activity_desc = f"launching {' '.join(argv[:3])}"
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not start Codex app-server. Install Codex CLI or set "
                "HERMES_CODEX_APP_SERVER_COMMAND."
            ) from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Codex app-server did not expose stdin/stdout pipes.")
        with self._proc_lock:
            self._proc = proc
        self._reader_threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._reader_threads:
            thread.start()

    def _prepare_codex_home(self, env: dict[str, str]) -> None:
        """Prepare Codex CLI auth from Hermes-managed OAuth when available."""
        if env.get("CODEX_HOME"):
            return
        try:
            from hermes_cli.auth import _read_codex_tokens, resolve_codex_runtime_credentials

            resolve_codex_runtime_credentials(refresh_if_expiring=True)
            data = _read_codex_tokens()
            tokens = data.get("tokens") if isinstance(data, dict) else None
            if not isinstance(tokens, dict):
                return
            if not tokens.get("access_token") or not tokens.get("refresh_token"):
                return
            tmp = tempfile.TemporaryDirectory(prefix="hermes-codex-app-")
            auth_path = Path(tmp.name) / "auth.json"
            payload = {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": tokens,
            }
            last_refresh = data.get("last_refresh")
            if last_refresh:
                payload["last_refresh"] = last_refresh
            auth_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            try:
                auth_path.chmod(0o600)
            except Exception:
                pass
            env["CODEX_HOME"] = tmp.name
            self._codex_home_tmp = tmp
        except Exception:
            return

    def _sync_codex_home_back_to_hermes(self) -> None:
        tmp = self._codex_home_tmp
        if tmp is None:
            return
        auth_path = Path(tmp.name) / "auth.json"
        if not auth_path.is_file():
            return
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
            tokens = payload.get("tokens")
            if not isinstance(tokens, dict):
                return
            if not tokens.get("access_token") or not tokens.get("refresh_token"):
                return
            from hermes_cli.auth import _save_codex_tokens

            _save_codex_tokens(tokens, payload.get("last_refresh"))
        except Exception:
            return

    def _cleanup_codex_home(self) -> None:
        tmp = self._codex_home_tmp
        self._codex_home_tmp = None
        if tmp is not None:
            try:
                tmp.cleanup()
            except Exception:
                shutil.rmtree(tmp.name, ignore_errors=True)

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except Exception:
                self._inbox.put({"raw": line})

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            tail = line.rstrip("\n")
            self._stderr_tail.append(tail)
            del self._stderr_tail[:-80]

    def _write(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("Codex app-server stdin is closed.")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self._next_request_id()
        self._write({"method": method, "id": request_id, "params": params or {}})
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            self._raise_if_dead()
            try:
                msg = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            response = self._handle_message(msg, expected_id=request_id)
            if response is _NoResponse:
                continue
            return response
        raise TimeoutError(f"Timed out waiting for Codex app-server response to {method}.")

    def _send_request_no_wait(self, method: str, params: dict[str, Any] | None = None) -> None:
        request_id = self._next_request_id()
        self._write({"method": method, "id": request_id, "params": params or {}})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"method": method, "params": params or {}})

    def _initialize(self) -> None:
        self._last_activity_desc = "initializing Codex app-server"
        self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "hermes_delegate",
                    "title": "Hermes Delegate",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self._notify("initialized", {})

    def _thread_start(self) -> dict[str, Any]:
        self._last_activity_desc = "starting Codex thread"
        params: dict[str, Any] = {
            "cwd": self.cwd,
            "ephemeral": True,
            "approvalPolicy": self.approval_policy,
        }
        if self.model:
            params["model"] = self.model
        if self.reasoning_effort:
            params["config"] = {"model_reasoning_effort": self.reasoning_effort}
        if self.sandbox:
            params["sandbox"] = self.sandbox
        result = self._request("thread/start", params)
        return result if isinstance(result, dict) else {}

    def _turn_start(self, thread_id: str, prompt: str) -> dict[str, Any]:
        self._last_activity_desc = "starting Codex turn"
        self._api_call_count += 1
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": self.cwd,
            "approvalPolicy": self.approval_policy,
        }
        if self.model:
            params["model"] = self.model
        if self.reasoning_effort:
            params["effort"] = self.reasoning_effort
        result = self._request("turn/start", params)
        return result if isinstance(result, dict) else {}

    def _wait_for_turn_completion(self, started: float) -> str:
        completed = False
        final_messages: list[str] = []
        delta_buffer: list[str] = []
        deadline = started + self.timeout_seconds
        while time.monotonic() < deadline:
            if self._interrupt_requested:
                break
            self._raise_if_dead(allow_running=True)
            try:
                msg = self._inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            method = msg.get("method")
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
            if method == "item/agentMessage/delta":
                delta = params.get("delta")
                if isinstance(delta, str):
                    delta_buffer.append(delta)
                    self._last_activity_desc = "streaming Codex response"
                    self._progress("codex.agent_message.delta", delta[:200])
                continue
            if method == "item/completed":
                item = params.get("item") if isinstance(params.get("item"), dict) else {}
                text = self._extract_agent_message_text(item)
                if text:
                    final_messages.append(text)
                    self._last_activity_desc = "received Codex message"
                continue
            if method == "turn/completed":
                completed = True
                self._last_activity_desc = "Codex turn completed"
                break
            self._handle_message(msg, expected_id=None)
        if not completed and not self._interrupt_requested:
            raise TimeoutError("Timed out waiting for Codex app-server turn/completed.")
        return "\n\n".join(m.strip() for m in final_messages if m.strip()) or "".join(delta_buffer).strip()

    def _handle_message(self, msg: dict[str, Any], *, expected_id: int | None) -> Any:
        if "raw" in msg:
            self._last_activity_desc = "received non-JSON app-server output"
            return _NoResponse
        if "id" in msg and (expected_id is None or msg.get("id") == expected_id) and (
            "result" in msg or "error" in msg
        ):
            if "error" in msg:
                raise RuntimeError(f"Codex app-server error: {msg['error']}")
            return msg.get("result")
        if "id" in msg and "method" in msg:
            self._handle_server_request(msg)
            return _NoResponse
        method = msg.get("method")
        if isinstance(method, str):
            self._handle_notification(method, msg.get("params") if isinstance(msg.get("params"), dict) else {})
        return _NoResponse

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        request_id = msg.get("id")
        method = msg.get("method")
        if method in _APPROVAL_METHODS:
            self._write({"id": request_id, "result": {"decision": "decline"}})
            self._last_activity_desc = f"declined Codex approval request {method}"
            return
        if method == _PERMISSIONS_APPROVAL_METHOD:
            self._write(
                {
                    "id": request_id,
                    "result": {
                        "permissions": {"fileSystem": None, "network": None},
                        "scope": "turn",
                    },
                }
            )
            self._last_activity_desc = "declined Codex permissions request"
            return
        if method == _USER_INPUT_METHOD:
            self._write({"id": request_id, "result": {"answers": {}}})
            self._last_activity_desc = "answered Codex user-input request with no input"
            return
        self._write({"id": request_id, "error": {"code": -32601, "message": f"Unsupported Hermes Codex app-server request: {method}"}})

    def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "turn/started":
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
            self._turn_id = self._extract_id(turn) or self._turn_id
            self._last_activity_desc = "Codex turn running"
            self._progress("codex.turn.started", self._turn_id or "")
        elif method == "item/started":
            item = params.get("item") if isinstance(params.get("item"), dict) else {}
            self._current_tool = item.get("type") if isinstance(item.get("type"), str) else None
            self._last_activity_desc = f"Codex item started: {self._current_tool or 'item'}"
            self._progress("codex.item.started", self._current_tool or "item")
        elif method == "item/completed":
            self._current_tool = None
            self._last_activity_desc = "Codex item completed"
        elif method == "error":
            self._last_activity_desc = f"Codex error notification: {params}"

    def _extract_agent_message_text(self, item: dict[str, Any]) -> str:
        item_type = item.get("type")
        if item_type in {"agentMessage", "agent_message"}:
            text = item.get("text") or item.get("message")
            return text if isinstance(text, str) else ""
        return ""

    def _extract_id(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("id", "threadId", "turnId"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    return candidate
        return None

    def _raise_if_dead(self, *, allow_running: bool = False) -> None:
        proc = self._proc
        if proc is None:
            raise RuntimeError("Codex app-server process was not started.")
        code = proc.poll()
        if code is not None:
            stderr = "\n".join(self._stderr_tail[-20:])
            raise RuntimeError(f"Codex app-server exited with code {code}. {stderr}".strip())

    def _progress(self, event: str, preview: str) -> None:
        if not self.tool_progress_callback:
            return
        try:
            self.tool_progress_callback(event, tool_name="codex-app-server", preview=preview)
        except Exception:
            pass


class _NoResponseType:
    pass


_NoResponse = _NoResponseType()
