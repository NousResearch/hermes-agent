"""Native Pi JSONL RPC client for delegated agents (no ACP bridge).

Option C replacement for the pi-acp bridge: Hermes speaks pi's own
``--mode rpc`` protocol directly, which exposes ``extension_ui_request``
(confirm / select / input / editor). Unlike ACP — which has no free-text
question channel — this lets a delegated pi agent ask the parent a
question and receive a real answer.

Contract parity with the pi-acp bridge is preserved:
- tool activity inside the child is surfaced as ``[pi-tool ...]`` /
  ``[pi-tool:ok|FAILED] ...`` markers in the reasoning stream (parsed by
  ``copilot_acp_client.parse_pi_tool_markers``), and
- every run appends a fenced ``pi-delegation-result`` JSON footer to the
  final message (parsed by ``parse_pi_result_footer``).

Interactive questions:
- An incoming ``extension_ui_request`` is registered in a module-level
  registry (``pending_questions``) and surfaced as a ``[pi-question]``
  marker in the reasoning stream plus the question text in the message
  stream, so the live delegation transcript shows it.
- The run then blocks (up to ``question_timeout_seconds``, default 600)
  waiting for an answer. ``tools.delegate_tool.steer_subagent`` checks
  the registry and routes a steer message to the oldest pending question
  as free text (``{"text": ...}`` for input/editor, option matching for
  select, yes/no parsing for confirm).
- On timeout the old auto-answer policy applies (confirm=true,
  select=first, input/editor=cancelled) so a delegation never hangs
  forever.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    _completion_to_stream_chunks,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
)
from tools.environments.local import hermes_subprocess_env

PI_RPC_MARKER_BASE_URL = "pi://rpc"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_DEFAULT_QUESTION_TIMEOUT = float(os.getenv("HERMES_PI_QUESTION_TIMEOUT", "600"))

# Module-level registry of live questions from all running pi children.
# Key: unique question id. Value: PendingQuestion.
_registry_lock = threading.Lock()
pending_questions: dict[str, "PendingQuestion"] = {}


class PendingQuestion:
    """One unanswered extension_ui_request from a pi child."""

    def __init__(self, method: str, title: str, options: list[str] | None, owner: "PiRPCClient | None" = None):
        self.id = f"q_{int(time.time() * 1000)}_{id(self):x}"
        self.method = method
        self.title = title
        self.options = list(options or [])
        self.owner = owner  # owning client; questions die with their client
        self.answered = threading.Event()
        self.answer: dict[str, Any] | None = None
        self.created_at = time.time()

    def answer_with(self, text: str) -> dict[str, Any]:
        """Map free text onto pi's extension_ui_response dialog shapes.

        Pi's RPC dialogs resolve via:
          input/select -> ``r.cancelled ? undefined : r.value``
          confirm      -> ``r.cancelled ? false : r.confirmed``
        So the wire payload is ``{value}`` / ``{confirmed}`` / ``{cancelled}``
        — NOT the third-party pi-ask-user ``{kind: ...}`` shape.
        """
        cleaned = (text or "").strip()
        low = cleaned.lower()
        if self.method == "confirm":
            is_yes = low in ("y", "yes", "true", "ok", "confirm", "t", "1", "yeah", "yep")
            payload = {"confirmed": is_yes}
        elif self.method == "select":
            # Match by option text first, then by 1‑based index, else pass the
            # raw text through as a freeform selection.
            match = None
            for option in self.options:
                if option and option.lower() == low:
                    match = option
                    break
            if match is None and low.isdigit():
                idx = int(low)
                if 1 <= idx <= len(self.options):
                    match = self.options[idx - 1]
            payload = {"value": match if match is not None else cleaned} if (match is not None or cleaned) else {"cancelled": True}
        else:  # input, editor — free text.
            payload = {"value": cleaned} if cleaned else {"cancelled": True}
        self.answer = payload
        self.answered.set()
        return payload

    def auto_answer(self) -> dict[str, Any]:
        """Default answer for fallback policy."""
        if self.method == "confirm":
            return {"confirmed": True}
        if self.method == "select":
            return {"value": self.options[0]} if self.options else {"cancelled": True}
        return {"cancelled": True}


def answer_oldest_pending_question(text: str) -> bool:
    """Route free text to the oldest pending pi question, if any.

    Questions whose owning client has closed are skipped (their run is
    gone; answering them would steal the text from a live question).
    Returns True when the text was consumed as a question answer.
    """
    with _registry_lock:
        oldest = None
        for question in list(pending_questions.values()):
            if question.owner is not None and question.owner.is_closed:
                pending_questions.pop(question.id, None)
                question.answered.set()  # unblock the dead waiter if any
                continue
            if oldest is None or question.created_at < oldest.created_at:
                oldest = question
        if oldest is None:
            return False
        pending_questions.pop(oldest.id, None)
    oldest.answer_with(text)
    return True


def pending_question_for_owner(owner: "PiRPCClient") -> PendingQuestion | None:
    """Return the oldest live question owned by one persistent Pi client."""
    with _registry_lock:
        matches = [
            question
            for question in pending_questions.values()
            if question.owner is owner and not owner.is_closed
        ]
    return min(matches, key=lambda question: question.created_at) if matches else None


def answer_pending_question_for_owner(owner: "PiRPCClient", text: str) -> bool:
    """Answer only a question belonging to *owner*; never steal another session's input."""
    with _registry_lock:
        matches = [
            question
            for question in pending_questions.values()
            if question.owner is owner and not owner.is_closed
        ]
        if not matches:
            return False
        question = min(matches, key=lambda item: item.created_at)
        pending_questions.pop(question.id, None)
    question.answer_with(text)
    return True


def _resolve_pi_bin() -> str:
    return (
        os.getenv("HERMES_PI_BIN", "").strip()
        or os.getenv("PI_BIN", "").strip()
        or "pi"
    )


def _resolve_acp_command(kwargs_command: str | None) -> str:
    # An explicitly passed command that exists on disk wins (lets callers
    # and tests pin the exact binary). Bare names (``pi`` / ``pi-acp`` from
    # the copilot-acp plumbing) resolve through env so HERMES_PI_BIN still
    # overrides; we drive pi natively either way.
    if kwargs_command and Path(kwargs_command).exists():
        return kwargs_command
    return _resolve_pi_bin()


class _PiChatCompletions:
    def __init__(self, client: "PiRPCClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _PiChatNamespace:
    def __init__(self, client: "PiRPCClient"):
        self.completions = _PiChatCompletions(client)


class PiRPCClient:
    """Minimal OpenAI-client-compatible facade over `pi --mode rpc`."""

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
        persistent_session: bool = False,
        session_id: str | None = None,
        session_name: str | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or "pi-rpc"
        self.base_url = base_url or PI_RPC_MARKER_BASE_URL
        self._pi_bin = _resolve_acp_command(command or acp_command)
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._persistent_session = bool(persistent_session)
        self._session_id = (session_id or "").strip() or None
        self._session_name = (session_name or "").strip() or None
        self.chat = _PiChatNamespace(self)
        self.is_closed = False
        self._proc: subprocess.Popen[str] | None = None
        self._stdin_lock = threading.Lock()
        self._prompt_lock = threading.Lock()
        self._pending: dict[int, list] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._settled = threading.Event()
        self.text_streamed = False

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.is_closed = True
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # -- shim entrypoint ---------------------------------------------------

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
        prompt_text = _format_messages_as_prompt(
            messages or [], model=model, tools=tools, tool_choice=tool_choice
        )
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text, timeout_seconds=effective_timeout
        )
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
        completion = SimpleNamespace(
            choices=[choice], usage=usage, model=model or "pi-rpc"
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    # -- pi protocol -------------------------------------------------------

    def _spawn(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self.is_closed:
            raise RuntimeError("pi rpc client is closed")
        argv = [self._pi_bin, "--mode", "rpc"]
        if self._persistent_session:
            if self._session_id:
                argv += ["--session-id", self._session_id]
            if self._session_name:
                argv += ["--name", self._session_name]
        else:
            argv.append("--no-session")
        model = os.getenv("HERMES_PI_MODEL", "").strip()
        if model:
            argv += ["--model", model]
        tools = os.getenv("HERMES_PI_TOOLS", "").strip()
        if tools:
            argv += ["--tools", tools]
        # Auto-load the first-party hermes_ask_user extension (ships with this
        # repo) and optionally pi-ask-user; users can force extra extensions
        # with HERMES_PI_EXTENSIONS (colon-separated).
        exts = []
        first_party = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "extensions", "hermes_ask_user.ts"
        )
        if os.path.isfile(first_party):
            exts.append(first_party)
        ask_user = os.path.join(
            os.path.expanduser("~"), ".pi", "agent", "npm", "node_modules",
            "pi-ask-user", "index.ts",
        )
        if os.path.isfile(ask_user):
            exts.append(ask_user)
        exts += [p for p in os.getenv("HERMES_PI_EXTENSIONS", "").split(":") if p]
        for ext in exts:
            argv += ["-e", ext]
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd,
                env=hermes_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start pi binary '{self._pi_bin}'. Install pi or set HERMES_PI_BIN."
            ) from exc
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("pi rpc process did not expose stdin/stdout pipes.")
        self._proc = proc
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        return proc

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(msg)
        except Exception:
            pass

    def _stderr_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip("\n"))

    def _send_pi(self, command: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("pi rpc child is not running")
        with self._stdin_lock:
            proc.stdin.write(json.dumps(command) + "\n")
            proc.stdin.flush()

    def _request_pi(self, command: dict, timeout: float = 60.0) -> dict:
        with self._pending_lock:
            self._next_id += 1
            request_id = self._next_id
            waiter = threading.Event()
            slot: list = [None]
            self._pending[request_id] = [waiter, slot]
        self._send_pi(dict(command, id=request_id))
        if not waiter.wait(timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"pi did not answer command {command.get('type')!r}")
        return slot[0] or {}

    def _dispatch(self, msg: dict) -> None:
        msg_type = msg.get("type")

        if msg_type == "response":
            request_id = msg.get("id")
            with self._pending_lock:
                entry = (
                    self._pending.pop(request_id, None)
                    if isinstance(request_id, int)
                    else None
                )
            if entry is not None:
                entry[1][0] = msg
                entry[0].set()
            return

        if msg_type == "extension_ui_request":
            self._handle_ui_request(msg)
            return

        if msg_type in ("tool_execution_start", "tool_execution_end"):
            name = str(msg.get("toolName") or "tool")
            if msg_type == "tool_execution_start":
                args = msg.get("args")
                args_txt = (
                    args
                    if isinstance(args, str)
                    else json.dumps(args, ensure_ascii=False, default=str)
                )
                line = f"[pi-tool] {name} {args_txt}".strip()
            else:
                result = msg.get("result")
                details = result.get("details") if isinstance(result, dict) else None
                ok = details.get("success") if isinstance(details, dict) else None
                mark = "ok" if ok is not False else "FAILED"
                size = (
                    len(result)
                    if isinstance(result, str)
                    else len(json.dumps(result, default=str))
                )
                line = f"[pi-tool:{mark}] {name} -> result {size} bytes"
            self._reasoning_parts.append(line[:400] + "\n")
            return

        if msg_type == "message_update":
            event = msg.get("assistantMessageEvent") or {}
            kind = event.get("type")
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                return
            if kind == "text_delta":
                self.text_streamed = True
                self._text_parts.append(delta)
            elif kind == "thinking_delta":
                self._reasoning_parts.append(delta)
            return

        if msg_type == "agent_settled":
            self._settled.set()
            return

    # -- interactive questions ----------------------------------------------

    def _handle_ui_request(self, msg: dict) -> None:
        method = str(msg.get("method") or "input")
        request_id = msg.get("id")
        title = str(msg.get("title") or "")
        options = msg.get("options") or []
        if method not in ("input", "select", "confirm", "editor"):
            # setStatus / notify / setWidget and similar are fire-and-forget
            # UI notifications from extensions, not questions. Acknowledge
            # immediately so the child never blocks on them.
            try:
                self._send_pi({"type": "extension_ui_response", "id": request_id, "ok": True})
            except Exception:
                pass
            return
        question = PendingQuestion(method, title, options, owner=self)
        with _registry_lock:
            pending_questions[question.id] = question

        self._reasoning_parts.append(
            f"[pi-question] {method}: {title}"
            + (f" options={options}" if options else "")
            + "\n"
        )
        self._text_parts.append(
            f"\n[Question from pi delegate — steer this delegation with your answer"
            f" (timeout {int(_DEFAULT_QUESTION_TIMEOUT)}s)]: {title}\n"
        )

        try:
            answered = question.answered.wait(_DEFAULT_QUESTION_TIMEOUT)
        finally:
            with _registry_lock:
                pending_questions.pop(question.id, None)

        payload = question.answer if answered else question.auto_answer()

        if not answered:
            self._reasoning_parts.append(
                f"[pi-question] timed out; auto-answered {method} -> {payload}\n"
            )
        else:
            self._reasoning_parts.append(
                f"[pi-question] answered with user text -> {payload}\n"
            )
        try:
            self._send_pi(
                {"type": "extension_ui_response", "id": request_id, **payload}
            )
        except Exception:
            pass

    # -- persistent-session control -------------------------------------------

    def start(self, *, timeout: float = 30.0) -> dict[str, Any]:
        """Start the Pi RPC process and return native session state."""
        self._spawn()
        response = self._request_pi({"type": "get_state"}, timeout=timeout)
        if response.get("success") is False:
            raise RuntimeError(response.get("error") or "pi get_state failed")
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    def get_state(self, *, timeout: float = 30.0) -> dict[str, Any]:
        self._spawn()
        response = self._request_pi({"type": "get_state"}, timeout=timeout)
        if response.get("success") is False:
            raise RuntimeError(response.get("error") or "pi get_state failed")
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    def get_messages(self, *, timeout: float = 30.0) -> list[Any]:
        self._spawn()
        response = self._request_pi({"type": "get_messages"}, timeout=timeout)
        if response.get("success") is False:
            raise RuntimeError(response.get("error") or "pi get_messages failed")
        data = response.get("data")
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data["messages"]
        return data if isinstance(data, list) else []

    def answer_pending_question(self, text: str) -> bool:
        return answer_pending_question_for_owner(self, text)

    def steer(self, message: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Steer a running Pi turn, or answer its pending extension question."""
        if self.answer_pending_question(message):
            return {"success": True, "command": "answer_question"}
        self._spawn()
        return self._request_pi({"type": "steer", "message": message}, timeout=timeout)

    def abort(self, *, timeout: float = 30.0) -> dict[str, Any]:
        self._spawn()
        return self._request_pi({"type": "abort"}, timeout=timeout)

    def run_session_prompt(
        self,
        message: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Run one turn against the same persistent Pi process/session.

        Unlike the OpenAI compatibility path, this method does not reconstruct
        prior conversation into the prompt: Pi owns the conversation state.
        """
        with self._prompt_lock:
            self._spawn()
            self._text_parts = []
            self._reasoning_parts = []
            self._settled = threading.Event()
            self.text_streamed = False
            started = time.monotonic()
            response = self._request_pi(
                {"type": "prompt", "message": message}, timeout=timeout_seconds
            )
            if not response.get("success"):
                raise RuntimeError(response.get("error") or "pi prompt rejected")
            self._settled.wait(timeout_seconds)
            if not self._settled.is_set():
                try:
                    self._send_pi({"type": "abort"})
                except Exception:
                    pass
                self._settled.wait(10)
                raise TimeoutError(f"pi session turn timed out after {timeout_seconds:.0f}s")
            if not self.text_streamed:
                try:
                    last = self._request_pi({"type": "get_last_assistant_text"})
                    text = (last.get("data") or {}).get("text")
                    if isinstance(text, str) and text:
                        self._text_parts.append(text)
                except Exception:
                    pass
            state = self.get_state(timeout=min(30.0, timeout_seconds))
            return {
                "success": True,
                "text": "".join(self._text_parts),
                "reasoning": "".join(self._reasoning_parts),
                "duration_s": round(time.monotonic() - started, 1),
                "state": state,
            }

    # -- run ------------------------------------------------------------------

    def _git_status_snapshot(self) -> dict[str, str] | None:
        try:
            proc = subprocess.run(
                ["git", "-C", self._cwd, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return {
            line[3:]: line for line in proc.stdout.splitlines() if len(line) > 3
        }

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        self._spawn()
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._settled = threading.Event()
        self.text_streamed = False

        before_status = self._git_status_snapshot()
        started = time.monotonic()

        policy = (
            "[Delegation policy] You are a delegated implementer. Make the "
            "requested changes in the working tree, but do NOT run git "
            "commit or git push — leave all changes uncommitted for the "
            "reviewing agent to review and land."
        )

        status = "end_turn"
        try:
            response = self._request_pi(
                {"type": "prompt", "message": policy + "\n\n" + prompt_text},
                timeout=timeout_seconds,
            )
            if not response.get("success"):
                error = response.get("error") or "prompt rejected"
                self._text_parts.append(f"\n[pi-rpc error] prompt rejected: {error}\n")
                status = "error"
            else:
                self._settled.wait(timeout_seconds)
                if not self._settled.is_set():
                    try:
                        self._send_pi({"type": "abort"})
                    except Exception:
                        pass
                    self._settled.wait(10)
                    self._text_parts.append(
                        f"\n[pi-rpc error] timed out after {timeout_seconds:.0f}s\n"
                    )
                    status = "error"
        except TimeoutError as exc:
            self._text_parts.append(f"\n[pi-rpc error] {exc}\n")
            status = "error"

        if status == "end_turn" and not self.text_streamed:
            try:
                last = self._request_pi({"type": "get_last_assistant_text"})
                text = (last.get("data") or {}).get("text")
                if isinstance(text, str) and text:
                    self._text_parts.append(text)
            except Exception:
                pass

        after_status = self._git_status_snapshot()
        touched = sorted(
            line
            for path, line in (after_status or {}).items()
            if (before_status or {}).get(path) != line
        ) if before_status is not None and after_status is not None else []
        footer = {
            "status": status,
            "duration_s": round(time.monotonic() - started, 1),
            "git_repo": after_status is not None,
            "touched_files": touched,
        }
        self._text_parts.append(
            "\n```pi-delegation-result\n"
            + json.dumps(footer, ensure_ascii=False)
            + "\n```\n"
        )
        return "".join(self._text_parts), "".join(self._reasoning_parts)
