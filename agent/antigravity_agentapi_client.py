"""Conversation transport for the Antigravity CLI's `agy agentapi` mode.

The CLI is asynchronous: `new-conversation` / `send-message` return as soon as
the message is accepted, and the model's reply is appended later to

    $HOME/.gemini/antigravity-cli/brain/<conversation_id>/.system_generated/logs/transcript.jsonl

as a `source=MODEL`, `type=PLANNER_RESPONSE`, `status=DONE` line carrying a
monotonically growing ``step_index``. This module owns that lifecycle —
starting the conversation, sending follow-up turns into it, and polling the
transcript for the reply belonging to the turn we just sent — so the ACP shim
in ``agent.copilot_acp_client`` only has to hand it a prompt.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

_DEFAULT_TITLE = "Hermes delegated task"
_POLL_INTERVAL_SECONDS = 0.25
_REQUIRED_ENV_VARS = ("ANTIGRAVITY_LS_ADDRESS", "ANTIGRAVITY_PROJECT_ID")

# Copilot ACP defaults / print-mode flags that `agy agentapi` does not accept.
_UNSUPPORTED_ARGS = {"--acp", "--stdio", "-p", "--print"}

_AGENTAPI_TOKEN = "agentapi"
_TITLE_PREFIX = "--title="


def is_agentapi_args(args: list[str] | None) -> bool:
    """True when the configured ACP args ask for `agy agentapi` mode."""

    return any(str(arg).strip().lower() == _AGENTAPI_TOKEN for arg in (args or []))


def _remaining_seconds(deadline: float) -> float:
    return deadline - time.monotonic()


def _loads_lenient(text: str) -> Any:
    """Parse JSON, tolerating banner/progress noise around the payload."""

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


class AntigravityAgentAPIClient:
    """Drives one `agy agentapi` conversation across multiple prompts."""

    def __init__(
        self,
        *,
        command: str,
        args: list[str] | None = None,
        cwd: str,
        env_factory: Callable[[], dict[str, str]],
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._env_factory = env_factory
        self._extra_args = self._clean_args(args or [])
        self._title = self._resolve_title(self._extra_args)
        self._conversation_id: str | None = None
        self._conversation_lock = threading.RLock()

    @property
    def conversation_id(self) -> str | None:
        with self._conversation_lock:
            return self._conversation_id

    def reset_conversation(self) -> None:
        with self._conversation_lock:
            self._reset_conversation_unlocked()

    def send_prompt(self, prompt_text: str, *, timeout_seconds: float) -> str:
        """Send ``prompt_text`` and return the model's final transcript reply.

        The timeout budget covers the CLI invocation *and* the transcript poll
        that follows it, so a slow CLI call shortens the wait rather than
        doubling the caller's deadline.
        """

        with self._conversation_lock:
            deadline = time.monotonic() + float(timeout_seconds)
            try:
                env = self._prepare_env()
                if self._conversation_id is None:
                    return self._start_conversation(prompt_text, env=env, deadline=deadline)
                return self._continue_conversation(prompt_text, env=env, deadline=deadline)
            except Exception:
                self._reset_conversation_unlocked()
                raise

    # -- lifecycle ---------------------------------------------------------

    def _start_conversation(self, prompt_text: str, *, env: dict[str, str], deadline: float) -> str:
        cmd = [
            self._command,
            _AGENTAPI_TOKEN,
            "new-conversation",
            *self._with_title(self._extra_args),
            prompt_text,
        ]
        stdout_text = self._invoke(cmd, env=env, deadline=deadline)
        conversation_id = self._parse_conversation_id(stdout_text)
        self._conversation_id = conversation_id
        return self._wait_for_response(
            conversation_id,
            after_step_index=-1,
            env=env,
            deadline=deadline,
        )

    def _continue_conversation(self, prompt_text: str, *, env: dict[str, str], deadline: float) -> str:
        conversation_id = str(self._conversation_id)
        transcript = self._transcript_path(conversation_id, env=env)
        # Snapshot the transcript before sending, so the previous turn's reply
        # can never be mistaken for this turn's.
        baseline = _max_step_index(transcript)
        cmd = [
            self._command,
            _AGENTAPI_TOKEN,
            "send-message",
            f"{_TITLE_PREFIX}{self._title}",
            conversation_id,
            prompt_text,
        ]
        self._invoke(cmd, env=env, deadline=deadline)
        return self._wait_for_response(
            conversation_id,
            after_step_index=baseline,
            env=env,
            deadline=deadline,
        )

    # -- subprocess --------------------------------------------------------

    def _invoke(self, cmd: list[str], *, env: dict[str, str], deadline: float) -> str:
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out before invoking `{self._command} {_AGENTAPI_TOKEN} {cmd[2]}`."
            )

        try:
            completed = subprocess.run(
                cmd,
                cwd=self._cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=remaining,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Antigravity CLI command '{self._command}'. "
                "Install `agy` or point Hermes at it explicitly via acp_command."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Timed out after {remaining:.0f}s waiting for "
                f"`{self._command} {_AGENTAPI_TOKEN} {cmd[2]}`."
            ) from exc

        stdout_text = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        if completed.returncode != 0:
            detail = stderr_text or stdout_text or f"exit code {completed.returncode}"
            raise RuntimeError(
                f"Antigravity agentapi `{cmd[2]}` failed ({completed.returncode}): {detail}"
            )
        if not stdout_text:
            detail = stderr_text or "CLI returned empty stdout"
            raise RuntimeError(f"Antigravity agentapi `{cmd[2]}` returned no response: {detail}")
        return stdout_text

    def _prepare_env(self) -> dict[str, str]:
        env = dict(self._env_factory())
        missing = [name for name in _REQUIRED_ENV_VARS if not str(env.get(name) or "").strip()]
        if missing:
            raise RuntimeError(
                "Antigravity `agy agentapi` requires "
                + " and ".join(missing)
                + " in the environment. Start the Antigravity CLI (which publishes the "
                "language-server address and project id) and export those variables, or "
                "drop the `agentapi` arg to use `agy -p` print mode instead."
            )
        return env

    # -- transcript --------------------------------------------------------

    def _transcript_path(self, conversation_id: str, *, env: dict[str, str]) -> Path:
        return (
            Path(env.get("HOME") or "")
            / ".gemini"
            / "antigravity-cli"
            / "brain"
            / conversation_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )

    def _wait_for_response(
        self,
        conversation_id: str,
        *,
        after_step_index: int,
        env: dict[str, str],
        deadline: float,
    ) -> str:
        transcript = self._transcript_path(conversation_id, env=env)
        while True:
            content = _final_response_after(transcript, after_step_index)
            if content is not None:
                return content
            remaining = _remaining_seconds(deadline)
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for an Antigravity agentapi response in "
                    f"{transcript}."
                )
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def _parse_conversation_id(self, stdout_text: str) -> str:
        payload = _loads_lenient(stdout_text)
        if not isinstance(payload, dict):
            raise RuntimeError(
                "Antigravity agentapi `new-conversation` returned malformed JSON: "
                f"{stdout_text[:400]}"
            )
        response = payload.get("response")
        new_conversation = response.get("newConversation") if isinstance(response, dict) else None
        conversation_id = ""
        if isinstance(new_conversation, dict):
            conversation_id = str(new_conversation.get("conversationId") or "").strip()
        if not conversation_id:
            raise RuntimeError(
                "Antigravity agentapi `new-conversation` did not return a conversationId: "
                f"{stdout_text[:400]}"
            )
        return conversation_id

    # -- args --------------------------------------------------------------

    @staticmethod
    def _clean_args(args: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen_agentapi = False
        for arg in args:
            token = str(arg).strip()
            lower = token.lower()
            if lower in _UNSUPPORTED_ARGS:
                continue
            if lower == _AGENTAPI_TOKEN and not seen_agentapi:
                seen_agentapi = True
                continue
            cleaned.append(token)
        return cleaned

    @staticmethod
    def _resolve_title(args: list[str]) -> str:
        for arg in args:
            if arg.startswith(_TITLE_PREFIX):
                title = arg[len(_TITLE_PREFIX) :].strip()
                if title:
                    return title
        return _DEFAULT_TITLE

    def _with_title(self, args: list[str]) -> list[str]:
        if any(arg.startswith(_TITLE_PREFIX) for arg in args):
            return list(args)
        return [*args, f"{_TITLE_PREFIX}{self._title}"]

    def _reset_conversation_unlocked(self) -> None:
        self._conversation_id = None


def _read_entries(transcript: Path) -> list[dict[str, Any]]:
    """Read the transcript, skipping lines that are absent/partial/malformed.

    The CLI appends while we read, so a torn final line is normal, not an error.
    """

    try:
        raw = transcript.read_bytes()
    except FileNotFoundError:
        return []
    except OSError:
        return []

    entries: list[dict[str, Any]] = []
    for raw_line in raw.splitlines():
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _step_index(entry: dict[str, Any]) -> int | None:
    step_index = entry.get("step_index")
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        return None
    return step_index


def _max_step_index(transcript: Path) -> int:
    highest = -1
    for entry in _read_entries(transcript):
        step_index = _step_index(entry)
        if step_index is not None and step_index > highest:
            highest = step_index
    return highest


def _final_response_after(transcript: Path, after_step_index: int) -> str | None:
    """Return the newest finished model reply past ``after_step_index``.

    Ignores everything that isn't a settled textual answer: user echoes, tool
    calls, in-progress chunks, and planner responses that only carry tool calls
    (those are followed by a later response with the actual text).
    """

    best_step: int | None = None
    best_content: str | None = None
    for entry in _read_entries(transcript):
        step_index = _step_index(entry)
        if step_index is None or step_index <= after_step_index:
            continue
        if str(entry.get("source") or "").strip().upper() != "MODEL":
            continue
        if str(entry.get("type") or "").strip().upper() != "PLANNER_RESPONSE":
            continue
        if str(entry.get("status") or "").strip().upper() != "DONE":
            continue
        if entry.get("tool_calls"):
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if best_step is None or step_index > best_step:
            best_step = step_index
            best_content = content.strip()
    return best_content
