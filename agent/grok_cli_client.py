"""OpenAI-compatible shim that forwards Hermes requests to the Grok CLI."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import _extract_tool_calls_from_text

GROK_CLI_MARKER_BASE_URL = "grok-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    for candidate in (
        os.getenv("HERMES_GROK_BUILD_COMMAND", "").strip(),
        os.getenv("GROK_CLI_PATH", "").strip(),
        os.path.expanduser("~/.grok/bin/grok"),
        os.path.expanduser("~/.local/bin/grok"),
        "grok",
    ):
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            continue
        return candidate
    return "grok"


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_GROK_BUILD_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    effort = (
        os.getenv("HERMES_GROK_BUILD_EFFORT", "").strip()
        or os.getenv("GROK_BUILD_EFFORT", "").strip()
        or "xhigh"
    )
    return [
        "--no-memory",
        "--disable-web-search",
        "--max-turns",
        "1",
        "--output-format",
        "plain",
        "--effort",
        effort,
    ]


def _normalize_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [
        getattr(timeout, attr, None)
        for attr in ("read", "write", "connect", "pool", "timeout")
    ]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    home = os.environ.get("HOME", "").strip() or os.path.expanduser("~")
    if home and home != "~":
        env["HOME"] = home
    # Non-interactive launchd/SSH services often miss user-local CLI install dirs.
    path = env.get("PATH", "")
    extras = [os.path.expanduser("~/.grok/bin"), os.path.expanduser("~/.local/bin")]
    env["PATH"] = os.pathsep.join([p for p in extras + [path] if p])
    return env


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are the active Grok Build backend for Hermes Agent.",
        "Complete the latest user request using the conversation transcript below.",
        "If a tool is needed, emit ONLY <tool_call>{...}</tool_call> blocks with JSON in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model: {model}")

    tool_specs: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
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
            "Available tools. To use one, emit a single <tool_call>{...}</tool_call> "
            "object with id/type/function{name,arguments}; arguments must be a JSON string.\n"
            + json.dumps(tool_specs, ensure_ascii=False)
        )
    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "context").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
        if rendered:
            transcript.append(f"{role.title()}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue from the latest user message.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
        inner = content.get("content")
        if isinstance(inner, str):
            return inner.strip()
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


def _has_flag(args: list[str], *flags: str) -> bool:
    return any(arg in flags or any(arg.startswith(flag + "=") for flag in flags) for arg in args)


class _GrokChatCompletions:
    def __init__(self, client: "GrokCliClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _GrokChatNamespace:
    def __init__(self, client: "GrokCliClient"):
        self.completions = _GrokChatCompletions(client)


class GrokCliClient:
    """Minimal OpenAI-client-compatible facade for `grok --single`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        grok_command: str | None = None,
        grok_args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "grok-build"
        self.base_url = base_url or GROK_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = grok_command or command or _resolve_command()
        self._args = list(grok_args or args or _resolve_args())
        self._cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _GrokChatNamespace(self)
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
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

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
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        response_text = self._run_prompt(
            prompt_text,
            model=model or "grok-build",
            timeout_seconds=_normalize_timeout(timeout),
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
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "grok-build")

    def _run_prompt(self, prompt_text: str, *, model: str, timeout_seconds: float) -> str:
        cmd = [self._command] + list(self._args)
        if not _has_flag(cmd, "--model", "-m"):
            cmd.extend(["--model", model])
        prompt_file_path: str | None = None
        if not _has_flag(cmd, "--single", "-p", "--prompt", "--prompt-file", "--prompt-json"):
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".txt",
                prefix="hermes-grok-",
                delete=False,
            ) as prompt_file:
                prompt_file.write(prompt_text)
                prompt_file_path = prompt_file.name
            cmd.extend(["--prompt-file", prompt_file_path])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            if prompt_file_path:
                try:
                    os.unlink(prompt_file_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"Could not start Grok CLI command '{self._command}'. "
                "Install xAI's Grok CLI or set HERMES_GROK_BUILD_COMMAND/GROK_CLI_PATH."
            ) from exc

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self.close()
            raise TimeoutError(f"Timed out waiting for Grok CLI after {timeout_seconds:.0f}s.") from exc
        finally:
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None
            if prompt_file_path:
                try:
                    os.unlink(prompt_file_path)
                except OSError:
                    pass

        if proc.returncode != 0:
            stderr_tail = "\n".join((stderr or "").splitlines()[-20:]).strip()
            detail = stderr_tail or (stdout or "").strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"Grok CLI returned exit code {proc.returncode}: {detail}")

        return (stdout or "").strip()
