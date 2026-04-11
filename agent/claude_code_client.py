from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ACP_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("CLAUDE_CODE_ACP_COMMAND", "").strip() or "claude"


def _resolve_args() -> list[str]:
    raw = os.getenv("CLAUDE_CODE_ACP_ARGS", "").strip()
    if not raw:
        return ["--print", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--no-session-persistence"]
    return shlex.split(raw)


class _CCChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CCChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _CCChatCompletions(client)


class ClaudeCodeClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _CCChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        prompt = _messages_to_prompt(messages or [])
        text = self._run_prompt(prompt, timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS))
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        msg = SimpleNamespace(
            content=text,
            tool_calls=[],
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "claude-code-acp")

    def _run_prompt(self, prompt: str, *, timeout_seconds: float) -> str:
        try:
            proc = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self._cwd,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude Code command '{self._command}'. "
                "Install Claude Code CLI or set CLAUDE_CODE_ACP_COMMAND."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Claude Code process did not expose stdin/stdout pipes.")

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()

        def _reader() -> None:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    pass

        threading.Thread(target=_reader, daemon=True).start()
        proc.stdin.write(prompt)
        proc.stdin.close()

        parts: list[str] = []
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None and inbox.empty():
                break
            try:
                evt = inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            evt_type = evt.get("type")
            if evt_type == "assistant":
                for block in (evt.get("message") or {}).get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
            elif evt_type == "result":
                result = evt.get("result", "")
                if result and not parts:
                    parts.append(result)
                break

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if not parts:
            stderr_out = (proc.stderr.read() or "").strip()
            raise RuntimeError(
                f"Claude Code returned no content. stderr: {stderr_out[:500]}"
            )
        return "".join(parts)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            texts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(texts)
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            role, role.title()
        )
        if str(content).strip():
            parts.append(f"{label}:\n{content}")
    parts.append("Continue from the latest user message.")
    return "\n\n".join(parts)
