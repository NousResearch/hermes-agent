"""OpenAI-compatible facade that routes Hermes inference through local Claude CLI.

This backend is intentionally text-in/text-out only for now. It shells out to
`claude -p --output-format json`, passes Claude Code's documented
`--disallowedTools` flags for the built-in tool set, and returns a minimal
response object that matches the parts of the OpenAI SDK shape Hermes expects.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.model_metadata import estimate_tokens_rough

CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_CLAUDE_CLI_DISABLED_TOOLS = (
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CLI_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _coerce_session_uuid(session_id: str | None) -> str:
    raw = str(session_id or "").strip()
    if not raw:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes:{raw}"))


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if isinstance(content.get("content"), str):
            return content["content"].strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(messages: list[dict[str, Any]], model: str | None = None) -> str:
    sections: list[str] = [
        "You are being used as the active Claude CLI backend for Hermes Agent.",
        "Respond directly in natural language.",
        "Do not emit tool-call JSON.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
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


class _ClaudeCLIChatCompletions:
    def __init__(self, client: "ClaudeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCLIChatNamespace:
    def __init__(self, client: "ClaudeCLIClient"):
        self.completions = _ClaudeCLIChatCompletions(client)


class ClaudeCLIClient:
    """Minimal OpenAI-client-compatible facade for local Claude CLI inference."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        session_id: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._command = command or acp_command or _resolve_command()
        self._args = list(args or acp_args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeCLIChatNamespace(self)
        self.is_closed = False
        self._hermes_session_uuid = _coerce_session_uuid(session_id)
        self._claude_session_id: str | None = None

    def _build_command(self, *, model: str, prompt_text: str, resume_session_id: str | None = None) -> list[str]:
        cmd = [self._command, "-p", "--output-format", "json", "--model", model]
        for tool_name in _CLAUDE_CLI_DISABLED_TOOLS:
            cmd.extend(["--disallowedTools", tool_name])
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        elif self._claude_session_id:
            cmd.extend(["--resume", self._claude_session_id])
        else:
            cmd.extend(["--session-id", self._hermes_session_uuid])
        cmd.extend(self._args)
        cmd.append(prompt_text)
        return cmd

    def _invoke(self, cmd: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self._cwd,
            timeout=timeout_seconds,
        )

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
        prompt_text = _format_messages_as_prompt(messages or [], model=model)
        payload = self._run_prompt(
            prompt_text,
            model=(model or "").split("/", 1)[-1],
            timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
        )
        cli_session_id = str(payload.get("session_id") or "").strip()
        if cli_session_id:
            self._claude_session_id = cli_session_id

        usage_payload = payload.get("usage") if isinstance(payload, dict) else {}
        input_tokens = int((usage_payload or {}).get("input_tokens") or 0)
        cache_read_tokens = int((usage_payload or {}).get("cache_read_input_tokens") or 0)
        cache_write_tokens = int((usage_payload or {}).get("cache_creation_input_tokens") or 0)
        completion_tokens = int((usage_payload or {}).get("output_tokens") or 0)
        prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
        total_tokens = prompt_tokens + completion_tokens

        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            raw_cli_usage=usage_payload,
        )

        content = str(payload.get("result") or "").strip()
        assistant_message = SimpleNamespace(
            content=content,
            tool_calls=[],
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or payload.get("model") or "claude-cli",
        )

    def _run_prompt(self, prompt_text: str, *, model: str, timeout_seconds: float) -> dict[str, Any]:
        cmd = self._build_command(model=model, prompt_text=prompt_text)
        try:
            proc = self._invoke(cmd, timeout_seconds=timeout_seconds)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude CLI command '{self._command}'. "
                "Install Claude Code CLI or set HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
            ) from exc

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        combined = f"{stdout}\n{stderr}".strip()
        if (
            proc.returncode != 0
            and not self._claude_session_id
            and self._hermes_session_uuid in combined
            and "already in use" in combined.lower()
        ):
            retry_cmd = self._build_command(
                model=model,
                prompt_text=prompt_text,
                resume_session_id=self._hermes_session_uuid,
            )
            proc = self._invoke(retry_cmd, timeout_seconds=timeout_seconds)
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
        if not stdout:
            raise RuntimeError(stderr or "Claude CLI returned no output.")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = {"result": stdout, "usage": {}}

        if not isinstance(payload, dict):
            payload = {"result": stdout, "usage": {}}

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
            payload["usage"] = usage

        # Fill missing token metrics with rough estimates so Hermes status/context
        # indicators remain useful even when Claude CLI omits detailed usage.
        if not any(usage.get(key) for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")):
            usage["input_tokens"] = estimate_tokens_rough(prompt_text)
            usage["output_tokens"] = estimate_tokens_rough(str(payload.get("result") or ""))

        if not payload.get("model"):
            payload["model"] = model
        if proc.returncode != 0 and not payload.get("result"):
            payload["result"] = stderr or stdout
        return payload
