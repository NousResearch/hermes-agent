"""OpenAI-compatible shim that forwards Hermes requests to local Claude CLI.

The Claude CLI is used only as an auth/inference transport. Hermes remains the
orchestrator, so this adapter is deliberately stateless: every request sends
the full Hermes transcript and never resumes or creates a Claude backend
session.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import _extract_tool_calls_from_text, _format_messages_as_prompt

CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip() or "claude"


def _normalize_model(model: str | None) -> str:
    text = str(model or "").strip()
    if not text:
        return "sonnet"
    if "/" in text:
        text = text.split("/", 1)[1]
    lowered = text.lower()
    if lowered in {"opus", "sonnet", "haiku"}:
        return lowered
    if lowered.startswith("claude-"):
        return text.replace(".", "-")
    return text


def _build_stdin_payload(prompt_text: str) -> str:
    """Encode a single full-transcript prompt for Claude CLI stream-json input."""
    payload = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


class _ClaudeChatCompletions:
    def __init__(self, client: "ClaudeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeChatNamespace:
    def __init__(self, client: "ClaudeCLIClient"):
        self.completions = _ClaudeChatCompletions(client)


class ClaudeCLIClient:
    """Minimal OpenAI-client-compatible facade for Claude CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        extra_args = list(acp_args or args or [])
        if extra_args:
            raise ValueError(
                "Claude CLI transport does not accept extra CLI args. "
                "Use HERMES_CLAUDE_CLI_COMMAND for a wrapper command if needed."
            )
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = acp_command or command or _resolve_command()
        self.chat = _ClaudeChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    @staticmethod
    def _effective_timeout_seconds(timeout: Any) -> float:
        if isinstance(timeout, (int, float)):
            return float(timeout)
        values: list[float] = []
        for attr in ("connect", "read", "write", "pool", "timeout"):
            value = getattr(timeout, attr, None)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return max(values) if values else _DEFAULT_TIMEOUT_SECONDS

    @staticmethod
    def _build_usage(usage_data: dict[str, Any] | None) -> SimpleNamespace:
        usage_data = usage_data if isinstance(usage_data, dict) else {}
        input_tokens = int(usage_data.get("input_tokens") or 0)
        output_tokens = int(usage_data.get("output_tokens") or 0)
        cache_read = int(usage_data.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage_data.get("cache_creation_input_tokens") or 0)
        prompt_tokens = input_tokens + cache_read + cache_creation
        total_tokens = prompt_tokens + output_tokens
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
        )

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
            list(messages or []),
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        response_text, reasoning_text, usage_data = self._run_prompt(
            prompt_text,
            model=model,
            timeout_seconds=self._effective_timeout_seconds(timeout),
        )
        usage = self._build_usage(usage_data)

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
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
            model=model or "claude-cli",
        )

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        model: str | None,
        timeout_seconds: float,
    ) -> tuple[str, str, dict[str, Any]]:
        cmd = [
            self._command,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--tools",
            "",
            "--no-session-persistence",
            "--model",
            _normalize_model(model),
        ]
        stdin_payload = _build_stdin_payload(prompt_text)

        try:
            proc = subprocess.run(
                cmd,
                input=stdin_payload,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude CLI command '{self._command}'. "
                "Install Claude Code or set HERMES_CLAUDE_CLI_COMMAND."
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(stderr or f"Claude CLI exited with status {proc.returncode}")

        final_text = ""
        assistant_text = ""
        delta_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_data: dict[str, Any] | None = None

        for raw_line in (proc.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            item_type = item.get("type")
            if item_type == "assistant":
                message = item.get("message") or {}
                content = message.get("content") or []
                parts = [
                    str(block.get("text") or "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    assistant_text = "".join(parts).strip()
                msg_usage = message.get("usage")
                if isinstance(msg_usage, dict):
                    usage_data = msg_usage
            elif item_type == "result":
                final_text = str(item.get("result") or "").strip()
                result_usage = item.get("usage")
                if isinstance(result_usage, dict):
                    usage_data = result_usage
            elif item_type == "stream_event":
                event = item.get("event") or {}
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage_data = event_usage
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        delta_parts.append(text)
                elif event.get("type") == "thinking_delta":
                    delta = event.get("delta") or {}
                    thinking = delta.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        reasoning_parts.append(thinking)

        response_text = final_text or assistant_text or "".join(delta_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        return response_text, reasoning_text, usage_data or {}
