"""OpenAI-compatible shim over the Cursor Agent SDK (`cursor-sdk`).

Hermes treats Cursor as a chat-completions backend. Each request formats the
conversation as a prompt, runs a local Cursor agent (default model ``auto``),
and maps the assistant text (+ optional Hermes ``<tool_call>`` blocks) back
into the OpenAI-shaped objects Hermes expects.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CURSOR_MARKER_BASE_URL = "cursor://agent"
_DEFAULT_MODEL = "auto"


def _estimate_usage(messages, tools, response_text, reasoning_text=""):
    from agent.copilot_acp_client import _estimate_usage as _est

    return _est(messages, tools, response_text, reasoning_text)


def _extract_tool_calls_from_text(text: str):
    from agent.copilot_acp_client import _extract_tool_calls_from_text as _ext

    return _ext(text)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    from agent.copilot_acp_client import _format_messages_as_prompt as _fmt

    return _fmt(messages, model=model, tools=tools)


def _import_cursor_sdk():
    try:
        from cursor_sdk import Agent, LocalAgentOptions  # type: ignore

        return Agent, LocalAgentOptions
    except ImportError as exc:
        raise RuntimeError(
            "Cursor provider requires the `cursor-sdk` package. "
            "Install with: pip install cursor-sdk"
        ) from exc


class _CursorChatCompletions:
    def __init__(self, client: "CursorAgentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CursorChatNamespace:
    def __init__(self, client: "CursorAgentClient"):
        self.completions = _CursorChatCompletions(client)


class CursorAgentClient:
    """Minimal OpenAI-client-compatible facade for Cursor Agent SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = (api_key or os.getenv("CURSOR_API_KEY", "")).strip()
        self.base_url = base_url or CURSOR_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = _CursorChatNamespace(self)
        self.is_closed = False
        self._lock = threading.Lock()
        self._agent = None

    def close(self) -> None:
        self.is_closed = True
        agent = self._agent
        self._agent = None
        if agent is None:
            return
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not set. Add it to Hermes .env "
                "(op://Infrastructure/Cursor API Key/credential)."
            )

        prompt = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
        )
        model_id = (model or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

        if stream:
            return self._stream_completion(
                prompt=prompt,
                model_id=model_id,
                messages=messages,
                tools=tools,
            )

        response_text = self._run_prompt(prompt, model_id=model_id)
        return self._build_completion(
            model=model_id,
            messages=messages,
            tools=tools,
            response_text=response_text,
        )

    def _build_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        response_text: str,
    ) -> SimpleNamespace:
        tool_calls, cleaned = _extract_tool_calls_from_text(response_text)
        usage = _estimate_usage(messages, tools, response_text)
        message = SimpleNamespace(
            content=cleaned,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=usage,
            model=model,
        )

    def _run_prompt(self, prompt: str, *, model_id: str) -> str:
        Agent, LocalAgentOptions = _import_cursor_sdk()
        with self._lock:
            # Prefer Agent.prompt one-shot when available; fall back to create+send.
            if hasattr(Agent, "prompt"):
                result = Agent.prompt(
                    prompt,
                    api_key=self.api_key,
                    model=model_id,
                    local=LocalAgentOptions(cwd=self._cwd),
                )
                status = getattr(result, "status", None)
                text = (
                    getattr(result, "result", None)
                    or getattr(result, "text", None)
                    or getattr(result, "output", "")
                    or ""
                )
                if status == "error":
                    raise RuntimeError(f"Cursor agent run failed: {text or status}")
                return str(text or "")

            with Agent.create(
                api_key=self.api_key,
                model=model_id,
                local=LocalAgentOptions(cwd=self._cwd),
            ) as agent:
                run = agent.send(prompt)
                wait = getattr(run, "wait", None)
                if callable(wait):
                    wait()
                text = getattr(run, "result", None) or ""
                if hasattr(run, "messages"):
                    parts: list[str] = []
                    for message in run.messages():
                        if getattr(message, "type", None) != "assistant":
                            continue
                        content = getattr(getattr(message, "message", None), "content", None)
                        if isinstance(content, list):
                            for block in content:
                                if getattr(block, "type", None) == "text":
                                    parts.append(str(getattr(block, "text", "") or ""))
                        elif isinstance(content, str):
                            parts.append(content)
                    if parts:
                        text = "".join(parts)
                return str(text or "")

    def _stream_completion(
        self,
        *,
        prompt: str,
        model_id: str,
        messages: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
    ) -> Iterator[SimpleNamespace]:
        text = self._run_prompt(prompt, model_id=model_id)
        completion = self._build_completion(
            model=model_id,
            messages=messages,
            tools=tools,
            response_text=text,
        )
        from agent.copilot_acp_client import _completion_to_stream_chunks

        yield from _completion_to_stream_chunks(completion)
