"""Relay-side accumulator for the chat_completions streaming wire.

Relay invokes its collector for every post-intercept chunk and then its finalizer as soon
as the provider stream ends — concurrently with Hermes' consumer thread, which may not have
read the last chunk yet. The finalizer therefore builds Relay's recorded response from
collector-observed state only, never from the consumer loop's closures. Sibling of
``relay_llm.AnthropicStreamAccumulator``; Bedrock and Codex follow the same contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent.chat_completion_helpers import _ToolCallAccumulator
from agent.message_content import flatten_message_text
from agent.reasoning_summaries import separate_glued_reasoning_blocks


def _tool_call_delta_view(tc_delta: Any) -> Any:
    """Attribute view of a JSON tool-call delta for ``_ToolCallAccumulator.feed`` (written
    against SDK objects). Only ``function`` is wrapped: ``feed`` passes ``extra_content``
    (a dict) straight through ``_dump_if_model``, so a recursive view would corrupt it."""
    if not isinstance(tc_delta, dict):
        return tc_delta
    function = tc_delta.get("function")
    return SimpleNamespace(**{**tc_delta,
        "function": SimpleNamespace(**function) if isinstance(function, dict) else function})


class RelayChatAccumulator:
    """Rebuild a chat.completion from Relay's post-intercept chunk dicts."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls = _ToolCallAccumulator()
        self._model = self._usage = self._finish_reason = None
        self._role = "assistant"

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        self._model = chunk.get("model") or self._model
        if chunk.get("usage"):
            self._usage = chunk["usage"]
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else None  # Hermes never requests n>1
        if not isinstance(choice, dict):
            return
        self._finish_reason = choice.get("finish_reason") or self._finish_reason
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        if delta.get("role"):
            self._role = delta["role"]
        text = flatten_message_text(delta.get("content"), sep="")
        if text:
            self._content.append(text)
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            self._reasoning.append(separate_glued_reasoning_blocks(
                self._reasoning[-1] if self._reasoning else "", reasoning))
        for tc_delta in delta.get("tool_calls") or []:
            self._tool_calls.feed(_tool_call_delta_view(tc_delta))

    def finalize(self) -> dict[str, Any]:
        acc = self._tool_calls.materialize()
        message = {"role": self._role, "content": "".join(self._content) or None,
            "reasoning_content": "".join(self._reasoning) or None,
            "tool_calls": [acc[i] for i in sorted(acc)] or None}
        # "stop" also covers Nous Portal ``lastOne`` usage frames, which carry no finish_reason.
        return {"model": self._model, "usage": self._usage,
            "choices": [{"message": message, "finish_reason": self._finish_reason or "stop"}]}
