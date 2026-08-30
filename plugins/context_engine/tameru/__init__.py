"""Hermes context-engine adapter for vendored Tameru 1.2.0."""
from __future__ import annotations

from typing import Any

from agent.context_compressor import ContextCompressor

from .hermes_extractive_engine import (
    apply_extractive_tool_prune,
    bulky_tools_dropped,
    query_facts_lost,
)


class ExtractiveContextEngine(ContextCompressor):
    """Tameru: built-in summarizer plus deterministic extractive pruning."""

    DISPLAY_NAME = "Tameru (貯める)"

    @property
    def name(self) -> str:
        return "tameru"

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    def get_automatic_compaction_status_message(
        self, *, phase: str, default_message: str, **context: Any
    ) -> str | None:
        del phase, context
        return f"🗜️ {self.DISPLAY_NAME} compaction — {default_message}"

    def __init__(self, model: str = "pending", **kwargs: Any) -> None:
        kwargs.setdefault("proactive_prune_tokens", 48_000)
        super().__init__(model=model, **kwargs)

    def prune_tool_results_only(self, messages, current_tokens=None):
        query = ""
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                query = str(msg.get("content") or "")
                break
        pruned, changed = apply_extractive_tool_prune(messages, query)
        more, parent_changed = super().prune_tool_results_only(pruned, current_tokens)
        return more, changed + parent_changed

    def compress(
        self,
        messages,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        query = focus_topic or ""
        if not query:
            for msg in reversed(messages or []):
                if msg.get("role") == "user":
                    query = str(msg.get("content") or "")
                    break
        pruned, _changed = apply_extractive_tool_prune(messages, query)
        summarised = super().compress(
            pruned,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        if query_facts_lost(messages, summarised, query) or bulky_tools_dropped(
            pruned, summarised
        ):
            return pruned
        return summarised


def register(ctx) -> None:
    ctx.register_context_engine(ExtractiveContextEngine())
