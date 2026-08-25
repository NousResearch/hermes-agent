"""Tameru (貯める) context engine — deterministic query-aware compaction.

This plugin carries the reviewed Tameru implementation locally so Hermes does
not depend on a separate checkout or host-specific paths.
"""
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

    def __init__(self, model: str = "pending", **kwargs):
        kwargs.setdefault("proactive_prune_tokens", 48_000)
        super().__init__(model=model, **kwargs)

    def prune_tool_results_only(self, messages, current_tokens=None):
        query = ""
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                query = str(msg.get("content") or "")
                break
        pruned, changed = apply_extractive_tool_prune(messages, query)
        more, changed_more = super().prune_tool_results_only(pruned, current_tokens)
        return more, changed + changed_more

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
        summarized = super().compress(
            pruned,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        if query_facts_lost(pruned, summarized, query) or bulky_tools_dropped(
            pruned, summarized
        ):
            return pruned
        return summarized


def register(ctx):
    ctx.register_context_engine(ExtractiveContextEngine())
