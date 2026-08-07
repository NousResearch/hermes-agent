"""Tests for structured-reasoning thinking-budget-exhaustion detection.

Covers _structured_reasoning_only() in agent/conversation_loop.py — the
companion to the <think>-tag detector in the finish_reason=="length" branch.
OpenAI-compatible backends (Ollama /v1, DeepSeek, OpenRouter) return
reasoning as a separate message field with content="" instead of inline
tags, so the tag-based detector never fires and the loop used to waste
continuation retries on responses with no visible text to continue.

Live shape that motivated this (Ollama hermes:latest, qwen35moe):
    {"content": "", "reasoning": "Here's a thinking process…",
     "finish_reason": "length"}
"""

from types import SimpleNamespace

from agent.conversation_loop import _structured_reasoning_only


def _msg(reasoning=None, provider_data=None):
    return SimpleNamespace(reasoning=reasoning, provider_data=provider_data)


class TestStructuredReasoningOnly:

    def test_ollama_v1_shape_reasoning_field_empty_content(self):
        assert _structured_reasoning_only(_msg(reasoning="Here's a thinking process…"), "")

    def test_none_content_with_reasoning(self):
        assert _structured_reasoning_only(_msg(reasoning="cot"), None)

    def test_whitespace_content_with_reasoning(self):
        assert _structured_reasoning_only(_msg(reasoning="cot"), "  \n\t ")

    def test_deepseek_reasoning_content_in_provider_data(self):
        msg = _msg(provider_data={"reasoning_content": "chain of thought"})
        assert _structured_reasoning_only(msg, "")

    def test_openrouter_reasoning_details_in_provider_data(self):
        msg = _msg(provider_data={"reasoning_details": [{"type": "reasoning.text"}]})
        assert _structured_reasoning_only(msg, "")

    def test_visible_content_wins_over_reasoning(self):
        # Model produced BOTH reasoning and real text — normal truncation,
        # continuation retries are appropriate.
        assert not _structured_reasoning_only(_msg(reasoning="cot"), "partial answer")

    def test_no_reasoning_no_content_is_not_exhaustion(self):
        # Plain empty turn (network stub, refusal, …) — other handlers own it.
        assert not _structured_reasoning_only(_msg(), "")

    def test_empty_provider_data_dict(self):
        assert not _structured_reasoning_only(_msg(provider_data={}), "")

    def test_provider_data_not_a_dict_is_tolerated(self):
        msg = SimpleNamespace(reasoning=None, provider_data="garbage")
        assert not _structured_reasoning_only(msg, "")

    def test_plain_dictless_object_without_attrs(self):
        assert not _structured_reasoning_only(object(), "")
