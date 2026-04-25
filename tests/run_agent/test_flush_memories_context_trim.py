"""Tests for flush_memories() context-window overflow prevention.

Two-layer defence:
  Layer 1 — _check_compression_model_feasibility now also resolves the
            flush_memories auxiliary model and caps the compression
            threshold at min(compression_ctx, flush_ctx).
  Layer 2 — flush_memories() itself trims oversized api_messages before
            calling call_llm, as a safety net for CLI /new and gateway
            reset paths that don't go through preflight compression.

Regression test for:
  ⚠ Auxiliary memory flush failed: HTTP 400: Error code: 400 -
  {'error': 'invalid params, context window exceeds limit (ref: ...)'}
"""

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key", "test")
        self.base_url = kwargs.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter"):
    """Build an AIAgent with mocked internals."""
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [
        {
            "type": "function",
            "function": {
                "name": "memory",
                "description": "Manage memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)

    agent = run_agent.AIAgent(
        api_key="test-key",
        base_url="https://test.example.com/v1",
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._memory_store = MagicMock()
    agent._memory_flush_min_turns = 1
    agent._user_turn_count = 5
    return agent


def _make_messages(n: int, chars_per_msg: int = 400) -> list:
    """Generate n alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * max(0, chars_per_msg - 15)
        msgs.append({"role": role, "content": content})
    return msgs


def _no_tool_calls_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="Nothing to save.", tool_calls=None),
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


# ── Layer 1: _check_compression_model_feasibility ───────────────────────


class TestFeasibilityChecksFlushModel:
    """_check_compression_model_feasibility must also resolve the
    flush_memories auxiliary model and cap the threshold at the smaller
    of the two auxiliary context windows."""

    def test_flush_model_smaller_than_compression_model_lowers_threshold(self, monkeypatch):
        """When flush_memories resolves to a model with a smaller context
        window than the compression model, the threshold is capped at the
        flush model's context."""
        agent = _make_agent(monkeypatch)
        # Set up a compressor with a large threshold
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")

        def _mock_get_text_aux(task, **kw):
            if task == "compression":
                return fake_client, "big-compression-model"
            elif task == "flush_memories":
                return fake_client, "small-flush-model"
            return None, None

        def _mock_ctx_length(model, **kw):
            if model == "big-compression-model":
                return 200_000
            elif model == "small-flush-model":
                return 80_000
            return 128_000

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    side_effect=_mock_get_text_aux), \
             patch("agent.model_metadata.get_model_context_length",
                    side_effect=_mock_ctx_length):
            agent._check_compression_model_feasibility()

        # Threshold should be lowered to the flush model's 80K
        assert agent.context_compressor.threshold_tokens == 80_000

    def test_same_model_for_both_tasks_no_double_penalty(self, monkeypatch):
        """When compression and flush_memories resolve to the same model,
        the threshold is capped at that model's context — no double penalty."""
        agent = _make_agent(monkeypatch)
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(fake_client, "same-model")), \
             patch("agent.model_metadata.get_model_context_length",
                    return_value=120_000):
            agent._check_compression_model_feasibility()

        # Both resolve to 120K > 100K threshold, so no change
        assert agent.context_compressor.threshold_tokens == 100_000

    def test_compression_model_smaller_still_works(self, monkeypatch):
        """When compression model is the smaller one, it still drives the
        threshold (existing behaviour preserved)."""
        agent = _make_agent(monkeypatch)
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")

        def _mock_get_text_aux(task, **kw):
            if task == "compression":
                return fake_client, "small-compression-model"
            elif task == "flush_memories":
                return fake_client, "big-flush-model"
            return None, None

        def _mock_ctx_length(model, **kw):
            if model == "small-compression-model":
                return 70_000
            elif model == "big-flush-model":
                return 200_000
            return 128_000

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    side_effect=_mock_get_text_aux), \
             patch("agent.model_metadata.get_model_context_length",
                    side_effect=_mock_ctx_length):
            agent._check_compression_model_feasibility()

        assert agent.context_compressor.threshold_tokens == 70_000

    def test_flush_model_resolution_failure_is_non_fatal(self, monkeypatch):
        """If flush_memories aux resolution raises, feasibility check still
        proceeds using the compression model's context only."""
        agent = _make_agent(monkeypatch)
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")

        call_count = [0]
        def _mock_get_text_aux(task, **kw):
            call_count[0] += 1
            if task == "compression":
                return fake_client, "compression-model"
            elif task == "flush_memories":
                raise RuntimeError("flush aux unavailable")
            return None, None

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    side_effect=_mock_get_text_aux), \
             patch("agent.model_metadata.get_model_context_length",
                    return_value=200_000):
            agent._check_compression_model_feasibility()

        # Should have tried both tasks
        assert call_count[0] == 2
        # No threshold change — compression model fits fine
        assert agent.context_compressor.threshold_tokens == 100_000


# ── Layer 2: flush_memories() inline trimming ────────────────────────────


class TestFlushMemoriesTrimming:
    """flush_memories trims oversized conversations before calling call_llm."""

    def test_oversized_conversation_trimmed(self, monkeypatch):
        """Large conversation is trimmed to fit the aux model's context."""
        agent = _make_agent(monkeypatch)
        agent._cached_system_prompt = "You are helpful."
        messages = _make_messages(200, chars_per_msg=500)

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")
        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(fake_client, "small-model")), \
             patch("agent.model_metadata.get_model_context_length",
                    return_value=8_000), \
             patch("agent.auxiliary_client.call_llm",
                    return_value=_no_tool_calls_response()) as mock_call:
            agent.flush_memories(messages)

            assert mock_call.called
            sent = mock_call.call_args.kwargs.get("messages", [])
            assert len(sent) < 100, (
                f"Expected trimmed messages, got {len(sent)}. "
                f"flush_memories should trim to fit 8K aux context."
            )

    def test_small_conversation_not_trimmed(self, monkeypatch):
        """Short conversations pass through untrimmed."""
        agent = _make_agent(monkeypatch)
        agent._cached_system_prompt = "You are helpful."
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Save this"},
        ]

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")
        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(fake_client, "big-model")), \
             patch("agent.model_metadata.get_model_context_length",
                    return_value=200_000), \
             patch("agent.auxiliary_client.call_llm",
                    return_value=_no_tool_calls_response()) as mock_call:
            agent.flush_memories(messages)

            sent = mock_call.call_args.kwargs.get("messages", [])
            # 1 system + 3 conv + 1 flush = 5
            assert len(sent) == 5

    def test_trim_failure_is_non_fatal(self, monkeypatch):
        """If trimming fails, flush still proceeds with full messages."""
        agent = _make_agent(monkeypatch)
        messages = _make_messages(10, chars_per_msg=100)

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    side_effect=RuntimeError("no provider")), \
             patch("agent.auxiliary_client.call_llm",
                    return_value=_no_tool_calls_response()) as mock_call:
            agent.flush_memories(messages)
            assert mock_call.called

    def test_sentinel_cleaned_up_after_trim(self, monkeypatch):
        """Flush sentinel is always removed regardless of trimming."""
        agent = _make_agent(monkeypatch)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Remember this"},
        ]
        original_len = len(messages)

        fake_client = SimpleNamespace(base_url="http://test", api_key="k")
        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                    return_value=(fake_client, "model")), \
             patch("agent.model_metadata.get_model_context_length",
                    return_value=128_000), \
             patch("agent.auxiliary_client.call_llm",
                    return_value=_no_tool_calls_response()):
            agent.flush_memories(messages)

        assert len(messages) == original_len
        assert not any(m.get("_flush_sentinel") for m in messages)
