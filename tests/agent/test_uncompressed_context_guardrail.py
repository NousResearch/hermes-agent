"""Unit tests for uncompressed context overflow guardrail (Issue #89297)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import TurnContext, build_turn_context
from tests.agent.test_turn_context import _FakeAgent, _build


class _FakeUncompressedAgent(_FakeAgent):
    """Agent stub with compression disabled (compression.enabled: False)."""

    def __init__(self, model="deepseek-v4-flash", context_length=10_000):
        super().__init__()
        self.model = model
        self.provider = "deepseek"
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2,
            protect_last_n=2,
            context_length=context_length,
            threshold_tokens=int(context_length * 0.75),
            last_prompt_tokens=-1,
        )

    def _warn_uncompressed_context_overflow(self, preflight_tokens: int, context_length: int) -> None:
        _warn_key = ("uncompressed_ctx_overflow", context_length)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            msg = (
                f"⚠️ Session context (~{preflight_tokens:,} tokens) exceeds the model "
                f"context window (~{context_length:,} tokens) with compression disabled "
                f"(compression.enabled: false). Use /compact to compress history or "
                f"enable compression in config.yaml."
            )
            self._emit_warning(msg)


def test_uncompressed_session_within_limits_emits_no_warning():
    agent = _FakeUncompressedAgent(context_length=128_000)
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    agent._emit_warning.assert_not_called()


def test_uncompressed_session_exceeding_context_limit_warns():
    # Model context length is 10,000 tokens (~40,000 chars)
    agent = _FakeUncompressedAgent(context_length=10_000)
    
    # Construct an oversized uncompressed history of ~15,000 tokens (>60,000 chars)
    large_turn = "Large context content " * 500  # ~2,500 tokens
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"Turn {i}: {large_turn}"})
        history.append({"role": "assistant", "content": f"Reply {i}: {large_turn}"})

    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    agent._emit_warning.assert_called_once()
    warning_msg = agent._emit_warning.call_args[0][0]
    assert "exceeds the model context window" in warning_msg
    assert "compression.enabled: false" in warning_msg
    assert "10,000 tokens" in warning_msg
