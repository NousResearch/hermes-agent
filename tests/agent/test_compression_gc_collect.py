"""Regression tests for gc.collect after context compression (#70684)."""

import gc
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        _ = c.context_length
        return c


def _make_long_messages(count: int = 12) -> list[dict]:
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(1, count):
        role = "user" if i % 2 == 1 else "assistant"
        messages.append({"role": role, "content": f"message {i}"})
    return messages


def test_compress_calls_gc_collect(compressor, monkeypatch):
    """compress() must call gc.collect() once before returning the result."""
    mock_collect = MagicMock()
    monkeypatch.setattr(gc, "collect", mock_collect)

    # Force a successful compression path without needing a real aux model.
    compressor._generate_summary = lambda *args, **kwargs: "summary"
    messages = _make_long_messages(12)

    compressor.compress(messages, current_tokens=100000, force=True)

    assert mock_collect.call_count == 1, "gc.collect() was not called by compress()"
