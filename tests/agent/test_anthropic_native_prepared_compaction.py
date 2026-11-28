"""The upstream prepared-summary splice and native on-demand summary must coexist."""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.prepared_compaction import PreparedCompaction, maybe_prepare


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop",
    )])


@pytest.mark.parametrize("force", [False, True], ids=["splice-prepared", "manual-native"])
def test_prepared_summary_precedes_native_unless_fresh_summary_is_requested(force):
    with patch("agent.context_compressor.get_model_context_length", return_value=40960):
        compressor = ContextCompressor(model="claude-opus-4-7", protect_first_n=3, protect_last_n=5,
                                       quiet_mode=True)
    compressor.threshold_tokens, compressor.tail_token_budget = 20000, 2500
    compressor.last_prompt_tokens, compressor._session_id = 19500, "S1"
    compressor.prepared_compaction = PreparedCompaction()
    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "detail " * 200}
        for i in range(30)
    ]
    agent = SimpleNamespace(context_compressor=compressor, compression_enabled=True, api_mode="anthropic_messages")
    native = Mock()
    native.covered_prefix.side_effect = len
    native.summarize.return_value = SimpleNamespace(
        text="native checkpoint", model="claude-opus-4-7", duration_ms=1, usage={"input_tokens": 10},
    )
    with patch("agent.context_compressor.call_llm", return_value=_response("prepared checkpoint")) as auxiliary:
        thread = maybe_prepare(agent, messages, origin="tool-batch")
        assert thread is not None
        thread.join(5)
        assert not thread.is_alive()
        prepared = compressor.prepared_compaction._entry
        assert prepared is not None and auxiliary.call_count == 1
        later = messages + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"new turn {i} " + "fact " * 200}
            for i in range(8)
        ]
        kwargs = {"force": force}
        # Exercise exact main too: fail on behavior, not on the new keyword being absent.
        if "native_summary" in inspect.signature(compressor.compress).parameters:
            kwargs["native_summary"] = native
        compressed = compressor.compress(later, current_tokens=21000, **kwargs)
    assert auxiliary.call_count == 1, "a prepared checkpoint or fresh native source avoids another auxiliary call"
    expected = "native checkpoint" if force else prepared.summary
    assert any(expected in str(row.get("content", "")) for row in compressed)
    assert native.summarize.call_count == int(force)
    assert compressor.prepared_compaction._entry is None
    if not force:
        for original, row in zip(later[prepared.compress_end:], compressed[-len(later[prepared.compress_end:]):]):
            assert original["content"] in row["content"], "all turns after the prepared boundary stay verbatim"
        telemetry = compressor._last_compression_telemetry
        assert telemetry is not None and telemetry["prepared_summary"] is True
