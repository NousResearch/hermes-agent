"""A short session must outgrow structural backoff before its context fills."""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


def _compressor():
    return ContextCompressor(
        model="test/model", config_context_length=65_536,
        threshold_percent=0.5, threshold_tokens_cap=32_768,
        protect_first_n=1, quiet_mode=True,
    )


def test_growth_retries_structural_skip_and_rearms_after_another_noop():
    compressor = _compressor()
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "investigate"}]
    with patch("time.monotonic", return_value=1000.0):
        assert compressor.compress(messages, current_tokens=38_483) is messages
        assert compressor._last_compression_telemetry["failure_class"] == "insufficient_messages"
        assert not compressor.should_compress(38_483)
        assert not compressor.should_compress(38_600)
        assert compressor.should_compress(43_000)
        compressor.compress(messages, current_tokens=43_000)
        assert not compressor.should_compress(43_000)
        messages += [
            {"role": role, "content": f"Record {i}: " + "details " * 1500}
            for i in range(6) for role in ("assistant", "user")
        ]
        assert compressor.should_compress(48_000)
        with patch.object(compressor, "_generate_summary", return_value="Work completed so far.") as summarize:
            result = compressor.compress(messages, current_tokens=48_000)
        summarize.assert_called_once()
        assert len(result) < len(messages)
        assert any(row.get("content") == messages[-1]["content"] for row in result)


@pytest.mark.parametrize("guard", ["cooldown", "ineffective"])
def test_growth_does_not_bypass_summary_failure_or_ineffectiveness(guard):
    compressor = _compressor()
    messages = [{"role": "user", "content": "investigate"}]
    with patch("time.monotonic", return_value=1000.0):
        compressor.compress(messages, current_tokens=38_483)
        if guard == "cooldown":
            compressor._record_compression_failure_cooldown(300, "summary unavailable")
        else:
            compressor._ineffective_compression_count = 2
        should, reason = compressor.should_compress_info(43_000)
        assert not should
        assert reason.startswith(guard)
