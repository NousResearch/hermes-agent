"""When terminal summary failures stack up before a success, compress() reports the most severe one.

Only a successful summary clears a terminal failure, so a later failure of another class adds to an
earlier one; the abort warning and telemetry name the higher-precedence class (access, network,
truncated, empty content, overload), whichever order the failures arrived in.
"""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor

_EMPTY = {"choices": [{"message": {"content": ""}}]}
_NETWORK = ConnectionError("Connection error.")


def _summary_call(outcome):
    if isinstance(outcome, Exception):
        return patch("agent.context_compressor.call_llm", side_effect=outcome)
    return patch("agent.context_compressor.call_llm", return_value=outcome)


def _msgs(n=12):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "x" * 40} for i in range(n)]


@pytest.mark.parametrize("first, second", [(_EMPTY, _NETWORK), (_NETWORK, _EMPTY)], ids=["empty_then_network",
                                                                                     "network_then_empty"])
def test_stacked_failures_report_the_higher_precedence_class(first, second):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = _msgs()
    for outcome in (first, second):
        with _summary_call(outcome):
            assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    assert compressor._last_compress_aborted is True
    assert compressor._last_compression_telemetry["failure_class"] == "summary_network_failure"


def test_a_successful_summary_clears_the_stacked_failure():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = _msgs()
    with patch("agent.context_compressor.call_llm", side_effect=_NETWORK):
        assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    with patch("agent.context_compressor.call_llm",
               return_value={"choices": [{"message": {"content": "## Goal\nA real summary."}}]}):
        assert compressor.compress(msgs, current_tokens=999999, force=True) != msgs
    assert compressor._last_compress_aborted is False
