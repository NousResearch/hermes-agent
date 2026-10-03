"""Throughput math measures the decode phase (first token → completion), not the wall clock.

A cold local model spends most of an API call loading before the first token
streams; dividing by the full duration then reports load wait as generation
slowness. These are behavior contracts on that relationship, not snapshots.
"""
from collections import deque
from types import SimpleNamespace

from agent.turn_usage import decode_duration_for_tps, record_response_usage, reset_throughput_history


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def _usage(output=20, prompt=100):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=output, total_tokens=prompt + output,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                           completion_tokens_details=None)


def _record(agent, duration, start=None, first_chunk=None, output=20):
    resp = SimpleNamespace(usage=_usage(output), id=None, model="m")
    if first_chunk is not None:
        agent._last_api_first_chunk_at = first_chunk
    else:
        agent._last_api_first_chunk_at = None
    record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                          api_duration=duration, api_start_time=start,
                          compression_attempts=0, max_compression_attempts=3)


def test_decode_excludes_time_to_first_token():
    assert decode_duration_for_tps(12.0, 100.0, 110.0) == 12.0 - 10.0


def test_decode_falls_back_without_chunk_timing():
    assert decode_duration_for_tps(12.0, None, None) == 12.0
    assert decode_duration_for_tps(12.0, 100.0, None) == 12.0
    assert decode_duration_for_tps(12.0, None, 105.0) == 12.0


def test_decode_ignores_implausible_chunk_timing():
    assert decode_duration_for_tps(12.0, 100.0, 99.0) == 12.0  # stale: before the call
    assert decode_duration_for_tps(12.0, 100.0, 200.0) == 12.0  # beyond the call
    assert decode_duration_for_tps(12.0, 100.0, float("nan")) == 12.0


def test_history_records_decode_duration(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, 10.0, start=1000.0, first_chunk=1008.0, output=20)
        assert list(a._api_latency_history) == [10.0]  # latency keeps the wall clock
        assert list(a._api_decode_duration_history) == [2.0]
        assert list(a._api_output_history) == [20]
    finally:
        a.close()


def test_history_records_wall_clock_without_chunk_timing(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, 10.0, output=20)
        assert list(a._api_latency_history) == [10.0]
        assert list(a._api_decode_duration_history) == [10.0]  # no chunk timing: decode is the call
    finally:
        a.close()


def test_reset_clears_histories(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        a._api_latency_history = deque([8.0, 2.0], maxlen=10)
        a._api_decode_duration_history = deque([7.0, 2.0], maxlen=10)
        a._api_output_history = deque([10, 20], maxlen=10)
        reset_throughput_history(a)
        assert list(a._api_latency_history) == []
        assert list(a._api_decode_duration_history) == []
        assert list(a._api_output_history) == []
    finally:
        a.close()


def test_reset_tolerates_missing_histories():
    reset_throughput_history(SimpleNamespace())
