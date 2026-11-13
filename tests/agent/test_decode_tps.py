"""Decode-only throughput: the status bar's ``decode_tps`` must measure the model's
decode window (first streamed chunk → last chunk), not the whole call.

The pre-existing ``avg_tps`` divides by the full call, so a long prefill or a small
tool-decision turn reads as "slow" while the model is decoding fast. The new figure
pairs ``_last_api_first_chunk_at`` with ``_last_api_stream_end_at``.

Contract:

- Streamed call: ``_last_api_decode_tps`` = output tokens / (end - first chunk), i.e.
  TTFT and prefill are excluded.
- Non-streamed call (no first-chunk stamp) or a stale end stamp alone: None, never a
  fabricated rate.
- The value rides the usage payloads both status-bar builders read.
"""

from __future__ import annotations

from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                   enabled_toolsets=["file"])


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
                           prompt_tokens_details=None, completion_tokens_details=None)


def _record(agent, *, prompt=1_000, completion=200, duration=8.0):
    from agent import turn_usage
    turn_usage.record_response_usage(
        agent, SimpleNamespace(usage=_usage(prompt, completion), id=None, model=agent.model),
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=duration, compression_attempts=0, max_compression_attempts=3)


def test_decode_tps_excludes_the_pre_first_token_wait(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        a._last_api_first_chunk_at = 1_000.0
        a._last_api_stream_end_at = 1_002.0  # 2s of decode; 6s of the 8s call was TTFT + prefill
        _record(a)
        assert a._last_api_decode_tps == 100.0
        # the end-to-end average is diluted by the wait, which is the whole point
        assert a._api_output_history[-1] == 200
    finally:
        a.close()


def test_decode_tps_is_none_without_a_streamed_window(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        a._last_api_first_chunk_at = None
        a._last_api_stream_end_at = 1_010.0  # stale end stamp alone must not invent a rate
        _record(a)
        assert a._last_api_decode_tps is None
    finally:
        a.close()


def test_single_chunk_stream_is_not_a_rate(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        a._last_api_first_chunk_at = 1_000.0
        a._last_api_stream_end_at = 1_000.01  # one buffered chunk: no window to measure
        _record(a)
        assert a._last_api_decode_tps is None
    finally:
        a.close()


def test_both_statusbar_builders_carry_decode_tps(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        a._last_api_first_chunk_at = 1_000.0
        a._last_api_stream_end_at = 1_001.0
        _record(a, completion=150)
        from gateway.session_usage import tui_usage, usage_stats
        from tui_gateway.server import _get_usage
        assert _get_usage(a)["decode_tps"] == 150.0
        assert tui_usage(usage_stats(a))["decode_tps"] == 150.0
    finally:
        a.close()
