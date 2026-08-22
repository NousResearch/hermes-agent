"""MoA per-reference duration/stats + metrics-lite (independent of save_traces)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.moa_loop import _RefAccounting
from agent.moa_trace import _slot_trace, save_moa_turn
from agent.usage_pricing import CanonicalUsage


def test_slot_trace_includes_duration_and_stats():
    acct = _RefAccounting(
        CanonicalUsage(input_tokens=10, output_tokens=5),
        duration_s=1.25,
        stats={"tokens_per_second": 40.0, "ttft_ms": 12},
    )
    row = _slot_trace(acct, "ref-a")
    assert row["duration_s"] == 1.25
    assert row["stats"]["tokens_per_second"] == 40.0
    assert row["usage"]["input_tokens"] == 10


def test_save_moa_turn_writes_metrics_when_traces_on(tmp_path, monkeypatch):
    """metrics/moa-refs.jsonl writes alongside the full trace when enabled."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.moa_trace._traces_enabled_and_dir", return_value=Path(tmp_path) / "moa-traces"):
        save_moa_turn(
            session_id="sess-1",
            preset_name="test-preset",
            reference_outputs=[
                (
                    "r1",
                    "hello",
                    _RefAccounting(
                        CanonicalUsage(input_tokens=3, output_tokens=2),
                        model="m1",
                        provider="local",
                        duration_s=0.5,
                        stats={"ttft_ms": 8},
                    ),
                )
            ],
            aggregator_label="agg",
            aggregator_model="agg-m",
            aggregator_provider="xai",
            aggregator_temperature=0.2,
            aggregator_input_messages=[],
            aggregator_output="done",
            aggregator_streamed=False,
        )

    metrics_path = Path(tmp_path) / "metrics" / "moa-refs.jsonl"
    assert metrics_path.is_file(), "metrics-lite must write when traces are on"
    assert (Path(tmp_path) / "moa-traces" / "sess-1.jsonl").is_file()
    line = metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["session_id"] == "sess-1"
    assert rec["preset"] == "test-preset"
    assert rec["references"][0]["duration_s"] == 0.5
    assert rec["references"][0]["stats"]["ttft_ms"] == 8
    # no message bodies in metrics-lite
    assert "input_messages" not in rec["references"][0]
    assert "output" not in rec["references"][0]


def test_save_moa_turn_writes_metrics_when_traces_off(tmp_path, monkeypatch):
    """metrics-lite still writes when save_traces is off; full trace does not."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.moa_trace._traces_enabled_and_dir", return_value=None):
        save_moa_turn(
            session_id="sess-off",
            preset_name="test-preset",
            reference_outputs=[
                (
                    "r1",
                    "hello",
                    _RefAccounting(
                        CanonicalUsage(input_tokens=3, output_tokens=2),
                        duration_s=0.5,
                        stats={},
                    ),
                )
            ],
            aggregator_label="agg",
            aggregator_model="agg-m",
            aggregator_provider="xai",
            aggregator_temperature=0.2,
            aggregator_input_messages=[],
            aggregator_output="done",
            aggregator_streamed=False,
        )

    metrics_path = Path(tmp_path) / "metrics" / "moa-refs.jsonl"
    assert metrics_path.is_file(), "metrics-lite must write even when traces are off"
    rec = json.loads(metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["session_id"] == "sess-off"
    assert rec["references"][0]["duration_s"] == 0.5
    assert "output" not in rec["references"][0]
    assert not (Path(tmp_path) / "moa-traces").exists()


def test_usage_summary_forwards_response_stats():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "ollama"
    agent.api_mode = "chat_completions"
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        stats={"tokens_per_second": 55.0},
    )
    summary = AIAgent._usage_summary_for_api_request_hook(agent, response)
    assert summary is not None
    assert summary["stats"]["tokens_per_second"] == 55.0
