"""Tests for MoA trace aggregator-output capture across streaming modes.

The MoA full-turn trace (opt-in ``moa.save_traces``) must record the
aggregator's acting output whether the aggregator ran non-streaming (inline
capture at call time) or streaming (captured after the fact from the caller's
resolved assistant text). Before the streamed-capture fix, a streamed
aggregator left ``output: null`` in the trace and only pointed at state.db,
so an offline audit of a benchmark run (which drives the streaming display
path via ``hermes chat --query``) couldn't see what the aggregator actually
produced without joining to the session DB by hand.

These exercise the real ``consume_and_save_trace`` → ``save_moa_turn`` path
with real file I/O against a temp HERMES_HOME — no mocks on the write path.
"""

from __future__ import annotations

import json

import pytest

from agent.moa_loop import MoAChatCompletions


def _enable_traces(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and turn moa.save_traces on."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # save_moa_turn reads config via hermes_cli.config.load_config; stub it to
    # return traces-on so the test doesn't depend on a real config file.
    import agent.moa_trace as moa_trace

    monkeypatch.setattr(
        moa_trace,
        "load_config",
        lambda: {"moa": {"save_traces": True}},
        raising=False,
    )
    # load_config is imported lazily inside _traces_enabled_and_dir; patch the
    # source module attribute it imports from as well.
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg, "load_config", lambda: {"moa": {"save_traces": True}}, raising=False
    )
    return hermes_home / "moa-traces"


def _make_completions_with_pending(streamed: bool, inline_output):
    """Build a MoAChatCompletions with a pending trace mimicking one turn."""
    mc = MoAChatCompletions.__new__(MoAChatCompletions)
    mc._pending_trace = {
        "preset": "closed",
        "reference_outputs": [],  # references not under test here
        "aggregator_label": "openrouter:anthropic/claude-opus-4.8",
        "aggregator_slot": {
            "model": "anthropic/claude-opus-4.8",
            "provider": "openrouter",
        },
        "aggregator_temperature": 0.4,
        "aggregator_input_messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the thing"},
        ],
        "aggregator_output": inline_output,
        "aggregator_streamed": streamed,
    }
    return mc


def _read_single_trace(trace_dir, session_id):
    path = trace_dir / f"{session_id}.jsonl"
    assert path.exists(), f"trace file not written: {path}"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    return json.loads(lines[0])


def test_streamed_aggregator_output_captured_from_fallback(tmp_path, monkeypatch):
    """Streaming turn: inline output is None, fallback text is embedded."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    mc = _make_completions_with_pending(streamed=True, inline_output=None)

    mc.consume_and_save_trace(
        "sess_streamed",
        aggregator_output_fallback="the acting aggregator answer",
    )

    rec = _read_single_trace(trace_dir, "sess_streamed")
    agg = rec["aggregator"]
    assert agg["streamed"] is True
    assert agg["output"] == "the acting aggregator answer"
    assert agg["output_location"] == "inline_from_stream"


def test_non_streaming_prefers_inline_over_fallback(tmp_path, monkeypatch):
    """Non-streaming turn keeps its inline capture even if a fallback is passed."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    mc = _make_completions_with_pending(
        streamed=False, inline_output="inline captured text"
    )

    mc.consume_and_save_trace(
        "sess_inline",
        aggregator_output_fallback="SHOULD NOT BE USED",
    )

    rec = _read_single_trace(trace_dir, "sess_inline")
    agg = rec["aggregator"]
    assert agg["streamed"] is False
    assert agg["output"] == "inline captured text"
    assert agg["output_location"] == "inline"


def test_streamed_without_fallback_points_to_session_db(tmp_path, monkeypatch):
    """Streaming turn with no resolvable text falls back to the state.db pointer."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    mc = _make_completions_with_pending(streamed=True, inline_output=None)

    mc.consume_and_save_trace("sess_nofb", aggregator_output_fallback=None)

    rec = _read_single_trace(trace_dir, "sess_nofb")
    agg = rec["aggregator"]
    assert agg["streamed"] is True
    assert agg["output"] is None
    assert agg["output_location"] == "assistant_message_in_session_db"


def test_pending_trace_cleared_after_flush(tmp_path, monkeypatch):
    """A second flush is a no-op (pending cleared) — never double-writes."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    mc = _make_completions_with_pending(streamed=True, inline_output=None)

    mc.consume_and_save_trace("sess_once", aggregator_output_fallback="x")
    # Second call: pending is None now, must not append a second line.
    mc.consume_and_save_trace("sess_once", aggregator_output_fallback="y")

    path = trace_dir / "sess_once.jsonl"
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1


def test_empty_fallback_string_treated_as_missing(tmp_path, monkeypatch):
    """An empty-string fallback must not override to '' — treated as absent."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    mc = _make_completions_with_pending(streamed=True, inline_output=None)

    mc.consume_and_save_trace("sess_empty", aggregator_output_fallback="")

    rec = _read_single_trace(trace_dir, "sess_empty")
    agg = rec["aggregator"]
    assert agg["output"] is None
    assert agg["output_location"] == "assistant_message_in_session_db"
