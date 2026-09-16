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


def test_trace_jsonl_redacts_nested_inputs_and_preserves_live_operands(tmp_path, monkeypatch):
    """Opt-in MoA diagnostics remain useful without durable raw secrets."""
    trace_dir = _enable_traces(tmp_path, monkeypatch)
    from agent.moa_trace import save_moa_turn

    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    ref_messages = [{"role": "user", "content": {"api_key": secret}}]
    ref_output = f"reference output {secret}"
    aggregator_input = [{"role": "user", "content": {"token": secret}}]
    aggregator_output = f"aggregator output {secret}"
    accounting = type("Accounting", (), {
        "messages": ref_messages, "output": ref_output, "usage": None,
        "model": "reference", "provider": "test", "temperature": 0,
        "cost_usd": 0, "cost_status": "known", "cost_source": "test",
    })()

    save_moa_turn(
        session_id="redacted-moa", preset_name="test", reference_outputs=[("ref", ref_output, accounting)],
        aggregator_label="agg", aggregator_model="model", aggregator_provider="test", aggregator_temperature=0,
        aggregator_input_messages=aggregator_input, aggregator_output=aggregator_output, aggregator_streamed=False,
    )

    persisted = (trace_dir / "redacted-moa.jsonl").read_text(encoding="utf-8")
    assert secret not in persisted
    assert "references" in persisted and "aggregator" in persisted and "input_messages" in persisted
    assert ref_messages[0]["content"]["api_key"] == secret
    assert aggregator_input[0]["content"]["token"] == secret
    assert accounting.output == ref_output


