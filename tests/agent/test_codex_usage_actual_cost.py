"""A provider-reported charge reaches the ``actual_cost_usd`` column on the codex path too.

``_record_codex_app_server_usage`` prices the turn through ``estimate_usage_cost``, which reports a
provider-supplied charge as ``status="actual"``. Forwarding only the status label left the amount
behind: the row said "actual" while ``actual_cost_usd`` stayed NULL, and the usage view — which adds
that column up — showed nothing. The turn path forwards the amount; this path has to as well.
"""
from types import SimpleNamespace

import pytest


class _FakeSessionDB:
    """Captures the kwargs `_queue_token_counts` would hand to the background writer."""

    def __init__(self):
        self.records = []

    def queue_token_counts(self, session_id, **kwargs):
        self.records.append((session_id, kwargs))


def _agent(session_db):
    return SimpleNamespace(
        _usage_anchor=None,
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
        context_compressor=None,
        event_callback=None,
        _session_db=session_db,
        _session_db_created=True,
        session_id="codex-session",
        model="grok-9-unpriced",
        provider="xai",
        base_url=None,
    )


def _turn(cost_ticks):
    return SimpleNamespace(
        token_usage_last={"inputTokens": 1_000, "cachedInputTokens": 0, "outputTokens": 100,
                          "reasoningOutputTokens": 0, "totalTokens": 1_100,
                          "cost_in_usd_ticks": cost_ticks},
        model_context_window=None,
    )


def test_a_reported_charge_is_forwarded_into_the_actual_column():
    from agent.codex_runtime import _record_codex_app_server_usage

    db = _FakeSessionDB()
    agent = _agent(db)

    returned = _record_codex_app_server_usage(agent, _turn(1_234_567_890))

    assert len(db.records) == 1
    _session_id, kwargs = db.records[0]
    assert kwargs["cost_status"] == "actual"
    assert kwargs["actual_cost_usd"] == pytest.approx(0.1234567890, rel=0, abs=1e-12)
    # The reply's own cost fields carry it too, so a caller reading the turn sees the same figure.
    assert returned["actual_cost_usd"] == pytest.approx(0.1234567890, rel=0, abs=1e-12)


def test_a_turn_without_a_reported_charge_leaves_the_actual_column_empty():
    from agent.codex_runtime import _record_codex_app_server_usage

    db = _FakeSessionDB()
    agent = _agent(db)

    _record_codex_app_server_usage(agent, _turn(None))

    _session_id, kwargs = db.records[0]
    assert kwargs["cost_status"] != "actual"
    assert kwargs["actual_cost_usd"] is None
