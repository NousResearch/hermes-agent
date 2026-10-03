"""End-to-end: does a usage-less call actually land in a real state.db?

The unit tests use a fake DB that records queue_token_counts calls. This one opens a real
SessionDB, drives record_response_usage against it, and reads the rows back — because the
interesting failure modes live below the call: has_usage gating, the session_model_usage
upsert, and the coalescer that merges adjacent deltas.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


class _Compressor:
    awaiting_real_usage_after_compression = False

    def update_from_response(self, _payload):
        pass

    def note_usage_less_response(self):
        pass


def _agent(db, session_id):
    agent = SimpleNamespace(
        context_compressor=_Compressor(),
        session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_reasoning_tokens=0, session_estimated_cost_usd=0.0,
        session_cost_status=None, session_cost_source=None,
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-opus-5", api_mode="chat_completions",
        session_id=session_id, _session_db=db, _session_db_created=True,
        _api_latency_history=[], _api_output_history=[], verbose_logging=False,
        _ensure_db_session=lambda: None,
    )
    return agent


def _record(agent, response):
    from agent.turn_usage import record_response_usage

    record_response_usage(agent, response, messages=[{"role": "user", "content": "hi"}],
                          api_call_count=1, api_duration=0.1,
                          compression_attempts=0, max_compression_attempts=3)


@pytest.fixture(autouse=True)
def _stub_session_source(monkeypatch):
    """Keep ``_agent_session_source`` off the import chain (see the unit-test file)."""
    import agent.turn_usage as turn_usage

    monkeypatch.setattr(turn_usage, "_agent_session_source", lambda _agent: "cli")


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    yield d
    d.close()


def _row(db, session_id):
    rows = db._read_all(
        "SELECT api_call_count, input_tokens, output_tokens, billing_provider, model "
        "FROM sessions WHERE id = ?", (session_id,))
    return rows[0] if rows else None


def _usage_rows(db, session_id):
    return db._read_all(
        "SELECT model, billing_provider, api_call_count, input_tokens, output_tokens "
        "FROM session_model_usage WHERE session_id = ?", (session_id,))


def test_usage_less_call_lands_in_real_db(db):
    """The bug, verified against SQLite rather than a mock."""
    db.create_session("s1", "cli")
    agent = _agent(db, "s1")

    _record(agent, SimpleNamespace(usage=None))
    db.flush_token_counts()

    row = _row(db, "s1")
    assert row is not None, "session row missing"
    assert row["api_call_count"] == 1, "the call happened but was never recorded"
    assert row["billing_provider"] == "openrouter", "route was known and must be stored"
    assert row["model"] == "anthropic/claude-opus-5"
    assert row["input_tokens"] == 0 and row["output_tokens"] == 0

    usage = _usage_rows(db, "s1")
    assert len(usage) == 1, "a call with no token data still gets an attribution row"
    assert usage[0]["api_call_count"] == 1


def test_usage_less_call_claims_no_price(db):
    """Unknown cost must not be recorded as a measured one.

    ``estimated_cost_usd`` is a running sum, so an unmeasured call contributes 0 — that is
    correct and unavoidable. What must stay unset is the *provenance*: ``cost_status`` and
    ``cost_source`` are what separate "we priced this" from "we could not price this", and a
    usage-less response must not leave them claiming a real estimate.
    """
    db.create_session("s2", "cli")
    agent = _agent(db, "s2")

    _record(agent, SimpleNamespace(usage=None))
    db.flush_token_counts()

    rows = db._read_all(
        "SELECT estimated_cost_usd, actual_cost_usd, cost_status, cost_source "
        "FROM sessions WHERE id = ?", ("s2",))
    row = rows[0]
    assert row["estimated_cost_usd"] == 0, "a running sum starts at zero"
    assert row["cost_status"] is None, "a usage-less call was never priced"
    assert row["cost_source"] is None
    assert row["actual_cost_usd"] is None


def test_two_usage_less_calls_coalesce_to_two(db):
    """The coalescer must sum api_call_count, not collapse the pair into one call."""
    db.create_session("s3", "cli")
    agent = _agent(db, "s3")

    _record(agent, SimpleNamespace(usage=None))
    _record(agent, SimpleNamespace(usage=None))
    db.flush_token_counts()

    assert _row(db, "s3")["api_call_count"] == 2


def test_usage_less_then_usage_bearing_both_count(db):
    """A no-usage call followed by a real one: 2 calls, the real tokens intact.

    This is the adjacent-deltas case the coalescer has to get right — merging must add the
    counts without discarding the tokens the second call reported.
    """
    db.create_session("s4", "cli")
    agent = _agent(db, "s4")

    _record(agent, SimpleNamespace(usage=None))
    usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    _record(agent, SimpleNamespace(usage=usage))
    db.flush_token_counts()

    row = _row(db, "s4")
    assert row is not None
    assert row["api_call_count"] == 2
    assert row["input_tokens"] == 100, "the measured call's tokens must survive coalescing"
    assert row["output_tokens"] == 50


def test_usage_bearing_only_is_unchanged(db):
    """Baseline: the common path keeps writing everything it always did."""
    db.create_session("s5", "cli")
    agent = _agent(db, "s5")

    usage = SimpleNamespace(
        prompt_tokens=200, completion_tokens=80, total_tokens=280,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    _record(agent, SimpleNamespace(usage=usage))
    db.flush_token_counts()

    row = _row(db, "s5")
    assert row is not None
    assert row["api_call_count"] == 1
    assert row["input_tokens"] == 200
    assert row["output_tokens"] == 80
    assert row["billing_provider"] == "openrouter"

def test_usage_less_call_does_not_claim_the_session_route(db):
    """The regression the adversarial review caught: ``first_accounted_route``.

    create_session records the requested route before any API call. If that provider fails
    and a fallback answers, the sessions row must land on the fallback — the one whose tokens
    are actually on the row — not on the provider that failed. A usage-less call cannot
    evidence which route served the session, so it must not win the route.

    Measured before the fix: ``model=m-A billing_provider=A`` with 100 real tokens belonging
    to B. After: ``model=m-B billing_provider=B``, api_call_count=2.
    """
    db.create_session("route", "cli")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET model = ?, billing_provider = ?, billing_base_url = ? WHERE id = ?",
        ("m-A", "A", "https://a.example", "route")))

    # The failed provider answers but sends no usage.
    db.queue_token_counts("route", source="cli", billing_provider="A",
                          billing_base_url="https://a.example", model="m-A",
                          api_call_count=1, route_authoritative=False)
    db.flush_token_counts()

    # The fallback answers and does report usage.
    db.queue_token_counts("route", source="cli", input_tokens=100, output_tokens=50,
                          billing_provider="B", billing_base_url="https://b.example",
                          model="m-B", api_call_count=1)
    db.flush_token_counts()

    row = _row(db, "route")
    assert row["billing_provider"] == "B", "the route must point at the tokens' origin"
    assert row["model"] == "m-B"
    assert row["input_tokens"] == 100
    assert row["api_call_count"] == 2, "both calls counted, including the usage-less one"


def test_deliberate_route_switch_still_claims_the_route(db):
    """A caller that says its write IS the route must still win, with no tokens involved.

    ``test_first_accounted_route_replaces_all_route_fields_atomically`` covers this at the
    DB level; this pins it through the public flag so the two stay tied together.
    """
    db.create_session("switch", "cli")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET model = ?, billing_provider = ? WHERE id = ?",
        ("primary", "primary-provider", "switch")))

    db.queue_token_counts("switch", source="cli", model="fallback",
                          billing_provider="fallback-provider", api_call_count=1,
                          route_authoritative=True)
    db.flush_token_counts()

    row = _row(db, "switch")
    assert row["billing_provider"] == "fallback-provider", "a /model switch must still apply"


def test_usage_less_call_before_any_measured_call_leaves_route_open(db):
    """The sentinel must survive a usage-less call so a later real call can still claim it."""
    db.create_session("open", "cli")
    db.queue_token_counts("open", source="cli", billing_provider="A", model="m-A",
                          api_call_count=1, route_authoritative=False)
    db.flush_token_counts()
    db.queue_token_counts("open", source="cli", billing_provider="B", model="m-B",
                          api_call_count=1, route_authoritative=True)
    db.flush_token_counts()

    rows = _usage_rows(db, "open")
    assert len(rows) == 2, "per-model attribution keeps both routes"
