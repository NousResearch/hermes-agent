"""Models analytics must attribute usage per API call, not per session (#71778).

The ``sessions`` row keeps only the final ``(model, billing_provider)`` pair of a
session. Aggregating the Models page from it attributed every token of a
session that switched models/providers mid-run (manually via ``/model`` or
silently via a fallback chain) to that single last pair, producing impossible
pairs on the dashboard. ``session_model_usage`` records each API call with the
route active at call time and is the correct source — the same one Insights
(``_compute_model_breakdown``) reads. Tool calls are session-level data and must
stay summed from ``sessions`` rather than being zeroed by the source switch.
"""

import time

import pytest

from hermes_cli.web_routers import analytics as web_server
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def models(monkeypatch, db):
    """Run _get_models_analytics against the test DB."""
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile, read_only=True: db,
    )
    # The helper closes the DB it is handed; keep it open for assertions.
    monkeypatch.setattr(db, "close", lambda: None)
    return lambda: web_server._get_models_analytics(days=30)


def _touch(db, session_id, model, provider, *, input_tokens, output_tokens, api_calls=1):
    """Record one main-loop API call's delta on the route active at call time."""
    db.update_token_counts(
        session_id, input_tokens=input_tokens, output_tokens=output_tokens,
        model=model, billing_provider=provider, api_call_count=api_calls,
    )


def _by_pair(result):
    return {(m["model"], m["provider"]): m for m in result["models"]}


def test_session_switching_providers_yields_separated_rows(db, models):
    """One session, two routes: both pairs appear with their own tokens (#71778)."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=30_000, output_tokens=1_000)
    _touch(db, "s1", "anthropic/claude-opus-5", "nous", input_tokens=31_000, output_tokens=2_000)
    # The sessions row keeps only the final pair — the old misattribution source.
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET model = ?, billing_provider = ? WHERE id = ?",
            ("anthropic/claude-opus-5", "nous", "s1"),
        )
        db._conn.commit()

    result = models()
    pairs = _by_pair(result)

    assert ("gpt-5.6-sol", "openai-codex") in pairs, "first-phase route must stay visible"
    assert ("anthropic/claude-opus-5", "nous") in pairs, "second-phase route must stay visible"
    # Tokens split per route, not all lumped on the final pair.
    assert pairs[("gpt-5.6-sol", "openai-codex")]["input_tokens"] == 30_000
    assert pairs[("anthropic/claude-opus-5", "nous")]["input_tokens"] == 31_000
    # A session counted once per pair it actually used, never COUNT(*)-inflated.
    assert pairs[("gpt-5.6-sol", "openai-codex")]["sessions"] == 1
    assert pairs[("anthropic/claude-opus-5", "nous")]["sessions"] == 1


def test_phantom_pair_never_appears_for_switched_session(db, models):
    """The sessions-only pair (model of phase 1, provider of phase 2) must not exist."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    _touch(db, "s1", "deepseek-v4-pro", "deepseek", input_tokens=200, output_tokens=20)
    # Fallback chains rewrite the sessions row without any user action (#71778 comment).
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET model = ?, billing_provider = ? WHERE id = ?",
            ("gpt-5.6-sol", "deepseek", "s1"),
        )
        db._conn.commit()

    pairs = _by_pair(models())
    assert ("gpt-5.6-sol", "deepseek") not in pairs
    assert set(pairs) == {("gpt-5.6-sol", "openai-codex"), ("deepseek-v4-pro", "deepseek")}


def test_tool_calls_survive_the_source_switch(db, models):
    """Tool calls stay summed from sessions; the query switch must not zero them."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    db.append_message("s1", "assistant", "running", tool_calls=[{"name": "terminal", "arguments": {}}])
    db.append_message("s1", "tool", "result")
    db.append_message("s1", "assistant", "done", tool_calls=[{"name": "read_file", "arguments": {}}])
    db.append_message("s1", "tool", "result")

    row = _by_pair(models())[("gpt-5.6-sol", "openai-codex")]
    assert row["tool_calls"] == 2


def test_fallback_provider_session_attributed_to_actual_route(db, models):
    """Aux rows are folded onto the main card without duplicating it (#23270, #89631)."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    db.record_auxiliary_usage(
        "s1", "vision", model="gemini-3-flash", billing_provider="gemini",
        input_tokens=500, output_tokens=50,
    )

    result = models()
    pairs = _by_pair(result)
    assert pairs[("gpt-5.6-sol", "openai-codex")]["input_tokens"] == 100
    assert pairs[("gemini-3-flash", "gemini")]["input_tokens"] == 500
    # Aux usage happened inside an already-counted session: the main card's
    # session count is not inflated by the aux row.
    assert pairs[("gpt-5.6-sol", "openai-codex")]["sessions"] == 1


def test_db_without_session_model_usage_falls_back_to_sessions(db, models):
    """Pre-v17 DBs (no session_model_usage) still render the Models page."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    db.append_message("s1", "assistant", "running", tool_calls=[{"name": "terminal", "arguments": {}}])
    db.append_message("s1", "tool", "result")
    with db._lock:
        db._conn.execute("DROP TABLE session_model_usage")
        db._conn.commit()

    row = _by_pair(models())[("gpt-5.6-sol", "openai-codex")]
    assert row["input_tokens"] == 100
    assert row["tool_calls"] == 1


def test_window_excludes_old_sessions(db, models):
    """The cutoff still filters by session start time."""
    db.create_session("s1", source="cli", model="gpt-5.6-sol")
    _touch(db, "s1", "gpt-5.6-sol", "openai-codex", input_tokens=100, output_tokens=10)
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (time.time() - 90 * 86400, "s1"))
        db._conn.commit()

    assert models()["models"] == []
