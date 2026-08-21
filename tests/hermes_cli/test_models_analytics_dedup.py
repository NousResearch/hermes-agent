"""Models analytics must emit one row per (model, provider) pair (#89631).

``_aux_usage_rows`` groups by (model, task, billing_provider), so a model
used for several auxiliary tasks returns several rows for the SAME (model,
billing_provider) pair. Appending them unconditionally rendered one
duplicate card per aux task, and their session counts summed past
``totals.total_sessions`` because aux usage happens inside sessions the
sessions-derived row already counted.
"""

import time

import pytest

from hermes_cli import web_server
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def analytics(monkeypatch, db):
    """Run _get_models_analytics against the test DB."""
    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile, read_only=True: db,
    )
    # The helper closes the DB it is handed; keep it open for assertions.
    monkeypatch.setattr(db, "close", lambda: None)
    return lambda: web_server._get_models_analytics(days=30)


def _session(db, session_id, model, provider):
    db.create_session(session_id, source="cli", model=model)
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET billing_provider = ?, started_at = ? WHERE id = ?",
            (provider, time.time(), session_id),
        )
        db._conn.commit()


def test_aux_tasks_do_not_duplicate_the_model_card(db, analytics):
    """Several aux tasks on one pair fold into that pair's single row."""
    _session(db, "s1", "gpt-5.6-terra", "openai-codex")
    for task in ("vision", "compression", "title_generation"):
        db.record_auxiliary_usage(
            "s1", task, model="gpt-5.6-terra",
            billing_provider="openai-codex",
            input_tokens=100, output_tokens=10,
        )

    result = analytics()
    pairs = [(m["model"], m["provider"]) for m in result["models"]]

    assert pairs == [("gpt-5.6-terra", "openai-codex")]
    # Session counts must stay reconcilable with the totals block.
    assert sum(m["sessions"] for m in result["models"]) == result["totals"]["total_sessions"]
    # Aux tokens are still accounted for, just on the one row.
    assert result["models"][0]["input_tokens"] == 300


def test_aux_only_model_still_gets_its_own_row(db, analytics):
    """Regression guard for #23270 — a dedicated aux model stays visible."""
    _session(db, "s1", "gpt-5.6-terra", "openai-codex")
    db.record_auxiliary_usage(
        "s1", "vision", model="gemini-3-flash", billing_provider="gemini",
        input_tokens=500, output_tokens=50,
    )

    result = analytics()
    pairs = {(m["model"], m["provider"]) for m in result["models"]}

    assert ("gemini-3-flash", "gemini") in pairs
    assert ("gpt-5.6-terra", "openai-codex") in pairs
