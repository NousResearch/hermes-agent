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

from hermes_cli.web_routers import analytics as web_server
from persistence.conversation.usage import TokenCounts
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


@pytest.fixture
def db(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        yield store


@pytest.fixture
def analytics(monkeypatch, db, tmp_path):
    """Run _get_models_analytics against the test DB."""
    from hermes_cli.web_server_sessions import _open_session_db_at_path

    monkeypatch.setattr(
        web_server,
        "_open_session_db_for_profile",
        lambda profile, read_only=True: _open_session_db_at_path(tmp_path / "state.db", read_only=True),
    )
    return lambda: web_server._get_models_analytics(days=30)


def _session(db, session_id, model, provider):
    seed_session(db, session_id, "cli", model=model)
    raw_holder(db)._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET billing_provider = ?, started_at = ? WHERE id = ?",
        (provider, time.time(), session_id),
    ))


def test_aux_tasks_do_not_duplicate_the_model_card(db, analytics):
    """Several aux tasks on one pair fold into that pair's single row."""
    _session(db, "s1", "gpt-5.6-terra", "openai-codex")
    for task in ("vision", "compression", "title_generation"):
        db.usage.record_auxiliary_usage(
            "s1", task, model="gpt-5.6-terra",
            billing_provider="openai-codex",
            tokens=TokenCounts(input_tokens=100, output_tokens=10),
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
    db.usage.record_auxiliary_usage(
        "s1", "vision", model="gemini-3-flash", billing_provider="gemini",
        tokens=TokenCounts(input_tokens=500, output_tokens=50),
    )

    result = analytics()
    pairs = {(m["model"], m["provider"]) for m in result["models"]}

    assert ("gemini-3-flash", "gemini") in pairs
    assert ("gpt-5.6-terra", "openai-codex") in pairs


def test_distinct_models_counts_aux_only_models(db, analytics):
    """The header count and the card list are two views of one row set.

    ``totals.distinct_models`` was a ``COUNT(DISTINCT model)`` over ``sessions``
    alone, so a model reached only through auxiliary usage drew a card the
    header never counted (#89631).
    """
    _session(db, "s1", "gpt-5.6-terra", "openai-codex")
    db.usage.record_auxiliary_usage(
        "s1", "title_generation", model="gpt-5.4-mini", billing_provider="openai-codex",
        tokens=TokenCounts(input_tokens=50, output_tokens=5),
    )

    result = analytics()

    assert {m["model"] for m in result["models"]} == {"gpt-5.6-terra", "gpt-5.4-mini"}
    assert result["totals"]["distinct_models"] == len({m["model"] for m in result["models"]})
