"""Tests for Proactive Mode opportunity detection."""

from __future__ import annotations

import time

import pytest

from agent.proactive import ProactiveEngine
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "proactive.db")
    yield session_db
    session_db.close()


def _add_session(db, sid: str, source: str, messages: list[str], *, days_ago: int = 1):
    now = time.time()
    started = now - days_ago * 86400
    db.create_session(session_id=sid, source=source, model="test-model")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
    for idx, text in enumerate(messages):
        db.append_message(
            sid,
            role="user",
            content=text,
            timestamp=started + idx + 1,
        )


def test_repeated_pr_review_workflow_surfaces_opportunity(db):
    _add_session(
        db,
        "s1",
        "cli",
        ["Review PR #123, run tests, and draft a Slack update for the release."],
        days_ago=1,
    )
    _add_session(
        db,
        "s2",
        "cli",
        ["Can you review PR 456, test it, and write the Slack release update?"],
        days_ago=2,
    )
    _add_session(
        db,
        "s3",
        "telegram",
        ["Review this pull request, run the test suite, and compose the Slack update."],
        days_ago=3,
    )

    report = ProactiveEngine(db).generate(days=30)

    assert report["opportunities"]
    opportunity = report["opportunities"][0]
    assert opportunity["artifact_type"] == "workflow"
    assert opportunity["message_count"] == 3
    assert opportunity["session_count"] == 3
    assert "pr code review" in opportunity["title"]
    assert len(opportunity["evidence"]) == 3


def test_low_signal_history_stays_empty(db):
    _add_session(
        db,
        "s1",
        "cli",
        [
            "Thanks, that makes sense.",
            "Can you help me with something general later?",
            "/model openrouter:some-model",
        ],
    )

    report = ProactiveEngine(db).generate(days=30)

    assert report["opportunities"] == []


def test_recurring_digest_recommends_cron(db):
    _add_session(
        db,
        "s1",
        "cli",
        ["Generate a weekly digest of AI papers and summarize the top links."],
        days_ago=1,
    )
    _add_session(
        db,
        "s2",
        "discord",
        ["Please make the weekly AI papers digest with a short summary."],
        days_ago=2,
    )

    report = ProactiveEngine(db).generate(days=30)

    assert report["opportunities"]
    assert report["opportunities"][0]["artifact_type"] == "cron"


def test_source_filter_limits_candidates(db):
    _add_session(
        db,
        "s1",
        "cli",
        ["Review PR #123, run tests, and draft a Slack update for the release."],
    )
    _add_session(
        db,
        "s2",
        "telegram",
        ["Review PR #456, run tests, and draft a Slack update for the release."],
    )

    report = ProactiveEngine(db).generate(days=30, source="telegram")

    assert report["opportunities"] == []
    assert report["candidate_messages"] == 1


def test_terminal_format_is_preview_only(db):
    _add_session(
        db,
        "s1",
        "cli",
        ["Review PR #123, run tests, and draft a Slack update for the release."],
    )
    _add_session(
        db,
        "s2",
        "cli",
        ["Review PR #456, run tests, and draft a Slack update for the release."],
    )
    engine = ProactiveEngine(db)
    report = engine.generate(days=30)

    text = engine.format_terminal(report)

    assert "Proactive opportunities" in text
    assert "Preview only: nothing was created" in text
    assert "Evidence:" in text
