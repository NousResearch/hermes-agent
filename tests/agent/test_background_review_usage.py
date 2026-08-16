"""Background-review usage attribution (issue #87250).

Background-review forks run with ``_session_db = None`` (persistence
isolation), so their provider-billed API calls were never recorded in
``session_model_usage``. ``_record_review_usage_to_parent`` closes that gap
by snapshotting the fork's in-memory counters and recording them against the
parent session via the aux-accounting chokepoint.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from agent import background_review
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


class _FakeParent:
    def __init__(self, session_db, session_id="sess-parent"):
        self._session_db = session_db
        self.session_id = session_id


def _usage(**overrides):
    base = {
        "model": "test-model",
        "provider": "test-provider",
        "base_url": "https://example.invalid/v1",
        "input_tokens": 12000,
        "output_tokens": 2400,
        "cache_read_tokens": 190000,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls": 5,
        "estimated_cost_usd": 0.05,
    }
    base.update(overrides)
    return base


def test_records_fork_usage_against_parent_session(db):
    db.create_session("sess-parent", source="cli")

    background_review._record_review_usage_to_parent(_FakeParent(db), _usage())

    rows = _usage_rows(db, "sess-parent")
    assert len(rows) == 1
    r = rows[0]
    assert r["task"] == "background_review"
    assert r["model"] == "test-model"
    assert r["billing_provider"] == "test-provider"
    assert r["input_tokens"] == 12000
    assert r["output_tokens"] == 2400
    assert r["cache_read_tokens"] == 190000
    assert r["api_call_count"] == 5
    assert r.get("estimated_cost_usd") == 0.05


def test_accumulates_repeated_forks_same_model(db):
    db.create_session("sess-parent", source="cli")
    parent = _FakeParent(db)

    background_review._record_review_usage_to_parent(parent, _usage(api_calls=5))
    background_review._record_review_usage_to_parent(parent, _usage(api_calls=7))

    rows = _usage_rows(db, "sess-parent")
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 24000
    assert rows[0]["api_call_count"] == 12


def test_noop_when_fork_made_no_calls(db):
    db.create_session("sess-parent", source="cli")

    background_review._record_review_usage_to_parent(
        _FakeParent(db),
        _usage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            api_calls=0,
        ),
    )

    assert _usage_rows(db, "sess-parent") == []


def test_noop_when_parent_has_no_session_db():
    background_review._record_review_usage_to_parent(_FakeParent(None), _usage())


def test_noop_when_parent_has_no_session_id(db):
    db.create_session("sess-parent", source="cli")

    background_review._record_review_usage_to_parent(
        _FakeParent(db, session_id=""), _usage()
    )

    assert _usage_rows(db, "sess-parent") == []


def test_survives_accounting_failure():
    class _BoomDB:
        def record_auxiliary_usage(self, *args, **kwargs):
            raise RuntimeError("simulated accounting failure")

    background_review._record_review_usage_to_parent(_FakeParent(_BoomDB()), _usage())


def test_classify_review_result():
    assert background_review._classify_review_result([]) == "none"
    assert background_review._classify_review_result(["Memory updated"]) == "memory"
    assert background_review._classify_review_result(["Skill 'x' patched"]) == "skill"
    assert (
        background_review._classify_review_result(
            ["Memory updated", "Skill 'x' created"]
        )
        == "skill+memory"
    )
    # Prefix-based — free-text "skill"/"memory" elsewhere must not misclassify.
    assert (
        background_review._classify_review_result(
            ["Skipped: no skill worth saving"]
        )
        == "none"
    )
    assert (
        background_review._classify_review_result(
            ["📝 Skill 'deploy' patched: \"a\" → \"b\""]
        )
        == "skill"
    )
    assert (
        background_review._classify_review_result(["User profile ➕ prefers terse"])
        == "memory"
    )


def test_enabled_config_failure_logs_warning(caplog):
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=RuntimeError("boom"),
    ), caplog.at_level(logging.WARNING, logger="agent.background_review"):
        assert background_review.is_background_review_enabled() is True
    assert any(
        "fail-open" in r.message.lower() or "leaving automatic" in r.message.lower()
        for r in caplog.records
    )


def test_spawn_reuses_provided_task_cfg_without_rereading():
    """One config load per spawn — the worker shares task_cfg."""
    task = {"enabled": True}
    agent = type("A", (), {})()
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=AssertionError("config must not be re-read when task_cfg is passed"),
    ):
        _target, prompt = background_review.spawn_background_review_thread(
            agent,
            messages_snapshot=[{"role": "user", "content": "hi"}],
            review_skills=True,
            task_cfg=task,
        )
        assert prompt  # built-in skill-review prompt selected
        assert callable(_target)

def test_log_review_completion_emits_thread_tag(caplog):
    with caplog.at_level(logging.INFO, logger="agent.background_review"):
        background_review._log_review_completion(
            _usage(api_calls=8, input_tokens=53000, output_tokens=400),
            "skill",
        )
    assert any(
        "thread=bg-review" in r.message
        and "calls=8" in r.message
        and "result=skill" in r.message
        for r in caplog.records
    )
