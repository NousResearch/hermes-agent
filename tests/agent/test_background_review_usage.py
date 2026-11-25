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
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

@pytest.fixture
def db(tmp_path):
    return storage_handle(tmp_path)

def _usage_rows(db, session_id):
    with raw_holder(db)._read_ctx() as conn:
        rows = conn.execute(
            "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]

class _FakeParent:
    def __init__(self, session_db, session_id="sess-parent"):
        self._session_db = session_db
        self.persistence = session_db
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
    seed_session(db, "sess-parent", "cli")

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
    seed_session(db, "sess-parent", "cli")
    parent = _FakeParent(db)

    background_review._record_review_usage_to_parent(parent, _usage(api_calls=5))
    background_review._record_review_usage_to_parent(parent, _usage(api_calls=7))

    rows = _usage_rows(db, "sess-parent")
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 24000
    assert rows[0]["api_call_count"] == 12

def test_records_fork_usage_through_the_parents_writer_handle(db, tmp_path):
    import persistence_assembly
    seed_session(db, "sess-parent", "cli")
    parent = _FakeParent(db)

    with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=tmp_path)) as handle:
        parent.persistence = handle
        background_review._record_review_usage_to_parent(parent, _usage(api_calls=5))
        background_review._record_review_usage_to_parent(parent, _usage(api_calls=7))

    rows = _usage_rows(db, "sess-parent")
    assert len(rows) == 1
    r = rows[0]
    assert (r["task"], r["model"], r["billing_provider"], r["billing_base_url"]) == (
        "background_review", "test-model", "test-provider", "https://example.invalid/v1")
    assert (r["input_tokens"], r["output_tokens"], r["cache_read_tokens"], r["api_call_count"]) == (24000, 4800, 380000, 12)
    assert r["estimated_cost_usd"] == pytest.approx(0.10)

def test_noop_when_fork_made_no_calls(db):
    seed_session(db, "sess-parent", "cli")

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

def test_noop_when_parent_has_no_session_id(db):
    seed_session(db, "sess-parent", "cli")

    background_review._record_review_usage_to_parent(
        _FakeParent(db, session_id=""), _usage()
    )

    assert _usage_rows(db, "sess-parent") == []

def test_the_fork_records_its_usage_after_the_stop_closed_the_parents_handle(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import persistence_assembly
    scope = persistence_assembly.profile_scope(home=tmp_path)
    handle = persistence_assembly.open_persistence(scope)
    seed_session(handle, "sess-parent", "cli")
    parent = SimpleNamespace(_session_db=handle, persistence=handle, session_id="sess-parent")

    def fork(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        def run_conversation(**kwargs):
            handle.close()  # the gateway's stop closes the parent's handle while the review is answering
        return SimpleNamespace(model="test-model", provider="test-provider", base_url=None, session_api_calls=1,
                               session_input_tokens=3, run_conversation=run_conversation, _session_messages=[]), {}, False
    monkeypatch.setattr(background_review, "_parent_can_emit_tool_calls", lambda agent: True)
    monkeypatch.setattr(background_review, "build_cache_parity_fork", fork)
    monkeypatch.setattr(background_review, "_review_tool_whitelist", lambda *args: (set(), set()))
    monkeypatch.setattr(background_review, "_track_review_fork", lambda *args, **kwargs: None)
    background_review._run_review_in_thread(parent, [], "review")
    with persistence_assembly.open_persistence(scope) as store:
        rows = _usage_rows(store, "sess-parent")
    assert [(r["task"], r["api_call_count"], r["input_tokens"]) for r in rows] == [("background_review", 1, 3)]

def test_survives_accounting_failure():
    class _BoomDB:
        read_only = False

        class usage:
            @staticmethod
            def record_auxiliary_usage(*args, **kwargs):
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
        assert background_review.load_background_review_settings()[0] is True
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
