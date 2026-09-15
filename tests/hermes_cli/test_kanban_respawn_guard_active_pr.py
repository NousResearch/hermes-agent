"""The ``active_pr`` respawn guard must exempt implementer-actionable PRs.

Card t_108177c6 looped ``respawn_guarded {"reason":"active_pr"}`` for 4+ hours
while its PR sat OPEN on a CHANGES_REQUESTED verdict: only the implementer could
clear it, and the guard was what kept the implementer from being respawned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/4774"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def clear_pr_guard_cache():
    """Verdicts are cached per-process; don't leak between cases.

    ``getattr`` with a default so this file still COLLECTS against a build that
    predates the fix — otherwise the regression case would fail as a collection
    error, which proves nothing about behaviour. On unpatched code the
    assertions below are what must go red.
    """
    getattr(kbd, "_PR_GUARD_CACHE", {}).clear()
    yield
    getattr(kbd, "_PR_GUARD_CACHE", {}).clear()


def _task_with_pr_comment(conn) -> str:
    task_id = kb.create_task(conn, title="ship the thing", assignee="a")
    kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
    conn.commit()
    return task_id


def _stub_status(monkeypatch, status):
    monkeypatch.setattr(
        kbd, "_fetch_pr_status", lambda owner, repo, number: status,
    )


def test_changes_requested_pr_does_not_guard_respawn(kanban_home, monkeypatch):
    """The deadlock repro: only the implementer can clear CHANGES_REQUESTED."""
    _stub_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [{"conclusion": "SUCCESS"}],
    })
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_green_pr_awaiting_review_still_guards(kanban_home, monkeypatch):
    """The guard's designed case: a green PR waiting on a reviewer."""
    _stub_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"conclusion": "SUCCESS"}, {"state": "SUCCESS"},
        ],
    })
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_failing_check_pr_does_not_guard_respawn(kanban_home, monkeypatch):
    """A red CI run is implementer-actionable too."""
    _stub_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"conclusion": None, "status": "IN_PROGRESS"},
            {"conclusion": "FAILURE"},
        ],
    })
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_merged_pr_does_not_guard_respawn(kanban_home, monkeypatch):
    """Finished work: guarding on it just freezes the card."""
    _stub_status(monkeypatch, {
        "state": "MERGED",
        "reviewDecision": "APPROVED",
        "statusCheckRollup": [],
    })
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) is None


def test_indeterminate_pr_status_fails_closed(kanban_home, monkeypatch):
    """No ``gh`` / network blip must not stampede duplicate workers."""
    _stub_status(monkeypatch, None)
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_pr_verdict_is_cached_within_ttl(kanban_home, monkeypatch):
    """Second check inside the TTL reuses the verdict instead of re-fetching."""
    _stub_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [],
    })
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(conn)
        assert kbd.check_respawn_guard(conn, task_id) is None

        def _boom(owner, repo, number):
            raise AssertionError("cached verdict must be reused")

        monkeypatch.setattr(kbd, "_fetch_pr_status", _boom)
        assert kbd.check_respawn_guard(conn, task_id) is None
