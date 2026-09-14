"""Decision table for the ``active_pr`` respawn-guard exemption.

Companion to ``test_kanban_respawn_guard_changes_requested_deadlock.py`` (the
deadlock repro). This file pins the *scoping* of the exemption, which round 1
of the fix got wrong: it exempted on ANY failing rollup entry, which on the
live board un-guarded 24 of 40 open PRs — 11 of them solely because the
advisory ``review-gate`` check is red BY DESIGN while a PR awaits review. Only
checks that actually gate the merge (``isRequired``) may justify a respawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


PR_URL = "https://github.com/VibeTechnologies/AgentPod/pull/4888"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _clear_cache():
    # ``getattr`` default keeps this file COLLECTABLE on a build predating the
    # fix, so the RED run reports behaviour rather than a collection error.
    getattr(kbd, "_PR_GUARD_CACHE", {}).clear()
    yield
    getattr(kbd, "_PR_GUARD_CACHE", {}).clear()


def _card(conn) -> str:
    task_id = kb.create_task(conn, title="ship it", assignee="a")
    kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
    conn.commit()
    return task_id


def _stub(monkeypatch, status):
    monkeypatch.setattr(kbd, "_fetch_pr_status", lambda o, r, n: status, raising=False)
    monkeypatch.setattr(
        kbd, "_pr_url_is_open",
        lambda _u: bool(status) and (status.get("state") == "OPEN"),
        raising=False,
    )


# --------------------------------------------------------------------------
# Scoping: advisory failure guards, required failure exempts
# --------------------------------------------------------------------------


def test_non_required_failing_check_still_guards(kanban_home, monkeypatch):
    """The REAL live shape: red advisory ``review-gate`` + skipped/pending checks.

    This is a green-PR-awaiting-review card. Exempting it is the AC4 widening.
    """
    _stub(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [
            {"name": "review-gate", "conclusion": "FAILURE", "isRequired": False},
            {"name": "test", "conclusion": "SUCCESS", "isRequired": False},
            {"name": "e2e", "conclusion": None, "status": "IN_PROGRESS", "isRequired": False},
            {"context": "legacy", "state": None, "isRequired": False},
        ],
    })
    with kbc.connect() as conn:
        task_id = _card(conn)
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_required_failing_check_exempts(kanban_home, monkeypatch):
    """A merge-gating red check is implementer-actionable."""
    _stub(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"name": "review-gate", "conclusion": "FAILURE", "isRequired": False},
            {"name": "test", "conclusion": "FAILURE", "isRequired": True},
        ],
    })
    with kbc.connect() as conn:
        task_id = _card(conn)
        exempt: list = []
        assert kbd.check_respawn_guard(conn, task_id, exempt_out=exempt) is None
        assert exempt == [{"reason": "pr_needs_author_action", "pr_url": PR_URL}]


def test_required_statuscontext_error_exempts(kanban_home, monkeypatch):
    """StatusContext carries ``state``, not ``conclusion`` — both shapes count."""
    _stub(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [{"context": "ci/legacy", "state": "ERROR", "isRequired": True}],
    })
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, _card(conn)) is None


def test_required_check_still_running_guards(kanban_home, monkeypatch):
    """A required check that has not concluded is not a failure."""
    _stub(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [
            {"name": "test", "conclusion": None, "status": "IN_PROGRESS", "isRequired": True},
        ],
    })
    with kbc.connect() as conn:
        assert kbd.check_respawn_guard(conn, _card(conn)) == "active_pr"


# --------------------------------------------------------------------------
# Other exemption arms + fail-closed
# --------------------------------------------------------------------------


def test_merged_pr_exempts_with_pr_not_open(kanban_home, monkeypatch):
    _stub(monkeypatch, {"state": "MERGED", "reviewDecision": "APPROVED", "statusCheckRollup": []})
    with kbc.connect() as conn:
        exempt: list = []
        assert kbd.check_respawn_guard(conn, _card(conn), exempt_out=exempt) is None
        assert exempt[0]["reason"] == "pr_not_open"


def test_indeterminate_status_fails_closed(kanban_home, monkeypatch):
    """No ``gh`` / network blip must not stampede duplicate workers."""
    _stub(monkeypatch, None)
    with kbc.connect() as conn:
        exempt: list = []
        assert kbd.check_respawn_guard(conn, _card(conn), exempt_out=exempt) == "active_pr"
        assert exempt == []


def test_verdict_cached_within_ttl(kanban_home, monkeypatch):
    _stub(monkeypatch, {
        "state": "OPEN", "reviewDecision": "CHANGES_REQUESTED", "statusCheckRollup": [],
    })
    with kbc.connect() as conn:
        task_id = _card(conn)
        assert kbd.check_respawn_guard(conn, task_id) is None

        def _boom(o, r, n):
            raise AssertionError("cached verdict must be reused")

        monkeypatch.setattr(kbd, "_fetch_pr_status", _boom, raising=False)
        assert kbd.check_respawn_guard(conn, task_id) is None


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------


def test_dispatch_emits_respawn_allowed_event(kanban_home, monkeypatch):
    """The exemption path must be distinguishable in ``task_events``."""
    _stub(monkeypatch, {
        "state": "OPEN", "reviewDecision": "CHANGES_REQUESTED", "statusCheckRollup": [],
    })
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    spawned: list = []
    with kbc.connect() as conn:
        task_id = _card(conn)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()

        kbd.dispatch_once(conn, spawn_fn=lambda *a, **kw: spawned.append(a) or 4242)

    with kbc.connect() as conn:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'respawn_allowed'",
            (task_id,),
        ).fetchall()
    assert len(rows) == 1, "exactly one respawn_allowed event expected"
    payload = json.loads(rows[0]["payload"])
    assert payload["reason"] == "pr_needs_author_action"
    assert payload["pr_url"] == PR_URL
