"""Decision table for the ``active_pr`` respawn-guard exemption.

Companion to ``test_kanban_respawn_guard_changes_requested_deadlock.py``.
This file pins the *scoping* of the exemption, which round 1 of the upstream
fix got wrong: it exempted on ANY failing rollup entry, which on the live
board un-guarded 24 of 40 open PRs — 11 of them solely because the advisory
``review-gate`` check is red BY DESIGN while a PR awaits review. Only checks
that actually gate the merge (``isRequired``) may justify a respawn.

Ported from upstream PR #110657 to the installed (Aug-25) monolith layout.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest


PR_URL = "https://github.com/VibeTechnologies/AgentPod/pull/4888"


@pytest.fixture()
def kb(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_pr_exemption_test_")
    for prof in ("a", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


@pytest.fixture(autouse=True)
def _clear_cache(kb):
    for name in ("_PR_GUARD_CACHE", "_PR_STATE_CACHE"):
        getattr(kb, name, {}).clear()
    yield
    for name in ("_PR_GUARD_CACHE", "_PR_STATE_CACHE"):
        getattr(kb, name, {}).clear()


def _card(kb, conn) -> str:
    task_id = kb.create_task(conn, title="ship it", assignee="a")
    kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
    conn.commit()
    return task_id


def _stub(kb, monkeypatch, status):
    monkeypatch.setattr(
        kb, "_fetch_pr_status", lambda o, r, n: status, raising=False
    )
    monkeypatch.setattr(
        kb, "_pr_url_is_open",
        lambda _u: bool(status) and status.get("state") == "OPEN",
        raising=False,
    )


# --------------------------------------------------------------------------
# Scoping: advisory failure guards, required failure exempts
# --------------------------------------------------------------------------


def test_non_required_failing_check_still_guards(kb, monkeypatch):
    """The REAL live shape: red advisory ``review-gate`` + pending checks."""
    _stub(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [
            {"name": "review-gate", "conclusion": "FAILURE", "isRequired": False},
            {"name": "test", "conclusion": "SUCCESS", "isRequired": False},
            {"name": "e2e", "conclusion": None, "status": "IN_PROGRESS",
             "isRequired": False},
            {"context": "legacy", "state": None, "isRequired": False},
        ],
    })
    with kb.connect_closing() as conn:
        assert kb.check_respawn_guard(conn, _card(kb, conn)) == "active_pr"


def test_required_failing_check_exempts(kb, monkeypatch):
    """A merge-gating red check is implementer-actionable."""
    _stub(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"name": "review-gate", "conclusion": "FAILURE", "isRequired": False},
            {"name": "test", "conclusion": "FAILURE", "isRequired": True},
        ],
    })
    with kb.connect_closing() as conn:
        exempt: list = []
        assert kb.check_respawn_guard(
            conn, _card(kb, conn), exempt_out=exempt
        ) is None
        assert exempt == [{"reason": "pr_needs_author_action", "pr_url": PR_URL}]


def test_required_statuscontext_error_exempts(kb, monkeypatch):
    """StatusContext carries ``state``, not ``conclusion`` — both shapes count."""
    _stub(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [
            {"context": "ci/legacy", "state": "ERROR", "isRequired": True},
        ],
    })
    with kb.connect_closing() as conn:
        assert kb.check_respawn_guard(conn, _card(kb, conn)) is None


def test_required_check_still_running_guards(kb, monkeypatch):
    """A required check that has not concluded is not a failure."""
    _stub(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": None,
        "statusCheckRollup": [
            {"name": "test", "conclusion": None, "status": "IN_PROGRESS",
             "isRequired": True},
        ],
    })
    with kb.connect_closing() as conn:
        assert kb.check_respawn_guard(conn, _card(kb, conn)) == "active_pr"


# --------------------------------------------------------------------------
# Other exemption arms + fail-closed
# --------------------------------------------------------------------------


def test_merged_pr_exempts_with_pr_not_open(kb, monkeypatch):
    _stub(kb, monkeypatch, {
        "state": "MERGED", "reviewDecision": "APPROVED", "statusCheckRollup": [],
    })
    with kb.connect_closing() as conn:
        exempt: list = []
        assert kb.check_respawn_guard(
            conn, _card(kb, conn), exempt_out=exempt
        ) is None
        assert exempt[0]["reason"] == "pr_not_open"


def test_indeterminate_status_fails_closed(kb, monkeypatch):
    """No ``gh`` / network blip must not stampede duplicate workers."""
    _stub(kb, monkeypatch, None)
    with kb.connect_closing() as conn:
        exempt: list = []
        assert kb.check_respawn_guard(
            conn, _card(kb, conn), exempt_out=exempt
        ) == "active_pr"
        assert exempt == []


def test_verdict_cached_within_ttl(kb, monkeypatch):
    _stub(kb, monkeypatch, {
        "state": "OPEN", "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [],
    })
    with kb.connect_closing() as conn:
        task_id = _card(kb, conn)
        assert kb.check_respawn_guard(conn, task_id) is None

        def _boom(o, r, n):
            raise AssertionError("cached verdict must be reused")

        monkeypatch.setattr(kb, "_fetch_pr_status", _boom, raising=False)
        assert kb.check_respawn_guard(conn, task_id) is None


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------


def test_dispatch_emits_respawn_allowed_event(kb, monkeypatch):
    """The exemption path must be distinguishable in ``task_events``."""
    _stub(kb, monkeypatch, {
        "state": "OPEN", "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [],
    })
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = _card(kb, conn)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()

    with kb.connect_closing() as conn:
        kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: 4242)

    with kb.connect_closing() as conn:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'respawn_allowed'",
            (task_id,),
        ).fetchall()
    assert len(rows) == 1, "exactly one respawn_allowed event expected"
    payload = json.loads(rows[0]["payload"])
    assert payload["reason"] == "pr_needs_author_action"
    assert payload["pr_url"] == PR_URL
