"""RED repro: ``active_pr`` deadlocks a card whose PR needs implementer action.

Card t_108177c6 emitted ``respawn_guarded {"reason":"active_pr"}`` every
60-90s for 4+ hours while its PR sat OPEN on a CHANGES_REQUESTED verdict.
Only the implementer can clear that verdict, and the guard was precisely
what kept the implementer from being respawned.

Port of the upstream PR #110657 tests to the installed (Aug-25) tree, where
the dispatcher still lives in the ``hermes_cli.kanban_db`` monolith.

* :func:`test_changes_requested_pr_must_respawn` — the defect. RED before fix.
* :func:`test_green_pr_awaiting_review_still_guards` — negative control that
  forbids widening the guard into "always respawn". GREEN before AND after.

The PR metadata seam is stubbed (``raising=False``) so the file COLLECTS on
an unpatched tree and reports behaviour, not a collection error.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/4774"


@pytest.fixture()
def kb(monkeypatch):
    """Fresh HERMES_HOME + kanban DB, with an 'a' profile that can spawn."""
    test_home = tempfile.mkdtemp(prefix="kanban_respawn_guard_test_")
    for prof in ("a", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # A dispatched worker inherits HERMES_KANBAN_* pins pointing at the REAL
    # board. pytest's conftest strips them, but this fixture is also reached
    # from ad-hoc `python -` reviews; strip here too or `create_task` below
    # writes "loop"/"ship the thing" cards onto the production board
    # (happened 2026-09-14, 10 junk cards).
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT"):
        monkeypatch.delenv(var, raising=False)
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
def _clear_pr_caches(kb):
    names = ("_PR_GUARD_CACHE", "_PR_STATE_CACHE")
    for name in names:
        getattr(kb, name, {}).clear()
    yield
    for name in names:
        getattr(kb, name, {}).clear()


def _card_with_open_pr(kb, conn) -> str:
    task_id = kb.create_task(conn, title="ship the thing", assignee="a")
    kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
    conn.commit()
    return task_id


def _stub_pr_status(kb, monkeypatch, status) -> None:
    """Stub every PR-metadata seam so no test shells out to ``gh``."""
    monkeypatch.setattr(
        kb, "_fetch_pr_status",
        lambda owner, repo, number: dict(status) if status else None,
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_pr_url_is_open",
        lambda _url: bool(status) and status.get("state") == "OPEN",
        raising=False,
    )


# ---------------------------------------------------------------------------
# 1. The defect — RED before the fix
# ---------------------------------------------------------------------------


def test_changes_requested_pr_must_respawn(kb, monkeypatch):
    """CHANGES_REQUESTED is implementer-actionable: the guard must release."""
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kb.connect_closing() as conn:
        task_id = _card_with_open_pr(kb, conn)

        assert kb.check_respawn_guard(conn, task_id) is None, (
            "an OPEN PR with CHANGES_REQUESTED must not be held by active_pr: "
            "only the implementer can clear the verdict"
        )

        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None, "card must be claimable once un-guarded"


# ---------------------------------------------------------------------------
# 2. Negative control — must stay GREEN before AND after the fix
# ---------------------------------------------------------------------------


def test_green_pr_awaiting_review_still_guards(kb, monkeypatch):
    """A green PR waiting on a reviewer is the case the guard exists for."""
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"name": "test", "conclusion": "SUCCESS"},
            {"name": "build", "conclusion": "SUCCESS"},
        ],
    })
    with kb.connect_closing() as conn:
        task_id = _card_with_open_pr(kb, conn)

        assert kb.check_respawn_guard(conn, task_id) == "active_pr", (
            "a green PR awaiting review must stay guarded"
        )
