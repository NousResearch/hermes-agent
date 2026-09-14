"""RED repro: ``active_pr`` deadlocks a card whose PR needs implementer action.

Card t_108177c6 emitted ``respawn_guarded {"reason":"active_pr"}`` every
60-90s for 4+ hours while its PR sat OPEN on a CHANGES_REQUESTED verdict.
Only the implementer can clear that verdict, and the guard was precisely
what kept the implementer from being respawned. Two full review rounds
produced zero commits.

On unpatched ``main`` :func:`hermes_cli.kanban_db_dispatch.check_respawn_guard`
never looks at PR state at all — it returns ``"active_pr"`` for any comment
carrying a PR URL inside the 24h window. This file pins both halves of the
required behaviour:

* :func:`test_changes_requested_pr_must_respawn` — the defect. RED on main.
* :func:`test_green_pr_awaiting_review_still_guards` — the negative control
  that forbids widening the guard into "always respawn" (AC4). GREEN on main
  and must STAY green after the fix.

The PR metadata source is stubbed (``raising=False``) rather than hitting
GitHub, so this file COLLECTS on a build that predates the fix: the seam the
fix introduces does not exist there yet, and a collection error would prove
nothing about behaviour. On unpatched code the stub is simply ignored and
assertion 1 goes red for the real reason.
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
def _clear_pr_caches():
    """PR verdicts are cached per-process; never leak between cases.

    ``getattr`` defaults keep this working on builds where the cache does
    not exist yet.
    """
    names = ("_PR_GUARD_CACHE", "_PR_STATE_CACHE", "_WORKFLOW_REVIEW_TRIGGER_CACHE")
    for name in names:
        getattr(kbd, name, {}).clear()
    yield
    for name in names:
        getattr(kbd, name, {}).clear()


def _card_with_open_pr(conn) -> str:
    """A ready card whose worker already commented an open PR URL."""
    task_id = kb.create_task(conn, title="ship the thing", assignee="a")
    kb.add_comment(conn, task_id, "worker", f"Opened PR: {PR_URL}")
    conn.commit()
    return task_id


def _stub_pr_status(monkeypatch, status: dict) -> None:
    """Stub every PR-metadata seam so no test ever shells out to ``gh``.

    ``raising=False``: on unpatched ``main`` these attributes do not exist.
    Creating them is harmless — the unpatched guard never calls them, which
    is exactly the defect this file pins.
    """
    monkeypatch.setattr(
        kbd, "_fetch_pr_status", lambda owner, repo, number: dict(status),
        raising=False,
    )
    monkeypatch.setattr(
        kbd, "_pr_url_is_open", lambda _url: status.get("state") == "OPEN",
        raising=False,
    )


# ---------------------------------------------------------------------------
# 1. The defect — RED on current main
# ---------------------------------------------------------------------------


def test_changes_requested_pr_must_respawn(kanban_home, monkeypatch):
    """CHANGES_REQUESTED is implementer-actionable: the guard must release.

    Guarding here is a deadlock by construction — the PR cannot progress
    without a new commit, and only a respawned implementer can push one.
    """
    _stub_pr_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "CHANGES_REQUESTED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kbc.connect() as conn:
        task_id = _card_with_open_pr(conn)

        assert kbd.check_respawn_guard(conn, task_id) is None, (
            "an OPEN PR with CHANGES_REQUESTED must not be held by active_pr: "
            "only the implementer can clear the verdict"
        )

        # The card must actually be claimable, not merely un-guarded.
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None, "card must be claimable once un-guarded"


# ---------------------------------------------------------------------------
# 2. Negative control — must stay GREEN before AND after the fix (AC4)
# ---------------------------------------------------------------------------


def test_green_pr_awaiting_review_still_guards(kanban_home, monkeypatch):
    """A green PR waiting on a reviewer is the case the guard exists for.

    Respawning here re-creates the duplicate-PR storm. This assertion is the
    fence against widening the exemption into "always respawn".
    """
    _stub_pr_status(monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"name": "test", "conclusion": "SUCCESS"},
            {"name": "build", "conclusion": "SUCCESS"},
        ],
    })
    with kbc.connect() as conn:
        task_id = _card_with_open_pr(conn)

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr", (
            "a green PR awaiting review must stay guarded"
        )
