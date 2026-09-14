"""Regression: synthesized review-handoff run attributed to the implementer (#111064).

``request_review`` captures the acting profile (the implementer) and then
rewrites ``tasks.assignee`` to the reviewer in the same UPDATE. The
zero-duration run synthesized for a never-claimed card must name the
implementer — the actor who performed the handoff — not the reviewer who
received it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    # request_review rejects reviewers that are not installed profiles (#106163).
    (home / "profiles" / "reviewer").mkdir(parents=True)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_synthesized_review_handoff_run_names_implementer_not_reviewer(
    kanban_home: Path,
) -> None:
    """The review_requested handoff row belongs to the implementer.

    Repro from #111064: an unclaimed card (assignee=builder) is handed off
    via request_review(reviewer=reviewer). The task is reassigned, but the
    synthesized zero-duration run must still record profile="builder".
    """
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Unclaimed card", assignee="builder")
        ok = kb.request_review(
            conn, task_id, summary="ready for review", reviewer="reviewer"
        )
        assert ok is True

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"

        handoff = kb.latest_run(conn, task_id)
        assert handoff is not None
        assert handoff.outcome == "review_requested"
        assert handoff.profile == "builder"


def test_synthesized_review_handoff_run_after_live_implementer_run(
    kanban_home: Path,
) -> None:
    """Same attribution when the implementer had a live run first.

    The implementer's own run keeps profile="builder"; the handoff row must
    too — otherwise per-profile tallies mix the two roles.
    """
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Claimed card", assignee="builder")
        claimed = kb.claim_task(conn, task_id, claimer="builder:1")
        assert claimed is not None

        ok = kb.request_review(
            conn,
            task_id,
            summary="done",
            reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True

        rows = conn.execute(
            "SELECT profile, outcome FROM task_runs WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
        handoffs = [r for r in rows if r["outcome"] == "review_requested"]
        assert len(handoffs) == 1
        assert handoffs[0]["profile"] == "builder"
