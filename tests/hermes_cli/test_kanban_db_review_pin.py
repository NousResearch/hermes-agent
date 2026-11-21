"""Review routing: the configured reviewer owns a first review, not the author."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_review as kbr
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# A first review belongs to the configured reviewer, never to the author
# ---------------------------------------------------------------------------

def test_default_reviewer_profile_reads_the_kanban_key(monkeypatch):
    """``kanban.reviewer_profile`` is the code-side default for a FIRST review.

    The key was documented and set in config.yaml while no code read it, so a
    handoff that carried no ``reviewer`` kept the author as the assignee, and the
    review lane spawns the row's own assignee.
    """
    import hermes_cli.config as config_mod
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(
        config_mod, "load_config_readonly",
        lambda: {"kanban": {"reviewer_profile": "reviewer"}})
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "reviewer")
    assert kbr.default_reviewer_profile() == "reviewer"

    # A name with no live profile would park the card on an assignee the
    # dispatcher can never spawn, so it degrades to the previous behaviour.
    monkeypatch.setattr(
        config_mod, "load_config_readonly",
        lambda: {"kanban": {"reviewer_profile": "ghost"}})
    assert kbr.default_reviewer_profile() is None

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {"kanban": {}})
    assert kbr.default_reviewer_profile() is None


def test_first_review_request_lands_on_the_configured_reviewer(kanban_home, monkeypatch):
    """Regression, measured 2026-11-05: 19 self-approvals in 60 handoffs / 14 days.

    ``kanban_request_review`` carries ``reviewer`` only when the model happens to
    supply one, and ``_prior_reviewer`` answers None on a first review. The card
    then kept the profile that did the work and the review lane spawned it, so
    the author graded itself while the reviewer lane sat idle.
    """
    monkeypatch.setattr(kb, "default_reviewer_profile", lambda: "reviewer")
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="self review trap", assignee="implementer")
        kbw.set_workspace_path(conn, t, kbw.resolve_workspace(kb.get_task(conn, t)))
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        assert kb.request_review(conn, t, summary="done", expected_run_id=run_id)
        task = kb.get_task(conn, t)
        assert task.status == "review"
        assert task.assignee == "reviewer", "the author must not own its own review"
        handoff = [e for e in kb.list_events(conn, t) if e.kind == "review_requested"][-1]
        assert handoff.payload["reviewer"] == "reviewer"
        assert handoff.payload["implementer"] == "implementer"

        # An explicit reviewer still wins over the configured default.
        t2 = kb.create_task(conn, title="explicit reviewer", assignee="implementer")
        kbw.set_workspace_path(conn, t2, kbw.resolve_workspace(kb.get_task(conn, t2)))
        kb.claim_task(conn, t2)
        assert kb.request_review(
            conn, t2, summary="done", reviewer="reviewer-2",
            expected_run_id=kb.get_task(conn, t2).current_run_id)
        assert kb.get_task(conn, t2).assignee == "reviewer-2"

        # An explicit reviewer naming the author is the same defect by another
        # route: it falls back to the configured reviewer, not the label.
        t3 = kb.create_task(conn, title="self-named reviewer", assignee="implementer")
        kbw.set_workspace_path(conn, t3, kbw.resolve_workspace(kb.get_task(conn, t3)))
        kb.claim_task(conn, t3)
        assert kb.request_review(
            conn, t3, summary="done", reviewer="implementer",
            expected_run_id=kb.get_task(conn, t3).current_run_id)
        assert kb.get_task(conn, t3).assignee == "reviewer"
