"""Regression tests for the review-lane self-review fix (``kanban.review_profile``).

A review row carries whatever assignee the card had when it entered the review
column; when nobody was reassigned at ``request_review`` that assignee is the
card's own implementer, so the review lane used to spawn the author to review
their own work. With ``kanban.review_profile`` set to an installed profile the
review lane spawns that profile instead — spawn-only, so the board keeps showing
the row's own assignee — and the key is ignored when it names nothing installed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def kanban_home(monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB and no ``config.yaml``."""
    test_home = tempfile.mkdtemp(prefix="kanban_review_profile_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Force-reimport so the fresh HERMES_HOME (and an empty config cache) is used.
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db

    yield kanban_db, Path(test_home)


def _write_review_profile(home: Path, profile: str) -> None:
    home.joinpath("config.yaml").write_text(
        f'kanban:\n  review_profile: "{profile}"\n', encoding="utf-8",
    )


def _review_card(kb, kbc, *, assignee: str) -> str:
    """A claimed card handed to the review column with no reviewer reassigned,
    so its assignee is still the implementer — the self-review scenario."""
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(conn, title="impl", assignee=assignee)
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary="Implementation complete", expected_run_id=run_id,
        ) is True
        row = conn.execute("SELECT status, assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "review"
        assert row["assignee"] == assignee  # nobody was reassigned
        return tid


def _spawn_recorder(seen: list):
    def _spawn(task, workspace, *args, **kwargs):
        seen.append({"assignee": task.assignee, "skills": list(task.skills or [])})
        return 4242

    return _spawn


def _db_assignee(kbc, task_id: str) -> str:
    with kbc.connect_closing() as conn:
        return conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()["assignee"]


def test_review_profile_spawns_configured_profile_not_the_row_assignee(kanban_home):
    """Invariant: with a resolvable ``kanban.review_profile`` the review run
    spawns that profile — even when the row's assignee is not an installed
    profile at all — while the task's stored assignee stays untouched."""
    kb, home = kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    _write_review_profile(home, "default")  # the default profile always exists
    task_id = _review_card(kb, kbc, assignee="implementer")  # not an installed profile
    seen: list = []

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(seen), dry_run=False)

    assert [s["assignee"] for s in seen] == ["default"]
    # The review lane's force-loaded skill survives the override.
    assert "sdlc-review" in seen[0]["skills"]
    assert any(spawned[0] == task_id and spawned[1] == "default" for spawned in res.spawned)
    # Spawn-only: the board keeps showing the row's own assignee.
    assert _db_assignee(kbc, task_id) == "implementer"


@pytest.mark.parametrize("configured", [None, "not-an-installed-profile"])
def test_unset_or_unresolvable_review_profile_keeps_row_assignee(kanban_home, configured):
    """Back-compat: unset, or a value naming no installed profile, spawns the
    row's own assignee exactly as before."""
    kb, home = kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    if configured is not None:
        _write_review_profile(home, configured)
    assert kbd.review_profile() == configured

    task_id = _review_card(kb, kbc, assignee="default")
    seen: list = []

    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(seen), dry_run=False)

    assert [s["assignee"] for s in seen] == ["default"]
    assert _db_assignee(kbc, task_id) == "default"
