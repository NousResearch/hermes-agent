"""kanban.dispatch_profiles: per-home claim allowlist for shared boards (#110995).

On a board shared across Hermes homes (one kanban.db mounted in several
containers) every home's ``profile_exists("default")`` is True, so any home's
dispatcher could claim cards assigned to ``default``. The allowlist wraps the
same predicate consumed by the spawn gate and the spawnable telemetry, so a
foreign assignee lands in ``skipped_nonspawnable`` and does not keep the
gateway wake-up loop hot.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_allowlist_without_default_skips_default_card(kanban_home, all_assignees_spawnable):
    """Fail-closed: the allowlist is set and ``default`` is not on it, so the
    card is neither spawnable (wake-up telemetry) nor claimed (spawn gate),
    even though every profile exists locally."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles:\n    - sage\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="foreign card", assignee="default")
        assert kbd.has_spawnable_ready(conn) is False
        res = kbd.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_nonspawnable == [tid]


def test_default_assignee_outside_allowlist_leaves_card_unassigned(
    kanban_home, all_assignees_spawnable,
):
    """Invariant: a ``default_assignee`` this home may not claim must not be
    persisted onto an unassigned shared-board card (no row write, no
    ``assigned`` event) — another home owns that card."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles:\n    - sage\n  default_assignee: default\n",
        encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unassigned card")
        res = kbd.dispatch_once(
            conn, dry_run=False, spawn_fn=lambda _t, _w: 4242,
            default_assignee="default",
        )
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.spawned == []
    assert res.auto_assigned_default == []
    assert task.assignee is None
    assert "assigned" not in kinds


def test_unset_allowlist_keeps_default_claimable(kanban_home, all_assignees_spawnable):
    """No key = upstream behaviour: any existing profile, ``default`` included."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="local card", assignee="default")
        assert kbd.has_spawnable_ready(conn) is True
        res = kbd.dispatch_once(conn, dry_run=True)
    assert [t for t, _a, _w in res.spawned] == [tid]
    assert res.skipped_nonspawnable == []


def test_empty_allowlist_claims_nothing_end_to_end(kanban_home, all_assignees_spawnable):
    """A set-but-empty allowlist is fail-closed end to end: no card is
    claimable and the wake-up probe reads the board as empty."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles: []\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="any card", assignee="sage")
        assert kbd.has_spawnable_ready(conn) is False
        res = kbd.dispatch_once(conn, dry_run=True)
    assert res.spawned == []
    assert res.skipped_nonspawnable == [tid]


def test_mixed_board_listed_spawns_foreign_bucketed(kanban_home, all_assignees_spawnable):
    """Both halves at once on a shared board: the listed assignee's card
    spawns while the foreign default card is bucketed, not silently ignored."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles:\n    - sage\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        foreign = kb.create_task(conn, title="foreign card", assignee="default")
        ours = kb.create_task(conn, title="our card", assignee="sage")
        res = kbd.dispatch_once(conn, dry_run=True)
    assert [t for t, _a, _w in res.spawned] == [ours]
    assert res.skipped_nonspawnable == [foreign]


def test_comma_string_allowlist_gates_like_list_form(kanban_home, all_assignees_spawnable):
    """The comma-separated string form gates the same way as the list form:
    the listed assignee spawns, a default-assigned card does not."""
    (kanban_home / "config.yaml").write_text(
        'kanban:\n  dispatch_profiles: "sage, researcher"\n', encoding="utf-8",
    )
    with kbc.connect() as conn:
        ours = kb.create_task(conn, title="local card", assignee="sage")
        foreign = kb.create_task(conn, title="foreign card", assignee="default")
        res = kbd.dispatch_once(conn, dry_run=True)
    assert [t for t, _a, _w in res.spawned] == [ours]
    assert res.skipped_nonspawnable == [foreign]
