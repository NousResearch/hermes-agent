"""Blocked cards must not stay owned by a profile that can never work them again.

A ``blocked`` card is handed to its assignee: only a worker/operator on that
profile can unblock or complete it. When that profile is deleted (tombstoned),
loses its identity marker, or never existed on this host, nobody is left to act
and the card deadlocks silently — the ``ready``-lane ``skipped_nonspawnable``
bucket never fires for it because a blocked row is not dispatchable at all.

The dispatcher therefore repairs ownership (never status) once per tick: route
the card to its creator when the creator is a live profile, else clear the
assignee and record ``route="human"``. Detection is ``profiles.profile_exists``
— the same predicate the spawn gate uses, so tombstones and identity-less ghost
shells count as dead without inventing a ``retired: true`` config contract.

Scratch board, no real host profiles.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _isolated_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _profile(home: Path, name: str) -> None:
    """Materialise a live on-disk profile (identity marker, no tombstone)."""
    d = home / "profiles" / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(f"name: {name}\n", encoding="utf-8")


def _tombstoned_profile(home: Path, name: str) -> None:
    """A profile ``hermes profile delete`` left behind: identity marker + tombstone."""
    from hermes_constants import mark_named_profile_deleted

    _profile(home, name)
    mark_named_profile_deleted(home / "profiles" / name)


def _blocked(conn, *, assignee, created_by) -> str:
    tid = kb.create_task(conn, title="card", assignee=assignee, created_by=created_by)
    assert kb.block_task(conn, tid, reason="waiting on the worker", kind="dependency")
    return tid


def _events(conn, tid: str) -> list:
    return [e for e in kb.list_events(conn, tid) if e.kind == "assignee_recovered"]


def test_deleted_assignee_is_routed_to_the_live_creator(tmp_path, monkeypatch):
    """(1) The deadlock: the block is addressed to a profile that no longer exists."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    spawns: list[str] = []
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        kbd.dispatch_once(conn, spawn_fn=lambda task, *a, **k: spawns.append(task.id))
        task = kb.get_task(conn, tid)
        events = _events(conn, tid)
    assert task.assignee == "pm"
    assert task.status == "blocked"
    assert len(events) == 1
    assert events[0].payload == {
        "assignee": "gone", "reassigned_to": "pm", "route": "created_by",
        "reason": "assignee_not_live",
    }
    assert spawns == []


def test_tombstoned_assignee_follows_the_same_route(tmp_path, monkeypatch):
    """(2) Upstream's retirement contract is the tombstone, not a config flag."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    _tombstoned_profile(home, "ghost")
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="ghost", created_by="pm")
        kbd.dispatch_once(conn, spawn_fn=lambda task, *a, **k: None)
        task = kb.get_task(conn, tid)
        events = _events(conn, tid)
    assert task.assignee == "pm"
    assert task.status == "blocked"
    assert len(events) == 1 and events[0].payload["route"] == "created_by"


def test_no_live_creator_takes_the_human_route(tmp_path, monkeypatch):
    """(3)+(4) Nothing to route to: the card is surfaced as ownerless, not faked.

    The creator is the same dead name here, so a ``created_by`` shortcut would
    "succeed" by writing a name that still cannot work the card.
    """
    home = _isolated_home(tmp_path, monkeypatch)
    _tombstoned_profile(home, "ghost")
    spawns: list[str] = []
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="ghost", created_by="ghost")
        kbd.dispatch_once(conn, spawn_fn=lambda task, *a, **k: spawns.append(task.id))
        task = kb.get_task(conn, tid)
        events = _events(conn, tid)
    assert task.assignee is None
    assert task.status == "blocked"
    assert len(events) == 1
    assert events[0].payload == {
        "assignee": "ghost", "reassigned_to": None, "route": "human",
        "reason": "assignee_not_live",
    }
    assert spawns == []


def test_recovery_never_completes_or_unblocks(tmp_path, monkeypatch):
    """(4) Ownership repair is not dispatch: status and block kind are untouched."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        before = kb.get_task(conn, tid)
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
        after = kb.get_task(conn, tid)
    assert after.status == before.status == "blocked"
    assert after.block_kind == before.block_kind
    assert after.completed_at is None


def test_three_ticks_are_one_recovery_episode(tmp_path, monkeypatch):
    """(5) The condition expires after the repair: no per-tick event spam."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
        assert len(_events(conn, tid)) == 1
        assert kb.get_task(conn, tid).assignee == "pm"


def test_dry_run_tick_repairs_nothing(tmp_path, monkeypatch):
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        kbd.dispatch_once(conn, dry_run=True, spawn_fn=lambda *a, **k: None)
        assert kb.get_task(conn, tid).assignee == "gone"
        assert _events(conn, tid) == []


def test_live_and_blank_assignees_are_untouched(tmp_path, monkeypatch):
    """(6) Only dead owners are repaired."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    with kbc.connect() as conn:
        live = _blocked(conn, assignee="pm", created_by="pm")
        blank = _blocked(conn, assignee=None, created_by="pm")
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
        assert kb.get_task(conn, live).assignee == "pm"
        assert kb.get_task(conn, blank).assignee is None
        assert _events(conn, live) == []
        assert _events(conn, blank) == []


def test_unreadable_profile_state_fails_closed(tmp_path, monkeypatch):
    """(7) An unreadable profile must never license a reassignment."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    from hermes_cli import profiles

    def _boom(name):  # pragma: no cover - raised, not returned
        raise OSError("profile home unreadable")

    monkeypatch.setattr(profiles, "profile_exists", _boom)
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
        assert kb.get_task(conn, tid).assignee == "gone"
        assert _events(conn, tid) == []


def test_missing_profile_introspection_fails_closed(tmp_path, monkeypatch):
    """(7) No ``profiles`` import on this host = no detection, so no write."""
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
        assert kb.get_task(conn, tid).assignee == "gone"
        assert _events(conn, tid) == []


def test_stale_assignment_cannot_be_overwritten(tmp_path, monkeypatch):
    """(8) The write is conditional on the observed owner (CAS), not a blind UPDATE.

    Simulates the concurrent write landing between the dispatcher's read and its
    repair: the stale repair must be dropped, with no event claiming it happened.
    """
    home = _isolated_home(tmp_path, monkeypatch)
    _profile(home, "pm")
    with kbc.connect() as conn:
        tid = _blocked(conn, assignee="gone", created_by="pm")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET assignee = 'ops' WHERE id = ?", (tid,))
        assert kbd._repair_blocked_assignee(conn, tid, "gone", "pm", route="created_by") is False
        assert kb.get_task(conn, tid).assignee == "ops"
        assert _events(conn, tid) == []
