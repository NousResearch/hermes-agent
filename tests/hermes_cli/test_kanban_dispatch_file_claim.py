"""Dispatcher file-claim guard: a colliding card queues instead of clobbering.

Card A declares it writes ``shared.txt``; card B declares the same path. On the
first dispatch tick A spawns while B lands in the new ``skipped_file_claimed``
bucket and is NOT spawned. After A completes, a second tick releases B and
spawns it - so a colliding card can never clobber a sibling's in-flight work.

This is the forced two-card collision from the acceptance command:
``python -m pytest tests/hermes_cli/test_kanban_dispatch_file_claim.py -q``.
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


def test_colliding_card_queues_then_spawns_after_holder_completes(
    kanban_home, all_assignees_spawnable,
):
    """The forced collision: B waits (never spawned) while A runs, then spawns."""
    shared = "repos/hot-repo/src/main.py"
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="card A", assignee="alice", declared_files=[shared])
        b = kb.create_task(conn, title="card B", assignee="bob", declared_files=[shared])

        spawned: list[str] = []
        first = kbd.dispatch_once(conn, spawn_fn=lambda t, _w: spawned.append(t.id) or 4242)

        # (1) A spawns; B is bucketed as file-claimed and NOT spawned.
        assert [t for t, _a, _w in first.spawned] == [a]
        assert first.skipped_file_claimed == [b]
        assert spawned == [a]
        # B is still queued (ready), not claimed/running.
        assert kb.get_task(conn, b).status == "ready"

        # A completes -> its file claim is released.
        assert kb.complete_task(conn, a, result="done", summary="done") is True

        # (2) A second tick now spawns B.
        second = kbd.dispatch_once(conn, spawn_fn=lambda t, _w: spawned.append(t.id) or 4242)
        assert [t for t, _a, _w in second.spawned] == [b]
        assert second.skipped_file_claimed == []
        assert spawned == [a, b]


def test_non_overlapping_claims_do_not_block(kanban_home, all_assignees_spawnable):
    """Cards declaring different files spawn together (no false deferral)."""
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="card A", assignee="alice", declared_files=["a.txt"])
        b = kb.create_task(conn, title="card B", assignee="bob", declared_files=["b.txt"])
        res = kbd.dispatch_once(conn, spawn_fn=lambda t, _w: 4242)
        assert sorted(t for t, _a, _w in res.spawned) == sorted([a, b])
        assert res.skipped_file_claimed == []


def test_card_without_declared_files_is_unaffected(kanban_home, all_assignees_spawnable):
    """A card that declares nothing never trips the guard (the common case)."""
    with kbc.connect() as conn:
        a = kb.create_task(conn, title="card A", assignee="alice", declared_files=["a.txt"])
        b = kb.create_task(conn, title="card B", assignee="bob")
        res = kbd.dispatch_once(conn, spawn_fn=lambda t, _w: 4242)
        assert sorted(t for t, _a, _w in res.spawned) == sorted([a, b])
        assert res.skipped_file_claimed == []
