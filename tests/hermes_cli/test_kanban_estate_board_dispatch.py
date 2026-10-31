"""A rehearsal/estate board must NOT be in the dispatcher's set (card t_17c9c847).

WHY (measured 2026-09-29 on the live estate board ``failure-acceptance-t4ffa46e3``)
-----------------------------------------------------------------------------------
``DEPLOY_TRAIN_DRY_RUN=1`` fails closed on the FILER side (card t_3a0f6437) — no
card is written. But the *redirect* those knobs exist for
(``DEPLOY_TRAIN_FAILURE_DB`` / ``DEPLOY_TRAIN_SEV1_BOARD``) was treated as
isolation, and it is not: the fleet dispatcher serves EVERY non-archived board
("each board has its own DB, workspaces directory, and dispatcher loop",
``hermes kanban boards --help``). Measured: ``~/.hermes/logs/gateway.log``
2026-09-29 18:12:16 ``kanban dispatcher [failure-acceptance-t4ffa46e3]:
spawned=1`` — a real ``default`` lane run dispatched at the reserved-tranche
priority 999000 to chase a failure that never happened, while the board held 13
ready phantom cards.

The fix is a BOARD-LEVEL admission flag, default OFF for every existing board:

* ``board.json`` gains ``"dispatch": false`` — set at creation
  (``hermes kanban boards create <slug> --no-dispatch``) or later
  (``hermes kanban boards set-dispatch <slug> off``), so an estate is safe BY
  CONSTRUCTION the moment it exists;
* the dispatcher enumerates through ``kanban_db.list_dispatch_boards()``, which
  drops it, AND ``dispatch_once`` refuses it before the lock/reclaim/claim, so a
  caller that bypasses enumeration (a CLI ``--board <estate> dispatch``) still
  cannot spawn a card from it.

Second measured route closed here: ARCHIVING an estate is not durable while a
worker's env pins its store path — the pinned worker's next ``connect(create=True)``
mints an empty ``kanban.db`` with no ``board.json``, which read as
``archived=False`` and re-entered the dispatch set. A directory with no
``board.json`` whose slug already has an ``_archived`` copy is that mint, and is
never admitted (``kanban_db.board_is_archived_stub``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_boards as kb_boards
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


ESTATE = "estate-probe"
LIVE = "live-probe"


def _spy():
    calls: list = []

    def spawn(task, workspace_path, board=None):
        calls.append((board, getattr(task, "id", task)))
        return 4242

    return calls, spawn


def _seed_boards():
    """One estate-flagged board and one normal board, each with the same ready card."""
    kb.create_board(ESTATE, dispatch=False)
    kb.create_board(LIVE)
    with kbc.connect(board=ESTATE) as conn:
        estate_tid = kb.create_task(conn, title="phantom SEV1", assignee="default")
    with kbc.connect(board=LIVE) as conn:
        live_tid = kb.create_task(conn, title="real work", assignee="default")
    return estate_tid, live_tid


# --- DoD 1: a ready card on an estate board spawns NOTHING, with a negative control ----------


def test_estate_board_ready_card_spawns_nothing_and_normal_board_spawns(
    kanban_home, all_assignees_spawnable,
):
    """The estate card never reaches the spawn path; the same card shape on a normal board does."""
    estate_tid, live_tid = _seed_boards()
    calls, spawn = _spy()

    with kbc.connect(board=ESTATE) as conn:
        estate_res = kbd.dispatch_once(conn, board=ESTATE, spawn_fn=spawn)
    assert estate_res.skipped_board_disabled is True
    assert estate_res.spawned == []
    assert calls == [], "the spawn path must never be reached on an estate board"

    calls.clear()
    with kbc.connect(board=LIVE) as conn:
        live_res = kbd.dispatch_once(conn, board=LIVE, spawn_fn=spawn)
    assert live_res.skipped_board_disabled is False
    assert [t for t, _a, _w in live_res.spawned] == [live_tid]
    assert [tid for _b, tid in calls] == [live_tid], (
        "negative control: the identical card shape on a normal board still spawns")
    assert estate_tid not in [t for t, _a, _w in live_res.spawned]


def test_estate_board_is_refused_even_when_the_tick_names_it_directly(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """``hermes kanban --board <estate> dispatch`` bypasses enumeration — the guard is in dispatch_once."""
    _seed_boards()
    calls, spawn = _spy()
    monkeypatch.setenv("HERMES_KANBAN_BOARD", ESTATE)
    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=spawn)  # no board kwarg: the pin decides
    assert res.skipped_board_disabled is True
    assert res.spawned == [] and calls == []


# --- DoD 3: board flag + dispatch enumeration (the measurement, carried by a test) -----------


def test_estate_flag_reaches_the_dispatchers_own_enumeration(kanban_home, all_assignees_spawnable):
    """``list_dispatch_boards`` drops the estate; ``list_boards`` still SHOWS it."""
    _seed_boards()
    all_slugs = [b["slug"] for b in kb.list_boards(include_archived=False)]
    assert ESTATE in all_slugs and LIVE in all_slugs, (
        "an estate board stays VISIBLE on the board list — it is simply never served")
    dispatch_slugs = [b["slug"] for b in kb.list_dispatch_boards()]
    assert LIVE in dispatch_slugs
    assert ESTATE not in dispatch_slugs
    assert kb.board_dispatch_enabled(LIVE) is True
    assert kb.board_dispatch_enabled(ESTATE) is False


def test_gateway_dispatcher_never_visits_an_estate_board(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """End of the spawn path: the per-board tick entry is never invoked for the estate."""
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher

    _seed_boards()
    settings = _DispatcherSettings(
        interval=60.0, max_spawn=None, max_in_progress=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT, stale_timeout_seconds=0,
        reconcile_orphans=True, default_assignee=None, max_in_progress_per_profile=None,
    )
    visited: list = []

    def spy_dispatch(conn, *, board=None, **kwargs):
        visited.append(board)
        return kbd.DispatchResult(spawned=[("t_probe", "default", "ws")])

    monkeypatch.setattr(kbd, "dispatch_once", spy_dispatch)
    _KanbanDispatcher(kb, settings).tick_once()

    assert LIVE in visited
    assert ESTATE not in visited


# --- evidence leg 2: the archive is not durable while a pinned worker holds the path ----------


def test_a_resurrected_archived_slug_is_not_readmitted(kanban_home):
    """A minted empty dir at an archived slug's path is NOT a board (no board.json)."""
    kb.create_board("ghost-probe")
    with kbc.connect(board="ghost-probe") as conn:
        kb.create_task(conn, title="phantom", assignee="default")

    res = kb.remove_board("ghost-probe", archive=True)
    assert res["action"] == "archived"

    # The measured mint: a live worker's env pins the store path, so its next
    # connect() recreates the directory — with a store but no metadata.
    minted = kb.board_dir("ghost-probe")
    minted.mkdir(parents=True, exist_ok=True)
    (minted / "kanban.db").write_bytes(b"")

    assert kb.board_is_archived_stub("ghost-probe") is True
    assert "ghost-probe" not in [b["slug"] for b in kb.list_boards()]
    assert "ghost-probe" not in [b["slug"] for b in kb.list_dispatch_boards()]
    assert kb.board_dispatch_enabled("ghost-probe") is False

    # An EXPLICIT re-create is the sanctioned re-admission (it writes board.json).
    kb.create_board("ghost-probe")
    assert "ghost-probe" in [b["slug"] for b in kb.list_dispatch_boards()]


# --- the flag itself: default-on, round-trips, and is not materialized by unrelated writes ----


def test_dispatch_flag_defaults_on_and_round_trips(kanban_home):
    kb.create_board(LIVE)
    assert kb.board_dispatch_enabled(LIVE) is True
    meta = kb.write_board_metadata(LIVE, dispatch=False)
    assert meta["dispatch"] is False
    assert kb.board_dispatch_enabled(LIVE) is False
    assert kb.write_board_metadata(LIVE, dispatch=True)["dispatch"] is True
    assert kb.board_dispatch_enabled(LIVE) is True


def test_unrelated_write_does_not_materialize_the_default(kanban_home):
    """A board that never touched the key keeps its ``board.json`` free of it."""
    kb.create_board(LIVE)
    assert "dispatch" not in (kb.board_dir(LIVE) / "board.json").read_text()
    kb.write_board_metadata(LIVE, name="Live Probe")
    assert "dispatch" not in (kb.board_dir(LIVE) / "board.json").read_text()


def test_create_board_no_dispatch_writes_the_flag(kanban_home):
    kb.create_board(ESTATE, dispatch=False)
    assert kb.read_board_metadata(ESTATE)["dispatch"] is False
    assert kb.board_dispatch_enabled(ESTATE) is False


def test_set_dispatch_verb_admits_and_excludes(kanban_home, capsys):
    """The CLI verb the estate teardown (or a human) uses in both directions."""
    kb.create_board(ESTATE)
    off = argparse.Namespace(slug=ESTATE, state="off", json=False)
    assert kb_boards._cmd_boards_set_dispatch(off) == 0
    assert kb.board_dispatch_enabled(ESTATE) is False
    on = argparse.Namespace(slug=ESTATE, state="on", json=False)
    assert kb_boards._cmd_boards_set_dispatch(on) == 0
    assert kb.board_dispatch_enabled(ESTATE) is True
    bad = argparse.Namespace(slug=ESTATE, state="maybe", json=False)
    assert kb_boards._cmd_boards_set_dispatch(bad) == 2
