"""Per-board kanban pause — the "Pause" control behind #141451.

Contract under test:
* ``pause``/``resume`` write and remove a per-board sentinel, survive a corrupt
  body (fail safe) and never leak to another board.
* The dispatcher spawns nothing on a paused board and says so
  (``DispatchResult.paused``) instead of looking stuck; resuming spawns again.
* The plugin API exposes the state on the board payload, the sidebar endpoint
  and POST /pause / /resume.
* The CLI round-trips through the same entry point the gateway slash command uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pause as pause


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _fake_spawn(*_args, **_kwargs):
    return 12345


def _ready_task(board: str | None = None, assignee: str = "default") -> str:
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title="t", assignee=assignee, board=board)


# ---------------------------------------------------------------------------
# Sentinel round-trip
# ---------------------------------------------------------------------------


def test_pause_roundtrip_and_state(fresh_home):
    assert pause.is_paused() is False
    assert pause.get_state() is None

    state = pause.pause(reason="switching provider")
    assert state["board"] == "default"
    assert state["reason"] == "switching provider"
    assert state["paused_at"]
    assert pause.is_paused() is True
    assert pause.pause_path("default").exists()

    # Idempotent: re-pausing refreshes, never raises.
    again = pause.pause(reason="still fixing")
    assert again["reason"] == "still fixing"
    state_after = pause.get_state()
    assert state_after is not None and state_after["reason"] == "still fixing"

    assert pause.resume() is True
    assert pause.is_paused() is False
    assert pause.resume() is False


def test_corrupt_sentinel_still_pauses(fresh_home):
    path = pause.pause_path("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    # Fail safe: a hand-written/garbled sentinel must still hold the board.
    assert pause.is_paused() is True
    assert pause.get_state() == {"board": "default", "reason": None, "paused_at": None}


def test_pause_is_per_board(fresh_home):
    kb.create_board("beta", name="Beta")
    pause.pause("default", reason="hold this one")
    assert pause.is_paused("default") is True
    assert pause.is_paused("beta") is False
    assert pause.paused_boards() == ["default"]
    assert not pause.pause_path("beta").exists()


def test_pause_unknown_board_raises(fresh_home):
    with pytest.raises(ValueError):
        pause.pause("nope")


# ---------------------------------------------------------------------------
# Dispatcher gating
# ---------------------------------------------------------------------------


def test_dispatch_spawns_nothing_while_paused(fresh_home):
    tid = _ready_task()
    pause.pause(reason="provider swap")

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
    assert res.paused is True
    assert res.spawned == []
    assert res.reclaimed == 0 and res.promoted == 0

    # The card is untouched: still ready, never claimed.
    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "ready"

    # Resuming lets the very next tick spawn it.
    pause.resume()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
    assert res.paused is False
    assert [s[0] for s in res.spawned] == [tid]


def test_paused_board_does_not_block_other_boards(fresh_home):
    kb.create_board("beta", name="Beta")
    tid_default = _ready_task("default")
    tid_beta = _ready_task("beta")
    pause.pause("default", reason="hold")

    with kbc.connect_closing(board="default") as conn:
        assert kbd.dispatch_once(conn, spawn_fn=_fake_spawn, board="default").paused is True
    with kbc.connect_closing(board="beta") as conn:
        beta_res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, board="beta")
    assert beta_res.paused is False
    assert [s[0] for s in beta_res.spawned] == [tid_beta]

    with kbc.connect_closing(board="default") as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid_default,)).fetchone()
    assert row["status"] == "ready"


def test_dispatch_reads_the_sentinel_of_the_board_it_dispatches(fresh_home, monkeypatch):
    """`--board beta dispatch` passes board=None: the guard must follow the connection.

    `board_dir(None)` answers for the *default* board while `connect()` resolves None to the
    *current* board, so a tick opened on beta used to read `boards/default/PAUSED` — pausing
    beta stopped nothing, and a paused default froze every board the CLI ticked.
    """
    kb.create_board("beta", name="Beta")
    ready_beta = _ready_task("beta")
    pause.pause("beta", reason="swapping provider")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")

    with kbc.connect_closing() as conn:
        held = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
    assert held.paused is True, "a paused current board must hold its own tick"
    assert held.spawned == []

    # The mirror image: beta resumed, default paused — beta must dispatch.
    pause.resume("beta")
    pause.pause("default", reason="hold")
    with kbc.connect_closing() as conn:
        beta_res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
    assert beta_res.paused is False, "the default board's sentinel must not freeze beta"
    assert [s[0] for s in beta_res.spawned] == [ready_beta]


def test_dispatch_result_reports_pause_in_telemetry(fresh_home):
    pause.pause()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
    assert "paused=1" in kbd.describe_suppression([res])


# ---------------------------------------------------------------------------
# Plugin API (desktop + web dashboard share this router)
# ---------------------------------------------------------------------------


def _board_payload(board: str = "default") -> dict:
    from plugins.kanban.dashboard import plugin_api

    return plugin_api.get_board(
        tenant=None, include_archived=False, board=board,
        workflow_template_id=None, current_step_key=None,
    )


def test_board_payload_and_endpoints_expose_pause(fresh_home):
    from plugins.kanban.dashboard import plugin_api

    assert _board_payload()["pause"] is None

    plugin_api.pause_board(plugin_api.PauseBody(reason="reloading model"), board="default")
    assert plugin_api.get_pause(board="default")["pause"]["reason"] == "reloading model"

    board = _board_payload()
    assert board["pause"]["board"] == "default"
    assert board["pause"]["reason"] == "reloading model"

    assert plugin_api.resume_board(board="default")["pause"] is None
    assert plugin_api.get_pause(board="default")["pause"] is None


def test_pause_endpoint_404s_unknown_board(fresh_home):
    from fastapi import HTTPException
    from plugins.kanban.dashboard import plugin_api

    with pytest.raises(HTTPException) as exc:
        plugin_api.pause_board(None, board="ghost")
    assert exc.value.status_code == 404


def test_paused_and_boards_lists_flag_each_held_board(fresh_home):
    """Pause is per board: the list surfaces must name every held board, or a
    single global "paused" chip would lie about the boards that still run."""
    from plugins.kanban.dashboard import plugin_api

    kb.create_board("sumaris")
    assert plugin_api.list_paused_boards() == {"paused": [], "count": 0}

    pause.pause("default", reason="provider swap")
    pause.pause("sumaris")

    listing = plugin_api.list_paused_boards()
    assert listing["count"] == 2
    assert {b["board"] for b in listing["paused"]} == {"default", "sumaris"}
    assert {b["name"].lower() for b in listing["paused"]} == {"default", "sumaris"}

    flags = {b["slug"]: bool(b["pause"]) for b in plugin_api.list_boards()["boards"]}
    assert flags == {"default": True, "sumaris": True}

    pause.resume("default")
    flags = {b["slug"]: bool(b["pause"]) for b in plugin_api.list_boards()["boards"]}
    assert flags == {"default": False, "sumaris": True}
    assert plugin_api.list_paused_boards()["count"] == 1


# ---------------------------------------------------------------------------
# CLI / slash surface
# ---------------------------------------------------------------------------


def test_cli_pause_resume_roundtrip(fresh_home):
    from hermes_cli import kanban as kc

    out = json.loads(kc.run_slash("pause --reason 'reloading model' --json"))
    assert out["board"] == "default"
    assert out["reason"] == "reloading model"

    out = json.loads(kc.run_slash("resume --json"))
    assert out == {"board": "default", "resumed": True}
    assert pause.is_paused() is False


# ---------------------------------------------------------------------------
# A bare call means "the board I am looking at"
#
# `board_dir(None)` answers for `default`, `connect(None)` opens the *current* board. Every
# pause entry point used to take the raw argument, so on a switched board the pause landed on
# `default`: the CLI held the wrong board, and a dashboard POST without `?board` paused one
# board while its own response named another.
# ---------------------------------------------------------------------------


def test_bare_pause_holds_the_current_board(fresh_home):
    kb.create_board("beta", name="Beta")
    kb.set_current_board("beta")

    state = pause.pause(reason="racing a provider switch")
    assert state["board"] == "beta"
    assert pause.pause_path("beta").exists()
    assert not pause.pause_path("default").exists()

    # The bare reads agree with the bare write.
    assert pause.is_paused() is True
    assert pause.get_state()["board"] == "beta"


def test_bare_resume_lifts_only_the_current_board(fresh_home):
    kb.create_board("beta", name="Beta")
    pause.pause("default", reason="operator hold")
    kb.set_current_board("beta")
    pause.pause("beta")

    assert pause.resume() is True
    assert not pause.pause_path("beta").exists()
    assert pause.pause_path("default").exists(), "resuming beta must not lift default's hold"


def test_api_pause_without_a_board_holds_the_current_board(fresh_home):
    from plugins.kanban.dashboard import plugin_api

    kb.create_board("beta", name="Beta")
    kb.set_current_board("beta")

    # `board=None` is what FastAPI hands the endpoint when the query param is omitted.
    out = plugin_api.pause_board(None, board=None)
    assert out["board"] == "beta"
    assert pause.pause_path("beta").exists()
    assert not pause.pause_path("default").exists()
    assert plugin_api.get_pause(board=None)["pause"]["board"] == "beta"

    assert plugin_api.resume_board(board=None)["pause"] is None
    assert not pause.pause_path("beta").exists()


def test_board_of_connection_maps_a_nested_db_to_its_board(fresh_home):
    """Workspaces keep their DBs below the board dir; the slug is the first segment."""
    import sqlite3

    kb.create_board("beta", name="Beta")
    nested = kb.board_dir("beta") / "workspaces" / "one"
    nested.mkdir(parents=True)
    conn = sqlite3.connect(nested / "kanban.db")
    try:
        assert pause.board_of_connection(conn) == "beta"
    finally:
        conn.close()
