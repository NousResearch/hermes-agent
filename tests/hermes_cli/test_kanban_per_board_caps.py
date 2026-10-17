"""Per-board dispatch caps via ``board.max_spawn`` (FR #118116, t_8a2a1ebc).

``kanban.max_spawn`` and ``kanban.max_in_progress_per_profile`` can now be
pinned per board with an optional ``board.json`` field. Precedence is
``CLI --max`` > ``board.json`` > global ``config.yaml``. A board.json that
omits the field keeps the global value (backward compatible); the global
memory-derived ``max_in_progress`` default is untouched.

Semantics reference: ``_tick_spawn_budget`` (kanban_db_dispatch.py) —
``max_spawn`` is a live per-board concurrency cap, ``max_in_progress`` a
host-wide cap. This module only changes the *resolution* before the call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_ops
from hermes_cli import config as hermes_config


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same shape as the
    host-cap suite) so every board.json path lands under ``tmp``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# 1. read_board_metadata carries the new optional cap fields (Schema)
# ---------------------------------------------------------------------------


def test_read_board_metadata_defaults_caps_to_none(kanban_home):
    """``board.json`` without cap fields → both caps read back as ``None``
    (meaning 'use the global config'), and ``db_path`` is still derived."""
    meta = kb.read_board_metadata(None)
    assert meta["max_spawn"] is None
    assert meta["max_in_progress_per_profile"] is None
    assert "db_path" in meta


def test_write_board_metadata_pins_caps(kanban_home):
    """Positive caps persist into ``board.json`` and read back."""
    kb.write_board_metadata(None, max_spawn=2, max_in_progress_per_profile=3)
    raw = json.loads(kb.board_metadata_path(None).read_text(encoding="utf-8"))
    assert raw["max_spawn"] == 2
    assert raw["max_in_progress_per_profile"] == 3
    # read back through the merged accessor
    meta = kb.read_board_metadata(None)
    assert meta["max_spawn"] == 2
    assert meta["max_in_progress_per_profile"] == 3


def test_write_board_metadata_omits_unset_caps(kanban_home):
    """Unset (``None``) caps are NOT persisted — board.json stays clean of nulls."""
    kb.write_board_metadata(None, name="Named")
    raw = json.loads(kb.board_metadata_path(None).read_text(encoding="utf-8"))
    assert "max_spawn" not in raw
    assert "max_in_progress_per_profile" not in raw
    assert raw["name"] == "Named"


def test_write_board_metadata_clears_cap_with_zero(kanban_home):
    """``0`` clears an existing pin back to the global value (drops the key)."""
    kb.write_board_metadata(None, max_spawn=2)
    raw_before = json.loads(kb.board_metadata_path(None).read_text(encoding="utf-8"))
    assert raw_before["max_spawn"] == 2
    kb.write_board_metadata(None, max_spawn=0)
    raw_after = json.loads(kb.board_metadata_path(None).read_text(encoding="utf-8"))
    assert "max_spawn" not in raw_after
    assert kb.read_board_metadata(None)["max_spawn"] is None


# ---------------------------------------------------------------------------
# 2. resolve_board_caps precedence (CLI > board > global)
# ---------------------------------------------------------------------------


def _set_cap(cfgdict, *, board=None, spawn=None, profile=None, cli=None):
    """Build a board.json + config and return (spawn, profile) resolved."""
    if spawn is not None or profile is not None:
        kb.write_board_metadata(
            board,
            max_spawn=spawn,
            max_in_progress_per_profile=profile,
        )
    return kanban_ops.resolve_board_caps(board, cli_max=cli)


def test_resolve_board_caps_board_wins_over_global(kanban_home, monkeypatch):
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **k: {"kanban": {"max_spawn": 8, "max_in_progress_per_profile": 9}},
    )
    spawn, profile = _set_cap(None, board=None, spawn=2, profile=3)
    assert spawn == 2      # board pinned
    assert profile == 3


def test_resolve_board_caps_falls_back_to_global(kanban_home, monkeypatch):
    """"board.json`` without the field → global value wins (AC-2: unchanged)."""
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **k: {"kanban": {"max_spawn": 8, "max_in_progress_per_profile": 9}},
    )
    spawn, profile = kanban_ops.resolve_board_caps(None, cli_max=None)
    assert spawn == 8
    assert profile == 9


def test_resolve_board_caps_all_unset_is_none(kanban_home, monkeypatch):
    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **k: {})
    spawn, profile = kanban_ops.resolve_board_caps(None, cli_max=None)
    assert spawn is None
    assert profile is None


def test_resolve_board_caps_cli_beats_board(kanban_home, monkeypatch):
    """"CLI ``--max`` is the most explicit signal → wins over board.json."""
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **k: {"kanban": {"max_spawn": 8}},
    )
    kb.write_board_metadata(None, max_spawn=2)
    spawn, profile = kanban_ops.resolve_board_caps(None, cli_max=5)
    assert spawn == 5
    assert profile is None


def test_resolve_board_caps_invalid_board_value_falls_through(kanban_home, monkeypatch):
    """A non-int board.json cap (hand-edited) is ignored → global wins."""
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **k: {"kanban": {"max_spawn": 7}},
    )
    p = kb.board_metadata_path(None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"slug": "default", "max_spawn": "lots"}), encoding="utf-8")
    spawn, _ = kanban_ops.resolve_board_caps(None, cli_max=None)
    assert spawn == 7


def test_resolve_board_caps_env_pin_routes_to_active_board(kanban_home, monkeypatch):
    """A ``HERMES_KANBAN_BOARD`` pin (the dispatch connect path) routes the cap
    lookup to that board — not the hardcoded default (spawn env consistency)."""
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **k: {"kanban": {"max_spawn": 8}},
    )
    kb.create_board("tight")
    kb.write_board_metadata("tight", max_spawn=3)
    # default board has no cap → global value would be 8 if routing is wrong.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "tight")
    spawn, _ = kanban_ops.resolve_board_caps(None, cli_max=None)
    assert spawn == 3


# ---------------------------------------------------------------------------
# 3. Gateway per-board resolution (_board_dispatch_caps)
# ---------------------------------------------------------------------------


def test_gateway_board_dispatch_caps_override(kanban_home, monkeypatch):
    """board.json max_spawn:2 → that board's tick gets 2; a board without the
    field keeps the global (boot) value."""
    from types import SimpleNamespace
    from gateway import kanban_watchers_dispatcher as kwd

    settings = kwd._DispatcherSettings(
        interval=60.0, max_spawn=9, max_in_progress=4,
        failure_limit=2, stale_timeout_seconds=0, reconcile_orphans=True,
        default_assignee=None, max_in_progress_per_profile=11,
    )
    dp = kwd._KanbanDispatcher(kb, settings)

    # tight board pins its own caps; loose board leaves them to global.
    kb.write_board_metadata("tight", max_spawn=2, max_in_progress_per_profile=1)
    kb.create_board("loose")

    tight_caps, tight_override = dp._board_dispatch_caps("tight")
    loose_caps, loose_override = dp._board_dispatch_caps("loose")

    assert tight_caps["max_spawn"] == 2
    assert tight_caps["max_in_progress_per_profile"] == 1
    assert tight_override is True

    # Loose board keeps the global (boot) values untouched.
    assert loose_caps["max_spawn"] == 9
    assert loose_caps["max_in_progress_per_profile"] == 11
    assert loose_override is False


def test_gateway_tick_passes_board_cap_to_dispatch_once(kanban_home, monkeypatch):
    """End-to-end: tick_once_for_board hands the board-local cap to dispatch_once,
    and a second board with no field uses the global cap."""
    from gateway import kanban_watchers_dispatcher as kwd

    settings = kwd._DispatcherSettings(
        interval=60.0, max_spawn=9, max_in_progress=4,
        failure_limit=2, stale_timeout_seconds=0, reconcile_orphans=True,
        default_assignee=None, max_in_progress_per_profile=None,
    )
    dp = kwd._KanbanDispatcher(kb, settings)

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured[kwargs.get("board")] = kwargs
        return kb.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    # connect is resolved by slug; keep the real one so boards differ.

    kb.create_board("tight")
    kb.create_board("loose")
    kb.write_board_metadata("tight", max_spawn=2)

    dp.tick_once_for_board("tight")
    dp.tick_once_for_board("loose")

    assert captured.get("tight", {}).get("max_spawn") == 2
    assert captured.get("loose", {}).get("max_spawn") == 9
    # max_in_progress is the host-wide global and is passed through identically.
    assert captured.get("tight", {}).get("max_in_progress") == 4
    assert captured.get("loose", {}).get("max_in_progress") == 4


# ---------------------------------------------------------------------------
# 4. Budget end-to-end: a board pinned to max_spawn=2 spawns only 2
# ---------------------------------------------------------------------------


def test_board_pinned_max_spawn_limits_spawns(kanban_home, all_assignees_spawnable):
    """board.json max_spawn:2 caps a board's concurrency; a global-only board does not.

    (AC-1 analogue at the budget level.)"""
    kb.create_board("tight")

    spawns: list = []
    with kbc.connect(board="tight") as conn:
        for title in ("t1", "t2", "t3", "t4"):
            kb.create_task(conn, title=title, assignee="alice")

        # Pin this board to 2 via the board.json accessor + real resolution.
        kb.write_board_metadata("tight", max_spawn=2)
        max_spawn, _ = kanban_ops.resolve_board_caps("tight", cli_max=None)
        res = kbd.dispatch_once(
            conn,
            board="tight",
            spawn_fn=_fake_spawn_factory(spawns),
            max_spawn=max_spawn,
            max_in_progress=99,  # host cap high enough to not bind
        )

    # budget = max_spawn - running = 2 → exactly two spawns, not all four.
    assert len(res.spawned) == 2
    assert len(spawns) == 2


def test_board_without_cap_is_unbounded(kanban_home, all_assignees_spawnable):
    """Without a board.json field and no global cap, all ready tasks spawn —
    the default behaviour is unchanged (AC-2 regression)."""
    kb.create_board("loose")

    spawns: list = []
    with kbc.connect(board="loose") as conn:
        for title in ("t1", "t2", "t3"):
            kb.create_task(conn, title=title, assignee="alice")
        max_spawn, _ = kanban_ops.resolve_board_caps("loose", cli_max=None)
        assert max_spawn is None
        res = kbd.dispatch_once(
            conn,
            board="loose",
            spawn_fn=_fake_spawn_factory(spawns),
            max_spawn=max_spawn,
            max_in_progress=99,
        )
    assert len(res.spawned) == 3
    assert len(spawns) == 3
