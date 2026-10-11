"""Regression tests for #27145 — kanban.default_assignee for unassigned ready tasks.

When the dispatcher hits an unassigned ready task and ``kanban.default_assignee``
is set, the dispatcher applies the assignment and spawns. Without the config,
the task is skipped (existing behavior preserved).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB."""
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345




def test_unassigned_task_auto_assigned_with_default_assignee(isolated_kanban_home):
    """Core #27145 contract: with default_assignee set, an unassigned ready
    task gets the assignment applied and dispatched on the same tick. The
    DB row is mutated (assignee column + an 'assigned' event)."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]
    assert not res.skipped_unassigned
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == task_id
    assert res.spawned[0][1] == "default"

    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] == "default"

    # 'assigned' event emitted for the audit trail
    with kbc.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ))
    assert len(evs) == 1
    payload = json.loads(evs[0][1])
    assert payload["assignee"] == "default"
    assert payload["source"] == "kanban.default_assignee"






def test_explicitly_assigned_task_untouched_by_default_assignee(isolated_kanban_home):
    """A task with an explicit assignee must NOT be touched by the
    default_assignee logic — that fallback only applies to genuinely
    unassigned rows."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee="default")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="someother",
        )
    assert task_id not in res.auto_assigned_default
    assert any(s[0] == task_id and s[1] == "default" for s in res.spawned)




# --- per-board overrides (kanban.boards.<slug>.default_assignee) -------------

def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_board_override_beats_passed_global(isolated_kanban_home):
    """Invariant: ``kanban.boards.<slug>.default_assignee`` overrides the global
    the caller passed for that board. On the 1e0c7730d7 baseline the board key
    is ignored, the (nonexistent) global resolves to None, and the card is
    skipped — red."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: default\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            board="tsa-mgmt", default_assignee="ghost-profile",
        )
    assert res.auto_assigned_default == [task_id]
    assert res.spawned and res.spawned[0][1] == "default"


def test_unset_board_key_keeps_passed_global(isolated_kanban_home):
    """Zero-change guard: a board with no override keeps the caller's global."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    svs:\n      default_assignee: default\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            board="tsa-mgmt", default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]


def test_board_override_is_opt_in_without_passed_global(isolated_kanban_home):
    """The standalone daemon passes no global; a board override still applies."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: default\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False, board="tsa-mgmt")
    assert res.auto_assigned_default == [task_id]


def test_board_without_override_and_no_global_skips(isolated_kanban_home):
    """Zero-change guard for the daemon: no board key + no passed global = skip."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    svs:\n      default_assignee: default\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False, board="tsa-mgmt")
    assert res.auto_assigned_default == []
    assert res.skipped_unassigned == [task_id]


def test_board_override_to_unknown_profile_is_ignored(isolated_kanban_home):
    """The board value does not bypass the existing profile-existence check;
    with no valid global to fall through to, the card stays unassigned."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: ghost\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False, board="tsa-mgmt")
    assert res.auto_assigned_default == []
    assert res.skipped_unassigned == [task_id]


def test_board_override_unknown_profile_falls_through_to_valid_global(isolated_kanban_home):
    """Review #135651: a board value naming an unknown profile must fall through
    to the caller's valid global, not swallow it and skip the card."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: ghost\n")
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            board="tsa-mgmt", default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]
    assert res.spawned and res.spawned[0][1] == "default"
    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] == "default"


def test_resolve_default_assignee_candidate_chain(isolated_kanban_home, monkeypatch):
    """Direct unit coverage of the per-candidate order: board (if it exists),
    then the caller's global, else None."""
    kb, home = isolated_kanban_home
    _write_config(home, "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: board-p\n")
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda n: n in {"board-p", "global-p"}))
    # Board value wins when it names an installed profile.
    assert kbd._resolve_default_assignee("global-p", board="tsa-mgmt") == "board-p"
    # Board value unknown -> the caller's valid global is used.
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda n: n in {"global-p"}))
    assert kbd._resolve_default_assignee("global-p", board="tsa-mgmt") == "global-p"
    # Neither candidate valid -> None (do not write the card).
    assert kbd._resolve_default_assignee(None, board="tsa-mgmt") is None


def test_resolve_default_assignee_multi_board_no_crosstalk(isolated_kanban_home, monkeypatch):
    kb, home = isolated_kanban_home
    _write_config(
        home,
        "kanban:\n  boards:\n    tsa-mgmt:\n      default_assignee: p1\n"
        "    svs:\n      default_assignee: p2\n",
    )
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda n: n in {"p1", "p2"}))
    assert kbd._resolve_default_assignee(None, board="tsa-mgmt") == "p1"
    assert kbd._resolve_default_assignee(None, board="svs") == "p2"
