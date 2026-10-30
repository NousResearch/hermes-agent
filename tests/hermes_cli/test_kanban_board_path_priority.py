"""Regression tests for _board_path priority: explicit board trumps env pin.

t_3f1c63a5: MCP-only sessions (dispatcher workers) always run with
HERMES_KANBAN_DB pinned to their own board. Before this fix, that ambient
env var was consulted BEFORE the caller's explicit board= argument, so
kanban_create(board="other-board") / kanban_show(board="other-board")
silently landed on the caller's own board instead of the requested one —
exactly the failure mode that stalled t_8239fc9c / t_09015699.
"""
import os
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root


@pytest.fixture(autouse=True)
def _clean_env_board(monkeypatch):
    """Remove env vars that would pin the board during these tests."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)


def test_explicit_board_trumps_kanban_db_env(monkeypatch, tmp_path):
    """When HERMES_KANBAN_DB pins to a dispatcher-injected path (the exact
    shape every kanban worker/MCP session runs with), an explicit
    board=<slug> must still resolve to that slug's own board dir."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pinned.db"))
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override env pin, got {path}"
    )


def test_explicit_board_trumps_kanban_board_env(monkeypatch, tmp_path):
    """When HERMES_KANBAN_BOARD pins to board A, explicit board=slug
    must resolve to slug's board dir."""
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "jarvis-os")
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override HERMES_KANBAN_BOARD, got {path}"
    )


def test_none_board_falls_back_to_kanban_db_env(monkeypatch, tmp_path):
    """When no board is passed, HERMES_KANBAN_DB should still win (back-compat
    for dispatcher-spawned workers with no board= override)."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    path = kb.kanban_db_path(board=None)
    assert path == pinned, f"None board with env pin should use env, got {path}"


def test_none_board_falls_back_to_kanban_board_env(monkeypatch, tmp_path):
    """When no board is passed, HERMES_KANBAN_BOARD should resolve normally
    IF the named board exists on disk."""
    kanban_home = tmp_path / "hermes_test"
    boards_dir = kanban_home / "kanban" / "boards" / "jarvis-os"
    boards_dir.mkdir(parents=True, exist_ok=True)
    (boards_dir / "board.json").write_text('{"name": "Jarvis OS"}', encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "jarvis-os")
    path = kb.kanban_db_path(board=None)
    assert "jarvis-os" in str(path), (
        f"None board should fall back to HERMES_KANBAN_BOARD, got {path}"
    )


def test_explicit_board_does_not_mutate_env(monkeypatch, tmp_path):
    """Calling kanban_db_path(board=...) must not alter env state — a
    subsequent call without board= must still honor the original env pin."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    _ = kb.kanban_db_path(board="sycode-trading")
    path2 = kb.kanban_db_path()
    assert path2 == pinned, (
        f"env pin should survive after explicit-board call, got {path2}"
    )


def test_explicit_board_works_without_any_env(monkeypatch):
    """Explicit board= should resolve correctly even without env vars."""
    path = kb.kanban_db_path(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should work without env, got {path}"
    )


def test_workspaces_root_also_honors_explicit_board_over_env(monkeypatch, tmp_path):
    """The same ambient-env-vs-explicit-board priority bug applied to every
    _board_path() caller, not just kanban_db_path — workspaces_root must
    honor board= over HERMES_KANBAN_WORKSPACES_ROOT too."""
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "pinned-workspaces"))
    path = kb.workspaces_root(board="sycode-trading")
    assert "sycode-trading" in str(path), (
        f"explicit board should override HERMES_KANBAN_WORKSPACES_ROOT, got {path}"
    )


# ---------------------------------------------------------------------------
# CLI --board boundary (os-reviewer P1 on PR#107195 / t_11c4afd8): a
# worker-pinned env (HERMES_KANBAN_DB/HERMES_KANBAN_BOARD) must not outrank
# an explicit `hermes kanban --board B ...` invocation, which resolves via
# scoped_current_board() rather than a direct board= call argument. The
# original fix only prioritized the direct argument; --board went through
# _board_path(..., board=None) and so still lost to the env pin.
# ---------------------------------------------------------------------------

def test_scoped_current_board_context_trumps_kanban_db_env(monkeypatch, tmp_path):
    """kb.scoped_current_board() (what CLI --board and the dashboard
    plugin_api use) must resolve like an explicit board= call even while
    HERMES_KANBAN_DB pins a different, worker-injected path."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "worker-pinned.db"))
    with kb.scoped_current_board("sycode-trading"):
        path = kb.kanban_db_path()
    assert "sycode-trading" in str(path), (
        f"scoped --board context should override HERMES_KANBAN_DB, got {path}"
    )


def test_cli_board_flag_trumps_worker_env_pin_end_to_end(monkeypatch, tmp_path):
    """Real boundary: a worker shell pinned via HERMES_KANBAN_DB/BOARD (the
    exact env every dispatcher-spawned kanban worker runs with) invokes
    `hermes kanban --board B create ...` through the actual CLI parser.
    Only board B's physical database may receive the write; the pinned
    board's database must not also contain the task."""
    import argparse

    from hermes_cli import kanban as kc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("worker-own-board")
    kb.create_board("target-board")

    # Simulate the ambient env every MCP/dispatcher-spawned worker carries:
    # pinned to its OWN board, distinct from the one it explicitly targets.
    worker_pinned_db = kb.kanban_db_path(board="worker-own-board")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(worker_pinned_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-own-board")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "--board", "target-board", "create", "cross-board task"])
    rc = kc.kanban_command(args)
    assert rc == 0

    target_db = kb.kanban_db_path(board="target-board")
    assert target_db != worker_pinned_db

    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect(board="target-board") as conn:
        rows = conn.execute("SELECT title FROM tasks").fetchall()
    assert any(r[0] == "cross-board task" for r in rows), (
        "explicit --board flag must land the write on the target board, "
        f"got rows={rows!r} in {target_db}"
    )

    # And the worker's own pinned board must NOT have received the write —
    # this is the exact silent-misroute failure class the fix closes.
    with kbc.connect(board="worker-own-board") as conn:
        own_rows = conn.execute("SELECT title FROM tasks").fetchall()
    assert not any(r[0] == "cross-board task" for r in own_rows), (
        f"--board must not fall through to the worker's env-pinned board, "
        f"but found the task there too: {own_rows!r}"
    )


# ---------------------------------------------------------------------------
# Fenced callers keep the pinned path (side-effect screen on PR#131913).
#
# The env pins are the dispatcher's worker isolation (5ec6baa: "workers
# physically cannot see other boards"), and kanban_path_is_fenced checks the
# PINNED path / fenced root. Explicit board intent therefore only outranks the
# pin where no fence applies: a dispatched worker (HERMES_KANBAN_TASK) or a
# delegated child / descendant (HERMES_DELEGATED_CHILD_CONTEXT) passing
# board=<other> must never reach that board, or it would both see and write a
# board outside the dispatcher's fence — and it is refused with an error rather
# than silently redirected onto the pinned board (#142435). Machine flows that enumerate
# boards (gateway notifier / watcher / dispatcher ticks) are fenced the same
# way via kanban_db.pin_first_board_resolution() — the enumerated slug is not
# caller intent.
# ---------------------------------------------------------------------------

def _pin_like_dispatcher(monkeypatch, tmp_path, own="own-board"):
    """The pins kanban_db_dispatch._default_spawn injects into every worker."""
    pinned = tmp_path / "worker-pinned.db"
    pinned_workspaces = tmp_path / "worker-pinned-workspaces"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(pinned_workspaces))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", own)
    return pinned, pinned_workspaces


def test_dispatched_worker_foreign_board_is_refused_not_redirected(monkeypatch, tmp_path):
    """#142435: a dispatched worker (HERMES_KANBAN_TASK) naming ANOTHER board gets a
    clear error from every pinned resolver: never that board's path (the fence) and
    never a silent redirect onto the pinned board. Its own board — or no board —
    still resolves through the pin."""
    pinned, pinned_workspaces = _pin_like_dispatcher(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_3f1c63a5")
    for resolve in (kb.kanban_db_path, kb.workspaces_root):
        with pytest.raises(ValueError, match="'other-board' is not accessible"):
            resolve(board="other-board")
    with pytest.raises(ValueError, match="not accessible"), kb.scoped_current_board("other-board"):
        kb.kanban_db_path()  # CLI --board / dashboard scope is the same explicit intent
    assert kb.kanban_db_path(board="own-board") == pinned
    assert kb.kanban_db_path(board="OWN-BOARD") == pinned
    assert kb.workspaces_root(board="own-board") == pinned_workspaces
    assert kb.kanban_db_path() == pinned


def test_worker_shell_cli_board_flag_is_refused_end_to_end(monkeypatch, tmp_path, capsys):
    """#142409: `hermes kanban --board B list --status blocked` from a dispatched
    worker's shell read the worker's own board under B's name and reported an
    empty queue (exit 0); a `create` landed on the wrong board. Both now fail
    loudly, neither database changes, and the worker's own board stays usable."""
    import argparse

    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("own")
    kb.create_board("target")
    with kbc.connect_closing(board="target") as conn:
        kb.block_task(conn, kb.create_task(conn, title="waiting on a human"), reason="phone")
    own_db, target_db = kb.kanban_db_path(board="own"), kb.kanban_db_path(board="target")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(own_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "own")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

    def run(*argv):
        parser = argparse.ArgumentParser(prog="hermes", add_help=False)
        kc.build_parser(parser.add_subparsers(dest="command"))
        return kc.kanban_command(parser.parse_args(["kanban", *argv]))

    def titles(db):
        with kbc.connect_closing(db_path=db) as conn:
            return [r[0] for r in conn.execute("SELECT title FROM tasks")]

    assert run("--board", "target", "list", "--status", "blocked") != 0
    assert "'target' is not accessible" in capsys.readouterr().err
    assert run("--board", "target", "create", "misrouted") != 0
    assert titles(own_db) == []
    assert titles(target_db) == ["waiting on a human"]
    assert run("--board", "own", "create", "mine") == 0
    assert titles(own_db) == ["mine"]


def test_fenced_pin_without_board_env_matches_own_board_by_path(monkeypatch, tmp_path):
    """A pin set without HERMES_KANBAN_BOARD still recognises its own board (by
    resolved path) and refuses any other."""
    pinned = kb.kanban_db_path(board="own-board")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    assert kb.kanban_db_path(board="own-board") == pinned
    with pytest.raises(ValueError, match="not accessible"):
        kb.kanban_db_path(board="default")


def test_delegated_child_foreign_board_is_refused(monkeypatch, tmp_path):
    """Delegated children (spawned descendant marker, or the in-process context)
    are fenced the same way; their own board stays on the fenced pin."""
    from agent.delegation_context import (
        DELEGATED_CHILD_ENV_MARKER, delegated_child_context, kanban_path_is_fenced,
    )

    pinned, _ = _pin_like_dispatcher(monkeypatch, tmp_path)
    with delegated_child_context(), pytest.raises(ValueError, match="not accessible"):
        kb.kanban_db_path(board="other-board")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(tmp_path))  # fenced root
    with pytest.raises(ValueError, match="not accessible"):
        kb.kanban_db_path(board="other-board")
    path = kb.kanban_db_path(board="own-board")
    assert path == pinned
    assert kanban_path_is_fenced(path), (
        "the explicit board escaped the dispatcher's pin-based fence"
    )


def test_fenced_board_enumeration_keeps_working(monkeypatch, tmp_path):
    """Enumerated slugs are not caller intent: machine flows keep aliasing to the
    pin, and read_board_metadata (\"never raises\") / list_boards keep listing
    every board for a fenced worker (`hermes kanban boards list` in its shell)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("own-board")
    kb.create_board("other-board")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="own-board")))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "own-board")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    with kb.pin_first_board_resolution():
        assert kb.kanban_db_path(board="other-board") == kb.kanban_db_path()
    slugs = {b["slug"] for b in kb.list_boards()}
    assert {"default", "own-board", "other-board"} <= slugs


def test_pin_first_board_resolution_maps_every_slug_to_the_pin(monkeypatch, tmp_path):
    """Machine flows (gateway notifier/watcher/dispatcher ticks) enumerate board
    slugs; inside pin_first_board_resolution() every slug must map to the
    pinned DB, and with no pin set the context must change nothing."""
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    with kb.pin_first_board_resolution():
        assert kb.kanban_db_path(board="default") == pinned
        assert kb.kanban_db_path(board="sycode-trading") == pinned
        assert kb.kanban_db_path(board=None) == pinned
    # Outside the machine-flow context the explicit board wins again.
    assert "sycode-trading" in str(kb.kanban_db_path(board="sycode-trading"))


def test_pin_first_board_resolution_without_pin_keeps_explicit_board(monkeypatch, tmp_path):
    """pin_first_board_resolution() is a no-op when no pin is set: multi-board
    notifier/dispatcher ticks on an unpinned box must keep resolving each slug
    to its own physical DB."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kb.pin_first_board_resolution():
        assert "sycode-trading" in str(kb.kanban_db_path(board="sycode-trading"))


def test_notifier_seam_connect_lands_on_pinned_db(monkeypatch, tmp_path):
    """Integration seam (the screen's repro shape): with HERMES_KANBAN_DB
    pinned, the notifier's per-slug connect(board=slug) inside the machine-flow
    context must open the pinned DB — the writes the worker made via
    connect() (no board) live there."""
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pinned = tmp_path / "board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    kb.init_db()
    with kbc.connect() as conn:  # what the dispatcher/notifier box writes
        kb.create_task(conn, title="pinned task", assignee="worker")
    # The notifier polls every enumerated slug; before the fix this opened
    # each slug's own (empty) DB and found nothing.
    with kb.pin_first_board_resolution():
        for slug in ("default", "some-other-board"):
            with kbc.connect(board=slug) as conn:
                rows = conn.execute("SELECT title FROM tasks").fetchall()
            assert any(r[0] == "pinned task" for r in rows), (
                f"per-slug connect lost the pinned DB for board={slug!r}"
            )
