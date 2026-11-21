"""A running card that carries no ``max_runtime_seconds`` can still be bounded.

2026-11-05: one ``manage_connections`` call stayed wedged for 4 h 59 inside a running
card. Nothing reclaimed it — the worker wrapper heartbeats every 60 s *while* a tool call
runs, so ``last_heartbeat_at`` never aged past the 1 h gap, and
``dispatch_stale_timeout_seconds`` (4 h) only applies to a heartbeat that went quiet.
A wall-clock cap is the only guard that ignores heartbeats, so
``kanban.default_max_runtime_seconds`` bounds a card that carries no cap of its own.

The shipped default is ``0`` (off): this adds the mechanism, and an operator opts into a
policy. A card's own ``max_runtime_seconds`` always wins.
"""
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_since(conn, tid: str, seconds: int) -> None:
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - seconds
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, tid),
        )


def test_the_default_cap_terminates_a_card_that_carries_none(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no cap of its own", assignee="worker")
        card = kb.get_task(conn, tid)
        assert card is not None and card.max_runtime_seconds is None
        _running_since(conn, tid, 3 * 3600)

        timed_out = kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None, default_seconds=7200)

        assert timed_out == [tid]
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "ready"
        assert "timed_out" in [event.kind for event in kb.list_events(conn, tid)]


def test_without_a_default_an_uncapped_card_is_left_alone(kanban_home):
    """0 = off: the historical behaviour, and the escape hatch for a genuinely long run."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no cap of its own", assignee="worker")
        _running_since(conn, tid, 3 * 3600)

        assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == []
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "running"


def test_a_cards_own_cap_wins_over_the_default(kanban_home):
    """A card that asks for more time than the default keeps its own cap."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="declared long job", assignee="worker",
                             max_runtime_seconds=6 * 3600)
        _running_since(conn, tid, 3 * 3600)

        assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None, default_seconds=7200) == []
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "running"


def test_a_cards_shorter_cap_still_applies(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="short cap", assignee="worker", max_runtime_seconds=60)
        _running_since(conn, tid, 600)

        assert kbd.enforce_max_runtime(
            conn, signal_fn=lambda *_: None, default_seconds=7200) == [tid]


def test_dispatch_once_applies_the_default_cap(kanban_home, monkeypatch):
    """End to end through the tick: the settings value reaches ``enforce_max_runtime``."""
    # Every sibling reclaimer is neutralised: the subject here is the cap, and a
    # crashed/stale/orphan sweep would otherwise take the card first (they run before
    # ``enforce_max_runtime``). No signal either: the pid is this test process and a
    # real os.kill would SIGTERM the suite.
    for name in ("reap_worker_zombies", "reconcile_orphaned_running",
                 "detect_stale_running", "detect_crashed_workers", "reap_terminal_workers"):
        monkeypatch.setattr(kbd, name, lambda *a, **k: [])
    monkeypatch.setattr(kb, "release_stale_claims", lambda *a, **k: 0)
    monkeypatch.setattr(kbd, "_kill_fn", lambda *_: None)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wedged", assignee="worker")
        _running_since(conn, tid, 3 * 3600)

        result = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: None, default_max_runtime_seconds=7200,
        )

        assert tid in result.timed_out


def test_the_shipped_default_is_off():
    """The mechanism ships disabled: enabling a cap is an operator decision, and a
    silent behaviour change would kill a legitimately long run on upgrade."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["default_max_runtime_seconds"] == 0


def test_dispatcher_settings_read_the_default_cap():
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    assert _resolve_dispatcher_settings(
        {"default_max_runtime_seconds": 1800}, kb).default_max_runtime_seconds == 1800
    assert _resolve_dispatcher_settings({}, kb).default_max_runtime_seconds == 0
    assert _resolve_dispatcher_settings(
        {"default_max_runtime_seconds": "nope"}, kb).default_max_runtime_seconds == 0
    assert _resolve_dispatcher_settings(
        {"default_max_runtime_seconds": -5}, kb).default_max_runtime_seconds == 0
