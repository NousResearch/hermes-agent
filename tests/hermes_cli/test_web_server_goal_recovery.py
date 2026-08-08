"""Tests for orphan goal recovery on the serve/web_server surface (#81109).

``hermes serve`` hosts the desktop chat's tui_gateway in-process. When the
browser detaches, the PTY reaper kills the chat child after the TTL, orphaning
any standing /goal. The server's recovery sweep re-drives the goal through the
embedded tui_gateway (session.resume + prompt.submit).

Covered here:
- the sweep's owns_goal callback treats a LIVE tui_gateway session as owned
  (no claim → no double-fire)
- _goal_recovery_drive_session drives a continuation through prompt.submit
  for a session that is no longer live (resume path)
- the periodic sweep loop is gated by config (0 disables)
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    from hermes_cli import web_server

    # The sweeper is cached on app.state — clear between tests.
    if hasattr(web_server.app.state, "_goal_sweeper_obj"):
        del web_server.app.state._goal_sweeper_obj
    yield web_server


@pytest.fixture()
def tui_server(hermes_home, monkeypatch):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    # patch.dict restores sys.modules on exit, dropping modules imported
    # inside the block. Re-register this instance so the web_server sweep
    # (which imports tui_gateway.server lazily) resolves to the SAME module
    # and sees the sessions this fixture populates.
    sys.modules["tui_gateway.server"] = mod
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


def _set_goal(home, session_id: str, *, owner_pid=None):
    from hermes_state import SessionDB
    from hermes_cli.goals import GoalState, save_goal

    db = SessionDB(db_path=Path(home) / "state.db")
    state = GoalState(
        goal=f"goal for {session_id}",
        status="active",
        turns_used=0,
        max_turns=5,
        created_at=time.time() - 600,
        last_turn_at=time.time() - 600,
    )
    if owner_pid is not None:
        state.owner_pid = owner_pid
        state.last_owner_seen_at = time.time() - 60
    else:
        state.last_owner_seen_at = time.time() - 3600
    save_goal(session_id, state, db=db)
    db.close()


class TestServeSweep:

    def test_sweep_homes_resolve_launch_profile(self, server, hermes_home):
        homes = server._goal_recovery_homes()
        assert str(hermes_home) in homes

    def test_live_tui_session_not_claimed(self, server, tui_server, hermes_home):
        """A session live in the in-process tui_gateway is owned — the sweep
        must not claim (and thus must not double-fire) its goal."""
        _set_goal(hermes_home, "live-tui-s1", owner_pid=4_000_000)
        tui_server._sessions["sid-live"] = {
            "session_key": "live-tui-s1",
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "cols": 120,
        }
        sweeper = server._goal_sweeper()
        assert sweeper.sweep() == []

    def test_orphaned_session_claimed_when_not_live(
        self, server, tui_server, hermes_home
    ):
        """A goal whose session is NOT live in the registry is claimed."""
        _set_goal(hermes_home, "orphan-tui-s1", owner_pid=4_000_000)
        sweeper = server._goal_sweeper()
        claims = sweeper.sweep()
        assert [c[1] for c in claims] == ["orphan-tui-s1"]

    def test_drive_session_calls_prompt_submit(
        self, server, tui_server, hermes_home
    ):
        """_goal_recovery_drive_session drives a continuation turn via the
        tui_gateway prompt.submit handler."""
        # A session that IS live — drive should reuse it (fast path) and
        # submit the continuation.
        tui_server._sessions["sid-drive"] = {
            "session_key": "drive-s1",
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "cols": 120,
        }
        with patch.object(
            tui_server, "_find_live_session_by_key", return_value=("sid-drive", None)
        ) as find_live, patch.object(
            tui_server, "_start_agent_build", return_value=None
        ):
            ok = server._goal_recovery_drive_session("drive-s1", "keep going")
        assert ok is True
        find_live.assert_called_once_with("drive-s1")

    def test_sweep_loop_disabled_by_config(self, server, monkeypatch):
        monkeypatch.setattr(server, "_goal_recovery_interval_minutes", lambda: 0)
        # The loop returns immediately when disabled — no sleep, no sweep.
        import asyncio

        async def _run():
            await server._goal_recovery_sweep_loop()

        asyncio.run(_run())
