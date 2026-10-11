"""Regression: reclaimed kanban workers clean up their browser daemons.

A browser_exec call starts an agent-browser session daemon outside the worker's
process tree. Killing only the worker during stale/reclaim left the daemon and
Chromium children alive until the generic orphan janitor eventually noticed.
The dispatcher must run the browser orphan cleanup for sessions owned by the
reclaimed worker PID, without touching sessions whose owner is still alive.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from tools import browser_tool_lifecycle as lifecycle


@pytest.fixture
def local_claim(monkeypatch):
    monkeypatch.setattr(kb, "_host_prefix", lambda: "host:")
    monkeypatch.setattr(kbd._kb, "_host_prefix", lambda: "host:")
    return "host:claim"


def test_reclaimed_worker_cleanup_runs_browser_owner_reap(monkeypatch, local_claim):
    calls: list[int] = []

    monkeypatch.setattr(kbd._kb, "_pid_alive", lambda pid: pid == 1234)
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda pid, started_at: True)
    monkeypatch.setattr(kbd, "_pid_recycled", lambda pid, started_at: False)
    monkeypatch.setattr(kbd, "_reap_browser_sessions_owned_by_pid", lambda pid: calls.append(pid) or 2)

    signalled: list[tuple[int, int]] = []

    def fake_signal(pid: int, sig: int) -> None:
        signalled.append((pid, sig))

    result = kbd._terminate_reclaimed_worker(
        1234,
        local_claim,
        signal_fn=fake_signal,
        started_at="1234|boot",
    )

    assert result["terminated"] is True
    assert result["browser_sessions_reaped"] == 2
    assert calls == [1234]
    assert signalled


def test_browser_owner_reap_skips_live_owner(monkeypatch, tmp_path):
    socket_dir = tmp_path / "agent-browser-h_live"
    socket_dir.mkdir()
    (socket_dir / "h_live.owner_pid").write_text("4321", encoding="utf-8")
    (socket_dir / "h_live.pid").write_text("9999", encoding="utf-8")

    monkeypatch.setattr(lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle, "_owner_pid_alive", lambda socket_dir, session_name: (4321, True))
    monkeypatch.setattr(lifecycle, "_reap_socket_dir", lambda *args, **kwargs: pytest.fail("live owner reaped"))

    assert lifecycle._reap_browser_sessions_owned_by_pid(4321) == 0
    assert socket_dir.exists()


def test_browser_owner_reap_only_targets_matching_dead_owner(monkeypatch, tmp_path):
    owned = tmp_path / "agent-browser-h_owned"
    other = tmp_path / "agent-browser-h_other"
    owned.mkdir()
    other.mkdir()
    (owned / "h_owned.owner_pid").write_text("1111", encoding="utf-8")
    (other / "h_other.owner_pid").write_text("2222", encoding="utf-8")

    monkeypatch.setattr(lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        lifecycle,
        "_owner_pid_alive",
        lambda socket_dir, session_name: (1111 if session_name == "h_owned" else 2222, False),
    )

    reaped: list[tuple[str, str]] = []

    def fake_reap(socket_dir: str, session_name: str, tracked_names: set) -> bool:
        reaped.append((os.path.basename(socket_dir), session_name))
        return True

    monkeypatch.setattr(lifecycle, "_reap_socket_dir", fake_reap)

    assert lifecycle._reap_browser_sessions_owned_by_pid(1111) == 1
    assert reaped == [("agent-browser-h_owned", "h_owned")]
