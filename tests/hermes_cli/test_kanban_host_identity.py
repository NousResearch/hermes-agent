"""Claim locks stay host-local when the machine's ``gethostname()`` changes (#73277).

On a Mac with no ``HostName`` set, ``gethostname()`` follows the network (``Mac``,
``Mac.localdomain``, ``<name>.local``). The dispatcher must still recognise its own live worker's
lock after such a flip; otherwise the expired claim is reclaimed without signalling the worker and
a duplicate spawns beside it.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def live_worker():
    proc = subprocess.Popen(["sleep", "60"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


def _running_task_with_expired_claim(conn, worker_pid: int) -> str:
    """A healthy worker mid long tool call: claim TTL lapsed, heartbeat 20 min old."""
    tid = kb.create_task(conn, title="long tool call", assignee="nobody")
    assert kb.claim_task(conn, tid) is not None
    kbd._set_worker_pid(conn, tid, worker_pid)
    now = int(time.time())
    conn.execute("UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                 (now - 60, now - 1200, tid))
    conn.commit()
    return tid


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("flipped", ["Mac", "Mac.localdomain"])
def test_live_worker_keeps_claim_after_macos_hostname_flip(conn, live_worker, monkeypatch, flipped):
    tid = _running_task_with_expired_claim(conn, live_worker.pid)
    monkeypatch.setattr(socket, "gethostname", lambda: flipped)

    signalled: list[tuple[int, int]] = []
    reclaimed = kb.release_stale_claims(conn, signal_fn=lambda pid, sig: signalled.append((pid, sig)))

    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert (reclaimed, row["status"], signalled) == (0, "running", [])
    assert row["claim_lock"] is not None
    assert live_worker.poll() is None


@pytest.mark.platforms("posix")
def test_lock_from_live_hostname_is_local_and_other_hosts_are_not(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "Build-Box.example.com")
    assert kb._is_local_lock("Build-Box.example.com:4242")
    assert kb._is_local_lock("build-box.EXAMPLE.com:4242")
    assert kb._is_local_lock(kb._claimer_id())
    assert not kb._is_local_lock("other-host:4242")
    assert not kb._is_local_lock("Build-Box.example.com")
    assert not kb._is_local_lock(None)


@pytest.mark.platforms("macos")
def test_macos_claimer_id_uses_bonjour_name_not_network_hostname(monkeypatch):
    local = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True,
                           timeout=10, check=True).stdout.strip()
    monkeypatch.setattr(socket, "gethostname", lambda: "Mac.localdomain")
    assert kb._claimer_id().split(":", 1)[0] == f"{local}.local"
