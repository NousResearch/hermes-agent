"""Dispatcher lease reclaim after an owner restart (2026-09-11).

Before: the heartbeat was a bare ``touch``; a restarted root gateway could not tell its own dead
previous incarnation from a healthy owner, so it waited out the 300 s stale window and dispatch
paused ~5 min after every root restart (pre-flight RED on "dispatcher heartbeat" meanwhile).

After: the heartbeat names its writer (``host:pid:create_time``) and an owner that gives the lease
up writes ``released:...``. A contender seizes at once when the named owner is gone or released;
a LIVE named owner is still never seized while its heartbeat is fresh.

The two ``_try_claim`` tests at the top use only the pre-existing API, so they are the negative
control: on the unfixed code a fresh heartbeat naming a dead pid blocks the claim.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin, _dispatcher_heartbeat_is_stale


class _Gw(GatewayKanbanWatchersMixin):
    def __init__(self, profile="default"):
        self._profile = profile
        self._kanban_dispatcher_lock_handle = None
        self._kanban_dispatcher_lease_handle = None

    def _active_profile_name(self):
        return self._profile


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _paths(tmp_path):
    (tmp_path / "kanban").mkdir()
    lock = tmp_path / "kanban" / ".dispatcher.lock"
    return lock, lock.with_name(".dispatcher.heartbeat")


def _fresh(hb: Path, content: str) -> None:
    hb.write_text(content)
    os.utime(hb, None)


# ── negative control: pre-existing API only ──────────────────────────────────

def test_fresh_heartbeat_naming_a_dead_pid_is_reclaimed_at_once(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, f"{socket.gethostname()}:{_dead_pid()}:0.000")
    assert not _dispatcher_heartbeat_is_stale(hb)          # the mtime rule alone would wait 300 s
    gw = _Gw()
    try:
        assert gw._try_claim_dispatcher_lease(lock, hb, tmp_path) is True
        assert gw._owns_kanban_dispatcher_lock()
    finally:
        gw._release_kanban_dispatcher_lock()


def test_fresh_heartbeat_marked_released_is_reclaimed_at_once(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, f"released:{socket.gethostname()}:{os.getpid()}:0.000")
    gw = _Gw()
    try:
        assert gw._try_claim_dispatcher_lease(lock, hb, tmp_path) is True
    finally:
        gw._release_kanban_dispatcher_lock()


# ── safety: a live owner is never seized while fresh ─────────────────────────

def test_fresh_heartbeat_naming_a_live_process_is_not_seized(tmp_path):
    from gateway import status as st
    lock, hb = _paths(tmp_path)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.2)
        ct = st.get_process_create_time_epoch(p.pid) or 0.0
        _fresh(hb, f"{socket.gethostname()}:{p.pid}:{ct:.3f}")
        gw = _Gw()
        assert gw._try_claim_dispatcher_lease(lock, hb, tmp_path) is False
        assert not gw._owns_kanban_dispatcher_lock()
    finally:
        p.kill(); p.wait()


def test_legacy_empty_heartbeat_keeps_the_mtime_rule(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, "")
    assert _Gw()._try_claim_dispatcher_lease(lock, hb, tmp_path) is False


def test_other_host_owner_keeps_the_mtime_rule(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, f"some-other-host.invalid:{_dead_pid()}:0.000")
    assert _Gw()._try_claim_dispatcher_lease(lock, hb, tmp_path) is False


def test_stale_heartbeat_still_seized_as_before(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, "")
    old = time.time() - 3600
    os.utime(hb, (old, old))
    gw = _Gw()
    try:
        assert gw._try_claim_dispatcher_lease(lock, hb, tmp_path) is True
    finally:
        gw._release_kanban_dispatcher_lock()


# ── the helpers ──────────────────────────────────────────────────────────────

def test_owner_heartbeat_names_this_process(tmp_path):
    from gateway.kanban_watchers import _touch_dispatcher_heartbeat
    hb = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(hb)
    host, pid, ct = hb.read_text().rsplit(":", 2)
    assert host == socket.gethostname() and int(pid) == os.getpid() and float(ct) > 0
    assert not _dispatcher_heartbeat_is_stale(hb)
    assert not list(tmp_path.glob("*.tmp"))


def test_pid_reused_by_another_process_counts_as_gone(tmp_path):
    from gateway.kanban_watchers import _dispatcher_owner_gone
    hb = tmp_path / ".dispatcher.heartbeat"
    hb.write_text(f"{socket.gethostname()}:{os.getpid()}:1000.000")   # alive pid, wrong birth time
    assert _dispatcher_owner_gone(hb) is True


def test_own_live_identity_is_not_gone(tmp_path):
    from gateway.kanban_watchers import _dispatcher_owner_gone, _touch_dispatcher_heartbeat
    hb = tmp_path / ".dispatcher.heartbeat"
    _touch_dispatcher_heartbeat(hb)
    assert _dispatcher_owner_gone(hb) is False


@pytest.mark.parametrize("junk", ["", "garbage", "host:notapid:1", "a:b", "\x00\x01"])
def test_unparseable_heartbeat_fails_safe(tmp_path, junk):
    from gateway.kanban_watchers import _dispatcher_owner_gone
    hb = tmp_path / ".dispatcher.heartbeat"
    hb.write_text(junk)
    assert _dispatcher_owner_gone(hb) is False
    assert _dispatcher_owner_gone(tmp_path / "absent") is False


def test_owner_shutdown_marks_released_and_successor_claims(tmp_path):
    lock, hb = _paths(tmp_path)
    old = _Gw()
    assert old._try_claim_dispatcher_lease(lock, hb, tmp_path) is True   # no heartbeat yet -> first claimant
    old._release_kanban_dispatcher_lease()                                # inter-tick sleep: still owner
    assert hb.read_text().startswith(socket.gethostname())
    successor = _Gw()
    assert successor._try_claim_dispatcher_lease(lock, hb, tmp_path) is False   # live owner: wait
    old._release_kanban_dispatcher_lock()                                 # shutdown
    assert hb.read_text().startswith("released:")
    assert not _dispatcher_heartbeat_is_stale(hb)                         # pre-flight stays green
    try:
        assert successor._try_claim_dispatcher_lease(lock, hb, tmp_path) is True
    finally:
        successor._release_kanban_dispatcher_lock()


def test_non_owner_release_does_not_mark(tmp_path):
    lock, hb = _paths(tmp_path)
    _fresh(hb, "keep-me")
    gw = _Gw()
    gw._kanban_dispatcher_heartbeat_path = hb
    gw._release_kanban_dispatcher_lock()
    assert hb.read_text() == "keep-me"
