"""A Desktop-over-SSH ``serve --isolated`` backend retires itself once the Desktop's ownership lock
names a newer spawn (#132034): a reconnect that cannot prove the old pid is its own drops the lock
without signalling it, so without this the superseded backend lingers as an extra ``state.db``
writer. It exits only between turns (the retirement fence proves idle and closes admission first),
and any lock it cannot validate keeps it up."""

from __future__ import annotations

import json
import signal
from pathlib import Path

from hermes_cli.dashboard_procs import _REAP_MIN_AGE_SECONDS, read_valid_backend_lock
from hermes_cli.web_server_owner_exit import should_retire_superseded, start_owner_watchdog

OID, ME, NEW = "f" * 32, "a" * 16, "b" * 16


def _lock(nonce: str) -> dict:
    return {"schemaVersion": 2, "protocolVersion": 1, "ownershipId": OID, "spawnNonce": nonce,
            "tokenFingerprint": "c" * 32, "pid": 4242, "port": 0, "profile": "default",
            "hermesPath": "/opt/hermes/bin/hermes", "hermesHome": "~/.hermes",
            "logPath": f"~/.hermes/desktop-ssh/{OID}/{nonce}.log", "startedAt": "2026-10-03T00:00:00Z"}


def test_retires_only_on_a_valid_lock_naming_another_spawn(tmp_path):
    lock_path = tmp_path / "desktop-ssh" / OID / "backend.lock.json"
    lock_path.parent.mkdir(parents=True)

    def verdict(body, age_s=600.0):
        if body is None:
            lock_path.unlink(missing_ok=True)
        else:
            lock_path.write_text(body if isinstance(body, str) else json.dumps(body))
        return should_retire_superseded(lock=read_valid_backend_lock(lock_path), my_nonce=ME, age_s=age_s)

    assert verdict(_lock(NEW)) is True
    assert verdict(_lock(ME)) is False  # still ours
    assert verdict(None) is False  # cleanup ran, replacement not written yet
    assert verdict("{not json") is False  # unreadable
    assert verdict({**_lock(NEW), "schemaVersion": 99}) is False  # another Desktop build's lock
    assert verdict(_lock(NEW), age_s=5.0) is False  # just spawned: the lock may not name us yet
    # Same settle window as the orphan reaper: both wait for the Desktop to write the lock.
    assert verdict(_lock(NEW), age_s=_REAP_MIN_AGE_SECONDS - 1) is False


class _Server:
    should_exit = False


class _Fence:
    def __init__(self, idle: bool):
        self.idle, self.committed = idle, False

    def prepare(self):
        return {"ok": True, "idle": True, "token": "t"} if self.idle else {"ok": False, "idle": False}

    def commit(self, token):
        self.committed = token == "t"
        return {"ok": self.committed}


def _run(fence, nonces):
    server, seq = _Server(), iter(nonces)
    clock = iter(range(0, 100_000, 1000))  # every poll is well past the young-process window
    def read_lock(_path):
        nonce = next(seq, nonces[-1])
        return _lock(nonce) if nonce else None

    start_owner_watchdog(
        server, lock_path=Path("backend.lock.json"), nonce=ME, fence=fence, poll_s=0.01,
        now=lambda: float(next(clock)), read_lock=read_lock, max_polls=len(nonces) + 2).join(timeout=5)
    return server


def test_superseded_backend_retires_through_the_fence_only_when_idle():
    idle = _Fence(idle=True)
    assert _run(idle, [NEW, NEW]).should_exit is True
    assert idle.committed is True  # admission closed before exit
    assert _run(_Fence(idle=True), [NEW, None, NEW, ME]).should_exit is False  # no 2 in a row
    busy = _Fence(idle=False)
    assert _run(busy, [NEW, NEW, NEW]).should_exit is False  # an in-flight turn keeps it up
    assert busy.committed is False


# ── Leftover same-slot sibling retirement (#132133) ──────────────────────────
# A gateway restart strands the slot's pre-owner-exit backend: the Desktop's reconnect drops the
# old lock and writes a new one naming the FRESH spawn, and the leftover — its token rotated — is
# unreachable, holding the desktop session. The successor must retire it once the lock names itself.

import os as _os
from unittest.mock import patch as _patch


def test_stale_slot_sibling_pids_targets_unnamed_backends_of_the_slot():
    from hermes_cli.web_server_owner_exit import stale_slot_sibling_pids

    lock = _lock(ME)
    lock["pid"] = 4242
    scanned = [
        (4242, f"hermes serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file ~/.hermes/desktop-ssh/{OID}/dead.token --ssh-owner-nonce {ME}"),  # the lock's spawn
        (5555, f"hermes serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file ~/.hermes/desktop-ssh/{OID}/old.token --ssh-owner-nonce {'c' * 16}"),  # #132133 leftover
        (6666, f"hermes serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file ~/.hermes/desktop-ssh/{'e' * 32}/x.token --ssh-owner-nonce {'d' * 16}"),  # another slot
        (7777, "hermes serve --host 0.0.0.0 --port 9119"),  # operator's remote serve
    ]
    with _patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=scanned), \
            _patch.dict(_os.environ, {}, clear=False):
        pids = stale_slot_sibling_pids(lock, my_pid=_os.getpid())
    assert pids == [5555]


def test_owner_watchdog_sigterms_leftover_slot_sibling_once_lock_names_this_spawn():
    terms: list[int] = []
    orphan_argv = (f"hermes serve --isolated --host 127.0.0.1 --port 0 "
                   f"--ssh-session-token-file ~/.hermes/desktop-ssh/{OID}/old.token --ssh-owner-nonce {'c' * 16}")
    lock = _lock(ME)
    lock["pid"] = 4242

    with (
        _patch("hermes_cli.dashboard_procs._scan_dashboard_processes",
               return_value=[(5555, orphan_argv), (4242, "hermes serve --isolated --ssh-owner-nonce " + ME)]),
        _patch("os.kill", side_effect=lambda pid, sig: terms.append((pid, sig))),
    ):
        _run(_Fence(idle=True), [ME, ME, ME, ME])

    assert terms == [(5555, signal.SIGTERM)]


def test_owner_watchdog_spares_sibling_while_lock_settles_or_names_another():
    terms: list[int] = []
    orphan_argv = (f"hermes serve --isolated --host 127.0.0.1 --port 0 "
                   f"--ssh-session-token-file ~/.hermes/desktop-ssh/{OID}/old.token --ssh-owner-nonce {'c' * 16}")

    with (
        _patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[(5555, orphan_argv)]),
        _patch("os.kill", side_effect=lambda pid, sig: terms.append((pid, sig))),
    ):
        # Lock never settles on this spawn: missing / another spawn's / fresh-young polls kill nothing.
        _run(_Fence(idle=True), [None, None, None, None])
        _run(_Fence(idle=True), [NEW, NEW, NEW, NEW])
        young_clock = iter([0.0, 10.0, 20.0, 30.0])  # never past _REAP_MIN_AGE_SECONDS
        start_owner_watchdog(
            _Server(), lock_path=Path("backend.lock.json"), nonce=ME, fence=_Fence(idle=True),
            poll_s=0.01, now=lambda: float(next(young_clock)), max_polls=4).join(timeout=5)

    assert terms == []
