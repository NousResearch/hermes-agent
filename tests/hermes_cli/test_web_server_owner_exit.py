"""A Desktop-over-SSH ``serve --isolated`` backend retires itself once the Desktop's ownership lock
names a newer spawn (#132034): a reconnect that cannot prove the old pid is its own drops the lock
without signalling it, so without this the superseded backend lingers as an extra ``state.db``
writer. It exits only between turns (the retirement fence proves idle and closes admission first),
and any lock it cannot validate keeps it up."""

from __future__ import annotations

import json
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


def test_lock_written_by_the_windows_ssh_runtime_lets_the_superseded_backend_retire(tmp_path, monkeypatch):
    """A Windows SSH host's lock comes from windows_ssh_runtime.write_lock, fed the Desktop's record
    (no logPath). The owner watchdog must accept it, or Windows hosts keep stacking backends."""
    from pathlib import PureWindowsPath

    from hermes_cli import windows_ssh_runtime

    # The runtime's root is a WindowsPath on that host; pass it as data instead of faking the OS.
    monkeypatch.setattr(windows_ssh_runtime, "_root",
                        lambda: PureWindowsPath(r"C:\Users\u\.hermes\desktop-ssh"))
    desktop_record = {k: v for k, v in _lock(NEW).items() if k != "logPath"}
    desktop_record.update(creationTimeNs="133700000000000000", hermesPath=r"C:\Hermes\hermes.exe")
    lock_path = tmp_path / "desktop-ssh" / OID / "backend.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(windows_ssh_runtime._lock_record(OID, desktop_record))

    assert should_retire_superseded(lock=read_valid_backend_lock(lock_path), my_nonce=ME, age_s=600.0)
