"""active_session_registry_snapshot must not probe liveness under lock (#115578).

_prune_dead() runs one psutil round-trip per entry (~7 ms on Windows for
the POSIX zombie probe). Holding the exclusive, unfair registry file lock
across that starves concurrent pollers once a handful of leases exist.
The snapshot reads raw entries under the lock, prunes after release, and
writes back only the lease ids proven dead — so a lease created between
the snapshot and the write-back survives.
"""

import json
import os

from hermes_cli import active_sessions


DEAD_PID = 2 ** 30  # never a live pid; _pid_exists() is False, no signal sent


def _entry(lease_id, session_id, pid):
    return {"lease_id": lease_id, "session_id": session_id, "pid": pid}


def _seed(home, entries):
    state_path = home / "runtime" / "active_sessions.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    active_sessions._write_entries(state_path, entries)
    return state_path


def _read_all(state_path):
    return json.loads(state_path.read_text(encoding="utf-8"))["entries"]


class _TrackingFileLock(active_sessions._FileLock):
    held = 0

    def __enter__(self):
        result = super().__enter__()
        type(self).held += 1
        return result

    def __exit__(self, *exc):
        type(self).held -= 1
        return super().__exit__(*exc)


def test_prune_dead_runs_after_lock_released(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _seed(home, [_entry("live-1", "s-live", os.getpid()), _entry("dead-1", "s-dead", DEAD_PID)])

    monkeypatch.setattr(active_sessions, "_FileLock", _TrackingFileLock)
    _TrackingFileLock.held = 0
    real_prune = active_sessions._prune_dead
    seen = {}

    def spy_prune(entries, **kwargs):
        seen["held_during_prune"] = _TrackingFileLock.held
        return real_prune(entries, **kwargs)

    monkeypatch.setattr(active_sessions, "_prune_dead", spy_prune)

    live = active_sessions.active_session_registry_snapshot(registry_home=home)

    assert seen["held_during_prune"] == 0
    assert [e["lease_id"] for e in live] == ["live-1"]


def test_snapshot_still_prunes_and_persists(tmp_path):
    home = tmp_path / "home"
    state_path = _seed(home, [_entry("live-1", "s-live", os.getpid()), _entry("dead-1", "s-dead", DEAD_PID)])

    live = active_sessions.active_session_registry_snapshot(registry_home=home)

    assert [e["lease_id"] for e in live] == ["live-1"]
    assert [e["lease_id"] for e in _read_all(state_path)] == ["live-1"]


def test_concurrent_lease_survives_prune_writeback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    state_path = _seed(home, [_entry("live-1", "s-live", os.getpid()), _entry("dead-1", "s-dead", DEAD_PID)])
    newcomer = _entry("new-1", "s-new", os.getpid())

    real_prune = active_sessions._prune_dead

    def spy_prune(entries, **kwargs):
        # A concurrent acquirer lands between our snapshot and our write-back.
        current = active_sessions._read_entries(state_path, strict=True)
        current.append(dict(newcomer))
        active_sessions._write_entries(state_path, current)
        return real_prune(entries, **kwargs)

    monkeypatch.setattr(active_sessions, "_prune_dead", spy_prune)

    live = active_sessions.active_session_registry_snapshot(registry_home=home)

    assert [e["lease_id"] for e in live] == ["live-1"]
    assert sorted(e["lease_id"] for e in _read_all(state_path)) == ["live-1", "new-1"]