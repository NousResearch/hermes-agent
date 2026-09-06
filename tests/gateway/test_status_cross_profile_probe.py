"""Probing a live foreign gateway must not delete its identity files.

``get_running_pid(foreign_pid_path)`` with a held lock used to fall through
to ``_cleanup_invalid_pid_path``, force-unlinking the foreign profile's live
``gateway.pid`` + ``gateway.lock`` (split-brain) and reporting it down. The
probe must return None and leave foreign files intact; genuinely stale
(dead-PID) records must still be cleaned.
"""

import json
import os

from gateway import status as status_mod


def _write_foreign_home(tmp_path, pid, home_label):
    home = tmp_path / home_label
    home.mkdir()
    pid_path = home / "gateway.pid"
    lock_path = home / "gateway.lock"
    record = {"pid": pid, "hermes_home": str(home)}
    pid_path.write_text(json.dumps(record), encoding="utf-8")
    lock_path.write_text(json.dumps(record), encoding="utf-8")
    return pid_path, lock_path


class TestCrossProfileProbe:
    def test_live_foreign_gateway_files_survive_probe(self, tmp_path, monkeypatch):
        foreign_pid, foreign_lock = _write_foreign_home(tmp_path, os.getpid(), "foreign")
        own_home = tmp_path / "own"
        own_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(own_home))
        monkeypatch.setattr(status_mod, "is_gateway_runtime_lock_active", lambda _p: True)

        assert status_mod.get_running_pid(foreign_pid) is None
        assert foreign_pid.exists()
        assert foreign_lock.exists()

    def test_stale_dead_record_still_cleaned(self, tmp_path, monkeypatch):
        dead_pid = 999999999
        foreign_pid, foreign_lock = _write_foreign_home(tmp_path, dead_pid, "foreign")
        own_home = tmp_path / "own"
        own_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(own_home))
        monkeypatch.setattr(status_mod, "is_gateway_runtime_lock_active", lambda _p: True)

        assert status_mod.get_running_pid(foreign_pid) is None
        assert not foreign_pid.exists()
        assert not foreign_lock.exists()

    def test_poisoned_own_record_with_live_foreign_pid_still_cleaned(
        self, tmp_path, monkeypatch
    ):
        """Own-home pid file naming another profile's LIVE gateway is still
        unlinked: the file is ours (poison), only the record is foreign.
        Guards the #89315 stop-refusal contract at the probe level."""
        own_home = tmp_path / "own"
        own_home.mkdir()
        other_home = tmp_path / "other"
        other_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(own_home))
        monkeypatch.setattr(status_mod, "is_gateway_runtime_lock_active", lambda _p: True)
        record = {"pid": os.getpid(), "hermes_home": str(other_home)}
        own_pid = own_home / "gateway.pid"
        own_lock = own_home / "gateway.lock"
        own_pid.write_text(json.dumps(record), encoding="utf-8")
        own_lock.write_text(json.dumps(record), encoding="utf-8")

        assert status_mod.get_running_pid() is None
        assert not own_pid.exists()
