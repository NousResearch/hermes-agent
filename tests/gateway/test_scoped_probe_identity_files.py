"""Tests for gateway runtime status tracking."""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import status


class TestScopedProbeIdentityFileSurvival:
    """A scoped liveness probe must never unlink another home's identity files.

    ``resolve_gateway_liveness(profile_dir=...)`` is the dashboard/kanban/cron-gate ladder for
    *another* profile's home. Its PID rung runs ``get_running_pid``, whose scoped branch
    hard-codes ``runtime_pid = None`` for the runtime-status fallback — so whenever the runtime
    lock reads inactive, the "lock inactive" rung reached ``_cleanup_invalid_pid_path`` with the
    default ``cleanup_stale=True`` and unlinked that home's ``gateway.pid`` + ``gateway.lock``,
    the exact pair rung 1 needs. ``get_running_pid``'s own docstring promises the opposite
    ("polling another profile must not delete its gateway.pid/gateway.lock"), so the default
    silently contradicted its documented contract (#106406).

    A gateway in another process holds its own ``flock``; from this process the lock file is
    therefore present but unheld — that is the trigger condition, and it is what these tests
    reproduce (no flock is taken below).
    """

    def _write_foreign_gateway(self, tmp_path):
        home = tmp_path / "profiles" / "wiki"
        home.mkdir(parents=True)
        record = {
            "pid": 4242,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway", "--profile", "wiki"],
            "start_time": 123,
            "hermes_home": str(home.resolve()),
        }
        pid_path = home / "gateway.pid"
        pid_path.write_text(json.dumps(record))
        lock_path = home / "gateway.lock"
        lock_path.write_text(json.dumps(record))
        return home, pid_path, lock_path

    def test_scoped_probe_keeps_the_foreign_homes_identity_files(self, tmp_path, monkeypatch):
        # The prober runs from the DEFAULT home, exactly like the dashboard backend.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default-home"))
        home, pid_path, lock_path = self._write_foreign_gateway(tmp_path)
        # Lock inactive as seen from another process: the file is there, nobody here holds it.
        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda *a, **k: False)
        monkeypatch.setattr(status, "multiplexer_liveness_for_profile", lambda *a, **k: None)

        liveness = status.resolve_gateway_liveness(profile_dir=home, use_cache=False)

        assert not liveness.probe_error
        # Rung 1's inputs must survive the probe: losing them degrades every later probe to the
        # process-table rung, and the Desktop backend then races the live gateway for its tick.
        assert pid_path.exists()
        assert lock_path.exists()

    def test_unscoped_probe_keeps_its_housekeeping(self, tmp_path, monkeypatch):
        """The counterpart contract: the guard above is scoped-only, so the process's OWN home
        still clears a record whose lock is gone (main's poison-file housekeeping, #89315)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        record = {
            "pid": 4242,
            "kind": "hermes-gateway",
            "argv": ["python", "-m", "hermes_cli.main", "gateway"],
            "start_time": 123,
            "hermes_home": str(tmp_path.resolve()),
        }
        (tmp_path / "gateway.pid").write_text(json.dumps(record))
        (tmp_path / "gateway.lock").write_text(json.dumps(record))
        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda *a, **k: False)

        status.resolve_gateway_liveness(use_cache=False)

        assert not (tmp_path / "gateway.pid").exists()
        assert not (tmp_path / "gateway.lock").exists()
