"""Cross-profile kill refusal regression tests (#89315).

A poisoned/contaminated ``gateway.pid`` inside one profile's HERMES_HOME can
truthfully name ANOTHER profile's live gateway (its ``hermes_home`` stamp
records the real owner).  ``gateway stop`` / the restart force-kill escalation
/ ``profile delete`` must refuse to signal such a PID instead of starting the
mutual cross-profile SIGTERM restart loop from the issue report.

These tests exercise the REAL code paths against real PID files, a real
flock-held gateway lock, and a real dummy child process — no mocks of the
code under test.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway.status import recorded_gateway_home_conflicts


def _spawn_gateway_lookalike(bin_dir: Path, lock_path: Path) -> subprocess.Popen:
    """Real child process whose argv matches the gateway runtime matcher."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "hermes"
    if sys.platform == "win32":
        body = "import time\ntime.sleep(120)\n"
    else:
        body = (
            "import fcntl, time\n"
            f"fh = open({str(lock_path)!r}, 'a+')\n"
            "fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "time.sleep(120)\n"
        )
    script.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    if sys.platform != "win32":
        script.chmod(0o755)
        cmd = [str(script), "gateway", "run"]
    else:
        cmd = [sys.executable, str(script), "gateway", "run"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not lock_path.exists():
        if proc.poll() is not None:
            raise RuntimeError("gateway lookalike died at startup")
        time.sleep(0.05)
    return proc


def _pid_record(proc: subprocess.Popen, script: Path, owner_home: Path) -> dict:
    from gateway.status import get_process_start_time

    return {
        "pid": proc.pid,
        "kind": "hermes-gateway",
        "argv": [str(script), "gateway", "run"],
        "start_time": get_process_start_time(proc.pid),
        "hermes_home": str(owner_home),
    }


class TestRecordedGatewayHomeConflicts:
    def test_conflicting_home_detected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "tim"))
        record = {"pid": 1, "hermes_home": str(tmp_path)}
        assert recorded_gateway_home_conflicts(record) is True

    def test_same_home_accepted(self, tmp_path, monkeypatch):
        home = tmp_path / "profiles" / "tim"
        monkeypatch.setenv("HERMES_HOME", str(home))
        record = {"pid": 1, "hermes_home": str(home)}
        assert recorded_gateway_home_conflicts(record) is False

    def test_legacy_record_without_home_proves_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert recorded_gateway_home_conflicts({"pid": 1}) is False
        assert recorded_gateway_home_conflicts(None) is False
        assert recorded_gateway_home_conflicts({"pid": 1, "hermes_home": "  "}) is False

    def test_expected_home_override(self, tmp_path):
        target = tmp_path / "profiles" / "tim"
        record = {"pid": 1, "hermes_home": str(tmp_path)}
        assert (
            recorded_gateway_home_conflicts(record, expected_home=target) is True
        )
        assert (
            recorded_gateway_home_conflicts(record, expected_home=tmp_path) is False
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock harness")
class TestCrossProfileStopRefusal:
    def test_stop_profile_gateway_refuses_other_profiles_pid(
        self, tmp_path, monkeypatch
    ):
        """Profile B's ``gateway stop`` must not SIGTERM profile A's gateway.

        On main this path is already safe upstream of any guard:
        ``get_running_pid()`` filters a pid record owned by another profile
        (and unlinks the poisoned pid file) before ``stop_profile_gateway``
        ever sees a pid — so the contract here is "returns False, other
        profile's process untouched, poisoned pid file gone", not a printed
        refusal.
        """
        root_home = tmp_path / "root-home"
        tim_home = tmp_path / "root-home" / "profiles" / "tim"
        tim_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(tim_home))

        proc = _spawn_gateway_lookalike(
            tmp_path / "bin", tim_home / "gateway.lock"
        )
        try:
            record = _pid_record(proc, tmp_path / "bin" / "hermes", root_home)
            (tim_home / "gateway.pid").write_text(json.dumps(record))

            from hermes_cli import gateway as gateway_cli

            assert gateway_cli.stop_profile_gateway() is False
            assert not (tim_home / "gateway.pid").exists(), (
                "poisoned cross-profile pid file should have been unlinked"
            )
            time.sleep(0.5)
            assert proc.poll() is None, (
                "cross-profile SIGTERM fired: profile A's gateway was killed"
            )
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_stop_profile_gateway_still_stops_own_gateway(
        self, tmp_path, monkeypatch
    ):
        """Same-home records keep stopping normally (no false refusal)."""
        tim_home = tmp_path / "profiles" / "tim"
        tim_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(tim_home))

        proc = _spawn_gateway_lookalike(
            tmp_path / "bin", tim_home / "gateway.lock"
        )
        try:
            record = _pid_record(proc, tmp_path / "bin" / "hermes", tim_home)
            (tim_home / "gateway.pid").write_text(json.dumps(record))

            from hermes_cli import gateway as gateway_cli

            assert gateway_cli.stop_profile_gateway() is True
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
            assert proc.poll() is not None, "own gateway was not stopped"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock harness")
class TestProfileDeleteStopRefusal:
    def test_stop_gateway_process_refuses_other_profiles_pid(
        self, tmp_path, capsys
    ):
        """``profile delete`` must not kill a gateway owned by another home."""
        root_home = tmp_path / "root-home"
        tim_home = root_home / "profiles" / "tim"
        tim_home.mkdir(parents=True)

        proc = _spawn_gateway_lookalike(
            tmp_path / "bin", tim_home / "gateway.lock"
        )
        try:
            record = _pid_record(proc, tmp_path / "bin" / "hermes", root_home)
            (tim_home / "gateway.pid").write_text(json.dumps(record))

            from hermes_cli.profiles import _stop_gateway_process

            _stop_gateway_process(tim_home)
            out = capsys.readouterr().out
            assert "Refusing to stop" in out
            time.sleep(0.5)
            assert proc.poll() is None, (
                "profile delete killed another profile's gateway"
            )
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_stop_gateway_process_still_stops_own_gateway(self, tmp_path):
        tim_home = tmp_path / "profiles" / "tim"
        tim_home.mkdir(parents=True)

        proc = _spawn_gateway_lookalike(
            tmp_path / "bin", tim_home / "gateway.lock"
        )
        try:
            record = _pid_record(proc, tmp_path / "bin" / "hermes", tim_home)
            (tim_home / "gateway.pid").write_text(json.dumps(record))

            from hermes_cli.profiles import _stop_gateway_process

            _stop_gateway_process(tim_home)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
            assert proc.poll() is not None, "own gateway was not stopped"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
