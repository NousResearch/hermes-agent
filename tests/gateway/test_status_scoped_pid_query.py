"""Scoped gateway status query regression tests (#106406).

A scoped ``?profile=X`` status query reads ``<X>/gateway.pid`` to report whether
profile X's gateway is running. Before #106406, ``get_running_pid`` validated that
record against THIS (serve) process's ``HERMES_HOME`` and, on the mismatch,
``cleanup_stale``-unlinked the foreign profile's ``gateway.pid``/``gateway.lock`` —
turning a read-only status query into data destruction plus a wrong "not running"
report.

These tests pin the scoped contract:

* a foreign profile's LIVE record is accepted (validated against the target home,
  not the serve home) and its PID returned;
* a third-profile POISON record inside the queried profile's dir is rejected;
* neither case, nor an inactive lock, ever ``cleanup_stale``-unlinks the foreign
  identity files — a scoped read is read-only.

The unscoped path's stale-file cleanup is preserved as a regression guard so the
fix does not silently relax the default gateway's poison-file housekeeping.

Only process-inspection DEPENDENCIES (lock liveness, /proc readers) are faked;
``get_running_pid`` itself runs unmocked.
"""

import json

import pytest

from gateway import status


def _record(pid: int, hermes_home, *, start_time: int = 123, argv=None) -> dict:
    return {
        "pid": pid,
        "kind": "hermes-gateway",
        "argv": argv or ["python", "-m", "hermes_cli.main", "gateway", "run"],
        "start_time": start_time,
        "hermes_home": str(hermes_home),
    }


def _write_identity_files(home, pid_record: dict, *, lock_record=None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway.pid").write_text(json.dumps(pid_record))
    if lock_record is not None:
        (home / "gateway.lock").write_text(json.dumps(lock_record))


@pytest.fixture
def serve_home(tmp_path, monkeypatch):
    """HERMES_HOME of the serve process issuing the scoped query (≠ the queried profile)."""
    home = tmp_path / "serve-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stub_live_process(monkeypatch, live_pid: int, cmdline: str):
    """Fake /proc readers so ``live_pid`` looks like a running gateway with ``cmdline``."""
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid == live_pid)
    monkeypatch.setattr(
        status,
        "_get_process_start_time",
        lambda pid: 123 if pid == live_pid else None,
    )
    monkeypatch.setattr(
        status,
        "_read_process_cmdline",
        lambda pid: cmdline if pid == live_pid else None,
    )


@pytest.mark.skipif(
    "sys.platform == 'win32'",
    reason="POSIX profile-path semantics; real flock path covered elsewhere",
)
class TestScopedStatusQueryPreservesForeignFiles:
    def test_scoped_returns_foreign_pid_without_unlinking(
        self, tmp_path, serve_home, monkeypatch
    ):
        """A scoped query reports the foreign profile's live gateway and leaves its files intact."""
        work_home = tmp_path / "root-home" / "profiles" / "work"
        work_home.mkdir(parents=True)
        foreign_pid = 4242
        record = _record(foreign_pid, work_home)
        _write_identity_files(work_home, record, lock_record=record)

        monkeypatch.setattr(
            status, "is_gateway_runtime_lock_active", lambda _lock_path: True
        )
        monkeypatch.setattr(status, "_read_pid_record", lambda _p: record)
        monkeypatch.setattr(status, "_read_gateway_lock_record", lambda _p: record)
        _stub_live_process(
            monkeypatch, foreign_pid, "hermes --profile work gateway run"
        )

        pid_path = work_home / "gateway.pid"
        result = status.get_running_pid(pid_path)

        assert result == foreign_pid, (
            "scoped query should report the foreign profile's live gateway"
        )
        assert pid_path.exists(), (
            "scoped read must NOT unlink the foreign profile's gateway.pid (#106406)"
        )
        assert (work_home / "gateway.lock").exists(), (
            "scoped read must NOT unlink the foreign gateway.lock (#106406)"
        )

    def test_scoped_third_profile_poison_returns_none_without_unlinking(
        self, tmp_path, serve_home, monkeypatch
    ):
        """A third-profile poison record is rejected but its host files are never deleted."""
        work_home = tmp_path / "root-home" / "profiles" / "work"
        work_home.mkdir(parents=True)
        third_home = tmp_path / "root-home" / "profiles" / "other"
        third_home.mkdir(parents=True)
        foreign_pid = 4242
        # Poison: the record sitting inside work_home names a THIRD profile's home.
        record = _record(foreign_pid, third_home)
        _write_identity_files(work_home, record, lock_record=record)

        monkeypatch.setattr(
            status, "is_gateway_runtime_lock_active", lambda _lock_path: True
        )
        monkeypatch.setattr(status, "_read_pid_record", lambda _p: record)
        monkeypatch.setattr(status, "_read_gateway_lock_record", lambda _p: record)
        _stub_live_process(
            monkeypatch, foreign_pid, "hermes --profile other gateway run"
        )

        pid_path = work_home / "gateway.pid"
        result = status.get_running_pid(pid_path)

        assert result is None, (
            "a third-profile poison record must not be lent as work's gateway"
        )
        assert pid_path.exists(), (
            "scoped read must NOT unlink even a poisoned foreign pid file (#106406)"
        )
        assert (work_home / "gateway.lock").exists(), (
            "scoped read must NOT unlink the poisoned foreign lock (#106406)"
        )

    def test_scoped_inactive_lock_returns_none_without_unlinking(
        self, tmp_path, serve_home, monkeypatch
    ):
        """A scoped query with an inactive foreign lock reports None and deletes nothing."""
        work_home = tmp_path / "root-home" / "profiles" / "work"
        work_home.mkdir(parents=True)
        record = _record(4242, work_home)
        _write_identity_files(work_home, record, lock_record=None)

        monkeypatch.setattr(
            status, "is_gateway_runtime_lock_active", lambda _lock_path: False
        )
        # A scoped query must not fall back to the (serve-process) runtime status.
        monkeypatch.setattr(
            status,
            "get_runtime_status_running_pid",
            lambda *a, **k: pytest.fail("scoped query must not consult runtime status"),
        )

        pid_path = work_home / "gateway.pid"
        result = status.get_running_pid(pid_path)

        assert result is None
        assert pid_path.exists(), (
            "scoped read with an inactive lock must NOT unlink the foreign pid file (#106406)"
        )


class TestUnscopedCleanupRegressionGuard:
    def test_unscoped_stale_pid_file_is_still_cleaned_up(self, serve_home, monkeypatch):
        """Regression guard: the unscoped default path still cleans up a stale pid file."""
        stale_pid = 99998
        record = _record(stale_pid, serve_home)
        _write_identity_files(serve_home, record, lock_record=None)

        monkeypatch.setattr(
            status, "is_gateway_runtime_lock_active", lambda _lock_path: False
        )
        monkeypatch.setattr(
            status, "get_runtime_status_running_pid", lambda *a, **k: None
        )
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)

        result = status.get_running_pid()

        assert result is None
        assert not (serve_home / "gateway.pid").exists(), (
            "unscoped stale pid file SHOULD be cleaned up (existing behavior preserved)"
        )
