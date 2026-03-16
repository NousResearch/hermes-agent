"""
Tests for gateway/orphan_cleanup.py

Covers:
- Subreaper setup (set_subreaper / is_subreaper)
- CLI session registry (register / unregister / stale detection)
- Process inspection helpers (_read_cmdline, _is_pid_alive, _get_process_parent)
- Orphan scanning and filtering
- Termination logic (SIGTERM → SIGKILL escalation)
- Integration: full scan_and_cleanup_orphans pipeline
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp directory for isolation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Also reset the module-level registry path cache
    import gateway.orphan_cleanup as oc
    oc._registry_path = None
    return tmp_path


@pytest.fixture
def sample_orphan_process():
    """
    Spawn a real child process that looks like a hermes --resume session.
    Yields (pid, proc).  Cleans up on teardown.
    """
    # Start a long-running sleep process with a hermes-like command name
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield proc.pid, proc
    # Cleanup
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subreaper tests
# ---------------------------------------------------------------------------

class TestSubreaper:
    @pytest.mark.skipif(sys.platform != "linux", reason="Linux-only prctl test")
    def test_set_subreaper_succeeds_on_linux(self):
        from gateway.orphan_cleanup import set_subreaper, is_subreaper
        result = set_subreaper()
        assert result is True
        assert is_subreaper() is True

    def test_set_subreaper_noop_on_non_linux(self):
        from gateway.orphan_cleanup import set_subreaper
        if sys.platform == "linux":
            pytest.skip("This test is for non-Linux platforms")
        result = set_subreaper()
        assert result is True  # no-op, returns True


# ---------------------------------------------------------------------------
# CLI session registry tests
# ---------------------------------------------------------------------------

class TestCLIRegistry:
    def test_register_and_unregister(self, tmp_hermes_home):
        from gateway.orphan_cleanup import (
            register_cli_session,
            unregister_cli_session,
            _read_registry,
        )

        register_cli_session("test-session-1", pid=12345)
        data = _read_registry()
        assert "test-session-1" in data
        assert data["test-session-1"]["pid"] == 12345
        assert "started_at" in data["test-session-1"]

        unregister_cli_session("test-session-1")
        data = _read_registry()
        assert "test-session-1" not in data

    def test_register_multiple_sessions(self, tmp_hermes_home):
        from gateway.orphan_cleanup import register_cli_session, _read_registry

        register_cli_session("sess-a", pid=1001)
        register_cli_session("sess-b", pid=1002)
        register_cli_session("sess-c", pid=1003)

        data = _read_registry()
        assert len(data) == 3
        assert all(s in data for s in ["sess-a", "sess-b", "sess-c"])

    def test_unregister_nonexistent_is_noop(self, tmp_hermes_home):
        from gateway.orphan_cleanup import unregister_cli_session, _read_registry

        unregister_cli_session("does-not-exist")
        data = _read_registry()
        assert isinstance(data, dict)

    def test_registry_persisted_across_reads(self, tmp_hermes_home):
        from gateway.orphan_cleanup import register_cli_session, _read_registry

        register_cli_session("persist-test", pid=9999)
        # Read again to verify disk persistence
        data2 = _read_registry()
        assert "persist-test" in data2
        assert data2["persist-test"]["pid"] == 9999


# ---------------------------------------------------------------------------
# Process inspection tests
# ---------------------------------------------------------------------------

class TestProcessInspection:
    def test_is_pid_alive_own_process(self):
        from gateway.orphan_cleanup import _is_pid_alive
        assert _is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_nonexistent(self):
        from gateway.orphan_cleanup import _is_pid_alive
        # PID 0 or a very high number should not exist
        assert _is_pid_alive(999999999) is False

    def test_read_cmdline_self(self):
        from gateway.orphan_cleanup import _read_cmdline
        cmdline = _read_cmdline(os.getpid())
        assert cmdline is not None
        assert "python" in cmdline.lower() or "pytest" in cmdline.lower()

    def test_read_cmdline_nonexistent(self):
        from gateway.orphan_cleanup import _read_cmdline
        assert _read_cmdline(999999999) is None

    def test_get_process_parent(self):
        from gateway.orphan_cleanup import _get_process_parent
        parent = _get_process_parent(os.getpid())
        # Parent should be a valid PID (could be None on some systems)
        if parent is not None:
            assert isinstance(parent, int)
            assert parent > 0


# ---------------------------------------------------------------------------
# Orphan pattern matching tests
# ---------------------------------------------------------------------------

class TestPatternMatching:
    def test_hermes_resume_matches(self):
        from gateway.orphan_cleanup import _looks_like_hermes_process
        assert _looks_like_hermes_process("hermes --resume abc123 --yolo") is True
        assert _looks_like_hermes_process("python hermes_cli.main --resume sess-1") is True

    def test_hermes_yolo_matches(self):
        from gateway.orphan_cleanup import _looks_like_hermes_process
        assert _looks_like_hermes_process("hermes --yolo") is True

    def test_non_hermes_does_not_match(self):
        from gateway.orphan_cleanup import _looks_like_hermes_process
        assert _looks_like_hermes_process("python app.py") is False
        assert _looks_like_hermes_process("node server.js") is False
        assert _looks_like_hermes_process("") is False
        assert _looks_like_hermes_process(None) is False

    def test_hermes_gateway_does_not_match(self):
        from gateway.orphan_cleanup import _looks_like_hermes_process
        # Gateway processes should NOT match (they're managed separately)
        assert _looks_like_hermes_process("hermes gateway run") is False


# ---------------------------------------------------------------------------
# Orphan scanning tests
# ---------------------------------------------------------------------------

class TestOrphanScanning:
    def test_scan_returns_list(self, tmp_hermes_home):
        from gateway.orphan_cleanup import scan_for_orphaned_hermes
        orphans = scan_for_orphaned_hermes(gateway_pid=os.getpid())
        assert isinstance(orphans, list)

    def test_scan_finds_registry_orphans(self, tmp_hermes_home):
        from gateway.orphan_cleanup import register_cli_session, scan_for_orphaned_hermes

        # Register a session with 0 TTL so it's immediately stale
        register_cli_session("stale-sess", pid=os.getpid())  # our own PID, will be alive
        orphans = scan_for_orphaned_hermes(
            gateway_pid=99999,  # different PID so our process looks orphaned
            stale_ttl=0,  # immediately stale
        )
        # Should find at least one (ourselves, registered as a stale session)
        assert len(orphans) >= 1
        found_pids = [o["pid"] for o in orphans]
        assert os.getpid() in found_pids

    def test_scan_skips_gateway_process(self, tmp_hermes_home):
        from gateway.orphan_cleanup import scan_for_orphaned_hermes

        my_pid = os.getpid()
        orphans = scan_for_orphaned_hermes(gateway_pid=my_pid, stale_ttl=0)
        # Our PID should NOT appear as it's the gateway
        found_pids = [o["pid"] for o in orphans]
        assert my_pid not in found_pids

    def test_scan_skips_fresh_registry_entries(self, tmp_hermes_home):
        from gateway.orphan_cleanup import register_cli_session, scan_for_orphaned_hermes

        register_cli_session("fresh-sess", pid=os.getpid())
        orphans = scan_for_orphaned_hermes(
            gateway_pid=99999,
            stale_ttl=3600,  # 1 hour TTL
        )
        # Should NOT find our process because it's not stale yet
        found_ids = [o.get("session_id") for o in orphans]
        assert "fresh-sess" not in found_ids

    def test_scan_cleans_dead_registry_entries(self, tmp_hermes_home):
        from gateway.orphan_cleanup import (
            register_cli_session,
            scan_for_orphaned_hermes,
            _read_registry,
        )

        # Register with a fake dead PID
        register_cli_session("dead-sess", pid=999999999)
        scan_for_orphaned_hermes(stale_ttl=0)

        # The dead entry should be cleaned from registry
        data = _read_registry()
        assert "dead-sess" not in data


# ---------------------------------------------------------------------------
# Termination tests
# ---------------------------------------------------------------------------

class TestTermination:
    def test_terminate_orphans_skips_dead_processes(self, tmp_hermes_home):
        from gateway.orphan_cleanup import terminate_orphans

        result = terminate_orphans([
            {"pid": 999999999, "cmdline": "fake", "age_seconds": 600},
        ])
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "already_dead"

    def test_terminate_orphans_handles_permission_error(self, tmp_hermes_home, monkeypatch):
        from gateway.orphan_cleanup import terminate_orphans

        # Mock _is_pid_alive to return True so we reach the kill attempt
        # (normally PID 1 returns False due to PermissionError in signal check)
        import gateway.orphan_cleanup as oc
        monkeypatch.setattr(oc, "_is_pid_alive", lambda pid: pid == 1)

        # PID 1 is init - we can't kill it (PermissionError)
        result = terminate_orphans([
            {"pid": 1, "cmdline": "init", "age_seconds": 600},
        ])
        assert len(result["failed"]) == 1
        assert "Permission" in result["failed"][0]["reason"] or "Operation not permitted" in result["failed"][0]["reason"]

    def test_terminate_kills_real_process(self, tmp_hermes_home, sample_orphan_process):
        from gateway.orphan_cleanup import terminate_orphans, _is_pid_alive

        pid, proc = sample_orphan_process
        assert _is_pid_alive(pid) is True

        result = terminate_orphans([
            {"pid": pid, "cmdline": "hermes --resume test --yolo", "age_seconds": 600},
        ], grace_period=2.0)

        # Process should be terminated
        terminated_or_killed = result["terminated"] + result["killed"]
        assert pid in terminated_or_killed


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_scan_and_cleanup_pipeline(self, tmp_hermes_home):
        from gateway.orphan_cleanup import scan_and_cleanup_orphans

        result = scan_and_cleanup_orphans(gateway_pid=os.getpid(), stale_ttl=0)
        assert "orphans_found" in result
        assert "cleanup_result" in result
        assert "dry_run" in result

    def test_dry_run_does_not_kill(self, tmp_hermes_home, sample_orphan_process):
        from gateway.orphan_cleanup import scan_and_cleanup_orphans, _is_pid_alive

        pid, proc = sample_orphan_process
        # Register it so the scanner finds it
        from gateway.orphan_cleanup import register_cli_session
        register_cli_session("dry-run-test", pid=pid)

        result = scan_and_cleanup_orphans(
            gateway_pid=99999,
            stale_ttl=0,
            dry_run=True,
        )

        # Process should still be alive (dry run)
        assert _is_pid_alive(pid) is True
        assert result["dry_run"] is True

    def test_cleanup_empty_system(self, tmp_hermes_home):
        """Scan on a clean system should return no orphans."""
        from gateway.orphan_cleanup import scan_and_cleanup_orphans

        result = scan_and_cleanup_orphans(gateway_pid=os.getpid())
        assert result["orphans_found"] == 0
        assert result["cleanup_result"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_registry_corrupted_json(self, tmp_hermes_home):
        from gateway.orphan_cleanup import _read_registry

        # Write corrupted JSON
        registry_path = tmp_hermes_home / "cli_sessions.json"
        registry_path.write_text("{not valid json")

        data = _read_registry()
        assert data == {}

    def test_registry_empty_file(self, tmp_hermes_home):
        from gateway.orphan_cleanup import _read_registry

        registry_path = tmp_hermes_home / "cli_sessions.json"
        registry_path.write_text("")

        data = _read_registry()
        assert data == {}

    def test_registry_nonexistent_file(self, tmp_hermes_home):
        from gateway.orphan_cleanup import _read_registry

        data = _read_registry()
        assert data == {}

    def test_concurrent_registry_writes(self, tmp_hermes_home):
        """Multiple rapid register calls should not corrupt the file."""
        from gateway.orphan_cleanup import register_cli_session, _read_registry

        for i in range(50):
            register_cli_session(f"concurrent-{i}", pid=10000 + i)

        data = _read_registry()
        assert len(data) == 50

    def test_subreaper_idempotent(self):
        """Calling set_subreaper multiple times should be safe."""
        from gateway.orphan_cleanup import set_subreaper
        if sys.platform != "linux":
            pytest.skip("Linux-only")

        assert set_subreaper() is True
        assert set_subreaper() is True  # should not error
