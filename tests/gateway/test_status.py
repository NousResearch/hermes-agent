"""Tests for gateway/status.py PID file handling."""

import os
from pathlib import Path
from unittest import mock

import pytest


def test_is_hermes_gateway_process_returns_false_for_nonexistent_pid():
    """Verify _is_hermes_gateway_process returns False for PIDs that don't exist."""
    from gateway.status import _is_hermes_gateway_process
    # Use a very high PID unlikely to exist
    result = _is_hermes_gateway_process(9999999)
    assert result is False


def test_is_hermes_gateway_process_returns_false_for_non_gateway():
    """Verify _is_hermes_gateway_process returns False for non-hermes processes."""
    from gateway.status import _is_hermes_gateway_process
    # Current process is running pytest, not hermes gateway
    result = _is_hermes_gateway_process(os.getpid())
    assert result is False


def test_get_running_pid_returns_none_when_no_pidfile(tmp_path: Path):
    """Verify get_running_pid returns None when PID file doesn't exist."""
    from gateway.status import get_running_pid
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        result = get_running_pid()
    assert result is None


def test_get_running_pid_cleans_stale_pidfile(tmp_path: Path):
    """Verify stale PID files (nonexistent process) are cleaned up."""
    from gateway.status import get_running_pid
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("9999999")  # Very high PID, unlikely to exist
    
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        result = get_running_pid()
    
    assert result is None
    assert not pid_file.exists(), "Stale PID file should be removed"


def test_get_running_pid_cleans_reused_pid(tmp_path: Path):
    """Verify PID files pointing to non-gateway processes are cleaned up.
    
    This tests the core fix for issue #576: when a PID is reused by a
    different process after a gateway crash, the startup should not fail.
    """
    from gateway.status import get_running_pid
    
    # Write current process PID (pytest, not gateway)
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text(str(os.getpid()))
    
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        result = get_running_pid()
    
    # Current process is pytest, not a gateway, so should return None
    assert result is None
    assert not pid_file.exists(), "PID file pointing to non-gateway should be removed"


def test_write_and_remove_pid_file(tmp_path: Path):
    """Test basic write and remove operations."""
    from gateway.status import write_pid_file, remove_pid_file, _get_pid_path
    
    with mock.patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        write_pid_file()
        pid_path = _get_pid_path()
        assert pid_path.exists()
        assert pid_path.read_text() == str(os.getpid())
        
        remove_pid_file()
        assert not pid_path.exists()
