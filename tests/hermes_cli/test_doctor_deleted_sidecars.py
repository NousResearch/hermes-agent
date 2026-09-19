import os
import signal
import pytest
from unittest.mock import MagicMock, call

from hermes_cli.doctor import Finding
from hermes_cli.doctor_state import _check_deleted_sidecars
import psutil

def test_check_deleted_sidecars_no_fix_appends_issue(monkeypatch):
    monkeypatch.setattr('hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders', lambda p: [(1234, 'wal')])
    f = Finding()
    _check_deleted_sidecars(f, False, "fake/path/state.db")
    assert any("run 'hermes doctor --fix' to terminate them" in issue for issue in f.issues)

def test_check_deleted_sidecars_with_fix_kills_and_checkpoints(monkeypatch):
    mock_holders = [(1234, 'wal')]
    
    # Mock the iterator to return our mock holder on first call, empty on subsequent
    def iter_mock(path):
        res = list(mock_holders)
        mock_holders.clear()
        return res
        
    monkeypatch.setattr('hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders', iter_mock)
    
    # Mock psutil
    mock_proc = MagicMock()
    mock_proc.name.return_value = 'python'
    mock_proc.cmdline.return_value = ['hermes', 'dashboard']
    monkeypatch.setattr('psutil.Process', lambda pid: mock_proc)
    
    # Mock os.kill
    mock_kill = MagicMock()
    monkeypatch.setattr('os.kill', mock_kill)
    
    # Mock exclusive guard
    mock_guard = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = (mock_guard, None)
    mock_ctx.__exit__.return_value = False
    monkeypatch.setattr('hermes_state_repair._exclusive_repair_db_guard', lambda p: mock_ctx)
    
    f = Finding()
    f.fixed = 0
    _check_deleted_sidecars(f, True, "fake/path/state.db")
    
    # Assertions
    mock_kill.assert_called_once_with(1234, signal.SIGTERM)
    assert f.fixed == 1
    mock_guard.execute.assert_called_once_with("PRAGMA wal_checkpoint(TRUNCATE)")

def test_check_deleted_sidecars_skips_unknown_processes(monkeypatch):
    monkeypatch.setattr('hermes_state_dbfile.iter_deleted_sqlite_sidecar_holders', lambda p: [(5678, 'wal')])
    
    # Mock psutil for an unknown process
    mock_proc = MagicMock()
    mock_proc.name.return_value = 'unknown_process'
    mock_proc.cmdline.return_value = ['/bin/sh']
    monkeypatch.setattr('psutil.Process', lambda pid: mock_proc)
    
    mock_kill = MagicMock()
    monkeypatch.setattr('os.kill', mock_kill)
    
    f = Finding()
    _check_deleted_sidecars(f, True, "fake/path/state.db")
    
    # Should not kill
    mock_kill.assert_not_called()
