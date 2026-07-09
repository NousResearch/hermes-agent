"""Tests for resource-dashboard.py."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\resource-dashboard.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("rd", SCRIPT)
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


# --- check_disk ---

def test_check_disk_ok(tmp_path):
    """Returns tuple of (total, used, free) in bytes."""
    total_bytes = 1000 * 1024**3
    free_bytes = 500 * 1024**3
    with patch.object(rd.shutil, "disk_usage", return_value=(total_bytes, total_bytes - free_bytes, free_bytes)):
        ok, msg = rd.check_disk()
    assert ok is True
    assert "50.0%" in msg


def test_check_disk_low(tmp_path):
    total_bytes = 1000 * 1024**3
    free_bytes = 50 * 1024**3  # 5% free
    with patch.object(rd.shutil, "disk_usage", return_value=(total_bytes, total_bytes - free_bytes, free_bytes)):
        ok, msg = rd.check_disk()
    assert ok is False
    assert "5.0%" in msg


def test_check_disk_failure():
    with patch.object(rd.shutil, "disk_usage", side_effect=OSError("no such dir")):
        ok, msg = rd.check_disk()
    assert ok is False
    assert "failed" in msg


# --- check_state_db_size ---

def test_check_state_db_size_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "STATE_DB", tmp_path / "state.db")
    (tmp_path / "state.db").write_bytes(b"x" * (100 * 1024 * 1024))  # 100 MB
    ok, msg = rd.check_state_db_size()
    assert ok is True
    assert "100" in msg


def test_check_state_db_size_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "STATE_DB", tmp_path / "state.db")
    (tmp_path / "state.db").write_bytes(b"x" * (600 * 1024 * 1024))  # 600 MB
    ok, msg = rd.check_state_db_size()
    assert ok is False
    assert "600" in msg


def test_check_state_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "STATE_DB", tmp_path / "missing.db")
    ok, msg = rd.check_state_db_size()
    assert ok is True
    assert "not found" in msg


# --- check_memory_md ---

def test_check_memory_md_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "HERMES_HOME", tmp_path)
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "MEMORY.md").write_text("a" * 3000)
    ok, msg = rd.check_memory_md()
    assert ok is True
    assert "50%" in msg


def test_check_memory_md_critical(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "HERMES_HOME", tmp_path)
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "MEMORY.md").write_text("a" * 5800)
    ok, msg = rd.check_memory_md()
    assert ok is False
    assert "96%" in msg or "97%" in msg


def test_check_memory_md_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "HERMES_HOME", tmp_path)
    ok, msg = rd.check_memory_md()
    assert ok is True
    assert "not found" in msg


# --- check_gateway ---

def test_check_gateway_200():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch.object(rd.urllib.request, "urlopen", return_value=mock_resp):
        ok, msg = rd.check_gateway()
    assert ok is True
    assert "200" in msg


def test_check_gateway_down():
    with patch.object(rd.urllib.request, "urlopen", side_effect=ConnectionError("refused")):
        ok, msg = rd.check_gateway()
    assert ok is False
    assert "down" in msg


# --- check_honcho ---

def test_check_honcho_running():
    with patch.object(rd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="honcho-api-1\nhoncho-deriver-1\nhoncho-database-1\n",
            stderr="",
        )
        ok, msg = rd.check_honcho()
    assert ok is True
    assert "3 container" in msg


def test_check_honcho_no_containers():
    with patch.object(rd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="\n", stderr="")
        ok, msg = rd.check_honcho()
    assert ok is False
    assert "no containers" in msg


def test_check_honcho_docker_error():
    with patch.object(rd.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="docker not running")
        ok, msg = rd.check_honcho()
    assert ok is False
    assert "docker error" in msg


def test_check_honcho_no_docker_binary():
    with patch.object(rd.subprocess, "run", side_effect=FileNotFoundError):
        ok, msg = rd.check_honcho()
    assert ok is True
    assert "docker not in PATH" in msg


# --- main flow ---

def test_main_all_green_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(rd, "STATE_DB", tmp_path / "missing.db")
    monkeypatch.setattr(rd, "HERMES_HOME", tmp_path)
    total_bytes = 1000 * 1024**3
    free_bytes = 500 * 1024**3
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch.object(rd.shutil, "disk_usage", return_value=(total_bytes, total_bytes - free_bytes, free_bytes)):
        with patch.object(rd.urllib.request, "urlopen", return_value=mock_resp):
            with patch.object(rd, "send_telegram", return_value=True):
                r = rd.main()
    assert r == 0
    log_text = (tmp_path / "log.txt").read_text()
    assert "all green" in log_text


def test_main_with_alerts_sends_telegram(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(rd, "STATE_DB", tmp_path / "missing.db")
    monkeypatch.setattr(rd, "HERMES_HOME", tmp_path)
    total_bytes = 1000 * 1024**3
    free_bytes = 5 * 1024**3  # 0.5% free — alert
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch.object(rd.shutil, "disk_usage", return_value=(total_bytes, total_bytes - free_bytes, free_bytes)):
        with patch.object(rd.urllib.request, "urlopen", return_value=mock_resp):
            with patch.object(rd, "send_telegram", return_value=True) as mock_tg:
                r = rd.main()
    assert r == 1
    assert mock_tg.called