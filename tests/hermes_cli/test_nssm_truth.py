"""Tests for nssm_truth.is_hermes_service_actually_running.

Coverage:
- test_returns_true_for_live_gateway: PID file has live python with hermes_cli/main + gateway
- test_returns_false_when_pid_dead: PID file has dead PID
- test_handles_missing_lock_file: no lock file, falls back to nssm
- test_handles_malformed_lock_file: lock file is invalid JSON
"""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the helper module
import sys
HELPERS_DIR = Path(__file__).resolve().parent.parent.parent / "hermes_cli"
sys.path.insert(0, str(HELPERS_DIR))

from nssm_truth import (  # noqa: E402
    is_hermes_service_actually_running,
    is_pid_alive,
    _lock_file_indicates_live_gateway,
    _read_lock_file,
)


# --- Fixtures ---

@pytest.fixture
def tmp_hermes_home(tmp_path: Path) -> Path:
    """Create a minimal hermes home dir for tests."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    # Mark it as a valid home by adding a memories/ subdir
    (home / "memories").mkdir()
    return home


@pytest.fixture
def live_pid() -> int:
    """Return the PID of the current pytest process (definitely alive)."""
    return os.getpid()


@pytest.fixture
def fake_hermes_argv() -> list[str]:
    """An argv list that matches the hermes-cli gateway heuristic."""
    return [
        r"C:\Users\bbask\AppData\Local\hermes\hermes-agent\hermes_cli\main.py",
        "gateway",
        "run",
        "--replace",
    ]


# --- Unit tests: is_pid_alive ---

def test_is_pid_alive_returns_true_for_current_process():
    """The pytest process is definitely alive."""
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_returns_false_for_zero():
    assert is_pid_alive(0) is False


def test_is_pid_alive_returns_false_for_negative():
    assert is_pid_alive(-1) is False


def test_is_pid_alive_returns_false_for_nonexistent_pid():
    """Pick a PID that almost certainly doesn't exist."""
    assert is_pid_alive(999_999_999) is False


# --- Unit tests: _lock_file_indicates_live_gateway ---

def test_lock_file_live_gateway(tmp_hermes_home: Path, live_pid: int, fake_hermes_argv: list[str]):
    data = {"pid": live_pid, "kind": "hermes-gateway", "argv": fake_hermes_argv}
    is_live, detail = _lock_file_indicates_live_gateway(data)
    assert is_live is True
    assert str(live_pid) in detail
    assert "live gateway" in detail


def test_lock_file_dead_pid(tmp_hermes_home: Path, fake_hermes_argv: list[str]):
    """Use a PID that almost certainly doesn't exist."""
    data = {"pid": 999_999_999, "kind": "hermes-gateway", "argv": fake_hermes_argv}
    is_live, detail = _lock_file_indicates_live_gateway(data)
    assert is_live is False
    assert "not alive" in detail


def test_lock_file_missing_pid(tmp_hermes_home: Path):
    data = {"kind": "hermes-gateway", "argv": ["foo"]}
    is_live, detail = _lock_file_indicates_live_gateway(data)
    assert is_live is False
    assert "pid" in detail.lower()


def test_lock_file_non_hermes_argv(tmp_hermes_home: Path, live_pid: int):
    """PID alive but argv doesn't match hermes gateway."""
    data = {"pid": live_pid, "kind": "other", "argv": ["python", "main.py"]}
    is_live, detail = _lock_file_indicates_live_gateway(data)
    assert is_live is False
    assert "gateway" in detail or "argv" in detail.lower()


def test_lock_file_no_gateway_command(tmp_hermes_home: Path, live_pid: int):
    """PID alive, argv has hermes_cli/main but not 'gateway' command."""
    argv = [
        r"C:\Users\bbask\AppData\Local\hermes\hermes-cli\main.py",
        "dashboard",
        "--port", "9720",
    ]
    data = {"pid": live_pid, "kind": "hermes-gateway", "argv": argv}
    is_live, detail = _lock_file_indicates_live_gateway(data)
    assert is_live is False


# --- Unit tests: _read_lock_file ---

def test_read_lock_file_prefers_gateway_lock(tmp_hermes_home: Path):
    """If both gateway.lock and gateway.pid exist, gateway.lock wins."""
    (tmp_hermes_home / "gateway.lock").write_text(
        json.dumps({"pid": 111, "argv": ["a"]}), encoding="utf-8"
    )
    (tmp_hermes_home / "gateway.pid").write_text(
        json.dumps({"pid": 222, "argv": ["b"]}), encoding="utf-8"
    )
    data = _read_lock_file(tmp_hermes_home)
    assert data["pid"] == 111


def test_read_lock_file_falls_back_to_pid(tmp_hermes_home: Path):
    """If only gateway.pid exists, use it."""
    (tmp_hermes_home / "gateway.pid").write_text(
        json.dumps({"pid": 333, "argv": ["c"]}), encoding="utf-8"
    )
    data = _read_lock_file(tmp_hermes_home)
    assert data["pid"] == 333


def test_read_lock_file_returns_none_when_missing(tmp_hermes_home: Path):
    assert _read_lock_file(tmp_hermes_home) is None


def test_read_lock_file_handles_malformed_json(tmp_hermes_home: Path):
    """Malformed JSON should not crash — return None."""
    (tmp_hermes_home / "gateway.lock").write_text("{this is not json", encoding="utf-8")
    assert _read_lock_file(tmp_hermes_home) is None


# --- Integration tests: is_hermes_service_actually_running ---

def test_returns_true_for_live_gateway(tmp_hermes_home: Path, live_pid: int, fake_hermes_argv: list[str]):
    """Happy path: PID file has a live hermes gateway python."""
    (tmp_hermes_home / "gateway.lock").write_text(
        json.dumps({"pid": live_pid, "kind": "hermes-gateway", "argv": fake_hermes_argv}),
        encoding="utf-8",
    )
    ok, detail = is_hermes_service_actually_running(
        "HermesGateway",
        hermes_home=tmp_hermes_home,
    )
    assert ok is True
    assert "live gateway" in detail
    assert str(live_pid) in detail


def test_returns_false_when_lock_dead_pid(tmp_hermes_home: Path, fake_hermes_argv: list[str]):
    """Lock file has dead PID — fall back to nssm. Mock nssm to return STOPPED."""
    (tmp_hermes_home / "gateway.lock").write_text(
        json.dumps({"pid": 999_999_999, "argv": fake_hermes_argv}),
        encoding="utf-8",
    )
    # Mock subprocess.run for the nssm call
    mock_result = MagicMock()
    mock_result.stdout = "HermesGateway:\n  STATE: SERVICE_STOPPED\n"
    mock_result.returncode = 0
    with patch("nssm_truth.subprocess.run", return_value=mock_result):
        ok, detail = is_hermes_service_actually_running(
            "HermesGateway",
            hermes_home=tmp_hermes_home,
        )
    assert ok is False
    assert "STOPPED" in detail


def test_handles_missing_lock_file(tmp_hermes_home: Path):
    """No lock file → fall back to nssm. Mock nssm to return RUNNING."""
    mock_result = MagicMock()
    mock_result.stdout = "HermesGateway:\n  STATE: SERVICE_RUNNING\n"
    mock_result.returncode = 0
    with patch("nssm_truth.subprocess.run", return_value=mock_result):
        ok, detail = is_hermes_service_actually_running(
            "HermesGateway",
            hermes_home=tmp_hermes_home,
        )
    assert ok is True
    assert "RUNNING" in detail


def test_handles_malformed_lock_file(tmp_hermes_home: Path):
    """Malformed lock file → falls through to nssm check."""
    (tmp_hermes_home / "gateway.lock").write_text("{not json", encoding="utf-8")
    mock_result = MagicMock()
    mock_result.stdout = "HermesGateway:\n  STATE: SERVICE_PAUSED\n"
    mock_result.returncode = 0
    with patch("nssm_truth.subprocess.run", return_value=mock_result):
        ok, detail = is_hermes_service_actually_running(
            "HermesGateway",
            hermes_home=tmp_hermes_home,
        )
    assert ok is False
    assert "PAUSED" in detail


def test_handles_nssm_not_in_path(tmp_hermes_home: Path):
    """nssm binary missing → graceful fallback."""
    with patch("nssm_truth.subprocess.run", side_effect=FileNotFoundError):
        ok, detail = is_hermes_service_actually_running(
            "HermesGateway",
            hermes_home=tmp_hermes_home,
        )
    assert ok is False
    assert "nssm not in PATH" in detail


def test_pid_file_takes_precedence_over_nssm(tmp_hermes_home: Path, live_pid: int, fake_hermes_argv: list[str]):
    """If PID file shows live gateway, nssm state is irrelevant."""
    (tmp_hermes_home / "gateway.lock").write_text(
        json.dumps({"pid": live_pid, "argv": fake_hermes_argv}),
        encoding="utf-8",
    )
    # Mock nssm to say PAUSED — but PID file should win
    mock_result = MagicMock()
    mock_result.stdout = "HermesGateway:\n  STATE: SERVICE_PAUSED\n"
    mock_result.returncode = 0
    with patch("nssm_truth.subprocess.run", return_value=mock_result) as mock_run:
        ok, detail = is_hermes_service_actually_running(
            "HermesGateway",
            hermes_home=tmp_hermes_home,
        )
    assert ok is True
    # nssm should NOT have been called
    mock_run.assert_not_called()


def test_can_disable_nssm_fallback(tmp_hermes_home: Path):
    """With fallback_to_nssm=False, no subprocess call is made."""
    ok, detail = is_hermes_service_actually_running(
        "HermesGateway",
        hermes_home=tmp_hermes_home,
        check_pid_file=False,
        fallback_to_nssm=False,
    )
    assert ok is False
    assert "no PID file" in detail or "nssm check disabled" in detail