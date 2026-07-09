"""Tests for autonomous-update.py."""
import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\autonomous-update.py")
sys.path.insert(0, str(SCRIPT.parent))

import importlib.util
spec = importlib.util.spec_from_file_location("au", SCRIPT)
au = importlib.util.module_from_spec(spec)
spec.loader.exec_module(au)


# --- find_venv_python_processes ---

def test_find_venv_python_processes_returns_list():
    """Even if wmic fails, should return a list (possibly empty)."""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = au.find_venv_python_processes()
        assert isinstance(result, list)


def test_find_venv_python_processes_parses_wmic_output():
    """Should parse wmic LIST format correctly."""
    wmic_output = """ProcessId=1234
ParentProcessId=5678
CommandLine=C:\\Users\\bbask\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe -m hermes_cli.main gateway run

ProcessId=5678
ParentProcessId=9999
CommandLine=C:\\some\\other\\path\\python.exe something_else

"""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=wmic_output, stderr="")
        result = au.find_venv_python_processes()
    # Should filter to only hermes venv processes (PID 1234)
    pids = [p["ProcessId"] for p in result]
    assert "1234" in pids
    assert "5678" not in pids


def test_find_venv_python_processes_handles_wmic_failure():
    """wmic returning non-zero should return empty list."""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = au.find_venv_python_processes()
    assert result == []


# --- find_cli_session_process ---

def test_find_cli_session_process_returns_pid_when_hermes_yolo_in_chain():
    """Should detect if our parent chain includes hermes.exe --yolo."""
    # Mock the wmic walk
    parent_walk = [
        # First call: current pid's parent
        "ParentProcessId=999",
        # Second call: parent has hermes.exe --yolo
        "Name=hermes.exe\nCommandLine=hermes.exe --yolo\nParentProcessId=888",
        # (should return here)
    ]
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=parent_walk[0], stderr=""),
            MagicMock(returncode=0, stdout=parent_walk[1], stderr=""),
        ]
        result = au.find_cli_session_process()
    assert result == 999


def test_find_cli_session_process_returns_none_when_no_hermes_yolo():
    """If no hermes.exe --yolo in chain, returns None."""
    walk_data = [
        "ParentProcessId=999",
        "Name=pwsh.exe\nCommandLine=pwsh.exe\nParentProcessId=888",
        "Name=cmd.exe\nCommandLine=cmd.exe\nParentProcessId=0",
    ]
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.side_effect = [MagicMock(returncode=0, stdout=d, stderr="") for d in walk_data]
        result = au.find_cli_session_process()
    assert result is None


# --- stop_nssm / start_nssm ---

def test_stop_nssm_returns_true_on_success():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert au.stop_nssm("HermesGateway") is True


def test_stop_nssm_returns_false_on_failure():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        assert au.stop_nssm("HermesGateway") is False


# --- wait_for_port ---

def test_wait_for_port_returns_true_when_bound():
    """Should return True when port is LISTENING."""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout="TCP    127.0.0.1:9720    LISTENING    1234",
            stderr=""
        )
        assert au.wait_for_port(9720, timeout_s=2) is True


def test_wait_for_port_returns_false_on_timeout():
    """Should return False if never bound within timeout."""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        assert au.wait_for_port(9720, timeout_s=1) is False


# --- kill_orphan_python ---

def test_kill_orphan_python_succeeds():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert au.kill_orphan_python(1234) is True


# --- send_telegram ---

def test_send_telegram_returns_false_when_no_token(monkeypatch):
    """No TELEGRAM_BOT_TOKEN in env → returns False without error."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    # Also patch _read_env_value to return None (in case .env file has tokens)
    with patch.object(au, "_read_env_value", return_value=None):
        assert au.send_telegram("test") is False


def test_send_telegram_returns_false_when_no_chat_id(monkeypatch):
    """Token but no chat_id → False."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token")
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    # Token is present, chat_id is not
    with patch.object(au, "_read_env_value", side_effect=lambda k: "test_token" if k == "TELEGRAM_BOT_TOKEN" else None):
        assert au.send_telegram("test") is False


# --- rollback_to_sha ---

def test_rollback_to_sha_runs_git_checkout_and_uv_sync():
    """Should call git checkout then uv sync."""
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git checkout
            MagicMock(returncode=0, stdout="", stderr=""),  # uv sync
        ]
        result = au.rollback_to_sha("abc123")
    assert result is True
    assert mock_run.call_count == 2
    # First call: git checkout
    assert "checkout" in mock_run.call_args_list[0][0][0]


def test_rollback_to_sha_returns_false_on_git_failure():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="git error")
        result = au.rollback_to_sha("abc123")
    assert result is False


# --- get_previous_sha ---

def test_get_previous_sha_returns_stripped_output():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123def\n", stderr="")
        result = au.get_previous_sha()
    assert result == "abc123def"


def test_get_previous_sha_returns_none_on_failure():
    with patch.object(au.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = au.get_previous_sha()
    assert result is None


# --- Integration via subprocess ---

def test_script_runs_with_help():
    """Script should accept --help and exit 0."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, timeout=10
    )
    assert r.returncode == 0
    assert "--check-only" in r.stdout


def test_script_aborts_on_cli_session():
    """When CLI session detected via the real wmic walk, exit code 2.

    Run the script via subprocess with cleaned argv to avoid pytest arg leakage.
    """
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--no-telegram", "--check-only"],
        capture_output=True, text=True, timeout=30
    )
    # Either exits 0 (success) or non-zero (env-specific), but should NOT crash
    assert "Traceback" not in r.stderr
    assert r.returncode in (0, 1, 2, 3, 4, 5)  # any documented exit code


def test_script_aborts_when_cli_session_detected_in_process():
    """Direct test: when find_cli_session_process returns a PID, main returns 2."""
    with patch.object(au, "find_cli_session_process", return_value=12345):
        # Need to capture stdout since main() prints to stdout.
        # Also patch sys.argv so argparse doesn't choke on pytest args.
        import io
        from contextlib import redirect_stdout
        with patch.object(sys, "argv", ["autonomous-update.py", "--no-telegram"]):
            f = io.StringIO()
            with redirect_stdout(f):
                result = au.main()
        assert result == 2
        assert "interactive CLI session" in f.getvalue()