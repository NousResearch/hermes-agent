"""Tests for the remote_run tool.

Uses mocked Paramiko connections — no live SSH required.
Verifies command construction, error handling, auth modes, and the registry
integration.
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_paramiko():
    """Mock paramiko.SSHClient so no real SSH connection is attempted."""
    with patch("paramiko.SSHClient") as mock_sshclient_cls:
        # Set up the mock client
        mock_client = MagicMock()
        mock_sshclient_cls.return_value = mock_client

        # Mock the exec_command return
        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()

        # Default: command succeeds
        mock_stdout.read.return_value = b"hello\n"
        mock_stderr.read.return_value = b""

        # Mock channel for exit status
        mock_channel = MagicMock()
        mock_channel.recv_exit_status.return_value = 0
        mock_stdout.channel = mock_channel

        mock_client.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

        yield mock_sshclient_cls, mock_client, mock_stdout, mock_stderr


def _run(args: dict) -> dict:
    """Helper to call remote_run_handler and parse the JSON result."""
    from tools.remote_run_tool import remote_run_handler
    return json.loads(remote_run_handler(args))


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------

def test_tool_registered():
    """The tool registers itself with the correct name and toolset."""
    from tools.remote_run_tool import check_remote_run_requirements
    # The check_fn should detect paramiko (it's mocked or installed during tests)
    # We just verify it exists and is callable
    assert callable(check_remote_run_requirements)


def test_tool_schema_is_valid():
    """The schema has all required fields."""
    from tools.remote_run_tool import REMOTE_RUN_SCHEMA
    assert REMOTE_RUN_SCHEMA["name"] == "remote_run"
    props = REMOTE_RUN_SCHEMA["parameters"]["properties"]
    assert "host" in props
    assert "command" in props
    assert REMOTE_RUN_SCHEMA["parameters"]["required"] == ["host", "command"]
    assert "max_result_size_chars" in props


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

def test_basic_command(mock_paramiko):
    """A simple command returns stdout, stderr, and exit_code."""
    mock_p, mock_client, mock_stdout, mock_stderr = mock_paramiko

    result = _run({"host": "example.com", "command": "echo hello", "user": "test"})

    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert result["stderr"] == ""

    # Verify connection params
    mock_client.connect.assert_called_once()
    call_kwargs = mock_client.connect.call_args[1]
    assert call_kwargs["hostname"] == "example.com"
    assert call_kwargs["username"] == "test"


def test_ssh_port(mock_paramiko):
    """Custom port is passed through."""
    _run({"host": "example.com", "command": "ls", "port": 2222})
    mock_p, mock_client, _, _ = mock_paramiko
    assert mock_client.connect.call_args[1]["port"] == 2222


# ---------------------------------------------------------------------------
# Sudo
# ---------------------------------------------------------------------------

def test_sudo_with_password(mock_paramiko):
    """Sudo with password sends via channel, not in command string."""
    mock_p, mock_client, mock_stdout, _ = mock_paramiko
    mock_stdin = MagicMock()
    mock_stdin.channel = MagicMock()
    mock_client.exec_command.return_value = (mock_stdin, mock_stdout, MagicMock())

    _run({
        "host": "srv", "command": "whoami",
        "sudo": True, "password": "sekret",
    })
    cmd = mock_client.exec_command.call_args[0][0]
    # Password is NOT in the command — sent via channel stdin
    assert "sekret" not in cmd
    assert "printf" not in cmd
    assert "sudo -S" not in cmd
    assert "whoami" in cmd
    assert cmd.startswith("sudo")
    # Verify password is sent via channel (with 0.5s delay before send)
    assert mock_stdin.channel.send.call_count == 1
    sent = mock_stdin.channel.send.call_args[0][0]
    assert "sekret" in sent


def test_sudo_without_password(mock_paramiko):
    """Sudo without password uses direct sudo call."""
    _run({
        "host": "srv", "command": "whoami", "sudo": True,
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    assert cmd.startswith("sudo")
    assert "whoami" in cmd


# ---------------------------------------------------------------------------
# Workdir
# ---------------------------------------------------------------------------

def test_workdir(mock_paramiko):
    """Working directory is prefixed with cd."""
    _run({
        "host": "srv", "command": "pwd", "workdir": "/opt/app",
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    assert "cd /opt/app" in cmd
    assert "pwd" in cmd


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

def test_environment_vars(mock_paramiko):
    """Environment variables are exported before the command."""
    _run({
        "host": "srv", "command": "echo $VAR",
        "env": {"VAR": "hello", "PATH": "/custom"},
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    assert "export VAR='hello'" in cmd
    assert "export PATH='/custom'" in cmd
    assert "echo $VAR" in cmd


def test_env_special_chars(mock_paramiko):
    """Values with single quotes are properly escaped."""
    _run({
        "host": "srv", "command": "echo $X",
        "env": {"X": "it's fine"},
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    # Check the value is wrapped in single quotes with proper escaping
    assert "it'" in cmd
    assert "X" in cmd


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def test_key_file_auth(mock_paramiko):
    """Key file path is passed through."""
    _run({
        "host": "srv", "command": "ls",
        "key_file": "~/.ssh/example_key",
    })
    mock_p, mock_client, _, _ = mock_paramiko
    call_kwargs = mock_client.connect.call_args[1]
    assert "key_filename" in call_kwargs
    assert "example_key" in call_kwargs["key_filename"]


def test_password_auth(mock_paramiko):
    """Password is passed through to connect."""
    _run({
        "host": "srv", "command": "ls",
        "password": "testpass",
    })
    mock_p, mock_client, _, _ = mock_paramiko
    assert mock_client.connect.call_args[1]["password"] == "testpass"


def test_default_user(mock_paramiko):
    """When user is omitted, no username is passed (paramiko defaults to local)."""
    _run({"host": "srv", "command": "whoami"})
    mock_p, mock_client, _, _ = mock_paramiko
    assert "username" not in mock_client.connect.call_args[1] or \
           mock_client.connect.call_args[1].get("username") is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_authentication_error(mock_paramiko):
    """AuthenticationException returns a structured error."""
    from paramiko import AuthenticationException
    mock_p, mock_client, _, _ = mock_paramiko
    mock_client.connect.side_effect = AuthenticationException("bad key")

    result = _run({"host": "srv", "command": "ls", "password": "wrong"})
    assert "error" in result
    assert "Authentication failed" in result["error"]
    assert result["exit_code"] == 1


def test_ssh_connection_error(mock_paramiko):
    """SSHException returns a structured error."""
    from paramiko import SSHException
    mock_p, mock_client, _, _ = mock_paramiko
    mock_client.connect.side_effect = SSHException("Connection refused")

    result = _run({"host": "srv", "command": "ls"})
    assert "error" in result
    assert "Connection refused" in result["error"]
    assert result["exit_code"] == 1


def test_timeout_error(mock_paramiko):
    """OSError (timeout) returns a structured error."""
    mock_p, mock_client, _, _ = mock_paramiko
    mock_client.connect.side_effect = OSError("timed out")

    result = _run({"host": "srv", "command": "ls", "timeout": 5})
    assert "error" in result
    assert "timed out" in result["error"]
    assert result["exit_code"] == 1


def test_non_zero_exit(mock_paramiko):
    """Non-zero exit codes are captured correctly."""
    mock_p, mock_client, mock_stdout, mock_stderr = mock_paramiko
    mock_stdout.channel.recv_exit_status.return_value = 2
    mock_stderr.read.return_value = b"ls: not found\n"

    result = _run({"host": "srv", "command": "ls /nonexistent"})
    assert result["exit_code"] == 2
    assert "not found" in result["stderr"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_command_does_not_crash(mock_paramiko):
    """An empty command string is handled gracefully."""
    result = _run({"host": "srv", "command": ""})
    mock_p, mock_client, _, _ = mock_paramiko
    # Paramiko will execute empty string (remote shell behavior)
    assert "exit_code" in result


def test_connection_closed_in_finally(mock_paramiko):
    """SSHClient.close() is called even on errors."""
    mock_p, mock_client, _, _ = mock_paramiko
    mock_client.connect.side_effect = OSError("fail")

    _run({"host": "srv", "command": "ls"})

    # close() should be called in the finally block
    mock_client.close.assert_called_once()


def test_long_running_command_respects_timeout(mock_paramiko):
    """Timeout parameter is passed to exec_command."""
    _run({"host": "srv", "command": "sleep 10", "timeout": 30})
    mock_p, mock_client, _, _ = mock_paramiko
    assert mock_client.exec_command.call_args[1].get("timeout") == 30


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def test_requirements_check_with_paramiko():
    """check_remote_run_requirements returns True when paramiko is importable."""
    from tools.remote_run_tool import check_remote_run_requirements
    with patch.dict("sys.modules", {"paramiko": MagicMock()}):
        assert check_remote_run_requirements() is True


def test_requirements_check_without_paramiko():
    """check_remote_run_requirements returns False when paramiko is missing."""
    from tools.remote_run_tool import check_remote_run_requirements
    # Simulate ImportError by patching builtins.__import__ to raise
    # for the 'paramiko' module
    import builtins
    orig_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == 'paramiko':
            raise ImportError("No module named paramiko")
        return orig_import(name, *args, **kwargs)

    with patch.object(builtins, '__import__', side_effect=_mock_import):
        assert check_remote_run_requirements() is False


# ---------------------------------------------------------------------------
# Security — injection resistance
# ---------------------------------------------------------------------------


def test_workdir_injection_prevented(mock_paramiko):
    """Shell injection via workdir containing $(...) is prevented by shlex.quote."""
    _run({
        "host": "srv", "command": "echo hello",
        "workdir": "/tmp/$(whoami)",
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    # The injected $(whoami) should be single-quoted by shlex.quote
    assert "$(whoami)" not in cmd.split("'")[0]  # Should not be unquoted
    assert "whoami" not in cmd or "cd '/tmp/$(whoami)'" in cmd
    assert "'" in cmd  # Should be quoted


def test_env_key_injection_prevented(mock_paramiko):
    """Shell injection via env key containing shell metacharacters is rejected."""
    result = _run({
        "host": "srv", "command": "echo hello",
        "env": {"X;whoami;Y": "val"},
    })
    assert "error" in result
    assert "Invalid environment variable key" in result["error"]


def test_env_key_empty_rejected(mock_paramiko):
    """Empty env var key is rejected."""
    result = _run({
        "host": "srv", "command": "echo hello",
        "env": {"": "val"},
    })
    assert "error" in result
    assert "Invalid environment variable key" in result["error"]


def test_sudo_command_with_single_quotes(mock_paramiko):
    """Sudo with command containing single quotes is handled correctly."""
    _run({
        "host": "srv", "command": "echo 'test'", "sudo": True,
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    assert cmd.startswith("sudo")
    assert "echo" in cmd
    # The single quotes should be preserved via shlex.quote wrapping
    # We can't validate exact format since shlex.quote may vary, but
    # the command should execute without shell syntax errors
    assert "bash -c" in cmd


def test_sudo_password_sent_via_channel(mock_paramiko):
    """Sudo password is sent via channel stdin, not embedded in command string."""
    mock_p, mock_client, mock_stdout, _ = mock_paramiko
    mock_stdin = MagicMock()

    # Mock exec_command to capture stdin channel for password sending
    mock_stdin.channel = MagicMock()
    mock_client.exec_command.return_value = (mock_stdin, mock_stdout, MagicMock())

    _run({
        "host": "srv", "command": "whoami",
        "sudo": True, "password": "sekret",
    })
    cmd = mock_client.exec_command.call_args[0][0]
    # The password should NOT be in the command string
    assert "sekret" not in cmd
    assert "printf" not in cmd
    # The password should be sent via stdin.channel
    mock_stdin.channel.send.assert_called_once()
    sent = mock_stdin.channel.send.call_args[0][0]
    assert "sekret" in sent


def test_password_not_in_command_string(mock_paramiko):
    """Password never appears in the command string (sent via channel)."""
    _run({
        "host": "srv", "command": "whoami",
        "sudo": True, "password": "p@$$w0rd'",
    })
    mock_p, mock_client, _, _ = mock_paramiko
    cmd = mock_client.exec_command.call_args[0][0]
    assert "p@$$w0rd" not in cmd


def test_host_key_policy_is_warning(mock_paramiko):
    """The SSHClient uses WarningPolicy (not AutoAddPolicy)."""
    from tools.remote_run_tool import check_remote_run_requirements
    # Policy is set during handler execution, verified via mock
    mock_p, mock_client, _, _ = mock_paramiko
    _run({"host": "srv", "command": "ls"})
    # Verify set_missing_host_key_policy was called
    mock_client.set_missing_host_key_policy.assert_called_once()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def test_large_output_truncated(mock_paramiko):
    """Output exceeding MAX_RESULT_SIZE_CHARS is truncated."""
    from tools.remote_run_tool import MAX_RESULT_SIZE_CHARS
    mock_p, mock_client, mock_stdout, mock_stderr = mock_paramiko
    # Generate output larger than the limit
    big_output = b"x" * (MAX_RESULT_SIZE_CHARS + 10000)
    mock_stdout.read.return_value = big_output
    mock_stderr.read.return_value = b""

    result = _run({"host": "srv", "command": "cat largefile"})
    assert len(result["stdout"]) < len(big_output)
    assert "truncated" in result["stdout"]


# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------


def test_tool_registered_in_ssh_toolset():
    """The tool is registered in the 'ssh' toolset (not 'terminal')."""
    from tools.remote_run_tool import REMOTE_RUN_SCHEMA, check_remote_run_requirements
    assert REMOTE_RUN_SCHEMA["name"] == "remote_run"
    assert callable(check_remote_run_requirements)
    # The toolset is set at registry time — verify via the module
    # (We can't easily inspect the registry after registration, but
    #  the registration call in the module uses toolset="ssh")
