"""SSH 执行器测试。"""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from scripts.ssh_executor import SSHExecutor, LocalExecutor, create_executor


class TestSSHExecutor:
    """SSHExecutor 测试。"""

    def test_build_cmd_basic(self):
        ssh = SSHExecutor("10.0.0.1", user="app")
        cmd = ssh._build_ssh_cmd()
        assert cmd[0] == "ssh"
        assert "-o" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert "app@10.0.0.1" in cmd
        assert "-p" in cmd
        assert "22" in cmd

    def test_build_cmd_with_key(self):
        ssh = SSHExecutor("10.0.0.1", user="app", key_path="~/.ssh/id_rsa")
        cmd = ssh._build_ssh_cmd()
        assert "-i" in cmd
        idx = cmd.index("-i")
        assert cmd[idx + 1] == "~/.ssh/id_rsa"

    def test_build_cmd_custom_port(self):
        ssh = SSHExecutor("10.0.0.1", user="app", port=2222)
        cmd = ssh._build_ssh_cmd()
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "2222"

    @patch("scripts.ssh_executor.subprocess.run")
    def test_run_calls_subprocess(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        ssh = SSHExecutor("10.0.0.1", user="app")
        result = ssh.run("jps -l", timeout=15)
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["timeout"] == 15
        assert "app@10.0.0.1" in call_args[0][0]
        assert result.stdout == "ok"

    @patch("scripts.ssh_executor.subprocess.run")
    def test_run_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
        ssh = SSHExecutor("10.0.0.1", user="app")
        with pytest.raises(subprocess.TimeoutExpired):
            ssh.run("sleep 100", timeout=30)

    def test_repr(self):
        ssh = SSHExecutor("10.0.0.1", user="app", port=2222)
        assert "app@10.0.0.1:2222" in repr(ssh)


class TestLocalExecutor:
    """LocalExecutor 测试。"""

    @patch("scripts.ssh_executor.subprocess.run")
    def test_run_uses_shell(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        local = LocalExecutor()
        local.run("echo hello", timeout=10)
        call_args = mock_run.call_args
        assert call_args[1]["shell"] is True

    def test_repr(self):
        assert "Local" in repr(LocalExecutor())


class TestCreateExecutor:
    """create_executor 测试。"""

    def test_localhost_returns_local(self):
        assert isinstance(create_executor({"host": "localhost"}), LocalExecutor)

    def test_127_returns_local(self):
        assert isinstance(create_executor({"host": "127.0.0.1"}), LocalExecutor)

    def test_no_host_returns_local(self):
        assert isinstance(create_executor({}), LocalExecutor)

    def test_remote_host_returns_ssh(self):
        target = {"host": "10.0.0.1", "ssh_user": "app", "ssh_key": "~/.ssh/id_rsa"}
        executor = create_executor(target)
        assert isinstance(executor, SSHExecutor)
        assert executor.host == "10.0.0.1"
        assert executor.user == "app"
        assert executor.key_path == "~/.ssh/id_rsa"

    def test_remote_host_default_user(self):
        executor = create_executor({"host": "10.0.0.1"})
        assert isinstance(executor, SSHExecutor)
        assert executor.user == "root"
