"""SSH 远程命令执行器 — 封装 subprocess 调用 ssh。"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class SSHExecutor:
    """通过 SSH 在远程节点执行命令。

    用法:
        ssh = SSHExecutor("10.0.0.1", user="app", key_path="~/.ssh/id_rsa")
        result = ssh.run("jps -l | grep catalina", timeout=15)
        print(result.stdout)
    """

    def __init__(
        self,
        host: str,
        user: str = "root",
        key_path: Optional[str] = None,
        port: int = 22,
    ):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.port = port

    def run(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """在远程节点执行命令，返回 CompletedProcess。"""
        ssh_cmd = self._build_ssh_cmd()
        ssh_cmd.append(cmd)
        return subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout
        )

    def _build_ssh_cmd(self) -> list:
        """构建 ssh 命令前缀。"""
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
        ]
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        cmd.extend(["-p", str(self.port), f"{self.user}@{self.host}"])
        return cmd

    def __repr__(self) -> str:
        return f"SSHExecutor({self.user}@{self.host}:{self.port})"


class LocalExecutor:
    """本地命令执行器，与 SSHExecutor 接口一致。

    当 target.host 为 localhost 或未指定时使用。
    """

    def __init__(self):
        pass

    def run(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )

    def __repr__(self) -> str:
        return "LocalExecutor()"


def create_executor(target: dict):
    """根据 target 配置创建执行器。有 host 且非 localhost 时用 SSH，否则用本地。

    Args:
        target: 节点配置字典，包含 host, ssh_user, ssh_key, ssh_port 等字段。

    Returns:
        SSHExecutor 或 LocalExecutor 实例。
    """
    host = target.get("host", "localhost")
    if host and host != "localhost" and host != "127.0.0.1":
        return SSHExecutor(
            host=host,
            user=target.get("ssh_user", "root"),
            key_path=target.get("ssh_key"),
            port=target.get("ssh_port", 22),
        )
    return LocalExecutor()
