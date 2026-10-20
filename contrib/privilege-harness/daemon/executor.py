"""
Executor — Command executor
=====================

Execute approved privileged commands as root。

Security：
- Auto-kill on timeout
- stdout/stderr size limits
"""

import logging
import os
import re
import signal
import subprocess
import time
from typing import Optional

logger = logging.getLogger("vipd.executor")


class Executor:
    """Command executor"""

    def __init__(self, timeout: int = 300, max_stdout: int = 50000):
        self._timeout = timeout
        self._max_stdout = max_stdout

    def execute(self, command: str, timeout: Optional[int] = None,
                env: Optional[dict] = None) -> dict:
        """
        Execute command。

        Args:
            command: shell 命令字符串
            timeout: 超时秒数（默认 self._timeout）
            env: 额外环境变量

        Returns:
            {stdout, stderr, exit_code, executed_at, duration_ms}
        """
        start = time.time()

        result = {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "executed_at": start,
            "duration_ms": 0,
        }

        actual_timeout = timeout or self._timeout

        try:
            # Strip any leading "sudo" from the command to avoid nested sudo
            clean_cmd = re.sub(r"^\s*sudo\s+", "", command, count=1, flags=re.IGNORECASE)
            proc = subprocess.Popen(
                ["/bin/sh", "-c", f"sudo {clean_cmd}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **(env or {})},
                preexec_fn=lambda: os.setsid(),  # 独立进程组，方便 kill 子树
            )

            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    timeout=actual_timeout
                )
            except subprocess.TimeoutExpired:
                # 超时：kill 整个进程组
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=2)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
                result["stderr"] = f"command timed out（{actual_timeout}s）"
                result["exit_code"] = -1
                end = time.time()
                result["duration_ms"] = int((end - start) * 1000)
                return result

            # Truncate output
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            if len(stdout_str) > self._max_stdout:
                stdout_str = stdout_str[:self._max_stdout] + "\n... (truncated)"
            if len(stderr_str) > self._max_stdout:
                stderr_str = stderr_str[:self._max_stdout] + "\n... (truncated)"

            result["stdout"] = stdout_str
            result["stderr"] = stderr_str
            result["exit_code"] = proc.returncode

        except FileNotFoundError:
            result["stderr"] = f"command not found：{command}"
            result["exit_code"] = 127
        except PermissionError:
            result["stderr"] = f"permission denied：{command}"
            result["exit_code"] = 126
        except Exception as exc:
            result["stderr"] = f"execution error：{exc}"
            result["exit_code"] = -1

        end = time.time()
        result["duration_ms"] = int((end - start) * 1000)
        logger.info("exec  exit_code=%d duration=%dms command=%s",
                     result["exit_code"], result["duration_ms"],
                     command[:80])

        return result
