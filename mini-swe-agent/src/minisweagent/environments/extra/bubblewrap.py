"""
[Bubblewrap](https://github.com/containers/bubblewrap) is a low-level, unprivileged sandboxing tool for Linux that enables running applications
in isolated environments with restricted access to the operating system and user data.
This environment uses bubblewrap to execute commands in a sandboxed environment.

!!! warning
    This environment is experimental.

!!! warning
    This environment is not supported on Windows.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class BubblewrapEnvironmentConfig(BaseModel):
    cwd: str = ""
    """Working directory for the sandbox."""
    env: dict[str, str] = {}
    """Dictionary of environment variables to set in the sandbox."""
    timeout: int = 30
    """Timeout for the command in seconds."""
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    """Path to the bubblewrap executable."""
    wrapper_args: list[str] = [
        "--unshare-user-try",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--new-session",
        "--setenv",
        "PATH",
        "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
    ]
    """Arguments to pass to the bubblewrap executable."""


class BubblewrapEnvironment:
    def __init__(
        self, *, config_class: type = BubblewrapEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """This class executes bash commands in a bubblewrap environment and a separate working
        directory for each environment. See `BubblewrapEnvironmentConfig` for kwargs.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.working_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
        self.working_dir.mkdir(parents=True)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the bubblewrap environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or str(self.working_dir)

        cmd = [self.config.executable] + self.config.wrapper_args + ["--bind", cwd, cwd, "--chdir", cwd]

        # Add environment variables
        for key, value in self.config.env.items():
            cmd.extend(["--setenv", key, value])

        cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

    def __del__(self):
        """Cleanup working_dir when object is destroyed."""
        self.cleanup()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
