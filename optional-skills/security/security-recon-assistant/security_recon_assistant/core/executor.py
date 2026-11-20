from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass
class ExecutionResult:
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration: float
    timeout: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timeout


class Executor:
    def __init__(self, default_timeout: int = 300, workers: int = 1):
        self.default_timeout = default_timeout
        self.workers = max(1, int(workers))

    def run(self, cmd: str | Sequence[str], timeout: Optional[int] = None) -> ExecutionResult:
        timeout = timeout if timeout is not None else self.default_timeout
        args = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            duration = time.perf_counter() - start
            return ExecutionResult(
                exit_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                duration=duration,
                timeout=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ExecutionResult(
                exit_code=None,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                duration=duration,
                timeout=True,
            )


def run(cmd: str, timeout: int = 300) -> Tuple[int, str, str]:
    result = Executor(default_timeout=timeout).run(cmd, timeout=timeout)
    return (result.exit_code if result.exit_code is not None else -1, result.stdout, result.stderr)
