"""Command execution abstractions for safe, testable probes."""

from __future__ import annotations

import subprocess
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        timeout: float,
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Run ``argv`` and return a result without raising for process failure."""


class SubprocessCommandRunner:
    """Real subprocess-backed command runner used by the doctor."""

    def __init__(self, *, env: Mapping[str, str] | None = None):
        self.env = dict(env) if env is not None else None

    def run(
        self,
        argv: Sequence[str],
        timeout: float,
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        process_env: Mapping[str, str] | None
        if self.env is None:
            if env is None:
                process_env = None
            else:
                merged = os.environ.copy()
                merged.update(dict(env))
                process_env = merged
        else:
            merged = dict(self.env)
            if env is not None:
                merged.update(dict(env))
            process_env = merged
        try:
            result = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
                input=input_text,
                env=process_env,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            if not stderr:
                stderr = f"timed out after {timeout}s"
            return CommandResult(-1, stdout, stderr, timed_out=True)
        except OSError as exc:
            return CommandResult(127, "", str(exc), timed_out=False)
        return CommandResult(result.returncode, result.stdout or "", result.stderr or "", timed_out=False)


class FakeCommandRunner:
    """Deterministic command runner for tests."""

    def __init__(self, results: dict[tuple[str, ...], CommandResult | Exception]):
        self.results = dict(results)
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        timeout: float,
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        key = tuple(str(part) for part in argv)
        self.calls.append(key)
        result = self.results.get(key)
        if isinstance(result, Exception):
            raise result
        if result is not None:
            return result
        return CommandResult(127, "", f"command not programmed: {' '.join(key)}")
