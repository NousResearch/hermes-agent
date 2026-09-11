"""Explicit uv construction, independent of PM tool discovery and live selection.

Callers own the destination, source/workspace, cache and publication lifecycle.
This module neither discovers profiles nor modifies the process environment.
"""
from __future__ import annotations

import codecs
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import TextIO

from pm.package import InstallError


def _run_streaming(command: list[str], *, cwd: Path, env: dict[str, str],
                   timeout: int, output: TextIO) -> subprocess.CompletedProcess:
    """Keep CI progress live, a bounded diagnostic tail, and a wall-clock timeout."""
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=0)
    pipe = proc.stdout
    assert isinstance(pipe, io.TextIOWrapper)  # Popen was given stdout=PIPE and text=True.
    tail = ""
    try:
        # A descendant can keep stdout open after proc exits. Nonblocking reads
        # bound that drain without leaving a thread stuck in readline()/close().
        # PM's Python >=3.14 supports nonblocking pipes on Windows as well as POSIX.
        os.set_blocking(pipe.fileno(), False)
        decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder(pipe.encoding)(errors="replace"), translate=True,
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout, stderr=tail)
            try:
                data = os.read(pipe.fileno(), 65536)
            except BlockingIOError:
                time.sleep(min(.05, remaining))
                continue
            text = decoder.decode(data, final=not data)
            if text:
                tail = (tail + text)[-2000:]
                output.write(text)
                output.flush()
            if not data:
                break
        # EOF can precede process exit; it does not grant another timeout budget.
        try:
            code = proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(command, timeout, stderr=tail) from None
    except BaseException:
        proc.kill()
        proc.wait(timeout=5)
        raise
    finally:
        pipe.close()
    return subprocess.CompletedProcess(command, code, "", tail)


@dataclass(frozen=True, kw_only=True)
class PythonEnvironment:
    uv: Path
    python: Path
    destination: Path
    cache: Path
    env: Mapping[str, str]
    offline: bool = False
    output: TextIO | None = None

    @property
    def executable(self) -> Path:
        return self.destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _run(self, args: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
        # The caller supplies policy (PM already sanitizes it via pm.ensure.uv).
        # Preserve intentional index credentials/settings; only explicit routing
        # inputs and project-config visibility override this child environment.
        env = {key: value for key, value in self.env.items() if key not in (
            "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
            "UV_WORKING_DIR", "UV_PROJECT", "UV_CONFIG_FILE", "UV_NO_CONFIG",
            "UV_MANAGED_PYTHON", "UV_NO_MANAGED_PYTHON", "UV_SYSTEM_PYTHON",
        )}
        env.update(UV_PYTHON=str(self.python), UV_PROJECT_ENVIRONMENT=str(self.destination),
                   UV_CACHE_DIR=str(self.cache), UV_PYTHON_DOWNLOADS="never")
        with tempfile.TemporaryDirectory(prefix="pm-uv-config-") as config:
            env.update(XDG_CONFIG_HOME=config, XDG_CONFIG_DIRS=config)
            command = [str(self.uv), *args]
            if self.offline:
                command.append("--offline")
            if self.output is not None:
                return _run_streaming(command, cwd=cwd, env=env, timeout=timeout, output=self.output)
            return subprocess.run(command, cwd=str(cwd), env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)

    def create(self) -> None:
        """Create at the final destination; callers must not move a live venv."""
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        result = self._run(
            ["venv", "--relocatable", "--no-project", "--no-config",
             "--python", str(self.python), str(self.destination)],
            cwd=self.destination.parent, timeout=120,
        )
        if result.returncode:
            raise InstallError("venv", f"uv venv failed: {result.stderr[-600:]}")

    def sync(self, source: Path, *, extras: Sequence[str] = (), frozen: bool = True,
             all_extras: bool = False, no_install_project: bool = False) -> None:
        """Install the root and every member; resolve only in a writable workspace.

        ``frozen=False`` is reserved for the caller-owned generated workspace,
        never the original project's lock. Seed/replay policy belongs to PM.
        """
        from pm.workspace import classify_uv_failure

        if not frozen:
            locked = self._run(["lock", "--python", str(self.python)], cwd=source, timeout=1800)
            if locked.returncode:
                raise classify_uv_failure("lock", locked.returncode, locked.stderr or locked.stdout)
        # Locking members alone is insufficient: plain sync only installs root deps.
        command = ["sync", "--frozen", "--all-packages", "--python", str(self.python)]
        if all_extras:
            command.append("--all-extras")
        if no_install_project:
            # --all-packages has no single selected project in uv, so
            # --no-install-project alone does not exclude the root. Name it
            # explicitly without dropping member dependencies or installations.
            import tomllib

            project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8-sig"))
            command += ["--no-install-project", "--no-install-package", project["project"]["name"]]
        for extra in sorted(set(extras)):
            command += ["--extra", extra]
        result = self._run(command, cwd=source, timeout=1800)
        if result.returncode:
            raise classify_uv_failure("sync", result.returncode, result.stderr or result.stdout)

    def check(self) -> None:
        result = self._run(
            ["pip", "check", "--no-config", "--python", str(self.executable)],
            cwd=self.destination.parent, timeout=60,
        )
        if result.returncode:
            raise InstallError("venv", f"dependency validation failed: {result.stderr[-600:]}")
