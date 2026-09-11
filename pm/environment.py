"""PM's private Python engine: tool routing, isolation and command execution.

Operations own source, destination and publication. Bootstrap may inject its
staged toolchain directly without recursing through the worker it is building.
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


def prune_site_pth(venv_dir: Path) -> None:
    """Drop .pth files that must never execute inside a shipped payload.

    ``uv sync`` leaves two behind: ``_virtualenv.pth`` (repoints
    ``sys.prefix`` at the venv) and the project's ``__editable__`` pointer
    (names the BUILD machine — the payload wires the repo snapshot itself,
    see scripts/build/launcher_wrapper.py). Bundled launchers process every
    other .pth with ``site.addsitedir()``; pywin32.pth is load-bearing on
    Windows (win32\\lib on sys.path is what makes ``import pywintypes``
    resolve, which portalocker/concurrent-log-handler need to write logs).
    """
    if (venv_dir / "Scripts").is_dir():
        sites = [venv_dir / "Lib" / "site-packages"]
    else:
        lib = venv_dir / "lib"
        sites = sorted(lib.glob("python*/site-packages")) if lib.is_dir() else []
    for site_dir in sites:
        if not site_dir.is_dir():
            continue
        for pth in site_dir.glob("*.pth"):
            if pth.name == "_virtualenv.pth" or pth.name.startswith("__editable__"):
                try:
                    pth.unlink()
                except OSError:
                    pass


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


def _base_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Ambient UV settings are never policy; explicit build index settings are."""
    index_settings = {
        "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_NO_INDEX", "UV_FIND_LINKS",
        "UV_INSECURE_HOST", "UV_KEYRING_PROVIDER", "UV_NATIVE_TLS",
    }
    return {key: value for key, value in (os.environ if env is None else env).items()
            if not key.startswith("PYTHON") and key != "VIRTUAL_ENV"
            and (not key.startswith("UV_") or
                 (env is not None and (key.startswith("UV_INDEX") or key in index_settings)))}


def managed_environment(destination: Path, *, python: Path | None = None,
                        cache: Path | None = None, env: Mapping[str, str] | None = None,
                        offline: bool = False, explicit: bool = False,
                        output: TextIO | None = None, realize: bool = True) -> PythonEnvironment:
    from pm._uv import _toolchain
    from pm.packages import uv_cache_dir

    tools = _toolchain(explicit=explicit, realize=realize)
    if tools is None:
        raise InstallError("venv", "PM's pinned toolchain is unavailable")
    uv, pinned_python = tools
    return PythonEnvironment(
        uv=uv, python=pinned_python if python is None else python.absolute(),
        destination=destination.absolute(), cache=uv_cache_dir() if cache is None else cache.absolute(),
        env=_base_environment(env), offline=offline, output=output,
    )


@dataclass(frozen=True, kw_only=True)
class PythonEnvironment:
    uv: Path
    python: Path
    destination: Path
    cache: Path
    env: Mapping[str, str]
    offline: bool = False
    output: TextIO | None = None
    no_config: bool = False

    @property
    def executable(self) -> Path:
        return self.destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _run(self, args: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
        # Explicit index credentials survive, but cannot redirect the project,
        # interpreter or cache selected by the operation.
        env = _base_environment(self.env)
        env.update(UV_PYTHON=str(self.python), UV_PROJECT_ENVIRONMENT=str(self.destination),
                   UV_CACHE_DIR=str(self.cache), UV_PYTHON_DOWNLOADS="never")
        with tempfile.TemporaryDirectory(prefix="pm-uv-config-") as config:
            env.update(XDG_CONFIG_HOME=config, XDG_CONFIG_DIRS=config)
            command = [str(self.uv), *args]
            if self.no_config and "--no-config" not in command:
                command.append("--no-config")
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

    def lock(self, source: Path, *, upgrade: bool = False, timeout: int = 1800) -> None:
        from pm.workspace import classify_uv_failure

        command = ["lock", "--python", str(self.python)]
        if upgrade:
            command.append("--upgrade")
        result = self._run(command, cwd=source, timeout=timeout)
        if result.returncode:
            raise classify_uv_failure("lock", result.returncode, result.stderr or result.stdout)

    def check_lock(self, source: Path) -> None:
        from pm.workspace import classify_uv_failure

        result = self._run(["lock", "--check", "--python", str(self.python)],
                           cwd=source, timeout=1800)
        if result.returncode:
            raise classify_uv_failure("lock", result.returncode, result.stderr or result.stdout)

    def sync(self, source: Path, *, extras: Sequence[str] = (), groups: Sequence[str] = (),
             timeout: int = 1800, frozen: bool = True, all_extras: bool = False,
             no_install_project: bool = False, locked: bool = False,
             no_default_groups: bool = False) -> None:
        """Install the root and every member; resolve only in a writable workspace.

        ``frozen=False`` is reserved for the caller-owned generated workspace,
        never the original project's lock. Seed/replay policy belongs to PM.
        """
        from pm.workspace import classify_uv_failure

        if not frozen:
            self.lock(source, timeout=timeout)
        # Locking members alone is insufficient: plain sync only installs root deps.
        command = ["sync", "--locked" if locked else "--frozen", "--all-packages",
                   "--python", str(self.python)]
        if no_default_groups:
            command.append("--no-default-groups")
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
        for group in sorted(set(groups)):
            command += ["--group", group]
        result = self._run(command, cwd=source, timeout=timeout)
        if result.returncode:
            raise classify_uv_failure("sync", result.returncode, result.stderr or result.stdout)

    def export_requirements(self, source: Path, out: Path, *, extras: Sequence[str] = (),
                            timeout: int = 1800) -> None:
        from pm.workspace import classify_uv_failure

        command = ["export", "--frozen", "--python", str(self.python), "--no-default-groups",
                   "--no-emit-project", "--no-hashes", "--no-annotate", "--no-header",
                   "--format", "requirements-txt", "--output-file", str(out)]
        for extra in sorted(set(extras)):
            command += ["--extra", extra]
        result = self._run(command, cwd=source, timeout=timeout)
        if result.returncode:
            raise classify_uv_failure("export", result.returncode, result.stderr or result.stdout)

    def install_requirements(self, requirements: Sequence[str], *, wheelhouse: Path | None = None) -> None:
        if not requirements:
            return
        # A file avoids command-line length limits and shell/marker quoting.
        with tempfile.TemporaryDirectory(prefix="pm-requirements-") as temporary:
            requirements_file = Path(temporary) / "requirements.txt"
            requirements_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")
            self._install_requirements_file(requirements_file, wheelhouse=wheelhouse)

    def _install_requirements_file(self, requirements: Path, *, wheelhouse: Path | None = None,
                                   timeout: int = 1800) -> None:
        from pm.workspace import classify_uv_failure

        command = ["pip", "install", "--no-config", "--python", str(self.executable),
                   "--requirements", str(requirements)]
        if wheelhouse is not None:
            command += ["--no-index", "--only-binary", ":all:",
                        "--find-links", str(wheelhouse.absolute())]
        result = self._run(command, cwd=requirements.parent, timeout=timeout)
        if result.returncode:
            raise classify_uv_failure("pip", result.returncode, result.stderr or result.stdout)

    def install_wheelhouse(self, source: Path, wheelhouse: Path, *, timeout: int = 1800) -> None:
        """Install rebuilt wheels whose hashes the bundle manifest owns, not uv.lock."""
        requirements = source / "requirements.txt"
        self.export_requirements(source, requirements, timeout=timeout)
        self._install_requirements_file(requirements, wheelhouse=wheelhouse, timeout=timeout)
        self.check()

    def prune_cache(self, *, ci: bool = False) -> None:
        command = ["cache", "prune", "--no-config"]
        if ci:
            command += ["--ci", "--force"]
        result = self._run(command, cwd=Path.cwd(), timeout=1800)
        if result.returncode:
            raise InstallError("uv", f"cache pruning failed: {result.stderr[-600:]}")

    def check(self) -> None:
        result = self._run(
            ["pip", "check", "--no-config", "--python", str(self.executable)],
            cwd=self.destination.parent, timeout=60,
        )
        if result.returncode:
            raise InstallError("venv", f"dependency validation failed: {result.stderr[-600:]}")
