"""The dependency manager's runtime, independent of the application graph.

Only the bootstrap below runs in the caller's interpreter. It never imports
application dependencies or adds the manager's dependencies to that process.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable
import uuid

from pm.filesystem import native
from pm.package import InstallError


def runtime_environment() -> dict[str, str]:
    """Do not let an activated application or a uv caller select PM's imports."""
    from hermes_constants import get_hermes_home
    from pm.paths import store_root

    from pm.environment import _base_environment

    env = _base_environment()
    env["HERMES_HOME"] = str(get_hermes_home())
    env["HERMES_RUNTIME_DIR"] = native(store_root())
    return env


def _python(environment: Path) -> Path:
    from pm.environments import venv_python

    return venv_python(environment)


def _inputs(project: Path, python: Path, *, as_spelled: bool = False) -> str:
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        digest.update((project / name).read_bytes())
        digest.update(b"\0")
    # A different interpreter must not reuse a venv pointing at the old one, but
    # the same one reached through a symlinked store (a per-task HERMES_HOME whose
    # tools/ links back) must. Only the directories resolve: the pinned
    # bin/python3 is itself a symlink, so resolving it would re-key every install.
    # A store spells its root extended-length on Windows; the identity must not.
    python = Path(native(python)).absolute()
    digest.update(str(python if as_spelled else python.parent.resolve() / python.name).encode())
    return digest.hexdigest()


def is_runtime() -> bool:
    if (Path(sys.prefix) / "pm-runtime.json").is_file():
        return True
    resident = _resident_runtime()
    return resident is not None and str(resident[1]) in sys.path


def _resident_runtime() -> tuple[Path, Path] | None:
    from pm.paths import repo_root

    project = repo_root()
    payload = project.parent if (project.parent / "manifest.json").is_file() else None
    if payload is not None:
        runtime = payload / "pm-runtime"
    else:
        from pm.paths import install_stamp_path

        stamp_path = install_stamp_path(project)
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise InstallError("pm-runtime", "invalid package install stamp", "reinstall this application") from exc
        if stamp.get("distribution") not in ("nix", "docker"):
            return None
        value = stamp.get("pmRuntime")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise InstallError("pm-runtime", "packaged PM runtime is missing", "reinstall this application")
        runtime = Path(value)
    try:
        marker = json.loads((runtime / "pm-runtime.json").read_text(encoding="utf-8"))
        python = (runtime / marker["python"]).resolve()
        site = (runtime / marker["sitePackages"]).resolve()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InstallError("pm-runtime", "packaged PM runtime is missing or invalid", "reinstall this application") from exc
    # Native payloads must remain self-contained. Nix's independent derivation
    # instead refers to its immutable interpreter/dependencies in /nix/store.
    if (not python.is_file() or not site.is_dir()
            or (payload is not None and (not python.is_relative_to(payload) or not site.is_relative_to(runtime)))):
        raise InstallError("pm-runtime", "packaged PM paths are missing or escape the payload", "reinstall this application")
    return python, site


def _pinned_python() -> str | None:
    """The Python version PM's own lock pins, or None when it pins none."""
    from pm.lock import Lockfile
    from pm.paths import lockfile_path

    return Lockfile(lockfile_path()).version("python") or None


def _pinned_python_minor(version: str) -> str:
    """The minor of a pinned Python version: "3.14.9+20251021" -> "3.14"."""
    parts = version.partition("+")[0].split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else ""


def _graph_minor() -> str:
    """The Python minor PM's runtime graph admits.

    ``pm/lock.json`` pins the interpreter PM installs (``3.14.9+20251021``),
    the way ``scripts/install.sh`` reads it. When the pin names no minor, the
    graph's own requirement (``pm/uv.lock``: ``==3.14.*``) says which one the
    runtime may be staged with -- uv enforces that requirement, so it is the
    authority on what an interpreter must be.
    """
    version = _pinned_python()
    if version:
        minor = _pinned_python_minor(version)
        if minor:
            return minor
    lock = Path(__file__).resolve().parent / "uv.lock"
    try:
        header = lock.read_text(encoding="utf-8").split("[[package]]", 1)[0]
    except OSError:
        return ""
    found = re.search(r'requires-python\s*=\s*"[^"]*?(\d+)\.(\d+)', header)
    return f"{found.group(1)}.{found.group(2)}" if found else ""


def _store_python(version: str, target: str) -> Path | None:
    """The pinned interpreter PM's store holds, when it holds a sound one."""
    from pm.paths import store_root
    from pm.registry import get_package

    package = get_package("python")
    entry = store_root() / package.store_entry(version, target)
    binary = package.binary(entry, target)
    if binary is None or not binary.is_file():
        return None
    return None if package.verify(entry, target) else binary


def _uv_python(uv: Path, minor: str) -> Path | None:
    """A managed interpreter of *minor*, the way the installer finds one.

    uv's own managed interpreter first, then a host one of that minor -- the
    order ``scripts/install.sh`` bootstraps with. Neither lookup downloads:
    this runs before PM may download anything.
    """
    for arguments, env in (
        (["--managed-python", minor], dict(os.environ, UV_SYSTEM_PYTHON="1", UV_NO_PROJECT="1")),
        (["--system", "--no-project", minor], dict(os.environ, UV_NO_PROJECT="1")),
    ):
        try:
            found = subprocess.run(
                [native(uv), "python", "find", *arguments],
                env=env, capture_output=True, text=True, timeout=120, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if found.returncode != 0:
            continue
        lines = found.stdout.strip().splitlines()
        if not lines:
            continue
        candidate = Path(lines[-1].strip())
        if candidate.is_file():
            return candidate
    return None


def _bootstrap_python(uv: Path) -> Path:
    """The interpreter PM's runtime graph is staged with, when the ambient one cannot.

    PM's runtime graph is locked to one Python minor (``pm/uv.lock``: PM pins
    ``==3.14.*``). Pairing it with an interpreter of another minor makes uv
    refuse the lock outright ("The requested interpreter resolved to Python
    3.11.15, which is incompatible with the project's Python requirement"), and
    the caller only learns that after the environment the sync built has been
    abandoned.

    Resolve an interpreter of the pinned minor instead -- PM's own pinned one
    when the store holds it, else the one uv manages -- the way
    ``scripts/install.sh`` bootstraps one, and keep the ambient interpreter
    only when it already has that minor.
    """
    minor = _graph_minor()
    if not minor:
        return Path(sys.executable)
    if ".".join(str(part) for part in sys.version_info[:2]) == minor:
        return Path(sys.executable)
    from pm.store import current_target

    version = _pinned_python()
    candidates = []
    if version:
        candidates.append(_store_python(version, current_target()))
    candidates.append(_uv_python(uv, minor))
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return Path(sys.executable)


def _validate(python: Path, env: dict[str, str]) -> str:
    try:
        checked = subprocess.run(
            [str(python), "-I", "-B", "-c",
             "import packaging, tomli_w, truststore; from ruamel.yaml import YAML"],
            env=env, capture_output=True, text=True, timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return checked.stderr.strip() or f"exit {checked.returncode}" if checked.returncode else ""


# Generations this process resolved for a child. The lease is taken under
# .prepare.lock, which the collector also holds, so a publish + `pm gc` between
# this return and the child's own lease cannot remove the child's generation.
# It lives as long as this process (one lease per generation, not per call):
# children may still be starting from an older generation after a newer one
# is chosen.
_HELD: dict[Path, Callable[[], None]] = {}


def _hold_for_children(environment: Path) -> None:
    from hermes_cli.runtime_state import lease_directory

    if environment not in _HELD:
        _HELD[environment] = lease_directory(environment)


def prepare_runtime(uv: Path, python: Path, root: Path, *, offline: bool = False,
                    project: Path | None = None, bootstrap: bool = True,
                    cache: Path | None = None) -> Path:
    """Publish a locked PM environment without resolving the application.

    Generations are immutable after publication. Failed preparation leaves the
    previous generation intact, including when an old worker is still running.
    """
    from pm.filesystem import lock_fd
    from pm.lock import _write
    from pm.runtime_stage import stage_runtime

    project = project or Path(__file__).resolve().parent
    identity = _inputs(project, python)
    env = runtime_environment()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as lock:
        lock_fd(lock.fileno(), wait=True)
        selected = root / "selected.json"
        try:
            fact = json.loads(selected.read_text(encoding="utf-8"))
        except FileNotFoundError:
            fact = {}
        # Records from before canonicalization spell the path as launched; for a
        # home under a symlink (/home -> /var/home) that is still this interpreter.
        if fact.get("inputs") in (identity, _inputs(project, python, as_spelled=True)):
            environment = root / fact["generation"]
            if (environment / "pm-runtime.json").is_file() and not _validate(_python(environment), env):
                _hold_for_children(environment)
                return _python(environment)
        if not bootstrap:
            raise InstallError("pm-runtime", "not installed or outdated and lazy installs are disabled",
                               "run `hermes pm install` to prepare the independent PM runtime")
        generation = Path("generations") / uuid.uuid4().hex
        environment = root / generation
        try:
            print("Preparing the isolated Hermes runtime…", file=sys.stderr, flush=True)
            executable = stage_runtime(uv, python, environment, project=project, offline=offline, cache=cache)
            (environment / ".lease-managed").touch()
            _write(environment / "pm-runtime.json", {"inputs": identity})
            _write(selected, {"inputs": identity, "generation": generation.as_posix()})
        except BaseException:
            shutil.rmtree(environment, ignore_errors=True)
            raise
        _hold_for_children(environment)
        return executable


def lease_current_runtime() -> None:
    """Pin the PM runtime this process runs from so the collector leaves it alone."""
    if (Path(sys.prefix) / "pm-runtime.json").is_file():
        from hermes_cli.runtime_state import lease_directory

        lease_directory(Path(sys.prefix))


def collect_runtime_generations(root: Path) -> list[Path]:
    """Remove PM runtime generations nothing can run from any more.

    Staging happens under ``.prepare.lock``, so with it held an unpublished generation
    (no ``pm-runtime.json``) is an aborted stage. A superseded published generation goes
    once every worker launched from it has exited; generations published before leases
    existed stay, as the application collector keeps its own.
    """
    from pm.filesystem import lock_fd
    from hermes_cli.runtime_state import leases_held

    generations = root / "generations"
    removed: list[Path] = []
    if not generations.is_dir():
        return removed
    with (root / ".prepare.lock").open("a+b") as lock:
        if not lock_fd(lock.fileno(), wait=False):
            return removed  # a stage is in flight; maintenance skips rather than queues
        try:
            selected = json.loads((root / "selected.json").read_text(encoding="utf-8-sig")).get("generation", "")
        except FileNotFoundError:
            selected = ""
        for generation in sorted(generations.iterdir()):
            if not generation.is_dir() or generation.is_symlink() or generation == root / selected:
                continue
            published = (generation / "pm-runtime.json").is_file()
            if published and (not (generation / ".lease-managed").is_file() or leases_held(generation)):
                continue
            shutil.rmtree(generation)
            removed.append(generation)
    return removed



def runtime_python(*, bootstrap: bool = True, cache: Path | None = None) -> Path:
    """Resolve PM without selecting, repairing, or importing the app environment."""
    if is_runtime():
        return Path(sys.executable)
    from pm.environments import install_state_dir
    from pm._uv import _toolchain
    from pm.paths import repo_root

    project = repo_root()
    resident = _resident_runtime()
    if resident is not None:
        return resident[0]
    tools = _toolchain(realize=False)
    if tools is None:
        if not bootstrap:
            raise InstallError("pm-runtime", "not installed and lazy installs are disabled",
                               "run `hermes pm install` to prepare the independent PM runtime")
        from pm.lock import Lockfile
        from pm.paths import lockfile_path, store_root
        from pm.registry import get_package
        from pm.store import current_target

        # Setup has already verified/extracted uv, but there are no PM facts
        # yet. Use it to acquire PM's TLS support BEFORE downloading Python.
        package = get_package("uv")
        version = Lockfile(lockfile_path()).version("uv")
        target = current_target()
        staged = package.binary(store_root() / package.store_entry(version, target), target) if version else None
        if staged is not None and staged.is_file():
            tools = staged, _bootstrap_python(staged)
        else:
            # Non-shell bootstrap callers (CI) already have a host interpreter.
            tools = _toolchain(explicit=True)
    if tools is None:
        raise InstallError("pm-runtime", "pinned uv and Python are unavailable")
    uv, python = tools
    return prepare_runtime(uv, python, install_state_dir(project) / "pm-runtime",
                           bootstrap=bootstrap, cache=cache)


def runtime_command(script: Path, args: tuple[str, ...] | list[str] = (), *,
                    bootstrap: bool = True, cache: Path | None = None) -> list[str]:
    """One launch contract for mutable venvs and resident signed payloads."""
    resident = _resident_runtime()
    if resident is None:
        python = runtime_python(bootstrap=bootstrap, cache=cache)
        return [str(python), "-I", "-B", str(script), *args]
    python, site = resident
    launcher = (
        "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
        "script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')"
    )
    return [str(python), "-I", "-S", "-B", "-c", launcher, str(site), str(script), *args]


def run_cli(argv: list[str]) -> int:
    result = subprocess.run(runtime_command(Path(__file__).with_name("launch.py"), argv),
                            env=runtime_environment(), check=False)
    return result.returncode
