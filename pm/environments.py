"""Dependency-environment layout: where a project's venv generations live, which one
is selected, and the interpreter inside any venv. Shared by PM and pre-import launchers.

Only stdlib and hermes_constants: environment selection must work before
any dependency from that environment has been imported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hermes_constants import get_default_hermes_root, project_venv_dir


def install_key(project_root: Path) -> str:
    canonical = str(Path(project_root).resolve())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dependency_home_root() -> Path:
    """Scope dependency state like a process launched in the active home."""
    from hermes_constants import get_default_hermes_root, get_hermes_home_override

    override = get_hermes_home_override()
    return get_default_hermes_root(home=override) if override else get_default_hermes_root()


def installs_root() -> Path:
    return dependency_home_root() / "installs"


def install_state_dir(project_root: Path) -> Path:
    return installs_root() / install_key(project_root)


def runtime_facts_path(project_root: Path) -> Path:
    return install_state_dir(project_root) / "facts.json"


# The files that decide the dependency set. `scripts/_hermes-python` re-activates
# when any of them differs in mtime from its stamp under activation_inputs_dir.
ACTIVATION_INPUTS = ("uv.lock", "pyproject.toml", "pm/lock.json")


def activation_inputs_dir(project_root: Path) -> Path:
    """Beside facts.json, so the prologue finds it from ``$__HERMES_ACTIVATED``."""
    return install_state_dir(project_root) / "inputs"


def activation_input_mtimes(project_root: Path) -> dict[str, int]:
    """Snapshot before installing, so an input edited mid-install records its
    pre-install mtime and the next run re-activates."""
    root = Path(project_root)
    return {name: (root / name).stat().st_mtime_ns for name in ACTIVATION_INPUTS if (root / name).is_file()}


def record_activation_inputs(stamps: Path, mtimes: dict[str, int], project_root: Path, *, test_environment: bool) -> None:
    """Give each stamp the exact mtime of the input the install was verified against.

    Recorded on every successful install, including no-op syncs: a checkout that
    rewrites an input without changing it moves the mtime, and only this record
    brings the stamp back to equal. The prologue compares for equality, not order,
    because switching branches can move an input's mtime in either direction.
    """
    import shutil

    shutil.rmtree(stamps, ignore_errors=True)
    # The sentinel is inherited by child shells; equal input mtimes in another
    # checkout must never make their test interpreter appear current here.
    stamps.mkdir(parents=True, exist_ok=True)
    (stamps / ".project-root").write_text(str(Path(project_root).resolve()), encoding="utf-8")
    if test_environment:
        (stamps / ".test-environment").touch()
    for name, mtime in mtimes.items():
        stamp = stamps / name
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
        os.utime(stamp, ns=(mtime, mtime))


def base_venv(project_root: Path) -> Path:
    root = Path(project_root).resolve()
    manifest_path = root.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if (root.parent / manifest.get("repo", "")).resolve() == root:
            venv = (root.parent / manifest["venv"]).resolve()
            if not venv.is_relative_to(root.parent):
                raise RuntimeError("payload environment escapes its root")
            return venv
    return project_venv_dir(root) or root / "venv"


def store_root(project_root: Path) -> Path:
    """Resolve a payload-relative or stamped store before PM imports."""
    override = os.environ.get("HERMES_RUNTIME_DIR")
    if override:
        return Path(override).resolve()
    root = Path(project_root).resolve()
    manifest_path = root.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if (root.parent / manifest.get("repo", "")).resolve() == root:
            store = (root.parent / manifest["store"]).resolve()
            if not store.is_relative_to(root.parent):
                raise RuntimeError("payload store escapes its root")
            return store
    from pm.paths import install_stamp_path

    for directory in (root, *root.parents):
        stamp = install_stamp_path(directory)
        if stamp.is_file():
            try:
                data = json.loads(stamp.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return get_default_hermes_root() / "tools"
            value = data.get("runtimeDir") if isinstance(data, dict) else None
            return Path(value).resolve() if value else get_default_hermes_root() / "tools"
    return get_default_hermes_root() / "tools"


def flush_before_selecting() -> None:
    """Make a finished generation tree durable before a selection record names it.

    The selection record (the install's facts.json, a side environment's
    active.json) is written last, atomically and fsynced, so it doubles as the
    generation's completion marker -- but only if every file it vouches for
    reached the disk first. Otherwise a power loss can persist the record while
    the tree's data is still in the page cache, selecting a half-written venv.
    One filesystem-wide sync instead of an fsync per file: a venv holds tens of
    thousands of files, and per-file flushes cost minutes on slow disks.

    Windows has no whole-filesystem flush reachable from Python (os.sync does
    not exist there), and FlushFileBuffers per file is the slow path rejected
    above. There the record's atomic write is the only guarantee: it is never
    torn, but NTFS journals metadata, not file contents, so power loss right
    after a publish can still leave the selected tree with incomplete files.
    """
    sync = getattr(os, "sync", None)
    if sync is not None:
        sync()


def selected_venv(project_root: Path) -> Path:
    """Use the committed environment, or the original install before first sync.

    A broken committed selection is an error, not permission to load an older
    dependency set silently. Reading this function never creates user state.
    The record itself is the completion marker: it is only written after
    ``flush_before_selecting``, so the ``pyvenv.cfg`` probe below is a sanity
    check against a vanished tree, not the durability guarantee.
    """
    path = runtime_facts_path(project_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return base_venv(project_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read dependency environment: {path}") from exc
    try:
        fact = data.get("packages", {}).get("venv", {})
        value = fact.get("environment")
    except AttributeError as exc:
        raise RuntimeError(f"invalid dependency environment record: {path}") from exc
    if value is None:
        return base_venv(project_root)
    if not isinstance(value, str):
        raise RuntimeError("invalid dependency environment path")
    environment = Path(value).resolve()
    generations = install_state_dir(project_root) / "environments"
    if not environment.is_relative_to(generations.resolve()) or not (environment / "pyvenv.cfg").is_file():
        raise RuntimeError(f"dependency environment is missing or outside this install: {environment}")
    return environment


def venv_bin_dir(venv: Path, *, windows: bool | None = None) -> Path:
    """``Scripts`` on Windows, ``bin`` elsewhere. Returned unconditionally — callers
    differ on whether a missing venv is an error. *windows* lets a POSIX process
    reason about a Windows layout (update hand-off, launcher repair)."""
    if windows is None:
        windows = os.name == "nt"
    return Path(venv) / ("Scripts" if windows else "bin")


def venv_python(venv: Path, *, windows: bool | None = None) -> Path:
    """The interpreter inside *venv* (may not exist)."""
    bin_dir = venv_bin_dir(venv, windows=windows)
    return bin_dir / ("python.exe" if bin_dir.name == "Scripts" else "python")


def project_python(project_root: Path) -> Path:
    """The interpreter of the committed dependency environment for *project_root*."""
    return venv_python(selected_venv(project_root))


def venv_python_version(venv: Path) -> tuple[int, int] | None:
    """The interpreter version a venv actually holds, or ``None``.

    ``site_packages`` must not date the tree from the CALLER's ``sys.version_info``:
    an update can rebuild the dependency environment with a different Python than
    the launcher that later imports it. Observed on an app-driven upgrade -- PM
    built the environment with CPython 3.14 while the PATH shim ran 3.11, so the
    shim composed ``lib/python3.11/site-packages`` inside a 3.14 venv, found no
    tree, and failed *after* a successful update.
    """
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8-sig").splitlines():
            key, _, value = line.partition("=")
            # stdlib venv uses version; uv uses version_info. Windows has
            # unversioned Lib/site-packages, so the config is its only signal.
            if key.strip() not in ("version", "version_info"):
                continue
            major, _, rest = value.strip().partition(".")
            minor, _, _ = rest.partition(".")
            if major.isdigit() and minor.isdigit():
                return int(major), int(minor)
    except OSError:
        pass
    try:
        candidates = sorted((venv / "lib").glob("python3*"))
    except OSError:
        return None
    for candidate in candidates:
        major, _, rest = candidate.name.removeprefix("python").partition(".")
        minor, _, _ = rest.partition(".")
        if major.isdigit() and minor.isdigit():
            return int(major), int(minor)
    return None


def site_packages(venv: Path) -> Path:
    import sys

    if os.name == "nt":
        return venv / "Lib/site-packages"
    version = venv_python_version(venv) or (sys.version_info.major, sys.version_info.minor)
    return venv / f"lib/python{version[0]}.{version[1]}/site-packages"


def running_from_selected_environment(project_root: Path) -> bool:
    """Does this process run on the environment PM selected for the install (base venv or committed
    generation)?

    A lazy sync from any other interpreter — a build_environment test venv, a developer's own venv,
    a Nix store Python — must not commit the install's selection: activation is a boot decision, so
    this process keeps running unchanged while every process booted afterwards swaps onto a
    generation that lacks whatever the foreign interpreter carried.

    activate_dependencies puts the selection's site-packages on sys.path without changing
    sys.prefix, so sys.path is the signal (the same one ensure_import reads after a sync).
    """
    import sys

    try:
        selected = site_packages(selected_venv(project_root)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return any(Path(entry).resolve() == selected for entry in sys.path if entry)


class DependencyPythonMismatch(RuntimeError):
    """The selected dependency tree needs its own interpreter before imports."""

    def __init__(self, environment: Path, version: tuple[int, int]):
        self.python = venv_python(environment)
        super().__init__(f"Python {version[0]}.{version[1]} dependencies at {environment} "
                         f"require their matching interpreter: {self.python}")


def activate_dependencies(project_root: Path) -> None:
    """Select the committed tree at process boot, before third-party imports.

    A process with no extension selection keeps its original launch contract.
    Already-running processes are never switched after a dependency install.
    """
    import sys

    state = install_state_dir(project_root)
    if state.is_dir():
        from hermes_cli.runtime_state import runtime_lock, recover_publication, lease_generation
        # The lock's holder may be another profile's backend running a full dependency rebuild;
        # this process only reads the committed selection, so it proceeds without waiting rather
        # than leaving the backend unbound (see runtime_lock).
        with runtime_lock(project_root) as held:
            if held:
                recover_publication(project_root)
            environment = selected_venv(project_root)
            release = lease_generation(environment)
            # Without the lock, an installer may commit a new generation between the
            # read and the lease, leaving the leased one unselected and collectable.
            while not held and (current := selected_venv(project_root)) != environment:
                release()
                environment, release = current, lease_generation(current)
            selected = site_packages(environment)
            if not selected.is_dir() and not runtime_facts_path(project_root).is_file():
                return
    else:
        # Older installs and sealed payloads still select once, before imports.
        # Never consult VIRTUAL_ENV: it can describe the invoking shell's Python.
        environment = base_venv(project_root)
        selected = site_packages(environment)
        if not selected.is_dir():
            return  # External/Nix interpreter owns its original sys.path.
    if not selected.is_dir():
        raise RuntimeError(f"dependency environment has no site-packages: {selected}")
    version = venv_python_version(environment)
    if version is not None and version != sys.version_info[:2]:
        # Tool acquisition can repin Python before the application sync commits.
        # Its failure keeps the old dependencies, not necessarily the launcher's
        # ABI. Refuse before executable .pth files or native extensions load.
        raise DependencyPythonMismatch(environment, version)
    import site

    sys.path[:] = [entry for entry in sys.path
                   if Path(entry).name not in ("site-packages", "dist-packages")
                   and Path(entry).resolve() != project_root.resolve()]
    # uv editable members are activated by .pth files, not by sys.path alone.
    site.addsitedir(str(selected))
    sys.path[:] = [str(project_root.resolve()), str(selected),
                   *[entry for entry in sys.path if Path(entry).resolve() != selected.resolve()]]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(project_root.resolve()), str(selected)])
    os.environ.pop("VIRTUAL_ENV", None)
    executable_dir = venv_bin_dir(environment)
    if executable_dir.is_dir():
        os.environ["PATH"] = os.pathsep.join([str(executable_dir), os.environ.get("PATH", "")])


def activation_environment(project_root: Path) -> dict[str, str]:
    """Read the installed PM environment; do not provision or switch imports."""
    from pm.install import env_for
    from pm.registry import all_packages

    env = env_for(*all_packages())
    selected = site_packages(selected_venv(project_root))
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONPATH"] = os.pathsep.join([str(project_root.resolve()), str(selected)])
    # The child-process sentinel. Its VALUE is the installed-state file this
    # environment was composed against, so a consumer learns that it inherited
    # an activated shell and which checkout/profile that shell came from. Its
    # directory also holds activation_inputs_dir, the input-mtime stamps
    # `scripts/_hermes-python` compares against to decide staleness.
    env["__HERMES_ACTIVATED"] = str(runtime_facts_path(project_root))
    # The suite's interpreter (pm.testenv): an isolated side environment, so it
    # never appears on PYTHONPATH/PATH above. scripts/run_tests.sh reads it.
    from pm.testenv import testenv_python

    test_python = testenv_python(project_root)
    if test_python is not None:
        env["__HERMES_TEST_PYTHON"] = str(test_python)
    return env


if __name__ == "__main__":
    print(json.dumps(activation_environment(Path(__file__).resolve().parents[1])))
