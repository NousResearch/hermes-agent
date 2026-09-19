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
    for directory in (root, *root.parents):
        stamp = directory / "install-stamp.json"
        if stamp.is_file():
            try:
                data = json.loads(stamp.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return get_default_hermes_root() / "tools"
            value = data.get("runtimeDir") if isinstance(data, dict) else None
            return Path(value).resolve() if value else get_default_hermes_root() / "tools"
    return get_default_hermes_root() / "tools"


def selected_venv(project_root: Path) -> Path:
    """Use the committed environment, or the original install before first sync.

    A broken committed selection is an error, not permission to load an older
    dependency set silently. Reading this function never creates user state.
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
    """The interpreter version a POSIX venv actually holds, or ``None``.

    ``site_packages`` must not date the tree from the CALLER's ``sys.version_info``:
    an update can rebuild the dependency environment with a different Python than
    the launcher that later imports it. Observed on an app-driven upgrade -- PM
    built the environment with CPython 3.14 while the PATH shim ran 3.11, so the
    shim composed ``lib/python3.11/site-packages`` inside a 3.14 venv, found no
    tree, and failed *after* a successful update.
    """
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() != "version":
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
            lease_generation(environment)
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
    # environment was composed against, so a consumer gets three things for
    # free: that it inherited an activated shell, which checkout/profile that
    # shell came from, and a staleness stamp — uv.lock / pyproject.toml /
    # pm/lock.json newer than this file means the shell's environment predates
    # its inputs. pm rewrites it on every real sync and no-ops otherwise, so a
    # `-nt` comparison settles back to "current" after one re-activation.
    env["__HERMES_ACTIVATED"] = str(runtime_facts_path(project_root))
    return env


if __name__ == "__main__":
    print(json.dumps(activation_environment(Path(__file__).resolve().parents[1])))
