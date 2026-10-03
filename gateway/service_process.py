"""Process and PATH primitives shared by gateway service backends."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from hermes_constants import get_hermes_home, is_wsl

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def python_path() -> str:
    """Interpreter selected by PM for this installation, else the current external interpreter."""
    from hermes_cli._launchers import resolve_store_python

    return str(resolve_store_python(PROJECT_ROOT) or sys.executable)


def build_user_local_paths(home: Path, path_entries: list[str]) -> list[str]:
    candidates = [
        str(home / ".local" / "bin"),
        str(home / ".cargo" / "bin"),
        str(home / "go" / "bin"),
        str(home / ".npm-global" / "bin"),
    ]
    return [p for p in candidates if p not in path_entries and Path(p).exists()]


def build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    if not is_wsl():
        return []
    candidates = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry.startswith("/mnt/")
    ]
    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(str(Path(resolved).parent))
    candidates += [
        entry
        for entry in (
            "/mnt/c/WINDOWS/system32",
            "/mnt/c/WINDOWS",
            "/mnt/c/WINDOWS/System32/Wbem",
            "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
            "/mnt/c/WINDOWS/System32/OpenSSH/",
        )
        if Path(entry).exists()
    ]
    result: list[str] = []
    seen = set(path_entries)
    for entry in candidates:
        if entry and entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


def remap_path_for_user(path: str, target_home_dir: str) -> str:
    current_home = Path.home()
    candidate = Path(path).expanduser()
    try:
        relative = candidate.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(candidate)


def stable_working_dir() -> str:
    try:
        home = get_hermes_home()
        if home and Path(home).is_dir():
            return str(Path(home).resolve())
    except Exception:
        pass
    return str(PROJECT_ROOT)


def service_path_dirs(project_root: Path | None = None) -> list[str]:
    """Stable service PATH additions; never persist a selected Python/dependency generation."""
    project_root = project_root or PROJECT_ROOT

    def is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    hermes_home = get_hermes_home()
    candidates: list[str] = []
    for extra in (
        project_root / "node_modules" / ".bin",
        hermes_home / "node" / "bin",
        hermes_home / "node_modules" / ".bin",
    ):
        if is_dir(extra):
            candidates.append(str(extra))
    return candidates


def pm_managed_node_dirs(home: Path) -> list[str]:
    """Node/npm/npx PATH dirs recorded by PM installed-state under *home*."""
    from pm.lock import Facts

    store = Path(home) / "tools"
    facts = Facts(store / "facts.json")
    dirs: list[str] = []
    for name in ("node", "npm", "npx"):
        for value in facts.env_for(name, store).get("PATH") or []:
            if value and Path(value).is_dir():
                entry = str(value)
                if entry not in dirs:
                    dirs.append(entry)
    return dirs


def append_node_dir(path_entries: list[str], hermes_root: Path | None = None) -> None:
    """Append the Node dirs a service should use, preferring PM installed-state."""
    home = Path(hermes_root) if hermes_root is not None else Path(get_hermes_home())
    try:
        managed_dirs = pm_managed_node_dirs(home)
    except Exception:
        managed_dirs = []

    for entry in managed_dirs:
        if entry not in path_entries:
            path_entries.append(entry)
    if managed_dirs:
        return

    from hermes_constants import hermes_managed_node_tree_present, iter_hermes_node_dirs

    managed = hermes_managed_node_tree_present(hermes_root)
    for directory in iter_hermes_node_dirs(hermes_root) if managed else ():
        entry = str(directory)
        try:
            present = directory.is_dir()
        except OSError:
            present = False
        if present and entry not in path_entries:
            path_entries.append(entry)
    if managed:
        return

    resolved = shutil.which("node")
    if resolved:
        node_dir = str(Path(resolved).parent)
        if node_dir not in path_entries:
            path_entries.append(node_dir)


def prepare_installation_launcher(
    project_root: Path | None = None,
    home: str | Path | None = None,
    owner: tuple[int, str] | None = None,
) -> None:
    """Publish the stable source-install launcher before a service persists it."""
    from hermes_cli._launchers import ENTRY_POINTS, ensure_install_launchers, resolve_store_python
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    project_root = Path(project_root or PROJECT_ROOT)
    target_home = Path(home) if home is not None else Path(get_hermes_home())
    token = set_hermes_home_override(target_home)
    try:
        if resolve_store_python(project_root) is None:
            return
        local = project_root / ".hermes" / "bin"
        paths = ensure_install_launchers(project_root, local)
        if len(paths) != len(ENTRY_POINTS):
            raise RuntimeError("Could not publish the gateway installation launcher")
        if owner is not None:
            import os as _os
            import pwd
            uid, username = owner
            gid = pwd.getpwnam(username).pw_gid
            for path in (local.parent, local, *map(Path, paths)):
                _os.chown(path, uid, gid)
    finally:
        reset_hermes_home_override(token)


def hermes_home_for_target_user(target_home_dir: str) -> str:
    raw = os.environ.get("HERMES_HOME", "").strip()
    current = Path(raw).expanduser() if raw else get_hermes_home()
    current_default = Path.home() / ".hermes"
    target_default = Path(target_home_dir) / ".hermes"
    try:
        return str(target_default / current.relative_to(current_default))
    except ValueError:
        return str(current)
