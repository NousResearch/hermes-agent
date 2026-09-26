"""Process and PATH primitives shared by gateway service backends."""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from hermes_constants import get_hermes_home, is_wsl

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def detect_venv_dir() -> Path | None:
    candidates: list[Path] = []
    if sys.prefix != sys.base_prefix:
        candidates.append(Path(sys.prefix))
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates += [PROJECT_ROOT / ".venv", PROJECT_ROOT / "venv"]
    return next((venv for venv in candidates if venv.is_dir()), None)


def python_path() -> str:
    venv = detect_venv_dir()
    if venv is not None:
        try:
            from hermes_constants import venv_python_path
        except ImportError:
            import hermes_constants
            venv_python_path = importlib.reload(hermes_constants).venv_python_path
        candidate = venv_python_path(venv, windows=sys.platform == "win32")
        if candidate.exists():
            return str(candidate)
    return sys.executable


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
    project_root = project_root or PROJECT_ROOT

    def is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    candidates: list[str] = []
    venv_bin = project_root / "venv" / "bin"
    if is_dir(venv_bin):
        candidates.append(str(venv_bin))
    elif sys.prefix != sys.base_prefix:
        candidates.append(str(Path(sys.prefix) / "bin"))

    hermes_home = get_hermes_home()
    for extra in (
        project_root / "node_modules" / ".bin",
        hermes_home / "node" / "bin",
        hermes_home / "node_modules" / ".bin",
    ):
        if is_dir(extra):
            candidates.append(str(extra))
    return candidates


def append_node_dir(path_entries: list[str], hermes_root: Path | None = None) -> None:
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


def service_venv_dir() -> str:
    venv = detect_venv_dir()
    return str(venv) if venv else str(PROJECT_ROOT / "venv")


def hermes_home_for_target_user(target_home_dir: str) -> str:
    raw = os.environ.get("HERMES_HOME", "").strip()
    current = Path(raw).expanduser() if raw else get_hermes_home()
    current_default = Path.home() / ".hermes"
    target_default = Path(target_home_dir) / ".hermes"
    try:
        return str(target_default / current.relative_to(current_default))
    except ValueError:
        return str(current)
