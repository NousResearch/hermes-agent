"""Virtual environment utilities for ARGUS.

Provides cross-platform venv detection and environment building for subprocess
calls. Subprocesses must inherit venv context or they fail to find hermes modules.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Platform constants for cross-platform compatibility
_IS_WINDOWS = os.name == 'nt' or sys.platform == 'win32'
_IS_MACOS = sys.platform == 'darwin'
_IS_LINUX = sys.platform.startswith('linux')


def is_running_in_venv() -> bool:
    """Check if current Python is in a virtual environment (venv, virtualenv, or conda)."""
    if hasattr(sys, 'real_prefix'):
        return True
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return True
    if os.environ.get('CONDA_DEFAULT_ENV'):
        return True
    if os.environ.get('VIRTUAL_ENV'):
        return True
    return False


def get_venv_path() -> Optional[Path]:
    """Get path to current virtual environment root, or None."""
    if venv_env := os.environ.get('VIRTUAL_ENV'):
        return Path(venv_env)
    if conda_prefix := os.environ.get('CONDA_PREFIX'):
        return Path(conda_prefix)
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return Path(sys.prefix)
    return None


def get_venv_bin_dir(venv_path: Optional[Union[str, Path]] = None) -> Path:
    """Get bin/ (POSIX) or Scripts/ (Windows) directory for a venv."""
    if venv_path is None:
        venv_path = get_venv_path()
        if venv_path is None:
            raise ValueError("Not in a virtual environment and no path provided")

    venv = Path(venv_path)

    if _IS_WINDOWS:
        return venv / 'Scripts'
    return venv / 'bin'


def get_venv_python(venv_path: Optional[Union[str, Path]] = None) -> str:
    """Get Python executable path for a virtual environment."""
    bin_dir = get_venv_bin_dir(venv_path)

    if _IS_WINDOWS:
        python_exe = bin_dir / 'python.exe'
        if python_exe.exists():
            return str(python_exe)
    else:
        python3_path = bin_dir / 'python3'
        if python3_path.exists():
            return str(python3_path)

    python_path = bin_dir / 'python'
    if python_path.exists():
        return str(python_path)

    return sys.executable


def detect_hermes_venv() -> Optional[Path]:
    """Detect hermes-agent venv location. Priority: HERMES_VENV, production, dev."""
    if env_venv := os.environ.get('HERMES_VENV'):
        if Path(env_venv).exists():
            return Path(env_venv)

    home = Path.home()
    prod_venv = home / '.hermes' / 'hermes-agent' / 'venv'
    if prod_venv.exists():
        return prod_venv

    dev_venv = home / 'Projects' / 'hermes-dev' / '.local' / 'venv'
    if dev_venv.exists():
        return dev_venv

    return None


def get_hermes_python() -> str:
    """Get best Python for Hermes/Argus. Priority: current venv, hermes venv, system."""
    if is_running_in_venv():
        return sys.executable

    if hermes_venv := detect_hermes_venv():
        try:
            return get_venv_python(hermes_venv)
        except ValueError:
            pass

    return shutil.which('python3') or sys.executable


def build_venv_aware_env(
    base_env: Optional[Dict[str, str]] = None,
    preserve_venv: bool = True,
    extra_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build environment dict preserving venv context for subprocess calls."""
    env = os.environ.copy() if base_env is None else dict(base_env)

    if preserve_venv and is_running_in_venv():
        if venv_path := get_venv_path():
            env['VIRTUAL_ENV'] = str(venv_path)
            venv_bin = str(get_venv_bin_dir(venv_path))
            current_path = env.get('PATH', '')
            if venv_bin not in current_path:
                env['PATH'] = f"{venv_bin}{os.pathsep}{current_path}"

    if extra_paths:
        path_parts = extra_paths + [p for p in env.get('PATH', '').split(os.pathsep) if p]
        seen = set()
        unique_paths = []
        for p in path_parts:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        env['PATH'] = os.pathsep.join(unique_paths)

    return env


def _is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux."""
    if not _IS_LINUX:
        return False
    if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'):
        return True
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            return 'microsoft' in version or 'wsl' in version
    except Exception:
        pass
    return False


def _get_platform_std_paths() -> List[str]:
    """Get platform-specific standard binary paths."""
    if _IS_MACOS:
        return ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']
    elif _IS_LINUX:
        # Standard Linux paths + common package manager locations
        paths = ['/usr/local/bin', '/usr/bin', '/bin']
        # Add snap packages if present
        snap_bin = os.path.expanduser('~/snap/bin')
        if os.path.isdir(snap_bin):
            paths.insert(0, snap_bin)
        # Add flatpak exports if present
        flatpak_bin = '/var/lib/flatpak/exports/bin'
        if os.path.isdir(flatpak_bin):
            paths.append(flatpak_bin)
        # WSL-specific: Windows paths may be available via /mnt/c/Windows
        if _is_wsl():
            # Windows system32 is often in PATH via WSL interop
            pass  # WSL handles this automatically
        return paths
    elif _IS_WINDOWS:
        # Windows uses PATH env and registry, no standard Unix-style paths
        return []
    return ['/usr/local/bin', '/usr/bin', '/bin']


def get_agathos_venv_paths() -> List[str]:
    """Get PATH entries for ARGUS subprocess operations in priority order."""
    paths = []

    if is_running_in_venv():
        try:
            paths.append(str(get_venv_bin_dir()))
        except ValueError:
            pass

    if hermes_venv := detect_hermes_venv():
        hermes_bin = str(get_venv_bin_dir(hermes_venv))
        if hermes_bin not in paths:
            paths.append(hermes_bin)

    home = Path.home()
    hermes_user_bin = str(home / 'hermes' / 'bin')
    if Path(hermes_user_bin).exists() and hermes_user_bin not in paths:
        paths.append(hermes_user_bin)

    # Use platform-specific standard paths
    for std_path in _get_platform_std_paths():
        if std_path not in paths:
            paths.append(std_path)

    return paths


def build_agathos_subprocess_env(
    inherit_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build complete environment for ARGUS subprocess calls with venv context."""
    env = os.environ.copy() if inherit_env is None else dict(inherit_env)

    argus_paths = get_agathos_venv_paths()
    current_parts = [p for p in env.get('PATH', '').split(os.pathsep) if p]

    seen = set(argus_paths)
    merged = list(argus_paths)
    for p in current_parts:
        if p not in seen:
            seen.add(p)
            merged.append(p)

    env['PATH'] = os.pathsep.join(merged)

    if 'HOME' not in env:
        env['HOME'] = str(Path.home())

    if 'HERMES_HOME' not in env:
        env['HERMES_HOME'] = str(Path.home() / '.hermes')

    if is_running_in_venv() and 'VIRTUAL_ENV' not in env:
        if venv_path := get_venv_path():
            env['VIRTUAL_ENV'] = str(venv_path)

    return env


def resolve_venv_python(python_cmd: Optional[str] = None) -> str:
    """Resolve Python command to appropriate venv Python."""
    if python_cmd is None or python_cmd in ('python3', 'python', sys.executable):
        return get_hermes_python()

    if Path(python_cmd).is_absolute():
        return python_cmd

    env = build_agathos_subprocess_env()
    for part in env.get('PATH', '').split(os.pathsep):
        candidate = Path(part) / python_cmd
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    return python_cmd


__all__ = [
    'is_running_in_venv',
    'get_venv_path',
    'get_venv_bin_dir',
    'get_venv_python',
    'detect_hermes_venv',
    'get_hermes_python',
    'build_venv_aware_env',
    'get_agathos_venv_paths',
    'build_agathos_subprocess_env',
    'resolve_venv_python',
]
