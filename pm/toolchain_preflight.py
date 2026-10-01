"""Fail fast when a venv sync would compile native deps without a C++ toolchain.

Matrix E2EE (`python-olm`) builds from sdist on many Linux/Python combos; PBS
interpreters default to ``clang++``. A missing compiler used to surface only
inside ``uv sync``, which crash-looped gateway units under Restart=always.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import subprocess
import sys

_MATRIX_EXTRA = "matrix"

_PREFLIGHT_ERROR = (
    "Hermes matrix/E2EE native build needs clang++ on PATH "
    "(e.g. apt install clang). Refusing to sync rather than crash-looping the gateway."
)


def _first_executable(token: str) -> str:
    return token.strip().split()[0] if token.strip() else ""


def _resolve_cxx(build_env: Mapping[str, str], python: Path | None) -> str | None:
    cxx = build_env.get("CXX")
    if cxx:
        return _first_executable(cxx) or None
    cc = build_env.get("CC")
    if cc:
        cc_bin = _first_executable(cc)
        if "clang" in Path(cc_bin).name:
            if cc_bin.endswith("clang"):
                return cc_bin + "++"
            return cc_bin
    if python is None:
        return None
    try:
        proc = subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_config_var('CXX') or '')"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    resolved = _first_executable(proc.stdout)
    return resolved or None


def _compiler_on_path(command: str, build_env: Mapping[str, str]) -> bool:
    if not command:
        return False
    path = Path(command)
    if path.is_file():
        return True
    return shutil.which(command, path=build_env.get("PATH")) is not None


def _sync_needs_matrix_native_build(extras: Sequence[str]) -> bool:
    if _MATRIX_EXTRA not in extras:
        return False
    from pm.extras import extra_supported

    return extra_supported(_MATRIX_EXTRA)


def _build_env_uses_clang(build_env: Mapping[str, str], python: Path | None) -> bool:
    for key in ("CXX", "CC"):
        value = build_env.get(key, "")
        if value and "clang" in _first_executable(value):
            return True
    cxx = _resolve_cxx(build_env, python)
    return cxx is not None and "clang" in Path(cxx).name


def require_native_cxx_for_sync(
    extras: Sequence[str],
    *,
    build_env: Mapping[str, str] | None = None,
    python: Path | None = None,
) -> None:
    """Raise before ``uv sync`` when a C++ compiler is required but missing."""
    env = dict(build_env if build_env is not None else os.environ)
    needs_matrix = _sync_needs_matrix_native_build(extras)
    uses_clang = _build_env_uses_clang(env, python)
    if not needs_matrix and not uses_clang:
        return

    cxx = _resolve_cxx(env, python)
    if needs_matrix and sys.platform == "linux":
        candidate = cxx or "clang++"
    elif uses_clang:
        candidate = cxx or "clang++"
    else:
        return

    if not _compiler_on_path(candidate, env):
        raise RuntimeError(_PREFLIGHT_ERROR)
