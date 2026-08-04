"""Import-safe helpers for inspecting a Python interpreter's linked SQLite.

This module intentionally depends only on the standard library.  Installer and
update code must be able to use it before Hermes' third-party dependencies are
healthy.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _version_tuple(parts: Iterable[object]) -> tuple[int, int, int]:
    values = [int(part) for part in parts]
    values.extend([0] * (3 - len(values)))
    return tuple(values[:3])


def is_sqlite_wal_reset_vulnerable(
    version_info: tuple[int, ...],
) -> bool:
    """Return whether *version_info* contains SQLite's WAL-reset bug."""
    info = _version_tuple(version_info)
    if info < (3, 7, 0):
        return False
    if info >= (3, 51, 3):
        return False
    if (3, 50, 7) <= info < (3, 51, 0):
        return False
    if (3, 44, 6) <= info < (3, 45, 0):
        return False
    return True


@dataclass(frozen=True)
class SQLiteRuntimeInfo:
    """SQLite details reported by one exact Python executable."""

    executable: Path
    base_prefix: Path
    python_version: tuple[int, int, int]
    sqlite_version: tuple[int, int, int]
    sqlite_version_string: str
    sqlite_source_id: str

    @property
    def wal_reset_vulnerable(self) -> bool:
        return is_sqlite_wal_reset_vulnerable(self.sqlite_version)


_PROBE_SCRIPT = """
import json
import sqlite3
import sys

conn = sqlite3.connect(":memory:")
try:
    row = conn.execute("SELECT sqlite_source_id()").fetchone()
finally:
    conn.close()

print(json.dumps({
    "base_prefix": sys.base_prefix,
    "executable": sys.executable,
    "python_version": list(sys.version_info[:3]),
    "sqlite_version": list(sqlite3.sqlite_version_info),
    "sqlite_version_string": sqlite3.sqlite_version,
    "sqlite_source_id": str(row[0]) if row and row[0] is not None else "",
}))
"""


def probe_sqlite_runtime(
    python: str | Path,
    *,
    timeout: float = 30.0,
) -> SQLiteRuntimeInfo | None:
    """Probe SQLite in *python*, never the caller's linked SQLite.

    ``None`` means the interpreter could not be executed or returned malformed
    data.  The child runs isolated from inherited Python path overrides.
    """
    executable = Path(python)
    env = dict(os.environ)
    for key in (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    try:
        result = subprocess.run(
            [str(executable), "-I", "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        return SQLiteRuntimeInfo(
            executable=Path(str(payload["executable"])),
            base_prefix=Path(str(payload["base_prefix"])),
            python_version=_version_tuple(payload["python_version"]),
            sqlite_version=_version_tuple(payload["sqlite_version"]),
            sqlite_version_string=str(payload["sqlite_version_string"]),
            sqlite_source_id=str(payload.get("sqlite_source_id", "")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
