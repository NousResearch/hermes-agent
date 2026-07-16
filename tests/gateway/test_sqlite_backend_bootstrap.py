"""Clean-process regression tests for sqlite backend bootstrap in gateway.run.

The optional ``modern-sqlite`` extra only helps if the backend selection runs
*before* gateway.run binds ``sqlite3`` at module top. A plain in-process import
isn't good enough here because the active interpreter may already have a cached
``sqlite3`` module from earlier test setup. These tests spawn a fresh Python
process, clear any preloaded sqlite modules, then import gateway.run and inspect
which module object got bound.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_pysqlite3(shim_root: Path) -> None:
    pkg = shim_root / "pysqlite3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            import sqlite3 as _stdlib_sqlite3

            SHIM_MARKER = "fake-pysqlite3"
            dbapi2 = _stdlib_sqlite3.dbapi2

            def __getattr__(name):
                return getattr(_stdlib_sqlite3, name)
            """
        )
    )


def _run_gateway_import(
    hermes_home: Path,
    shim_root: Path,
    initial_env: dict[str, str],
    block_pysqlite3: bool = False,
) -> dict[str, str | None]:
    script = textwrap.dedent(
        f"""
        import json, os, sys

        sys.modules.pop("sqlite3", None)
        sys.modules.pop("sqlite3.dbapi2", None)
        sys.modules.pop("pysqlite3", None)
        sys.path.insert(0, {str(shim_root)!r})
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        if {block_pysqlite3!r}:
            # Force the default-install path regardless of whether pysqlite3
            # happens to be present in site-packages: block the import so the
            # backend hook falls through to stdlib sqlite3.
            import builtins
            _real_import = builtins.__import__

            def _blocking_import(name, *args, **kwargs):
                if name == "pysqlite3" or name.startswith("pysqlite3."):
                    raise ImportError("blocked by test")
                return _real_import(name, *args, **kwargs)

            builtins.__import__ = _blocking_import

        try:
            from gateway import run
        except Exception as exc:
            print(f"IMPORT_ERROR:{{type(exc).__name__}}:{{exc}}", file=sys.stderr)
            sys.exit(2)

        sqlite_mod = run.sqlite3
        sys_sqlite = sys.modules.get("sqlite3")
        print(json.dumps({{
            "gateway_name": getattr(sqlite_mod, "__name__", None),
            "gateway_marker": getattr(sqlite_mod, "SHIM_MARKER", None),
            "sys_name": getattr(sys_sqlite, "__name__", None),
            "sys_marker": getattr(sys_sqlite, "SHIM_MARKER", None),
        }}))
        """
    )
    env = dict(initial_env)
    env["HERMES_HOME"] = str(hermes_home)
    for key in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME"):
        if key in os.environ and key not in env:
            env[key] = os.environ[key]

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"gateway.run import failed (rc={result.returncode})\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    return json.loads(result.stdout)


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


@pytest.fixture
def fake_pysqlite3_root(tmp_path: Path) -> Path:
    root = tmp_path / "shim"
    root.mkdir()
    _write_fake_pysqlite3(root)
    return root


def test_gateway_import_selects_pysqlite3_before_top_level_sqlite_import(
    hermes_home: Path, fake_pysqlite3_root: Path,
) -> None:
    info = _run_gateway_import(hermes_home, fake_pysqlite3_root, initial_env={})

    assert info["gateway_name"] == "pysqlite3"
    assert info["gateway_marker"] == "fake-pysqlite3"
    assert info["sys_name"] == "pysqlite3"
    assert info["sys_marker"] == "fake-pysqlite3"


def test_gateway_import_uses_stdlib_sqlite_when_pysqlite3_absent(
    hermes_home: Path, tmp_path: Path,
) -> None:
    """Default install path: when pysqlite3 cannot be imported, gateway.run
    binds the stdlib sqlite3 module. We block the pysqlite3 import inside the
    subprocess so the result is deterministic regardless of whether the extra
    happens to be installed in the test venv."""
    empty_root = tmp_path / "empty-shim"
    empty_root.mkdir()
    info = _run_gateway_import(
        hermes_home, empty_root, initial_env={}, block_pysqlite3=True
    )

    assert info["gateway_name"] == "sqlite3"
    assert info["gateway_marker"] is None
    assert info["sys_name"] == "sqlite3"
    assert info["sys_marker"] is None
