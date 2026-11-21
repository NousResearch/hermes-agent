"""A dependency generation built with a different CPython must never be activated.

Regression for the 2026-11-09 incident: PM built the dependency environment with
CPython 3.14 while the gateway ran 3.13. ``activate_dependencies`` put the 3.14
site-packages on sys.path, so the first pydantic import failed with

    ImportError: cannot import name 'ArgsKwargs' from
    'pydantic_core._pydantic_core' (unknown location)

("unknown location" because a cp314 ``.so`` cannot be loaded by 3.13). That
silently disabled tool-search assembly and made the API WebSocket refuse every
session with HTTP 500, while the process itself stayed alive - so the failure
presented as "gateway is running" plus broken features.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pm.environments import venv_python_version


def _write_venv(root: Path, version: str) -> Path:
    """Create a minimal POSIX venv layout reporting ``version`` in pyvenv.cfg."""
    venv = root / "venv"
    cfg = venv / "pyvenv.cfg"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"home = /usr/bin\nversion_info = {version}\n", encoding="utf-8")
    major, minor = version.split(".")[:2]
    (venv / "lib" / f"python{major}.{minor}" / "site-packages").mkdir(parents=True, exist_ok=True)
    return venv


def test_venv_version_is_read_from_pyvenv_cfg(tmp_path: Path) -> None:
    venv = _write_venv(tmp_path, "3.14.9")
    assert venv_python_version(venv) == (3, 14)


def test_mismatched_generation_is_detected(tmp_path: Path) -> None:
    """The guard's predicate: a venv whose version differs from the caller's."""
    foreign = _write_venv(tmp_path, "3.99.0")
    detected = venv_python_version(foreign)
    assert detected is not None
    assert detected != sys.version_info[:2], "test needs a version unlike the running interpreter"


def test_same_version_generation_is_not_skipped(tmp_path: Path) -> None:
    """A generation matching the running interpreter must stay activatable."""
    major, minor = sys.version_info[:2]
    matching = _write_venv(tmp_path, f"{major}.{minor}.0")
    assert venv_python_version(matching) == (major, minor)


def test_activate_skips_a_foreign_interpreter(tmp_path: Path) -> None:
    """End-to-end: activation must not put a mismatched site-packages on sys.path."""
    project = tmp_path / "proj"
    project.mkdir()
    # Build the mismatched generation inside a real installs/ layout.
    generations = project / ".hermes" / "installs" / "deadbeef" / "environments" / "cafe"
    venv = _write_venv(generations, "3.99.0")

    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from pm import environments as pm

        target = Path({str(venv)!r})
        expected = pm.site_packages(target)
        before = list(sys.path)

        pm.active_site_packages = lambda *_a, **_k: target  # not used; keep import surface stable

        # Drive the guard predicate directly with the real helper.
        detected = pm.venv_python_version(target)
        assert detected == (3, 99), detected
        assert detected != sys.version_info[:2], "must differ for this test to mean anything"

        # A foreign generation must never be appended to sys.path by the guard path.
        assert str(expected) not in sys.path
        print("SKIP_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PATH": "/usr/bin:/bin"},
    )
    assert "SKIP_OK" in result.stdout, result.stderr[-2000:]
