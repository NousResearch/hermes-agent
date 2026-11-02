"""A corrupt entry in PM's uv cache must not fail every later install.

A machine stop during an install can leave an unpacked wheel in uv's cache
whose files read back as NUL bytes: uv renames its unpack directory into
``archive-v0`` without flushing the data first. uv then reuses that entry on
every sync and fails with "The wheel is invalid", so retrying never helps. PM
purges the damaged package(s) through ``uv cache clean`` and retries, and
gives up with a clear message when a fresh copy is broken too.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.environment import PythonEnvironment
from pm.uv_cache_health import corrupt_cached_packages, failed_cached_packages
from tests.pm._fixtures import _run, _wheel

@pytest.mark.parametrize("output", [
    "error: No solution found when resolving dependencies:\n  Because alpha==2 was not found ...",
    "error: Failed to fetch: `https://pypi.org/simple/alpha/`\n  Caused by: Request failed after 3 retries",
    "error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
    "  cause: failed to hardlink file from /c/archive-v0/abc/alpha/__init__.py to /v/lib/alpha/__init__.py: "
    "No space left on device (os error 28)",
    "error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
    "  cause: failed to create directory `/v/lib/alpha`: Permission denied (os error 13)",
    "  Caused by: The wheel is invalid: Invalid Wheel-Version",  # no install step to name a package
    "error: Failed to install: --force-1.0-py3-none-any.whl (x==1)\n  cause: The wheel is invalid: x",
])
def test_other_failures_are_not_cache_corruption(output):
    assert failed_cached_packages(output) == set()


@pytest.fixture
def project(tmp_path):
    uv = shutil.which("uv")
    assert uv, "cache recovery tests require uv on PATH"
    source = tmp_path / "source"
    source.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in ("alpha_dep", "beta_dep"):
        _wheel(wheels, name)
    (source / "pyproject.toml").write_text(
        '[project]\nname="cache-root"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["alpha-dep==1.0", "beta-dep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    env.update(HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home), XDG_CONFIG_DIRS=str(home))
    environment = _environment(Path(uv), tmp_path, "first", env)
    environment.lock(source, timeout=120)
    environment.create()
    environment.sync(source, timeout=120)  # populates the cache
    return source, Path(uv), env, wheels


def _environment(uv: Path, root: Path, name: str, env, output=None) -> PythonEnvironment:
    return PythonEnvironment(uv=uv, python=Path(sys.executable), destination=root / name / "venv",
                             cache=root / "cache", env=env, offline=True, output=output)


def _cached(cache: Path, name: str) -> Path:
    matches = list((cache / "archive-v0").glob(f"*/{name}-1.0.dist-info"))
    assert len(matches) == 1, matches
    return matches[0]


def _nul_fill(path: Path) -> None:
    """What a crash leaves: the file keeps its size, its data never reached the disk."""
    size = max(path.stat().st_size, 1)
    path.write_bytes(b"\0" * size)


def _record_uv_calls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []
    real = PythonEnvironment._run

    def recording(self, args, *, cwd, timeout):
        calls.append(list(args))
        return real(self, args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(PythonEnvironment, "_run", recording)
    return calls


def test_sync_recovers_from_nul_filled_cache_entries(project, tmp_path, monkeypatch):
    """The live shape: two packages damaged by one crash, the first a NUL WHEEL file."""
    source, uv, env, _ = project
    cache = tmp_path / "cache"
    _nul_fill(_cached(cache, "alpha_dep") / "WHEEL")
    for part in _cached(cache, "beta_dep").iterdir():
        _nul_fill(part)
    assert corrupt_cached_packages(cache) == {"alpha-dep", "beta-dep"}
    calls = _record_uv_calls(monkeypatch)
    notes = io.StringIO()

    second = _environment(uv, tmp_path, "second", env, output=notes)
    second.create()
    second.sync(source, timeout=120)

    assert _run([str(second.executable), "-I", "-c", "import alpha_dep, beta_dep; print(alpha_dep.__version__)"],
                cwd=tmp_path, env=env) == "1.0"
    # One purge covered both packages, then exactly one retry.
    assert [c[:2] for c in calls].count(["cache", "clean"]) == 1
    assert ["cache", "prune"] not in [c[:2] for c in calls]  # prune breaks PM's seeded cache
    assert sorted(next(c for c in calls if c[:2] == ["cache", "clean"])[2:]) == ["alpha-dep", "beta-dep"]
    assert sum(c[0] == "sync" for c in calls) == 2
    assert corrupt_cached_packages(cache) == set()
    assert "alpha-dep, beta-dep is corrupt" in notes.getvalue()
