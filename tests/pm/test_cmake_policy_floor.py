"""Every uv build tier hands cmake a policy floor.

cmake 4.x removed compatibility with ``cmake_minimum_required(VERSION < 3.5)``,
so an sdist that still declares an older minimum aborts at CONFIGURE time
before a compiler is ever reached. python-olm 3.2.16 (pulled in by
``mautrix[encryption]``) vendors libolm with ``VERSION 3.4``. Without
``CMAKE_POLICY_VERSION_MINIMUM`` the install dies, so this pins the contract on
the env uv actually spawns build backends with — the ``uv sync`` tier and the
``uv pip install`` requirements tier alike, since they are the two ways a
sdist can enter a PM environment.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _record_uv_envs(monkeypatch) -> list[dict]:
    """Capture the env of every env-bearing uv spawn, reporting success."""
    seen: list[dict] = []

    def record(command, **kwargs):
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", record)
    return seen


def _environment(tmp_path: Path):
    from pm.environment import PythonEnvironment

    return PythonEnvironment(uv=tmp_path / "uv", python=Path(sys.executable),
                             destination=tmp_path / "venv", cache=tmp_path / "cache", env={})


def test_both_install_tiers_receive_the_cmake_policy_floor(tmp_path, monkeypatch):
    """`uv sync` and `uv pip install` are independent spawns; neither may be
    left without the floor, or a sdist compiles on one path and fails on the other."""
    seen = _record_uv_envs(monkeypatch)

    _environment(tmp_path).sync(tmp_path)
    _environment(tmp_path).install_requirements(["python-olm==3.2.16"])

    # One spawn each: sync, then the requirements tier.
    assert len(seen) == 2
    assert [env["CMAKE_POLICY_VERSION_MINIMUM"] for env in seen] == ["3.5", "3.5"]


def test_ambient_cmake_policy_version_is_not_overwritten(tmp_path, monkeypatch):
    """The floor is a default, not a mandate: a user who exports their own
    policy version (a newer minimum, or a build that must see none) keeps it.
    Resolved through managed_environment, which is where the ambient shell env
    actually becomes the build env for a lazy extra install."""
    from pm.environment import managed_environment

    monkeypatch.setenv("CMAKE_POLICY_VERSION_MINIMUM", "3.10")
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (tmp_path / "uv", Path(sys.executable)))
    seen = _record_uv_envs(monkeypatch)

    managed_environment(tmp_path / "candidate", cache=tmp_path).sync(tmp_path)

    assert seen[0]["CMAKE_POLICY_VERSION_MINIMUM"] == "3.10"
