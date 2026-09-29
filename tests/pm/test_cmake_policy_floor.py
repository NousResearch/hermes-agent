"""uv's sdist build children must tolerate CMake 4's raised policy floor (#127795).

python-olm vendors libolm's ``CMakeLists.txt`` with ``cmake_minimum_required(VERSION 3.4)``;
CMake >= 4 refuses to configure it unless ``CMAKE_POLICY_VERSION_MINIMUM`` relaxes the
floor. The build backend inherits PM's uv environment, so ``PythonEnvironment._run`` seeds
the variable while an explicit user export still wins.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _environment(tmp_path: Path):
    from pm.environment import PythonEnvironment

    return PythonEnvironment(
        uv=Path("/nonexistent/uv"), python=Path(sys.executable),
        destination=tmp_path / "venv", cache=tmp_path / "cache", env=None,
    )


def _child_env_after_sync(tmp_path: Path, monkeypatch) -> dict[str, str]:
    captured: dict[str, dict[str, str]] = {}

    def fake_run(command, *, cwd, env, capture_output, text, encoding, errors, timeout):
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0, "", "")

    import pm.environment

    monkeypatch.setattr(pm.environment.subprocess, "run", fake_run)
    environment = _environment(tmp_path)
    environment.sync(tmp_path, frozen=True)
    assert captured, "the sync path must reach the uv subprocess"
    return captured["env"]


def test_uv_children_carry_the_cmake_policy_floor(tmp_path, monkeypatch):
    child_env = _child_env_after_sync(tmp_path, monkeypatch)
    assert child_env.get("CMAKE_POLICY_VERSION_MINIMUM") == "3.5"


def test_explicit_cmake_policy_floor_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("CMAKE_POLICY_VERSION_MINIMUM", "3.10")
    child_env = _child_env_after_sync(tmp_path, monkeypatch)
    assert child_env["CMAKE_POLICY_VERSION_MINIMUM"] == "3.10"
