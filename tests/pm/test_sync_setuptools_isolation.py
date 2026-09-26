"""RED for #123540: pm's uv sync must bypass the uv 0.12.3 setuptools-83 WHEEL parse bug.

uv 0.12.3 fails with "Invalid Wheel-Version in WHEEL file: None" when
installing setuptools-83.0.0 into an isolated build env, even though the
wheel is valid. The working bypass is `--no-build-isolation-package
setuptools`, which has no env-var/config equivalent and is forced
invisible by pm's --no-config + XDG redirection, so pm itself must pass it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def test_sync_bypasses_isolated_setuptools_build(tmp_path, monkeypatch):
    from pm.environment import PythonEnvironment

    source = tmp_path / "project"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="setuptools-bypass"\nversion="1"\n', encoding="utf-8"
    )
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    env = PythonEnvironment(
        uv=Path("/fake/uv"), python=Path("/fake/python"),
        destination=tmp_path / "venv", cache=tmp_path / "cache", env={},
    )
    env.sync(source)
    assert seen, "sync must invoke uv"
    cmd = seen[0]
    assert "--no-build-isolation-package" in cmd, f"missing bypass flag: {cmd}"
    idx = cmd.index("--no-build-isolation-package")
    assert cmd[idx + 1] == "setuptools", f"bypass must target setuptools: {cmd}"
