"""_build_web_ui: treat dist output as success signal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as m

    return m


@dataclass
class _RunResult:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""


def test_build_web_ui_nonzero_exit_but_dist_exists_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main_mod
) -> None:
    (tmp_path / "package.json").write_text("{}")

    dist = tmp_path / "dist"
    dist.mkdir()
    # If dist exists before build, it should be removed to avoid stale success.
    (dist / "index.html").write_text("stale")

    monkeypatch.setattr(main_mod.shutil, "which", lambda _: "npm")

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], cwd: Path, capture_output: bool):  # noqa: ANN001
        calls.append(cmd)
        if cmd[:2] == ["npm", "install"]:
            return _RunResult(returncode=0)
        if cmd[:3] == ["npm", "run", "build"]:
            # Simulate a Windows false-negative exit code, but write dist output.
            out = cwd / "dist"
            out.mkdir(exist_ok=True)
            (out / "index.html").write_text("ok")
            return _RunResult(returncode=1)
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(main_mod.subprocess, "run", _fake_run)

    assert main_mod._build_web_ui(tmp_path, fatal=True) is True
    assert calls == [["npm", "install", "--silent"], ["npm", "run", "build"]]


def test_build_web_ui_nonzero_exit_without_dist_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main_mod
) -> None:
    (tmp_path / "package.json").write_text("{}")
    monkeypatch.setattr(main_mod.shutil, "which", lambda _: "npm")

    def _fake_run(cmd: list[str], cwd: Path, capture_output: bool):  # noqa: ANN001
        if cmd[:2] == ["npm", "install"]:
            return _RunResult(returncode=0)
        if cmd[:3] == ["npm", "run", "build"]:
            return _RunResult(returncode=1)
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(main_mod.subprocess, "run", _fake_run)

    assert main_mod._build_web_ui(tmp_path, fatal=True) is False

