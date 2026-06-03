from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _launcher_namespace() -> dict[str, object]:
    return runpy.run_path(str(REPO_ROOT / "hermes"), run_name="hermes_launcher_test")


def test_root_hermes_launcher_prefers_dot_venv(tmp_path: Path) -> None:
    namespace = _launcher_namespace()
    dot_venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python = tmp_path / "venv" / "bin" / "python"
    dot_venv_python.parent.mkdir(parents=True)
    venv_python.parent.mkdir(parents=True)
    dot_venv_python.write_text("")
    venv_python.write_text("")

    candidate_local_python = namespace["_candidate_local_python"]

    assert candidate_local_python(tmp_path) == dot_venv_python


def test_root_hermes_launcher_reexecs_with_repo_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _launcher_namespace()
    script = tmp_path / "hermes"
    local_python = tmp_path / "venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    script.write_text("")
    local_python.write_text("")
    calls: list[tuple[str, list[str]]] = []

    def fake_execv(path: str, argv: list[str]) -> None:
        calls.append((path, argv))
        raise SystemExit

    monkeypatch.setattr(os, "execv", fake_execv)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(sys, "argv", ["./hermes", "--version"])

    with pytest.raises(SystemExit):
        namespace["_maybe_reexec_from_local_venv"](script)

    resolved_python = str(local_python.resolve())
    assert calls == [(resolved_python, [resolved_python, str(script), "--version"])]


def test_root_hermes_launcher_skips_reexec_inside_repo_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _launcher_namespace()
    script = tmp_path / "hermes"
    local_python = tmp_path / "venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    script.write_text("")
    local_python.write_text("")

    def fail_execv(path: str, argv: list[str]) -> None:
        raise AssertionError(f"unexpected execv({path!r}, {argv!r})")

    monkeypatch.setattr(os, "execv", fail_execv)
    monkeypatch.setattr(sys, "executable", str(local_python.absolute()))

    namespace["_maybe_reexec_from_local_venv"](script)


def test_root_hermes_launcher_reexecs_when_sys_executable_is_resolved_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _launcher_namespace()
    script = tmp_path / "hermes"
    local_python = tmp_path / "venv" / "bin" / "python"
    base_python = tmp_path / "base-python"
    local_python.parent.mkdir(parents=True)
    script.write_text("")
    base_python.write_text("")
    local_python.symlink_to(base_python)
    calls: list[tuple[str, list[str]]] = []

    def fake_execv(path: str, argv: list[str]) -> None:
        calls.append((path, argv))
        raise SystemExit

    monkeypatch.setattr(os, "execv", fake_execv)
    monkeypatch.setattr(sys, "executable", str(base_python))
    monkeypatch.setattr(sys, "argv", ["./hermes"])

    with pytest.raises(SystemExit):
        namespace["_maybe_reexec_from_local_venv"](script)

    absolute_python = str(local_python.absolute())
    assert calls == [(absolute_python, [absolute_python, str(script)])]
