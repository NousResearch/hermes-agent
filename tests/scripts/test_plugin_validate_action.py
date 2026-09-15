"""Behavioral coverage for the composite action's Hermes install source."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_FILE = REPO_ROOT / ".github" / "actions" / "plugin-validate" / "action.yml"


def _bash_executable() -> str | None:
    candidates: list[Path] = []
    if os.name == "nt":
        for name in ("ProgramFiles", "ProgramW6432"):
            if program_files := os.environ.get(name):
                candidates.append(Path(program_files) / "Git" / "bin" / "bash.exe")
    if bash := shutil.which("bash"):
        candidates.append(Path(bash))
    return next((str(path) for path in candidates if path.exists()), None)


def _install_script() -> str:
    yaml = pytest.importorskip("yaml")
    action = yaml.safe_load(ACTION_FILE.read_text(encoding="utf-8"))
    step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("name") == "Install hermes-agent"
    )
    return step["run"]


def _run_install_script(tmp_path: Path, hermes_ref: str) -> list[str]:
    # Prefer Git Bash on Windows: System32\bash.exe is the WSL launcher and
    # does not provide the shell contract used by a Windows Actions runner.
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is required to exercise the composite action shell")

    action_path = tmp_path / ".github" / "actions" / "plugin-validate"
    action_path.mkdir(parents=True)
    # The default branch resolves three parents from github.action_path.
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'fake-hermes'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    calls_file = tmp_path / "pip-calls"
    mock_pip = mock_bin / "pip"
    mock_pip.write_bytes(
        b"#!/usr/bin/env bash\n"
        b'printf "%s\\n" "$@" > "$MOCK_PIP_CALLS"\n',
    )
    mock_pip.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "_HERMES_REF": hermes_ref,
            "_ACTION_PATH": action_path.as_posix(),
            "MOCK_PIP_CALLS": calls_file.as_posix(),
            "PATH": os.pathsep.join((mock_bin.as_posix(), env.get("PATH", ""))),
        }
    )
    result = subprocess.run(
        [bash, "-c", _install_script()],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return calls_file.read_text(encoding="utf-8").splitlines()


def test_install_source_is_local_by_default_and_explicit_ref_when_requested(
    tmp_path: Path,
) -> None:
    """The action never contacts moving upstream/main implicitly."""

    # Keep spaces in the checkout path so the quoted local install argument is
    # exercised rather than merely inspected.
    default_root = tmp_path / "default checkout"
    default_root.mkdir(parents=True)
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is required to exercise the composite action shell")
    expected_root = subprocess.run(
        [bash, "-c", 'cd "$1" && pwd', "bash", default_root.as_posix()],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    assert _run_install_script(default_root, "") == [
        "install",
        expected_root,
    ]
    assert _run_install_script(tmp_path / "explicit", "v0.21.1") == [
        "install",
        "git+https://github.com/NousResearch/hermes-agent@v0.21.1",
    ]
