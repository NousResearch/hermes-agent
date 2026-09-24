"""Behavioral tests for the canonical shell test runner."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _isolated_runner_repo(tmp_path: Path) -> Path:
    """Copy the executable runner into a repo with no virtualenv."""
    source_root = Path(__file__).resolve().parent.parent
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(source_root / "scripts" / "run_tests.sh", scripts / "run_tests.sh")
    (scripts / "run_tests_parallel.py").write_text(
        "import pytest, sys\n"
        "print(f'RUNNER_PYTHON={sys.executable}')\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return repo


def test_shell_runner_falls_back_to_python_on_path(tmp_path: Path) -> None:
    """A checkout without a configured venv uses the invoking Python command."""
    bash = shutil.which("bash")
    python = shutil.which("python")
    assert bash is not None, "run_tests.sh requires bash"
    assert python is not None, "test requires the invoking python command"

    expected = subprocess.run(
        [python, "-c", "import sys; print(sys.executable)"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()
    repo = _isolated_runner_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("HERMES_PYTHON", None)

    proc = subprocess.run(
        [bash, str(repo / "scripts" / "run_tests.sh")],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout
    assert "using invoking Python from PATH" in proc.stdout
    runner_python = next(
        line.removeprefix("RUNNER_PYTHON=")
        for line in proc.stdout.splitlines()
        if line.startswith("RUNNER_PYTHON=")
    )
    assert os.path.normcase(runner_python) == os.path.normcase(expected)


def test_shell_runner_failure_lists_every_probe_and_direct_command(
    tmp_path: Path,
) -> None:
    """A missing pytest interpreter reports evidence and a direct equivalent."""
    bash = shutil.which("bash")
    assert bash is not None, "run_tests.sh requires bash"

    repo = _isolated_runner_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = "/usr/bin"
    env["HERMES_PYTHON"] = "/usr/bin/false"

    proc = subprocess.run(
        [
            bash,
            str(repo / "scripts" / "run_tests.sh"),
            "tests/test_example.py",
            "-q",
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert proc.returncode == 1, proc.stdout
    assert "no Python interpreter with pytest was found" in proc.stdout
    assert "Probed virtualenvs:" in proc.stdout
    rendered = proc.stdout.replace("\\", "/")
    assert "/repo/.venv" in rendered
    assert "/repo/venv" in rendered
    assert "/home/.hermes/hermes-agent/venv" in rendered
    assert "Probed HERMES_PYTHON: /usr/bin/false" in proc.stdout
    assert "Probed invoking Python from PATH: <not found>" in proc.stdout
    assert "python -m pytest tests/test_example.py -q" in proc.stdout
