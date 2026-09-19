"""The canonical runner must reject environments unable to run async tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="Controlled interpreter fixtures require POSIX executable scripts and bash",
)


@pytest.fixture
def runner_tree(tmp_path):
    repo = tmp_path / "checkout"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts" / "run_tests.sh",
        scripts / "run_tests.sh",
    )
    home = tmp_path / "home"
    home.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text("#!/bin/sh\nexit 0\n")
    git.chmod(0o755)
    return repo, home, commands


def _interpreter(venv: Path, *, complete: bool, layout: str = "bin") -> Path:
    """Execute real import probes with isolated stub packages, never host packages."""
    binaries = venv / layout
    binaries.mkdir(parents=True)
    (binaries / "activate").touch()
    modules = venv / "modules"
    modules.mkdir()
    (modules / "pytest.py").touch()
    if complete:
        (modules / "pytest_asyncio.py").touch()
    python = binaries / ("python.exe" if layout == "Scripts" else "python")
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        f"    env = dict(os.environ, PYTHONPATH={str(modules)!r})\n"
        "    raise SystemExit(subprocess.call([sys.executable, '-S', *sys.argv[1:]], env=env))\n"
        f"print('SELECTED:' + {json.dumps(str(python))} + ':' + ' '.join(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    return python


def _run(tree, *, explicit_python: Path | None = None):
    repo, home, commands = tree
    env = {
        "HOME": str(home),
        "PATH": str(commands) + os.pathsep + os.defpath,
    }
    if explicit_python is not None:
        env["HERMES_PYTHON"] = str(explicit_python)
    return subprocess.run(
        ["bash", str(repo / "scripts" / "run_tests.sh"), "tests/test_example.py"],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )


@pytest.mark.parametrize("layout", ["bin", "Scripts"])
def test_incomplete_local_venv_does_not_hide_usable_test_environment(runner_tree, layout):
    repo, _, _ = runner_tree
    rejected = _interpreter(repo / ".venv", complete=False, layout=layout)
    selected = _interpreter(repo / "venv", complete=True, layout=layout)

    result = _run(runner_tree)

    assert result.returncode == 0, result.stderr
    assert f"SELECTED:{selected}:" in result.stdout
    assert f"SELECTED:{rejected}:" not in result.stdout
    assert "run_tests_parallel.py tests/test_example.py" in result.stdout
    assert "pytest_asyncio" in result.stderr


def test_incomplete_release_venv_falls_back_to_explicit_dev_interpreter(runner_tree):
    repo, home, _ = runner_tree
    rejected = _interpreter(home / ".hermes" / "hermes-agent" / "venv", complete=False)
    selected = _interpreter(repo / "external-dev", complete=True)

    result = _run(runner_tree, explicit_python=selected)

    assert result.returncode == 0, result.stderr
    assert f"SELECTED:{selected}:" in result.stdout
    assert f"SELECTED:{rejected}:" not in result.stdout
    assert "pytest_asyncio" in result.stderr


def test_missing_async_plugin_fails_before_compilation_or_test_dispatch(runner_tree):
    repo, home, _ = runner_tree
    _interpreter(home / ".hermes" / "hermes-agent" / "venv", complete=False)
    explicit = _interpreter(repo / "external-dev", complete=False)

    result = _run(runner_tree, explicit_python=explicit)

    assert result.returncode != 0
    assert "SELECTED:" not in result.stdout
    assert "pytest_asyncio" in result.stderr


def test_complete_local_environment_needs_no_optional_test_plugins(runner_tree):
    repo, home, _ = runner_tree
    selected = _interpreter(repo / ".venv", complete=True)
    unused = _interpreter(home / ".hermes" / "hermes-agent" / "venv", complete=True)

    result = _run(runner_tree)

    assert result.returncode == 0, result.stderr
    assert f"SELECTED:{selected}:" in result.stdout
    assert f"SELECTED:{unused}:" not in result.stdout
