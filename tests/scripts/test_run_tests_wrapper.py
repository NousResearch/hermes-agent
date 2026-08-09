from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE_RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_tests.sh"


def _python_launcher(path: Path, label: str) -> Path:
    bin_dir = path / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text("# test fixture\n", encoding="utf-8")
    launcher = bin_dir / "python"
    launcher.write_text(
        "#!/bin/sh\n"
        f"export SELECTED_LAUNCHER={label!r}\n"
        f"exec {str(Path(sys.executable))!r} \"$@\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


@pytest.fixture
def runner_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_tests.sh"
    shutil.copy2(SOURCE_RUNNER, runner)
    runner.chmod(0o755)
    (scripts / "run_tests_parallel.py").write_text(
        "import os\nprint('selected=' + os.environ['SELECTED_LAUNCHER'])\n",
        encoding="utf-8",
    )
    local = _python_launcher(repo / ".venv", "local")
    explicit = _python_launcher(tmp_path / "explicit-venv", "explicit")
    return repo, local, explicit


def _run(runner_repo: tuple[Path, Path, Path], *, override: bool) -> subprocess.CompletedProcess[str]:
    repo, _local, explicit = runner_repo
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(repo.parent / "home"),
    }
    if override:
        env["HERMES_PYTHON"] = str(explicit)
    return subprocess.run(
        [str(repo / "scripts" / "run_tests.sh"), "-q"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_explicit_hermes_python_overrides_discovered_local_venv(runner_repo):
    result = _run(runner_repo, override=True)

    assert result.returncode == 0, result.stderr
    assert "selected=explicit" in result.stdout
    assert "selected=local" not in result.stdout


def test_local_venv_is_used_when_no_explicit_override(runner_repo):
    result = _run(runner_repo, override=False)

    assert result.returncode == 0, result.stderr
    assert "selected=local" in result.stdout
