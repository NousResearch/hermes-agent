"""``python -m scripts.code_health.ruff_runner`` runs the pyproject-pinned ruff, whatever is on PATH."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.code_health import ruff_runner
from scripts.code_health.ruff_runner import pinned_version

REPO = Path(__file__).resolve().parents[2]


def test_resolver_uses_the_activated_test_environment(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[dependency-groups]\ndev = ["ruff==0.15.22"]\n', encoding="utf-8",
    )
    test_bin = tmp_path / "test-environment" / "bin"
    test_bin.mkdir(parents=True)
    test_python = test_bin / "python"
    test_python.touch()
    ruff = test_bin / "ruff"
    ruff.touch()
    monkeypatch.setenv("__HERMES_TEST_PYTHON", str(test_python))
    monkeypatch.setattr(ruff_runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        ruff_runner, "_version_of", lambda command: "0.15.22" if command == [str(ruff)] else None,
    )

    assert ruff_runner.resolve_ruff(repo) == [str(ruff)]


def test_module_runs_the_pinned_ruff_with_the_given_arguments() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.code_health.ruff_runner", "--version"],
        cwd=REPO, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ruff", pinned_version(REPO)]


def test_module_forwards_the_ruff_exit_status(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.code_health.ruff_runner", "check", "--isolated",
         "--no-cache", "--select", "F401", str(bad)],
        cwd=REPO, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "F401" in proc.stdout
