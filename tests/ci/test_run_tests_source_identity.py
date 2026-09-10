from __future__ import annotations

import subprocess
from pathlib import Path


def test_worktree_runner_never_borrows_live_editable_venv(tmp_path: Path) -> None:
    repository = tmp_path / "worktree"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "scripts" / "run_tests.sh"
    runner = scripts / "run_tests.sh"
    runner.write_bytes(source.read_bytes())

    fake_python = tmp_path / "home" / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    (fake_python.parent / "activate").touch()

    completed = subprocess.run(
        ["bash", str(runner), "tests"],
        cwd=repository,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "no virtualenv with pytest found" in completed.stderr
    assert "using Nix dev venv" not in completed.stdout
