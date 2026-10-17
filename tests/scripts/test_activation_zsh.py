"""``scripts/_activation.sh`` stays usable when ``source ./activate`` runs under zsh.

zsh's default ``nomatch`` aborts a function the moment a glob matches nothing, where
bash passes the pattern through. ``hermes_bootstrap_python`` probes several runtime
stores for ``python-*``; the first (``<repo>/../tools``) usually does not exist, so
zsh died there and never reached the store that holds the interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(shutil.which("zsh") is None, reason="needs zsh"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATION = REPO_ROOT / "scripts" / "_activation.sh"


def _bootstrap_python_under_zsh(repo: Path, hermes_home: Path) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ["PATH"], "HOME": str(hermes_home.parent), "HERMES_HOME": str(hermes_home)}
    return subprocess.run(
        ["zsh", "-f", "-c", f'source "{ACTIVATION}" && hermes_bootstrap_python "{repo}"'],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )


def test_bootstrap_python_found_past_stores_with_no_match(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()  # no .venv, and no <repo>/../tools: those globs match nothing
    python = tmp_path / "home" / "tools" / "python-3.14.0" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    result = _bootstrap_python_under_zsh(repo, tmp_path / "home")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)


def test_bootstrap_python_reports_no_interpreter_without_a_zsh_error(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()

    result = _bootstrap_python_under_zsh(repo, tmp_path / "home")

    assert result.returncode == 1
    assert "no matches found" not in result.stderr
