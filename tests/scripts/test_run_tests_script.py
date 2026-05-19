"""Tests for scripts/run_tests.sh guardrails."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_with_fake_python(tmp_path: Path, script_args: list[str]):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise scripts/run_tests.sh")

    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts"
    venv_bin = repo / ".venv" / "bin"
    scripts_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)

    script_dst = scripts_dir / "run_tests.sh"
    shutil.copy2(REPO_ROOT / "scripts" / "run_tests.sh", script_dst)

    (venv_bin / "activate").write_text("", encoding="utf-8")
    fake_python = venv_bin / "python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail

            if [ "${1:-}" = "-c" ]; then
              exit 0
            fi

            {
              printf 'nofile=%s\\n' "$(ulimit -n)"
              printf 'args='
              printf '[%s]' "$@"
              printf '\\n'
            } >> "$FAKE_PYTHON_LOG"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    log_path = repo / "fake-python.log"
    quoted_args = " ".join(script_args)
    env = {
        **os.environ,
        "FAKE_PYTHON_LOG": str(log_path),
        "HERMES_TEST_NOFILE_LIMIT": "128",
    }
    result = subprocess.run(
        [
            bash,
            "-c",
            f"ulimit -S -n 64 || exit 77; scripts/run_tests.sh {quoted_args}",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("shell could not lower the soft open-file limit")

    return result, log_path


@pytest.mark.skipif(sys.platform == "win32", reason="ulimit is POSIX-only")
def test_run_tests_raises_soft_nofile_limit_before_pytest(tmp_path):
    """The canonical runner must raise macOS's low default before xdist starts."""
    result, log_path = _run_with_fake_python(tmp_path, ["tests/example.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "raised open-file limit from 64 to 128" in result.stdout

    log_text = log_path.read_text(encoding="utf-8")
    nofile_line = next(line for line in log_text.splitlines() if line.startswith("nofile="))
    assert int(nofile_line.removeprefix("nofile=")) >= 128
    assert "[-m][pytest]" in log_text
    assert "[-n][4]" in log_text


@pytest.mark.skipif(sys.platform == "win32", reason="ulimit is POSIX-only")
def test_run_tests_allows_no_pytest_args(tmp_path):
    """The full-suite invocation must not trip set -u before pytest starts."""
    result, log_path = _run_with_fake_python(tmp_path, [])

    assert result.returncode == 0, result.stdout + result.stderr

    log_text = log_path.read_text(encoding="utf-8")
    assert "[-m][pytest]" in log_text
    assert "[tests/example.py]" not in log_text
