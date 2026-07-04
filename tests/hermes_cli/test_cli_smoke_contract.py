import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERMES = ROOT / "hermes"


def _run_hermes(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    return subprocess.run(
        [sys.executable, str(HERMES), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_cli_help_smoke_proves_launcher_imports_and_argparse():
    result = _run_hermes("--help")

    assert result.returncode == 0
    assert "Hermes Agent - AI assistant" in result.stdout
    assert "usage: hermes" in result.stdout
    assert result.stderr == ""


def test_cli_bad_input_reports_clear_nonzero_failure():
    result = _run_hermes("--not-a-real-hermes-flag")

    assert result.returncode == 2
    assert "usage: hermes" in result.stderr
    assert "unrecognized arguments: --not-a-real-hermes-flag" in result.stderr
