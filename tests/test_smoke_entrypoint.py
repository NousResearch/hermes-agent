import os
import subprocess
from pathlib import Path


def test_root_smoke_script_runs_without_credentials(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = repo_root / "smoke.sh"

    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        ["bash", str(smoke)],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout
    assert "ok: smoke passed" in result.stdout
