"""First-run guidance on the real CLI without a configured executor."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("argv", [[], ["chat"], ["-z", "hello"]])
def test_headless_first_run_exits_with_setup_guidance_before_gateway(tmp_path, argv):
    home = tmp_path / "state"
    home.mkdir(mode=0o700)
    user = tmp_path / "user"
    user.mkdir()
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *argv], cwd=tmp_path, env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "hermes config set model.provider custom" in result.stdout
    assert "Run setup now?" not in result.stdout
    assert "Session:" not in result.stderr
    assert not (home / "gateway.pid").exists()
    assert not (home / "state.db").exists()
