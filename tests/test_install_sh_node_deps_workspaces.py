"""Bootstrap does not ask npm to resolve an unrelated desktop workspace."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_node_stage_does_not_invoke_ambient_npm(tmp_path):
    sentinel = tmp_path / "npm-called"
    script = f"""source "{(ROOT / 'scripts/install.sh').as_posix()}" --manifest
npm() {{ touch "{sentinel.as_posix()}"; return 99; }}
INSTALL_DIR="{tmp_path.as_posix()}"
stage_node_deps
"""
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=tmp_path.as_posix())
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()
