from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "workflows" / "dashboard"


def test_workflows_plugin_build_matches_tracked_assets(tmp_path):
    env = os.environ.copy()
    env["WORKFLOWS_PLUGIN_OUT_DIR"] = str(tmp_path)
    subprocess.run(
        ["npm", "run", "build:workflows", "--workspace", "web"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert (tmp_path / "index.js").read_bytes() == (PLUGIN / "dist/index.js").read_bytes()
    assert (tmp_path / "style.css").read_bytes() == (PLUGIN / "dist/style.css").read_bytes()