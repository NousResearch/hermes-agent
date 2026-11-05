"""Bundle commands run the prepared pnpm, including Nix's shell wrapper."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.bundles.desktop import pnpm_command


def test_bundle_pnpm_runs_workspace_from_unrelated_directory(tmp_path):
    node = shutil.which("node")
    assert node, "the build acceptance lane must provide Node/pnpm"
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    (workspace / "package.json").write_text(json.dumps({
        "name": "bundle-command-fixture", "version": "1.0.0",
        "scripts": {"build": "node probe.cjs"},
    }), encoding="utf-8")
    (workspace / "probe.cjs").write_text(
        "console.log(JSON.stringify({args:process.argv.slice(2),pnpm:process.env.npm_execpath}));"
        "process.exitCode=7", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if key.lower() != "npm_execpath"}
    env.update(HOME=str(tmp_path), pnpm_config_store_dir=str(tmp_path / "store"),
               pnpm_config_cache_dir=str(tmp_path / "cache"), pnpm_config_state_dir=str(tmp_path / "state"),
               pnpm_config_offline="true")
    result = subprocess.run([*pnpm_command(node), "--dir", str(workspace),
                             "run", "--silent", "build", "two words", "$(literal)"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 7, result.stdout + result.stderr
    child = json.loads(result.stdout)
    assert child["args"] == ["two words", "$(literal)"]
    assert "pnpm" in Path(child["pnpm"]).name
