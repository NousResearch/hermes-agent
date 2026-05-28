import json
import os
import subprocess
import sys
from pathlib import Path


def run_hermes(tmp_path, *args):
    env = os.environ.copy()
    env.update({"HERMES_HOME": str(tmp_path / ".hermes"), "PYTHONPATH": str(Path.cwd())})
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_project_init_cli_bootstraps_home(tmp_path):
    project_home = tmp_path / "projects" / "demo"
    result = run_hermes(
        tmp_path,
        "project",
        "init",
        "--slug",
        "demo",
        "--title",
        "Demo",
        "--goal",
        "Make demo restartable",
        "--board",
        "demo",
        "--root-task",
        "t_root",
        "--project-home",
        str(project_home),
        "--repo-org",
        "summation",
        "--repo-name",
        "Code",
        "--canonical-repo",
        "/Users/vsletten/src/summation/Code/main",
        "--final-branch",
        "feat/demo-pr",
    )

    assert result.returncode == 0, result.stderr
    assert "BOOTSTRAPPED" in result.stdout
    doc = json.loads((project_home / "project.json").read_text())
    assert doc["slug"] == "demo"
