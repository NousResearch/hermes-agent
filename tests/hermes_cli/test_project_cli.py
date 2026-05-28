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


def test_project_cleanup_inventory_cli_writes_inventory(tmp_path):
    project_home = tmp_path / "projects" / "demo"
    init_result = run_hermes(
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
    assert init_result.returncode == 0, init_result.stderr

    doc = json.loads((project_home / "project.json").read_text())
    doc["repo"]["worktree_namespace"] = str(tmp_path / "src" / "summation" / "Code")
    doc["final_worktree_path"] = str(
        tmp_path / "src" / "summation" / "Code" / "feat" / "demo-pr"
    )
    doc["task_graph"] = {
        "nodes": [
            {
                "id": "t_done",
                "title": "slice one",
                "status": "done",
                "workspace_kind": "worktree",
                "workspace_path": str(
                    tmp_path / "src" / "summation" / "Code" / "feat" / "slice-one"
                ),
                "branch_name": "feat/slice-one",
            }
        ],
        "edges": [],
    }
    (project_home / "project.json").write_text(json.dumps(doc), encoding="utf-8")

    result = run_hermes(tmp_path, "project", "cleanup-inventory", str(project_home))

    assert result.returncode == 0, result.stderr
    assert "CLEANUP_INVENTORY" in result.stdout
    saved = json.loads((project_home / "project.json").read_text())
    assert saved["cleanup"]["state"] == "inventory_ready"
    assert Path(saved["cleanup"]["inventory_path"]).exists()
