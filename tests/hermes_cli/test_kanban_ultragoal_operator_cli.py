from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _authority_file(tmp_path: Path, task_id="BO-203") -> Path:
    p = tmp_path / "authority.json"
    p.write_text(json.dumps({
        "authority": "kanban",
        "taskId": task_id,
        "publicId": task_id,
        "status": "ready",
        "routingVerdict": "direct-kanban",
        "executionApproved": True,
        "snapshotHash": "sha256:a",
        "doneCriteriaHash": "sha256:d",
        "doneCriteria": ["ship reviewed PR"],
    }))
    return p


def test_operator_cli_run_resume_status_and_parent_mode(tmp_path):
    authority = _authority_file(tmp_path)
    cmd = [sys.executable, "-m", "hermes_cli.main", "kanban-ultragoal", "--workdir", str(tmp_path), "--json"]

    run = subprocess.run(
        cmd + ["run", "BO-203", "--authority-json", str(authority), "--root-objective", "Bring one reviewed PR"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    run_data = json.loads(run.stdout)
    assert run_data["state"] == "running"
    assert run_data["dispatcherUsed"] is False
    assert run_data["pendingAction"]["executor"] == "hermes-direct-goal-loop"

    status = subprocess.run(cmd + ["status", "BO-203"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert json.loads(status.stdout)["runId"] == "BO-203"

    resume = subprocess.run(
        cmd + ["resume", "BO-203", "--authority-json", str(authority)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert json.loads(resume.stdout)["dispatcherUsed"] is False

    parent_auth = tmp_path / "parent-authority.json"
    parent_auth.write_text(json.dumps({
        **json.loads(authority.read_text()),
        "children": [{"id": "BO-204", "relationType": "hierarchy"}],
    }))
    parent = subprocess.run(
        cmd + ["run", "BO-PARENT", "--mode", "parent", "--authority-json", str(parent_auth), "--root-objective", "Complete parent", "--force"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert parent.returncode != 0  # wrong authority task id must fail closed

    fixed = tmp_path / "parent-authority-fixed.json"
    fixed.write_text(json.dumps({
        **json.loads(authority.read_text()),
        "taskId": "BO-PARENT",
        "publicId": "BO-PARENT",
        "children": [{"id": "BO-204", "relationType": "hierarchy"}],
    }))
    parent = subprocess.run(
        cmd + ["run", "BO-PARENT", "--mode", "parent", "--authority-json", str(fixed), "--root-objective", "Complete parent", "--force"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    parent_data = json.loads(parent.stdout)
    assert parent_data["targetMode"] == "parent"
    assert parent_data["scope"]["childTaskIds"] == ["BO-204"]
