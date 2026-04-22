import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


WORKTREE_SCRIPT_ROOT = Path(os.environ.get("HERMES_SCRIPT_ROOT", str(Path.home() / ".hermes" / "scripts")))
WORKTREE_SCRIPT_PATH = WORKTREE_SCRIPT_ROOT / "minos_worktree_run.py"
worktree_spec = importlib.util.spec_from_file_location("minos_worktree_run", WORKTREE_SCRIPT_PATH)
minos_worktree_run = importlib.util.module_from_spec(worktree_spec)
worktree_spec.loader.exec_module(minos_worktree_run)

GATE_SCRIPT_PATH = WORKTREE_SCRIPT_ROOT / "minos_gate.py"
gate_spec = importlib.util.spec_from_file_location("minos_gate", GATE_SCRIPT_PATH)
minos_gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(minos_gate)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _task_pack(repo: Path, verifier_commands: list[str]) -> Path:
    task_pack = repo / "task-pack.md"
    body = [
        "# Minos task pack",
        "",
        "## Verifier commands",
        "Replace these examples with task-specific commands before execution. Run them before stopping and record the result:",
    ]
    body.extend([f"- {cmd}" for cmd in verifier_commands])
    body.extend(["", "## Stop conditions", "- done"])
    task_pack.write_text("\n".join(body) + "\n", encoding="utf-8")
    return task_pack


def _bootstrap(repo: Path, verifier_commands: list[str], run_id: str = "run-001") -> dict[str, str]:
    return minos_worktree_run.bootstrap_run(
        repo_path=repo,
        task_pack_path=_task_pack(repo, verifier_commands),
        run_id=run_id,
    )


def _seed_builder_artifacts(bootstrap: dict[str, str], exit_code: int = 0) -> Path:
    artifact_dir = Path(bootstrap["artifact_dir"])
    (artifact_dir / "builder-summary.md").write_text("summary\n", encoding="utf-8")
    (artifact_dir / "builder-result.json").write_text(
        json.dumps({"exit_code": exit_code}) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "git-status.txt").write_text(" M README.md\n", encoding="utf-8")
    (artifact_dir / "git-diff.patch").write_text(
        "diff --git a/README.md b/README.md\nindex ce01362..95d8f88 100644\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-hello\n+patched from builder\n",
        encoding="utf-8",
    )
    return artifact_dir


def test_gate_creates_fresh_validation_lane(clean_repo: Path):
    bootstrap = _bootstrap(clean_repo, ["python3 -c \"print('ok')\""])
    _seed_builder_artifacts(bootstrap)
    with pytest.raises(RuntimeError, match="Gate worktree path already exists"):
        with pytest.MonkeyPatch.context() as mp:
            def _no_cleanup(repo_path, worktree_path, branch_name):
                return None
            mp.setattr(minos_gate, "_cleanup_gate_worktree", _no_cleanup)
            minos_gate.run_gate(bootstrap)
            minos_gate.run_gate(bootstrap)


def test_gate_replays_verifier_commands_from_task_pack(clean_repo: Path):
    bootstrap = _bootstrap(clean_repo, ["python3 -c \"from pathlib import Path; print(Path('README.md').read_text().strip())\""])
    _seed_builder_artifacts(bootstrap)
    result = minos_gate.run_gate(bootstrap)

    summary = Path(result["gate_summary_path"]).read_text(encoding="utf-8")
    command_log = Path(result["gate_log_path"]).read_text(encoding="utf-8")
    assert "python3 -c \"from pathlib import Path; print(Path('README.md').read_text().strip())\"" in summary
    assert "patched from builder" in command_log
    assert result["gate_passed"] is True


def test_gate_checks_required_builder_artifacts(clean_repo: Path):
    bootstrap = _bootstrap(clean_repo, ["python3 -c \"print('ok')\""])
    artifact_dir = Path(bootstrap["artifact_dir"])
    (artifact_dir / "builder-summary.md").write_text("summary\n", encoding="utf-8")
    # intentionally omit builder-result.json and other required artifacts

    result = minos_gate.run_gate(bootstrap)

    persisted = json.loads(Path(result["gate_result_path"]).read_text(encoding="utf-8"))
    assert result["gate_passed"] is False
    assert persisted["gate_passed"] is False
    missing = "\n".join(persisted["missing_artifacts"])
    assert "builder-result.json" in missing
    assert "git-diff.patch" in missing


def test_gate_writes_final_pass_fail_bundle(clean_repo: Path):
    bootstrap = _bootstrap(clean_repo, ["python3 -c \"import sys; sys.exit(2)\""])
    _seed_builder_artifacts(bootstrap, exit_code=0)

    result = minos_gate.run_gate(bootstrap)

    result_path = Path(result["gate_result_path"])
    summary_path = Path(result["gate_summary_path"])
    decision_path = Path(result["decision_path"])
    assert result_path.exists()
    assert summary_path.exists()
    assert decision_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert payload["gate_passed"] is False
    assert payload["command_results"][0]["exit_code"] == 2
    assert payload["builder_exit_code"] == 0
    assert payload["status_mismatch"] is None
    assert decision["run_id"] == "run-001"
    assert decision["builder_status"] == "passed"
    assert decision["gate_status"] == "failed"
    assert decision["next_action"] == "retry"
    assert decision["decision_reason"] == "verifier_failed"
    assert decision["publish_recommended"] is False
    assert decision["retry_recommended"] is True
    assert decision["human_review_required"] is False
    assert "FAILED" in summary_path.read_text(encoding="utf-8")



def test_gate_fails_when_artifact_status_mismatches_patch(clean_repo: Path):
    bootstrap = _bootstrap(clean_repo, ["python3 -c \"print('ok')\""])
    artifact_dir = _seed_builder_artifacts(bootstrap, exit_code=0)
    (artifact_dir / "git-status.txt").write_text(" M SOME_OTHER_FILE\n", encoding="utf-8")

    result = minos_gate.run_gate(bootstrap)

    payload = json.loads(Path(result["gate_result_path"]).read_text(encoding="utf-8"))
    decision = json.loads(Path(result["decision_path"]).read_text(encoding="utf-8"))
    assert payload["gate_passed"] is False
    assert payload["status_mismatch"] is not None
    assert "README.md" in payload["status_mismatch"]["actual"]
    assert decision["next_action"] == "escalate"
    assert decision["decision_reason"] == "status_mismatch"
    assert decision["blocking_findings"]
    assert decision["human_review_required"] is True
