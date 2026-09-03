import json
import subprocess
from pathlib import Path

import pytest

from plugins.builder_adapter import activation
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.preparation import (
    inspect_repository,
    prepare_bundle,
    write_bundle,
)


def _repository(path: Path, remote: str) -> str:
    subprocess.run(["/usr/bin/git", "init", "-q", str(path)], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(path), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["/usr/bin/git", "-C", str(path), "remote", "add", "origin", remote], check=True)
    (path / "README.md").write_text("test\n")
    subprocess.run(["/usr/bin/git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_activate_binds_governance_worktree_and_runtime_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source"
    governance = tmp_path / "governance"
    source.mkdir()
    governance.mkdir()
    remote = "https://example.invalid/project.git"
    source_head = _repository(source, remote)
    governance_head = _repository(governance, "https://example.invalid/governance.git")

    class Snapshot:
        PREFIX = "ai-engineering-orchestrator/"

        def __init__(self, _repository, commit, *, registered_contract_path=None):
            self.commit = commit
            self.registered_contract_path = registered_contract_path
            self.contract = {
                "contract_id": "TEMPLATE",
                "title": "Template",
                "objective": {},
                "acceptance_criteria": [],
                "artifact_bindings": {
                    "allowed_path_manifest": {"path": "old", "sha256": "0" * 64}
                },
            }

        def value(self, artifact_id):
            assert artifact_id == "allowed_path_manifest_schema"
            return {}

    monkeypatch.setattr(activation, "GovernanceSnapshot", Snapshot)
    config = tmp_path / "runtime.json"
    config.write_text(
        json.dumps(
            {
                "socket_path": str(tmp_path / "adapter.sock"),
                "state_path": str(tmp_path / "state.db"),
                "auth_file": str(tmp_path / "auth.json"),
                "governance_repo": str(governance),
                "governance_commit": governance_head,
                "repository_allowlist": {"my-project": remote},
                "validation_profile_id": "strict.v1",
                "board": "test",
                "cycle_registry": {},
            }
        )
    )
    config.chmod(0o600)
    proposal = prepare_bundle(
        cycle_id="FEATURE_EXAMPLE_001",
        contract_id="FEATURE-EXAMPLE-001",
        repository_id="my-project",
        repository=inspect_repository(source),
        goal="Build it",
        acceptance_criteria=["Tests pass"],
        allowed_paths=["README.md"],
        planned_branch="feat/example",
        planned_worktree=str(tmp_path / "worktrees" / "example"),
        validation_profile_id="strict.v1",
        max_runtime_seconds=1800,
        heartbeat_timeout_seconds=180,
        registered_remote=remote,
    )
    proposal_path = tmp_path / "proposal.json"
    write_bundle(proposal_path, proposal)

    result = activation.activate_proposal(config, proposal_path)
    updated = json.loads(config.read_text())
    cycle = updated["cycle_registry"]["FEATURE_EXAMPLE_001"]
    assert result["state"] == "ACTIVATED_RESTART_REQUIRED"
    assert Path(cycle["worktree_path"]).is_dir()
    assert cycle["expected_head_sha"] == source_head
    assert cycle["governance_commit"] != governance_head
    assert updated["governance_commit"] == cycle["governance_commit"]
    assert cycle["proposal_sha256"] == proposal["bundle_sha256"]
    assert subprocess.check_output(
        ["/usr/bin/git", "-C", cycle["worktree_path"], "branch", "--show-current"],
        text=True,
    ).strip() == "feat/example"


def test_activation_rejects_tampered_proposal(tmp_path):
    proposal = {"schema_version": "1.0.0", "bundle_kind": "hermes.builder_job_proposal", "bundle_sha256": "0" * 64}
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal))
    path.chmod(0o600)
    with pytest.raises(AdapterError, match="hash"):
        activation.load_proposal(path)
