"""Exact Stage 1 preflight/tool-context integration.

Successful validation, CAS commit, and completion evidence remain intentionally
out of scope until the governed disposable container/VM runner exists.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from plugins.builder_adapter import plugin_tools
from plugins.builder_adapter.adapter import BuilderDispatchAdapter
from plugins.builder_adapter.attestation import (
    GovernanceSnapshot,
    HermesProfileResolver,
)
from plugins.builder_adapter.gitops import GitVerifier
from plugins.builder_adapter.native import BUILDER_WORKER_POLICY, NativeKanbanBackend
from plugins.builder_adapter.schemas import SchemaRegistry
from plugins.builder_adapter.store import DispatchStore
from plugins.builder_adapter.validation import ValidationRunner
from tests.plugins.test_builder_adapter_schema import request_payload


GOVERNANCE_REPOSITORY = Path("/opt/bots")
APPROVED_GOVERNANCE_SNAPSHOT = (
    "93ac9675b3edfee34b5769899b7726c542436460"
)
EXPECTED_TOOLS = tuple(sorted(BUILDER_WORKER_POLICY["tool_allowlist"]))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _schema_registry(snapshot: GovernanceSnapshot, root: Path) -> SchemaRegistry:
    schema_artifacts = {
        "dispatch_request": "dispatch_request_schema",
        "dispatch_result": "dispatch_result_schema",
        "completion_evidence": "completion_evidence_schema",
        "allowed_manifest": "allowed_path_manifest_schema",
    }
    root.mkdir(parents=True)
    paths = {}
    for name, artifact_id in schema_artifacts.items():
        path = root / f"{name}.json"
        path.write_bytes(snapshot.raw(artifact_id))
        paths[name] = path
    return SchemaRegistry(paths)


@pytest.mark.integration
def test_exact_stage1_snapshot_preflight_tool_context_and_fail_closed_validation(
    tmp_path, monkeypatch
):
    implementation_repo = Path(__file__).resolve().parents[2]
    snapshot = GovernanceSnapshot(
        GOVERNANCE_REPOSITORY, APPROVED_GOVERNANCE_SNAPSHOT
    )
    policy, _interface = snapshot.load()
    manifest = GitVerifier({}).manifest_from_artifact(
        snapshot.raw("allowed_path_manifest")
    )
    profile = snapshot.value("validation_profile")

    repository = tmp_path / "repository"
    worktree = tmp_path / "worktree"
    subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "-q",
            "--no-hardlinks",
            str(implementation_repo),
            str(repository),
        ],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "-q",
            "-b",
            "stage1-integration",
            str(worktree),
            manifest.base_sha,
        ],
        check=True,
    )
    canonical_remote = str(implementation_repo)
    board = "stage1-exact-integration"
    hermes_home = tmp_path / "hermes"
    profile_dir = hermes_home / "profiles" / "deepseek-builder"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(hermes_home))

    store = DispatchStore(tmp_path / "dispatch.db")
    _write_json(
        profile_dir / "config.yaml",
        {
            "model": {
                "provider": policy["provider"],
                "default": policy["model"],
            },
            "fallback_providers": [],
            "platform_toolsets": {"cli": ["builder_adapter", "no_mcp"]},
            "plugins": {"enabled": ["builder_adapter"]},
            "builder_dispatch": {"confinement": policy["confinement"]},
        },
    )
    _write_json(
        hermes_home / "config.yaml",
        {
            "builder_dispatch": {
                "state_path": str(store.path),
                "repository_allowlist": {"hermes-agent": canonical_remote},
                "governance_repo": str(GOVERNANCE_REPOSITORY),
                "governance_commit": APPROVED_GOVERNANCE_SNAPSHOT,
            }
        },
    )

    profile_resolver = HermesProfileResolver()
    effective = profile_resolver.resolve(policy)
    assert effective.provider == "deepseek"
    assert effective.model == "deepseek-v4-pro"
    assert effective.fallback_chain == ()
    assert effective.allowed_tools == EXPECTED_TOOLS

    backend = NativeKanbanBackend(board=board)
    validation = ValidationRunner(
        {profile["profile_id"]: profile}, python=os.sys.executable
    )
    adapter = BuilderDispatchAdapter(
        store=store,
        schemas=_schema_registry(snapshot, tmp_path / "schemas"),
        git=GitVerifier({"hermes-agent": canonical_remote}),
        kanban=backend,
        validation=validation,
        governance_repo=GOVERNANCE_REPOSITORY,
        governance_attestor=snapshot,
        profile_resolver=profile_resolver,
        cycle_registry={
            "FEAT-HERMES-BUILDER-DISPATCH-001": {
                "revision": 1,
                "contract_id": "FEAT-HERMES-BUILDER-DISPATCH-001",
                "repository_id": "hermes-agent",
                "governance_repository_id": "orchestrator",
                "canonical_remote": canonical_remote,
                "worktree_path": str(worktree),
                "branch": "stage1-integration",
                "expected_head_sha": manifest.base_sha,
                "validation_profile_id": profile["profile_id"],
                "timeout_policy": {
                    "max_runtime_seconds": 60,
                    "heartbeat_timeout_seconds": 30,
                },
                "retry_policy": {
                    "max_attempts": 1,
                    "retryable_terminal_states": [],
                },
            }
        },
    )
    payload = request_payload(tmp_path)
    payload.update(
        {
            "cycle_id": "FEAT-HERMES-BUILDER-DISPATCH-001",
            "contract_id": "FEAT-HERMES-BUILDER-DISPATCH-001",
            "expected_cycle_revision": 1,
        }
    )
    accepted = adapter.dispatch("stage1-integration", payload)
    assert accepted["status"] == "ACCEPTED", json.dumps(accepted, sort_keys=True)
    task_id = accepted["kanban_task_id"]
    record = store.get(payload["dispatch_id"])
    assert record["phase"] == "TASK_CREATED"
    assert store.assert_packet_identity(payload["dispatch_id"])["packet_sha256"]

    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_POLICY", BUILDER_WORKER_POLICY["policy_id"]
    )
    monkeypatch.setenv(
        "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
        json.dumps(BUILDER_WORKER_POLICY["tool_allowlist"]),
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(worktree))

    packet = json.loads(plugin_tools.handle_packet({}))
    assert packet["ok"]
    read = json.loads(
        plugin_tools.handle_read({"path": "hermes_cli/kanban_db.py"})
    )
    assert read["ok"]
    original = read["result"]["content"]
    patched = original + "\n# exact Stage 1 integration probe\n"
    assert json.loads(
        plugin_tools.handle_patch(
            {
                "path": "hermes_cli/kanban_db.py",
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "content": patched,
            }
        )
    )["ok"]
    assert "hermes_cli/kanban_db.py" in json.loads(
        plugin_tools.handle_search({"pattern": "kanban_db.py"})
    )["result"]["paths"]

    write = json.loads(
        plugin_tools.handle_write(
            {
                "path": "plugins/builder_adapter/stage1-probe.txt",
                "content": "governed write\n",
            }
        )
    )
    assert write["ok"]
    assert (worktree / "plugins/builder_adapter/stage1-probe.txt").read_text() == (
        "governed write\n"
    )
    denied = json.loads(
        plugin_tools.handle_write({"path": ".env", "content": "forbidden\n"})
    )
    assert denied["ok"] is False
    assert denied["errors"][0]["code"] == "MANIFEST_MISMATCH"

    validation_result = json.loads(
        plugin_tools.handle_validation(
            {
                "profile_id": profile["profile_id"],
                "expected_sha": manifest.base_sha,
            }
        )
    )
    assert validation_result["ok"] is False
    assert (
        validation_result["errors"][0]["code"]
        == "VALIDATION_CONTAINMENT_UNAVAILABLE"
    )
    assert policy["policy_id"] == "HERMES_DEEPSEEK_BUILDER_V1"
