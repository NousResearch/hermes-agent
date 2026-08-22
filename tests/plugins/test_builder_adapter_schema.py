from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from plugins.builder_adapter.adapter import BuilderDispatchAdapter
from plugins.builder_adapter.canonical import canonical_json_bytes, canonical_sha256
from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.gitops import AllowedPathManifest
from plugins.builder_adapter.native import CancellationProof, TaskSnapshot
from plugins.builder_adapter.schemas import SchemaRegistry
from plugins.builder_adapter.store import DispatchStore


POLICY = {
    "schema_version": "1.0.0",
    "policy_id": "HERMES_DEEPSEEK_BUILDER_V1",
    "profile": "deepseek-builder",
    "provider": "deepseek",
    "model": "deepseek-v4-pro",
    "fallback_chain": [],
    "allowed_tools": ["builder_read_file"],
    "forbidden_tools": ["terminal"],
    "authority": {"allowed": ["implement"], "forbidden": ["review"]},
    "secrets": {"may_read_provider_secret": False, "may_emit_secrets": False},
    "live_execution_affected": False,
}


def policy_hash(policy=POLICY):
    return hashlib.sha256(
        (json.dumps(policy, ensure_ascii=False, indent=2) + "\n").encode()
    ).hexdigest()


def request_payload(tmp_path: Path) -> dict:
    return {
        "schema_version": "1.0.0",
        "dispatch_id": str(uuid4()),
        "idempotency_key": "dispatch-key-" + "a" * 32,
        "cycle_id": "FEAT_TEST_001",
        "contract_id": "FEAT_TEST_001",
        "repository_id": "hermes-agent",
        "builder_role": "primary_builder",
        "expected_cycle_revision": 7,
        "completion_schema_version": "1.0.0",
    }


class AcceptingSchemas:
    def validate(self, name, value):
        if name == "dispatch_request" and "unexpected" in value:
            raise AdapterError("INVALID_REQUEST", "unexpected property")


class FakeGit:
    manifest = AllowedPathManifest(
        {
            "default_access": "forbidden",
            "base_sha": "3" * 40,
            "read_policy": {
                "source": "git_tracked_regular_files",
                "snapshot": "base_sha",
                "deny_patterns": [".env", ".git/**"],
            },
            "symlinks": "reject",
            "submodules": "reject",
            "rules": [{"pattern": "plugins/builder_adapter/**", "access": "read_write"}],
        }
    )

    def verify_artifact(self, repo, ref, code):
        if code == "CONTRACT_MISMATCH":
            return json.dumps({"contract_id": "FEAT_TEST_001"}).encode()
        return json.dumps(
            {
                "default_access": "forbidden",
                "base_sha": "3" * 40,
                "read_policy": {
                    "source": "git_tracked_regular_files",
                    "snapshot": "base_sha",
                    "deny_patterns": [".env", ".git/**"],
                },
                "symlinks": "reject",
                "submodules": "reject",
                "rules": [
                    {
                        "pattern": "plugins/builder_adapter/**",
                        "access": "read_write",
                    }
                ],
            }
        ).encode()

    def verify_worktree(self, request):
        return Path(request.worktree_path)

    def manifest_from_artifact(self, raw):
        return self.manifest


class FakeKanban:
    def __init__(self):
        self.created = 0
        self.cancelled = True
        self.status = "ready"
        self.released = 0

    def create_task(self, request_sha256, request):
        self.created += 1
        return "t_12345678"

    def snapshot(self, task_id):
        return TaskSnapshot(task_id, self.status, ["1"], 1)

    def cancel(self, task_id, reason):
        return CancellationProof(
            confirmed=self.cancelled,
            process_tree_terminated=self.cancelled,
            task_archived=self.cancelled,
            detail="test proof",
        )

    def completion_exclusive(self, task_id):
        return self.status == "done"

    def release_completion_lease(self, task_id):
        self.released += 1
        return True


class FakeValidation:
    _profiles = {"hermes-builder-adapter-strict.v1": {}}


class FakeGovernance:
    def __init__(self, policy):
        self.policy = policy
        self.commit = "1" * 40
        self.contract_sha256 = "2" * 64
        self.REGISTERED_CONTRACT_PATH = "contracts/active/FEAT_TEST_001.json"
        self.contract_raw = json.dumps({"contract_id": "FEAT_TEST_001"}).encode()
        self.contract = {
            "contract_id": "FEAT_TEST_001",
            "objective": {},
            "artifact_bindings": {
                "allowed_path_manifest": {
                    "path": "contracts/manifests/test.json",
                    "sha256": "4" * 64,
                }
            },
        }
        self.bindings = {
            "allowed_path_manifest": (
                "contracts/manifests/test.json",
                b"",
                {},
            )
        }

    def load(self):
        return self.policy, {"capability_id": "hermes.builder_dispatch.v1"}

    def raw(self, artifact_id):
        return FakeGit().verify_artifact(None, None, "MANIFEST_MISMATCH")

    def value(self, artifact_id):
        assert artifact_id == "validation_profile"
        return {"profile_id": "hermes-builder-adapter-strict.v1"}


class FakeEffectiveProfile:
    def __init__(self, policy):
        self.policy = policy

    def evidence(self):
        return {
            "provider": self.policy["provider"],
            "model": self.policy["model"],
            "profile": self.policy["profile"],
            "profile_configuration_sha256": "9" * 64,
            "fallback_chain": list(self.policy["fallback_chain"]),
            "fallback_used": False,
            "attested_by": "hermes.builder_dispatch.v1",
        }


class FakeProfileResolver:
    def resolve(self, policy):
        if (
            policy.get("provider") != "deepseek"
            or policy.get("model") != "deepseek-v4-pro"
            or policy.get("fallback_chain") != []
        ):
            raise AdapterError("PROFILE_POLICY_MISMATCH", "effective builder route")
        return FakeEffectiveProfile(policy)


def make_adapter(tmp_path, *, kanban=None, policy=POLICY):
    return BuilderDispatchAdapter(
        store=DispatchStore(tmp_path / "journal.db"),
        schemas=AcceptingSchemas(),
        git=FakeGit(),
        kanban=kanban or FakeKanban(),
        validation=FakeValidation(),
        governance_repo=tmp_path,
        governance_attestor=FakeGovernance(policy),
        profile_resolver=FakeProfileResolver(),
        cycle_registry={
            "FEAT_TEST_001": {
                "revision": 7,
                "contract_id": "FEAT_TEST_001",
                "repository_id": "hermes-agent",
                "governance_repository_id": "orchestrator",
                "canonical_remote": "git@example.invalid:hermes.git",
                "worktree_path": str(tmp_path),
                "branch": "feat/test",
                "expected_head_sha": "3" * 40,
                "validation_profile_id": "hermes-builder-adapter-strict.v1",
                "timeout_policy": {
                    "max_runtime_seconds": 60,
                    "heartbeat_timeout_seconds": 30,
                },
                "retry_policy": {
                    "max_attempts": 2,
                    "retryable_terminal_states": ["CRASHED"],
                },
            }
        },
    )


def test_canonical_request_hash_is_order_independent():
    left = {"z": [2, 1], "a": {"b": True}}
    right = {"a": {"b": True}, "z": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_cycle_specific_governance_snapshot_is_exactly_bound(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    state = adapter.cycle_registry["FEAT_TEST_001"]
    state["governance_commit"] = "a" * 40
    state["contract_path"] = "contracts/active/FEATURE-ONE.json"
    seen = {}

    class Snapshot:
        def __init__(self, repository, commit, *, registered_contract_path):
            seen.update(
                repository=repository,
                commit=commit,
                registered_contract_path=registered_contract_path,
            )

    monkeypatch.setattr("plugins.builder_adapter.attestation.GovernanceSnapshot", Snapshot)
    result = adapter._snapshot_for_cycle(state)
    assert isinstance(result, Snapshot)
    assert seen == {
        "repository": tmp_path,
        "commit": "a" * 40,
        "registered_contract_path": "contracts/active/FEATURE-ONE.json",
    }


def test_registered_schema_fully_rejects_nested_and_extra_fields(tmp_path):
    schema_path = tmp_path / "request.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": ["nested"],
                "properties": {
                    "nested": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value"],
                        "properties": {"value": {"const": "expected"}},
                    }
                },
            }
        )
    )
    registry = SchemaRegistry({"request": schema_path})
    registry.validate("request", {"nested": {"value": "expected"}})
    with pytest.raises(AdapterError, match="schema validation failed"):
        registry.validate("request", {"nested": {"value": "wrong", "extra": True}})


def test_adapter_rejects_unknown_request_before_kanban(tmp_path):
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban)
    payload = request_payload(tmp_path)
    payload["unexpected"] = True
    result = adapter.dispatch("orchestrator-mcp", payload)
    assert result["status"] == "REJECTED"
    assert result["side_effects_state"] == "NONE"
    assert kanban.created == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("governance_commit", "0" * 40),
        ("worktree_path", "/tmp/attacker"),
        ("validation_profile", "historical.v1"),
        ("timeout_policy", {"max_runtime_seconds": 999}),
        ("allowed_path_manifest", {"path": "historical.json"}),
    ],
)
def test_caller_cannot_select_any_authoritative_coordinate(
    tmp_path, field, value
):
    adapter = make_adapter(tmp_path)
    payload = request_payload(tmp_path)
    payload[field] = value
    result = adapter.dispatch("principal", payload)
    assert result["status"] == "REJECTED"
    assert adapter.store.get(payload["dispatch_id"]) is None


def test_stale_or_unregistered_cycle_is_rejected_before_reservation(tmp_path):
    adapter = make_adapter(tmp_path)
    payload = request_payload(tmp_path)
    payload["expected_cycle_revision"] = 6
    result = adapter.dispatch("principal", payload)
    assert result["status"] == "REJECTED"
    assert adapter.store.get(payload["dispatch_id"]) is None


def test_wrong_effective_policy_fails_before_kanban_side_effect(tmp_path):
    bad = deepcopy(POLICY)
    bad["provider"] = "openrouter"
    kanban = FakeKanban()
    adapter = make_adapter(tmp_path, kanban=kanban, policy=bad)
    result = adapter.dispatch("orchestrator-mcp", request_payload(tmp_path))
    assert result["status"] == "REJECTED"
    assert result["errors"][0]["code"] == "PROFILE_POLICY_MISMATCH"
    assert kanban.created == 0
