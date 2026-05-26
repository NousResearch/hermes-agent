from __future__ import annotations

from datetime import datetime, timezone

import pytest

from autonomous_contracts import (
    ContractValidationError,
    compile_ledger_seed,
    compute_contract_sha256,
    create_contract_lock,
    generate_worker_packet,
    schema_map,
    validate_contract,
    write_schema_files,
)


def sample_contract() -> dict:
    return {
        "schemaVersion": "autonomous-contract/v1",
        "contractId": "sample-contract",
        "contractVersion": "0.1.0",
        "contractStatus": "approved",
        "project": {
            "id": "sample-project",
            "name": "Sample Project",
            "publicName": "Sample",
            "repoPath": "/tmp/sample-project",
            "primaryBranch": "main",
            "supervisorProfile": "galt",
            "pmProfile": "sample-pm",
        },
        "chainOfCommand": {
            "owner": "Benjamin",
            "supervisor": "John Galt",
            "projectManager": "sample-pm",
            "escalationPolicy": "galt_first",
            "benjaminEscalationAllowedOnlyBy": "galt",
        },
        "communication": {
            "primary": "discord",
            "profileHomeChannel": "123",
            "supervisorMention": "<@456>",
            "botToBotPolicy": "mentions_required",
            "ackPolicy": "reaction_for_ack_text_for_substance",
            "unavailabilityPolicy": "local_log_retry_5m_then_block_to_galt_when_available",
            "messageClasses": ["status", "blocker", "review_request", "checkpoint", "ack"],
        },
        "credentials": {
            "inheritancePolicy": "explicit_shared_trusted_pm",
            "claudeCode": "inherit_from_galt_allowed",
            "codex": "inherit_from_galt_allowed",
            "github": "profile_scoped_or_supervisor_push",
            "mcp": "curated_by_profile_spec",
            "secretsPolicy": "do_not_print_or_commit",
            "smokeTestsRequired": True,
        },
        "authority": {
            "allowed": ["read_repo", "edit_allowed_paths", "run_tests"],
            "gated": ["install_dependencies", "push_branch", "delete_files"],
            "forbidden": ["direct_benjamin_escalation", "secret_printing", "raw_contract_execution_without_normalization"],
        },
        "modelPolicy": {
            "defaultImplementer": "codex",
            "defaultReviewer": "claude",
            "codexModel": "gpt-5.5",
            "codexReasoningEffort": "high",
            "claudeModel": "sonnet",
            "claudeEffort": "high",
            "overrideAllowedBy": "galt",
            "localModelUse": "scratch_or_low_risk_only",
        },
        "budgets": {
            "maxWorkerCallsPerSprint": 6,
            "maxReviewerCallsPerSprint": 3,
            "maxSprintWallClockMinutes": 180,
            "maxDiffFilesPerSprint": 40,
            "maxDiffLinesPerSprint": 2500,
            "maxRetriesPerWorkerPacket": 1,
            "maxConsecutiveFailedSprints": 2,
        },
        "kanban": {
            "enabled": True,
            "platform": "hermes_kanban",
            "role": "generated_projection",
            "sourceOfTruth": "ledger",
            "conflictPolicy": "ledger_wins_projection_repaired",
        },
        "cleanupPolicy": {
            "closeLedgerItems": True,
            "closeKanbanProjectionItems": True,
            "summarizeDiscordThreadsWhenDone": True,
            "killIdleWorkerSessions": True,
            "archiveOrRemoveTempWorktrees": True,
            "reconcileBackgroundProcesses": True,
            "reconcileOpenPorts": True,
            "reconcileProjectCronJobs": True,
            "reconcileContainers": True,
            "recordCleanupEvidence": True,
        },
        "gates": [
            {
                "id": "G.APPROVED",
                "type": "human_approval",
                "owner": "galt",
                "severity": "blocking",
                "description": "Galt approval before execution.",
                "blocksSprintIds": ["PRE.1"],
                "resolutionCondition": "galt_records_approval_in_ledger",
                "evidenceRequired": ["approval_record"],
                "expiresAfter": None,
            }
        ],
        "sections": [
            {
                "id": "PRE",
                "title": "Preflight",
                "objective": "Prepare execution substrate.",
                "order": 0,
            }
        ],
        "sprints": [
            {
                "id": "PRE.1",
                "section": "PRE",
                "title": "Host tools",
                "order": 1,
                "dependsOn": [],
                "priority": 100,
                "parallelSafe": False,
                "materialType": "environment",
                "allowedPaths": ["docs/**", ".contract-ledger/**"],
                "forbiddenPaths": ["~/.hermes/auth.json", "~/.claude/**", "~/.codex/**"],
                "objective": "Verify host tools and write tooling report.",
                "requiredInputs": ["global.agentRules"],
                "requiredContext": ["Use scoped paths only."],
                "implementationRequirements": ["Record versions."],
                "acceptance": [
                    {
                        "id": "PRE.1.AC1",
                        "text": "Tool command passes.",
                        "verification": "command",
                        "command": "python3 --version",
                    },
                    {
                        "id": "PRE.1.AC2",
                        "text": "Report exists.",
                        "verification": "file_exists",
                        "path": "docs/TOOLING_VERSIONS.md",
                    },
                ],
                "gates": ["G.APPROVED"],
                "stopConditions": [
                    {"id": "PRE.1.STOP1", "text": "Tool missing.", "escalation": "galt"}
                ],
                "evidenceRequired": ["command_log", "file_snapshot", "handoff"],
                "review": {"required": False},
                "closeout": {"requiresCleanupAudit": True, "requiresGaltGate": False},
            }
        ],
    }


def test_validate_contract_accepts_minimum_v1_shape() -> None:
    contract = validate_contract(sample_contract())
    assert contract.contractId == "sample-contract"
    assert contract.project.pmProfile == "sample-pm"


def test_validate_contract_rejects_duplicate_sprint_ids() -> None:
    data = sample_contract()
    data["sprints"].append(dict(data["sprints"][0]))
    with pytest.raises(ContractValidationError, match="duplicate sprint ids"):
        validate_contract(data)


def test_validate_contract_rejects_missing_gate_reference() -> None:
    data = sample_contract()
    data["sprints"][0]["gates"] = ["G.MISSING"]
    with pytest.raises(ContractValidationError, match="missing gate"):
        validate_contract(data)


def test_validate_contract_rejects_kanban_as_authority() -> None:
    data = sample_contract()
    data["kanban"]["sourceOfTruth"] = "kanban"
    with pytest.raises(ContractValidationError, match="sourceOfTruth"):
        validate_contract(data)


def test_compile_ledger_seed_is_deterministic_and_scoped() -> None:
    seed = compile_ledger_seed(sample_contract())
    assert seed.contractId == "sample-contract"
    assert [record.sprintId for record in seed.sprints] == ["PRE.1"]
    assert seed.contractSha256 == compute_contract_sha256(validate_contract(sample_contract()))
    assert seed.sprints[0].objective == "Verify host tools and write tooling report."
    assert seed.sprints[0].acceptance[1].verification == "file_exists"
    assert seed.sprints[0].allowedPaths == ["docs/**", ".contract-ledger/**"]
    assert seed.gates[0].gateId == "G.APPROVED"
    assert seed.gates[0].resolved is False
    assert seed.cleanupRegistry.records == []


def test_contract_hash_is_canonical_for_dict_order() -> None:
    data = sample_contract()
    contract = validate_contract(data)
    assert compute_contract_sha256(contract) == compute_contract_sha256(contract.model_dump(mode="json"))


def test_create_contract_lock_binds_hash_and_approval() -> None:
    approved_at = datetime(2026, 5, 23, tzinfo=timezone.utc)
    contract = validate_contract(sample_contract())
    lock = create_contract_lock(contract, approved_by="galt", approved_at=approved_at)
    assert lock.contractId == contract.contractId
    assert lock.contractSha256 == compute_contract_sha256(contract)
    assert lock.approvedBy == "galt"
    assert lock.active is True


def test_generate_worker_packet_excludes_master_contract() -> None:
    packet = generate_worker_packet(
        sample_contract(),
        "PRE.1",
        worker_role="implementer",
        assigned_worker="codex",
        session_id="session-1",
        context={"contractExcerptPath": ".contract-ledger/packets/PRE.1.excerpt.md"},
    )
    dumped = packet.model_dump()
    assert packet.projectId == "sample-project"
    assert packet.verificationCommands == ["python3 --version"]
    assert [criterion.id for criterion in packet.acceptanceCriteria] == ["PRE.1.AC1", "PRE.1.AC2"]
    assert packet.acceptanceCriteria[1].path == "docs/TOOLING_VERSIONS.md"
    assert "sprints" not in dumped
    assert packet.context["contractExcerptPath"].endswith("PRE.1.excerpt.md")


def test_validate_contract_rejects_indirect_dependency_cycles() -> None:
    data = sample_contract()
    second = dict(data["sprints"][0])
    second["id"] = "PRE.2"
    second["title"] = "Second"
    second["order"] = 2
    second["dependsOn"] = ["PRE.1"]
    data["sprints"][0]["dependsOn"] = ["PRE.2"]
    data["sprints"].append(second)
    with pytest.raises(ContractValidationError, match="dependency cycle"):
        validate_contract(data)


def test_compile_ledger_seed_can_embed_contract_lock() -> None:
    seed = compile_ledger_seed(sample_contract(), approved_by="galt")
    assert seed.contractLock is not None
    assert seed.contractLock.contractSha256 == seed.contractSha256
    assert seed.contractLock.approvedBy == "galt"


def test_schema_export_writes_phase_one_schema_files(tmp_path) -> None:
    schemas = schema_map()
    assert "contract.schema.json" in schemas
    assert "worker-packet.schema.json" in schemas
    written = write_schema_files(tmp_path)
    names = {path.name for path in written}
    assert {"contract.schema.json", "ledger-seed.schema.json", "worker-packet.schema.json"} <= names
    assert (tmp_path / "contract.schema.json").read_text(encoding="utf-8").endswith("\n")
