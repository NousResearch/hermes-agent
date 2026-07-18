from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package as cron_continuity
from gateway import production_cron_migration
from gateway.production_capability_prerequisites import (
    production_capability_topology_identity_sha256,
)
from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_host_authority as host_authority
from scripts.canary import production_cutover_initial_collector as initial_collector
from scripts.canary import production_cutover_owner_launcher as owner
from scripts.canary import production_database_recovery_gate as database_recovery
from tests.gateway.test_canonical_writer_production_cutover import (
    MemoryJournal,
    NOW,
    Services,
    Snapshots,
    _database_recovery_receipt,
    _db_receipt,
    _freeze,
    _isolated_canary_goal_prerequisite,
    _runtime_attestation,
    _snapshot,
)
from tests.gateway.test_production_cron_continuity_package import (
    EDGE_BOOT_ID_SHA256,
    EDGE_OBSERVED_AT_UNIX,
    _collector_package,
    _collector_execution_readiness,
    _operational_edge_receipt,
    _source_store,
)
from tests.scripts.canary.test_package_production_cutover_artifacts import (
    _release,
    _unit_inputs,
)
from tests.scripts.canary.test_production_cutover_owner_launcher import (
    REVISION,
    _collector_receipt,
)


@pytest.fixture(autouse=True)
def _sealed_owner_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        owner,
        "_active_owner_runtime_attestation",
        lambda revision: _runtime_attestation(revision),
    )
    monkeypatch.setattr(
        owner.cutover_passkey,
        "build_claim_frame",
        lambda *, publication, passkey_proof, now_unix: {
            "schema": owner.cutover_passkey.CUTOVER_CLAIM_FRAME_SCHEMA,
            "publication": publication,
            "passkey_proof": passkey_proof,
            "claim_sha256": "5" * 64,
        },
    )


def _workflow_passkey_proof() -> dict:
    return {
        "proof_sha256": "1" * 64,
        "action_envelope": {
            "request_id": "a" * 64,
            "envelope_sha256": "2" * 64,
            "action_payload_sha256": "6" * 64,
            "authority_release_sha": REVISION,
        },
        "authorization_receipt": {
            "receipt_sha256": "3" * 64,
            "consume_attempt_id": "4" * 64,
            "execution_window_expires_at_unix": NOW + 3600,
        },
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hashed(unsigned: dict, field: str) -> dict:
    return {
        **unsigned,
        field: hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _initial_from_full(full: dict) -> dict:
    unsigned = {
        "schema": owner.INITIAL_COLLECTOR_SCHEMA,
        **{
            name: copy.deepcopy(full[name])
            for name in (
                "release_revision",
                "target",
                "artifacts",
                "gateway_before",
                "writer_before",
                "connector_before",
                "initial_snapshot",
                "cron_inventory",
                "cron_continuity_plan",
                "mechanical_job_host_facts",
                "mechanical_job_package",
                "observed_at_unix",
                "source_boot_id_sha256",
                "secret_material_recorded",
                "secret_digest_recorded",
            )
        },
    }
    return _hashed(unsigned, "receipt_sha256")


def _host_receipt_from_full(full: dict, initial: dict) -> dict:
    request = host_authority.build_host_authority_request(
        initial_collector_receipt=initial,
        release_manifest_sha256="a" * 64,
        gateway_target_identity=full["gateway_target_identity"],
        writer_target_identity=full["writer_target_identity"],
        connector_target_identity=full["connector_target_identity"],
        host_transition=full["host_transition"],
        capability_topology=full["capability_topology"],
        cron_continuity_plan=full["cron_continuity_plan"],
    )
    readback = [
        {
            "name": name,
            "sha256": full["host_transition"]["files"][name]["sha256"],
            "size": 1,
            "staged_uid": 0,
            "staged_gid": 0,
            "staged_mode": 0o400,
            "target_pre": copy.deepcopy(full["host_transition"]["files"][name]["pre"]),
        }
        for name in sorted(package.HOST_ARTIFACT_TARGETS)
    ]
    unsigned = {
        "schema": host_authority.RECEIPT_SCHEMA,
        "release_revision": REVISION,
        "request_sha256": request["request_sha256"],
        "initial_collector_receipt_sha256": initial["receipt_sha256"],
        "release_manifest_sha256": "a" * 64,
        "host_artifact_contract_sha256": "b" * 64,
        "gateway_target_identity": copy.deepcopy(full["gateway_target_identity"]),
        "writer_target_identity": copy.deepcopy(full["writer_target_identity"]),
        "connector_target_identity": copy.deepcopy(full["connector_target_identity"]),
        "host_transition": copy.deepcopy(full["host_transition"]),
        "capability_topology": copy.deepcopy(full["capability_topology"]),
        "cron_continuity_plan": copy.deepcopy(full["cron_continuity_plan"]),
        "readback_file_count": len(package.HOST_ARTIFACT_TARGETS),
        "readback_files": readback,
        "readback_set_sha256": hashlib.sha256(
            _canonical({"files": readback})
        ).hexdigest(),
        "observed_at_unix": NOW,
        "source_boot_id_sha256": initial["source_boot_id_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(unsigned, "receipt_sha256")


def test_release_manifest_binds_every_required_host_artifact(
    tmp_path: Path,
) -> None:
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )

    contract = manifest["host_artifact_contract"]
    assert contract["schema"] == package.HOST_ARTIFACT_CONTRACT_SCHEMA
    assert contract["required_file_count"] == len(
        package.HOST_ARTIFACT_TARGETS
    )
    assert contract["all_files_require_readback"] is True
    assert set(contract["files"]) == set(package.HOST_ARTIFACT_TARGETS)
    assert len({
        item["target_path"] for item in contract["files"].values()
    }) == len(package.HOST_ARTIFACT_TARGETS)
    assert len({
        item["staged_path"] for item in contract["files"].values()
    }) == len(package.HOST_ARTIFACT_TARGETS)
    assert all(
        item["required_readback"] is True
        and item["actual_sha256_bound_by"] == host_authority.RECEIPT_SCHEMA
        for item in contract["files"].values()
    )
    assert all(
        (item["package_sha256"] is not None)
        == (
            package.HOST_ARTIFACT_TARGETS[name][1]
            in {"release_sealed_payload", "release_reviewed_source"}
        )
        for name, item in contract["files"].items()
    )
    assert (
        package.verify_release_artifacts(
            release,
            REVISION,
            unit_inputs=_unit_inputs(),
        )["host_artifact_contract"]
        == contract
    )


def _write(root: Path, logical: str, raw: bytes, mode: int) -> Path:
    path = root.joinpath(*Path(logical).parts[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    # macOS temp roots can inherit wheel (gid 0) even when the test process
    # runs as staff.  Establish the exact staged identity the test passes to
    # the production validator instead of weakening that validator.
    os.chown(path, os.getuid(), os.getgid())
    path.chmod(mode)
    return path


def _retime_service_digest(value: dict, digest: str) -> dict:
    unsigned = {
        **{name: item for name, item in value.items() if name != "observation_sha256"},
        "fragment_sha256": digest,
    }
    return _hashed(unsigned, "observation_sha256")


def test_full_host_collector_reads_back_every_file_and_composes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = Services()
    full = _collector_receipt(NOW, services)
    topology = copy.deepcopy(full["capability_topology"])
    transition = copy.deepcopy(full["host_transition"])
    filesystem = (tmp_path / "host").resolve()
    filesystem.mkdir()

    payloads = {
        name: f"reviewed:{name}\n".encode() for name in package.HOST_ARTIFACT_TARGETS
    }
    digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
    topology["phase_b"]["fragment_sha256"] = digests["phase_b_unit"]
    topology["routeback_edge"]["fragment_sha256"] = digests["routeback_unit"]
    topology["routeback_edge"]["config_sha256"] = digests["routeback_config"]
    topology["mac_ops"]["fragment_sha256"] = digests["mac_ops_unit"]
    topology["mac_ops"]["config_sha256"] = digests["mac_ops_config"]
    topology["browser"]["fragment_sha256"] = digests["browser_unit"]
    topology["browser"]["config_sha256"] = digests["browser_config"]
    topology["isolated_worker"]["socket_fragment_sha256"] = digests[
        "isolated_worker_socket_unit"
    ]
    topology["isolated_worker"]["service_fragment_sha256"] = digests[
        "isolated_worker_service_unit"
    ]
    topology["isolated_worker"]["config_sha256"] = digests["isolated_worker_config"]
    topology["public_connector"]["fragment_sha256"] = digests["connector_unit"]

    gateway_target = copy.deepcopy(full["gateway_target_identity"])
    gateway_target["fragment_sha256"] = digests["gateway_unit"]
    gateway_target["drop_in_sha256"] = {
        cutover.GATEWAY_CONNECTOR_DROP_IN: digests["gateway_connector_drop_in"]
    }
    writer_target = copy.deepcopy(full["writer_target_identity"])
    writer_target["fragment_sha256"] = digests["writer_unit"]
    connector_target = copy.deepcopy(full["connector_target_identity"])
    connector_target["fragment_sha256"] = digests["connector_unit"]

    old_gateway = b"legacy gateway unit\n"
    old_config = b"legacy gateway config\n"
    old_gateway_digest = hashlib.sha256(old_gateway).hexdigest()
    old_config_digest = hashlib.sha256(old_config).hexdigest()
    full["gateway_before"] = _retime_service_digest(
        full["gateway_before"], old_gateway_digest
    )

    for name, item in transition["files"].items():
        item["sha256"] = digests[name]
        item["pre"] = {
            "state": "absent",
            "sha256": None,
            "uid": None,
            "gid": None,
            "mode": None,
        }
        _write(filesystem, item["staged_path"], payloads[name], 0o400)
    gateway_path = _write(
        filesystem,
        transition["files"]["gateway_unit"]["target_path"],
        old_gateway,
        0o644,
    )
    config_path = _write(
        filesystem,
        transition["files"]["gateway_config"]["target_path"],
        old_config,
        0o640,
    )
    transition["files"]["gateway_unit"]["pre"] = {
        "state": "present",
        "sha256": old_gateway_digest,
        "uid": gateway_path.stat().st_uid,
        "gid": gateway_path.stat().st_gid,
        "mode": 0o644,
    }
    transition["files"]["gateway_config"]["pre"] = {
        "state": "present",
        "sha256": old_config_digest,
        "uid": config_path.stat().st_uid,
        "gid": config_path.stat().st_gid,
        "mode": 0o640,
    }
    transition_unsigned = {
        name: item for name, item in transition.items() if name != "manifest_sha256"
    }
    transition = _hashed(transition_unsigned, "manifest_sha256")

    initial = _initial_from_full(full)
    contract_files = {}
    for name, (target, binding_class) in package.HOST_ARTIFACT_TARGETS.items():
        contract_files[name] = {
            "target_path": target,
            "staged_path": transition["files"][name]["staged_path"],
            "binding_class": binding_class,
            "package_sha256": (
                digests[name]
                if binding_class
                in {"release_sealed_payload", "release_reviewed_source"}
                else None
            ),
            "actual_sha256_bound_by": host_authority.RECEIPT_SCHEMA,
            "required_readback": True,
        }
    contract_unsigned = {
        "schema": package.HOST_ARTIFACT_CONTRACT_SCHEMA,
        "files": contract_files,
        "required_file_count": len(package.HOST_ARTIFACT_TARGETS),
        "all_files_require_readback": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    contract = _hashed(contract_unsigned, "contract_sha256")
    manifest = {
        "manifest_sha256": "d" * 64,
        "host_artifact_contract": contract,
        "unit_inputs": {
            "writer_capability_public_key_id": transition[
                "discord_key_foundation"
            ]["writer"]["public_key_id"],
            "operational_edge_key_foundation_sha256": transition[
                "operational_edge_key_foundation_sha256"
            ],
            "operational_edge_receipt_public_key_ids": transition[
                "operational_edge_receipt_public_key_ids"
            ],
            "release_owner_uid": transition["release_owner_uid"],
            "release_owner_gid": transition["release_owner_gid"],
        },
    }
    monkeypatch.setattr(
        package,
        "verify_release_artifacts",
        lambda *_args, **_kwargs: manifest,
    )
    request = host_authority.build_host_authority_request(
        initial_collector_receipt=initial,
        release_manifest_sha256=manifest["manifest_sha256"],
        gateway_target_identity=gateway_target,
        writer_target_identity=writer_target,
        connector_target_identity=connector_target,
        host_transition=transition,
        capability_topology=topology,
        cron_continuity_plan=full["cron_continuity_plan"],
    )
    release = (tmp_path / "release").resolve()
    release.mkdir()
    receipt = host_authority.collect_host_authority(
        request,
        release_root=release,
        filesystem_root=filesystem,
        unit_inputs=_unit_inputs(),
        require_root=False,
        staged_uid=os.getuid(),
        staged_gid=os.getgid(),
        clock=lambda: NOW,
        boot_reader=lambda: initial["source_boot_id_sha256"],
    )

    assert receipt["readback_file_count"] == len(
        package.HOST_ARTIFACT_TARGETS
    )
    assert [item["name"] for item in receipt["readback_files"]] == sorted(
        package.HOST_ARTIFACT_TARGETS
    )
    assert receipt["host_artifact_contract_sha256"] == contract["contract_sha256"]
    composed = host_authority.compose_full_authority_receipt(
        initial_collector_receipt=initial,
        host_authority_request=request,
        host_authority_receipt=receipt,
        release_revision=REVISION,
        now_unix=NOW,
    )
    assert composed["host_transition"] == transition
    assert composed["gateway_target_identity"] == gateway_target
    assert composed["cron_continuity_plan"]["cutover_executable"] is True

    substituted = copy.deepcopy(receipt)
    substituted["request_sha256"] = "f" * 64
    substituted["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value
            for key, value in substituted.items()
            if key != "receipt_sha256"
        })
    ).hexdigest()
    with pytest.raises(
        host_authority.HostAuthorityError,
        match="host_authority_receipt_identity_invalid",
    ):
        host_authority.validate_host_authority_receipt(
            substituted,
            host_authority_request=request,
            initial_collector_receipt=initial,
            release_revision=REVISION,
            now_unix=NOW,
        )

    staged = Path(transition["files"]["api_approval_verifier"]["staged_path"])
    physical = filesystem.joinpath(*staged.parts[1:])
    physical.chmod(0o600)
    physical.write_bytes(b"drifted\n")
    physical.chmod(0o400)
    physical.chmod(0o400)
    with pytest.raises(
        host_authority.HostAuthorityError,
        match="host_authority_staged_digest_invalid",
    ):
        host_authority.collect_host_authority(
            request,
            release_root=release,
            filesystem_root=filesystem,
            unit_inputs=_unit_inputs(),
            require_root=False,
            staged_uid=os.getuid(),
            staged_gid=os.getgid(),
            clock=lambda: NOW,
            boot_reader=lambda: initial["source_boot_id_sha256"],
        )


def _stage_receipt(publication: dict) -> dict:
    if publication["action"] == "freeze-authority":
        outputs = (
            (cutover.STAGED_FREEZE_PLAN_PATH, publication["documents"]["plan"]),
            (
                cutover.STAGED_FREEZE_APPROVAL_PATH,
                publication["documents"]["approval"],
            ),
        )
    else:
        outputs = ((
            cutover.STAGED_CUTOVER_PLAN_PATH,
            publication["documents"]["plan"],
        ),)
    unsigned = {
        "schema": "muncho-production-cutover-publication-receipt.v1",
        "action": publication["action"],
        "release_revision": REVISION,
        "publication_sha256": publication["publication_sha256"],
        "files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(_canonical(document)).hexdigest(),
                "created": True,
            }
            for path, document in outputs
        ],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(unsigned, "receipt_sha256")


def _terminal(plan: cutover.CutoverPlan, approval_sha256: str) -> dict:
    canary_goal = plan.value["freeze_plan"]["cutover_authority"][
        "isolated_canary_goal_prerequisite"
    ]
    equivalence = cutover._build_production_isolation_equivalence(
        plan=plan,
        evidence=canary_goal,
    )
    unsigned = {
        "schema": cutover.TERMINAL_SCHEMA,
        "plan_sha256": plan.sha256,
        "freeze_plan_sha256": plan.value["freeze_plan_sha256"],
        "freeze_approval_sha256": plan.value["freeze_approval_sha256"],
        "approval_sha256": approval_sha256,
        "final_tail_receipt_sha256": plan.value["final_tail_receipt_sha256"],
        "capability_prerequisite_receipt_sha256": "1" * 64,
        "capability_prerequisite_file_sha256": "2" * 64,
        "isolated_canary_goal_continuation_terminal_sha256": canary_goal[
            "goal_continuation_terminal_sha256"
        ],
        "isolated_canary_workspace_gateway_receipt_sha256": canary_goal[
            "workspace_gateway_receipt_sha256"
        ],
        "isolation_equivalence_projection_sha256": equivalence[
            "projection_sha256"
        ],
        "zero_canonical_database_mutation_observed": True,
        "pre_db_zero_write_observation_sha256": "e" * 64,
        "capability_topology_identity_sha256": (
            production_capability_topology_identity_sha256(
                plan.value["capability_topology"]
            )
        ),
        "database_apply_receipt_sha256": "3" * 64,
        "host_apply_receipt_sha256": "4" * 64,
        "host_boot_commit_receipt_sha256": "5" * 64,
        "activation_commit_intent_receipt_sha256": "6" * 64,
        "database_postflight_receipt_sha256": "7" * 64,
        "gateway_observation_sha256": "8" * 64,
        "writer_observation_sha256": "9" * 64,
        "connector_observation_sha256": "a" * 64,
        "direct_discord_disabled": True,
        "discord_dm_allowed": False,
        "rollback_used": False,
        "secret_material_recorded": False,
        "completed_at_unix": NOW,
    }
    return _hashed(unsigned, "receipt_sha256")


class _WorkflowTransport:
    def __init__(
        self,
        initial: dict,
        host: dict,
        services: Services,
        *,
        fail_capture: bool = False,
        clock=lambda: NOW,
        cron_stage: dict | None = None,
    ) -> None:
        self.initial = initial
        self.host = host
        self.services = services
        self.fail_capture = fail_capture
        self.clock = clock
        self.cron_stage = cron_stage
        self.calls: list[str] = []
        self.freeze: cutover.FreezePlan | None = None
        self.approval: dict | None = None
        self.cutover_plan: cutover.CutoverPlan | None = None
        self.journal = MemoryJournal()

    def invoke(
        self,
        _revision: str,
        action: str,
        *,
        publication=None,
        authority_request=None,
    ) -> dict:
        self.calls.append(action)
        if action == "collect-initial":
            assert publication is None and authority_request is None
            return self.initial
        if action == "collect-authority":
            assert authority_request["request_sha256"]
            assert publication is None
            return self.host
        if action == "stage-publication":
            if publication.get("schema") == owner.cutover_passkey.CUTOVER_CLAIM_FRAME_SCHEMA:
                proof = publication["passkey_proof"]
                publication = publication["publication"]
            else:
                proof = None
            if publication["action"] == "freeze-authority":
                assert proof is not None
                self.freeze = cutover.FreezePlan.from_mapping(
                    publication["documents"]["plan"]
                )
                self.approval = cutover.CutoverApproval.from_mapping(
                    publication["documents"]["approval"],
                    plan=self.freeze,
                    now_unix=publication["documents"]["approval"][
                        "issued_at_unix"
                    ],
                ).value
                recorded_at = self.approval["issued_at_unix"]
                self.journal.append(
                    self.freeze.sha256,
                    "passkey_claim",
                    {
                        "schema": cutover.PASSKEY_CLAIM_SCHEMA,
                        "freeze_plan_sha256": self.freeze.sha256,
                        "freeze_approval_sha256": self.approval[
                            "approval_sha256"
                        ],
                        "freeze_publication_sha256": publication[
                            "publication_sha256"
                        ],
                        "passkey_proof_sha256": proof["proof_sha256"],
                        "authorization_receipt_sha256": proof[
                            "authorization_receipt"
                        ]["receipt_sha256"],
                        "action_envelope_sha256": proof[
                            "action_envelope"
                        ]["envelope_sha256"],
                        "action_payload_sha256": proof[
                            "action_envelope"
                        ]["action_payload_sha256"],
                        "request_id": proof["action_envelope"][
                            "request_id"
                        ],
                        "consume_attempt_id": proof[
                            "authorization_receipt"
                        ]["consume_attempt_id"],
                        "authority_release_sha": proof[
                            "action_envelope"
                        ]["authority_release_sha"],
                        "execution_window_expires_at_unix": proof[
                            "authorization_receipt"
                        ]["execution_window_expires_at_unix"],
                    },
                    recorded_at,
                )
                cutover._append_authority(
                    self.journal,
                    self.freeze.sha256,
                    cutover.CutoverApproval.from_mapping(
                        self.approval,
                        plan=self.freeze,
                        now_unix=self.approval["issued_at_unix"],
                    ),
                    recorded_at,
                )
            else:
                self.cutover_plan = cutover.CutoverPlan.from_mapping(
                    publication["documents"]["plan"]
                )
            return _stage_receipt(publication)
        if action == "capture-final-tail":
            if self.fail_capture:
                raise RuntimeError("injected capture failure")
            assert self.freeze is not None and self.approval is not None
            receipt = cutover.execute_final_tail_capture(
                self.freeze,
                self.approval,
                cutover.FreezeDependencies(
                    services=self.services,
                    snapshots=Snapshots(_snapshot(14_081, marker="2")),
                    journal=self.journal,
                    lock=nullcontext,
                ),
                now_unix=int(self.clock()),
            )
            return receipt.to_mapping()
        if action == "collect-stopped":
            assert self.freeze is not None and self.approval is not None
            observed_at = int(self.clock())
            for name in ("gateway", "writer", "connector"):
                current = getattr(self.services, name).to_mapping()
                unsigned = {
                    **{
                        field: item
                        for field, item in current.items()
                        if field != "observation_sha256"
                    },
                    "observed_at_unix": observed_at,
                }
                setattr(
                    self.services,
                    name,
                    cutover.ServiceObservation.from_mapping(
                        _hashed(unsigned, "observation_sha256")
                    ),
                )
            return initial_collector.collect_stopped_services(
                REVISION,
                freeze_plan=self.freeze.to_mapping(),
                freeze_approval=self.approval,
                services=self.services,
                clock=lambda: observed_at,
                boot_reader=lambda: self.initial["source_boot_id_sha256"],
                require_root=False,
            )
        if action == "stage-cron-continuity":
            assert self.freeze is not None
            if self.cron_stage is not None:
                return copy.deepcopy(self.cron_stage)
            continuity = self.freeze.value["cutover_authority"][
                "cron_continuity_plan"
            ]
            unsigned = {
                "schema": owner.CRON_STAGE_NOOP_SCHEMA,
                "release_revision": REVISION,
                "freeze_plan_sha256": self.freeze.sha256,
                "continuity_plan_sha256": continuity[
                    "owner_approved_plan_sha256"
                ],
                "legacy_noop": True,
                "artifacts_staged": False,
                "units_installed": False,
                "timers_enabled": False,
                "timers_started": False,
                "jobs_store_mutated": False,
                "secret_material_recorded": False,
            }
            return _hashed(unsigned, "receipt_sha256")
        if action == "phase-b-preflight":
            assert self.cutover_plan is not None
            return _db_receipt(
                self.cutover_plan,
                "muncho-production-legacy-cutover-preflight.v1",
                "database_postflight",
            )
        if action == "apply-cutover":
            assert self.cutover_plan is not None and self.approval is not None
            return _terminal(
                self.cutover_plan,
                self.approval["approval_sha256"],
            )
        if action == "abort-freeze":
            assert self.freeze is not None and self.approval is not None
            unsigned = {
                "schema": cutover.FREEZE_ABORT_SCHEMA,
                "freeze_plan_sha256": self.freeze.sha256,
                "approval_sha256": self.approval["approval_sha256"],
                "trigger": "owner_abort",
                "gateway_legacy_restarted": True,
                "writer_stopped": True,
                "connector_stopped": True,
                "database_mutated": False,
                "host_mutated": False,
                "secret_material_recorded": False,
                "completed_at_unix": NOW,
            }
            return _hashed(unsigned, "receipt_sha256")
        raise AssertionError(action)


def _workflow_inputs() -> tuple[Services, dict, dict, dict]:
    services = Services()
    full = _collector_receipt(NOW, services)
    initial = _initial_from_full(full)
    host = _host_receipt_from_full(full, initial)
    plan = {
        name: copy.deepcopy(host[name]) for name in owner._HOST_AUTHORITY_PLAN_FIELDS
    }
    return services, initial, host, plan


def _recovery_gate_runner(*_args: object) -> dict:
    return dict(_database_recovery_receipt(rechecked_at_unix=NOW))


def _bridge_request(document: dict) -> dict:
    legacy_request_id = "L" * 32
    unsigned = {
        "schema": owner.BRIDGE_REQUEST_SCHEMA,
        "release_revision": document["release_revision"],
        "freeze_plan_sha256": document["freeze_plan_sha256"],
        "freeze_approval_sha256": document["freeze_approval_sha256"],
        "freeze_publication_sha256": document[
            "freeze_publication_sha256"
        ],
        "v2_request_id": document["v2_request_id"],
        "v2_transaction_id": document["v2_transaction_id"],
        "v2_approval_url_sha256": document["v2_approval_url_sha256"],
        "v2_action_payload_sha256": document[
            "v2_action_payload_sha256"
        ],
        "bootstrap_input_sha256": document["document_sha256"],
        "legacy_passkey_request_id": legacy_request_id,
        "legacy_passkey_request_sha256": "1" * 64,
        "legacy_approval_url": (
            f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/approve/"
            f"{legacy_request_id}"
        ),
        "bridge_action_sha256": "2" * 64,
        "route_contract_sha256": "3" * 64,
        "original_caddy_sha256": "4" * 64,
        "approval_bridge_template_sha256": "5" * 64,
        "approval_bridge_caddy_sha256": "6" * 64,
        "default_local_v1_route_preserved": True,
        "production_mutation_performed": False,
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "requested_at_unix": NOW,
    }
    return _hashed(unsigned, "receipt_sha256")


def _bridge_receipt(document: dict, request: dict) -> dict:
    unsigned = {
        "schema": owner.BRIDGE_RECEIPT_SCHEMA,
        "release_revision": document["release_revision"],
        "freeze_plan_sha256": document["freeze_plan_sha256"],
        "freeze_approval_sha256": document["freeze_approval_sha256"],
        "freeze_publication_sha256": document[
            "freeze_publication_sha256"
        ],
        "v2_request_id": document["v2_request_id"],
        "v2_transaction_id": document["v2_transaction_id"],
        "v2_approval_url_sha256": document["v2_approval_url_sha256"],
        "v2_action_payload_sha256": document[
            "v2_action_payload_sha256"
        ],
        "bootstrap_input_sha256": document["document_sha256"],
        "bridge_request_receipt_sha256": request["receipt_sha256"],
        "legacy_passkey_request_id": request[
            "legacy_passkey_request_id"
        ],
        "legacy_passkey_request_sha256": request[
            "legacy_passkey_request_sha256"
        ],
        "legacy_passkey_grant_id": "G" * 43,
        "legacy_passkey_grant_sha256": "7" * 64,
        "legacy_passkey_consumed_grant_sha256": "8" * 64,
        "legacy_passkey_consume_entry_sha256": "9" * 64,
        "legacy_service_active_before_sha256": "a" * 64,
        "legacy_service_inactive_sha256": "b" * 64,
        "legacy_service_active_after_sha256": "c" * 64,
        "legacy_service_local_health_sha256": "d" * 64,
        "bridge_action_sha256": request["bridge_action_sha256"],
        "route_contract_sha256": request["route_contract_sha256"],
        "original_caddy_sha256": request["original_caddy_sha256"],
        "approval_bridge_caddy_sha256": request[
            "approval_bridge_caddy_sha256"
        ],
        "active_route_projection_sha256": "e" * 64,
        "default_local_v1_route_preserved": True,
        "exact_v2_approval_routes_only": True,
        "caddy_validated": True,
        "caddy_reloaded": True,
        "caddy_readback_verified": True,
        "rollback_mode": "pre_migration_exact_bytes",
        "caller_selected_input_accepted": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "activated_at_unix": NOW,
    }
    return _hashed(unsigned, "receipt_sha256")


def test_prepare_then_resume_consumes_once_before_first_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    prepare_transport = _WorkflowTransport(initial, host, services)

    class PasskeyBoundary:
        def __init__(self) -> None:
            self.request_id = "a" * 64
            self.consume_calls = 0

        def request(self, publication: dict) -> dict:
            plan_sha256 = publication["documents"]["plan"][
                "plan_sha256"
            ]
            return {
                "request_id": self.request_id,
                "action_envelope_sha256": "6" * 64,
                "challenge_record_sha256": "7" * 64,
                "expires_at_unix": NOW + 900,
                "release_sha": REVISION,
                "plan_sha256": plan_sha256,
                "freeze_publication_sha256": publication[
                    "publication_sha256"
                ],
                "action_payload_sha256": "8" * 64,
                "transaction_id": "9" * 64,
                "approval_url": (
                    f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/"
                    f"approve/{self.request_id}"
                ),
                "passkey_only": True,
                "single_use": True,
                "production_mutation_performed": False,
            }

        def consume(
            self,
            *,
            freeze_publication: dict,
            request_id: str,
            consume_attempt_id: str,
        ) -> dict:
            self.consume_calls += 1
            assert request_id == self.request_id
            return {
                "request_id": request_id,
                "consume_attempt_id": consume_attempt_id,
                "disposition": "authorized_once",
                "passkey_proof": _workflow_passkey_proof(),
                "release_sha": REVISION,
                "plan_sha256": freeze_publication["documents"]["plan"][
                    "plan_sha256"
                ],
                "single_use": True,
                "production_mutation_performed": False,
            }

    boundary = PasskeyBoundary()

    class BridgeBootstrap:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.consume_calls = 0
            self.request: dict | None = None

        def prepare(self, document: dict) -> dict:
            self.prepare_calls += 1
            self.request = _bridge_request(document)
            return self.request

        def consume_and_install(self, document: dict) -> dict:
            self.consume_calls += 1
            assert self.request is not None
            return _bridge_receipt(document, self.request)

    bridge = BridgeBootstrap()
    workspace = owner.execute_production_cutover_workflow(
        release_revision=REVISION,
        owner_identity=object(),
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        host_authority_plan=plan,
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        passkey_boundary=boundary,
        prepare_only=True,
        transport_factory=lambda _identity: prepare_transport,
        database_recovery_gate_runner=_recovery_gate_runner,
        now_unix=NOW,
    )

    assert workspace["state"] == "awaiting_bridge_bootstrap"
    assert workspace["advertised_approval_url"] is None
    assert prepare_transport.calls == ["collect-initial", "collect-authority"]
    assert boundary.consume_calls == 0

    resume_transport = _WorkflowTransport(initial, host, services)
    bridge_workspace = owner.resume_prepared_production_cutover_workflow(
        workspace=workspace,
        owner_identity=object(),
        passkey_boundary=boundary,
        bridge_bootstrap=bridge,
        transport_factory=lambda _identity: resume_transport,
        now_unix=NOW,
    )
    assert bridge_workspace["state"] == "awaiting_bridge_passkey"
    assert bridge_workspace["advertised_approval_url"] is None
    assert bridge.prepare_calls == 1
    assert bridge.consume_calls == 0
    assert boundary.consume_calls == 0
    assert resume_transport.calls == []

    passkey_workspace = owner.resume_prepared_production_cutover_workflow(
        workspace=bridge_workspace,
        owner_identity=object(),
        passkey_boundary=boundary,
        bridge_bootstrap=bridge,
        transport_factory=lambda _identity: resume_transport,
        now_unix=NOW,
    )
    assert passkey_workspace["state"] == "awaiting_cutover_passkey"
    assert (
        passkey_workspace["advertised_approval_url"]
        == workspace["passkey_request"]["approval_url"]
    )
    assert passkey_workspace["production_mutation_performed"] is True
    assert bridge.consume_calls == 1
    assert boundary.consume_calls == 0
    assert resume_transport.calls == []

    receipt = owner.resume_prepared_production_cutover_workflow(
        workspace=passkey_workspace,
        owner_identity=object(),
        passkey_boundary=boundary,
        bridge_bootstrap=bridge,
        transport_factory=lambda _identity: resume_transport,
        now_unix=NOW,
    )

    assert receipt["schema"] == owner.WORKFLOW_RECEIPT_SCHEMA
    assert boundary.consume_calls == 1
    assert resume_transport.calls[0] == "stage-publication"


def test_resume_rejects_tampered_approval_url_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, initial, host, plan = _workflow_inputs()

    class Boundary:
        consume_calls = 0

        def request(self, publication: dict) -> dict:
            request_id = "a" * 64
            return {
                "request_id": request_id,
                "action_envelope_sha256": "6" * 64,
                "challenge_record_sha256": "7" * 64,
                "expires_at_unix": NOW + 900,
                "release_sha": REVISION,
                "plan_sha256": publication["documents"]["plan"][
                    "plan_sha256"
                ],
                "freeze_publication_sha256": publication[
                    "publication_sha256"
                ],
                "action_payload_sha256": "8" * 64,
                "transaction_id": "9" * 64,
                "approval_url": (
                    f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/"
                    f"approve/{request_id}"
                ),
                "passkey_only": True,
                "single_use": True,
                "production_mutation_performed": False,
            }

        def consume(self, **_kwargs) -> dict:
            self.consume_calls += 1
            raise AssertionError("tampered workspace reached consume")

    boundary = Boundary()
    workspace = dict(owner.execute_production_cutover_workflow(
        release_revision=REVISION,
        owner_identity=object(),
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        host_authority_plan=plan,
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        passkey_boundary=boundary,
        prepare_only=True,
        transport_factory=lambda _identity: _WorkflowTransport(
            initial, host, services
        ),
        database_recovery_gate_runner=_recovery_gate_runner,
        now_unix=NOW,
    ))
    workspace["passkey_request"] = dict(workspace["passkey_request"])
    workspace["passkey_request"]["approval_url"] = "https://evil.example/"
    workspace["workspace_sha256"] = hashlib.sha256(_canonical({
        name: item
        for name, item in workspace.items()
        if name != "workspace_sha256"
    })).hexdigest()

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_passkey_request_invalid",
    ):
        owner.resume_prepared_production_cutover_workflow(
            workspace=workspace,
            owner_identity=object(),
            passkey_boundary=boundary,
            bridge_bootstrap=object(),
            transport_factory=lambda _identity: None,
            now_unix=NOW,
        )

    assert boundary.consume_calls == 0


@pytest.mark.parametrize(
    "request_id",
    ("A" * 64, "a" * 63, "_" * 64),
)
def test_prepare_rejects_non_sha256_cutover_request_id(
    request_id: str,
) -> None:
    services, initial, host, plan = _workflow_inputs()

    class Boundary:
        def request(self, publication: dict) -> dict:
            return {
                "request_id": request_id,
                "action_envelope_sha256": "6" * 64,
                "challenge_record_sha256": "7" * 64,
                "expires_at_unix": NOW + 900,
                "release_sha": REVISION,
                "plan_sha256": publication["documents"]["plan"][
                    "plan_sha256"
                ],
                "freeze_publication_sha256": publication[
                    "publication_sha256"
                ],
                "action_payload_sha256": "8" * 64,
                "transaction_id": "9" * 64,
                "approval_url": (
                    f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/"
                    f"approve/{request_id}"
                ),
                "passkey_only": True,
                "single_use": True,
                "production_mutation_performed": False,
            }

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_passkey_request_invalid",
    ):
        owner.execute_production_cutover_workflow(
            release_revision=REVISION,
            owner_identity=object(),
            owner_subject_sha256="a" * 64,
            private_key=Ed25519PrivateKey.generate(),
            host_authority_plan=plan,
            isolated_canary_goal_prerequisite=(
                _isolated_canary_goal_prerequisite()
            ),
            truth_mode="start_new_truth_epoch",
            passkey_boundary=Boundary(),
            prepare_only=True,
            transport_factory=lambda _identity: _WorkflowTransport(
                initial, host, services
            ),
            database_recovery_gate_runner=_recovery_gate_runner,
            now_unix=NOW,
        )


@pytest.mark.parametrize("legacy_request_id", ("L" * 31, "L" * 33))
def test_bridge_request_rejects_non_exact_legacy_request_id(
    legacy_request_id: str,
) -> None:
    unsigned = {
        "schema": owner.BRIDGE_BOOTSTRAP_INPUT_SCHEMA,
        "release_revision": REVISION,
        "freeze_plan_sha256": "1" * 64,
        "freeze_approval_sha256": "2" * 64,
        "freeze_publication_sha256": "3" * 64,
        "v2_request_id": "a" * 64,
        "v2_transaction_id": "4" * 64,
        "v2_approval_url_sha256": hashlib.sha256(
            (
                f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/"
                f"approve/{'a' * 64}"
            ).encode("ascii")
        ).hexdigest(),
        "v2_action_payload_sha256": "5" * 64,
    }
    document = _hashed(unsigned, "document_sha256")
    request = _bridge_request(document)
    request["legacy_passkey_request_id"] = legacy_request_id
    request["legacy_approval_url"] = (
        f"{owner.cutover_passkey.protocol.PRODUCTION_ORIGIN}/approve/"
        f"{legacy_request_id}"
    )
    request["receipt_sha256"] = hashlib.sha256(_canonical({
        name: item
        for name, item in request.items()
        if name != "receipt_sha256"
    })).hexdigest()

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_bridge_request_invalid",
    ):
        owner._validate_bridge_request(request, document=document)


def test_workflow_order_is_fixed_and_first_mutation_has_signed_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    transport = _WorkflowTransport(initial, host, services)
    key = Ed25519PrivateKey.generate()

    receipt = owner.execute_production_cutover_workflow(
        release_revision=REVISION,
        owner_identity=object(),
        owner_subject_sha256="a" * 64,
        private_key=key,
        host_authority_plan=plan,
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        passkey_proof=_workflow_passkey_proof(),
        transport_factory=lambda _identity: transport,
        database_recovery_gate_runner=_recovery_gate_runner,
        now_unix=NOW,
    )

    assert transport.calls == [
        "collect-initial",
        "collect-authority",
        "stage-publication",
        "capture-final-tail",
        "collect-stopped",
        "stage-cron-continuity",
        "stage-publication",
        "phase-b-preflight",
        "apply-cutover",
    ]
    assert receipt["schema"] == owner.WORKFLOW_RECEIPT_SCHEMA
    assert [item["stage"] for item in receipt["gates"]] == [
        "initial_read_only_collected",
        "host_authority_read_only_collected",
        "full_authority_composed",
        "database_recovery_validated",
        "freeze_owner_signed",
        "single_use_passkey_consumed",
        "freeze_authority_staged",
        "final_tail_captured",
        "stopped_services_collected",
        "cron_continuity_stage_accepted",
        "cutover_plan_composed",
        "cutover_plan_staged",
        "phase_b_preflight_accepted",
        "cutover_terminal_accepted",
    ]


def test_workflow_samples_fresh_time_at_each_long_running_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    tick = {"value": NOW}

    def advancing_clock() -> int:
        tick["value"] += 45
        return tick["value"]

    transport = _WorkflowTransport(
        initial,
        host,
        services,
        clock=advancing_clock,
    )
    receipt = owner.execute_production_cutover_workflow(
        release_revision=REVISION,
        owner_identity=object(),
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        host_authority_plan=plan,
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        passkey_proof=_workflow_passkey_proof(),
        transport_factory=lambda _identity: transport,
        database_recovery_gate_runner=lambda *_args: (
            _database_recovery_receipt(rechecked_at_unix=tick["value"])
        ),
        clock=advancing_clock,
    )

    assert receipt["schema"] == owner.WORKFLOW_RECEIPT_SCHEMA
    assert tick["value"] >= NOW + 45 * 7


def test_workflow_blocks_before_freeze_authoring_when_recovery_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    transport = _WorkflowTransport(initial, host, services)

    def blocked(*_args: object) -> dict:
        raise database_recovery.ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_transport_unavailable"
        )

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_database_recovery_failed",
    ):
        owner.execute_production_cutover_workflow(
            release_revision=REVISION,
            owner_identity=object(),
            owner_subject_sha256="a" * 64,
            private_key=Ed25519PrivateKey.generate(),
            host_authority_plan=plan,
            isolated_canary_goal_prerequisite=(
                _isolated_canary_goal_prerequisite()
            ),
            truth_mode="start_new_truth_epoch",
            transport_factory=lambda _identity: transport,
            database_recovery_gate_runner=blocked,
            now_unix=NOW,
        )

    assert transport.calls == ["collect-initial", "collect-authority"]


def test_workflow_stages_exact_packaged_cron_before_cutover_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        production_cron_migration,
        "_now",
        lambda: "2027-01-15T08:00:00Z",
    )
    services = Services()
    full = _collector_receipt(NOW, services)
    source_store, _catalog = _source_store(monkeypatch)
    inventory = production_cron_migration.inventory_jobs_bytes(source_store)
    build = cron_continuity.build_packaged_continuity_plan(
        source_store=source_store,
        collector_package=_collector_package(),
        collector_execution_readiness=_collector_execution_readiness(),
        operational_edge_readiness=_operational_edge_receipt(),
        mechanical_job_package_manifest_sha256=full[
            "mechanical_job_package"
        ]["manifest_sha256"],
        cutover_runtime_sha256="d" * 64,
        cutover_entrypoint_sha256="e" * 64,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    )
    full["cron_inventory"] = inventory
    full["cron_continuity_plan"] = build.plan
    full["receipt_sha256"] = hashlib.sha256(
        _canonical({
            name: item
            for name, item in full.items()
            if name != "receipt_sha256"
        })
    ).hexdigest()
    initial = _initial_from_full(full)
    host = _host_receipt_from_full(full, initial)
    plan = {
        name: copy.deepcopy(host[name])
        for name in owner._HOST_AUTHORITY_PLAN_FIELDS
    }
    artifact_index = cron_continuity.write_packaged_continuity_artifacts(
        output_root=tmp_path / "cron-stage",
        build=build,
        inventory=inventory,
    )
    transport = _WorkflowTransport(
        initial,
        host,
        services,
        cron_stage=artifact_index,
    )

    receipt = owner.execute_production_cutover_workflow(
        release_revision=REVISION,
        owner_identity=object(),
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        host_authority_plan=plan,
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        truth_mode="start_new_truth_epoch",
        passkey_proof=_workflow_passkey_proof(),
        transport_factory=lambda _identity: transport,
        database_recovery_gate_runner=_recovery_gate_runner,
        now_unix=NOW,
    )

    assert receipt["schema"] == owner.WORKFLOW_RECEIPT_SCHEMA
    assert transport.calls.index("stage-cron-continuity") < (
        transport.calls.index("stage-publication", 4)
    )
    assert "cron_continuity_stage_accepted" in {
        item["stage"] for item in receipt["gates"]
    }


def test_publication_stage_receipt_rejects_swapped_path_or_digest() -> None:
    services = Services()
    full = _collector_receipt(NOW, services)
    freeze, approval, publication = owner.author_freeze(
        collector_receipt=full,
        release_revision=REVISION,
        owner_subject_sha256="a" * 64,
        private_key=Ed25519PrivateKey.generate(),
        owner_runtime_attestation=_runtime_attestation(),
        isolated_canary_goal_prerequisite=(
            _isolated_canary_goal_prerequisite()
        ),
        database_recovery_receipt=_database_recovery_receipt(
            rechecked_at_unix=NOW
        ),
        truth_mode="start_new_truth_epoch",
        now_unix=NOW,
    )
    assert freeze.sha256 and approval["approval_sha256"]
    receipt = _stage_receipt(dict(publication))
    receipt["files"][0]["path"] = receipt["files"][1]["path"]
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical({
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        })
    ).hexdigest()

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_publication_stage_invalid",
    ):
        owner._validate_publication_stage_receipt(
            receipt,
            publication=publication,
            expected_file_count=2,
        )


def test_packaged_cron_stage_receipt_binds_all_forty_five_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    services = Services()
    private = Ed25519PrivateKey.generate()
    base = _freeze(private, services)
    old_authority = base.value["cutover_authority"]
    monkeypatch.setattr(
        production_cron_migration,
        "_now",
        lambda: "2027-01-15T08:00:00Z",
    )
    source_store, _catalog = _source_store(monkeypatch)
    inventory = production_cron_migration.inventory_jobs_bytes(source_store)
    build = cron_continuity.build_packaged_continuity_plan(
        source_store=source_store,
        collector_package=_collector_package(),
        collector_execution_readiness=_collector_execution_readiness(),
        operational_edge_readiness=_operational_edge_receipt(),
        mechanical_job_package_manifest_sha256=old_authority[
            "mechanical_job_package"
        ]["manifest_sha256"],
        cutover_runtime_sha256="d" * 64,
        cutover_entrypoint_sha256="e" * 64,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    )
    gateway = cutover.ServiceObservation.from_mapping(
        base.value["gateway_before"]
    )
    writer = cutover.ServiceObservation.from_mapping(
        base.value["writer_before"]
    )
    connector = cutover.ServiceObservation.from_mapping(
        base.value["connector_before"]
    )
    authority = cutover.build_cutover_authority(
        release_revision=REVISION,
        artifacts=old_authority["artifacts"],
        gateway_before=gateway,
        writer_before=writer,
        connector_before=connector,
        gateway_target_identity=old_authority["gateway_target_identity"],
        writer_target_identity=old_authority["writer_target_identity"],
        connector_target_identity=old_authority["connector_target_identity"],
        host_transition=old_authority["host_transition"],
        capability_topology=old_authority["capability_topology"],
        cron_inventory=inventory,
        cron_continuity_plan=build.plan,
        mechanical_job_host_facts=old_authority[
            "mechanical_job_host_facts"
        ],
        mechanical_job_package=old_authority["mechanical_job_package"],
        isolated_canary_goal_prerequisite=old_authority[
            "isolated_canary_goal_prerequisite"
        ],
        database_recovery_receipt=old_authority[
            "database_recovery_receipt"
        ],
        legacy_truth_decision=old_authority["legacy_truth_decision"],
        max_appended_rows=old_authority["final_tail_bounds"][
            "max_appended_rows"
        ],
        max_capture_delay_seconds=old_authority["final_tail_bounds"][
            "max_capture_delay_seconds"
        ],
    )
    freeze = cutover.build_freeze_plan(
        release_revision=REVISION,
        target=base.value["target"],
        owner_subject_sha256=base.value["owner_subject_sha256"],
        owner_public_key_ed25519_hex=base.value[
            "owner_public_key_ed25519_hex"
        ],
        gateway_before=gateway,
        writer_before=writer,
        connector_before=connector,
        initial_snapshot=cutover.LegacySnapshot.from_mapping(
            base.value["initial_snapshot"]
        ),
        cutover_authority=authority,
        owner_runtime_attestation=_runtime_attestation(),
    )
    artifact_index = cron_continuity.write_packaged_continuity_artifacts(
        output_root=tmp_path / "cron-stage",
        build=build,
        inventory=inventory,
    )

    accepted = owner._validate_cron_continuity_stage_receipt(
        artifact_index,
        freeze_plan=freeze,
    )
    assert accepted["file_count"] == 45

    changed = copy.deepcopy(artifact_index)
    changed["files"][0], changed["files"][1] = (
        changed["files"][1],
        changed["files"][0],
    )
    changed["artifact_index_sha256"] = hashlib.sha256(
        _canonical({
            name: item
            for name, item in changed.items()
            if name != "artifact_index_sha256"
        })
    ).hexdigest()
    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_cron_continuity_stage_invalid",
    ):
        owner._validate_cron_continuity_stage_receipt(
            changed,
            freeze_plan=freeze,
        )


def test_failure_after_signed_freeze_invokes_abort_before_any_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    transport = _WorkflowTransport(
        initial,
        host,
        services,
        fail_capture=True,
    )

    with pytest.raises(RuntimeError, match="injected capture failure"):
        owner.execute_production_cutover_workflow(
            release_revision=REVISION,
            owner_identity=object(),
            owner_subject_sha256="a" * 64,
            private_key=Ed25519PrivateKey.generate(),
            host_authority_plan=plan,
            isolated_canary_goal_prerequisite=(
                _isolated_canary_goal_prerequisite()
            ),
            truth_mode="start_new_truth_epoch",
            passkey_proof=_workflow_passkey_proof(),
            transport_factory=lambda _identity: transport,
            database_recovery_gate_runner=_recovery_gate_runner,
            now_unix=NOW,
        )

    assert transport.calls == [
        "collect-initial",
        "collect-authority",
        "stage-publication",
        "capture-final-tail",
        "abort-freeze",
    ]
    assert "phase-b-preflight" not in transport.calls
    assert "apply-cutover" not in transport.calls


def test_cutover_and_abort_base_failures_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()

    class OwnerStop(BaseException):
        pass

    primary = OwnerStop("injected owner stop")
    abort_error = RuntimeError("injected abort failure")

    class DualFailureTransport(_WorkflowTransport):
        def invoke(
            self,
            revision: str,
            action: str,
            *,
            publication=None,
            authority_request=None,
        ) -> dict:
            if action == "capture-final-tail":
                self.calls.append(action)
                raise primary
            if action == "abort-freeze":
                self.calls.append(action)
                raise abort_error
            return super().invoke(
                revision,
                action,
                publication=publication,
                authority_request=authority_request,
            )

    transport = DualFailureTransport(initial, host, services)
    with pytest.raises(BaseExceptionGroup) as caught:
        owner.execute_production_cutover_workflow(
            release_revision=REVISION,
            owner_identity=object(),
            owner_subject_sha256="a" * 64,
            private_key=Ed25519PrivateKey.generate(),
            host_authority_plan=plan,
            isolated_canary_goal_prerequisite=(
                _isolated_canary_goal_prerequisite()
            ),
            truth_mode="start_new_truth_epoch",
            passkey_proof=_workflow_passkey_proof(),
            transport_factory=lambda _identity: transport,
            database_recovery_gate_runner=_recovery_gate_runner,
            now_unix=NOW,
        )

    assert caught.value.message == (
        "production cutover failed and freeze abort was incomplete"
    )
    assert caught.value.exceptions == (primary, abort_error)
    assert transport.calls[-2:] == ["capture-final-tail", "abort-freeze"]
    assert "apply-cutover" not in transport.calls


def test_cron_stage_mismatch_aborts_freeze_before_cutover_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.canary.production_cutover_public_stager.time.time",
        lambda: NOW,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_production_cutover.time.time",
        lambda: NOW,
    )
    services, initial, host, plan = _workflow_inputs()
    transport = _WorkflowTransport(
        initial,
        host,
        services,
        cron_stage={},
    )

    with pytest.raises(
        owner.OwnerCutoverError,
        match="owner_cutover_cron_continuity_stage_invalid",
    ):
        owner.execute_production_cutover_workflow(
            release_revision=REVISION,
            owner_identity=object(),
            owner_subject_sha256="a" * 64,
            private_key=Ed25519PrivateKey.generate(),
            host_authority_plan=plan,
            isolated_canary_goal_prerequisite=(
                _isolated_canary_goal_prerequisite()
            ),
            truth_mode="start_new_truth_epoch",
            passkey_proof=_workflow_passkey_proof(),
            transport_factory=lambda _identity: transport,
            database_recovery_gate_runner=_recovery_gate_runner,
            now_unix=NOW,
        )

    assert transport.calls[-2:] == ["stage-cron-continuity", "abort-freeze"]
    assert "phase-b-preflight" not in transport.calls
    assert "apply-cutover" not in transport.calls
