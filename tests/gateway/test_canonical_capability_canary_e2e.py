from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_capability_canary_e2e as canary
from gateway import canonical_capability_canary_producers as producers


NOW = 1_800_000_000_000
PRODUCER_READINESS_SHA256 = hashlib.sha256(
    b"producer-readiness-final"
).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _full_canary_terminal_receipt(
    revision: str = "1" * 40,
) -> dict[str, object]:
    unsigned = {
        "schema": "muncho-full-canary-session-bound-owner-receipt.v1",
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": revision,
        "coordinator_input_sha256": _sha("full-coordinator-input"),
        "full_canary_plan_sha256": "b" * 64,
        "owner_approval_sha256": _sha("original-full-owner-approval"),
        "phase_b_readiness_anchor_sha256": _sha("full-phase-b-anchor"),
        "api_session_key_sha256": _sha("full-api-session-key"),
        "fixture_sha256": _sha("full-fixture"),
        "discord_token_install_receipt_sha256": _sha(
            "full-discord-token-install"
        ),
        "coordinator_receipt_sha256": _sha("full-coordinator-receipt"),
        "live_driver_receipt_sha256": _sha("full-live-driver-receipt"),
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": NOW // 1_000 - 60,
    }
    return {**unsigned, "receipt_sha256": canary._digest(unsigned)}


def _private_keys() -> dict[str, Ed25519PrivateKey]:
    return {role: Ed25519PrivateKey.generate() for role in canary.AUTHORITY_ROLES}


def _fixture(
    keys: dict[str, Ed25519PrivateKey],
    revision: str = "1" * 40,
) -> dict:
    terminal = _full_canary_terminal_receipt(revision)
    semantic_config_contract = canary.build_semantic_config_contract()
    public_discord_target = {
        "target_type": "public_channel",
        "guild_id": canary.PRODUCTION_DISCORD_GUILD_ID,
        "channel_id": canary.PRODUCTION_DISCORD_CANARY_CHANNEL_ID,
    }
    discord_bot_identities = {
        "production_bot_user_id": canary.PRODUCTION_DISCORD_BOT_USER_ID,
        "connector_bot_user_id": (
            canary.PRODUCTION_DISCORD_CONNECTOR_BOT_USER_ID
        ),
        "routeback_bot_user_id": (
            canary.PRODUCTION_DISCORD_ROUTEBACK_BOT_USER_ID
        ),
    }
    role_topology = canary.build_capability_role_topology_contract(
        public_discord_target=public_discord_target,
        discord_bot_identities=discord_bot_identities,
    )
    return {
        "schema": canary.FIXTURE_SCHEMA,
        "release_sha": revision,
        "release_root": f"/opt/muncho-canary-releases/{revision}",
        "release_artifact_sha256": _sha("release"),
        "capability_plan_sha256": "a" * 64,
        "full_canary_plan_sha256": "b" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal[
            "owner_approval_sha256"
        ],
        "plan_publication_receipt_sha256": _sha(
            "capability-plan-publication"
        ),
        "installed_wheel_manifest_sha256": _sha("wheel"),
        "effective_config_sha256": _sha("config"),
        "tool_inventory_sha256": _sha("tools"),
        "run_id": "11111111-1111-4111-8111-111111111111",
        "owner_id": "1279454038731264061",
        "host_identity_sha256": _sha("host"),
        "business_edge_service_identity_sha256": _sha("business-edge-service"),
        "bitrix_operational_edge_contract": {
            "revision": revision,
            "service_unit": canary.BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT,
            "service_identity_sha256": _sha("business-edge-service"),
            "asset_manifest_sha256": _sha("bitrix-asset-manifest"),
            "asset_names": list(canary.BITRIX_OPERATIONAL_EDGE_ASSET_NAMES),
            "asset_manifest_path": (
                f"/opt/muncho-canary-releases/{revision}/ops/muncho/runtime/"
                "operational-assets/manifest.json"
            ),
            "rendered_unit_sha256": _sha("bitrix-rendered-unit"),
            "rendered_unit_path": canary.BITRIX_OPERATIONAL_EDGE_UNIT_PATH,
            "rendered_config_sha256": _sha("bitrix-rendered-config"),
            "rendered_config_path": canary.BITRIX_OPERATIONAL_EDGE_CONFIG_PATH,
            "rendered_trust_sha256": _sha("bitrix-rendered-trust"),
            "rendered_trust_path": canary.BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
            "identity_bootstrap": {
                "service_user": canary.BITRIX_OPERATIONAL_EDGE_SERVICE_USER,
                "service_group": canary.BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP,
                "service_uid": 2101,
                "service_gid": 2101,
                "socket_client_group": (
                    canary.BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP
                ),
                "socket_client_gid": 2102,
                "receipt_sha256": _sha("bitrix-identity-bootstrap"),
            },
            "credential_projection": {
                "name": "bitrix-webhook-url",
                "source_path": canary.BITRIX_WEBHOOK_SOURCE_PATH,
                "projected_path": canary.BITRIX_WEBHOOK_PROJECTION_PATH,
                "bind_target_path": canary.BITRIX_WEBHOOK_SOURCE_PATH,
                "source_owner_uid": 0,
                "source_owner_gid": 0,
                "source_mode": "0400",
                "service_reads_projection": True,
                "original_source_inaccessible": True,
                "value_or_digest_recorded": False,
            },
            "receipt_key_contract": {
                "private_credential_name": "receipt-private-key",
                "private_source_path": canary.BITRIX_RECEIPT_PRIVATE_KEY_PATH,
                "private_projection_path": (
                    canary.BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
                ),
                "private_owner_uid": 0,
                "private_owner_gid": 0,
                "private_mode": "0400",
                "public_path": canary.BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
                "public_key_id": _sha("bitrix-receipt-public-key"),
                "public_trust_sha256": _sha("bitrix-rendered-trust"),
                "writer_public_key_credential_name": "writer-public-key",
                "writer_public_key_source_path": canary.WRITER_PUBLIC_KEY_PATH,
                "writer_public_key_projection_path": (
                    canary.WRITER_PUBLIC_KEY_PROJECTION_PATH
                ),
                "key_bootstrap_receipt_sha256": _sha(
                    "bitrix-key-bootstrap"
                ),
                "create_only": True,
                "retire_private_on_stop": True,
                "retire_public_on_stop": True,
                "private_content_or_digest_recorded": False,
            },
            "expected_active_service_state": {
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
                "unit_file_state": "disabled",
            },
            "expected_cleanup_service_state": {
                "active_state": "inactive",
                "sub_state": "dead",
                "overlay_retired_or_prior_restored": True,
            },
            "credential_binding": "bitrix_operational_edge_webhook",
            "staging_protocol": canary.BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        },
        "valid_from_unix_ms": NOW,
        "valid_until_unix_ms": NOW + 3_600_000,
        "producer_foundation_sha256": _sha("producer-foundation"),
        "model_route": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": "gpt-5.6-sol",
            "initial_effort": "high",
            "adaptive_max_effort": "max",
            "max_turns": 90,
        },
        "semantic_config_contract": semantic_config_contract,
        "semantic_config_contract_sha256": canary._digest(
            semantic_config_contract
        ),
        "required_toolsets": list(canary.REQUIRED_TOOLSETS),
        "required_toolsets_sha256": canary._digest(
            {"toolsets": list(canary.REQUIRED_TOOLSETS)}
        ),
        "public_discord_target": public_discord_target,
        "discord_bot_identities": discord_bot_identities,
        "capability_role_topology_contract": role_topology,
        "capability_role_topology_contract_sha256": canary._digest(
            role_topology
        ),
        "authority_keys": {
            role: {
                "key_id": hashlib.sha256(
                    key.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                ).hexdigest(),
                "algorithm": canary.AUTHORITY_ALGORITHMS[role],
                "public_key_ed25519_hex": key
                .public_key()
                .public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
                .hex(),
            }
            for role, key in keys.items()
        },
    }


def _common(schema: str, fixture: dict, fixture_sha256: str, offset: int) -> dict:
    return {
        "schema": schema,
        "run_id": fixture["run_id"],
        "release_sha": fixture["release_sha"],
        "fixture_sha256": fixture_sha256,
        "observed_at_unix_ms": NOW + offset,
    }


def _signed(
    role: str,
    payload: dict,
    *,
    fixture: dict,
    keys: dict[str, Ed25519PrivateKey],
    slot: str | None = None,
) -> dict:
    schema_slots = {
        canary.RUNTIME_RECEIPT_SCHEMA: "runtime",
        canary.TASK_WORKSPACE_GATEWAY_SCHEMA: "workspace_gateway",
        canary.TASK_WORKSPACE_WRITER_SCHEMA: "workspace_writer",
        canary.PLAN_APPROVAL_SCHEMA: "workspace_owner",
        canary.CAPABILITY_DENIAL_SCHEMA: "capability_denials",
        canary.DATABASE_RECONCILIATION_SCHEMA: "database_reconciliation",
        canary.BITRIX_EDGE_SCHEMA: "bitrix_edge",
        canary.BITRIX_WRITER_SCHEMA: "bitrix_writer",
        canary.DISCORD_EDGE_SCHEMA: "discord_edge",
        canary.DISCORD_WRITER_SCHEMA: "discord_writer",
        canary.FAILURE_GATEWAY_SCHEMA: "failure_gateway",
        canary.FAILURE_WRITER_SCHEMA: "failure_writer",
        canary.CLEANUP_RECEIPT_SCHEMA: "cleanup",
    }
    slot = slot or schema_slots[payload["schema"]]
    algorithm = fixture["authority_keys"][role]["algorithm"]
    unsigned = {
        "schema": canary.SIGNED_RECEIPT_SCHEMA,
        "authority_role": role,
        "key_id": fixture["authority_keys"][role]["key_id"],
        "signature_algorithm": algorithm,
        "payload": payload,
    }
    if role != "owner":
        bindings = [
            {
                "kind": kind,
                "source_identity_sha256": _sha(
                    f"{slot}:{kind}:source"
                ),
                "artifact_sha256": _sha(f"{slot}:{kind}:artifact"),
                "verification_receipt_sha256": _sha(
                    f"{slot}:{kind}:verification"
                ),
            }
            for kind in canary.SLOT_NATIVE_BINDING_KINDS[slot]
        ]
        if slot == "cleanup":
            next(
                item
                for item in bindings
                if item["kind"] == "production_diff_observation"
            )["artifact_sha256"] = payload["production_diff_sha256"]
        unsigned["native_evidence"] = {
            "schema": canary.NATIVE_EVIDENCE_SCHEMA,
            "producer_readiness_sha256": PRODUCER_READINESS_SHA256,
            "bindings": bindings,
        }
    message = canary._canonical_bytes(unsigned)
    signature = (
        _sshsig_signature(keys[role], message)
        if algorithm == "sshsig-ed25519-sha512"
        else keys[role].sign(message).hex()
    )
    return {
        **unsigned,
        "signature": signature,
    }


def _sshsig_signature(
    key: Ed25519PrivateKey,
    message: bytes,
    *,
    namespace_text: str = canary.OWNER_SSHSIG_NAMESPACE,
) -> str:
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    namespace = namespace_text.encode("ascii")
    reserved = b""
    hash_algorithm = b"sha512"
    signed = (
        b"SSHSIG"
        + canary._ssh_string(namespace)
        + canary._ssh_string(reserved)
        + canary._ssh_string(hash_algorithm)
        + canary._ssh_string(hashlib.sha512(message).digest())
    )
    public_blob = canary._ssh_string(b"ssh-ed25519") + canary._ssh_string(public)
    signature_blob = canary._ssh_string(b"ssh-ed25519") + canary._ssh_string(
        key.sign(signed)
    )
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + canary._ssh_string(public_blob)
        + canary._ssh_string(namespace)
        + canary._ssh_string(reserved)
        + canary._ssh_string(hash_algorithm)
        + canary._ssh_string(signature_blob)
    )
    encoded = base64.b64encode(envelope).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _resign(
    receipt: dict,
    *,
    fixture: dict,
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    role = receipt["authority_role"]
    producer_readiness_sha256 = (
        receipt.get("native_evidence", {}).get("producer_readiness_sha256")
    )
    slot = next(
        (
            name
            for name, expected_role in canary.SLOT_ROLE.items()
            if expected_role == role
            and (
                role == "owner"
                or [binding["kind"] for binding in receipt["native_evidence"]["bindings"]]
                == list(canary.SLOT_NATIVE_BINDING_KINDS[name])
            )
        ),
        None,
    )
    replacement = _signed(
        role, receipt["payload"], fixture=fixture, keys=keys, slot=slot
    )
    if role != "owner":
        replacement["native_evidence"]["producer_readiness_sha256"] = (
            producer_readiness_sha256
        )
        _resign_exact_envelope(replacement, keys=keys)
    receipt.clear()
    receipt.update(replacement)


def _resign_exact_envelope(
    receipt: dict,
    *,
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    role = receipt["authority_role"]
    unsigned = {
        key: value for key, value in receipt.items() if key != "signature"
    }
    message = canary._canonical_bytes(unsigned)
    receipt["signature"] = (
        _sshsig_signature(keys[role], message)
        if receipt["signature_algorithm"] == "sshsig-ed25519-sha512"
        else keys[role].sign(message).hex()
    )


def _terminal(
    label: str,
    *,
    state: str = "completed",
    resumed: bool = False,
) -> dict:
    blocked = state == "blocked"
    return {
        "case_id": f"case:{label}",
        "plan_id": f"plan:{label}",
        "revision": 3,
        "state": state,
        "terminal_event_id": f"event:{label}:terminal",
        "terminal_event_sha256": _sha(f"{label}-terminal"),
        "completed_step_ids": [f"step:{label}:1", f"step:{label}:2"],
        "verification_event_ids": [f"verify:{label}:1", f"verify:{label}:2"],
        "pending_step_count": 0,
        "blocked_step_count": 1 if blocked else 0,
        "resumed_after_restart": resumed,
        "replayed_mutation_count": 0,
        "blocker_event_id": f"event:{label}:blocked" if blocked else None,
        "blocker_receipt_sha256": _sha(f"{label}-blocked") if blocked else None,
    }


def _producer_authority_and_activation(
    *,
    fixture: dict,
    fixture_sha256: str,
    keys: dict[str, Ed25519PrivateKey],
    owner_grant: dict,
) -> tuple[dict, dict]:
    owner = owner_grant["payload"]
    command_bytes = (b"command-1", b"command-2")
    catalog = producers.build_probe_catalog(
        release_sha=fixture["release_sha"],
        capability_plan_sha256=fixture["capability_plan_sha256"],
        full_canary_plan_sha256=fixture["full_canary_plan_sha256"],
        fixture_sha256=fixture_sha256,
        run_id=fixture["run_id"],
        session_id=owner["session_id"],
        capability_epoch_sha256=owner["capability_epoch_sha256"],
        case_ids={
            name: f"case:{name.replace('_', '-')}"
            for name in (
                "workspace_continuation",
                "capability_denials",
                "database_reconciliation",
                "bitrix_boundary",
                "discord_routeback",
                "failure_recovery",
            )
        },
        workspace={
            "first_path_probe_id": "probe:workspace:first",
            "alternate_path_probe_id": "probe:workspace:alternate",
            "worker_restart_checkpoint_step_id": "step:workspace:restart",
        },
        commands={
            "allowed": [
                {
                    "command_id": f"command:{index}",
                    "command_b64": base64.b64encode(raw).decode("ascii"),
                    "command_sha256": hashlib.sha256(raw).hexdigest(),
                    "max_uses": 1,
                }
                for index, raw in enumerate(command_bytes, start=1)
            ],
            "denied": [
                {
                    "kind": kind,
                    "command": {
                        "command_id": f"denied:{kind}",
                        "command_b64": base64.b64encode(
                            raw := f"denied:{kind}".encode()
                        ).decode("ascii"),
                        "command_sha256": hashlib.sha256(raw).hexdigest(),
                        "max_uses": 1,
                    },
                }
                for kind in producers.DENIAL_KINDS
            ],
        },
        database={
            "row_key": "row:one",
            "idempotency_key": "idempotency:database",
            "read_probe_id": "probe:database:read",
            "write_probe_id": "probe:database:write",
            "lost_response_probe_id": "probe:database:lost",
        },
        bitrix={
            "handoff_id": "handoff-bitrix",
            "selected_edge_id": "operational-edge:bitrix",
            "read_operation_id": "bitrix.crm.status_list",
            "read_arguments": dict(producers.BITRIX_CANARY_READ_ARGUMENTS),
            "initial_read_probe_id": "probe-bitrix-status-initial",
            "readback_probe_id": "probe-bitrix-status-readback",
            "normalized_equality_excluded_fields": ["generated_at_utc"],
            "mutation_operation_id": "bitrix.crm.lead_add",
            "mutation_arguments": dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
            "mutation_probe_id": "probe-bitrix-lead-add-denied",
        },
        discord={
            "public_target": copy.deepcopy(fixture["public_discord_target"]),
            "public_idempotency_key": "discord:public:one",
            "private_target_kind": "dm",
            "private_probe_id": "discord:private:one",
        },
        failure={
            "probes": [
                {
                    "component": component,
                    "failure_id": f"failure:{component}",
                    "alternative_available": True,
                    "alternative_id": f"alternative:{component}",
                }
                for component in producers.FAILURE_COMPONENTS
            ]
        },
    )
    assert sorted(
        item["command_sha256"] for item in catalog["commands"]["allowed"]
    ) == owner["command_sha256s"]
    outer_unsigned = {
        "schema": producers.API_ADMISSION_OWNER_AUTHORITY_SCHEMA,
        "challenge_sha256": _sha("api-admission-challenge"),
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_id": fixture["run_id"],
        "session_id": owner["session_id"],
        "capability_epoch_sha256": owner["capability_epoch_sha256"],
        "issued_at_unix_ms": owner["observed_at_unix_ms"],
        "valid_until_unix_ms": (
            owner["observed_at_unix_ms"] + owner["ttl_seconds"] * 1000
        ),
        "catalog": catalog,
        "workspace_owner_receipt": copy.deepcopy(owner_grant),
    }
    authority = {
        **outer_unsigned,
        "owner_signature": _sshsig_signature(
            keys["owner"], canary._canonical_bytes(outer_unsigned)
        ),
    }
    authority_sha256 = canary._digest(authority)
    endpoint_readiness = {
        role: {
            "foundation_sha256": fixture["producer_foundation_sha256"],
            "release_sha": fixture["release_sha"],
            "capability_plan_sha256": fixture["capability_plan_sha256"],
            "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
            "key_id": fixture["authority_keys"][role]["key_id"],
            "public_key_ed25519_hex": fixture["authority_keys"][role][
                "public_key_ed25519_hex"
            ],
        }
        for role in producers.ENDPOINT_ROLES
    }
    activation_unsigned = {
        "schema": producers.PRODUCER_FLEET_READINESS_SCHEMA,
        "foundation_sha256": fixture["producer_foundation_sha256"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_id": fixture["run_id"],
        "run_receipt_root": (
            f"/var/lib/muncho-capability-canary-evidence/{fixture['run_id']}"
        ),
        "owner_id": fixture["owner_id"],
        "valid_from_unix_ms": fixture["valid_from_unix_ms"],
        "valid_until_unix_ms": fixture["valid_until_unix_ms"],
        "receipt_slots": list(producers.RECEIPT_SLOTS),
        "slot_filenames": dict(producers.SLOT_FILENAME),
        "slot_roles": dict(producers.SLOT_ROLE),
        "slot_native_binding_kinds": {
            slot: list(producers.SLOT_NATIVE_BINDING_KINDS[slot])
            for slot in producers.RECEIPT_SLOTS
        },
        "authority_keys": copy.deepcopy(fixture["authority_keys"]),
        "endpoint_readiness": endpoint_readiness,
        "owner_authority": {
            "producer_kind": "pre_staged_sshsig_grant",
            "owner_id": fixture["owner_id"],
            "key_id": fixture["authority_keys"]["owner"]["key_id"],
            "public_key_source_sha256": _sha("owner-public-key-source"),
            "grant_sha256": canary._digest(owner_grant),
            "grant_path": str(producers.DEFAULT_OWNER_GRANT_PATH),
            "catalog_sha256": catalog["catalog_sha256"],
            "catalog_path": str(producers.DEFAULT_PROBE_CATALOG_PATH),
            "authority_sha256": authority_sha256,
        },
        "discord_bot_identities": {
            **copy.deepcopy(fixture["discord_bot_identities"]),
            "routeback_identity_attestation_sha256": _sha("routeback-attestation"),
            "routeback_credential_file_metadata_sha256": _sha(
                "routeback-credential-file"
            ),
        },
        "producer_protocol": "role_local_native_evidence_v1",
        "root_can_sign_non_observer_roles": False,
        "token_or_token_digest_recorded": False,
        "observed_at_unix_ms": NOW + 14,
    }
    activation = {
        **activation_unsigned,
        "readiness_sha256": canary._digest(activation_unsigned),
    }
    return authority, activation


def _production_diff(fixture: dict, fixture_sha256: str) -> dict:
    target = {
        "project": "adventico-ai-platform",
        "zone": "europe-west3-a",
        "vm": "ai-platform-runtime-01",
        "instance_id": "1094477181810932795",
    }
    surface_names = (
        "code",
        "config",
        "identities_permissions",
        "jobs",
        "migration_assets",
    )
    surface_diffs = {
        name: {
            "before_sha256": _sha(f"production-{name}"),
            "after_sha256": _sha(f"production-{name}"),
            "changed": False,
        }
        for name in surface_names
    }
    expected_contract = {
        "schema": "muncho-production-capability-production-no-change-contract.v1",
        "run_id": fixture["run_id"],
        "canary_revision": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "target": target,
        "expected_changed_surfaces": [],
        "boot_identity_change_allowed": True,
    }
    static_sha256 = _sha("production-static-observation")
    unsigned = {
        "schema": "muncho-production-capability-production-diff.v1",
        "run_id": fixture["run_id"],
        "canary_revision": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "target": target,
        "before_envelope_sha256": _sha("production-before-envelope"),
        "after_envelope_sha256": _sha("production-after-envelope"),
        "before_observation_sha256": _sha("production-before-observation"),
        "after_observation_sha256": _sha("production-after-observation"),
        "before_observed_at_unix_ms": NOW + 1,
        "after_observed_at_unix_ms": NOW + 89,
        "static_before_sha256": static_sha256,
        "static_after_sha256": static_sha256,
        "changed_surfaces": [],
        "surface_diffs": surface_diffs,
        "expected_change_contract_sha256": canary._digest(expected_contract),
        "unexpected_change_count": 0,
        "production_mutation_observed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_job_content_recorded": False,
    }
    return {**unsigned, "diff_sha256": canary._digest(unsigned)}


def _goal_continuation_evidence(
    *,
    fixture: dict,
    fixture_sha256: str,
    owner_approval_receipt_sha256: str,
    production_diff_sha256: str,
) -> dict:
    replay_binding = {
        "run_id": fixture["run_id"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt_sha256,
    }

    def bound(schema: str, digest_field: str, **fields: object) -> dict:
        unsigned = {
            "schema": schema,
            **replay_binding,
            **fields,
        }
        return {**unsigned, digest_field: canary._digest(unsigned)}

    session_id = f"capability_{fixture['run_id']}"
    generation_id = "goal-generation-1"
    ingress = bound(
        canary.DISCORD_OWNER_INGRESS_SCHEMA,
        "receipt_sha256",
        challenge_id="goal-owner-challenge-1",
        challenge_sha256=_sha("goal-owner-challenge"),
        wait_started_at_unix_ms=NOW + 13,
        discord_event_id="1527000000000000001",
        discord_event_sha256=_sha("goal-owner-discord-event"),
        target_type="public_guild_channel",
        guild_id=fixture["public_discord_target"]["guild_id"],
        channel_id=fixture["public_discord_target"]["channel_id"],
        owner_user_id=fixture["owner_id"],
        connector_bot_user_id=fixture["discord_bot_identities"][
            "connector_bot_user_id"
        ],
        delivery_id_sha256=_sha("goal-owner-delivery"),
        connector_journal_row_sha256=_sha("goal-owner-journal-row"),
        journal_state="acked",
        offered_at_unix_ms=NOW + 14,
        acked_at_unix_ms=NOW + 15,
        acked=True,
        message_content_recorded=False,
    )
    outcome_specs = (
        ("goal-turn-1", "continue", 15, 16),
        ("goal-turn-2", "continue", 19, 22),
        ("goal-turn-3", "complete", 23, 24),
    )
    goal_state_sha256s = [
        _sha(f"goal-state-{ordinal}") for ordinal in range(len(outcome_specs) + 1)
    ]
    outcomes = [
        bound(
            canary.GOAL_MODEL_OUTCOME_SCHEMA,
            "receipt_sha256",
            session_id=session_id,
            goal_generation_id=generation_id,
            turn_id=turn_id,
            ordinal=ordinal,
            outcome=outcome,
            reason_sha256=_sha(f"goal-outcome-reason-{ordinal}"),
            todo_tool_call_id_sha256=_sha(f"goal-todo-call-{ordinal}"),
            goal_state_before_sha256=goal_state_sha256s[ordinal - 1],
            goal_state_after_sha256=goal_state_sha256s[ordinal],
            turn_started_at_unix_ms=NOW + started_offset,
            observed_at_unix_ms=NOW + observed_offset,
            structured_outcome_source="todo.goal_outcome",
            goal_manager="hermes_cli.goals.GoalManager",
            goal_max_turns=0,
        )
        for ordinal, (turn_id, outcome, started_offset, observed_offset) in enumerate(
            outcome_specs,
            start=1,
        )
    ]

    def service_identity(
        *, pid: int, invocation_id: str, monotonic: int, n_restarts: int
    ) -> dict:
        unsigned = {
            "service_unit": "hermes-cloud-gateway.service",
            "main_pid": pid,
            "invocation_id": invocation_id,
            "active_enter_timestamp_monotonic": monotonic,
            "n_restarts": n_restarts,
        }
        return {**unsigned, "identity_sha256": canary._digest(unsigned)}

    restart = bound(
        canary.GATEWAY_RESTART_RECEIPT_SCHEMA,
        "receipt_sha256",
        service_unit="hermes-cloud-gateway.service",
        restart_count=1,
        continuation_turn_id="goal-turn-2",
        continuation_started_at_unix_ms=NOW + 19,
        restart_requested_at_unix_ms=NOW + 19,
        restart_completed_at_unix_ms=NOW + 20,
        pre_service_identity=service_identity(
            pid=21001,
            invocation_id="1" * 32,
            monotonic=1000,
            n_restarts=0,
        ),
        post_service_identity=service_identity(
            pid=21002,
            invocation_id="2" * 32,
            monotonic=2000,
            n_restarts=1,
        ),
        controlled_restart=True,
    )
    ctw_recovery = bound(
        canary.CTW_RECOVERY_RECEIPT_SCHEMA,
        "receipt_sha256",
        session_id=session_id,
        goal_generation_id=generation_id,
        case_id="goal-case-1",
        plan_id="goal-plan-1",
        plan_revision=3,
        checkpoint_event_id="goal-checkpoint-1",
        checkpoint_event_sha256=_sha("goal-checkpoint-event"),
        recovery_readback_sha256=_sha("goal-recovery-readback"),
        next_step_id="goal-step-3",
        terminal_event_id="goal-terminal-1",
        terminal_event_sha256=_sha("goal-terminal-event"),
        terminal_state="completed",
        recovery_boundary="gateway_restart_resume",
        resumed_after_restart=True,
        replayed_mutation_count=0,
        observed_at_unix_ms=NOW + 24,
    )
    model_sha256 = hashlib.sha256(b"gpt-5.6-sol").hexdigest()
    prompt_sha256 = _sha("goal-system-prompt")
    tool_schema_sha256 = _sha("goal-tool-schema")
    pre_identity_sha256 = restart["pre_service_identity"]["identity_sha256"]
    post_identity_sha256 = restart["post_service_identity"]["identity_sha256"]
    api_call_observations = [
        {
            "ordinal": 1,
            "phase": "pre_restart",
            "turn_id": "goal-turn-1",
            "gateway_process_identity_sha256": pre_identity_sha256,
            "pre_api_request_frame_sha256": _sha("goal-pre-api-1"),
            "post_api_request_frame_sha256": _sha("goal-post-api-1"),
            "system_prompt_sha256": prompt_sha256,
            "tool_schema_sha256": tool_schema_sha256,
            "response_model_sha256": model_sha256,
            "started_at_unix_ms": NOW + 15,
            "completed_at_unix_ms": NOW + 16,
        },
        {
            "ordinal": 2,
            "phase": "post_restart",
            "turn_id": "goal-turn-2",
            "gateway_process_identity_sha256": post_identity_sha256,
            "pre_api_request_frame_sha256": _sha("goal-pre-api-2"),
            "post_api_request_frame_sha256": _sha("goal-post-api-2"),
            "system_prompt_sha256": prompt_sha256,
            "tool_schema_sha256": tool_schema_sha256,
            "response_model_sha256": model_sha256,
            "started_at_unix_ms": NOW + 21,
            "completed_at_unix_ms": NOW + 22,
        },
        {
            "ordinal": 3,
            "phase": "post_restart",
            "turn_id": "goal-turn-3",
            "gateway_process_identity_sha256": post_identity_sha256,
            "pre_api_request_frame_sha256": _sha("goal-pre-api-3"),
            "post_api_request_frame_sha256": _sha("goal-post-api-3"),
            "system_prompt_sha256": prompt_sha256,
            "tool_schema_sha256": tool_schema_sha256,
            "response_model_sha256": model_sha256,
            "started_at_unix_ms": NOW + 23,
            "completed_at_unix_ms": NOW + 24,
        },
    ]
    model_route = bound(
        canary.MODEL_ROUTE_RECEIPT_SCHEMA,
        "receipt_sha256",
        provider="openai-codex",
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        base_url_sha256=_sha("openai-codex-base-url"),
        fallback_configured=False,
        fallback_used=False,
        model_call_count=3,
        response_model_sha256s=[model_sha256, model_sha256, model_sha256],
        api_call_observations=api_call_observations,
        observed_at_unix_ms=NOW + 24,
    )
    stability = bound(
        canary.PROMPT_TOOL_STABILITY_RECEIPT_SCHEMA,
        "receipt_sha256",
        pre_restart_system_prompt_sha256=prompt_sha256,
        post_restart_system_prompt_sha256=prompt_sha256,
        pre_restart_tool_schema_sha256=tool_schema_sha256,
        post_restart_tool_schema_sha256=tool_schema_sha256,
        prompt_cache_stable=True,
        tool_schema_stable=True,
        process_reconstruction_exact=True,
        observed_at_unix_ms=NOW + 24,
    )
    preemption = bound(
        canary.USER_PREEMPTION_QUEUE_E2E_RECEIPT_SCHEMA,
        "receipt_sha256",
        queue_path="gateway.GatewayRunner._post_turn_goal_continuation",
        owner_event_id="1527000000000000002",
        owner_event_sha256=_sha("goal-owner-preemption-event"),
        goal_ingress_connector_row_sha256=_sha("goal-ingress-journal"),
        preemption_ingress_connector_row_sha256=_sha(
            "goal-preemption-journal"
        ),
        automatic_continuation_was_pending=True,
        real_user_event_preempted_automatic_continuation=True,
        queued_user_event_identity_preserved=True,
        connector_ack_readback_after_restart=True,
        transport_redelivery_required=False,
        duplicate_transport_delivery_policy=(
            "same_delivery_id_exact_ack_replay_only"
        ),
        duplicate_model_turn_count=0,
        live_observed=True,
        observed_at_unix_ms=NOW + 24,
    )
    terminal_routeback = bound(
        canary.GOAL_TERMINAL_ROUTEBACK_RECEIPT_SCHEMA,
        "receipt_sha256",
        session_id=session_id,
        goal_generation_id=generation_id,
        case_id="goal-case-1",
        event_id="goal-terminal-routeback-1",
        event_sha256=_sha("goal-terminal-routeback-event"),
        canonical_content_sha256=_sha("goal-terminal-routeback-content"),
        message_id="1527000000000000099",
        channel_id=fixture["public_discord_target"]["channel_id"],
        content_sha256=_sha("goal-terminal-routeback-message"),
        public_receipt_sha256=_sha("goal-terminal-routeback-receipt"),
        adapter_receipt=True,
        receipt_readback_verified=True,
        writer_origin="routeback_finalize_sent",
        writer_appended_at="2026-07-13T12:00:00+00:00",
        observed_at_unix_ms=NOW + 24,
    )
    isolation_projection = canary.normalize_goal_continuation_isolation_projection(
        fixture=fixture
    )
    terminal = bound(
        canary.GOAL_CONTINUATION_TERMINAL_SCHEMA,
        "terminal_sha256",
        discord_owner_ingress_receipt_sha256=ingress["receipt_sha256"],
        session_id=session_id,
        goal_generation_id=generation_id,
        continue_outcome_receipt_sha256s=[
            item["receipt_sha256"] for item in outcomes[:-1]
        ],
        completion_outcome_receipt_sha256=outcomes[-1]["receipt_sha256"],
        gateway_restart_receipt_sha256=restart["receipt_sha256"],
        ctw_recovery_receipt_sha256=ctw_recovery["receipt_sha256"],
        model_route_receipt_sha256=model_route["receipt_sha256"],
        prompt_tool_stability_receipt_sha256=stability["receipt_sha256"],
        user_preemption_queue_e2e_receipt_sha256=preemption["receipt_sha256"],
        terminal_routeback_receipt_sha256=terminal_routeback[
            "receipt_sha256"
        ],
        production_diff_sha256=production_diff_sha256,
        isolation_equivalence_projection=isolation_projection,
        isolation_equivalence_projection_sha256=canary._digest(
            isolation_projection
        ),
        completed_at_unix_ms=NOW + 24,
    )
    return bound(
        canary.GOAL_CONTINUATION_EVIDENCE_SCHEMA,
        "evidence_sha256",
        discord_owner_ingress=ingress,
        model_outcomes=outcomes,
        gateway_restart=restart,
        ctw_recovery=ctw_recovery,
        model_route=model_route,
        prompt_tool_stability=stability,
        user_preemption_queue_e2e=preemption,
        terminal_routeback=terminal_routeback,
        terminal=terminal,
    )


def _evidence(
    fixture: dict,
    fixture_sha256: str,
    keys: dict[str, Ed25519PrivateKey],
) -> dict:
    sign = lambda role, payload: _signed(  # noqa: E731 - compact fixture builder
        role, payload, fixture=fixture, keys=keys
    )
    runtime = sign(
        "gateway_observer",
        {
            **_common(canary.RUNTIME_RECEIPT_SCHEMA, fixture, fixture_sha256, 10),
            "host_identity_sha256": fixture["host_identity_sha256"],
            "release_artifact_sha256": fixture["release_artifact_sha256"],
            "installed_wheel_manifest_sha256": fixture[
                "installed_wheel_manifest_sha256"
            ],
            "effective_config_sha256": fixture["effective_config_sha256"],
            "tool_inventory_sha256": fixture["tool_inventory_sha256"],
            **fixture["model_route"],
            "semantic_config_contract": copy.deepcopy(
                fixture["semantic_config_contract"]
            ),
            "semantic_config_contract_sha256": fixture[
                "semantic_config_contract_sha256"
            ],
            "toolsets": list(canary.REQUIRED_TOOLSETS),
            "ordered_toolsets_sha256": fixture["required_toolsets_sha256"],
            "capability_role_topology_contract": copy.deepcopy(
                fixture["capability_role_topology_contract"]
            ),
            "capability_role_topology_contract_sha256": fixture[
                "capability_role_topology_contract_sha256"
            ],
            "kanban_auxiliary_planning_enabled": False,
            "kanban_auto_decompose": False,
            "kanban_dispatch_in_gateway": False,
            "prompt_cache_stable": True,
            "message_alternation_valid": True,
            "gateway_process_identity_sha256": _sha("gateway-process"),
            "connector_bot_user_id": fixture["discord_bot_identities"][
                "connector_bot_user_id"
            ],
            "connector_bot_user_id_provenance": (
                canary.CONNECTOR_BOT_ID_PROVENANCE
            ),
            "connector_readiness_receipt_sha256": _sha("connector-readiness"),
        },
    )

    commands = sorted([_sha("command-1"), _sha("command-2")])
    workspace_terminal = _terminal("workspace", resumed=True)
    owner_grant = sign(
        "owner",
        {
            **_common(canary.PLAN_APPROVAL_SCHEMA, fixture, fixture_sha256, 12),
            "approval_id": "approval-workspace",
            "owner_id": fixture["owner_id"],
            "session_id": f"capability_{fixture['run_id']}",
            "capability_epoch_sha256": _sha("epoch-workspace"),
            "command_sha256s": commands,
            "ttl_seconds": 7200,
            "max_uses": 10,
        },
    )
    owner_grant_sha256 = canary._digest(owner_grant)
    production_diff = _production_diff(fixture, fixture_sha256)
    goal_continuation = _goal_continuation_evidence(
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_grant_sha256,
        production_diff_sha256=production_diff["diff_sha256"],
    )
    workspace = {
        "gateway_receipt": sign(
            "gateway_observer",
            {
                **_common(
                    canary.TASK_WORKSPACE_GATEWAY_SCHEMA,
                    fixture,
                    fixture_sha256,
                    20,
                ),
                "session_id": f"capability_{fixture['run_id']}",
                "capability_epoch_sha256": _sha("epoch-workspace"),
                "transcript_sha256": _sha("workspace-transcript"),
                "task_workspace_evidence_sha256s": sorted(
                    [
                        _sha("canonical-plan-progress"),
                        _sha("canonical-restart-resume"),
                    ]
                ),
                "first_path_failure_receipt_sha256": _sha("first-path-failure"),
                "alternate_read_receipt_sha256": _sha("alternate-read"),
                "model_requested_effort": "max",
                "later_request_effort": "max",
                "reasoning_tool_call_id": "call-reasoning-max",
                "restart_count": 1,
                "used_command_sha256s": commands,
                "mutation_receipt_sha256s": [_sha("mutation-1"), _sha("mutation-2")],
                "approval_prompt_count": 0,
                "microapproval_prompt_count": 0,
                "replayed_mutation_count": 0,
                "owner_grant_id": "approval-workspace",
                "owner_grant_sha256": owner_grant_sha256,
                "consumed_command_sha256s": commands,
                "terminal_plan_id": workspace_terminal["plan_id"],
                "terminal_plan_revision": workspace_terminal["revision"],
                "goal_continuation_evidence": goal_continuation,
            },
        ),
        "writer_receipt": sign(
            "canonical_writer",
            {
                **_common(
                    canary.TASK_WORKSPACE_WRITER_SCHEMA,
                    fixture,
                    fixture_sha256,
                    21,
                ),
                "session_id": f"capability_{fixture['run_id']}",
                "capability_epoch_sha256": _sha("epoch-workspace"),
                "owner_grant_id": "approval-workspace",
                "owner_grant_sha256": owner_grant_sha256,
                "consumed_command_sha256s": commands,
                "terminal_ctw": workspace_terminal,
            },
        ),
        "owner_approval_receipt": owner_grant,
    }

    denials = sign(
        "canonical_writer",
        {
            **_common(canary.CAPABILITY_DENIAL_SCHEMA, fixture, fixture_sha256, 30),
            "session_id": "session-denials",
            "capability_epoch_sha256": _sha("epoch-denials"),
            "denials": [
                {
                    "kind": kind,
                    "denied": True,
                    "dispatch_attempted": False,
                    "receipt_sha256": _sha(f"denial-{kind}"),
                }
                for kind in canary.DENIAL_KINDS
            ],
        },
    )

    database = sign(
        "canonical_writer",
        {
            **_common(
                canary.DATABASE_RECONCILIATION_SCHEMA, fixture, fixture_sha256, 40
            ),
            "read_receipt_sha256": _sha("db-read"),
            "transaction_receipt_sha256": _sha("db-transaction"),
            "idempotency_key_sha256": _sha("db-idempotency"),
            "lost_response_observed": True,
            "reconciled_before_retry": True,
            "readback_verified": True,
            "readback_receipt_sha256": _sha("db-readback"),
            "durable_row_count": 1,
            "duplicate_row_count": 0,
            "terminal_ctw": _terminal("database"),
        },
    )

    bitrix = {
        "edge_receipt": sign(
            "business_edge",
            {
                **_common(canary.BITRIX_EDGE_SCHEMA, fixture, fixture_sha256, 50),
                "handoff_id": "handoff-bitrix",
                "selected_edge_id": "operational-edge:bitrix",
                "operational_edge_contract_sha256": canary._digest(
                    fixture["bitrix_operational_edge_contract"]
                ),
                "operational_edge_service_identity_sha256": fixture[
                    "bitrix_operational_edge_contract"
                ]["service_identity_sha256"],
                "operational_edge_asset_manifest_sha256": fixture[
                    "bitrix_operational_edge_contract"
                ]["asset_manifest_sha256"],
                "read_probe": {
                    "selected_edge_id": "operational-edge:bitrix",
                    "read_operation_id": "bitrix.crm.status_list",
                    "read_arguments": {"entity_id": "STATUS"},
                    "initial_read_probe_id": "probe-bitrix-status-initial",
                    "readback_probe_id": "probe-bitrix-status-readback",
                    "normalized_equality_excluded_fields": [
                        "generated_at_utc"
                    ],
                    "stable_normalized_equality": True,
                },
                "request_sha256": _sha("bitrix-request"),
                "operation_id": "bitrix.crm.status_list",
                "arguments_sha256": canary._digest({"entity_id": "STATUS"}),
                "operational_edge_receipt_sha256": _sha(
                    "bitrix-operational-edge-receipt"
                ),
                "authenticated_live_readback_sha256": _sha(
                    "bitrix-authenticated-live-readback"
                ),
                "readback_verified": True,
            },
        ),
        "writer_receipt": sign(
            "canonical_writer",
            {
                **_common(canary.BITRIX_WRITER_SCHEMA, fixture, fixture_sha256, 51),
                "handoff_id": "handoff-bitrix",
                "selected_edge_id": "operational-edge:bitrix",
                "selection_event_id": "event-bitrix-selection",
                "selection_event_sha256": _sha("bitrix-selection"),
                "mutation_probe": {
                    "selected_edge_id": "operational-edge:bitrix",
                    "mutation_operation_id": "bitrix.crm.lead_add",
                    "mutation_arguments": {
                        "title": "Muncho capability canary dry-run",
                        "requester": "capability-canary",
                        "reason": (
                            "Verify pre-dispatch denial without mutation"
                        ),
                        "execute": False,
                    },
                    "mutation_probe_id": "probe-bitrix-lead-add-denied",
                },
                "mutation_approval_present": False,
                "mutation_dispatched": False,
                "mutation_denial_receipt_sha256": _sha(
                    "bitrix-mutation-denial"
                ),
                "blocked_event_id": "event-bitrix-mutation-blocked",
                "blocked_receipt_sha256": _sha("bitrix-blocked"),
                "terminal_ctw": _terminal("bitrix"),
            },
        ),
    }

    discord_terminal = _terminal("discord")
    discord_source_refs = {"thread_id": "thread:discord-canary-source"}
    sent_receipt = {
        "platform": "discord",
        "adapter_receipt": True,
        "receipt_readback_verified": True,
        "message_id": "1504852355588423999",
        "channel_id": fixture["public_discord_target"]["channel_id"],
        "content_sha256": _sha("discord-content"),
        "public_receipt_sha256": _sha("discord-public-receipt"),
    }
    private_receipt = {
        "private_denial_receipt_sha256": _sha("discord-dm-denial"),
        "dispatch_attempted": False,
    }
    sent_event_binding = {
        "schema": canary.CANONICAL_ROUTEBACK_EVENT_BINDING_SCHEMA,
        "event_id": "event-route-back-sent",
        "event_type": "route_back.sent",
        "case_id": discord_terminal["case_id"],
        "event_sha256": _sha("route-back-sent"),
        "canonical_content_sha256": _sha("route-back-sent-content"),
        "source_refs": discord_source_refs,
        "returned_receipt": copy.deepcopy(sent_receipt),
        "payload_receipt": copy.deepcopy(sent_receipt),
        "route_back_receipt": copy.deepcopy(sent_receipt),
    }
    blocked_event_binding = {
        "schema": canary.CANONICAL_ROUTEBACK_EVENT_BINDING_SCHEMA,
        "event_id": "event-route-back-blocked",
        "event_type": "route_back.blocked",
        "case_id": discord_terminal["case_id"],
        "event_sha256": _sha("route-back-blocked"),
        "canonical_content_sha256": _sha("route-back-blocked-content"),
        "source_refs": discord_source_refs,
        "returned_receipt": copy.deepcopy(private_receipt),
        "payload_receipt": copy.deepcopy(private_receipt),
        "route_back_receipt": copy.deepcopy(private_receipt),
        "payload_dispatch_attempted": False,
        "route_back_dispatch_attempted": False,
    }
    discord = {
        "edge_receipt": sign(
            "discord_edge",
            {
                **_common(canary.DISCORD_EDGE_SCHEMA, fixture, fixture_sha256, 60),
                **fixture["public_discord_target"],
                "idempotency_key_sha256": _sha("discord-idempotency"),
                "request_sha256": _sha("discord-request"),
                "content_sha256": _sha("discord-content"),
                "platform_message_id": "1504852355588423999",
                "adapter_accepted": True,
                "public_readback_verified": True,
                "public_receipt_sha256": _sha("discord-public-receipt"),
                "private_target_kind": "dm",
                "private_dispatch_attempted": False,
                "journal_unchanged_after_private_probe": True,
                "private_denial_receipt_sha256": _sha("discord-dm-denial"),
                "routeback_bot_user_id": fixture["discord_bot_identities"][
                    "routeback_bot_user_id"
                ],
                "routeback_bot_user_id_provenance": (
                    canary.ROUTEBACK_BOT_ID_PROVENANCE
                ),
            },
        ),
        "writer_receipt": sign(
            "canonical_writer",
            {
                **_common(canary.DISCORD_WRITER_SCHEMA, fixture, fixture_sha256, 61),
                "sent_event": sent_event_binding,
                "sent_after_verified_readback": True,
                "blocked_event": blocked_event_binding,
                "blocked_before_dispatch": True,
                "terminal_ctw": discord_terminal,
            },
        ),
    }

    failures = {
        "gateway_receipt": sign(
            "gateway_observer",
            {
                **_common(canary.FAILURE_GATEWAY_SCHEMA, fixture, fixture_sha256, 70),
                "transcript_sha256": _sha("failure-transcript"),
                "failures": [
                    {
                        "component": component,
                        "failure_observed": True,
                        "failure_receipt_sha256": _sha(f"failure-{component}"),
                        "alternative_available": True,
                        "alternative_attempted": True,
                        "alternative_receipt_sha256": _sha(f"alternative-{component}"),
                    }
                    for component in canary.FAILURE_COMPONENTS
                ],
                "model_retained_tool_control": True,
            },
        ),
        "writer_receipt": sign(
            "canonical_writer",
            {
                **_common(canary.FAILURE_WRITER_SCHEMA, fixture, fixture_sha256, 71),
                "terminal_ctw": _terminal("failure"),
            },
        ),
    }

    observer_identity = {
        "role": "gateway_observer",
        "service_unit": canary.OBSERVER_PRODUCER_SERVICE_UNIT,
        "live": True,
        "signing_only": True,
        "credential_read_access": False,
        "service_state_sha256": _sha("observer-live-service-state"),
        "producer_foundation_sha256": fixture[
            "producer_foundation_sha256"
        ],
        "unit_bundle_manifest_sha256": _sha("producer-unit-bundle"),
        "credential_inaccessibility_contract_sha256": _sha(
            "producer-credential-inaccessibility"
        ),
    }
    credential_stop_unsigned = {
        "schema": canary.CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA,
        "plan_sha256": fixture["capability_plan_sha256"],
        "non_observer_stop_order": list(canary.NON_OBSERVER_SERVICE_UNITS),
        "non_observer_services_state_sha256": _sha(
            "non-observer-services-state"
        ),
        "all_credential_consumers_stopped": True,
        "observer_service_unit": canary.OBSERVER_PRODUCER_SERVICE_UNIT,
        "observer_state_sha256": observer_identity["service_state_sha256"],
        "observer_live_signing_only": True,
        "observer_credential_read_access": False,
        "producer_foundation_sha256": fixture[
            "producer_foundation_sha256"
        ],
        "unit_bundle_manifest_sha256": observer_identity[
            "unit_bundle_manifest_sha256"
        ],
        "credential_inaccessibility_contract_sha256": observer_identity[
            "credential_inaccessibility_contract_sha256"
        ],
        "observed_at_unix": NOW // 1000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    credential_stop_proof = {
        **credential_stop_unsigned,
        "stop_proof_sha256": canary._digest(credential_stop_unsigned),
    }

    retirements = {}
    credential_absence = {}
    for index, binding in enumerate(canary.CREDENTIAL_LEASES):
        lease_id = f"lease-{binding.replace('_', '-')}"
        journal_root = f"/var/lib/muncho-capability-canary-leases/{lease_id}"
        target_path = f"/run/credentials/muncho-canary/{binding}"
        completion_unsigned = {
            "operation": "retirement_completion",
            "state": "retired",
            "kind": canary.CREDENTIAL_LEASE_KINDS[binding],
            "credential_binding": binding,
            "revision": fixture["release_sha"],
            "plan_sha256": fixture["capability_plan_sha256"],
            "full_canary_plan_sha256": fixture[
                "full_canary_plan_sha256"
            ],
            "lease_id": lease_id,
            "target_path": target_path,
            "target_device": 1,
            "target_inode": 10_000 + index,
            "target_uid": 1000 + index,
            "target_gid": 1000 + index,
            "target_mode": "0400",
            "target_size": 128,
            "target_mtime_ns": 100_000 + index,
            "target_ctime_ns": 200_000 + index,
            "install_receipt_path": f"{journal_root}/install-receipt.json",
            "install_receipt_sha256": _sha(f"install-receipt-{binding}"),
            "retirement_intent_path": (
                f"{journal_root}/retirement-intent.json"
            ),
            "retirement_intent_sha256": _sha(
                f"retirement-intent-{binding}"
            ),
            "service_stop_proof_sha256": credential_stop_proof[
                "stop_proof_sha256"
            ],
            "service_stop_observed_at_unix": credential_stop_proof[
                "observed_at_unix"
            ],
            "removed": True,
            "absent": True,
            "absent_after_stop": True,
            "retired_at_unix": NOW // 1000,
            "schema": canary.SECRET_RETIREMENT_COMPLETION_SCHEMA,
            "receipt_path": f"{journal_root}/retirement-completion.json",
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        retirements[binding] = {
            **completion_unsigned,
            "receipt_sha256": canary._digest(completion_unsigned),
        }
        credential_absence[binding] = {
            "path": target_path,
            "absent": True,
        }
    key_contract = fixture["bitrix_operational_edge_contract"][
        "receipt_key_contract"
    ]
    service_stop_proof_sha256 = credential_stop_proof[
        "stop_proof_sha256"
    ]
    key_retirement_root = (
        "/var/lib/muncho-capability-canary-control/"
        f"bitrix-key-retirements/{key_contract['public_key_id']}"
    )
    key_retirement_unsigned = {
        "schema": canary.INTERNAL_KEY_RETIREMENT_SCHEMA,
        "operation": "retire_bitrix_receipt_key_pair",
        "reason": "service_stop",
        "revision": fixture["release_sha"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "key_bootstrap_receipt_path": (
            "/var/lib/muncho-capability-canary-control/"
            f"bitrix-key-bootstraps/{key_contract['public_key_id']}/bootstrap.json"
        ),
        "key_bootstrap_receipt_sha256": key_contract[
            "key_bootstrap_receipt_sha256"
        ],
        "retirement_intent_path": (
            f"{key_retirement_root}/service_stop-intent.json"
        ),
        "retirement_intent_sha256": _sha("bitrix-key-retirement-intent"),
        "public_key_id": key_contract["public_key_id"],
        "private_path": key_contract["private_source_path"],
        "public_path": key_contract["public_path"],
        "private_absent": True,
        "public_absent": True,
        "both_pair_members_absent": True,
        "service_stop_proof_sha256": service_stop_proof_sha256,
        "retired_at_unix": NOW // 1000,
        "private_content_or_digest_recorded": False,
        "receipt_path": f"{key_retirement_root}/service_stop-completion.json",
    }
    key_retirement = {
        **key_retirement_unsigned,
        "receipt_sha256": canary._digest(key_retirement_unsigned),
    }

    cleanup = sign(
        "gateway_observer",
        {
            **_common(canary.CLEANUP_RECEIPT_SCHEMA, fixture, fixture_sha256, 90),
            "non_observer_service_units": list(
                canary.NON_OBSERVER_SERVICE_UNITS
            ),
            "non_observer_services_stopped": True,
            "non_observer_services_state_sha256": credential_stop_proof[
                "non_observer_services_state_sha256"
            ],
            "gateway_observer_signer_identity": observer_identity,
            "credential_consumer_stop_proof": credential_stop_proof,
            "credential_leases": list(canary.CREDENTIAL_LEASES),
            "credential_leases_retired": True,
            "retirements": retirements,
            "retirement_receipt_sha256s": {
                binding: receipt["receipt_sha256"]
                for binding, receipt in retirements.items()
            },
            "credential_absence": credential_absence,
            "credentials_absent": True,
            "bitrix_receipt_key_retirement": key_retirement,
            "bitrix_receipt_key_absence": {
                "private_path": key_contract["private_source_path"],
                "private_absent": True,
                "public_path": key_contract["public_path"],
                "public_absent": True,
                "both_pair_members_absent": True,
            },
            "discord_credential_topology": dict(
                canary.DISCORD_CREDENTIAL_TOPOLOGY
            ),
            "browser_session_retired": True,
            "isolated_worker_lease_cleanup_verified": True,
            "production_diff_sha256": production_diff["diff_sha256"],
        },
    )

    producer_owner_authority, producer_activation = (
        _producer_authority_and_activation(
            fixture=fixture,
            fixture_sha256=fixture_sha256,
            keys=keys,
            owner_grant=owner_grant,
        )
    )
    producer_readiness_sha256 = producer_activation["readiness_sha256"]
    non_owner_receipts = [
        runtime,
        workspace["gateway_receipt"],
        workspace["writer_receipt"],
        denials,
        database,
        bitrix["edge_receipt"],
        bitrix["writer_receipt"],
        discord["edge_receipt"],
        discord["writer_receipt"],
        failures["gateway_receipt"],
        failures["writer_receipt"],
        cleanup,
    ]
    for receipt in non_owner_receipts:
        receipt["native_evidence"]["producer_readiness_sha256"] = (
            producer_readiness_sha256
        )
        _resign_exact_envelope(receipt, keys=keys)

    observer_stop_unsigned = {
        "schema": canary.OBSERVER_STOP_RECEIPT_SCHEMA,
        "plan_sha256": fixture["capability_plan_sha256"],
        "service_unit": canary.OBSERVER_PRODUCER_SERVICE_UNIT,
        "service_state_sha256": _sha("observer-stopped-service-state"),
        "stopped": True,
        "stopped_at_unix_ms": NOW + 91,
        "secret_material_recorded": False,
    }
    observer_stop = {
        **observer_stop_unsigned,
        "receipt_sha256": canary._digest(observer_stop_unsigned),
    }
    final_stop_unsigned = {
        "schema": canary.CAPABILITY_SERVICE_STOP_PROOF_SCHEMA,
        "plan_sha256": fixture["capability_plan_sha256"],
        "stop_order": list(canary.SERVICE_UNITS),
        "services_state_sha256": _sha("all-services-stopped-state"),
        "all_services_stopped": True,
        "observed_at_unix": NOW // 1000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    final_stop_proof = {
        **final_stop_unsigned,
        "stop_proof_sha256": canary._digest(final_stop_unsigned),
    }
    fleet_retirement_unsigned = {
        "schema": canary.PRODUCER_FLEET_RETIREMENT_SCHEMA,
        "readiness_sha256": producer_readiness_sha256,
        "foundation_sha256": fixture["producer_foundation_sha256"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_id": fixture["run_id"],
        "path": canary.PRODUCER_ACTIVATION_PATH,
        "retirement_intent_sha256": _sha("producer-retirement-intent"),
        "retired": True,
        "absence_verified": True,
        "retired_at_unix_ms": NOW + 92,
    }
    fleet_retirement = {
        **fleet_retirement_unsigned,
        "receipt_sha256": canary._digest(fleet_retirement_unsigned),
    }
    admission_retirement_unsigned = {
        "schema": canary.API_ADMISSION_RETIREMENT_SCHEMA,
        "run_id": fixture["run_id"],
        "fixture_sha256": fixture_sha256,
        "session_id": f"capability_{fixture['run_id']}",
        "capability_epoch_sha256": _sha("capability-epoch"),
        "challenge_sha256": _sha("api-admission-challenge"),
        "owner_authority_path": (
            f"/var/lib/muncho-capability-canary-evidence/{fixture['run_id']}/"
            "api-admission-owner-authority.json"
        ),
        "owner_authority_sha256": canary._digest(producer_owner_authority),
        "install_publication_sha256": _sha("api-admission-install"),
        "intent_sha256": _sha("api-admission-retirement-intent"),
        "catalog_sha256": producer_owner_authority["catalog"]["catalog_sha256"],
        "owner_grant_sha256": producer_activation["owner_authority"][
            "grant_sha256"
        ],
        "catalog_absent": True,
        "owner_grant_absent": True,
        "retired_at_unix_ms": NOW + 93,
    }
    admission_retirement = {
        **admission_retirement_unsigned,
        "receipt_sha256": canary._digest(admission_retirement_unsigned),
    }
    watchdog_unsigned = {
        "operation": "normal_lifecycle_disarm",
        "watchdog_id": "f" * 32,
        "watchdog_authority_sha256": _sha("watchdog-authority"),
        "disarm_intent_path": (
            "/var/lib/muncho-capability-canary-control/watchdog-disarm-intent.json"
        ),
        "disarm_intent_sha256": _sha("watchdog-disarm-intent"),
        "timer_name": "muncho-capability-expiry-f.timer",
        "timer_disabled": True,
        "timer_wants_absent": True,
        "service_absent": True,
        "timer_absent": True,
        "completed_at_unix": (NOW + 94) // 1000,
        "ok": True,
        "schema": canary.CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA,
        "receipt_path": (
            "/var/lib/muncho-capability-canary-control/"
            "watchdog-disarm-completion.json"
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    watchdog_completion = {
        **watchdog_unsigned,
        "receipt_sha256": canary._digest(watchdog_unsigned),
    }
    watchdog_retirement = {
        "watchdog_count": 1,
        "retired": [watchdog_completion],
        "all_timers_disabled": True,
        "all_unit_files_absent": True,
    }
    finalization_unsigned = {
        "schema": canary.CLEANUP_FINALIZATION_SCHEMA,
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_id": fixture["run_id"],
        "cleanup_receipt_sha256": canary._digest(cleanup),
        "observer_stop_receipt": observer_stop,
        "service_stop_proof": final_stop_proof,
        "producer_fleet_retirement": fleet_retirement,
        "admission_input_retirement": admission_retirement,
        "expiry_watchdog_retirement": watchdog_retirement,
        "producer_activation_absent": True,
        "admission_inputs_absent": True,
        "credentials_absent": True,
        "bitrix_receipt_key_pair_absent": True,
        "full_canary_stopped_preflight_sha256": _sha(
            "full-canary-stopped-preflight"
        ),
        "finalized_at_unix_ms": NOW + 95,
    }
    cleanup_finalization = {
        **finalization_unsigned,
        "finalization_sha256": canary._digest(finalization_unsigned),
    }

    return {
        "schema": canary.EVIDENCE_SCHEMA,
        "execution_mode": "live_production_shaped_canary",
        "synthetic": False,
        "fixture_sha256": fixture_sha256,
        "release_sha": fixture["release_sha"],
        "release_artifact_sha256": fixture["release_artifact_sha256"],
        "installed_wheel_manifest_sha256": fixture["installed_wheel_manifest_sha256"],
        "producer_readiness_sha256": producer_readiness_sha256,
        "producer_activation": producer_activation,
        "producer_owner_authority": producer_owner_authority,
        "run_id": fixture["run_id"],
        "started_at_unix_ms": NOW + 5,
        "api_started_at_unix_ms": NOW + 15,
        "api_completed_at_unix_ms": NOW + 80,
        "completed_at_unix_ms": NOW + 100,
        "runtime_receipt": runtime,
        "bundles": {
            "workspace_continuation": workspace,
            "capability_denials": denials,
            "database_reconciliation": database,
            "bitrix_boundary": bitrix,
            "discord_routeback": discord,
            "failure_recovery": failures,
        },
        "cleanup_receipt": cleanup,
        "cleanup_finalization": cleanup_finalization,
    }


@pytest.fixture
def valid_bundle():
    keys = _private_keys()
    fixture = _fixture(keys)
    fixture_sha256 = canary._digest(fixture)
    evidence = _evidence(fixture, fixture_sha256, keys)
    return keys, fixture, evidence


def build_signed_production_prerequisite_evidence_for_test(
    revision: str = "1" * 40,
) -> tuple[
    dict, str, dict, dict, dict
]:
    """Return signed goal/cleanup envelopes and their exact native diff."""

    keys = _private_keys()
    fixture = _fixture(keys, revision=revision)
    fixture_sha256 = canary._digest(fixture)
    evidence = _evidence(fixture, fixture_sha256, keys)
    workspace_gateway_envelope = copy.deepcopy(
        evidence["bundles"]["workspace_continuation"]["gateway_receipt"]
    )
    cleanup_envelope = copy.deepcopy(evidence["cleanup_receipt"])
    production_diff = _production_diff(fixture, fixture_sha256)
    return (
        fixture,
        fixture_sha256,
        workspace_gateway_envelope,
        cleanup_envelope,
        production_diff,
    )


def build_signed_goal_continuation_workspace_gateway_for_test(
    revision: str = "1" * 40,
) -> tuple[dict, str, dict]:
    """Return a complete fixture and its real signed workspace envelope."""

    fixture, fixture_sha256, workspace, _cleanup, _diff = (
        build_signed_production_prerequisite_evidence_for_test(revision)
    )
    return fixture, fixture_sha256, workspace


def _verify(fixture: dict, evidence: dict) -> dict:
    return canary.verify_capability_canary(
        fixture,
        evidence,
        fixture_sha256=canary._digest(fixture),
        evidence_sha256=canary._digest(evidence),
    )


def test_valid_signed_six_bundle_evidence_verifies(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    result = _verify(fixture, evidence)
    assert result["ok"] is True
    assert result["schema"] == canary.VERIFICATION_SCHEMA
    assert result["invariants"] == list(canary.INVARIANTS)
    assert result["verification_receipt_sha256"] == canary._digest({
        key: value
        for key, value in result.items()
        if key != "verification_receipt_sha256"
    })


def test_legacy_capability_evidence_v1_is_rejected(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    evidence["schema"] = "muncho-production-capability-canary-evidence.v1"

    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)

    assert raised.value.code == "evidence_invalid"


def test_legacy_discord_writer_v1_is_rejected_even_when_resigned(valid_bundle):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["discord_routeback"]["writer_receipt"]
    writer["payload"]["schema"] = (
        "muncho-production-capability-discord-writer.v1"
    )
    _resign(writer, fixture=fixture, keys=keys)

    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)

    assert raised.value.code == "discord_bundle_invalid"


def test_tampered_signed_receipt_is_rejected(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    evidence["bundles"]["database_reconciliation"]["payload"]["readback_verified"] = (
        False
    )
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "database_bundle_invalid"


def test_non_owner_receipt_requires_exact_native_evidence_kinds(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["runtime_receipt"]
    receipt["native_evidence"]["bindings"][0]["kind"] = "claimed_success"
    _resign_exact_envelope(receipt, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "runtime_receipt_invalid"


def test_all_non_owner_receipts_bind_one_final_producer_activation(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["capability_denials"]
    receipt["native_evidence"]["producer_readiness_sha256"] = _sha(
        "different-activation"
    )
    _resign_exact_envelope(receipt, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "evidence_invalid"


def test_owner_pregrant_remains_distinct_sshsig_without_native_envelope(
    valid_bundle,
):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["workspace_continuation"][
        "owner_approval_receipt"
    ]
    receipt["native_evidence"] = {
        "schema": canary.NATIVE_EVIDENCE_SCHEMA,
        "producer_readiness_sha256": PRODUCER_READINESS_SHA256,
        "bindings": [],
    }
    _resign_exact_envelope(receipt, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda evidence: evidence.__setitem__("synthetic", True), "evidence_invalid"),
        (
            lambda evidence: evidence["runtime_receipt"]["payload"].__setitem__(
                "kanban_dispatch_in_gateway", True
            ),
            "runtime_receipt_invalid",
        ),
        (
            lambda evidence: evidence["bundles"]["workspace_continuation"][
                "gateway_receipt"
            ]["payload"].__setitem__("microapproval_prompt_count", 1),
            "task_workspace_bundle_invalid",
        ),
        (
            lambda evidence: evidence["bundles"]["discord_routeback"]["writer_receipt"][
                "payload"
            ].__setitem__("sent_after_verified_readback", False),
            "discord_bundle_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"].__setitem__(
                "credential_leases_retired", False
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "non_observer_service_units"
            ].remove(
                "muncho-capability-producer-business-edge.service"
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "non_observer_service_units"
            ].append(canary.OBSERVER_PRODUCER_SERVICE_UNIT),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "gateway_observer_signer_identity"
            ].__setitem__("live", False),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "credential_consumer_stop_proof"
            ].__setitem__("observer_credential_read_access", True),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "retirements"
            ]["bitrix_operational_edge_webhook"].__setitem__(
                "absent_after_stop", False
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "retirements"
            ]["api_control"].__setitem__(
                "service_stop_proof_sha256", _sha("unbound-consumer-stop")
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "retirements"
            ]["api_control"].__setitem__(
                "install_receipt_path", "/tmp/unbound-install-receipt.json"
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"].__setitem__(
                "credentials_absent", False
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "bitrix_receipt_key_retirement"
            ].__setitem__("public_absent", False),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "bitrix_receipt_key_absence"
            ].__setitem__("public_absent", False),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "bitrix_receipt_key_retirement"
            ].__setitem__(
                "service_stop_proof_sha256", _sha("unbound-stop-proof")
            ),
            "cleanup_receipt_invalid",
        ),
        (
            lambda evidence: evidence["cleanup_receipt"]["payload"][
                "discord_credential_topology"
            ].__setitem__("routeback_discord_transport", "gateway_websocket"),
            "cleanup_receipt_invalid",
        ),
    ],
)
def test_signed_but_invalid_invariants_are_rejected(valid_bundle, mutate, code):
    keys, fixture, evidence = valid_bundle
    mutate(evidence)
    for receipt in _all_receipts(evidence):
        _resign(receipt, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == code


def _all_receipts(evidence: dict) -> list[dict]:
    receipts = [evidence["runtime_receipt"], evidence["cleanup_receipt"]]
    for bundle in evidence["bundles"].values():
        if bundle.get("schema") == canary.SIGNED_RECEIPT_SCHEMA:
            receipts.append(bundle)
        else:
            receipts.extend(bundle.values())
    return receipts


def test_cleanup_production_diff_must_bind_native_observation(valid_bundle):
    keys, fixture, evidence = valid_bundle
    cleanup = evidence["cleanup_receipt"]
    binding = next(
        item
        for item in cleanup["native_evidence"]["bindings"]
        if item["kind"] == "production_diff_observation"
    )
    binding["artifact_sha256"] = _sha("unrelated-production-diff")
    _resign_exact_envelope(cleanup, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "cleanup_receipt_invalid"


def _redigest_cleanup_finalization(evidence: dict) -> None:
    finalization = evidence["cleanup_finalization"]
    observer_stop = finalization["observer_stop_receipt"]
    observer_stop["receipt_sha256"] = canary._digest({
        key: value
        for key, value in observer_stop.items()
        if key != "receipt_sha256"
    })
    stop_proof = finalization["service_stop_proof"]
    stop_proof["stop_proof_sha256"] = canary._digest({
        key: value
        for key, value in stop_proof.items()
        if key != "stop_proof_sha256"
    })
    retirement = finalization["producer_fleet_retirement"]
    retirement["receipt_sha256"] = canary._digest({
        key: value
        for key, value in retirement.items()
        if key != "receipt_sha256"
    })
    admission = finalization["admission_input_retirement"]
    admission["receipt_sha256"] = canary._digest({
        key: value
        for key, value in admission.items()
        if key != "receipt_sha256"
    })
    for completion in finalization["expiry_watchdog_retirement"]["retired"]:
        completion["receipt_sha256"] = canary._digest({
            key: value
            for key, value in completion.items()
            if key != "receipt_sha256"
        })
    finalization["finalization_sha256"] = canary._digest({
        key: value
        for key, value in finalization.items()
        if key != "finalization_sha256"
    })


@pytest.mark.parametrize(
    "mutation",
    (
        "cleanup_digest",
        "observer_not_stopped",
        "observer_not_last",
        "activation_present",
        "admission_inputs_present",
        "admission_retirement_not_absent",
        "watchdog_timer_not_absent",
        "retirement_not_absent",
        "retirement_wrong_activation",
        "finalization_predates_cleanup",
    ),
)
def test_cleanup_finalization_rejects_impossible_or_unbound_truths(
    valid_bundle, mutation
):
    _keys, fixture, evidence = valid_bundle
    finalization = evidence["cleanup_finalization"]
    if mutation == "cleanup_digest":
        finalization["cleanup_receipt_sha256"] = _sha("other-cleanup")
    elif mutation == "observer_not_stopped":
        finalization["observer_stop_receipt"]["stopped"] = False
    elif mutation == "observer_not_last":
        stop_order = finalization["service_stop_proof"]["stop_order"]
        stop_order[0], stop_order[-1] = stop_order[-1], stop_order[0]
    elif mutation == "activation_present":
        finalization["producer_activation_absent"] = False
    elif mutation == "admission_inputs_present":
        finalization["admission_inputs_absent"] = False
    elif mutation == "admission_retirement_not_absent":
        finalization["admission_input_retirement"]["catalog_absent"] = False
    elif mutation == "watchdog_timer_not_absent":
        finalization["expiry_watchdog_retirement"]["retired"][0][
            "timer_absent"
        ] = False
    elif mutation == "retirement_not_absent":
        finalization["producer_fleet_retirement"][
            "absence_verified"
        ] = False
    elif mutation == "retirement_wrong_activation":
        finalization["producer_fleet_retirement"][
            "readiness_sha256"
        ] = _sha("other-activation")
    elif mutation == "finalization_predates_cleanup":
        finalization["finalized_at_unix_ms"] = NOW + 89
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(mutation)
    _redigest_cleanup_finalization(evidence)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "cleanup_finalization_invalid"


def test_nonterminal_task_workspace_is_rejected_even_when_resigned(valid_bundle):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["workspace_continuation"]["writer_receipt"]
    writer["payload"]["terminal_ctw"] = _terminal("workspace", state="blocked")
    _resign(writer, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


def test_workspace_requires_model_authored_max_effort_on_later_request(valid_bundle):
    keys, fixture, evidence = valid_bundle
    gateway = evidence["bundles"]["workspace_continuation"]["gateway_receipt"]
    gateway["payload"]["later_request_effort"] = "high"
    _resign(gateway, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


def test_workspace_replayed_mutation_is_rejected(valid_bundle):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["workspace_continuation"]["writer_receipt"]
    writer["payload"]["terminal_ctw"]["replayed_mutation_count"] = 1
    _resign(writer, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


def test_discord_dm_must_be_denied_before_dispatch(valid_bundle):
    keys, fixture, evidence = valid_bundle
    edge = evidence["bundles"]["discord_routeback"]["edge_receipt"]
    edge["payload"]["private_dispatch_attempted"] = True
    _resign(edge, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "discord_bundle_invalid"


@pytest.mark.parametrize(
    ("event_name", "receipt_name", "field", "replacement"),
    [
        ("sent_event", "returned_receipt", "public_receipt_sha256", "e" * 64),
        ("sent_event", "payload_receipt", "message_id", "1504852355588423998"),
        ("sent_event", "route_back_receipt", "public_receipt_sha256", "e" * 64),
        (
            "blocked_event",
            "returned_receipt",
            "private_denial_receipt_sha256",
            "e" * 64,
        ),
        (
            "blocked_event",
            "payload_receipt",
            "private_denial_receipt_sha256",
            "e" * 64,
        ),
        (
            "blocked_event",
            "route_back_receipt",
            "private_denial_receipt_sha256",
            "e" * 64,
        ),
        ("blocked_event", "returned_receipt", "dispatch_attempted", True),
        ("blocked_event", "payload_receipt", "dispatch_attempted", True),
        ("blocked_event", "route_back_receipt", "dispatch_attempted", True),
    ],
)
def test_discord_writer_receipt_substitution_fails_at_every_bound_level(
    valid_bundle,
    event_name,
    receipt_name,
    field,
    replacement,
):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["discord_routeback"]["writer_receipt"]
    writer["payload"][event_name][receipt_name][field] = replacement
    _resign(writer, fixture=fixture, keys=keys)

    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)

    assert raised.value.code == "discord_bundle_invalid"


@pytest.mark.parametrize(
    "field",
    [
        "payload_dispatch_attempted",
        "route_back_dispatch_attempted",
    ],
)
def test_discord_writer_blocked_event_dispatch_flags_are_exact(
    valid_bundle,
    field,
):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["discord_routeback"]["writer_receipt"]
    writer["payload"]["blocked_event"][field] = True
    _resign(writer, fixture=fixture, keys=keys)

    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)

    assert raised.value.code == "discord_bundle_invalid"


def test_failure_bundle_accepts_honestly_blocked_terminal_ctw(valid_bundle):
    keys, fixture, evidence = valid_bundle
    writer = evidence["bundles"]["failure_recovery"]["writer_receipt"]
    writer["payload"]["terminal_ctw"] = _terminal("failure", state="blocked")
    _resign(writer, fixture=fixture, keys=keys)
    assert _verify(fixture, evidence)["ok"] is True


def test_unavailable_alternative_must_not_claim_attempt_or_receipt(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["failure_recovery"]["gateway_receipt"]
    database = receipt["payload"]["failures"][2]
    database["alternative_available"] = False
    database["alternative_attempted"] = True
    _resign(receipt, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "failure_bundle_invalid"


def test_unknown_or_secret_bearing_field_is_rejected(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["bitrix_boundary"]["edge_receipt"]
    receipt["payload"]["session_cookie"] = "must-not-be-accepted"
    _resign(receipt, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "bitrix_bundle_invalid"


def test_missing_execution_bundle_is_rejected(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    evidence["bundles"].pop("bitrix_boundary")
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "evidence_invalid"


def test_wrong_owner_key_cannot_sign_plan_approval(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["workspace_continuation"]["owner_approval_receipt"]
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    receipt["signature"] = _sshsig_signature(
        keys["gateway_observer"], canary._canonical_bytes(unsigned)
    )
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


def test_owner_grant_is_honestly_prestaged_before_api_execution(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    owner = evidence["bundles"]["workspace_continuation"][
        "owner_approval_receipt"
    ]
    assert owner["payload"]["observed_at_unix_ms"] <= evidence[
        "api_started_at_unix_ms"
    ]
    assert _verify(fixture, evidence)["ok"] is True


def test_owner_grant_created_after_mutation_started_is_rejected(valid_bundle):
    keys, fixture, evidence = valid_bundle
    owner = evidence["bundles"]["workspace_continuation"][
        "owner_approval_receipt"
    ]
    owner["payload"]["observed_at_unix_ms"] = NOW + 22
    _resign(owner, fixture=fixture, keys=keys)
    grant_sha256 = canary._digest(owner)
    for receipt_name in ("gateway_receipt", "writer_receipt"):
        receipt = evidence["bundles"]["workspace_continuation"][receipt_name]
        receipt["payload"]["owner_grant_sha256"] = grant_sha256
        _resign(receipt, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "evidence_invalid"


def test_owner_sshsig_wrong_namespace_is_rejected(valid_bundle):
    keys, fixture, evidence = valid_bundle
    receipt = evidence["bundles"]["workspace_continuation"]["owner_approval_receipt"]
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    receipt["signature"] = _sshsig_signature(
        keys["owner"],
        canary._canonical_bytes(unsigned),
        namespace_text="unrelated-owner-purpose",
    )
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "task_workspace_bundle_invalid"


def test_bitrix_requires_operational_edge_read_not_mac_ops_proxy(valid_bundle):
    keys, fixture, evidence = valid_bundle
    outer = evidence["bundles"]["bitrix_boundary"]["edge_receipt"]
    assert "mac_ops" not in json.dumps(outer["payload"])
    assert "gitlab" not in json.dumps(outer["payload"])
    outer["payload"]["operation_id"] = "task.read"
    _resign(outer, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "bitrix_bundle_invalid"


def test_authority_keys_must_be_role_separated(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    fixture["authority_keys"]["business_edge"]["public_key_ed25519_hex"] = fixture[
        "authority_keys"
    ]["canonical_writer"]["public_key_ed25519_hex"]
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "fixture_invalid"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("connector_bot_user_id", "routeback_bot_user_id"),
        ("connector_bot_user_id", "production_bot_user_id"),
        ("routeback_bot_user_id", "production_bot_user_id"),
    ],
)
def test_discord_bot_identities_must_be_pairwise_distinct(
    valid_bundle, left, right
):
    _keys, fixture, evidence = valid_bundle
    fixture["discord_bot_identities"][left] = fixture["discord_bot_identities"][right]
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "fixture_invalid"


@pytest.mark.parametrize(
    "channel_id", sorted(canary.LOCKED_PRIVATE_DISCORD_CHANNEL_IDS)
)
def test_locked_private_discord_targets_are_rejected(valid_bundle, channel_id):
    _keys, fixture, evidence = valid_bundle
    fixture["public_discord_target"]["channel_id"] = channel_id
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "fixture_invalid"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("production_bot_user_id", "1526999999999999991"),
        ("connector_bot_user_id", "1526999999999999992"),
        ("routeback_bot_user_id", "1526999999999999993"),
    ],
)
def test_discord_bot_identities_are_exactly_staged(
    valid_bundle, field, replacement
):
    _keys, fixture, evidence = valid_bundle
    fixture["discord_bot_identities"][field] = replacement
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == "fixture_invalid"


@pytest.mark.parametrize(
    ("receipt_path", "field", "replacement", "code"),
    [
        (
            ("runtime_receipt",),
            "connector_bot_user_id_provenance",
            "environment_hint",
            "runtime_receipt_invalid",
        ),
        (
            ("bundles", "discord_routeback", "edge_receipt"),
            "routeback_bot_user_id_provenance",
            "configured_application_id",
            "discord_bundle_invalid",
        ),
    ],
)
def test_discord_bot_identity_requires_live_provenance(
    valid_bundle, receipt_path, field, replacement, code
):
    keys, fixture, evidence = valid_bundle
    receipt = evidence
    for component in receipt_path:
        receipt = receipt[component]
    receipt["payload"][field] = replacement
    _resign(receipt, fixture=fixture, keys=keys)
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        _verify(fixture, evidence)
    assert raised.value.code == code


def test_artifact_verifier_rejects_wrong_digest_without_echoing_content(
    valid_bundle, tmp_path: Path
):
    _keys, fixture, evidence = valid_bundle
    fixture_path = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    fixture_path.write_bytes(canary._canonical_bytes(fixture))
    evidence_path.write_bytes(canary._canonical_bytes(evidence))

    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        canary.verify_files(
            fixture_path=fixture_path,
            fixture_sha256="0" * 64,
            evidence_path=evidence_path,
            evidence_sha256=canary._digest(evidence),
        )
    assert raised.value.code == "fixture_artifact_invalid"
    assert "must-not" not in str(raised.value)


def test_cli_emits_only_stable_failure_code(valid_bundle, tmp_path: Path, capsys):
    _keys, fixture, evidence = valid_bundle
    fixture_path = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    fixture_path.write_bytes(canary._canonical_bytes(fixture))
    evidence_path.write_bytes(canary._canonical_bytes(evidence))

    result = canary.main([
        "verify",
        "--fixture",
        str(fixture_path),
        "--fixture-sha256",
        canary._digest(fixture),
        "--evidence",
        str(evidence_path),
        "--evidence-sha256",
        "0" * 64,
    ])
    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output == {
        "schema": canary.VERIFICATION_SCHEMA,
        "ok": False,
        "failure_code": "evidence_artifact_invalid",
    }


def test_cli_verifies_canonical_files(valid_bundle, tmp_path: Path, capsys):
    _keys, fixture, evidence = valid_bundle
    fixture_path = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    fixture_path.write_bytes(canary._canonical_bytes(fixture))
    evidence_path.write_bytes(canary._canonical_bytes(evidence))

    result = canary.main([
        "verify",
        "--fixture",
        str(fixture_path),
        "--fixture-sha256",
        canary._digest(fixture),
        "--evidence",
        str(evidence_path),
        "--evidence-sha256",
        canary._digest(evidence),
    ])
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["ok"] is True
    assert output["schema"] == canary.VERIFICATION_SCHEMA


def test_caller_supplied_noncanonical_digest_is_rejected(valid_bundle):
    _keys, fixture, evidence = valid_bundle
    with pytest.raises(canary.CapabilityCanaryEvidenceError) as raised:
        canary.verify_capability_canary(
            copy.deepcopy(fixture),
            copy.deepcopy(evidence),
            fixture_sha256="f" * 64,
            evidence_sha256=canary._digest(evidence),
        )
    assert raised.value.code == "fixture_digest_noncanonical"
