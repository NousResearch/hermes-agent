"""Focused tests for production-shaped capability-canary packaging."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import stat
import time
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import gateway.canonical_capability_canary_runtime as runtime
import gateway.canonical_capability_canary_producer_units as producer_units
from gateway import canonical_capability_canary_producers as producers
from gateway.canonical_full_canary_runtime import (
    FullCanaryIdentities,
    FullCanaryPlan,
)
from plugins.muncho_canary_evidence import (
    CanaryEvidenceHookMultiplexer,
    CanaryEvidencePlugin,
    GoalContinuationEvidencePlugin,
)


def _full_plan() -> FullCanaryPlan:
    plan = object.__new__(FullCanaryPlan)
    identities = FullCanaryIdentities(
        writer_user="muncho_writer",
        writer_group="muncho_writer",
        writer_uid=2101,
        writer_gid=2201,
        gateway_user="hermes_gateway",
        gateway_group="hermes_gateway",
        gateway_uid=max(os.getuid(), 1),
        gateway_gid=max(os.getgid(), 1),
        socket_client_group="muncho_writer_clients",
        socket_client_gid=2203,
        edge_user="muncho-discord-egress",
        edge_group="muncho-discord-egress",
        edge_uid=2103,
        edge_gid=2204,
    )
    for name, value in {
        "revision": "a" * 40,
        "sha256": "b" * 64,
        "release": {
            "artifact_root": "/opt/muncho-canary-releases/" + "a" * 40,
            "artifact_sha256": "c" * 64,
            "interpreter": "/opt/muncho-canary-releases/"
            + "a" * 40
            + "/venv/bin/python",
        },
        "identities": identities,
    }.items():
        object.__setattr__(plan, name, value)
    return plan


def _full_canary_terminal_receipt(
    full_plan: FullCanaryPlan | None = None,
) -> dict[str, object]:
    full = _full_plan() if full_plan is None else full_plan
    unsigned = {
        "schema": runtime.FULL_CANARY_TERMINAL_RECEIPT_SCHEMA,
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": full.revision,
        "coordinator_input_sha256": "0" * 64,
        "full_canary_plan_sha256": full.sha256,
        "owner_approval_sha256": "1" * 64,
        "phase_b_readiness_anchor_sha256": "2" * 64,
        "api_session_key_sha256": "3" * 64,
        "fixture_sha256": "4" * 64,
        "discord_token_install_receipt_sha256": "5" * 64,
        "coordinator_receipt_sha256": "6" * 64,
        "live_driver_receipt_sha256": "7" * 64,
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": 1_700_000_000,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _plan(
    full_plan: FullCanaryPlan | None = None,
    *,
    runtime_dependency_manifest_sha256: str = "4" * 64,
) -> runtime.CapabilityCanaryPlan:
    full = _full_plan() if full_plan is None else full_plan
    terminal = _full_canary_terminal_receipt(full)
    return runtime.build_capability_plan(
        full_plan=full,
        full_canary_terminal_receipt=terminal,
        full_canary_terminal_receipt_sha256=terminal["receipt_sha256"],
        mac_ops_uid=2104,
        mac_ops_gid=2205,
        connector_uid=2105,
        connector_gid=2206,
        bitrix_operational_edge_uid=2108,
        bitrix_operational_edge_gid=2210,
        bitrix_operational_edge_client_gid=2211,
        browser_uid=2106,
        browser_gid=2207,
        worker_uid=2107,
        worker_gid=2208,
        worker_client_gid=2209,
        connector_bot_user_id="1600000000000000001",
        routeback_bot_user_id="1600000000000000002",
        connector_allowed_guild_ids=("1282725267068157972",),
        connector_allowed_channel_ids=(runtime.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID,),
        connector_allowed_user_ids=("1279454038731264061",),
        browser_node_sha256="5" * 64,
        browser_wrapper_sha256="6" * 64,
        browser_native_sha256="7" * 64,
        browser_executable_sha256="3" * 64,
        agent_browser_config_sha256="8" * 64,
        worker_bwrap_sha256="9" * 64,
        worker_shell_sha256="a" * 64,
        runtime_dependency_manifest_sha256=runtime_dependency_manifest_sha256,
        bitrix_operational_edge_asset_manifest_sha256="b" * 64,
        bitrix_operational_edge_rendered_unit_sha256="c" * 64,
        bitrix_operational_edge_rendered_config_sha256="d" * 64,
        bitrix_operational_edge_rendered_trust_sha256="e" * 64,
        bitrix_operational_edge_identity_bootstrap_receipt_sha256="f" * 64,
        bitrix_operational_edge_receipt_public_key_id="1" * 64,
        bitrix_operational_edge_key_bootstrap_receipt_sha256="2" * 64,
    )


def _service_identity_observation(
    plan: runtime.CapabilityCanaryPlan,
    *,
    role: str,
    state: str,
) -> dict[str, object]:
    if role == "mac_ops":
        user = plan.identities.mac_ops_user
        group = plan.identities.mac_ops_group
        uid = plan.identities.mac_ops_uid
        gid = plan.identities.mac_ops_gid
    else:
        user = plan.identities.connector_user
        group = plan.identities.connector_group
        uid = plan.identities.connector_uid
        gid = plan.identities.connector_gid
    unsigned = {
        "schema": runtime.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "role": role,
        "state": state,
        "user": user,
        "group": group,
        "uid": uid,
        "gid": gid,
        "home": "/nonexistent",
        "shell": "/usr/sbin/nologin",
        "group_members": (None if state == "absent_create_only_slot" else []),
        "supplementary_group_ids": ([gid] if state == "present_exact" else None),
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _producer_identity_observation(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    identities = producer_units.planned_producer_role_identities()
    unsigned = {
        "schema": producer_units.PRODUCER_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "roles": {
            role: {
                "state": "present_exact",
                "user": item["user"],
                "group": item["group"],
                "uid": item["uid"],
                "gid": item["gid"],
                "home": "/nonexistent",
                "shell": "/usr/sbin/nologin",
                "group_members": [],
                "supplementary_group_ids": [item["gid"]],
                "create_only_eligible": True,
            }
            for role, item in identities.items()
        },
        "receipt_writer_group": {
            "state": "present_exact",
            "group": producer_units.PRODUCER_RECEIPT_WRITER_GROUP,
            "gid": producer_units.PRODUCER_RECEIPT_WRITER_GID,
            "members": [],
            "create_only_eligible": True,
        },
        "planned_identities": identities,
        "persistent_supplementary_memberships": False,
        "service_time_supplementary_groups_only": True,
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _production_observation_wait_request(
    plan: runtime.CapabilityCanaryPlan,
    *,
    phase: str = "before",
) -> dict[str, object]:
    return {
        "schema": runtime.CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA,
        "phase": phase,
        "canary_revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": "c" * 64,
        "run_id": "capability-run-observed",
        "owner_subject_sha256": "d" * 64,
        "timeout_seconds": 30,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _stop_proof(
    plan: runtime.CapabilityCanaryPlan,
    *,
    observed_at_unix: int | None = None,
) -> dict[str, object]:
    stopped = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": 0,
        "UnitFileState": "disabled",
        "DropInPaths": "",
    }
    return dict(
        runtime.build_capability_stop_proof(
            plan,
            {unit: dict(stopped) for unit in runtime.CAPABILITY_STOP_ORDER},
            observed_at_unix=(
                int(time.time()) if observed_at_unix is None else observed_at_unix
            ),
        )
    )


def _routeback_file_metadata(*, inode: int = 41) -> dict[str, int]:
    return {
        "device": 7,
        "inode": inode,
        "uid": _full_plan().identities.edge_uid,
        "gid": _full_plan().identities.edge_gid,
        "mode": 0o400,
        "size": 96,
        "mtime_ns": 1_700_000_000_000_000_001,
        "ctime_ns": 1_700_000_000_000_000_002,
    }


def _plan_publication_inputs(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    unsigned = {
        "schema": runtime.CAPABILITY_PLAN_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "identities": dict(runtime.CAPABILITY_PLANNED_IDENTITIES),
        "discord": {
            "connector_bot_user_id": plan.connector_bot_user_id,
            "routeback_bot_user_id": plan.routeback_bot_user_id,
            "allowed_guild_ids": list(plan.connector_allowed_guild_ids),
            "allowed_channel_ids": list(plan.connector_allowed_channel_ids),
            "allowed_user_ids": list(plan.connector_allowed_user_ids),
        },
        "artifacts": {
            "browser_node_sha256": plan.browser_node_sha256,
            "browser_wrapper_sha256": plan.browser_wrapper_sha256,
            "browser_native_sha256": plan.browser_native_sha256,
            "browser_executable_sha256": plan.browser_executable_sha256,
            "agent_browser_config_sha256": plan.agent_browser_config_sha256,
            "worker_bwrap_sha256": plan.worker_bwrap_sha256,
            "worker_shell_sha256": plan.worker_shell_sha256,
            "runtime_dependency_manifest_sha256": (
                plan.runtime_dependency_manifest_sha256
            ),
            "bitrix_operational_edge_asset_manifest_sha256": (
                plan.bitrix_operational_edge_asset_manifest_sha256
            ),
            "bitrix_operational_edge_rendered_unit_sha256": (
                plan.bitrix_operational_edge_rendered_unit_sha256
            ),
            "bitrix_operational_edge_rendered_config_sha256": (
                plan.bitrix_operational_edge_rendered_config_sha256
            ),
            "bitrix_operational_edge_rendered_trust_sha256": (
                plan.bitrix_operational_edge_rendered_trust_sha256
            ),
            "bitrix_operational_edge_identity_bootstrap_receipt_sha256": (
                plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
            ),
            "bitrix_operational_edge_receipt_public_key_id": (
                plan.bitrix_operational_edge_receipt_public_key_id
            ),
            "bitrix_operational_edge_key_bootstrap_receipt_sha256": (
                plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
            ),
        },
    }
    return {**unsigned, "inputs_sha256": runtime._sha256_json(unsigned)}


def _foundation_authoring_context(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    unsigned = {
        "schema": runtime.CAPABILITY_FOUNDATION_AUTHORING_CONTEXT_SCHEMA,
        "revision": plan.revision,
        "staged_plan_path": str(runtime.DEFAULT_STAGED_FULL_CANARY_PLAN_PATH),
        "staged_plan_file_sha256": "8" * 64,
        "staged_plan_identity": {
            "device": 1,
            "inode": 2,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "size": 3,
            "mtime_ns": 4,
        },
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "release_artifact_sha256": plan.release_artifact_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "identities": {
            "service_uid": plan.identities.bitrix_operational_edge_uid,
            "service_gid": plan.identities.bitrix_operational_edge_gid,
            "socket_client_gid": (plan.identities.bitrix_operational_edge_client_gid),
            "business_edge_uid": plan.identities.mac_ops_uid,
        },
        "identity_observation": {
            "service_user": "muncho-edge-bitrix",
            "service_group": "muncho-edge-bitrix",
            "service_uid": plan.identities.bitrix_operational_edge_uid,
            "service_gid": plan.identities.bitrix_operational_edge_gid,
            "socket_client_group": "muncho-edge-bitrix-c",
            "socket_client_gid": (plan.identities.bitrix_operational_edge_client_gid),
            "state": "present_exact",
        },
        "asset_manifest_sha256": (plan.bitrix_operational_edge_asset_manifest_sha256),
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _bitrix_foundation_receipt(
    plan: runtime.CapabilityCanaryPlan,
    foundation_context: dict[str, object],
) -> dict[str, object]:
    unsigned = {
        "schema": runtime.CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA,
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "foundation_authoring_context_receipt_sha256": foundation_context[
            "receipt_sha256"
        ],
        "asset_manifest_sha256": (plan.bitrix_operational_edge_asset_manifest_sha256),
        "rendered_unit_sha256": (plan.bitrix_operational_edge_rendered_unit_sha256),
        "rendered_config_sha256": (plan.bitrix_operational_edge_rendered_config_sha256),
        "rendered_trust_sha256": (plan.bitrix_operational_edge_rendered_trust_sha256),
        "identity_bootstrap_receipt_sha256": (
            plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
        ),
        "receipt_public_key_id": (plan.bitrix_operational_edge_receipt_public_key_id),
        "key_bootstrap_receipt_sha256": (
            plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _plan_authoring_context(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    foundation = _foundation_authoring_context(plan)
    bitrix = _bitrix_foundation_receipt(plan, foundation)
    inputs = _plan_publication_inputs(plan)

    def observation(schema: str, **extra: object) -> dict[str, object]:
        unsigned_observation = {
            "schema": schema,
            "plan_sha256": plan.sha256,
            **extra,
        }
        return {
            **unsigned_observation,
            "receipt_sha256": runtime._sha256_json(unsigned_observation),
        }

    unsigned = {
        "schema": runtime.CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA,
        "revision": plan.revision,
        "staged_plan_path": str(runtime.DEFAULT_STAGED_FULL_CANARY_PLAN_PATH),
        "staged_plan_file_sha256": "8" * 64,
        "staged_plan_identity": {
            "device": 1,
            "inode": 2,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "size": 3,
            "mtime_ns": 4,
        },
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "foundation_authoring_context": foundation,
        "foundation_authoring_context_receipt_sha256": foundation["receipt_sha256"],
        "bitrix_foundation_receipt": bitrix,
        "bitrix_foundation_receipt_sha256": bitrix["receipt_sha256"],
        "plan_inputs": inputs,
        "host_identity_observations": {
            "browser": observation(runtime.CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA),
            "execution": observation(runtime.CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA),
            "mac_ops": observation(
                runtime.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA,
                role="mac_ops",
                state="present_exact",
                create_only_eligible=True,
            ),
            "connector": observation(
                runtime.CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA,
                role="connector",
                state="present_exact",
                create_only_eligible=True,
            ),
            "producer": _producer_identity_observation(plan),
        },
        "capability_inputs_sha256": inputs["inputs_sha256"],
        "capability_plan_sha256": plan.sha256,
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}


def _plan_publication_authority(
    plan: runtime.CapabilityCanaryPlan,
) -> dict[str, object]:
    authoring = _plan_authoring_context(plan)
    inputs = {
        key: value
        for key, value in authoring["plan_inputs"].items()
        if key
        not in {
            "schema",
            "full_canary_terminal_receipt",
            "full_canary_terminal_receipt_sha256",
            "inputs_sha256",
        }
    }
    unsigned = {
        "schema": runtime.CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA,
        "scope": runtime.CAPABILITY_PLAN_PUBLICATION_SCOPE,
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_authoring_context": authoring,
        "plan_authoring_context_receipt_sha256": authoring["receipt_sha256"],
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "d" * 64,
        "authority_kind": "trusted_gcloud_owner_explicit_plan_digest",
        "cryptographic_owner_proof": False,
        "inputs": inputs,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "authority_sha256": runtime._sha256_json(unsigned),
    }


def _patch_publication_host_identity_observations(
    monkeypatch: pytest.MonkeyPatch,
    authority: dict[str, object],
) -> None:
    observations = authority["plan_authoring_context"]["host_identity_observations"]
    monkeypatch.setattr(
        runtime,
        "browser_host_identity_receipt",
        lambda *_args, **_kwargs: observations["browser"],
    )
    monkeypatch.setattr(
        runtime,
        "execution_host_identity_receipt",
        lambda *_args, **_kwargs: observations["execution"],
    )
    monkeypatch.setattr(
        runtime,
        "service_host_identity_receipt",
        lambda *_args, role, **_kwargs: observations[role],
    )
    monkeypatch.setattr(
        producer_units,
        "producer_host_identity_receipt",
        lambda *_args, **_kwargs: observations["producer"],
    )
    monkeypatch.setattr(
        runtime,
        "_observe_bitrix_foundation_identity",
        lambda **_kwargs: authority["plan_authoring_context"][
            "foundation_authoring_context"
        ]["identity_observation"],
    )


def _bitrix_foundation_authority(
    full: FullCanaryPlan,
    *,
    now_unix: int | None = None,
) -> dict[str, object]:
    issued = int(time.time()) if now_unix is None else now_unix
    plan = _plan(full)
    foundation = _foundation_authoring_context(plan)
    unsigned = {
        "schema": runtime.CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA,
        "scope": runtime.CAPABILITY_BITRIX_FOUNDATION_SCOPE,
        "revision": full.revision,
        "full_canary_plan_sha256": full.sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "foundation_authoring_context_receipt_sha256": foundation["receipt_sha256"],
        "release_artifact_sha256": full.release["artifact_sha256"],
        "owner_subject_sha256": "a" * 64,
        "authority_kind": "trusted_gcloud_owner_explicit_foundation_digest",
        "cryptographic_owner_proof": False,
        "issued_at_unix": issued,
        "expires_at_unix": issued + 900,
        "identities": {
            "service_uid": 2108,
            "service_gid": 2210,
            "socket_client_gid": 2211,
            "business_edge_uid": 2104,
        },
        "asset_manifest_sha256": "b" * 64,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {
        **unsigned,
        "authority_sha256": runtime._sha256_json(unsigned),
    }


def _jwt(expiry: int) -> bytearray:
    def segment(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return bytearray(
        f"{segment({'alg': 'none'})}.{segment({'exp': expiry})}.sig".encode()
    )


def _capability_approval(
    plan: runtime.CapabilityCanaryPlan,
    *,
    approved_at_unix: int,
    expires_at_unix: int,
    stopped_preflight_state_sha256: str = "4" * 64,
    stopped_preflight_report_sha256: str = "6" * 64,
    nonce_sha256: str = "3" * 64,
) -> runtime.CapabilityCanaryOwnerApproval:
    return runtime.CapabilityCanaryOwnerApproval.from_mapping({
        "schema": runtime.CAPABILITY_APPROVAL_SCHEMA,
        "scope": "production_capability_canary_runtime_start",
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_publication_receipt_sha256": "8" * 64,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": "1" * 64,
        "approval_source_sha256": "2" * 64,
        "stopped_preflight_state_sha256": (stopped_preflight_state_sha256),
        "stopped_preflight_report_sha256": (stopped_preflight_report_sha256),
        "stopped_preflight_observed_at_unix": max(approved_at_unix - 1, 0),
        "fixture_sha256": "7" * 64,
        "fixture_publication_receipt_sha256": "9" * 64,
        "lease_install_receipt_sha256_by_binding": {
            binding: hashlib.sha256(binding.encode()).hexdigest()
            for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
        },
        "bitrix_expiry_watchdog_authority_sha256": "a" * 64,
        "approval_not_after_unix": expires_at_unix,
        "nonce_sha256": nonce_sha256,
        "approved_at_unix": approved_at_unix,
        "expires_at_unix": expires_at_unix,
    })


def _stopped_preflight_for_approval(
    plan: runtime.CapabilityCanaryPlan,
    approval: runtime.CapabilityCanaryOwnerApproval,
) -> dict[str, object]:
    value = approval.value
    return {
        "schema": runtime.CAPABILITY_PREFLIGHT_SCHEMA,
        "phase": "stopped",
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": dict(plan.full_canary_terminal_receipt),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_publication_receipt_sha256": value["plan_publication_receipt_sha256"],
        "fixture_sha256": value["fixture_sha256"],
        "fixture_publication_receipt_sha256": value[
            "fixture_publication_receipt_sha256"
        ],
        "lease_install_receipt_sha256_by_binding": dict(
            value["lease_install_receipt_sha256_by_binding"]
        ),
        "bitrix_expiry_watchdog_authority_sha256": value[
            "bitrix_expiry_watchdog_authority_sha256"
        ],
        "approval_not_after_unix": value["approval_not_after_unix"],
        "state_sha256": value["stopped_preflight_state_sha256"],
        "report_sha256": value["stopped_preflight_report_sha256"],
        "observed_at_unix": value["stopped_preflight_observed_at_unix"],
        "ok": True,
    }


def test_plan_renders_exact_model_owned_first_wave():
    plan = _plan()
    restored = runtime.CapabilityCanaryPlan.from_mapping(plan.to_mapping())
    assert restored == plan
    assert plan.full_canary_terminal_receipt == _full_canary_terminal_receipt()
    assert (
        plan.full_canary_terminal_receipt_sha256
        == plan.full_canary_terminal_receipt["receipt_sha256"]
    )
    assert (
        plan.original_full_canary_owner_approval_sha256
        == plan.full_canary_terminal_receipt["owner_approval_sha256"]
    )
    config = yaml.safe_load(runtime.render_gateway_config(plan))

    assert config["model"] == {
        "default": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    assert config["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "max",
    }
    assert config["agent"] == {
        "reasoning_effort": "high",
        "max_turns": 90,
        "adaptive_reasoning": {"enabled": True, "max_effort": "max"},
        "tool_use_enforcement": True,
        "task_completion_guidance": True,
        "parallel_tool_call_guidance": True,
        "background_review_enabled": False,
        "verification_ledger_enabled": False,
        "verify_on_stop": False,
    }
    assert config["tools"] == {"tool_search": {"enabled": "off"}}
    assert all(
        route["provider"] == "openai-codex"
        and route["model"] == "gpt-5.6-sol"
        and route["fallback_chain"] == []
        for route in config["auxiliary"].values()
    )
    assert config["platform_toolsets"]["api_server"] == list(
        runtime.FIRST_WAVE_TOOLSETS
    )
    assert config["platform_toolsets"]["relay"] == list(runtime.FIRST_WAVE_TOOLSETS)
    assert config["gateway"]["isolated_runtime"] is False
    assert set(config["platforms"]) == {"api_server", "relay"}
    assert config["platforms"]["relay"]["extra"]["relay_url"] == (
        "unix:///run/muncho-discord-connector/connector.sock"
    )
    assert "mac_ops" in config["platform_toolsets"]["api_server"]
    assert config["kanban"] == {
        "auxiliary_planning_enabled": False,
        "auto_decompose": False,
        "dispatch_in_gateway": False,
    }
    assert config["cron"] == {"enabled": False}
    assert config["plugins"] == {"enabled": [runtime.CAPABILITY_OBSERVER_PLUGIN]}
    assert config["hooks"] == {}
    assert config["memory"] == {
        "provider": "",
        "memory_enabled": True,
        "user_profile_enabled": True,
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("agent", "tool_use_enforcement", "auto"),
        ("agent", "task_completion_guidance", False),
        ("agent", "parallel_tool_call_guidance", False),
        ("agent", "background_review_enabled", True),
        ("agent", "verification_ledger_enabled", True),
        ("agent", "verify_on_stop", True),
        ("tools", "tool_search", {"enabled": "auto"}),
    ),
)
def test_gateway_config_rejects_each_sovereignty_policy_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    config = yaml.safe_load(runtime.render_gateway_config(_plan()))
    config[section][field] = value

    with pytest.raises(ValueError):
        runtime.validate_capability_gateway_config(config)


def test_plan_rejects_terminal_receipt_projection_or_tamper():
    mapping = _plan().to_mapping()
    projected = json.loads(json.dumps(mapping))
    projected["full_canary_terminal_receipt"].pop("coordinator_receipt_sha256")
    with pytest.raises(ValueError, match="terminal receipt"):
        runtime.CapabilityCanaryPlan.from_mapping(projected)

    tampered = json.loads(json.dumps(mapping))
    tampered["full_canary_terminal_receipt"]["services_stopped"] = False
    with pytest.raises(ValueError, match="terminal receipt"):
        runtime.CapabilityCanaryPlan.from_mapping(tampered)


def test_plan_binds_two_canary_bots_distinct_from_production():
    plan = _plan()
    connector = plan.to_mapping()["discord_connector"]

    assert connector["connector_bot_user_id"] == plan.connector_bot_user_id
    assert connector["routeback_bot_user_id"] == plan.routeback_bot_user_id
    assert connector["production_bot_user_id"] == (
        runtime.PRODUCTION_DISCORD_BOT_USER_ID
    )
    assert (
        len({
            connector["connector_bot_user_id"],
            connector["routeback_bot_user_id"],
            connector["production_bot_user_id"],
        })
        == 3
    )

    tampered = plan.to_mapping()
    tampered["discord_connector"]["connector_bot_user_id"] = (
        runtime.PRODUCTION_DISCORD_BOT_USER_ID
    )
    with pytest.raises(ValueError, match="bot identities are not isolated"):
        runtime.CapabilityCanaryPlan.from_mapping(tampered)


def test_live_routeback_identity_is_plan_bound_and_secret_free(monkeypatch):
    plan = _plan()
    full = _full_plan()
    metadata = _routeback_file_metadata()
    calls = []

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            calls.append(("current_user", timeout_seconds))
            return {
                "id": plan.routeback_bot_user_id,
                "bot": True,
                "username": "must-not-enter-the-attestation",
            }

    class FakeAdapter:
        def __init__(self):
            self._api = FakeApi()

        def close(self):
            calls.append(("close",))

    def factory(path, **kwargs):
        calls.append(("factory", path, kwargs))
        return FakeAdapter()

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(metadata),
    )
    receipt = runtime._attest_live_routeback_bot_identity(
        plan,
        full,
        adapter_factory=factory,
        now_unix=1_700_000_000,
    )

    assert receipt["schema"] == runtime.CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA
    assert receipt["plan_sha256"] == plan.sha256
    assert receipt["full_canary_plan_sha256"] == full.sha256
    assert receipt["live_bot_user_id"] == plan.routeback_bot_user_id
    assert receipt["planned_routeback_bot_user_id"] == plan.routeback_bot_user_id
    assert receipt["connector_bot_user_id"] == plan.connector_bot_user_id
    assert receipt["production_bot_user_id"] == (runtime.PRODUCTION_DISCORD_BOT_USER_ID)
    assert receipt["pairwise_distinct"] is True
    assert receipt["credential_file_metadata_sha256"] == runtime._sha256_json(metadata)
    assert receipt["provenance"] == {
        "source": "discord_rest_api_v10_current_user",
        "http_method": "GET",
        "resource": "/users/@me",
        "credential_boundary": "sealed_routeback_credential_file",
    }
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert receipt["attestation_sha256"] == runtime._sha256_json({
        key: value for key, value in receipt.items() if key != "attestation_sha256"
    })
    serialized = json.dumps(receipt)
    assert "must-not-enter-the-attestation" not in serialized
    assert "token" not in serialized
    assert calls == [
        (
            "factory",
            runtime.DEFAULT_EDGE_TOKEN_PATH,
            {
                "credentials_directory": runtime.DEFAULT_EDGE_TOKEN_DIRECTORY,
                "expected_owner_uid": full.identities.edge_uid,
                "timeout_seconds": 5.0,
            },
        ),
        ("current_user", 5.0),
        ("close",),
    ]
    assert (
        runtime._require_routeback_credential_binding(plan, full, receipt) == metadata
    )


def test_routeback_credential_metadata_requires_root_sealed_parent(
    monkeypatch,
    tmp_path,
):
    full = _full_plan()
    directory = tmp_path / "discord-edge-credentials"
    credential_path = directory / "bot-token"
    directory.mkdir()
    credential_path.write_bytes(b"opaque-placeholder-not-read-by-this-test")
    directory_uid = [0]
    file_metadata = _routeback_file_metadata()
    real_lstat = os.lstat

    def fake_lstat(path):
        if Path(path) == directory:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o750,
                st_uid=directory_uid[0],
            )
        if Path(path) == credential_path:
            return SimpleNamespace(
                st_mode=stat.S_IFREG | file_metadata["mode"],
                st_nlink=1,
                st_uid=file_metadata["uid"],
                st_gid=file_metadata["gid"],
                st_size=file_metadata["size"],
                st_dev=file_metadata["device"],
                st_ino=file_metadata["inode"],
                st_mtime_ns=file_metadata["mtime_ns"],
                st_ctime_ns=file_metadata["ctime_ns"],
            )
        return real_lstat(path)

    monkeypatch.setattr(runtime, "DEFAULT_EDGE_TOKEN_DIRECTORY", directory)
    monkeypatch.setattr(runtime, "DEFAULT_EDGE_TOKEN_PATH", credential_path)
    monkeypatch.setattr(runtime.os, "lstat", fake_lstat)

    assert runtime._routeback_credential_file_metadata(full) == file_metadata
    directory_uid[0] = full.identities.edge_uid
    with pytest.raises(RuntimeError, match="directory is writable by the edge"):
        runtime._routeback_credential_file_metadata(full)


@pytest.mark.parametrize("failure_kind", ("mismatch", "unavailable"))
def test_routeback_identity_failure_prevents_edge_start(
    monkeypatch,
    failure_kind,
):
    plan = _plan()
    full = _full_plan()
    calls = []
    original_attestor = runtime._attest_live_routeback_bot_identity

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            calls.append(("current_user", timeout_seconds))
            if failure_kind == "unavailable":
                raise RuntimeError("Discord API unavailable")
            return {
                "id": runtime.PRODUCTION_DISCORD_BOT_USER_ID,
                "bot": True,
            }

    class FakeAdapter:
        def __init__(self):
            self._api = FakeApi()

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: _routeback_file_metadata(),
    )
    monkeypatch.setattr(
        runtime,
        "_attest_live_routeback_bot_identity",
        lambda candidate, foundation: original_attestor(
            candidate,
            foundation,
            adapter_factory=lambda *_args, **_kwargs: FakeAdapter(),
        ),
    )
    lifecycle = runtime.CapabilityCanaryLifecycle(
        plan,
        full,
        runner=lambda command: calls.append(("systemd", command.argv)),
    )

    with pytest.raises(RuntimeError):
        lifecycle._start_routeback_edge(require_approval=lambda: None)

    assert not any(call[0] == "systemd" for call in calls)
    assert calls[-1] == ("close",)


def test_routeback_credential_swap_after_start_stops_edge(monkeypatch):
    plan = _plan()
    full = _full_plan()
    original_metadata = _routeback_file_metadata(inode=41)
    swapped_metadata = _routeback_file_metadata(inode=42)

    class FakeApi:
        def current_user(self, *, timeout_seconds):
            assert timeout_seconds == 5.0
            return {"id": plan.routeback_bot_user_id, "bot": True}

    class FakeAdapter:
        _api = FakeApi()

        def close(self):
            return None

    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(original_metadata),
    )
    identity = runtime._attest_live_routeback_bot_identity(
        plan,
        full,
        adapter_factory=lambda *_args, **_kwargs: FakeAdapter(),
    )
    observations = iter((original_metadata, swapped_metadata))
    monkeypatch.setattr(
        runtime,
        "_routeback_credential_file_metadata",
        lambda _full: dict(next(observations)),
    )
    monkeypatch.setattr(
        runtime,
        "_attest_live_routeback_bot_identity",
        lambda _plan, _full: identity,
    )
    commands = []
    approval_checks = []

    def runner(command):
        commands.append(command.argv)
        return runtime.subprocess.CompletedProcess(
            command.argv,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    lifecycle = runtime.CapabilityCanaryLifecycle(plan, full, runner=runner)
    with pytest.raises(RuntimeError, match="credential object changed"):
        lifecycle._start_routeback_edge(
            require_approval=lambda: approval_checks.append(len(commands))
        )

    assert approval_checks == [0]
    assert commands == [
        runtime.edge_start_command().argv,
        (runtime.SYSTEMCTL, "stop", runtime.EDGE_UNIT_NAME),
    ]


def test_plan_publication_authority_is_exact_and_digest_bound():
    plan = _plan()
    authority = _plan_publication_authority(plan)

    assert runtime.validate_plan_publication_authority(authority) == authority
    assert (
        runtime.build_plan_from_publication_authority(authority, _full_plan()) == plan
    )

    tampered = json.loads(json.dumps(authority))
    tampered["inputs"]["identities"]["browser_uid"] += 1
    with pytest.raises(ValueError, match="fixed inventory"):
        runtime.validate_plan_publication_authority(tampered)

    wrong_digest = json.loads(json.dumps(authority))
    wrong_digest["plan_sha256"] = "e" * 64
    authoring = wrong_digest["plan_authoring_context"]
    authoring["capability_plan_sha256"] = "e" * 64
    for observation in authoring["host_identity_observations"].values():
        observation["plan_sha256"] = "e" * 64
        observation_unsigned = {
            key: value for key, value in observation.items() if key != "receipt_sha256"
        }
        observation["receipt_sha256"] = runtime._sha256_json(observation_unsigned)
    authoring_unsigned = {
        key: value for key, value in authoring.items() if key != "receipt_sha256"
    }
    authoring["receipt_sha256"] = runtime._sha256_json(authoring_unsigned)
    wrong_digest["plan_authoring_context_receipt_sha256"] = authoring["receipt_sha256"]
    unsigned = {
        key: value for key, value in wrong_digest.items() if key != "authority_sha256"
    }
    wrong_digest["authority_sha256"] = runtime._sha256_json(unsigned)
    with pytest.raises(PermissionError, match="approved capability plan digest"):
        runtime.build_plan_from_publication_authority(wrong_digest, _full_plan())


def test_plan_publication_rejects_missing_or_tampered_authoring_chain():
    authority = _plan_publication_authority(_plan())
    missing = json.loads(json.dumps(authority))
    missing.pop("plan_authoring_context")
    missing.pop("plan_authoring_context_receipt_sha256")
    missing_unsigned = {
        key: value for key, value in missing.items() if key != "authority_sha256"
    }
    missing["authority_sha256"] = runtime._sha256_json(missing_unsigned)
    with pytest.raises(ValueError, match="fields are not exact"):
        runtime.validate_plan_publication_authority(missing)

    tampered = json.loads(json.dumps(authority))
    tampered["plan_authoring_context"]["mutation_performed"] = True
    tampered_unsigned = {
        key: value for key, value in tampered.items() if key != "authority_sha256"
    }
    tampered["authority_sha256"] = runtime._sha256_json(tampered_unsigned)
    with pytest.raises(ValueError, match="authoring context is invalid"):
        runtime.validate_plan_publication_authority(tampered)


def test_foundation_authoring_collector_uses_only_fixed_sealed_facts(
    monkeypatch,
):
    full = _full_plan()
    plan = _plan(full)
    terminal = dict(plan.full_canary_terminal_receipt)
    request_unsigned = {
        "schema": runtime.CAPABILITY_FOUNDATION_AUTHORING_REQUEST_SCHEMA,
        "revision": full.revision,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    request = {
        **request_unsigned,
        "request_sha256": runtime._sha256_json(request_unsigned),
    }
    staged_identity = {
        "device": 1,
        "inode": 2,
        "uid": 0,
        "gid": 0,
        "mode": "0400",
        "size": 3,
        "mtime_ns": 4,
    }
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        runtime,
        "_require_root_linux",
        lambda: calls.append(("root", None)),
    )
    monkeypatch.setattr(
        runtime,
        "_read_staged_full_canary_plan",
        lambda: (full, b"sealed-staged-plan", staged_identity),
    )

    def verify_assets(**kwargs):
        calls.append(("assets", kwargs))
        return {"manifest_sha256": plan.bitrix_operational_edge_asset_manifest_sha256}

    identity = _foundation_authoring_context(plan)["identity_observation"]

    def observe_identity(**kwargs):
        calls.append(("identity", kwargs))
        return identity

    monkeypatch.setattr(runtime, "verify_packaged_operational_assets", verify_assets)
    monkeypatch.setattr(
        runtime,
        "_observe_bitrix_foundation_identity",
        observe_identity,
    )

    context = runtime.collect_foundation_authoring_context(request)

    assert context["staged_plan_path"] == str(
        runtime.DEFAULT_STAGED_FULL_CANARY_PLAN_PATH
    )
    assert context["staged_plan_identity"] == staged_identity
    assert context["full_canary_terminal_receipt"] == terminal
    assert context["identities"] == {
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_gid": 2211,
        "business_edge_uid": 2104,
    }
    assert context["identity_observation"] == identity
    assert context["mutation_performed"] is False
    assert calls == [
        ("root", None),
        (
            "assets",
            {
                "release_root": Path(full.release["artifact_root"]),
                "revision": full.revision,
                "expected_uid": 0,
                "expected_gid": 0,
            },
        ),
        (
            "identity",
            {
                "service_uid": 2108,
                "service_gid": 2210,
                "socket_client_gid": 2211,
                "allow_absence": True,
            },
        ),
    ]

    tampered = json.loads(json.dumps(request))
    tampered["full_canary_terminal_receipt"]["services_stopped"] = False
    with pytest.raises(ValueError, match="terminal receipt"):
        runtime.collect_foundation_authoring_context(tampered)
    assert calls[-1] == ("root", None)


def test_plan_authoring_collector_derives_fixed_inventory_without_mutation(
    monkeypatch,
):
    full = _full_plan()
    manifest_raw = b"sealed-runtime-dependency-manifest"
    plan = _plan(
        full,
        runtime_dependency_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    foundation = _foundation_authoring_context(plan)
    bitrix = _bitrix_foundation_receipt(plan, foundation)
    terminal = dict(plan.full_canary_terminal_receipt)
    request_unsigned = {
        "schema": runtime.CAPABILITY_PLAN_AUTHORING_REQUEST_SCHEMA,
        "revision": full.revision,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "foundation_authoring_context": foundation,
        "foundation_authoring_context_receipt_sha256": foundation["receipt_sha256"],
        "bitrix_foundation_receipt": bitrix,
        "bitrix_foundation_receipt_sha256": bitrix["receipt_sha256"],
        "connector_bot_user_id": plan.connector_bot_user_id,
        "routeback_bot_user_id": plan.routeback_bot_user_id,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    request = {
        **request_unsigned,
        "request_sha256": runtime._sha256_json(request_unsigned),
    }
    staged_identity = {
        "device": 1,
        "inode": 2,
        "uid": 0,
        "gid": 0,
        "mode": "0400",
        "size": 3,
        "mtime_ns": 4,
    }
    runtime_manifest = {
        "agent_browser": {
            "node_sha256": plan.browser_node_sha256,
            "wrapper_sha256": plan.browser_wrapper_sha256,
            "native_sha256": plan.browser_native_sha256,
            "config_sha256": plan.agent_browser_config_sha256,
        },
        "chrome": {"executable_sha256": plan.browser_executable_sha256},
    }
    expected_observations = _plan_authoring_context(plan)["host_identity_observations"]
    observed_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_read_staged_full_canary_plan",
        lambda: (full, b"sealed-staged-plan", staged_identity),
    )
    monkeypatch.setattr(
        runtime,
        "verify_release_runtime_dependency_manifest",
        lambda release_root, revision: (
            runtime_manifest
            if release_root == Path(full.release["artifact_root"])
            and revision == full.revision
            else (_ for _ in ()).throw(AssertionError("unexpected release"))
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda path, **_kwargs: (
            (
                manifest_raw,
                SimpleNamespace(),
            )
            if path
            == Path(full.release["artifact_root"])
            / runtime.RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
            else (_ for _ in ()).throw(AssertionError("unexpected mutable path"))
        ),
    )

    def executable_digest(path, label):
        observed_calls.append((path, label))
        return {
            runtime.BWRAP_PATH: plan.worker_bwrap_sha256,
            runtime.SHELL_PATH: plan.worker_shell_sha256,
        }[path]

    monkeypatch.setattr(runtime, "_stable_executable_sha256", executable_digest)
    monkeypatch.setattr(
        runtime,
        "browser_host_identity_receipt",
        lambda *_args, **_kwargs: expected_observations["browser"],
    )
    monkeypatch.setattr(
        runtime,
        "execution_host_identity_receipt",
        lambda *_args, **_kwargs: expected_observations["execution"],
    )
    monkeypatch.setattr(
        runtime,
        "service_host_identity_receipt",
        lambda *_args, role, **_kwargs: expected_observations[role],
    )
    monkeypatch.setattr(
        producer_units,
        "producer_host_identity_receipt",
        lambda *_args, **_kwargs: expected_observations["producer"],
    )

    context = runtime.collect_plan_authoring_context(request)

    assert context["plan_inputs"] == _plan_publication_inputs(plan)
    assert context["capability_plan_sha256"] == plan.sha256
    assert context["host_identity_observations"] == expected_observations
    assert context["full_canary_terminal_receipt"] == terminal
    assert context["mutation_performed"] is False
    assert observed_calls == [
        (runtime.BWRAP_PATH, "isolated worker bwrap"),
        (runtime.SHELL_PATH, "isolated worker shell"),
    ]


@pytest.mark.parametrize(
    "locked_channel_id", sorted(runtime.LOCKED_NONPUBLIC_CHANNEL_IDS)
)
def test_plan_publication_authority_rejects_each_locked_discord_channel(
    locked_channel_id,
):
    authority = _plan_publication_authority(_plan())
    authority["inputs"]["discord"]["allowed_channel_ids"] = [locked_channel_id]
    unsigned = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    authority["authority_sha256"] = runtime._sha256_json(unsigned)

    with pytest.raises(ValueError, match="public Discord target is invalid"):
        runtime.validate_plan_publication_authority(authority)


def test_bitrix_preplan_bootstrap_retires_both_key_halves_and_requires_fresh_authority_to_reboot(
    tmp_path,
    monkeypatch,
):
    full = _full_plan()
    authority = _bitrix_foundation_authority(full)
    private_path = tmp_path / "keys/bitrix-private.pem"
    public_path = tmp_path / "trust/bitrix-public.pem"
    writer_path = tmp_path / "keys/writer-public.pem"
    writer_path.parent.mkdir(parents=True)
    writer_key = runtime.Ed25519PrivateKey.generate().public_key()
    writer_path.write_bytes(runtime._ed25519_public_pem(writer_key))
    os.chown(writer_path, os.geteuid(), os.getegid())
    writer_path.chmod(0o444)

    identity = {
        "service_user": "muncho-edge-bitrix",
        "service_group": "muncho-edge-bitrix",
        "service_uid": 2108,
        "service_gid": 2210,
        "socket_client_group": "muncho-edge-bitrix-c",
        "socket_client_gid": 2211,
        "state": "present_exact",
    }

    def observe_identity(**_kwargs):
        return dict(identity)

    def verify_assets(**_kwargs):
        return {
            "manifest_sha256": "b" * 64,
            "verification_sha256": "c" * 64,
            "files": [
                {"asset_id": asset_id}
                for asset_id in runtime.BITRIX_OPERATIONAL_EDGE_ASSET_IDS.values()
            ],
        }

    arguments = {
        "full_plan": full,
        "runner": lambda command: runtime.subprocess.CompletedProcess(
            command.argv, 0, b"", b""
        ),
        "identity_observer": observe_identity,
        "asset_verifier": verify_assets,
        "private_key_path": private_path,
        "public_key_path": public_path,
        "writer_public_key_path": writer_path,
        "identity_receipt_path": tmp_path / "state/identity.json",
        "foundation_root": tmp_path / "state/foundations",
        "key_bootstrap_root": tmp_path / "state/key-bootstraps",
        "require_root": False,
        "clock": lambda: authority["issued_at_unix"],
    }
    first = runtime.bootstrap_bitrix_foundation(authority, **arguments)
    first_key_id = first["receipt_public_key_id"]
    assert (
        first["full_canary_terminal_receipt"]
        == authority["full_canary_terminal_receipt"]
    )
    assert (
        first["full_canary_terminal_receipt_sha256"]
        == authority["full_canary_terminal_receipt_sha256"]
    )
    assert (
        first["original_full_canary_owner_approval_sha256"]
        == authority["original_full_canary_owner_approval_sha256"]
    )
    assert (
        first["foundation_authoring_context_receipt_sha256"]
        == authority["foundation_authoring_context_receipt_sha256"]
    )
    assert private_path.exists() and public_path.exists()
    assert first["read_peer_uids"] == [
        full.identities.writer_uid,
        2104,
    ]
    assert first["mutation_peer_uid"] == full.identities.writer_uid
    assert '"plan_sha256":' not in json.dumps(first)
    key_receipt = runtime._load_lease_artifact(
        Path(first["key_bootstrap_receipt_path"]),
        schema=runtime.CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
    )
    assert key_receipt["retire_private_on_stop"] is True
    assert key_receipt["retire_public_on_stop"] is True
    assert "private_sha256" not in json.dumps(key_receipt)
    for index in range(20):
        (Path(arguments["key_bootstrap_root"]) / f"{index:064x}").mkdir()
    indexed = runtime._bitrix_key_receipts_for_authority(
        authority["authority_sha256"],
        root=arguments["key_bootstrap_root"],
    )
    assert [item["receipt_sha256"] for item in indexed] == [
        key_receipt["receipt_sha256"]
    ]
    authority_index_path = (
        Path(arguments["key_bootstrap_root"])
        / ".authority-index"
        / authority["authority_sha256"]
        / "key-bootstrap.json"
    )
    authority_index_raw = authority_index_path.read_bytes()
    corrupted_index = json.loads(authority_index_raw)
    corrupted_index["key_bootstrap_receipt_sha256"] = "0" * 64
    corrupted_index["receipt_sha256"] = runtime._sha256_json({
        key: value for key, value in corrupted_index.items() if key != "receipt_sha256"
    })
    authority_index_path.chmod(0o600)
    authority_index_path.write_bytes(runtime._canonical_bytes(corrupted_index))
    authority_index_path.chmod(0o400)
    with pytest.raises(RuntimeError, match="index binding drifted"):
        runtime._bitrix_key_receipts_for_authority(
            authority["authority_sha256"],
            root=arguments["key_bootstrap_root"],
        )
    authority_index_path.chmod(0o600)
    authority_index_path.write_bytes(authority_index_raw)
    authority_index_path.chmod(0o400)
    for index in range(20):
        (Path(arguments["foundation_root"]) / f"{index + 100:064x}").mkdir()
    bound_plan = replace(
        _plan(full),
        bitrix_operational_edge_receipt_public_key_id=first["receipt_public_key_id"],
        bitrix_operational_edge_key_bootstrap_receipt_sha256=key_receipt[
            "receipt_sha256"
        ],
        bitrix_operational_edge_asset_manifest_sha256=first["asset_manifest_sha256"],
        bitrix_operational_edge_identity_bootstrap_receipt_sha256=first[
            "identity_bootstrap_receipt_sha256"
        ],
        bitrix_operational_edge_rendered_unit_sha256=first["rendered_unit_sha256"],
        bitrix_operational_edge_rendered_config_sha256=first["rendered_config_sha256"],
        bitrix_operational_edge_rendered_trust_sha256=first["rendered_trust_sha256"],
    )
    monkeypatch.setattr(
        runtime,
        "DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH",
        private_path,
    )
    monkeypatch.setattr(runtime, "DEFAULT_BITRIX_TRUST_PATH", public_path)
    read_exact_file = runtime._read_exact_file
    current_owned = {
        Path(first["rendered_unit_stage_path"]),
        Path(first["rendered_config_stage_path"]),
        public_path,
    }

    def read_with_test_owner(path, **kwargs):
        if Path(path) in current_owned:
            kwargs["uid"] = os.geteuid()
            kwargs["gid"] = os.getegid()
        return read_exact_file(path, **kwargs)

    monkeypatch.setattr(runtime, "_read_exact_file", read_with_test_owner)
    ready = runtime.validate_bitrix_foundation_for_plan(
        bound_plan,
        full,
        foundation_root=arguments["foundation_root"],
        key_bootstrap_root=arguments["key_bootstrap_root"],
        identity_receipt_path=arguments["identity_receipt_path"],
        now_unix=authority["issued_at_unix"],
    )
    assert ready["foundation_receipt_sha256"] == first["receipt_sha256"]
    assert ready["ready"] is True

    retirement = runtime.retire_bitrix_foundation_key_pair(
        key_receipt,
        reason="foundation_expired",
        now_unix=authority["expires_at_unix"],
        retirement_root=tmp_path / "state/key-retirements",
        require_root=False,
    )
    assert retirement["private_absent"] is True
    assert retirement["public_absent"] is True
    assert retirement["both_pair_members_absent"] is True
    assert not private_path.exists() and not public_path.exists()

    arguments["clock"] = lambda: authority["expires_at_unix"]
    with pytest.raises(
        RuntimeError,
        match="bootstrap failed and was retired",
    ):
        runtime.bootstrap_bitrix_foundation(authority, **arguments)
    assert not private_path.exists() and not public_path.exists()

    fresh_authority = _bitrix_foundation_authority(
        full,
        now_unix=authority["expires_at_unix"] + 1,
    )
    arguments["clock"] = lambda: fresh_authority["issued_at_unix"]
    second = runtime.bootstrap_bitrix_foundation(
        fresh_authority,
        **arguments,
    )
    assert private_path.exists() and public_path.exists()
    assert second["receipt_public_key_id"] != first_key_id
    assert "private_sha256" not in json.dumps(second)


def test_bitrix_foundation_authority_forbids_capability_plan_self_reference():
    authority = _bitrix_foundation_authority(_full_plan())
    authority["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fields are not exact"):
        runtime.validate_bitrix_foundation_authority(authority)


def test_bitrix_bootstrap_abort_synchronously_retires_partial_key_pair(
    tmp_path,
):
    authority = _bitrix_foundation_authority(_full_plan())
    private_path = tmp_path / "keys/private.pem"
    public_path = tmp_path / "trust/public.pem"
    runtime._stage_bitrix_receipt_key_pair(
        private_path=private_path,
        public_path=public_path,
        require_root=False,
    )
    public_path.unlink()
    result = runtime._retire_bitrix_bootstrap_pair_after_abort(
        authority,
        private_path=private_path,
        public_path=public_path,
        abort_root=tmp_path / "abort",
        now_unix=authority["issued_at_unix"] + 1,
        require_root=False,
    )
    assert result["private_absent"] is True
    assert result["public_absent"] is True
    assert result["both_pair_members_absent"] is True
    assert result["private_content_or_digest_recorded"] is False
    assert not private_path.exists() and not public_path.exists()
    assert (
        runtime._retire_bitrix_bootstrap_pair_after_abort(
            authority,
            private_path=private_path,
            public_path=public_path,
            abort_root=tmp_path / "abort",
            now_unix=authority["issued_at_unix"] + 2,
            require_root=False,
        )
        == result
    )


def test_invalid_bitrix_authority_never_retires_unrelated_fixed_pair(tmp_path):
    full = _full_plan()
    authority = _bitrix_foundation_authority(full)
    authority["plan_sha256"] = "f" * 64
    private_path = tmp_path / "keys/private.pem"
    public_path = tmp_path / "trust/public.pem"
    runtime._stage_bitrix_receipt_key_pair(
        private_path=private_path,
        public_path=public_path,
        require_root=False,
    )
    with pytest.raises(ValueError):
        runtime.bootstrap_bitrix_foundation(
            authority,
            full_plan=full,
            private_key_path=private_path,
            public_key_path=public_path,
            writer_public_key_path=tmp_path / "writer.pem",
            identity_receipt_path=tmp_path / "identity.json",
            foundation_root=tmp_path / "foundations",
            key_bootstrap_root=tmp_path / "key-bootstraps",
            require_root=False,
            clock=lambda: authority["issued_at_unix"],
        )
    assert private_path.exists() and public_path.exists()


def _watchdog_stopped_services():
    return {
        unit: {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": 0,
            "FragmentPath": "",
            "DropInPaths": "",
            "Type": "",
            "NotifyAccess": "",
            "StatusText": "",
        }
        for unit in runtime.CAPABILITY_STOP_ORDER
    }


def test_processless_socket_observation_normalizes_only_fixed_empty_properties():
    def result_for(unit, *, drop=()):
        values = {
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "FragmentPath": "",
            "DropInPaths": "",
            "Type": "",
            "NotifyAccess": "none",
            "StatusText": "",
        }
        for name in drop:
            values.pop(name)
        stdout = "".join(f"{name}={value}\n" for name, value in values.items())
        return runtime.subprocess.CompletedProcess([], 0, stdout.encode(), b"")

    socket = runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME
    socket_missing = tuple(runtime._PROCESSLESS_UNIT_PROPERTY_DEFAULTS[socket])
    state = runtime.collect_capability_service_state(
        socket,
        runner=lambda _command: result_for(socket, drop=socket_missing),
    )
    assert state["MainPID"] == 0
    assert state["Type"] == ""
    assert state["NotifyAccess"] == ""
    assert state["StatusText"] == ""

    service = runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME
    with pytest.raises(RuntimeError, match="fields are not exact"):
        runtime.collect_capability_service_state(
            service,
            runner=lambda _command: result_for(service, drop=("MainPID",)),
        )

    unexpected = result_for(socket)
    unexpected = runtime.subprocess.CompletedProcess(
        unexpected.args,
        unexpected.returncode,
        unexpected.stdout + b"Unexpected=x\n",
        unexpected.stderr,
    )
    with pytest.raises(RuntimeError, match="fields are not exact"):
        runtime.collect_capability_service_state(
            socket,
            runner=lambda _command: unexpected,
        )


def test_expiry_watchdog_is_absolute_persistent_and_idempotent(tmp_path):
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    first = runtime.arm_capability_expiry_watchdog(**arguments)
    second = runtime.arm_capability_expiry_watchdog(**arguments)
    assert second == first
    assert first["cleanup_at_unix"] == first["expires_at_unix"]
    paths = runtime._expiry_watchdog_paths(
        first["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    service = Path(paths["service_path"]).read_text("ascii")
    timer = Path(paths["timer_path"]).read_text("ascii")
    assert "Restart=on-failure\n" in service
    assert "expiry-cleanup --watchdog-id " + first["watchdog_id"] in service
    assert "OnCalendar=@1800000000\n" in timer
    assert "Persistent=true\n" in timer
    assert Path(paths["timer_wants_path"]).is_symlink()
    assert os.readlink(paths["timer_wants_path"]) == ("../" + str(paths["timer_name"]))
    assert all("enable" not in argv for argv in calls)
    assert calls.count((runtime.SYSTEMCTL, "start", paths["timer_name"])) == 2


def test_completed_watchdog_generation_is_rejected_without_name_error(tmp_path):
    import gateway.canonical_capability_canary_producers as producers

    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    armed = runtime.arm_capability_expiry_watchdog(**arguments)
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    authority = runtime._load_expiry_watchdog_authority(
        armed["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    active_run = producers._no_active_api_admission_retirement(
        observed_at_unix_ms=1_800_000_000_000,
    )
    runtime._append_lease_artifact(
        Path(paths["completion"]),
        schema=runtime.CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA,
        value={
            "operation": "persistent_expiry_cleanup",
            "watchdog_id": armed["watchdog_id"],
            "watchdog_authority_sha256": authority["receipt_sha256"],
            "active_run_retirement": active_run,
            "active_run_retirement_sha256": active_run["receipt_sha256"],
            "ok": True,
        },
    )

    with pytest.raises(PermissionError, match="completed capability"):
        runtime.arm_capability_expiry_watchdog(**arguments)


def test_normal_watchdog_disarm_is_durable_and_same_generation_cannot_rearm(
    tmp_path,
):
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    armed = runtime.arm_capability_expiry_watchdog(**arguments)
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    first = runtime.disarm_all_capability_expiry_watchdogs(
        runner=runner,
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
        require_root=False,
    )
    second = runtime.disarm_all_capability_expiry_watchdogs(
        runner=runner,
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
        require_root=False,
    )
    assert second == first
    assert first["all_timers_disabled"] is True
    assert first["all_unit_files_absent"] is True
    assert Path(paths["disarm_intent"]).is_file()
    assert Path(paths["disarm_completion"]).is_file()
    assert not os.path.lexists(paths["service_path"])
    assert not os.path.lexists(paths["timer_path"])
    assert not os.path.lexists(paths["timer_wants_path"])
    with pytest.raises(PermissionError, match="normally disarmed"):
        runtime.arm_capability_expiry_watchdog(**arguments)


def test_normal_watchdog_disarm_recovers_after_units_removed_before_completion(
    monkeypatch,
    tmp_path,
):
    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    armed = runtime.arm_capability_expiry_watchdog(**arguments)
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    append_artifact = runtime._append_lease_artifact

    def interrupt_completion(path, *, schema, value):
        if schema == runtime.CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA:
            raise SystemExit("injected normal disarm completion window")
        return append_artifact(path, schema=schema, value=value)

    monkeypatch.setattr(runtime, "_append_lease_artifact", interrupt_completion)
    with pytest.raises(SystemExit, match="completion window"):
        runtime.disarm_all_capability_expiry_watchdogs(
            runner=runner,
            state_root=arguments["state_root"],
            systemd_root=arguments["systemd_root"],
            require_root=False,
        )
    assert Path(paths["disarm_intent"]).is_file()
    assert not Path(paths["disarm_completion"]).exists()
    assert not os.path.lexists(paths["service_path"])
    assert not os.path.lexists(paths["timer_path"])
    assert not os.path.lexists(paths["timer_wants_path"])

    monkeypatch.setattr(runtime, "_append_lease_artifact", append_artifact)
    recovered = runtime.disarm_all_capability_expiry_watchdogs(
        runner=runner,
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
        require_root=False,
    )
    assert recovered["all_timers_disabled"] is True
    assert recovered["all_unit_files_absent"] is True
    assert Path(paths["disarm_completion"]).is_file()
    with pytest.raises(PermissionError, match="normally disarmed"):
        runtime.arm_capability_expiry_watchdog(**arguments)


def test_one_watchdog_rearm_recovery_never_disarms_unrelated_generation(
    tmp_path,
):
    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    common = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    first_arguments = {**common, "authority_sha256": "d" * 64}
    second_arguments = {**common, "authority_sha256": "e" * 64}
    first = runtime.arm_capability_expiry_watchdog(**first_arguments)
    second = runtime.arm_capability_expiry_watchdog(**second_arguments)
    first_paths = runtime._expiry_watchdog_paths(
        first["watchdog_id"],
        state_root=common["state_root"],
        systemd_root=common["systemd_root"],
    )
    second_paths = runtime._expiry_watchdog_paths(
        second["watchdog_id"],
        state_root=common["state_root"],
        systemd_root=common["systemd_root"],
    )
    runtime._disarm_capability_expiry_watchdog(
        first["watchdog_id"],
        runner=runner,
        state_root=common["state_root"],
        systemd_root=common["systemd_root"],
        require_root=False,
    )
    with pytest.raises(PermissionError, match="normally disarmed"):
        runtime.arm_capability_expiry_watchdog(**first_arguments)
    assert os.path.lexists(first_paths["disarm_completion"])
    assert os.path.lexists(second_paths["service_path"])
    assert os.path.lexists(second_paths["timer_path"])
    assert os.path.lexists(second_paths["timer_wants_path"])
    assert not os.path.lexists(second_paths["disarm_intent"])
    assert not os.path.lexists(second_paths["disarm_completion"])

    selected = runtime.disarm_all_capability_expiry_watchdogs(
        runner=runner,
        state_root=common["state_root"],
        systemd_root=common["systemd_root"],
        require_root=False,
        expected_authority_receipt_sha256s=(
            second["authority_receipt_sha256"],
        ),
    )
    assert selected["watchdog_count"] == 1
    assert selected["authority_receipt_sha256s"] == [
        second["authority_receipt_sha256"]
    ]
    assert os.path.lexists(second_paths["disarm_completion"])


def test_selective_watchdog_disarm_rejects_missing_and_duplicate_authorities(
    tmp_path,
):
    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    arguments = {
        "kind": "bitrix_foundation",
        "revision": "a" * 40,
        "full_canary_plan_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "interpreter": Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        "expires_at_unix": 1_800_000_000,
        "authority_sha256": "d" * 64,
        "plan_sha256": None,
        "credential_binding": None,
        "runner": runner,
        "state_root": tmp_path / "state",
        "systemd_root": tmp_path / "systemd",
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    armed = runtime.arm_capability_expiry_watchdog(**arguments)
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=arguments["state_root"],
        systemd_root=arguments["systemd_root"],
    )
    with pytest.raises(RuntimeError, match="authority is absent"):
        runtime.disarm_all_capability_expiry_watchdogs(
            runner=runner,
            state_root=arguments["state_root"],
            systemd_root=arguments["systemd_root"],
            require_root=False,
            expected_authority_receipt_sha256s=("f" * 64,),
        )
    with pytest.raises(ValueError, match="not unique"):
        runtime.disarm_all_capability_expiry_watchdogs(
            runner=runner,
            state_root=arguments["state_root"],
            systemd_root=arguments["systemd_root"],
            require_root=False,
            expected_authority_receipt_sha256s=(
                armed["authority_receipt_sha256"],
                armed["authority_receipt_sha256"],
            ),
        )
    with pytest.raises(ValueError, match="empty"):
        runtime.disarm_all_capability_expiry_watchdogs(
            runner=runner,
            state_root=arguments["state_root"],
            systemd_root=arguments["systemd_root"],
            require_root=False,
            expected_authority_receipt_sha256s=(),
        )
    assert os.path.lexists(paths["service_path"])
    assert os.path.lexists(paths["timer_path"])
    assert not os.path.lexists(paths["disarm_intent"])


def test_cleanup_watchdog_authorities_derive_from_exact_seven_bound_receipts(
    tmp_path,
):
    plan = _plan()

    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"
    common = {
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "release_artifact_sha256": plan.release_artifact_sha256,
        "interpreter": plan.interpreter,
        "runner": runner,
        "state_root": state_root,
        "systemd_root": systemd_root,
        "require_root": False,
        "now_unix": 1_799_999_000,
    }
    bitrix = runtime.arm_capability_expiry_watchdog(
        **common,
        kind="bitrix_foundation",
        expires_at_unix=1_800_001_000,
        authority_sha256="1" * 64,
        plan_sha256=None,
        credential_binding=None,
    )
    install_by_binding = {}
    retirements = {}
    expected = {bitrix["authority_receipt_sha256"]}
    for index, binding in enumerate(runtime.CAPABILITY_CREDENTIAL_BINDINGS, start=2):
        watchdog = runtime.arm_capability_expiry_watchdog(
            **common,
            kind="credential_lease",
            expires_at_unix=1_800_001_000 + index,
            authority_sha256=f"{index:x}" * 64,
            plan_sha256=plan.sha256,
            credential_binding=binding,
        )
        install = runtime._append_lease_artifact(
            tmp_path / "leases" / binding / "install.json",
            schema=runtime.CAPABILITY_LEASE_RECEIPT_SCHEMA,
            value={
                "operation": "install",
                "plan_sha256": plan.sha256,
                "full_canary_plan_sha256": plan.full_canary_plan_sha256,
                "credential_binding": binding,
                "expires_at_unix": watchdog["expires_at_unix"],
                "expiry_watchdog": watchdog,
            },
        )
        install_by_binding[binding] = install["receipt_sha256"]
        retirements[binding] = {
            "install_receipt_path": install["receipt_path"],
            "install_receipt_sha256": install["receipt_sha256"],
        }
        expected.add(watchdog["authority_receipt_sha256"])
    approval_retirement = {
        "lease_install_receipt_sha256_by_binding": install_by_binding,
        "bitrix_expiry_watchdog_authority_sha256": bitrix[
            "authority_receipt_sha256"
        ],
    }
    derived = runtime._expected_cleanup_expiry_watchdog_authorities(
        plan,
        approval_retirement=approval_retirement,
        lease_retirements=retirements,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    assert derived == tuple(sorted(expected))
    assert len(derived) == 7

    with pytest.raises(RuntimeError, match="Bitrix watchdog authority"):
        runtime._expected_cleanup_expiry_watchdog_authorities(
            plan,
            approval_retirement={
                **approval_retirement,
                "bitrix_expiry_watchdog_authority_sha256": next(
                    value for value in expected if value != bitrix["authority_receipt_sha256"]
                ),
            },
            lease_retirements=retirements,
            state_root=state_root,
            systemd_root=systemd_root,
        )
    with pytest.raises(RuntimeError, match="binding drifted"):
        runtime._expected_cleanup_expiry_watchdog_authorities(
            plan,
            approval_retirement=approval_retirement,
            lease_retirements={
                **retirements,
                runtime.CAPABILITY_CREDENTIAL_BINDINGS[0]: {
                    **retirements[runtime.CAPABILITY_CREDENTIAL_BINDINGS[0]],
                    "install_receipt_sha256": "f" * 64,
                },
            },
            state_root=state_root,
            systemd_root=systemd_root,
        )


def test_expiry_watchdog_retires_unjournaled_bitrix_pair_and_disarms(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", tmp_path / "no-plan.json")
    monkeypatch.setattr(
        runtime,
        "DEFAULT_APPROVAL_PATH",
        tmp_path / "etc/owner-approval.json",
    )
    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        interpreter=Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    private_path = tmp_path / "keys/private.pem"
    public_path = tmp_path / "trust/public.pem"
    runtime._stage_bitrix_receipt_key_pair(
        private_path=private_path,
        public_path=public_path,
        require_root=False,
    )
    credential_paths = {
        binding: tmp_path / "credentials" / binding
        for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    observations = []

    def observe_services(**_kwargs):
        observations.append(len(observations) + 1)
        return _watchdog_stopped_services()

    completion = runtime.run_capability_expiry_cleanup(
        armed["watchdog_id"],
        runner=runner,
        now_unix=2_000,
        state_root=state_root,
        systemd_root=systemd_root,
        credential_paths=credential_paths,
        private_key_path=private_path,
        public_key_path=public_path,
        key_bootstrap_root=tmp_path / "key-bootstraps",
        key_retirement_root=tmp_path / "key-retirements",
        service_observer=observe_services,
        require_root=False,
    )
    assert completion["ok"] is True
    assert completion["all_six_credentials_absent_readback"] is True
    assert completion["bitrix_pair_absent"] is True
    assert completion["approval_absent"] is True
    assert completion["approval_retirement"] == {
        "path": str(runtime.DEFAULT_APPROVAL_PATH),
        "removed": False,
        "absent": True,
    }
    assert completion["approval_retirement_sha256"] == runtime._sha256_json(
        completion["approval_retirement"]
    )
    assert (
        completion["bitrix_key_pair_cleanup"]["private_content_or_digest_recorded"]
        is False
    )
    assert not private_path.exists() and not public_path.exists()
    assert not list(systemd_root.glob("muncho-capability-canary-expiry-*"))
    atomic_no_replace = runtime._atomic_no_replace_file

    def interrupt_reconciliation(path, payload, **kwargs):
        if Path(path).name == "0000000000000002.json":
            temporary = Path(path).parent / kwargs["temporary_name"]
            temporary.write_bytes(payload)
            temporary.chmod(0o400)
            raise SystemExit("injected reconciliation SIGKILL window")
        return atomic_no_replace(path, payload, **kwargs)

    monkeypatch.setattr(
        runtime,
        "_atomic_no_replace_file",
        interrupt_reconciliation,
    )
    with pytest.raises(SystemExit, match="reconciliation SIGKILL window"):
        runtime.run_capability_expiry_cleanup(
            armed["watchdog_id"],
            runner=runner,
            now_unix=2_001,
            state_root=state_root,
            systemd_root=systemd_root,
            credential_paths=credential_paths,
            private_key_path=private_path,
            public_key_path=public_path,
            key_bootstrap_root=tmp_path / "key-bootstraps",
            key_retirement_root=tmp_path / "key-retirements",
            service_observer=observe_services,
            require_root=False,
        )
    monkeypatch.setattr(
        runtime,
        "_atomic_no_replace_file",
        atomic_no_replace,
    )
    repeated = runtime.run_capability_expiry_cleanup(
        armed["watchdog_id"],
        runner=runner,
        now_unix=2_001,
        state_root=state_root,
        systemd_root=systemd_root,
        credential_paths=credential_paths,
        private_key_path=private_path,
        public_key_path=public_path,
        key_bootstrap_root=tmp_path / "key-bootstraps",
        key_retirement_root=tmp_path / "key-retirements",
        service_observer=observe_services,
        require_root=False,
    )
    assert repeated == completion
    reconciliation_files = sorted(
        Path(
            runtime._expiry_watchdog_paths(
                armed["watchdog_id"],
                state_root=state_root,
                systemd_root=systemd_root,
            )["reconciliations"]
        ).glob("*.json")
    )
    assert len(reconciliation_files) == 3
    assert not list(reconciliation_files[0].parent.glob(".*.tmp"))
    assert observations == [1, 2, 3, 4, 5, 6]
    attempted = [
        argv[-1]
        for argv in calls
        if argv[:2] == (runtime.SYSTEMCTL, "stop")
        and argv[-1] in runtime.CAPABILITY_STOP_ORDER
    ]
    assert attempted == [
        *runtime.CAPABILITY_STOP_ORDER,
        *runtime.CAPABILITY_STOP_ORDER,
        *runtime.CAPABILITY_STOP_ORDER,
    ]


def test_expiry_watchdog_retires_and_binds_installed_owner_approval(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    full = _full_plan()
    plan_path = tmp_path / "etc/runtime-plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"published")
    approval_path = tmp_path / "etc/owner-approval.json"
    approval_path.write_bytes(b"installed")
    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_PATH", approval_path)
    monkeypatch.setattr(runtime, "load_capability_plan", lambda: plan)
    monkeypatch.setattr(runtime, "load_full_canary_plan", lambda: full)

    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"

    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision=plan.revision,
        full_canary_plan_sha256=full.sha256,
        release_artifact_sha256=full.release["artifact_sha256"],
        interpreter=plan.interpreter,
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    approval_receipt = {
        "path": str(approval_path),
        "approval_sha256": "e" * 64,
        "retirement_receipt_sha256": "f" * 64,
        "removed": True,
        "absent": True,
    }

    def retire_approval(actual_plan, actual_full):
        assert actual_plan is plan and actual_full is full
        approval_path.unlink()
        return approval_receipt

    monkeypatch.setattr(
        runtime,
        "_remove_installed_capability_approval",
        retire_approval,
    )
    monkeypatch.setattr(
        runtime,
        "retire_secret_leases_best_effort",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        runtime,
        "load_bitrix_key_bootstrap_receipt",
        lambda **_k: {},
    )
    monkeypatch.setattr(
        runtime,
        "retire_bitrix_foundation_key_pair",
        lambda *_a, **_k: {"receipt_sha256": "1" * 64},
    )
    credential_paths = {
        binding: tmp_path / "credentials" / binding
        for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    completion = runtime.run_capability_expiry_cleanup(
        armed["watchdog_id"],
        runner=runner,
        now_unix=2_000,
        state_root=state_root,
        systemd_root=systemd_root,
        credential_paths=credential_paths,
        private_key_path=tmp_path / "keys/private.pem",
        public_key_path=tmp_path / "trust/public.pem",
        key_bootstrap_root=tmp_path / "key-bootstraps",
        key_retirement_root=tmp_path / "key-retirements",
        service_observer=lambda **_kwargs: _watchdog_stopped_services(),
        require_root=False,
    )
    assert completion["ok"] is True
    assert completion["approval_absent"] is True
    assert completion["approval_retirement"] == approval_receipt
    assert completion["approval_retirement_sha256"] == runtime._sha256_json(
        approval_receipt
    )
    reconciliation = runtime._load_lease_artifact(
        sorted(
            Path(
                runtime._expiry_watchdog_paths(
                    armed["watchdog_id"],
                    state_root=state_root,
                    systemd_root=systemd_root,
                )["reconciliations"]
            ).glob("*.json")
        )[0],
        schema=runtime.CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
    )
    assert reconciliation["approval_absent"] is True
    assert reconciliation["approval_retirement_sha256"] == runtime._sha256_json(
        approval_receipt
    )


def test_expiry_cleanup_recovers_fixed_active_run_before_terminal_completion(
    monkeypatch,
    tmp_path,
):
    import gateway.canonical_capability_canary_producers as producers

    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", tmp_path / "no-plan.json")
    monkeypatch.setattr(
        runtime,
        "DEFAULT_APPROVAL_PATH",
        tmp_path / "etc/owner-approval.json",
    )
    catalog_path = tmp_path / "etc/probe-catalog.json"
    owner_grant_path = tmp_path / "etc/owner-grant.json"
    readiness_path = tmp_path / "run/producer-activation.json"
    for path, payload in (
        (catalog_path, b"catalog"),
        (owner_grant_path, b"grant"),
        (readiness_path, b"activation"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(producers, "DEFAULT_PROBE_CATALOG_PATH", catalog_path)
    monkeypatch.setattr(producers, "DEFAULT_OWNER_GRANT_PATH", owner_grant_path)
    monkeypatch.setattr(producers, "DEFAULT_READINESS_PATH", readiness_path)
    monkeypatch.setattr(
        producers,
        "validate_active_api_admission_retirement",
        lambda value: dict(value),
    )

    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"

    def runner(command):
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        interpreter=Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=state_root,
        systemd_root=systemd_root,
    )
    durable_retirement = tmp_path / "retirement/active-run.json"
    attempts = []

    def retire_active_run(*, retired_at_unix_ms):
        attempts.append(retired_at_unix_ms)
        if not durable_retirement.exists():
            for path in (catalog_path, owner_grant_path, readiness_path):
                path.unlink()
            durable_retirement.parent.mkdir(parents=True)
            durable_retirement.write_bytes(b"durable")
            raise SystemExit("injected death after durable fixed-artifact retirement")
        unsigned = {
            "schema": runtime.CAPABILITY_EXPIRY_ACTIVE_RUN_RETIREMENT_SCHEMA,
            "outcome": "retired_active_run",
            "run_id": "run-fixed-expiry",
            "release_sha": "a" * 40,
            "capability_plan_sha256": "e" * 64,
            "full_canary_plan_sha256": "b" * 64,
            "fixture_sha256": "f" * 64,
            "readiness_sha256": "1" * 64,
            "catalog_absent": True,
            "owner_grant_absent": True,
            "producer_activation_absent": True,
            "admission_retirement": {"receipt_sha256": "2" * 64},
            "fleet_retirement": {"receipt_sha256": "3" * 64},
            "observed_at_unix_ms": retired_at_unix_ms,
        }
        return {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}

    common = {
        "runner": runner,
        "now_unix": 2_000,
        "state_root": state_root,
        "systemd_root": systemd_root,
        "credential_paths": {
            binding: tmp_path / "credentials" / binding
            for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
        },
        "private_key_path": tmp_path / "keys/private.pem",
        "public_key_path": tmp_path / "trust/public.pem",
        "key_bootstrap_root": tmp_path / "key-bootstraps",
        "key_retirement_root": tmp_path / "key-retirements",
        "service_observer": lambda **_kwargs: _watchdog_stopped_services(),
        "active_run_retirer": retire_active_run,
        "require_root": False,
    }
    with pytest.raises(BaseExceptionGroup, match="will retry"):
        runtime.run_capability_expiry_cleanup(armed["watchdog_id"], **common)
    assert durable_retirement.is_file()
    assert not catalog_path.exists()
    assert not owner_grant_path.exists()
    assert not readiness_path.exists()
    assert not os.path.lexists(paths["completion"])
    assert os.path.lexists(paths["timer_path"])
    assert not os.path.lexists(paths["disarm_intent"])

    completion = runtime.run_capability_expiry_cleanup(
        armed["watchdog_id"],
        **common,
    )
    assert completion["active_run_retirement"]["outcome"] == "retired_active_run"
    assert completion["catalog_absent"] is True
    assert completion["owner_grant_absent"] is True
    assert completion["producer_activation_absent"] is True
    assert len(attempts) == 2
    assert os.path.lexists(paths["disarm_completion"])
    reconciliation = runtime._load_lease_artifact(
        sorted(Path(paths["reconciliations"]).glob("*.json"))[0],
        schema=runtime.CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
    )
    assert reconciliation["active_run_retirement_sha256"] == completion[
        "active_run_retirement_sha256"
    ]
    assert reconciliation["producer_activation_absent"] is True


def test_expiry_accepts_bound_published_run_that_never_reached_admission():
    import gateway.canonical_capability_canary_producers as producers

    fixture = {
        "run_id": "run-before-admission",
        "release_sha": "a" * 40,
        "capability_plan_sha256": "e" * 64,
        "full_canary_plan_sha256": "b" * 64,
    }
    retirement = producers._reconciled_published_run_without_admission(
        fixture,
        fixture_sha256="f" * 64,
        observed_at_unix_ms=2_000_000,
    )

    assert runtime._validate_expiry_active_run_retirement(
        retirement,
        authority={
            "plan_sha256": "e" * 64,
            "full_canary_plan_sha256": "b" * 64,
        },
        require_current_absence=False,
    ) == retirement
    assert retirement["outcome"] == (
        "reconciled_published_run_without_admission"
    )


def test_expiry_watchdog_failure_is_retryable_and_attempts_every_stop(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", tmp_path / "no-plan.json")
    calls = []

    def runner(command):
        calls.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    state_root = tmp_path / "state"
    systemd_root = tmp_path / "systemd"
    armed = runtime.arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision="a" * 40,
        full_canary_plan_sha256="b" * 64,
        release_artifact_sha256="c" * 64,
        interpreter=Path("/opt/muncho-canary-releases")
        / ("a" * 40)
        / "venv/bin/python",
        expires_at_unix=2_000,
        authority_sha256="d" * 64,
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=False,
        now_unix=1_000,
    )
    states = _watchdog_stopped_services()
    states[runtime.GATEWAY_UNIT_NAME]["ActiveState"] = "active"
    credential_paths = {
        binding: tmp_path / "credentials" / binding
        for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    with pytest.raises(BaseExceptionGroup, match="will retry"):
        runtime.run_capability_expiry_cleanup(
            armed["watchdog_id"],
            runner=runner,
            now_unix=2_000,
            state_root=state_root,
            systemd_root=systemd_root,
            credential_paths=credential_paths,
            private_key_path=tmp_path / "keys/private.pem",
            public_key_path=tmp_path / "trust/public.pem",
            key_bootstrap_root=tmp_path / "key-bootstraps",
            key_retirement_root=tmp_path / "key-retirements",
            service_observer=lambda **_kwargs: states,
            require_root=False,
        )
    paths = runtime._expiry_watchdog_paths(
        armed["watchdog_id"],
        state_root=state_root,
        systemd_root=systemd_root,
    )
    assert not os.path.lexists(paths["completion"])
    assert os.path.lexists(paths["timer_path"])
    attempted = [
        argv[-1]
        for argv in calls
        if argv[:2] == (runtime.SYSTEMCTL, "stop")
        and argv[-1] in runtime.CAPABILITY_STOP_ORDER
    ]
    assert attempted == list(runtime.CAPABILITY_STOP_ORDER)


def test_publish_plan_is_atomic_idempotent_and_tamper_evident(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    authority = _plan_publication_authority(plan)
    _patch_publication_host_identity_observations(monkeypatch, authority)
    plan_path = tmp_path / "etc/muncho/capability-canary/runtime-plan.json"
    receipt_root = tmp_path / "state/plan-publications"

    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT",
        receipt_root,
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "build_plan_from_publication_authority",
        lambda value, full: (
            plan
            if value == authority and full.sha256 == plan.full_canary_plan_sha256
            else (_ for _ in ()).throw(AssertionError("unexpected plan authority"))
        ),
    )
    monkeypatch.setattr(
        runtime,
        "validate_dedicated_canary_host",
        lambda _plan: {"dedicated": True},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_release_manifest",
        lambda _plan: {"release": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "runtime_dependency_manifest_preflight",
        lambda _plan: {"runtime": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "browser_executable_preflight",
        lambda _plan: {"browser": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "worker_executables_preflight",
        lambda _plan: {"worker": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "validate_bitrix_foundation_for_plan",
        lambda _plan, _full: {"bitrix_foundation": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "collect_service_state",
        lambda unit: {"unit": unit, "state": "stopped"},
    )
    monkeypatch.setattr(
        runtime,
        "evaluate_service_states",
        lambda _states, *, phase: {"all_stopped": phase == "stopped"},
    )
    lock_state = {"held": False, "entries": 0}

    @contextlib.contextmanager
    def lifecycle_lock():
        assert lock_state["held"] is False
        lock_state["held"] = True
        lock_state["entries"] += 1
        try:
            yield
        finally:
            lock_state["held"] = False

    monkeypatch.setattr(runtime, "_lifecycle_lock", lifecycle_lock)
    monkeypatch.setattr(
        runtime,
        "ensure_service_identities_create_only",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plan publication must remain read-only")
        ),
    )

    for name in (
        "validate_dedicated_canary_host",
        "_validate_release_manifest",
        "browser_host_identity_receipt",
        "execution_host_identity_receipt",
        "service_host_identity_receipt",
        "_observe_bitrix_foundation_identity",
        "runtime_dependency_manifest_preflight",
        "browser_executable_preflight",
        "worker_executables_preflight",
        "validate_bitrix_foundation_for_plan",
        "collect_service_state",
        "evaluate_service_states",
    ):
        operation = getattr(runtime, name)

        def require_lock(*args, _operation=operation, **kwargs):
            assert lock_state["held"] is True, name
            return _operation(*args, **kwargs)

        monkeypatch.setattr(runtime, name, require_lock)

    def publish(path: Path, payload: bytes) -> None:
        assert lock_state["held"] is True
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read(path: Path, *, maximum: int) -> bytes:
        item = path.stat()
        assert stat.S_IMODE(item.st_mode) == 0o400
        raw = path.read_bytes()
        assert len(raw) <= maximum
        return raw

    monkeypatch.setattr(runtime, "_atomic_publish_root_file", publish)
    monkeypatch.setattr(runtime, "_read_published_plan_file", read)

    def read_exact(path: Path, *, maximum: int):
        raw = read(path, maximum=maximum)
        item = path.stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    monkeypatch.setattr(runtime, "_read_exact_publication_file", read_exact)
    monkeypatch.setattr(
        runtime,
        "_require_same_file_identity",
        lambda path, expected: (
            None
            if (path.stat().st_dev, path.stat().st_ino)
            == (expected.st_dev, expected.st_ino)
            else (_ for _ in ()).throw(AssertionError("publication replaced"))
        ),
    )

    first = runtime.publish_capability_plan(authority)
    receipt_path = Path(first["receipt_path"])
    assert first["operation"] == "publish_capability_plan"
    assert first["full_canary_terminal_receipt"] == (plan.full_canary_terminal_receipt)
    assert first["full_canary_terminal_receipt_sha256"] == (
        plan.full_canary_terminal_receipt_sha256
    )
    assert first["original_full_canary_owner_approval_sha256"] == (
        plan.original_full_canary_owner_approval_sha256
    )
    assert first["plan_authoring_context"] == authority["plan_authoring_context"]
    assert (
        first["plan_authoring_context_receipt_sha256"]
        == authority["plan_authoring_context_receipt_sha256"]
    )
    assert plan_path.read_bytes() == runtime._canonical_bytes(plan.to_mapping())
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert runtime.publish_capability_plan(authority) == first
    assert lock_state == {"held": False, "entries": 2}

    def stable_read(path, **_kwargs):
        raw = Path(path).read_bytes()
        item = Path(path).stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    monkeypatch.setattr(runtime, "_read_stable_file", stable_read)
    assert runtime.load_bound_plan_publication_receipt(plan) == first

    receipt = json.loads(receipt_path.read_text())
    receipt["owner_subject_sha256"] = "f" * 64
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(runtime._canonical_bytes(receipt))
    receipt_path.chmod(0o400)
    with pytest.raises(RuntimeError, match="binding drifted"):
        runtime.load_bound_plan_publication_receipt(plan)
    with pytest.raises(RuntimeError, match="receipt drifted"):
        runtime.publish_capability_plan(authority)


def test_bound_reviewed_fixture_publication_loader_rejects_rehashed_tamper(
    monkeypatch,
    tmp_path,
):
    from gateway import canonical_capability_canary_live_driver as live
    from tests.gateway.test_canonical_capability_canary_live_driver import (
        _installed_contract,
    )

    monkeypatch.setattr(
        live,
        "load_bound_plan_publication_receipt",
        lambda current_plan: {
            "receipt_sha256": current_plan.plan_publication_receipt_sha256,
        },
    )
    installed = _installed_contract(tmp_path)
    fixture = installed["fixture"]
    plan = installed["plan"]
    full_plan = installed["full_plan"]
    receipt_path = live._fixture_publication_receipt_path(
        root=installed["publication_root"],
        plan_sha256=plan.sha256,
        run_id=fixture["run_id"],
        fixture_sha256=installed["fixture_sha256"],
    )
    monkeypatch.setattr(live, "DEFAULT_REVIEWED_FIXTURE", installed["source"])
    monkeypatch.setattr(
        live,
        "DEFAULT_FIXTURE_PUBLICATION_ROOT",
        installed["publication_root"],
    )
    monkeypatch.setattr(
        runtime,
        "load_bound_plan_publication_receipt",
        lambda _plan: {"receipt_sha256": fixture["plan_publication_receipt_sha256"]},
    )

    def stable_read(path, **_kwargs):
        path = Path(path)
        return path.read_bytes(), path.stat()

    monkeypatch.setattr(runtime, "_read_stable_file", stable_read)
    bound = runtime.load_bound_reviewed_fixture_publication(plan, full_plan)
    assert bound["ready"] is True
    assert bound["fixture_sha256"] == installed["fixture_sha256"]
    assert (
        bound["publication_receipt_sha256"]
        == json.loads(receipt_path.read_text())["receipt_sha256"]
    )

    tampered = json.loads(receipt_path.read_text())
    tampered["capability_plan_sha256"] = "f" * 64
    tampered_unsigned = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = runtime._sha256_bytes(
        runtime._canonical_bytes(tampered_unsigned)
    )
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(runtime._canonical_bytes(tampered))
    receipt_path.chmod(0o400)
    with pytest.raises(RuntimeError, match="publication binding drifted"):
        runtime.load_bound_reviewed_fixture_publication(plan, full_plan)


def test_publish_plan_reconciles_exact_plan_only_sigkill_orphan(monkeypatch, tmp_path):
    plan = _plan()
    authority = _plan_publication_authority(plan)
    _patch_publication_host_identity_observations(monkeypatch, authority)
    plan_path = tmp_path / "runtime-plan.json"
    plan_path.write_bytes(runtime._canonical_bytes(plan.to_mapping()))
    plan_path.chmod(0o400)

    monkeypatch.setattr(runtime, "DEFAULT_PLAN_PATH", plan_path)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT",
        tmp_path / "receipts",
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "build_plan_from_publication_authority",
        lambda value, full: (
            plan
            if value == authority and full.sha256 == plan.full_canary_plan_sha256
            else (_ for _ in ()).throw(AssertionError("unexpected plan authority"))
        ),
    )
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda _p: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _p: {})
    monkeypatch.setattr(runtime, "runtime_dependency_manifest_preflight", lambda _p: {})
    monkeypatch.setattr(runtime, "browser_executable_preflight", lambda _p: {})
    monkeypatch.setattr(runtime, "worker_executables_preflight", lambda _p: {})
    monkeypatch.setattr(
        runtime,
        "validate_bitrix_foundation_for_plan",
        lambda _plan, _full: {},
    )
    monkeypatch.setattr(runtime, "collect_service_state", lambda unit: {"unit": unit})
    monkeypatch.setattr(
        runtime,
        "evaluate_service_states",
        lambda _states, *, phase: {"stopped": phase == "stopped"},
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)

    def read_exact(path: Path, *, maximum: int):
        raw = path.read_bytes()
        assert len(raw) <= maximum
        item = path.stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    def publish(path: Path, payload: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(runtime, "_read_exact_publication_file", read_exact)
    monkeypatch.setattr(
        runtime,
        "_read_published_plan_file",
        lambda path, *, maximum: path.read_bytes(),
    )
    monkeypatch.setattr(runtime, "_atomic_publish_root_file", publish)
    monkeypatch.setattr(runtime, "_require_same_file_identity", lambda *_a: None)

    receipt = runtime.publish_capability_plan(authority)
    assert receipt["plan_sha256"] == plan.sha256
    assert Path(receipt["receipt_path"]).is_file()
    assert runtime.publish_capability_plan(authority) == receipt
    plan_path.unlink()
    with pytest.raises(RuntimeError, match="receipt exists without its plan"):
        runtime.publish_capability_plan(authority)


def test_gateway_config_rejects_any_unapproved_semantic_extension():
    original = yaml.safe_load(runtime.render_gateway_config(_plan()))
    mutations = (
        lambda value: value.__setitem__("semantic_dispatcher", {"enabled": True}),
        lambda value: value["plugins"]["enabled"].append("rewrite_plugin"),
        lambda value: value["plugins"].__setitem__("autoload", True),
        lambda value: value["hooks"].__setitem__(
            "pre_tool_call", {"command": "/tmp/rewrite"}
        ),
        lambda value: value.__setitem__(
            "auxiliary", {"goal_judge": {"provider": "auto"}}
        ),
        lambda value: value.__setitem__("goals", {"max_turns": 20}),
    )
    for mutate in mutations:
        candidate = json.loads(json.dumps(original))
        mutate(candidate)
        with pytest.raises(ValueError):
            runtime.validate_capability_gateway_config(candidate)


def _extension_surface(plan: runtime.CapabilityCanaryPlan):
    plugin_instance = object()
    callbacks = {}
    for hook_name in runtime.CAPABILITY_OBSERVER_HOOKS:

        def callback(_self, _hook_name=hook_name):
            return _hook_name

        callback.__name__ = hook_name
        callback.__module__ = "hermes_plugins.muncho_canary_evidence"
        callbacks[hook_name] = [
            callback.__get__(plugin_instance, type(plugin_instance))
        ]
    module = SimpleNamespace(
        __name__="hermes_plugins.muncho_canary_evidence",
        __file__=str(
            plan.release_root
            / "plugins"
            / runtime.CAPABILITY_OBSERVER_PLUGIN
            / "__init__.py"
        ),
        _PLUGIN=plugin_instance,
    )
    manifest = SimpleNamespace(
        name=runtime.CAPABILITY_OBSERVER_PLUGIN,
        key=runtime.CAPABILITY_OBSERVER_PLUGIN,
        kind="standalone",
        source="bundled",
        path=str(plan.release_root / "plugins" / runtime.CAPABILITY_OBSERVER_PLUGIN),
        provides_tools=[],
        provides_hooks=list(runtime.CAPABILITY_OBSERVER_HOOKS),
    )
    loaded = SimpleNamespace(
        manifest=manifest,
        module=module,
        tools_registered=[],
        hooks_registered=list(runtime.CAPABILITY_OBSERVER_HOOKS),
        middleware_registered=[],
        commands_registered=[],
        enabled=True,
        error=None,
        deferred=False,
    )
    manager = SimpleNamespace(
        _discovered=True,
        _isolated_allowlist=frozenset({runtime.CAPABILITY_OBSERVER_PLUGIN}),
        _isolated_discovery_failure=None,
        _plugins={runtime.CAPABILITY_OBSERVER_PLUGIN: loaded},
        _hooks=callbacks,
        _middleware={},
        _plugin_tool_names=set(),
        _plugin_platform_names=set(),
        _cli_commands={},
        _plugin_commands={},
        _context_engine=None,
        _plugin_skills={},
        _aux_tasks={},
        _slack_action_handlers=[],
        _cli_ref=None,
    )
    gateway_hooks = SimpleNamespace(_handlers={}, _loaded_hooks=[])
    return manager, gateway_hooks


def _provider_registry():
    profile = SimpleNamespace(
        name="openai-codex",
        aliases=("codex", "openai_codex"),
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        auth_type="oauth_external",
        env_vars=(),
    )
    return SimpleNamespace(
        _discovered=True,
        _discovery_error=None,
        _isolated_provider_allowlist=frozenset({"openai-codex"}),
        _isolated_discovery_validated=True,
        _REGISTRY={"openai-codex": profile},
        _ALIASES={"codex": "openai-codex", "openai_codex": "openai-codex"},
    )


def test_extension_surface_allows_only_the_sealed_observer():
    plan = _plan()
    manager, gateway_hooks = _extension_surface(plan)
    runtime.validate_capability_extension_surface(
        manager,
        gateway_hooks,
        _provider_registry(),
        plan=plan,
    )


@pytest.mark.parametrize("mode", ("api_only", "goal_only", "dual"))
def test_extension_surface_accepts_exact_bound_observer_multiplexer(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    manager, gateway_hooks = _extension_surface(plan)
    api = object.__new__(CanaryEvidencePlugin) if mode != "goal_only" else None
    goal = (
        object.__new__(GoalContinuationEvidencePlugin)
        if mode != "api_only"
        else None
    )
    if api is not None:
        api._session_id = None
    if goal is not None:
        goal._session_id = None
    observer = CanaryEvidenceHookMultiplexer(api, goal)
    for hook_name in runtime.CAPABILITY_OBSERVER_HOOKS:
        function = getattr(CanaryEvidenceHookMultiplexer, hook_name)
        monkeypatch.setattr(
            function,
            "__module__",
            "hermes_plugins.muncho_canary_evidence",
        )
        manager._hooks[hook_name] = [getattr(observer, hook_name)]
    manager._plugins[runtime.CAPABILITY_OBSERVER_PLUGIN].module._PLUGIN = observer

    runtime.validate_capability_extension_surface(
        manager,
        gateway_hooks,
        _provider_registry(),
        plan=plan,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "not_discovered",
        "discovery_error",
        "broad_allowlist",
        "not_validated",
        "extra_provider",
        "alias_drift",
        "profile_drift",
    ),
)
def test_capability_provider_registry_fails_closed_on_each_drift(mutation):
    provider = _provider_registry()
    if mutation == "not_discovered":
        provider._discovered = False
    elif mutation == "discovery_error":
        provider._discovery_error = RuntimeError("failed")
    elif mutation == "broad_allowlist":
        provider._isolated_provider_allowlist = None
    elif mutation == "not_validated":
        provider._isolated_discovery_validated = False
    elif mutation == "extra_provider":
        provider._REGISTRY["other"] = object()
    elif mutation == "alias_drift":
        provider._ALIASES["other"] = "openai-codex"
    else:
        provider._REGISTRY["openai-codex"].base_url = "https://example.invalid"

    with pytest.raises(RuntimeError, match="provider"):
        runtime.validate_capability_provider_registry(provider)


def test_gateway_runner_pins_capability_provider_before_later_lookup(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig

    calls = []
    monkeypatch.setattr(
        gateway_run,
        "_configure_gateway_provider_discovery",
        lambda isolated, production, capability: calls.append(
            (isolated, production, capability)
        ) or True,
    )

    runner = gateway_run.GatewayRunner(
        GatewayConfig(),
        require_capability_canary=True,
    )

    assert runner._require_capability_canary is True
    assert calls == [(False, False, True)]


@pytest.mark.parametrize(
    "field",
    (
        "model",
        "provider",
        "api_mode",
        "base_url",
        "reasoning_config",
        "_adaptive_reasoning_policy",
        "_tool_use_enforcement",
        "_task_completion_guidance",
        "_parallel_tool_call_guidance",
        "_background_review_enabled",
        "_fallback_chain",
        "_fallback_model",
        "request_overrides",
        "service_tier",
    ),
)
def test_capability_agent_attestation_rejects_each_policy_drift(field):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._require_capability_canary = True
    agent = SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning_config={"enabled": True, "effort": "high"},
        _adaptive_reasoning_policy={"enabled": True, "max_effort": "max"},
        _tool_use_enforcement=True,
        _task_completion_guidance=True,
        _parallel_tool_call_guidance=True,
        _background_review_enabled=False,
        _fallback_chain=[],
        _fallback_model=None,
        request_overrides={},
        service_tier=None,
    )
    setattr(
        agent,
        field,
        [dict(provider="other", model="other")]
        if field == "_fallback_chain"
        else (
            {"enabled": False}
            if field in {"reasoning_config", "_adaptive_reasoning_policy"}
            else {"provider": "other", "model": "other"}
            if field == "_fallback_model"
            else {"reasoning": {"effort": "none"}}
            if field == "request_overrides"
            else "priority"
            if field == "service_tier"
            else True
            if field == "_background_review_enabled"
            else False
            if field.startswith("_")
            else "drifted"
        ),
    )

    with pytest.raises(RuntimeError, match="agent policy"):
        runner._attest_capability_agent_policy(agent)


def test_capability_runtime_ignores_persisted_session_route_and_requires_exact_runtime(
    monkeypatch,
):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._require_production_model_sovereignty = False
    runner._require_capability_canary = True
    runner._session_model_overrides = {
        "session": {"model": "stale", "provider": "stale"}
    }
    runner.config = None
    runner._last_resolved_model = {}
    runner._rehydrate_session_model_override = lambda _key: (_ for _ in ()).throw(
        AssertionError("sealed route must not rehydrate an override")
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda _config: "gpt-5.6-sol",
    )
    exact_runtime = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: dict(exact_runtime),
    )

    model, resolved = runner._resolve_session_agent_runtime(
        session_key="session",
        user_config={},
    )

    assert model == "gpt-5.6-sol"
    assert resolved == exact_runtime

    for field, drifted in (
        ("provider", "other"),
        ("base_url", "https://example.invalid"),
        ("api_mode", "chat_completions"),
    ):
        candidate = {**exact_runtime, field: drifted}
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda candidate=candidate: dict(candidate),
        )
        with pytest.raises(RuntimeError, match="model route"):
            runner._resolve_session_agent_runtime(
                session_key="session",
                user_config={},
            )


def test_capability_runtime_ignores_session_reasoning_disable(monkeypatch):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._require_production_model_sovereignty = False
    runner._require_capability_canary = True
    runner._session_reasoning_overrides = {
        "session": {"enabled": False},
    }
    loaded = []
    runner._load_reasoning_config = lambda: loaded.append(True) or {
        "enabled": True,
        "effort": "high",
    }

    assert runner._resolve_session_reasoning_config(session_key="session") == {
        "enabled": True,
        "effort": "high",
    }
    assert loaded == [True]


def test_native_model_call_attests_final_per_turn_policy_after_mutations():
    import gateway.run as gateway_run

    events = []
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._require_capability_canary = True

    class Agent:
        model = "gpt-5.6-sol"
        provider = "openai-codex"
        api_mode = "codex_responses"
        base_url = "https://chatgpt.com/backend-api/codex"
        reasoning_config = {"enabled": True, "effort": "high"}
        _adaptive_reasoning_policy = {"enabled": True, "max_effort": "max"}
        _tool_use_enforcement = True
        _task_completion_guidance = True
        _parallel_tool_call_guidance = True
        _background_review_enabled = False
        _fallback_chain = []
        _fallback_model = None
        request_overrides = {}
        service_tier = None

        def run_conversation(self, *args, **kwargs):
            events.append(("model", args, kwargs))
            return {"final_response": "ok"}

    agent = Agent()
    original_attest = runner._attest_capability_agent_policy

    def record_attestation(candidate):
        events.append(
            (
                "attest",
                dict(candidate.reasoning_config),
                dict(candidate._adaptive_reasoning_policy),
                dict(candidate.request_overrides),
            )
        )
        original_attest(candidate)

    runner._attest_capability_agent_policy = record_attestation

    result = runner._run_conversation_with_capability_attestation(
        agent,
        "complex task",
        task_id="session",
    )

    assert result == {"final_response": "ok"}
    assert events == [
        (
            "attest",
            {"enabled": True, "effort": "high"},
            {"enabled": True, "max_effort": "max"},
            {},
        ),
        ("model", ("complex task",), {"task_id": "session"}),
    ]

    agent.request_overrides = {"reasoning": {"effort": "none"}}
    with pytest.raises(RuntimeError, match="agent policy"):
        runner._run_conversation_with_capability_attestation(agent, "blocked")
    assert [event[0] for event in events] == ["attest", "model", "attest"]


def test_capability_api_agent_uses_only_the_sealed_model_route(monkeypatch):
    import gateway.run as gateway_run
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    config = yaml.safe_load(runtime.render_gateway_config(_plan()))
    created = []

    class ExactAgent:
        def __init__(self, **kwargs):
            created.append(dict(kwargs))
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.api_mode = kwargs["api_mode"]
            self.base_url = kwargs["base_url"]
            self.reasoning_config = kwargs["reasoning_config"]
            self._adaptive_reasoning_policy = {
                "enabled": True,
                "max_effort": "max",
            }
            self._tool_use_enforcement = True
            self._task_completion_guidance = True
            self._parallel_tool_call_guidance = True
            self._background_review_enabled = False
            self._fallback_chain = []
            self._fallback_model = None
            self.request_overrides = kwargs.get("request_overrides") or {}
            self.service_tier = kwargs.get("service_tier")

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True),
        run_admission_callback=lambda _session, _epoch: {},
        require_capability_canary=True,
    )
    monkeypatch.setattr(
        adapter,
        "_session_model_override_for",
        lambda _key: (_ for _ in ()).throw(
            AssertionError("sealed API route must not read session overrides")
        ),
    )
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "api_key": "opaque-test-credential",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda: "gpt-5.6-sol",
    )
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(
        gateway_run,
        "_isolated_gateway_runtime_active",
        lambda: False,
    )
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_load_reasoning_config",
        staticmethod(lambda: {"enabled": True, "effort": "high"}),
    )
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_load_fallback_model",
        staticmethod(
            lambda: (_ for _ in ()).throw(
                AssertionError("sealed API route must not load fallbacks")
            )
        ),
    )
    monkeypatch.setattr("run_agent.AIAgent", ExactAgent)

    agent = adapter._create_agent(reuse_cached_agent=False)

    assert agent.model == "gpt-5.6-sol"
    assert created[0]["fallback_model"] is None
    assert created[0]["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }
    with pytest.raises(RuntimeError, match="request model routes"):
        adapter._create_agent(
            route={"model": "other", "provider": "other"},
            reuse_cached_agent=False,
        )


@pytest.mark.parametrize(
    ("model", "runtime_kwargs"),
    (
        (
            "other",
            {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
            },
        ),
        (
            "gpt-5.6-sol",
            {
                "provider": "other",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
            },
        ),
        (
            "gpt-5.6-sol",
            {
                "provider": "openai-codex",
                "base_url": "https://example.invalid",
                "api_mode": "codex_responses",
            },
        ),
        (
            "gpt-5.6-sol",
            {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "chat_completions",
            },
        ),
        (
            "gpt-5.6-sol",
            {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
                "command": "external-provider",
            },
        ),
        (
            "gpt-5.6-sol",
            {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_mode": "codex_responses",
                "args": ["--semantic-router"],
            },
        ),
    ),
)
def test_capability_model_runtime_route_rejects_each_transport_drift(
    model,
    runtime_kwargs,
):
    with pytest.raises(RuntimeError, match="model route"):
        runtime.validate_capability_model_runtime_route(model, runtime_kwargs)


@pytest.mark.parametrize(
    "mutation",
    (
        "general_discovery",
        "extra_plugin",
        "middleware",
        "tool",
        "command",
        "context_engine",
        "auxiliary_task",
        "callback_substitution",
        "gateway_hook",
        "module_substitution",
    ),
)
def test_extension_surface_fails_closed_on_behavior_changing_state(mutation):
    plan = _plan()
    manager, gateway_hooks = _extension_surface(plan)
    if mutation == "general_discovery":
        manager._isolated_allowlist = None
    elif mutation == "extra_plugin":
        manager._plugins["extra"] = manager._plugins[runtime.CAPABILITY_OBSERVER_PLUGIN]
    elif mutation == "middleware":
        manager._middleware["llm_request"] = [lambda **_kwargs: {}]
    elif mutation == "tool":
        manager._plugin_tool_names.add("semantic_router")
    elif mutation == "command":
        manager._plugin_commands["dispatch"] = {"handler": lambda: None}
    elif mutation == "context_engine":
        manager._context_engine = object()
    elif mutation == "auxiliary_task":
        manager._aux_tasks["goal_judge"] = {"provider": "auto"}
    elif mutation == "callback_substitution":
        manager._hooks["pre_api_request"] = [lambda **_kwargs: None]
    elif mutation == "gateway_hook":
        gateway_hooks._handlers["agent:start"] = [lambda *_args: None]
    else:
        manager._plugins[runtime.CAPABILITY_OBSERVER_PLUGIN].module.__file__ = str(
            plan.release_root / "substituted.py"
        )
    with pytest.raises(RuntimeError):
        runtime.validate_capability_extension_surface(
            manager,
            gateway_hooks,
            _provider_registry(),
            plan=plan,
        )


def test_gateway_and_mac_units_keep_secrets_outside_gateway():
    plan = _plan()
    gateway = runtime.render_gateway_unit(plan)
    mac = runtime.render_mac_ops_unit(plan)
    browser = runtime.render_browser_unit(plan)
    worker_socket = runtime.render_worker_socket_unit(plan)
    worker_service = runtime.render_worker_service_unit(plan)

    assert "-m gateway.run" in gateway
    assert "--require-capability-canary" in gateway
    assert "127.0.0.1" not in gateway  # API host remains sealed config, not argv.
    assert "EnvironmentFile=" not in gateway
    assert "DISCORD_BOT_TOKEN=" not in gateway
    assert "GITLAB_TOKEN=" not in gateway
    assert "UnsetEnvironment=" in gateway
    assert "DISCORD_TOKEN" in gateway
    assert "GITLAB_BASE_URL" in gateway
    assert "GITLAB_TOKEN" in gateway
    assert f"InaccessiblePaths={runtime.DEFAULT_MAC_OPS_CREDENTIAL_DIR}" in gateway
    assert f"BindReadOnlyPaths={runtime.DEFAULT_GATEWAY_AUTH_STORE}" in gateway
    assert f"InaccessiblePaths={runtime.DEFAULT_GATEWAY_AUTH_STORE}" not in gateway
    assert "GATEWAY_RELAY_PLATFORMS=discord" in gateway
    assert "TERMINAL_ENV=isolated_worker" in gateway
    assert f"TERMINAL_ISOLATED_WORKER_SOCKET={runtime.DEFAULT_WORKER_SOCKET}" in gateway
    assert runtime.DEFAULT_WORKER_CLIENT_GROUP in gateway
    assert runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME in gateway
    assert runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME in gateway
    assert "AGENT_BROWSER_EXECUTABLE_PATH=" not in gateway
    assert f"InaccessiblePaths={plan.browser_executable}" in gateway
    assert f"AssertPathExists={plan.browser_executable}" not in gateway
    assert f"ReadOnlyPaths={runtime.DEFAULT_BROWSER_CONFIG}" in gateway
    assert f"ReadOnlyPaths={runtime.DEFAULT_BROWSER_SOCKET.parent}" in gateway
    assert "/usr/bin/chromium" not in gateway
    assert "/srv/" not in gateway
    assert "docker" not in gateway.lower()
    assert "BROWSER_CDP_URL" not in gateway
    assert "remote-debugging" not in gateway
    assert "9222" not in gateway
    assert f"BindsTo={runtime.WRITER_UNIT_NAME}" in gateway
    assert "gateway.mac_ops_edge_service" in mac
    assert f"ReadOnlyPaths={runtime.DEFAULT_MAC_OPS_CREDENTIAL}" in mac
    assert f"User={plan.identities.mac_ops_user}" in mac
    assert "UnsetEnvironment=" in mac
    assert "OPENAI_API_KEY" in mac
    assert f"User={plan.identities.browser_user}" in browser
    assert f"Group={plan.identities.browser_group}" in browser
    assert f"# PrincipalUID={plan.identities.browser_uid}" in browser
    assert f"# PrincipalGID={plan.identities.browser_gid}" in browser
    assert plan.identities.gateway_user not in browser
    assert "--no-sandbox" not in browser
    assert "RestrictNamespaces" not in browser
    assert "Type=notify" in browser
    assert "NotifyAccess=main" in browser
    assert "gateway.browser_controller" in browser
    assert "remote-debugging" not in browser
    assert "9222" not in browser
    assert f"ListenStream={runtime.DEFAULT_WORKER_SOCKET}" in worker_socket
    assert "PrivateNetwork=yes" in worker_service
    assert "TemporaryFileSystem=" in worker_service
    assert "docker" not in worker_service.lower()


def test_connector_plan_unit_and_six_lease_bindings_are_exact():
    plan = _plan()
    config = json.loads(runtime.render_connector_config(plan))
    unit = runtime.render_connector_unit(plan)

    assert config["service"]["canary_history_reader"] == {
        "service_unit": runtime.CANARY_HISTORY_READER_SERVICE_UNIT,
        "service_user": runtime.CANARY_HISTORY_READER_SERVICE_USER,
        "requester_user_id": runtime.CANARY_REQUESTER_USER_ID,
    }
    assert config["discord"]["allowed_guild_ids"] == ["1282725267068157972"]
    assert config["discord"]["allowed_channel_ids"] == [
        runtime.PRODUCTION_CANARY_PUBLIC_CHANNEL_ID
    ]
    assert config["discord"]["allowed_user_ids"] == ["1279454038731264061"]
    assert config["discord"]["reviewed_cron_history_targets"] == {}
    assert config["discord"]["allow_bot_authors"] is False
    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "Restart=no" in unit
    assert "RuntimeMaxSec=900s" in unit
    assert "DISCORD_BOT_TOKEN=" not in unit
    assert set(plan.to_mapping()["credential_bindings"]) == set(
        runtime.CAPABILITY_CREDENTIAL_BINDINGS
    )
    assert tuple(runtime.CAPABILITY_START_ORDER) == (
        runtime.PHASE_B_READINESS_UNIT_NAME,
        runtime.EDGE_UNIT_NAME,
        runtime.DEFAULT_DISCORD_CONNECTOR_UNIT,
        runtime.MAC_OPS_UNIT_NAME,
        runtime.DEFAULT_WORKER_SOCKET_UNIT_NAME,
        runtime.DEFAULT_WORKER_SERVICE_UNIT_NAME,
        runtime.DEFAULT_BROWSER_UNIT_NAME,
        runtime.WRITER_UNIT_NAME,
        runtime.BITRIX_OPERATIONAL_EDGE_UNIT,
        *(
            runtime.CAPABILITY_PRODUCER_SERVICE_UNITS[role]
            for role in runtime.CAPABILITY_PRODUCER_ROLES
        ),
        runtime.GATEWAY_UNIT_NAME,
    )
    assert tuple(runtime.CAPABILITY_STOP_ORDER) == (
        *runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER,
        runtime.CAPABILITY_OBSERVER_UNIT,
    )
    assert runtime.CAPABILITY_OBSERVER_UNIT not in (
        runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER
    )


@pytest.mark.parametrize("failure_index", range(len(runtime.CAPABILITY_STOP_ORDER)))
def test_cleanup_attempts_all_stops_after_each_injected_failure(
    failure_index,
):
    attempted = []

    def stop(unit):
        attempted.append(unit)
        if len(attempted) - 1 == failure_index:
            raise RuntimeError("injected stop failure")

    stopped, errors = runtime._attempt_capability_stop_order(stop)

    assert attempted == list(runtime.CAPABILITY_STOP_ORDER)
    assert len(errors) == 1
    assert len(stopped) == len(runtime.CAPABILITY_STOP_ORDER) - 1


@pytest.mark.parametrize(
    (
        "fixture_expiry",
        "earliest_lease_expiry",
        "watchdog_expiry",
        "expected_not_after",
        "expected_chain_complete",
    ),
    (
        (2_500, 2_500, 2_500, 1_895, True),
        (1_500, 2_500, 2_500, 1_495, True),
        (2_500, 1_400, 2_500, 1_395, True),
        (2_500, 2_500, 1_300, 1_295, True),
        (1_035, 2_500, 2_500, 1_030, False),
    ),
)
def test_stopped_preflight_computes_exact_fresh_approval_bound(
    monkeypatch,
    fixture_expiry,
    earliest_lease_expiry,
    watchdog_expiry,
    expected_not_after,
    expected_chain_complete,
):
    plan = _plan()
    full = _full_plan()
    observed_at = 1_000
    fixture = {
        "fixture_sha256": "b" * 64,
        "publication_receipt_sha256": "c" * 64,
        "fixture_valid_until_unix_ms": fixture_expiry * 1_000 + 999,
    }
    bitrix = {
        "foundation_receipt_sha256": "d" * 64,
        "expiry_watchdog_authority_sha256": "e" * 64,
        "expiry_watchdog_expires_at_unix": watchdog_expiry,
        "expires_at_unix": watchdog_expiry,
    }
    lease_expiry_by_binding = {
        binding: 2_500 for binding in runtime.CAPABILITY_CREDENTIAL_BINDINGS
    }
    lease_expiry_by_binding[runtime.CAPABILITY_CREDENTIAL_BINDINGS[0]] = (
        earliest_lease_expiry
    )
    monkeypatch.setattr(
        runtime,
        "validate_dedicated_canary_host",
        lambda *_args, **_kwargs: {"dedicated": True},
    )
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _full: {})
    monkeypatch.setattr(
        runtime,
        "load_bound_plan_publication_receipt",
        lambda _plan: {"receipt_sha256": "a" * 64},
    )
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "load_service_identity_foundation_receipt",
        lambda *_args, **_kwargs: {"receipt_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "load_bound_reviewed_fixture_publication",
        lambda _plan, _full: fixture,
    )
    monkeypatch.setattr(
        runtime,
        "validate_bitrix_foundation_for_plan",
        lambda *_args, **_kwargs: bitrix,
    )
    monkeypatch.setattr(
        runtime,
        "_producer_foundation_preflight",
        lambda *_args, **_kwargs: {
            "ready": True,
            "producer_identity_foundation_receipt_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        producer_units,
        "producer_host_identity_receipt",
        lambda *_args, **_kwargs: _producer_identity_observation(plan),
    )
    monkeypatch.setattr(
        runtime,
        "load_service_identity_foundation_receipt",
        lambda *_args, **_kwargs: {"receipt_sha256": "9" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "service_host_identity_receipt",
        lambda *_args, **_kwargs: {"receipt_sha256": "8" * 64},
    )
    monkeypatch.setattr(runtime, "_overlay_targets_are_absent", lambda *_a: True)
    monkeypatch.setattr(
        runtime, "runtime_dependency_manifest_preflight", lambda _plan: {}
    )
    monkeypatch.setattr(runtime, "browser_executable_preflight", lambda _plan: {})
    monkeypatch.setattr(runtime, "worker_executables_preflight", lambda _plan: {})
    monkeypatch.setattr(runtime, "worker_systemd252_preflight", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runtime, "execution_host_identity_receipt", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        runtime,
        "browser_host_identity_receipt",
        lambda *_a, **_k: {
            "browser": {"state": "absent_create_only_slot"},
            "receipt_sha256": "f" * 64,
        },
    )
    monkeypatch.setattr(runtime, "browser_userns_preflight", lambda: {})

    def active_lease(_plan, *, kind, **_kwargs):
        binding = runtime._CREDENTIAL_BINDING_BY_KIND[kind]
        return {
            "install_receipt_sha256": hashlib.sha256(kind.encode()).hexdigest(),
            "expires_at_unix": lease_expiry_by_binding[binding],
        }

    monkeypatch.setattr(runtime, "_active_lease_receipt", active_lease)
    monkeypatch.setattr(
        runtime,
        "_capability_services",
        lambda **_kwargs: _watchdog_stopped_services(),
    )

    if expected_chain_complete:
        preflight = runtime.collect_capability_preflight(
            plan,
            full,
            phase="stopped",
            approval_window_started_at_unix=observed_at,
        )
    else:
        with pytest.raises(runtime.CapabilityCanaryPreflightError) as caught:
            runtime.collect_capability_preflight(
                plan,
                full,
                phase="stopped",
                approval_window_started_at_unix=observed_at,
            )
        preflight = caught.value.report

    assert preflight["approval_not_after_unix"] == expected_not_after
    assert (
        preflight["approval_not_after_unix"]
        == min(
            observed_at + 900,
            fixture["fixture_valid_until_unix_ms"] // 1_000,
            min(lease_expiry_by_binding.values()),
            watchdog_expiry,
        )
        - 5
    )
    assert (
        preflight["checks"]["approval.fresh_evidence_chain"] is expected_chain_complete
    )
    assert preflight["full_canary_terminal_receipt"] == (
        plan.full_canary_terminal_receipt
    )
    assert preflight["full_canary_terminal_receipt_sha256"] == (
        plan.full_canary_terminal_receipt_sha256
    )
    assert preflight["original_full_canary_owner_approval_sha256"] == (
        plan.original_full_canary_owner_approval_sha256
    )


def test_capability_approval_rejects_terminal_and_plan_mismatch():
    plan = _plan()
    approval = _capability_approval(
        plan,
        approved_at_unix=100,
        expires_at_unix=200,
    )
    wrong_plan = dict(approval.value)
    wrong_plan["plan_sha256"] = "f" * 64
    mismatched = runtime.CapabilityCanaryOwnerApproval.from_mapping(wrong_plan)
    with pytest.raises(PermissionError, match="does not authorize"):
        mismatched.require(
            plan_sha256=plan.sha256,
            full_canary_plan_sha256=plan.full_canary_plan_sha256,
            now_unix=150,
        )

    tampered_terminal = json.loads(json.dumps(approval.value))
    tampered_terminal["full_canary_terminal_receipt"]["services_stopped"] = False
    with pytest.raises(ValueError, match="terminal receipt"):
        runtime.CapabilityCanaryOwnerApproval.from_mapping(tampered_terminal)


def test_runtime_start_ignores_any_stale_full_canary_approval(monkeypatch):
    plan = _plan()
    full = _full_plan()
    fresh = _capability_approval(
        plan,
        approved_at_unix=100,
        expires_at_unix=200,
    )
    calls: list[tuple[object, ...]] = []

    class Lifecycle:
        def __init__(self, actual_plan, actual_full):
            calls.append(("init", actual_plan, actual_full))

        def start(self, approval):
            calls.append(("start", approval))
            return {"ok": True}

    monkeypatch.setattr(runtime, "load_capability_plan", lambda: plan)
    monkeypatch.setattr(runtime, "load_full_canary_plan", lambda: full)
    monkeypatch.setattr(runtime, "validate_plan_against_full", lambda *_a: None)
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _full: {})
    monkeypatch.setattr(runtime, "load_capability_approval", lambda: fresh)
    monkeypatch.setattr(runtime, "CapabilityCanaryLifecycle", Lifecycle)
    monkeypatch.setattr(runtime, "_emit", lambda value: calls.append(("emit", value)))
    monkeypatch.setattr(
        runtime,
        "load_full_canary_approval",
        lambda: (_ for _ in ()).throw(AssertionError("stale full approval loaded")),
        raising=False,
    )

    assert runtime.main(["start"]) == 0
    assert calls == [
        ("init", plan, full),
        ("start", fresh),
        ("emit", {"ok": True}),
    ]


def test_lifecycle_receipt_carries_complete_terminal_chain(tmp_path, monkeypatch):
    plan = _plan()
    monkeypatch.setattr(runtime, "load_full_canary_plan", _full_plan)
    monkeypatch.setattr(
        runtime,
        "load_service_identity_foundation_receipt",
        lambda *_args, **_kwargs: {"receipt_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "DEFAULT_LIFECYCLE_RECEIPT_ROOT",
        tmp_path / "lifecycle",
    )

    def ensure_root_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)

    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        ensure_root_directory,
    )

    def write_exclusive(path, payload, *, mode):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    monkeypatch.setattr(
        runtime,
        "load_bound_plan_publication_receipt",
        lambda _plan: {"receipt_sha256": "a" * 64},
    )

    receipt = runtime._write_lifecycle_receipt(
        plan,
        stage="started",
        value={"operation": "start"},
    )

    assert receipt["full_canary_terminal_receipt"] == (
        plan.full_canary_terminal_receipt
    )
    assert receipt["full_canary_terminal_receipt_sha256"] == (
        plan.full_canary_terminal_receipt_sha256
    )
    assert receipt["original_full_canary_owner_approval_sha256"] == (
        plan.original_full_canary_owner_approval_sha256
    )
    assert receipt["plan_publication_receipt_sha256"] == "a" * 64
    assert receipt["service_identity_foundation_receipt_sha256"] == "b" * 64


def test_deferred_lifecycle_state_recovers_from_receipts_and_rejects_conflicts(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    approval = _capability_approval(
        plan,
        approved_at_unix=1_000,
        expires_at_unix=1_600,
    )
    root = tmp_path / "lifecycle"
    monkeypatch.setattr(runtime, "DEFAULT_LIFECYCLE_RECEIPT_ROOT", root)
    sequence = {"value": 0}

    def write(stage, value):
        sequence["value"] += 1
        directory = root / plan.revision / plan.sha256 / stage
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"{sequence['value']}-123-{'a' * 31}{sequence['value']:x}.json"
        )
        unsigned = {
            **value,
            "schema": runtime.CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA,
            "stage": stage,
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "full_canary_terminal_receipt": plan.full_canary_terminal_receipt,
            "full_canary_terminal_receipt_sha256": (
                plan.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                plan.original_full_canary_owner_approval_sha256
            ),
            "plan_publication_receipt_sha256": "1" * 64,
            "service_identity_foundation_receipt_sha256": "2" * 64,
            "receipt_path": str(path),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}
        path.write_bytes(runtime._canonical_bytes(receipt))
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(0o400)
        return receipt

    mapping_fields = {
        field: {}
        for field in (
            "installed_artifacts",
            "connector_state",
            "phase_b_current_readiness",
            "phase_b_full_canary_anchor",
            "writer_runtime_readiness",
            "gateway_runtime_readiness",
            "observer_config",
            "browser_identity_foundation",
            "browser_principal_smoke",
            "execution_identity_foundation",
            "worker_mountpoint",
            "execution_readiness",
            "routeback_bot_identity",
            "producer_foundation",
        )
    }
    core = write(
        runtime.CAPABILITY_GATEWAY_CORE_READY_STAGE,
        {
            "operation": "start_core_before_api_admission",
            "owner_approval_sha256": approval.sha256,
            "full_canary_stopped_preflight_sha256": "3" * 64,
            "stopped_preflight_sha256": "4" * 64,
            "core_start_order": list(runtime.CAPABILITY_DEFERRED_CORE_START_ORDER),
            "producer_units_started": False,
            "api_admission_pending": True,
            **mapping_fields,
        },
    )
    assert runtime._load_deferred_lifecycle_state(plan, approval) == (core, None)

    terminal = write(
        runtime.CAPABILITY_RUNTIME_PENDING_ACK_STAGE,
        {
            "operation": "runtime_live_pending_gateway_commit_ack",
            "owner_approval_sha256": approval.sha256,
            "core_start_receipt_sha256": core["receipt_sha256"],
        },
    )
    assert runtime._load_deferred_lifecycle_state(plan, approval) == (core, terminal)

    write(
        runtime.CAPABILITY_RUNTIME_PENDING_ACK_STAGE,
        {
            "operation": "runtime_live_pending_gateway_commit_ack",
            "owner_approval_sha256": approval.sha256,
            "core_start_receipt_sha256": core["receipt_sha256"],
        },
    )
    with pytest.raises(RuntimeError, match="duplicate terminal"):
        runtime._load_deferred_lifecycle_state(plan, approval)


def test_deferred_lifecycle_dirty_failure_remains_explicit(tmp_path, monkeypatch):
    plan = _plan()
    approval = _capability_approval(
        plan,
        approved_at_unix=1_000,
        expires_at_unix=1_600,
    )
    root = tmp_path / "lifecycle"
    monkeypatch.setattr(runtime, "DEFAULT_LIFECYCLE_RECEIPT_ROOT", root)

    def write(stage, name, value):
        directory = root / plan.revision / plan.sha256 / stage
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}-123-{'b' * 32}.json"
        unsigned = {
            **value,
            "schema": runtime.CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA,
            "stage": stage,
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "full_canary_terminal_receipt": plan.full_canary_terminal_receipt,
            "full_canary_terminal_receipt_sha256": (
                plan.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                plan.original_full_canary_owner_approval_sha256
            ),
            "receipt_path": str(path),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}
        path.write_bytes(runtime._canonical_bytes(receipt))
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(0o400)
        return receipt

    core = write(
        runtime.CAPABILITY_GATEWAY_CORE_READY_STAGE,
        "1",
        {
            "operation": "start_core_before_api_admission",
            "owner_approval_sha256": approval.sha256,
            "full_canary_stopped_preflight_sha256": "3" * 64,
            "stopped_preflight_sha256": "4" * 64,
            "core_start_order": list(runtime.CAPABILITY_DEFERRED_CORE_START_ORDER),
            "producer_units_started": False,
            "api_admission_pending": True,
            **{
                field: {}
                for field in (
                    "installed_artifacts",
                    "connector_state",
                    "phase_b_current_readiness",
                    "phase_b_full_canary_anchor",
                    "writer_runtime_readiness",
                    "gateway_runtime_readiness",
                    "observer_config",
                    "browser_identity_foundation",
                    "browser_principal_smoke",
                    "execution_identity_foundation",
                    "worker_mountpoint",
                    "execution_readiness",
                    "routeback_bot_identity",
                    "producer_foundation",
                )
            },
        },
    )
    write(
        "failure",
        "2",
        {
            "operation": "complete_start_after_api_admission",
            "owner_approval_sha256": approval.sha256,
            "core_start_receipt_sha256": core["receipt_sha256"],
            "cleanup_complete": False,
        },
    )
    with pytest.raises(RuntimeError, match="requires exact reconciliation"):
        runtime._load_deferred_lifecycle_state(plan, approval)


def test_deferred_phase_two_resumes_an_already_started_generation(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    approval = _capability_approval(
        plan,
        approved_at_unix=1_000,
        expires_at_unix=1_600,
    )
    core_path = tmp_path / "core.json"
    core_unsigned = {
        "stage": runtime.CAPABILITY_GATEWAY_CORE_READY_STAGE,
        "operation": "start_core_before_api_admission",
        "plan_sha256": plan.sha256,
        "owner_approval_sha256": approval.sha256,
        "producer_units_started": False,
        "api_admission_pending": True,
        "receipt_path": str(core_path),
    }
    core = {
        **core_unsigned,
        "receipt_sha256": runtime._sha256_json(core_unsigned),
    }
    core_path.write_bytes(runtime._canonical_bytes(core))
    core_path.chmod(0o400)
    run_id = "phase-two-recovery"
    session_id = f"capability_{run_id}"
    epoch_sha = "1" * 64
    catalog_sha = "2" * 64
    authority_sha = "3" * 64
    publication = {
        "run_id": run_id,
        "session_id": session_id,
        "capability_epoch_sha256": epoch_sha,
        "catalog_sha256": catalog_sha,
        "authority_sha256": authority_sha,
        "readback_verified": True,
        "receipt_sha256": "4" * 64,
    }
    prepared = {
        "receipt_sha256": "5" * 64,
        "api_admission_publication": publication,
    }
    ack_unsigned = {
        "schema": "hermes.api.run-admission-ready-ack.v1",
        "session_id": session_id,
        "capability_epoch_sha256": epoch_sha,
        "challenge_sha256": "6" * 64,
        "ready_receipt_sha256": "7" * 64,
        "acknowledged": True,
    }
    ready_ack = {
        **ack_unsigned,
        "receipt_sha256": runtime._sha256_json(ack_unsigned),
    }
    pending = {
        "gateway_readiness": {"gateway_pid": 991},
        "full_preflight": {"report_sha256": "8" * 64},
        "preflight": {"report_sha256": "9" * 64},
        **{
            field: {}
            for field in (
                "installed",
                "connector_state",
                "phase_b_current",
                "installed_phase_b_anchor",
                "writer_readiness",
                "observer",
                "browser_identity_foundation",
                "browser_principal_smoke",
                "execution_identity_foundation",
                "worker_mountpoint",
                "execution_readiness",
                "routeback_bot_identity",
                "producer_foundation",
            )
        },
    }
    activation = SimpleNamespace(
        readiness={
            "catalog_sha256": catalog_sha,
            "owner_authority_sha256": authority_sha,
            "readiness_sha256": "a" * 64,
        }
    )
    require_stopped_values = []
    starts = []
    lifecycle = object.__new__(runtime.CapabilityCanaryLifecycle)
    lifecycle.plan = plan
    lifecycle.full_plan = _full_plan()
    lifecycle.runner = lambda _command: None
    lifecycle.metadata_reader = None
    lifecycle.local_identity_reader = None
    lifecycle._pending_deferred_start = None
    lifecycle._require_host = lambda: {}

    def validate(_approval, _core, *, require_producers_stopped):
        require_stopped_values.append(require_producers_stopped)
        return pending

    lifecycle._validate_deferred_core_runtime = validate
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime.time, "time", lambda: 1_100)
    monkeypatch.setattr(
        runtime,
        "_require_installed_capability_approval",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        runtime,
        "_load_deferred_lifecycle_state",
        lambda *_args: (core, None),
    )
    monkeypatch.setattr(
        runtime,
        "_load_bound_deferred_stage",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda *_args, **_kwargs: (runtime._canonical_bytes(core), core_path.stat()),
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)
    monkeypatch.setattr(
        runtime,
        "_run_checked",
        lambda command, **_kwargs: starts.append(command.argv),
    )
    monkeypatch.setattr(
        runtime,
        "_await_runtime_ready",
        lambda _probe, **_kwargs: {"ready": True},
    )
    monkeypatch.setattr(
        runtime,
        "collect_capability_preflight",
        lambda *_args, **_kwargs: {"report_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "_write_lifecycle_receipt",
        lambda _plan, *, stage, value: {
            **value,
            "stage": stage,
            "receipt_sha256": "c" * 64,
        },
    )

    receipt, recovered_activation, recovered_publication = (
        lifecycle.start_admitted_producers(
            approval,
            expected_gateway_pid=991,
            expected_run_id=run_id,
            expected_session_id=session_id,
            expected_capability_epoch_sha256=epoch_sha,
            expected_catalog_sha256=catalog_sha,
            expected_owner_authority_sha256=authority_sha,
            api_admission_ready_ack=ready_ack,
            admission_publisher=lambda: publication,
            producer_fleet_activator=lambda: activation,
            producer_activation_retirer=lambda _value: {},
            admission_input_retirer=lambda: {},
        )
    )

    assert require_stopped_values == [False]
    assert len(starts) == len(runtime.CAPABILITY_PRODUCER_ROLES)
    assert recovered_activation is activation
    assert recovered_publication == publication
    assert receipt["runtime_started"] is True


def test_expired_capability_approval_causes_zero_mutations(monkeypatch):
    plan = _plan()
    approval = _capability_approval(
        plan,
        approved_at_unix=1,
        expires_at_unix=61,
    )
    mutations = []
    lifecycle = runtime.CapabilityCanaryLifecycle(
        plan,
        _full_plan(),
        runner=lambda command: mutations.append(command),
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime.time, "time", lambda: 61)

    with pytest.raises(PermissionError, match="does not authorize"):
        lifecycle.start(approval)
    assert mutations == []


def _approval_crash_recovery_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan()
    full = _full_plan()
    now = int(time.time())
    approval = _capability_approval(
        plan,
        approved_at_unix=now - 1,
        expires_at_unix=now + 299,
    )
    approval_path = tmp_path / "etc/owner-approval.json"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_PATH", approval_path)
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _p: {})
    monkeypatch.setattr(
        runtime,
        "collect_capability_preflight",
        lambda *_a, **_k: _stopped_preflight_for_approval(plan, approval),
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: contextlib.nullcontext())

    def ensure_root_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(0o700)

    def write_exclusive(path, payload, *, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            current = path.read_bytes()
            if current == payload:
                return
            if len(current) < len(payload) and payload.startswith(current):
                path.unlink()
            else:
                raise RuntimeError("exclusive test publication drifted")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def read_as_root(path, **_kwargs):
        path = Path(path)
        raw = path.read_bytes()
        item = path.stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    monkeypatch.setattr(runtime, "_ensure_root_directory", ensure_root_directory)
    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    monkeypatch.setattr(runtime, "_read_stable_file", read_as_root)
    monkeypatch.setattr(runtime, "_require_same_file_identity", lambda *_a: None)
    return plan, full, approval, approval_path, write_exclusive, read_as_root


def test_approval_install_is_exclusive_nonce_journaled_and_retired(
    tmp_path, monkeypatch
):
    plan = _plan()
    full = _full_plan()
    now = int(time.time())
    approval = _capability_approval(
        plan,
        approved_at_unix=now - 1,
        expires_at_unix=now + 299,
    )
    approval_path = tmp_path / "etc/owner-approval.json"
    receipt_root = tmp_path / "receipts"
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_PATH", approval_path)
    monkeypatch.setattr(runtime, "DEFAULT_APPROVAL_RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "validate_dedicated_canary_host", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_validate_release_manifest", lambda _p: {})
    active_approval = {"value": approval}
    monkeypatch.setattr(
        runtime,
        "collect_capability_preflight",
        lambda *_a, **_k: _stopped_preflight_for_approval(
            plan,
            active_approval["value"],
        ),
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: contextlib.nullcontext())

    def ensure_root_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, os.geteuid(), os.getegid())
        path.chmod(0o700)

    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        ensure_root_directory,
    )

    def write_exclusive(path, payload, *, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            current = path.read_bytes()
            if current == payload:
                return
            if len(current) < len(payload) and payload.startswith(current):
                path.unlink()
            else:
                raise RuntimeError("exclusive test publication drifted")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    def read_as_root(path, **_kwargs):
        path = Path(path)
        raw = path.read_bytes()
        item = path.stat()
        return raw, SimpleNamespace(
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_size=item.st_size,
            st_mtime_ns=item.st_mtime_ns,
            st_ctime_ns=item.st_ctime_ns,
        )

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    monkeypatch.setattr(runtime, "_read_stable_file", read_as_root)
    monkeypatch.setattr(runtime, "_require_same_file_identity", lambda *_a: None)

    stale_value = dict(approval.value)
    stale_value["stopped_preflight_state_sha256"] = "5" * 64
    stale = runtime.CapabilityCanaryOwnerApproval.from_mapping(stale_value)
    with pytest.raises(PermissionError, match="complete fresh capability preflight"):
        runtime.install_capability_approval(plan, full, stale)
    assert not approval_path.exists()

    # Simulate SIGKILL after the exact approval inode became reachable but
    # before its nonce receipt was published.  The next bounded install must
    # complete that exact pair rather than reject the host forever.
    write_exclusive(
        approval_path,
        runtime._canonical_bytes(approval.value),
        mode=0o400,
    )
    receipt = runtime.install_capability_approval(plan, full, approval)
    assert approval_path.exists()
    assert receipt["approval_sha256"] == approval.sha256
    assert receipt["full_canary_terminal_receipt"] == (
        plan.full_canary_terminal_receipt
    )
    assert receipt["full_canary_terminal_receipt_sha256"] == (
        plan.full_canary_terminal_receipt_sha256
    )
    assert receipt["original_full_canary_owner_approval_sha256"] == (
        plan.original_full_canary_owner_approval_sha256
    )
    assert (
        receipt["lease_install_receipt_sha256_by_binding"]
        == approval.value["lease_install_receipt_sha256_by_binding"]
    )
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    retirement = runtime._remove_installed_capability_approval(plan, full)
    assert retirement["removed"] is True
    retirement_receipt = Path(retirement["retirement_receipt_path"])
    assert retirement_receipt.is_file()
    assert (
        json.loads(retirement_receipt.read_text())["receipt_sha256"]
        == (retirement["retirement_receipt_sha256"])
    )
    assert not approval_path.exists()
    assert runtime._remove_installed_capability_approval(plan, full) == retirement

    second = _capability_approval(
        plan,
        approved_at_unix=now - 1,
        expires_at_unix=now + 299,
        nonce_sha256="4" * 64,
    )
    active_approval["value"] = second
    runtime.install_capability_approval(plan, full, second)
    append_artifact = runtime._append_lease_artifact

    def interrupt_after_unlink(path, *, schema, value):
        if schema == runtime.CAPABILITY_APPROVAL_RETIREMENT_RECEIPT_SCHEMA:
            raise SystemExit("injected approval retirement SIGKILL window")
        return append_artifact(path, schema=schema, value=value)

    monkeypatch.setattr(
        runtime,
        "_append_lease_artifact",
        interrupt_after_unlink,
    )
    with pytest.raises(SystemExit, match="SIGKILL window"):
        runtime._remove_installed_capability_approval(plan, full)
    assert not approval_path.exists()
    monkeypatch.setattr(runtime, "_append_lease_artifact", append_artifact)

    # A later fresh approval first reconciles the old unlink/completion
    # half-state under the same lifecycle lock, then installs normally.  The
    # old unfinished intent can never poison retirement of the newer inode.
    third = _capability_approval(
        plan,
        approved_at_unix=now - 1,
        expires_at_unix=now + 299,
        nonce_sha256="6" * 64,
    )
    active_approval["value"] = third
    runtime.install_capability_approval(plan, full, third)
    assert approval_path.exists()
    histories = runtime._approval_retirement_histories(plan, full)
    second_history = next(item for item in histories if item[1].sha256 == second.sha256)
    assert second_history[3] is not None
    recovered = runtime._remove_installed_capability_approval(plan, full)
    assert recovered["approval_sha256"] == third.sha256
    assert recovered["removed"] is True
    assert recovered["absent"] is True

    with pytest.raises(PermissionError, match="nonce was already consumed"):
        runtime.install_capability_approval(plan, full, approval)


def test_approval_install_recovers_kill_truncated_approval_then_retires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        plan,
        full,
        approval,
        approval_path,
        _write_exclusive,
        _read_as_root,
    ) = _approval_crash_recovery_harness(tmp_path, monkeypatch)
    payload = runtime._canonical_bytes(approval.value)
    approval_path.parent.mkdir(parents=True)
    approval_path.write_bytes(payload[: len(payload) // 2])
    approval_path.chmod(0o400)

    receipt = runtime.install_capability_approval(plan, full, approval)

    assert approval_path.read_bytes() == payload
    assert receipt["approval_sha256"] == approval.sha256
    retirement = runtime._remove_installed_capability_approval(plan, full)
    assert retirement["removed"] is True
    assert retirement["absent"] is True
    assert not approval_path.exists()


def test_approval_install_recovers_kill_truncated_receipt_then_retires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        plan,
        full,
        approval,
        approval_path,
        write_exclusive,
        read_as_root,
    ) = _approval_crash_recovery_harness(tmp_path, monkeypatch)
    approval_payload = runtime._canonical_bytes(approval.value)
    write_exclusive(approval_path, approval_payload, mode=0o400)
    _raw, target = read_as_root(approval_path)
    receipt_path = runtime._approval_install_receipt_path(plan, approval)
    expected_receipt = runtime._build_approval_install_receipt(
        plan,
        full,
        approval,
        target=target,
        receipt_path=receipt_path,
    )
    receipt_payload = runtime._canonical_bytes(expected_receipt)
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_payload[: len(receipt_payload) // 2])
    receipt_path.chmod(0o400)

    with pytest.raises(PermissionError, match="nonce was already consumed"):
        runtime.install_capability_approval(plan, full, approval)

    assert approval_path.read_bytes() == approval_payload
    assert receipt_path.read_bytes() == receipt_payload
    retirement = runtime._remove_installed_capability_approval(plan, full)
    assert retirement["removed"] is True
    assert retirement["absent"] is True
    assert not approval_path.exists()


def test_connector_cleanup_requires_no_unacked_or_unresolved_work():
    safe = {
        "schema": "discord-public-connector-cleanup-snapshot.v1",
        "event_state_counts": {"acked": 2},
        "send_state_counts": {"verified": 1, "blocked": 1},
        "unresolved_dispatch_count": 0,
        "unacked_event_count": 0,
        "safe_to_retire": True,
    }
    assert runtime._connector_cleanup_snapshot_is_safe(safe)
    for event_state in ("pending", "delivering"):
        unsafe = dict(safe)
        unsafe["event_state_counts"] = {event_state: 1}
        unsafe["unacked_event_count"] = 1
        unsafe["safe_to_retire"] = False
        assert not runtime._connector_cleanup_snapshot_is_safe(unsafe)
    for send_state in ("prepared", "dispatching", "uncertain"):
        unsafe = dict(safe)
        unsafe["send_state_counts"] = {send_state: 1}
        unsafe["unresolved_dispatch_count"] = 1
        unsafe["safe_to_retire"] = False
        assert not runtime._connector_cleanup_snapshot_is_safe(unsafe)


def test_secret_frames_are_plan_bound_and_never_include_refresh_token():
    now = int(time.time())
    token = _jwt(now + 3_600)
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=token,
        plan_sha256="a" * 64,
        owner_subject_sha256="b" * 64,
        now_unix=now,
        lease_id="c" * 32,
    )
    metadata, decoded = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="codex_access_token",
        now_unix=now,
    )
    assert metadata["plan_sha256"] == "a" * 64
    assert decoded == token
    assert b"refresh_token" not in frame

    # The opaque secret intentionally has no digest (including in receipts).
    # Framing still rejects truncation/trailing-byte substitution.
    frame.append(0)
    with pytest.raises(ValueError):
        runtime.read_secret_lease_frame(
            io.BytesIO(frame),
            expected_kind="codex_access_token",
            now_unix=now,
        )


@pytest.mark.parametrize("kind", tuple(runtime._SECRET_LEASE_MAGIC_BY_KIND))
def test_all_six_secret_leases_are_half_open_and_reserve_bounded(kind):
    plan = _plan()
    issued = 1_000
    payload = _jwt(issued + 1_000) if kind == "codex_access_token" else b"opaque-secret"
    frame = bytes(
        runtime.build_secret_lease_frame(
            kind=kind,
            secret=payload,
            plan_sha256=plan.sha256,
            owner_subject_sha256="b" * 64,
            now_unix=issued,
            ttl_seconds=60,
            lease_id="c" * 32,
        )
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind=kind,
        now_unix=issued + 59,
    )
    assert secret == payload
    assert (
        runtime._validate_lease_metadata(
            plan,
            metadata,
            secret,
            now_unix=issued,
            minimum_reserve_seconds=60,
        )
        == kind
    )
    with pytest.raises(runtime.CapabilityLeaseReserveError):
        runtime._validate_lease_metadata(
            plan,
            metadata,
            secret,
            now_unix=issued + 1,
            minimum_reserve_seconds=60,
        )
    with pytest.raises(ValueError, match="metadata is invalid"):
        runtime.read_secret_lease_frame(
            io.BytesIO(frame),
            expected_kind=kind,
            now_unix=issued + 60,
        )


def test_operation_clock_rejects_regression_and_provision_uses_lifecycle_first(
    monkeypatch,
):
    values = iter((1_000, 999))
    clock = runtime._OperationClock(lambda: next(values))
    assert clock.sample("first") == 1_000
    with pytest.raises(RuntimeError, match="clock regressed"):
        clock.sample("second")

    events = []
    held = {"value": False}

    @contextlib.contextmanager
    def lifecycle_lock():
        assert held["value"] is False
        held["value"] = True
        events.append("lifecycle-enter")
        try:
            yield
        finally:
            events.append("lifecycle-exit")
            held["value"] = False

    def provision_locked(*_args, **_kwargs):
        assert held["value"] is True
        events.append("lease-journal-work")
        return {"ok": True}

    monkeypatch.setattr(runtime, "_lifecycle_lock", lifecycle_lock)
    monkeypatch.setattr(runtime, "_provision_secret_lease_locked", provision_locked)
    assert runtime.provision_secret_lease(
        _plan(),
        {},
        bytearray(b"secret"),
        full_plan=_full_plan(),
    ) == {"ok": True}
    assert events == [
        "lifecycle-enter",
        "lease-journal-work",
        "lifecycle-exit",
    ]


@pytest.mark.parametrize("live_unit", runtime.CAPABILITY_STOP_ORDER)
def test_each_live_consumer_blocks_secret_provision_before_any_mutation(
    monkeypatch,
    live_unit,
):
    plan = _plan()
    secret = bytearray(b"opaque-secret")
    metadata = {
        "schema": runtime.CAPABILITY_LEASE_FRAME_SCHEMA,
        "kind": "api_server_control_key",
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "b" * 64,
        "lease_id": "c" * 32,
        "issued_at_unix": 1_000,
        "expires_at_unix": 1_900,
        "secret_bytes": len(secret),
    }
    services = _watchdog_stopped_services()
    services[live_unit] = {
        **services[live_unit],
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": 4242,
        "UnitFileState": "enabled",
    }
    mutations = []
    lock_held = {"value": False}

    @contextlib.contextmanager
    def lifecycle_lock():
        assert lock_held["value"] is False
        lock_held["value"] = True
        try:
            yield
        finally:
            lock_held["value"] = False

    def observe_services(*, runner):
        assert lock_held["value"] is True
        assert runner is runtime._runner
        return services

    def forbidden_mutation(*_args, **_kwargs):
        mutations.append("unexpected")
        raise AssertionError("secret provision mutated before stopped proof")

    monkeypatch.setattr(runtime, "_capability_services", observe_services)
    for name in (
        "arm_capability_expiry_watchdog",
        "_lease_journal_lock",
        "_append_lease_artifact",
        "_prepare_secret_parent",
        "_atomic_no_replace_file",
    ):
        monkeypatch.setattr(runtime, name, forbidden_mutation)

    with lifecycle_lock():
        with pytest.raises(RuntimeError, match="all capability consumers stopped"):
            runtime._provision_secret_lease_locked(
                plan,
                metadata,
                secret,
                full_plan=_full_plan(),
                operation_clock=runtime._OperationClock(lambda: 1_100),
            )

    assert mutations == []
    assert secret == bytearray(len(secret))


def test_secret_provision_rechecks_consumers_immediately_before_publication(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    secret = bytearray(b"opaque-secret")
    metadata = {
        "schema": runtime.CAPABILITY_LEASE_FRAME_SCHEMA,
        "kind": "api_server_control_key",
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "b" * 64,
        "lease_id": "d" * 32,
        "issued_at_unix": 1_000,
        "expires_at_unix": 1_900,
        "secret_bytes": len(secret),
    }
    spec = runtime._SecretLeaseTarget(
        kind="api_server_control_key",
        credential_binding="api_control",
        path=tmp_path / "secret",
        journal=tmp_path / "journal",
        uid=os.geteuid(),
        gid=os.getegid(),
        mode=0o400,
        parent_uid=os.geteuid(),
        parent_gid=os.getegid(),
        parent_mode=0o700,
        maximum_bytes=512,
    )
    checks = []
    mutations = []

    def require_stopped():
        checks.append("stopped")
        if len(checks) == 2:
            raise RuntimeError(
                "secret lease provisioning requires all capability consumers stopped"
            )
        return _watchdog_stopped_services()

    def append_intent(path, *, schema, value):
        mutations.append("install-intent")
        return {
            **value,
            "schema": schema,
            "receipt_path": str(path),
            "receipt_sha256": "e" * 64,
        }

    def forbidden_publish(*_args, **_kwargs):
        raise AssertionError("secret inode published after a live-consumer recheck")

    monkeypatch.setattr(runtime, "_lease_target", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        runtime,
        "_require_secret_provision_consumers_stopped",
        require_stopped,
    )
    monkeypatch.setattr(
        runtime,
        "arm_capability_expiry_watchdog",
        lambda **_kwargs: mutations.append("watchdog") or {},
    )
    monkeypatch.setattr(
        runtime,
        "_lease_journal_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime, "_journal_states", lambda _path: [])
    monkeypatch.setattr(runtime, "_prepare_journal_directory", lambda _path: None)
    monkeypatch.setattr(runtime, "_append_lease_artifact", append_intent)
    monkeypatch.setattr(
        runtime,
        "_prepare_secret_parent",
        lambda *_args, **_kwargs: mutations.append("secret-parent"),
    )
    monkeypatch.setattr(runtime, "_atomic_no_replace_file", forbidden_publish)

    with pytest.raises(RuntimeError, match="all capability consumers stopped"):
        runtime._provision_secret_lease_locked(
            plan,
            metadata,
            secret,
            full_plan=_full_plan(),
            operation_clock=runtime._OperationClock(lambda: 1_100),
        )

    assert checks == ["stopped", "stopped"]
    assert mutations == ["watchdog", "install-intent", "secret-parent"]
    assert secret == bytearray(len(secret))
    assert not spec.path.exists()


def test_failed_secret_compensation_uses_default_runner_before_live_guard(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    full = _full_plan()
    lease_id = "e" * 32
    spec = runtime._SecretLeaseTarget(
        kind="api_server_control_key",
        credential_binding="api_control",
        path=tmp_path / "secret",
        journal=tmp_path / "journal",
        uid=os.geteuid(),
        gid=os.getegid(),
        mode=0o400,
        parent_uid=os.geteuid(),
        parent_gid=os.getegid(),
        parent_mode=0o700,
        maximum_bytes=512,
    )
    spec.journal.mkdir()
    calls = []
    stopped = _watchdog_stopped_services()
    live_unit = runtime.CAPABILITY_STOP_ORDER[0]
    stopped[live_unit] = {
        **stopped[live_unit],
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": 4242,
        "UnitFileState": "enabled",
    }

    def collect(unit, *, runner):
        assert runner is runtime._runner
        calls.append(unit)
        return stopped[unit]

    monkeypatch.setattr(runtime, "_lease_target", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        runtime,
        "_lease_journal_lock",
        lambda _path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime,
        "_journal_states",
        lambda _path: [
            {
                "lease_id": lease_id,
                "install_abort": None,
                "retirement_completion": None,
            }
        ],
    )
    monkeypatch.setattr(runtime, "collect_capability_service_state", collect)
    monkeypatch.setattr(
        runtime,
        "_retire_secret_slot_best_effort",
        lambda *_args, **_kwargs: pytest.fail("live-consumer guard was bypassed"),
    )

    with pytest.raises(RuntimeError, match="live consumer"):
        runtime._compensate_failed_secret_provision_locked(
            plan,
            full,
            metadata={"kind": spec.kind, "lease_id": lease_id},
            now_unix=1_100,
        )

    assert len(calls) == len(runtime.CAPABILITY_STOP_ORDER)
    assert set(calls) == set(runtime.CAPABILITY_STOP_ORDER)


def test_failed_secret_compensation_retires_real_published_lease(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    full = _full_plan()
    kind = "api_server_control_key"
    lease_id = "f" * 32
    target = tmp_path / "credential/api-control-key"
    journal = tmp_path / "control/api-control-leases"
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind=kind,
        secret=b"published-control-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="a" * 64,
        now_unix=now,
        lease_id=lease_id,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind=kind,
        now_unix=now,
    )
    install = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        full_plan=full,
        api_control_path=target,
        journal_path=journal,
        clock=lambda: now,
    )
    spec = runtime._lease_target(
        plan,
        kind=kind,
        full_plan=full,
        api_control_path=target,
        journal_path=journal,
    )
    assert target.is_file()
    assert install["lease_id"] == lease_id
    assert secret == bytearray(len(secret))

    calls = []
    stopped = _watchdog_stopped_services()

    def collect(unit, *, runner):
        assert runner is runtime._runner
        calls.append(unit)
        return stopped[unit]

    monkeypatch.setattr(runtime, "_lease_target", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(runtime, "collect_capability_service_state", collect)

    compensated = runtime._compensate_failed_secret_provision_locked(
        plan,
        full,
        metadata=metadata,
        now_unix=now + 1,
    )

    assert compensated is not None
    assert compensated["state"] == "install_bound_retired"
    assert compensated["retirement_completion"]["lease_id"] == lease_id
    assert compensated["retirement_completion"]["install_receipt_sha256"] == (
        install["receipt_sha256"]
    )
    assert compensated["absent"] is True
    assert not target.exists()
    states = runtime._journal_states(journal)
    assert len(states) == 1
    assert states[0]["lease_id"] == lease_id
    assert states[0]["install_receipt"]["receipt_sha256"] == install["receipt_sha256"]
    assert states[0]["install_abort"] is None
    assert states[0]["retirement_completion"]["receipt_sha256"] == (
        compensated["retirement_receipt_sha256"]
    )
    assert len(calls) == len(runtime.CAPABILITY_STOP_ORDER)
    assert set(calls) == set(runtime.CAPABILITY_STOP_ORDER)


@pytest.mark.parametrize("kind", tuple(runtime._SECRET_LEASE_MAGIC_BY_KIND))
def test_all_six_failed_provision_commits_compensate_before_lifecycle_unlock(
    monkeypatch, kind
):
    events = []
    held = {"value": False}
    secret = bytearray(b"opaque-secret")
    metadata = {
        "kind": kind,
        "lease_id": "c" * 32,
    }

    @contextlib.contextmanager
    def lifecycle_lock():
        held["value"] = True
        events.append("lifecycle-enter")
        try:
            yield
        finally:
            events.append("lifecycle-exit")
            held["value"] = False

    def fail_commit(*_args, **_kwargs):
        assert held["value"] is True
        events.append(f"{kind}:commit-failed")
        raise RuntimeError("injected post-publication failure")

    def compensate(_plan, _full, *, metadata, now_unix):
        assert held["value"] is True
        assert metadata["kind"] == kind
        assert type(now_unix) is int
        events.append(f"{kind}:compensated")
        return {"state": "retired_before_unlock"}

    monkeypatch.setattr(runtime, "_lifecycle_lock", lifecycle_lock)
    monkeypatch.setattr(runtime, "_provision_secret_lease_locked", fail_commit)
    monkeypatch.setattr(
        runtime,
        "_compensate_failed_secret_provision_locked",
        compensate,
    )
    with pytest.raises(RuntimeError, match="post-publication failure"):
        runtime.provision_secret_lease(
            _plan(),
            metadata,
            secret,
            full_plan=_full_plan(),
            clock=lambda: 1_000,
        )
    assert secret == bytearray(len(secret))
    assert events == [
        "lifecycle-enter",
        f"{kind}:commit-failed",
        f"{kind}:compensated",
        "lifecycle-exit",
    ]


@pytest.mark.parametrize("kind", tuple(runtime._SECRET_LEASE_MAGIC_BY_KIND))
def test_all_six_partial_publications_append_abort_and_remove_exact_inode(
    tmp_path, kind
):
    plan = _plan()
    binding = runtime._CREDENTIAL_BINDING_BY_KIND[kind]
    target = tmp_path / kind / "secret"
    journal = tmp_path / "journals" / kind
    target.parent.mkdir(parents=True, mode=0o700)
    target.write_bytes(b"partial-secret")
    target.chmod(0o400)
    target_identity = target.lstat()
    parent_identity = target.parent.lstat()
    owner = target_identity.st_uid
    group = target_identity.st_gid
    spec = runtime._SecretLeaseTarget(
        kind=kind,
        credential_binding=binding,
        path=target,
        journal=journal,
        uid=owner,
        gid=group,
        mode=0o400,
        parent_uid=parent_identity.st_uid,
        parent_gid=parent_identity.st_gid,
        parent_mode=0o700,
        maximum_bytes=1024,
    )
    lease_id = "c" * 32
    paths = runtime._lease_artifact_paths(journal, lease_id)
    runtime._prepare_journal_directory(journal)
    runtime._prepare_journal_directory(paths.root)
    intent = runtime._append_lease_artifact(
        paths.install_intent,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "kind": kind,
            "credential_binding": binding,
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "owner_subject_sha256": "b" * 64,
            "lease_id": lease_id,
            "issued_at_unix": 1_000,
            "expires_at_unix": 1_900,
            "target_path": str(target),
            "target_uid": owner,
            "target_gid": group,
            "target_mode": "0400",
            "target_parent_uid": parent_identity.st_uid,
            "target_parent_gid": parent_identity.st_gid,
            "target_parent_mode": "0700",
            "intent_at_unix": 1_000,
            "expiry_watchdog": {},
        },
    )
    result = runtime._retire_secret_slot_best_effort(
        plan,
        spec,
        stop_proof={},
        now_unix=1_100,
    )
    assert result["state"] == "incomplete_install_retired"
    assert result["install_intent_sha256"] == intent["receipt_sha256"]
    assert result["install_abort"]["target_absent"] is True
    assert result["install_abort"]["temporary_absent"] is True
    assert Path(result["install_abort"]["receipt_path"]).is_file()
    assert not target.exists()


def test_codex_provision_and_retirement_receipts_are_secret_free(tmp_path, monkeypatch):
    plan = _plan()
    now = int(time.time())
    token = _jwt(now + 3_600)
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=token,
        plan_sha256=plan.sha256,
        owner_subject_sha256="f" * 64,
        now_unix=now,
        lease_id="9" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    auth_path = tmp_path / "profile/auth.json"
    journal = tmp_path / "control/codex.json"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        auth_path=auth_path,
        journal_path=journal,
    )
    store = json.loads(auth_path.read_text())
    entry = store["credential_pool"]["openai-codex"][0]
    assert entry["access_token"]
    assert "refresh_token" not in entry
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert bytes(token) not in json.dumps(receipt).encode()
    assert secret == bytearray(len(secret))

    monkeypatch.setenv("HERMES_HOME", str(auth_path.parent))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entries = pool.entries()
    assert len(entries) == 1
    assert (
        entries[0].access_token
        == store["credential_pool"]["openai-codex"][0]["access_token"]
    )
    assert entries[0].refresh_token is None

    retired = runtime.retire_secret_lease(
        kind="codex_access_token",
        target=auth_path,
        journal=journal,
        stop_proof=_stop_proof(plan),
    )
    assert retired["absent"] is True
    assert not auth_path.exists()
    assert retired["secret_digest_recorded"] is False


def test_secret_slot_prefers_new_active_and_latest_completed_generation(tmp_path):
    plan = _plan()
    kind = "api_server_control_key"
    target = tmp_path / "credentials/api-control-key"
    journal = tmp_path / "control/api-control-leases"
    spec = runtime._lease_target(
        plan,
        kind=kind,
        api_control_path=target,
        journal_path=journal,
    )
    aborted_id = "a" * 32
    aborted_paths = runtime._lease_artifact_paths(journal, aborted_id)
    runtime._prepare_journal_directory(journal)
    runtime._prepare_journal_directory(aborted_paths.root)
    runtime._append_lease_artifact(
        aborted_paths.install_intent,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "kind": kind,
            "credential_binding": spec.credential_binding,
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "owner_subject_sha256": "1" * 64,
            "lease_id": aborted_id,
            "issued_at_unix": 1_000,
            "expires_at_unix": 1_900,
            "target_path": str(target),
            "target_uid": spec.uid,
            "target_gid": spec.gid,
            "target_mode": f"{spec.mode:04o}",
            "target_parent_uid": spec.parent_uid,
            "target_parent_gid": spec.parent_gid,
            "target_parent_mode": f"{spec.parent_mode:04o}",
            "intent_at_unix": 1_000,
            "expiry_watchdog": {},
        },
    )
    aborted = runtime._retire_secret_slot_best_effort(
        plan,
        spec,
        stop_proof={},
        now_unix=1_100,
    )
    assert aborted["state"] == "incomplete_install_retired"

    def install(lease_id, secret):
        now = int(time.time())
        frame = runtime.build_secret_lease_frame(
            kind=kind,
            secret=secret,
            plan_sha256=plan.sha256,
            owner_subject_sha256="2" * 64,
            now_unix=now,
            lease_id=lease_id,
        )
        metadata, decoded = runtime.read_secret_lease_frame(
            io.BytesIO(frame),
            expected_kind=kind,
            now_unix=now,
        )
        return runtime.provision_secret_lease(
            plan,
            metadata,
            decoded,
            api_control_path=target,
            journal_path=journal,
        )

    second = install("b" * 32, b"second-generation-control-key")
    retired_second = runtime._retire_secret_slot_best_effort(
        plan,
        spec,
        stop_proof=_stop_proof(plan),
    )
    assert retired_second["state"] == "install_bound_retired"
    assert retired_second["retirement_completion"]["lease_id"] == "b" * 32
    assert (
        retired_second["retirement_completion"]["install_receipt_sha256"]
        == (second["receipt_sha256"])
    )

    third = install("c" * 32, b"third-generation-control-key")
    runtime.retire_secret_lease(
        kind=kind,
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
    )
    latest = runtime._retire_secret_slot_best_effort(
        plan,
        spec,
        stop_proof=_stop_proof(plan),
    )
    assert latest["retirement_completion"]["lease_id"] == "c" * 32
    assert (
        latest["retirement_completion"]["install_receipt_sha256"]
        == third["receipt_sha256"]
    )
    assert len(latest["retirement_history_receipt_sha256s"]) == 2
    assert not target.exists()


def test_discord_connector_lease_is_owned_and_secret_free(tmp_path):
    plan = _plan()
    plan = replace(
        plan,
        identities=replace(
            plan.identities,
            connector_uid=os.getuid(),
            connector_gid=os.getgid(),
        ),
    )
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="discord_connector_token",
        secret=bytearray(b"opaque.connector.token"),
        plan_sha256=plan.sha256,
        owner_subject_sha256="6" * 64,
        now_unix=now,
        lease_id="5" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="discord_connector_token",
        now_unix=now,
    )
    target = tmp_path / "connector/bot-token"
    journal = tmp_path / "control/connector.json"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        connector_path=target,
        journal_path=journal,
    )
    assert target.read_bytes() == b"opaque.connector.token"
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert receipt["kind"] == "discord_connector_token"
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False
    assert b"opaque.connector.token" not in json.dumps(receipt).encode()
    assert secret == bytearray(len(secret))
    retired = runtime.retire_secret_lease(
        kind="discord_connector_token",
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
    )
    assert retired["absent"] is True


@pytest.mark.parametrize(
    "kind,secret,path_argument,identity_fields",
    (
        (
            "api_server_control_key",
            b"generated-api-control-key",
            "api_control_path",
            (),
        ),
        (
            "bitrix_operational_edge_webhook",
            b"https://example.bitrix24.eu/rest/1/opaque/",
            "bitrix_path",
            (),
        ),
        (
            "mac_ops_gitlab_env",
            b"GITLAB_BASE_URL=https://gitlab.example\nGITLAB_TOKEN=opaque\n",
            "mac_path",
            ("mac_ops_uid", "mac_ops_gid"),
        ),
    ),
)
def test_additional_credential_leases_are_append_only_and_secret_free(
    tmp_path,
    kind,
    secret,
    path_argument,
    identity_fields,
):
    plan = _plan()
    if identity_fields:
        plan = replace(
            plan,
            identities=replace(
                plan.identities,
                **{
                    identity_fields[0]: os.getuid(),
                    identity_fields[1]: os.getgid(),
                },
            ),
        )
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind=kind,
        secret=secret,
        plan_sha256=plan.sha256,
        owner_subject_sha256="1" * 64,
        now_unix=now,
        lease_id="2" * 32,
    )
    metadata, decoded = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind=kind,
        now_unix=now,
    )
    target = tmp_path / "credentials" / "secret"
    journal = tmp_path / "control" / f"{kind}-leases"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        decoded,
        journal_path=journal,
        **{path_argument: target},
    )
    assert receipt["credential_binding"] == runtime._CREDENTIAL_BINDING_BY_KIND[kind]
    assert decoded == bytearray(len(decoded))
    assert target.exists()
    lease_root = journal / ("2" * 32)
    assert sorted(path.name for path in lease_root.iterdir()) == [
        "install-intent.json",
        "install-receipt.json",
    ]
    journal_bytes = b"".join(
        path.read_bytes() for path in lease_root.iterdir() if path.is_file()
    )
    assert secret not in journal_bytes
    assert b"secret_digest" in journal_bytes

    completion = runtime.retire_secret_lease(
        kind=kind,
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    )
    assert completion["absent_after_stop"] is True
    assert completion["install_receipt_sha256"] == receipt["receipt_sha256"]
    assert not target.exists()
    assert sorted(path.name for path in lease_root.iterdir()) == [
        "install-intent.json",
        "install-receipt.json",
        "retirement-completion.json",
        "retirement-intent.json",
    ]
    assert (
        runtime.retire_secret_lease(
            kind=kind,
            target=target,
            journal=journal,
            stop_proof=_stop_proof(plan),
            plan=plan,
        )
        == completion
    )


def test_partial_prestart_cleanup_attempts_all_six_and_removes_later_active_slots(
    tmp_path,
    monkeypatch,
):
    plan = replace(
        _plan(),
        identities=replace(
            _plan().identities,
            connector_uid=os.getuid(),
            connector_gid=os.getgid(),
        ),
    )
    full = _full_plan()
    monkeypatch.setattr(
        runtime,
        "DEFAULT_LIFECYCLE_RECEIPT_ROOT",
        tmp_path / "lifecycle",
    )
    monkeypatch.setattr(
        runtime,
        "load_bound_plan_publication_receipt",
        lambda _plan: {"receipt_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )

    def write_exclusive(path, payload, *, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, payload)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write_exclusive)
    by_kind = {}
    for index, kind in enumerate((
        "api_server_control_key",
        "bitrix_operational_edge_webhook",
        "discord_routeback_token",
        "discord_connector_token",
        "mac_ops_gitlab_env",
        "codex_access_token",
    )):
        mode = 0o600 if kind == "codex_access_token" else 0o400
        maximum = {
            "codex_access_token": 128 * 1024,
            "mac_ops_gitlab_env": runtime._MAX_SECRET_BYTES,
            "discord_connector_token": 512,
        }.get(kind, 8 * 1024)
        by_kind[kind] = runtime._SecretLeaseTarget(
            kind=kind,
            credential_binding=runtime._CREDENTIAL_BINDING_BY_KIND[kind],
            path=tmp_path / f"targets/{index}-{kind}",
            journal=tmp_path / f"journals/{index}-{kind}",
            uid=os.getuid(),
            gid=os.getgid(),
            mode=mode,
            parent_uid=os.getuid(),
            parent_gid=os.getgid(),
            parent_mode=0o700,
            maximum_bytes=maximum,
        )

    now = int(time.time())
    for offset, (kind, secret, path_argument) in enumerate((
        (
            "bitrix_operational_edge_webhook",
            b"https://example.bitrix24.eu/rest/1/opaque/",
            "bitrix_path",
        ),
        (
            "discord_connector_token",
            b"opaque.connector.token",
            "connector_path",
        ),
    )):
        spec = by_kind[kind]
        frame = runtime.build_secret_lease_frame(
            kind=kind,
            secret=secret,
            plan_sha256=plan.sha256,
            owner_subject_sha256="7" * 64,
            now_unix=now,
            lease_id=f"{offset + 1}" * 32,
        )
        metadata, decoded = runtime.read_secret_lease_frame(
            io.BytesIO(frame), expected_kind=kind, now_unix=now
        )
        runtime.provision_secret_lease(
            plan,
            metadata,
            decoded,
            journal_path=spec.journal,
            **{path_argument: spec.path},
        )
        assert spec.path.exists()

    ordered = tuple(
        by_kind[kind]
        for kind in (
            "api_server_control_key",
            "bitrix_operational_edge_webhook",
            "discord_routeback_token",
            "discord_connector_token",
            "mac_ops_gitlab_env",
            "codex_access_token",
        )
    )
    result = runtime.retire_secret_leases_best_effort(
        plan,
        full,
        targets=ordered,
        stop_proof=_stop_proof(plan),
    )

    assert result["ok"] is True
    assert result["all_six_credentials_absent_readback"] is True
    assert result["all_six_install_bound_retirement_completions"] is False
    assert result["slots"]["api_control"]["state"] == "never_installed_absent"
    assert (
        result["slots"]["bitrix_operational_edge_webhook"]["state"]
        == "install_bound_retired"
    )
    assert (
        result["slots"]["discord_public_session_bot_token"]["state"]
        == "install_bound_retired"
    )
    assert all(not spec.path.exists() for spec in ordered)


def test_routeback_lease_parent_is_root_controlled_and_not_edge_writable(
    tmp_path,
):
    full = _full_plan()
    production_spec = runtime._lease_target(
        _plan(full),
        kind="discord_routeback_token",
        full_plan=full,
        routeback_path=tmp_path / "unused/bot-token",
        journal_path=tmp_path / "unused-journal",
    )
    assert production_spec.parent_uid == os.geteuid()
    assert production_spec.parent_gid == full.identities.edge_gid
    assert production_spec.parent_mode == 0o750
    assert production_spec.parent_mode & stat.S_IWGRP == 0
    assert production_spec.uid == full.identities.edge_uid
    assert production_spec.mode == 0o400

    object.__setattr__(
        full,
        "identities",
        replace(
            full.identities,
            edge_uid=os.getuid(),
            edge_gid=os.getgid(),
        ),
    )
    plan = _plan(full)
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="discord_routeback_token",
        secret=b"opaque-routeback-token",
        plan_sha256=plan.sha256,
        owner_subject_sha256="3" * 64,
        now_unix=now,
        lease_id="4" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="discord_routeback_token",
        now_unix=now,
    )
    target = tmp_path / "routeback" / "bot-token"
    journal = tmp_path / "control" / "routeback-leases"
    receipt = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        full_plan=full,
        routeback_path=target,
        journal_path=journal,
    )
    assert receipt["credential_binding"] == ("discord_canonical_routeback_bot_token")
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o750
    assert (
        runtime.retire_secret_lease(
            kind="discord_routeback_token",
            target=target,
            journal=journal,
            stop_proof=_stop_proof(plan),
            plan=plan,
        )["absent"]
        is True
    )


def test_exact_provision_retry_is_idempotent_but_different_secret_fails(tmp_path):
    plan = _plan()
    now = int(time.time())
    arguments = {
        "kind": "api_server_control_key",
        "secret": b"first-generated-control-key",
        "plan_sha256": plan.sha256,
        "owner_subject_sha256": "5" * 64,
        "now_unix": now,
        "lease_id": "6" * 32,
    }
    frame = runtime.build_secret_lease_frame(**arguments)
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    first = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    assert (
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            api_control_path=target,
            journal_path=journal,
        )
        == first
    )
    different = runtime.build_secret_lease_frame(**{
        **arguments,
        "secret": b"other-generated-control-key",
    })
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(different),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    with pytest.raises(RuntimeError, match="different secret bytes"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            api_control_path=target,
            journal_path=journal,
        )
    assert secret == bytearray(len(secret))


def test_install_crash_after_intent_recovers_only_with_exact_retry(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"crash-recovery-control-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="7" * 64,
        now_unix=now,
        lease_id="8" * 32,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    original = runtime._atomic_no_replace_file
    interrupted = False

    def crash_once(path, payload, **kwargs):
        nonlocal interrupted
        if path == target and not interrupted:
            interrupted = True
            raise OSError("simulated install interruption")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(runtime, "_atomic_no_replace_file", crash_once)
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    with pytest.raises(OSError, match="simulated install interruption"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            api_control_path=target,
            journal_path=journal,
        )
    lease_root = journal / ("8" * 32)
    assert (lease_root / "install-intent.json").exists()
    assert not (lease_root / "install-receipt.json").exists()
    assert not target.exists()

    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    recovered = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    assert recovered["state"] == "provisioned"
    assert target.exists()


def test_retirement_crash_after_unlink_recovers_from_bound_intent(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"retirement-recovery-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="9" * 64,
        now_unix=now,
        lease_id="a" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api" / "key"
    journal = tmp_path / "control" / "api-leases"
    runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    original = runtime._append_lease_artifact
    interrupted = False

    def crash_once(path, *, schema, value):
        nonlocal interrupted
        if schema == runtime.CAPABILITY_RETIREMENT_RECEIPT_SCHEMA and not interrupted:
            interrupted = True
            raise OSError("simulated retirement interruption")
        return original(path, schema=schema, value=value)

    monkeypatch.setattr(runtime, "_append_lease_artifact", crash_once)
    with pytest.raises(OSError, match="simulated retirement interruption"):
        runtime.retire_secret_lease(
            kind="api_server_control_key",
            target=target,
            journal=journal,
            stop_proof=_stop_proof(plan),
            plan=plan,
        )
    assert not target.exists()
    assert (journal / ("a" * 32) / "retirement-intent.json").exists()
    completion = runtime.retire_secret_lease(
        kind="api_server_control_key",
        target=target,
        journal=journal,
        stop_proof=_stop_proof(plan),
        plan=plan,
    )
    assert completion["absent_after_stop"] is True


def test_retirement_time_is_post_stop_and_cannot_reuse_install_ctime(tmp_path):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="api_server_control_key",
        secret=b"post-stop-ordering-key",
        plan_sha256=plan.sha256,
        owner_subject_sha256="4" * 64,
        now_unix=now,
        lease_id="d" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame),
        expected_kind="api_server_control_key",
        now_unix=now,
    )
    target = tmp_path / "api/key"
    journal = tmp_path / "control/api-leases"
    installed = runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        api_control_path=target,
        journal_path=journal,
    )
    installed_at = installed["installed_at_unix"]
    stale = _stop_proof(plan, observed_at_unix=installed_at - 1)
    with pytest.raises(PermissionError, match="post-install stop proof"):
        runtime.retire_secret_lease(
            kind="api_server_control_key",
            target=target,
            journal=journal,
            stop_proof=stale,
            plan=plan,
            now_unix=installed_at + 10,
        )
    assert target.exists()

    stopped_at = installed_at + 5
    proof = _stop_proof(plan, observed_at_unix=stopped_at)
    completion = runtime.retire_secret_lease(
        kind="api_server_control_key",
        target=target,
        journal=journal,
        stop_proof=proof,
        plan=plan,
        now_unix=stopped_at + 1,
    )
    intent = json.loads((journal / ("d" * 32) / "retirement-intent.json").read_text())
    assert intent["requested_at_unix"] >= stopped_at
    assert intent["service_stop_proof_sha256"] == proof["stop_proof_sha256"]
    assert completion["service_stop_observed_at_unix"] == stopped_at
    assert completion["retired_at_unix"] >= stopped_at
    assert completion["retired_at_unix"] != installed_at
    assert completion["absent_after_stop"] is True


def _services_with_only_cleanup_observer_live():
    services = _watchdog_stopped_services()
    services[runtime.CAPABILITY_OBSERVER_UNIT] = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4242,
        "FragmentPath": str(
            Path("/etc/systemd/system") / runtime.CAPABILITY_OBSERVER_UNIT
        ),
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
        "StatusText": "",
    }
    return services


def test_credential_consumer_proof_keeps_only_blind_observer_live(monkeypatch):
    plan = _plan()
    services = _services_with_only_cleanup_observer_live()
    monkeypatch.setattr(
        runtime,
        "_producer_credential_inaccessibility_contract",
        lambda: {
            "paths": ["/etc/muncho/keys"],
            "applies_to_roles": list(runtime.CAPABILITY_PRODUCER_ROLES),
            "unit_hash_bound": True,
            "cleanup_observer_has_no_credential_read_access": True,
        },
    )
    proof = runtime.build_credential_consumer_stop_proof(
        plan,
        services,
        producer_foundation={
            "revision": plan.revision,
            "foundation_sha256": "1" * 64,
            "unit_bundle_manifest_sha256": "2" * 64,
            "ready": True,
            "mutation_performed": False,
        },
        observed_at_unix=100,
    )

    assert proof["schema"] == (runtime.CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA)
    assert proof["non_observer_stop_order"] == list(
        runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER
    )
    assert proof["observer_service_unit"] == runtime.CAPABILITY_OBSERVER_UNIT
    assert proof["observer_live_signing_only"] is True
    assert proof["observer_credential_read_access"] is False
    assert (
        runtime._validate_capability_stop_proof(
            plan,
            proof,
            installed_at_unix=90,
            now_unix=110,
        )
        == proof
    )
    with pytest.raises(RuntimeError, match="live service"):
        runtime.build_capability_stop_proof(plan, services)

    tampered = json.loads(json.dumps(proof))
    tampered["observer_credential_read_access"] = True
    tampered["stop_proof_sha256"] = runtime._sha256_json({
        key: value for key, value in tampered.items() if key != "stop_proof_sha256"
    })
    with pytest.raises(PermissionError, match="consumer stop proof"):
        runtime._validate_capability_stop_proof(
            plan,
            tampered,
            installed_at_unix=90,
            now_unix=110,
        )


def _install_cleanup_transaction_test_io(tmp_path, monkeypatch, run_id):
    receipt_root = tmp_path / "receipts"
    run_root = receipt_root / run_id
    run_root.mkdir(parents=True)
    run_root.chmod(0o3770)
    monkeypatch.setattr(producers, "DEFAULT_RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(
        runtime,
        "_cleanup_transaction_run_gid",
        lambda _run_id: os.getgid(),
    )

    def publish(path, payload, *, mode, **_kwargs):
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError("checkpoint collision")
        else:
            path.write_bytes(payload)
            path.chmod(mode)
        return path.stat()

    def read(path, *, maximum, mode, **_kwargs):
        item = path.stat()
        raw = path.read_bytes()
        if len(raw) > maximum or stat.S_IMODE(item.st_mode) != mode:
            raise RuntimeError("checkpoint read failed")
        return raw, item

    monkeypatch.setattr(runtime, "_atomic_no_replace_file", publish)
    monkeypatch.setattr(runtime, "_read_exact_file", read)
    return run_root


def _cleanup_transaction_snapshot(*, run_id: str, fixture_sha256: str):
    cleanup_facts = {"facts_sha256": "1" * 64}
    snapshot = {
        "stopped": list(runtime.CAPABILITY_PRE_CLEANUP_STOP_ORDER),
        "execution_cleanup": {},
        "connector_cleanup": {},
        "full_gateway_unit_restore": {},
        "removed_overlay_artifacts": {},
        "full_canary_stopped_preflight": {"report_sha256": "2" * 64},
        "approval_retirement": {},
        "producer_foundation": {},
        "credential_consumer_stop_proof": {},
        "bitrix_receipt_key_pair_retirement": {},
        "bitrix_receipt_key_pair_absence": {
            "both_pair_members_absent": True
        },
        "retirements": {},
        "retirement_receipt_sha256s": {},
        "credential_absence": {},
        "credentials_absent": True,
        "cleanup_facts": cleanup_facts,
    }
    publication = {
        "facts": cleanup_facts,
        "facts_file_sha256": "3" * 64,
    }
    cleanup_receipt = {
        "authority_role": runtime.CAPABILITY_OBSERVER_ROLE,
        "payload": {
            "run_id": run_id,
            "fixture_sha256": fixture_sha256,
        },
    }
    return snapshot, publication, cleanup_receipt


def _publish_cleanup_transaction_prefix(
    plan,
    *,
    run_id,
    fixture_sha256,
    stop_after=None,
):
    snapshot, publication, cleanup_receipt = _cleanup_transaction_snapshot(
        run_id=run_id,
        fixture_sha256=fixture_sha256,
    )
    stage_payloads = (
        ("facts_collected", snapshot),
        (
            "facts_published",
            {"cleanup_facts_publication": publication},
        ),
        (
            "signed_receipt_verified",
            {
                "cleanup_facts_sha256": snapshot["cleanup_facts"][
                    "facts_sha256"
                ],
                "cleanup_facts_file_sha256": publication[
                    "facts_file_sha256"
                ],
                "cleanup_receipt": cleanup_receipt,
                "cleanup_receipt_file_sha256": runtime._sha256_bytes(
                    runtime._canonical_bytes(cleanup_receipt)
                ),
                "receipt_readback_verified": True,
                "signature_and_native_evidence_verified": True,
            },
        ),
    )
    transaction = dict(runtime.load_capability_cleanup_transaction(
        plan,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    ))
    for stage, payload in stage_payloads:
        if stage in transaction:
            continue
        checkpoint = runtime.publish_capability_cleanup_transaction_checkpoint(
            plan,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            stage=stage,
            payload=payload,
            existing=transaction,
            recorded_at_unix_ms=1_000 + len(transaction),
        )
        transaction[stage] = checkpoint
        if stage == stop_after:
            break
    return transaction


@pytest.mark.parametrize(
    "crash_stage",
    tuple(stage for _ordinal, stage, _name in runtime.CAPABILITY_CLEANUP_TRANSACTION_STAGES),
)
def test_cleanup_transaction_recovers_after_every_durable_stage(
    crash_stage,
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    run_id = "run-crash-window"
    fixture_sha256 = "f" * 64
    _install_cleanup_transaction_test_io(tmp_path, monkeypatch, run_id)
    prefix_stage = crash_stage if crash_stage in {
        "facts_collected",
        "facts_published",
        "signed_receipt_verified",
    } else None
    _publish_cleanup_transaction_prefix(
        plan,
        run_id=run_id,
        fixture_sha256=fixture_sha256,
        stop_after=prefix_stage,
    )
    # Simulated process restart: only fixed files survive.
    transaction = dict(runtime.load_capability_cleanup_transaction(
        plan,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    ))
    if "signed_receipt_verified" not in transaction:
        transaction = _publish_cleanup_transaction_prefix(
            plan,
            run_id=run_id,
            fixture_sha256=fixture_sha256,
        )

    final_services = {
        unit: {"unit": unit} for unit in runtime.CAPABILITY_STOP_ORDER
    }
    monkeypatch.setattr(runtime, "_service_stopped", lambda _state: True)
    monkeypatch.setattr(
        runtime,
        "_attempt_capability_stop_order",
        lambda _stop, *, stop_order: (list(stop_order), []),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_services",
        lambda **_kwargs: final_services,
    )
    monkeypatch.setattr(
        runtime,
        "build_capability_stop_proof",
        lambda _plan, _services, **kwargs: {
            "observed_at_unix": kwargs.get("observed_at_unix") or 10,
            "proof": "stopped",
        },
    )
    monkeypatch.setattr(
        runtime,
        "build_capability_observer_stop_receipt",
        lambda _plan, _state, **kwargs: {
            "stopped_at_unix_ms": kwargs.get("stopped_at_unix_ms") or 20,
            "receipt": "observer-stopped",
        },
    )
    monkeypatch.setattr(
        runtime,
        "_expected_cleanup_expiry_watchdog_authorities",
        lambda *_args, **_kwargs: ("a" * 64,),
    )
    watchdog = {
        "all_timers_disabled": True,
        "all_unit_files_absent": True,
        "authority_receipt_sha256s": ["a" * 64],
    }
    monkeypatch.setattr(
        runtime,
        "disarm_all_capability_expiry_watchdogs",
        lambda **_kwargs: watchdog,
    )

    def finalization(_plan, **kwargs):
        finalized_at = kwargs.get("finalized_at_unix_ms") or 30
        unsigned = {
            "schema": runtime.CAPABILITY_CLEANUP_FINALIZATION_SCHEMA,
            "run_id": run_id,
            "fixture_sha256": fixture_sha256,
            "finalized_at_unix_ms": finalized_at,
        }
        return {
            **unsigned,
            "finalization_sha256": runtime._sha256_json(unsigned),
        }

    monkeypatch.setattr(
        runtime,
        "build_capability_cleanup_finalization",
        finalization,
    )
    for path in (
        producers.DEFAULT_READINESS_PATH,
        producers.DEFAULT_PROBE_CATALOG_PATH,
        producers.DEFAULT_OWNER_GRANT_PATH,
    ):
        monkeypatch.setattr(
            producers,
            next(
                name
                for name, value in vars(producers).items()
                if value is path and name.startswith("DEFAULT_")
            ),
            tmp_path / f"absent-{path.name}",
        )
    callback_calls = []
    admission_unsigned = {
        "schema": "muncho-production-capability-api-admission-retirement.v1",
        "run_id": run_id,
        "fixture_sha256": fixture_sha256,
        "catalog_absent": True,
        "owner_grant_absent": True,
    }
    admission = {
        **admission_unsigned,
        "receipt_sha256": runtime._sha256_json(admission_unsigned),
    }
    fleet = {
        "run_id": run_id,
        "fixture_sha256": fixture_sha256,
        "retired": True,
        "absence_verified": True,
    }

    lifecycle = object.__new__(runtime.CapabilityCanaryLifecycle)
    lifecycle.plan = plan
    lifecycle.full_plan = _full_plan()
    lifecycle.runner = lambda _command: None
    lifecycle._pending_deferred_start = {"pending": True}
    original_publish = runtime.publish_capability_cleanup_transaction_checkpoint
    crashed = False

    def crash_after_publish(*args, **kwargs):
        nonlocal crashed
        checkpoint = original_publish(*args, **kwargs)
        if kwargs["stage"] == crash_stage and crash_stage not in {
            "facts_collected",
            "facts_published",
            "signed_receipt_verified",
        } and not crashed:
            crashed = True
            raise RuntimeError(f"crash-after-{crash_stage}")
        return checkpoint

    monkeypatch.setattr(
        runtime,
        "publish_capability_cleanup_transaction_checkpoint",
        crash_after_publish,
    )

    def retire_admission():
        callback_calls.append("admission")
        return admission

    def retire_fleet():
        callback_calls.append("fleet")
        return fleet

    if crash_stage in {
        "observer_stopped",
        "runtime_retired",
        "watchdogs_disarmed",
        "finalized",
    }:
        with pytest.raises(RuntimeError, match=f"crash-after-{crash_stage}"):
            lifecycle._complete_live_cleanup_transaction_locked(
                transaction=transaction,
                cleanup_run_id=run_id,
                cleanup_fixture_sha256=fixture_sha256,
                producer_activation_retirer=retire_fleet,
                admission_input_retirer=retire_admission,
            )
        transaction = dict(runtime.load_capability_cleanup_transaction(
            plan,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
        ))
    monkeypatch.setattr(
        runtime,
        "publish_capability_cleanup_transaction_checkpoint",
        original_publish,
    )
    result = lifecycle._complete_live_cleanup_transaction_locked(
        transaction=transaction,
        cleanup_run_id=run_id,
        cleanup_fixture_sha256=fixture_sha256,
        producer_activation_retirer=retire_fleet,
        admission_input_retirer=retire_admission,
    )

    assert result["services_stopped"] is True
    assert result["cleanup_finalization"]["run_id"] == run_id
    assert tuple(result["cleanup_transaction"]) == tuple(
        stage
        for _ordinal, stage, _name in runtime.CAPABILITY_CLEANUP_TRANSACTION_STAGES
    )
    assert callback_calls.count("admission") == 1
    assert callback_calls.count("fleet") == 1


def test_cleanup_transaction_never_stops_observer_before_verified_receipt(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    run_id = "run-no-receipt"
    fixture_sha256 = "e" * 64
    _install_cleanup_transaction_test_io(tmp_path, monkeypatch, run_id)
    transaction = _publish_cleanup_transaction_prefix(
        plan,
        run_id=run_id,
        fixture_sha256=fixture_sha256,
        stop_after="facts_published",
    )
    stop_calls = []
    monkeypatch.setattr(
        runtime,
        "_attempt_capability_stop_order",
        lambda *_args, **_kwargs: stop_calls.append(True),
    )
    lifecycle = object.__new__(runtime.CapabilityCanaryLifecycle)
    lifecycle.plan = plan

    with pytest.raises(RuntimeError, match="prefix is incomplete"):
        lifecycle._complete_live_cleanup_transaction_locked(
            transaction=transaction,
            cleanup_run_id=run_id,
            cleanup_fixture_sha256=fixture_sha256,
            producer_activation_retirer=None,
            admission_input_retirer=None,
        )

    assert stop_calls == []


def test_cleanup_finalization_is_strictly_after_observer_and_activation_stop():
    plan = _plan()
    all_stopped = _watchdog_stopped_services()
    stop = runtime.build_capability_stop_proof(
        plan,
        all_stopped,
        observed_at_unix=200,
    )
    observer = runtime.build_capability_observer_stop_receipt(
        plan,
        all_stopped[runtime.CAPABILITY_OBSERVER_UNIT],
        stopped_at_unix_ms=200_100,
    )
    fleet_unsigned = {
        "schema": "muncho-production-capability-fleet-retirement.v1",
        "readiness_sha256": "1" * 64,
        "foundation_sha256": "2" * 64,
        "release_sha": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": "3" * 64,
        "run_id": "capability-run-1",
        "path": "/run/muncho-capability-canary/producer-activation.json",
        "retirement_intent_sha256": "7" * 64,
        "retired": True,
        "absence_verified": True,
        "retired_at_unix_ms": 200_200,
    }
    fleet = {
        **fleet_unsigned,
        "receipt_sha256": runtime._sha256_json(fleet_unsigned),
    }
    cleanup = {
        "schema": "muncho-production-capability-canary-signed-receipt.v1",
        "authority_role": "gateway_observer",
        "key_id": "4" * 64,
        "signature_algorithm": "ed25519",
        "payload": {
            "run_id": "capability-run-1",
            "fixture_sha256": "3" * 64,
            "observed_at_unix_ms": 200_000,
        },
        "native_evidence": {},
        "signature": "5" * 128,
    }
    admission_unsigned = {
        "schema": "muncho-production-capability-api-admission-retirement.v1",
        "run_id": "capability-run-1",
        "fixture_sha256": "3" * 64,
        "session_id": "capability-session-1",
        "capability_epoch_sha256": "8" * 64,
        "challenge_sha256": "9" * 64,
        "owner_authority_path": (
            "/var/lib/muncho-capability-canary-evidence/capability-run-1/"
            "api-admission-owner-authority.json"
        ),
        "owner_authority_sha256": "a" * 64,
        "install_publication_sha256": "b" * 64,
        "intent_sha256": "c" * 64,
        "catalog_sha256": "d" * 64,
        "owner_grant_sha256": "e" * 64,
        "catalog_absent": True,
        "owner_grant_absent": True,
        "retired_at_unix_ms": 200_250,
    }
    admission = {
        **admission_unsigned,
        "receipt_sha256": runtime._sha256_json(admission_unsigned),
    }
    watchdog_paths = runtime._expiry_watchdog_paths("f" * 32)
    watchdog_unsigned = {
        "operation": "normal_lifecycle_disarm",
        "watchdog_id": "f" * 32,
        "watchdog_authority_sha256": "1" * 64,
        "disarm_intent_path": str(watchdog_paths["disarm_intent"]),
        "disarm_intent_sha256": "2" * 64,
        "timer_name": watchdog_paths["timer_name"],
        "timer_disabled": True,
        "timer_wants_absent": True,
        "service_absent": True,
        "timer_absent": True,
        "completed_at_unix": 200,
        "ok": True,
        "schema": runtime.CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA,
        "receipt_path": str(watchdog_paths["disarm_completion"]),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    watchdog_completion = {
        **watchdog_unsigned,
        "receipt_sha256": runtime._sha256_json(watchdog_unsigned),
    }
    watchdog_retirement = {
        "watchdog_count": 1,
        "authority_receipt_sha256s": ["1" * 64],
        "authority_set_sha256": runtime._sha256_json({
            "authority_receipt_sha256s": ["1" * 64],
        }),
        "retired": [watchdog_completion],
        "all_timers_disabled": True,
        "all_unit_files_absent": True,
    }
    finalization = runtime.build_capability_cleanup_finalization(
        plan,
        cleanup_receipt=cleanup,
        observer_stop_receipt=observer,
        service_stop_proof=stop,
        producer_fleet_retirement=fleet,
        admission_input_retirement=admission,
        expiry_watchdog_retirement=watchdog_retirement,
        expected_expiry_watchdog_authority_sha256s=("1" * 64,),
        producer_activation_absent=True,
        admission_inputs_absent=True,
        credentials_absent=True,
        bitrix_receipt_key_pair_absent=True,
        full_canary_stopped_preflight_sha256="6" * 64,
        finalized_at_unix_ms=200_300,
    )
    assert finalization["cleanup_receipt_sha256"] == runtime._sha256_json(cleanup)
    assert finalization["observer_stop_receipt"] == observer
    assert finalization["service_stop_proof"] == stop
    assert finalization["producer_fleet_retirement"] == fleet
    assert finalization["admission_input_retirement"] == admission
    assert finalization["expiry_watchdog_retirement"] == watchdog_retirement

    def finalize_with(
        retirement,
        expected,
    ):
        return runtime.build_capability_cleanup_finalization(
            plan,
            cleanup_receipt=cleanup,
            observer_stop_receipt=observer,
            service_stop_proof=stop,
            producer_fleet_retirement=fleet,
            admission_input_retirement=admission,
            expiry_watchdog_retirement=retirement,
            expected_expiry_watchdog_authority_sha256s=expected,
            producer_activation_absent=True,
            admission_inputs_absent=True,
            credentials_absent=True,
            bitrix_receipt_key_pair_absent=True,
            full_canary_stopped_preflight_sha256="6" * 64,
            finalized_at_unix_ms=200_300,
        )

    empty_authorities = {
        "watchdog_count": 0,
        "authority_receipt_sha256s": [],
        "authority_set_sha256": runtime._sha256_json({
            "authority_receipt_sha256s": [],
        }),
        "retired": [],
        "all_timers_disabled": True,
        "all_unit_files_absent": True,
    }
    with pytest.raises(ValueError, match="expiry watchdog retirement"):
        finalize_with(empty_authorities, ())
    missing_completion = {
        **watchdog_retirement,
        "retired": [],
    }
    with pytest.raises(ValueError, match="expiry watchdog retirement"):
        finalize_with(missing_completion, ("1" * 64,))
    foreign_authority = {
        **watchdog_retirement,
        "authority_receipt_sha256s": ["4" * 64],
        "authority_set_sha256": runtime._sha256_json({
            "authority_receipt_sha256s": ["4" * 64],
        }),
    }
    with pytest.raises(ValueError, match="expiry watchdog retirement"):
        finalize_with(foreign_authority, ("1" * 64,))
    with pytest.raises(ValueError, match="expiry watchdog retirement"):
        finalize_with(watchdog_retirement, ("1" * 64, "1" * 64))

    with pytest.raises(ValueError, match="terminal truth"):
        runtime.build_capability_cleanup_finalization(
            plan,
            cleanup_receipt=cleanup,
            observer_stop_receipt=observer,
            service_stop_proof=stop,
            producer_fleet_retirement=fleet,
            admission_input_retirement=admission,
            expiry_watchdog_retirement=watchdog_retirement,
            expected_expiry_watchdog_authority_sha256s=("1" * 64,),
            producer_activation_absent=True,
            admission_inputs_absent=True,
            credentials_absent=True,
            bitrix_receipt_key_pair_absent=True,
            full_canary_stopped_preflight_sha256="6" * 64,
            finalized_at_unix_ms=200_199,
        )


def test_append_only_artifact_collision_fails_without_overwrite(tmp_path):
    path = tmp_path / "journal" / ("b" * 32) / "install-intent.json"
    first = runtime._append_lease_artifact(
        path,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "lease_id": "b" * 32,
            "marker": "first",
        },
    )
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="collided with different bytes"):
        runtime._append_lease_artifact(
            path,
            schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
            value={
                "operation": "install_intent",
                "lease_id": "b" * 32,
                "marker": "second",
            },
        )
    assert path.read_bytes() == before
    assert json.loads(before)["receipt_sha256"] == first["receipt_sha256"]


def test_lease_journal_promotes_exact_fsynced_temporary_after_sigkill(tmp_path):
    journal = tmp_path / "journal"
    lease_id = "d" * 32
    paths = runtime._lease_artifact_paths(journal, lease_id)
    runtime._prepare_journal_directory(journal)
    runtime._prepare_journal_directory(paths.root)
    intent = runtime._append_lease_artifact(
        paths.install_intent,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "lease_id": lease_id,
            "marker": "exact-crash-recovery",
        },
    )
    temporary = paths.root / f".{paths.install_intent.name}.tmp"
    os.rename(paths.install_intent, temporary)

    [state] = runtime._journal_states(journal)

    assert state["install_intent"] == intent
    assert paths.install_intent.is_file()
    assert not os.path.lexists(temporary)


def test_lease_journal_rejects_conflicting_temporary_without_overwrite(tmp_path):
    journal = tmp_path / "journal"
    lease_id = "e" * 32
    paths = runtime._lease_artifact_paths(journal, lease_id)
    runtime._prepare_journal_directory(journal)
    runtime._prepare_journal_directory(paths.root)
    runtime._append_lease_artifact(
        paths.install_intent,
        schema=runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        value={
            "operation": "install_intent",
            "lease_id": lease_id,
            "marker": "winner",
        },
    )
    winner = paths.install_intent.read_bytes()
    unsigned = {
        "operation": "install_intent",
        "lease_id": lease_id,
        "marker": "conflict",
        "schema": runtime.CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
        "receipt_path": str(paths.install_intent),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    conflict = {
        **unsigned,
        "receipt_sha256": runtime._sha256_json(unsigned),
    }
    temporary = paths.root / f".{paths.install_intent.name}.tmp"
    temporary.write_bytes(runtime._canonical_bytes(conflict))
    temporary.chmod(0o400)

    with pytest.raises(RuntimeError, match="half-state is inconsistent"):
        runtime._journal_states(journal)

    assert paths.install_intent.read_bytes() == winner
    assert temporary.is_file()


def test_secret_install_rejects_dangling_target_and_retirement_substitution(
    tmp_path,
):
    plan = _plan()
    now = int(time.time())
    frame = runtime.build_secret_lease_frame(
        kind="codex_access_token",
        secret=_jwt(now + 3_600),
        plan_sha256=plan.sha256,
        owner_subject_sha256="8" * 64,
        now_unix=now,
        lease_id="7" * 32,
    )
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    auth_path = tmp_path / "profile/auth.json"
    auth_path.parent.mkdir(mode=0o700)
    os.chown(
        auth_path.parent,
        plan.identities.gateway_uid,
        plan.identities.gateway_gid,
    )
    auth_path.parent.chmod(0o700)
    auth_path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.provision_secret_lease(
            plan,
            metadata,
            secret,
            auth_path=auth_path,
            journal_path=tmp_path / "control/codex.json",
        )

    auth_path.unlink()
    metadata, secret = runtime.read_secret_lease_frame(
        io.BytesIO(frame), expected_kind="codex_access_token", now_unix=now
    )
    journal = tmp_path / "control/codex.json"
    runtime.provision_secret_lease(
        plan,
        metadata,
        secret,
        auth_path=auth_path,
        journal_path=journal,
    )
    original_mode = stat.S_IMODE(auth_path.stat().st_mode)
    auth_path.unlink()
    auth_path.write_text("substitute", encoding="utf-8")
    auth_path.chmod(original_mode)
    with pytest.raises(RuntimeError, match="identity is unsafe"):
        runtime.retire_secret_lease(
            kind="codex_access_token",
            target=auth_path,
            journal=journal,
            stop_proof=_stop_proof(plan),
        )


def test_overlay_cleanup_removes_only_exact_installed_bytes(tmp_path, monkeypatch):
    plan = _plan()
    target = tmp_path / "overlay.json"
    payload = b'{"sealed":true}\n'
    target.write_bytes(payload)
    target.chmod(0o600)
    identity = target.stat()
    binding = {
        "overlay": (
            target,
            payload,
            0o600,
            identity.st_uid,
            identity.st_gid,
            frozenset(),
        )
    }
    monkeypatch.setattr(runtime, "_capability_artifact_bindings", lambda *_: binding)

    removed = runtime._remove_exact_overlay_artifacts(plan, _full_plan())
    assert removed["overlay"]["removed"] is True
    assert not target.exists()

    target.write_bytes(b'{"substituted":true}\n')
    target.chmod(0o600)
    with pytest.raises(RuntimeError, match="substitution"):
        runtime._remove_exact_overlay_artifacts(plan, _full_plan())
    assert target.exists()


def test_browser_controller_config_is_exact_af_unix_and_rejects_tamper():
    plan = _plan()
    service = json.loads(runtime.render_browser_config(plan))
    client = runtime.capability_browser_controller_client_mapping(plan)

    assert service["socket_path"] == str(runtime.DEFAULT_BROWSER_SOCKET)
    assert service["allowed_client_uid"] == plan.identities.gateway_uid
    assert service["node_path"] == str(plan.browser_node)
    assert service["chrome_path"] == str(plan.browser_executable)
    assert client == {
        "schema": runtime.BROWSER_CONTROLLER_CLIENT_SCHEMA,
        "socket_path": str(runtime.DEFAULT_BROWSER_SOCKET),
        "server_uid": plan.identities.browser_uid,
        "artifact_root": str(runtime.DEFAULT_BROWSER_ARTIFACT_ROOT),
        "connect_timeout_seconds": 5,
        "request_timeout_seconds": runtime.BROWSER_COMMAND_TIMEOUT_SECONDS,
    }
    encoded = json.dumps({"service": service, "client": client})
    assert "cdp" not in encoded.lower()
    assert "remote-debugging" not in encoded
    assert "9222" not in encoded

    tampered = plan.to_mapping()
    tampered["browser"]["socket_path"] = "/tmp/controller.sock"
    tampered["capability_plan_sha256"] = runtime._sha256_json({
        key: value for key, value in tampered.items() if key != "capability_plan_sha256"
    })
    with pytest.raises(ValueError, match="AF_UNIX controller"):
        runtime.CapabilityCanaryPlan.from_mapping(tampered)


def _patch_browser_principals(monkeypatch, plan, *, browser_present=True):
    browser_user = SimpleNamespace(
        pw_name=plan.identities.browser_user,
        pw_uid=plan.identities.browser_uid,
        pw_gid=plan.identities.browser_gid,
        pw_dir=runtime.DEFAULT_BROWSER_HOME,
        pw_shell=runtime.DEFAULT_BROWSER_SHELL,
    )
    browser_group = SimpleNamespace(
        gr_name=plan.identities.browser_group,
        gr_gid=plan.identities.browser_gid,
        gr_mem=[],
    )
    projector_user = SimpleNamespace(
        pw_name=runtime.DEFAULT_PROJECTOR_USER,
        pw_uid=2198,
        pw_gid=2298,
    )
    projector_group = SimpleNamespace(
        gr_name=runtime.DEFAULT_PROJECTOR_GROUP,
        gr_gid=2298,
        gr_mem=[],
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: (
            browser_user
            if name == plan.identities.browser_user and browser_present
            else projector_user
            if name == runtime.DEFAULT_PROJECTOR_USER
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: (
            browser_group
            if name == plan.identities.browser_group and browser_present
            else projector_group
            if name == runtime.DEFAULT_PROJECTOR_GROUP
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda uid: (
            browser_user
            if uid == plan.identities.browser_uid and browser_present
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda gid: (
            browser_group
            if gid == plan.identities.browser_gid and browser_present
            else None
        ),
    )
    monkeypatch.setattr(
        runtime.os,
        "getgrouplist",
        lambda _user, primary_gid: [primary_gid],
    )
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda _name, _uid, _gid: (
            (
                [
                    (
                        plan.identities.browser_user,
                        plan.identities.browser_uid,
                        plan.identities.browser_gid,
                        runtime.DEFAULT_BROWSER_HOME,
                        runtime.DEFAULT_BROWSER_SHELL,
                    )
                ],
                [plan.identities.browser_user],
            )
            if browser_present
            else ([], [])
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_group_slot_inventory",
        lambda _name, _gid: (
            [
                (
                    plan.identities.browser_group,
                    plan.identities.browser_gid,
                    (),
                )
            ]
            if browser_present
            else []
        ),
    )


def test_browser_host_receipt_is_exact_and_allows_only_create_only_absence(
    monkeypatch,
):
    plan = _plan()
    _patch_browser_principals(monkeypatch, plan)
    present = runtime.browser_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=False
    )
    assert present["browser"]["state"] == "present_exact"
    assert present["browser"]["supplementary_group_ids"] == [
        plan.identities.browser_gid
    ]
    assert present["projector"] == {
        "state": "present",
        "user": runtime.DEFAULT_PROJECTOR_USER,
        "group": runtime.DEFAULT_PROJECTOR_GROUP,
        "uid": 2198,
        "gid": 2298,
    }
    assert present["receipt_sha256"] == runtime._sha256_json({
        key: value for key, value in present.items() if key != "receipt_sha256"
    })

    _patch_browser_principals(monkeypatch, plan, browser_present=False)
    absent = runtime.browser_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=True
    )
    assert absent["browser"]["state"] == "absent_create_only_slot"
    with pytest.raises(RuntimeError, match="principal is absent"):
        runtime.browser_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )


def test_browser_host_receipt_rejects_raw_nss_aliases(monkeypatch):
    plan = _plan()
    _patch_browser_principals(monkeypatch, plan)
    exact_passwd_inventory = runtime._capability_passwd_slot_inventory
    exact_group_inventory = runtime._capability_group_slot_inventory

    def hidden_primary(*args):
        rows, names = exact_passwd_inventory(*args)
        return rows, [*names, "unrelated-primary-user"]

    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        hidden_primary,
    )
    with pytest.raises(RuntimeError, match="slot collides or drifted"):
        runtime.browser_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )

    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        exact_passwd_inventory,
    )

    def duplicate_group(*args):
        rows = exact_group_inventory(*args)
        return [*rows, ("browser-group-alias", plan.identities.browser_gid, ())]

    monkeypatch.setattr(
        runtime,
        "_capability_group_slot_inventory",
        duplicate_group,
    )
    with pytest.raises(RuntimeError, match="slot collides or drifted"):
        runtime.browser_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )


@pytest.mark.parametrize("role", ("mac_ops", "connector"))
def test_service_host_receipt_rejects_hidden_primary_nss_alias(
    monkeypatch,
    role,
):
    plan = _plan()
    identities = plan.identities
    user_name = getattr(identities, f"{role}_user")
    group_name = getattr(identities, f"{role}_group")
    uid = getattr(identities, f"{role}_uid")
    gid = getattr(identities, f"{role}_gid")
    user = SimpleNamespace(
        pw_name=user_name,
        pw_uid=uid,
        pw_gid=gid,
        pw_dir="/nonexistent",
        pw_shell="/usr/sbin/nologin",
    )
    group = SimpleNamespace(gr_name=group_name, gr_gid=gid, gr_mem=[])
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: user if name == user_name else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda value: user if value == uid else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: group if name == group_name else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda value: group if value == gid else None,
    )
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda *_args: (
            [(user_name, uid, gid, "/nonexistent", "/usr/sbin/nologin")],
            [user_name],
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_group_slot_inventory",
        lambda *_args: [(group_name, gid, ())],
    )
    monkeypatch.setattr(runtime.os, "getgrouplist", lambda *_args: [gid])

    exact = runtime.service_host_identity_receipt(
        plan,
        _full_plan(),
        role=role,
        allow_create_only_absence=False,
    )
    assert exact["state"] == "present_exact"

    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda *_args: (
            [(user_name, uid, gid, "/nonexistent", "/usr/sbin/nologin")],
            [user_name, "unrelated-primary-user"],
        ),
    )
    with pytest.raises(RuntimeError, match="slot collides or drifted"):
        runtime.service_host_identity_receipt(
            plan,
            _full_plan(),
            role=role,
            allow_create_only_absence=False,
        )


def test_bitrix_identity_accepts_only_exact_create_sequence_and_rejects_aliases(
    monkeypatch,
):
    service_uid = 2110
    service_gid = 2210
    client_gid = 2211
    service_user = SimpleNamespace(
        pw_name="muncho-edge-bitrix",
        pw_uid=service_uid,
        pw_gid=service_gid,
        pw_dir="/nonexistent",
        pw_shell="/usr/sbin/nologin",
    )
    service_group = SimpleNamespace(
        gr_name="muncho-edge-bitrix",
        gr_gid=service_gid,
        gr_mem=[],
    )
    client_group = SimpleNamespace(
        gr_name="muncho-edge-bitrix-c",
        gr_gid=client_gid,
        gr_mem=[],
    )
    present = {"user": True, "service_group": True, "client_group": True}
    client_primary_users: list[str] = []

    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: (
            service_user if name == "muncho-edge-bitrix" and present["user"] else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda value: (
            service_user if value == service_uid and present["user"] else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: (
            service_group
            if name == "muncho-edge-bitrix" and present["service_group"]
            else client_group
            if name == "muncho-edge-bitrix-c" and present["client_group"]
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda value: (
            service_group
            if value == service_gid and present["service_group"]
            else client_group
            if value == client_gid and present["client_group"]
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda *_args: (
            (
                [
                    (
                        "muncho-edge-bitrix",
                        service_uid,
                        service_gid,
                        "/nonexistent",
                        "/usr/sbin/nologin",
                    )
                ],
                ["muncho-edge-bitrix"],
            )
            if present["user"]
            else ([], [])
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_group_slot_inventory",
        lambda name, _gid: (
            [("muncho-edge-bitrix", service_gid, ())]
            if name == "muncho-edge-bitrix" and present["service_group"]
            else [("muncho-edge-bitrix-c", client_gid, ())]
            if name == "muncho-edge-bitrix-c" and present["client_group"]
            else []
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_primary_group_user_names",
        lambda _gid: list(client_primary_users),
    )
    monkeypatch.setattr(runtime.os, "getgrouplist", lambda *_args: [service_gid])

    observe = lambda allow: runtime._observe_bitrix_foundation_identity(
        service_uid=service_uid,
        service_gid=service_gid,
        socket_client_gid=client_gid,
        allow_absence=allow,
    )
    assert observe(False)["state"] == "present_exact"
    client_primary_users.append("unrelated-primary-user")
    with pytest.raises(RuntimeError, match="slot collides or drifted"):
        observe(False)
    client_primary_users.clear()

    present["user"] = False
    assert observe(True)["state"] == ("groups_present_user_absent_create_only_slot")
    present["client_group"] = False
    assert observe(True)["state"] == "service_group_present_create_only_slot"
    present["service_group"] = False
    assert observe(True)["state"] == "absent_create_only_slot"


def test_service_identity_foundation_creates_only_clean_slots_after_publication(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    full = _full_plan()
    commands: list[tuple[str, ...]] = []
    publications: list[tuple[Path, bytes]] = []
    observations = {
        role: iter(("absent_create_only_slot", "present_exact"))
        for role in ("mac_ops", "connector")
    }

    def observe(_plan, _full, *, role, **_kwargs):
        return _service_identity_observation(
            plan,
            role=role,
            state=next(observations[role]),
        )

    def run(command):
        commands.append(command.argv)
        return runtime.subprocess.CompletedProcess(command.argv, 0, b"", b"")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_SERVICE_IDENTITY_FOUNDATION_ROOT",
        tmp_path / "service-identities",
    )

    receipt = runtime.ensure_service_identities_create_only(
        plan,
        full,
        runner=run,
        observer=observe,
        plan_publication_loader=lambda _plan: {"receipt_sha256": "a" * 64},
        publisher=lambda path, payload: publications.append((path, payload)),
    )

    assert commands == [
        (
            runtime.GROUPADD,
            "--system",
            "--gid",
            str(plan.identities.mac_ops_gid),
            "--",
            plan.identities.mac_ops_group,
        ),
        (
            runtime.USERADD,
            "--system",
            "--uid",
            str(plan.identities.mac_ops_uid),
            "--gid",
            plan.identities.mac_ops_group,
            "--home-dir",
            "/nonexistent",
            "--no-create-home",
            "--shell",
            "/usr/sbin/nologin",
            "--",
            plan.identities.mac_ops_user,
        ),
        (
            runtime.GROUPADD,
            "--system",
            "--gid",
            str(plan.identities.connector_gid),
            "--",
            plan.identities.connector_group,
        ),
        (
            runtime.USERADD,
            "--system",
            "--uid",
            str(plan.identities.connector_uid),
            "--gid",
            plan.identities.connector_group,
            "--home-dir",
            "/nonexistent",
            "--no-create-home",
            "--shell",
            "/usr/sbin/nologin",
            "--",
            plan.identities.connector_user,
        ),
    ]
    assert receipt["created"] == [
        "mac_ops_group",
        "mac_ops_user",
        "connector_group",
        "connector_user",
    ]
    assert receipt["plan_publication_receipt_sha256"] == "a" * 64
    assert receipt["retained_dormant_on_rollback"] is True
    assert publications == [
        (Path(receipt["receipt_path"]), runtime._canonical_bytes(receipt))
    ]


def test_service_identity_foundation_rejects_collision_before_any_mutation(
    monkeypatch,
):
    plan = _plan()
    calls: list[object] = []

    def collision(*_args, role, **_kwargs):
        calls.append(("observe", role))
        raise RuntimeError(f"capability {role} UID is owned by another user")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(runtime, "_lifecycle_lock", contextlib.nullcontext)
    with pytest.raises(RuntimeError, match="UID is owned"):
        runtime.ensure_service_identities_create_only(
            plan,
            _full_plan(),
            runner=lambda command: calls.append(("mutate", command.argv)),
            observer=collision,
            plan_publication_loader=lambda _plan: {"receipt_sha256": "a" * 64},
            publisher=lambda *_args: calls.append("publish"),
        )

    assert calls == [("observe", "mac_ops")]


def test_browser_identity_foundation_is_create_only_and_receipted(monkeypatch):
    plan = _plan()
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    before = {
        "receipt_sha256": "1" * 64,
        "browser": {"state": "absent_create_only_slot"},
    }
    after = {
        "receipt_sha256": "2" * 64,
        "browser": {"state": "present_exact"},
    }
    observations = iter((before, after))
    commands = []

    def observer(*_args, **_kwargs):
        return next(observations)

    def runner(command):
        commands.append(command.argv)
        return __import__("subprocess").CompletedProcess(
            command.argv, 0, stdout=b"", stderr=b""
        )

    receipt = runtime.ensure_browser_identity_create_only(
        plan,
        _full_plan(),
        runner=runner,
        observer=observer,
    )
    assert [command[0] for command in commands] == [
        runtime.GROUPADD,
        runtime.USERADD,
    ]
    assert receipt["created_group"] is True
    assert receipt["created_user"] is True
    assert receipt["retained_dormant_on_rollback"] is True
    assert receipt["host_identity"] == after
    assert receipt["receipt_sha256"] == runtime._sha256_json({
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_sha256", "host_identity"}
    })


def test_browser_userns_and_principal_version_smokes_are_exact():
    plan = _plan()
    observed_paths = []

    def reader(path):
        observed_paths.append(path)
        return 1 if "unprivileged" in str(path) else 15452

    userns = runtime.browser_userns_preflight(reader=reader)
    assert userns["sandbox_required"] is True
    assert len(observed_paths) == 2
    with pytest.raises(RuntimeError, match="sandbox is disabled"):
        runtime.browser_userns_preflight(reader=lambda _path: 0)

    commands = []

    def runner(command):
        commands.append(command.argv)
        return __import__("subprocess").CompletedProcess(
            command.argv,
            0,
            stdout=(
                f"Google Chrome for Testing {runtime.RELEASE_CHROME_VERSION}\n"
            ).encode(),
            stderr=b"",
        )

    proof = runtime.browser_principal_version_smoke(plan, runner=runner)
    assert proof["uid"] == plan.identities.browser_uid
    assert commands[0][:6] == (
        runtime.RUNUSER,
        "--user",
        plan.identities.browser_user,
        "--group",
        plan.identities.browser_group,
        "--",
    )


def test_live_browser_preflight_binds_af_unix_socket_to_controller_mainpid():
    plan = _plan()
    state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4242,
        "FragmentPath": str(runtime.DEFAULT_BROWSER_UNIT_PATH),
        "DropInPaths": "",
        "Type": "notify",
        "NotifyAccess": "main",
    }
    socket_state = SimpleNamespace(
        st_mode=stat.S_IFSOCK | 0o660,
        st_uid=plan.identities.browser_uid,
        st_gid=plan.identities.browser_gid,
        st_dev=11,
        st_ino=12,
    )

    proof = runtime.browser_service_runtime_preflight(
        plan,
        state,
        proc_stat=lambda _path: SimpleNamespace(
            st_uid=plan.identities.browser_uid,
            st_gid=plan.identities.browser_gid,
        ),
        socket_lstat=lambda _path: socket_state,
        listener_paths=lambda _pid: {str(runtime.DEFAULT_BROWSER_SOCKET)},
    )
    assert proof["main_pid"] == 4242
    assert proof["transport"] == "authenticated_af_unix"
    assert proof["socket_path"] == str(runtime.DEFAULT_BROWSER_SOCKET)
    with pytest.raises(RuntimeError, match="identity drifted"):
        runtime.browser_service_runtime_preflight(
            plan,
            state,
            proc_stat=lambda _path: SimpleNamespace(
                st_uid=plan.identities.gateway_uid,
                st_gid=plan.identities.gateway_gid,
            ),
            socket_lstat=lambda _path: socket_state,
            listener_paths=lambda _pid: {str(runtime.DEFAULT_BROWSER_SOCKET)},
        )


def test_canary_dependency_manifest_binds_exact_production_package(
    monkeypatch,
):
    plan = _plan()
    unsigned = {
        "schema": runtime.RUNTIME_DEPENDENCY_MANIFEST_SCHEMA,
        "release_revision": plan.revision,
        "agent_browser": {
            "version": runtime.RELEASE_AGENT_BROWSER_VERSION,
            "node_path": str(plan.browser_node),
            "node_sha256": plan.browser_node_sha256,
            "wrapper_path": str(plan.browser_wrapper),
            "wrapper_sha256": plan.browser_wrapper_sha256,
            "native_path": str(plan.browser_native),
            "native_sha256": plan.browser_native_sha256,
            "config_path": str(plan.agent_browser_config),
            "config_sha256": plan.agent_browser_config_sha256,
        },
        "chrome": {
            "version": runtime.RELEASE_CHROME_VERSION,
            "executable_path": str(plan.browser_executable),
            "executable_sha256": plan.browser_executable_sha256,
        },
        "python": {
            "distributions": {
                name: {"version": version}
                for name, version in runtime.RELEASE_DDGS_DISTRIBUTIONS.items()
            }
        },
        "secret_material_recorded": False,
    }
    manifest = {
        **unsigned,
        "manifest_sha256": runtime._sha256_json(unsigned),
    }
    raw = runtime._canonical_bytes(manifest) + b"\n"
    plan = replace(
        plan,
        runtime_dependency_manifest_sha256=runtime._sha256_bytes(raw),
    )
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda *_args, **_kwargs: (raw, SimpleNamespace()),
    )
    monkeypatch.setattr(
        runtime,
        "verify_release_runtime_dependency_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    proof = runtime.runtime_dependency_manifest_preflight(plan)

    assert proof["chrome_version"] == runtime.RELEASE_CHROME_VERSION
    assert proof["agent_browser_version"] == runtime.RELEASE_AGENT_BROWSER_VERSION
    assert proof["ddgs_version"] == "9.14.4"
    assert "/usr/bin/chromium" not in str(plan.browser_executable)


def test_contract_is_non_semantic_and_secret_free():
    contract = runtime.runtime_contract()
    assert contract["normal_gateway_loop"] is True
    assert contract["model_semantic_authority"] is True
    assert contract["codex_refresh_token_leased"] is False
    assert contract["discord_credential_in_gateway"] is False
    assert contract["mac_ops_credential_in_gateway"] is False
    assert contract["goal_judge_enabled"] is False
    assert contract["model_authored_goal_outcome_enabled"] is True
    assert contract["goal_continuations_enabled"] is True
    assert contract["goal_outcome_source"] == "todo.goal_outcome"
    assert contract["goal_manager"] == "hermes_cli.goals.GoalManager"
    assert contract["goal_max_turns"] == 0
    assert yaml.safe_load(runtime.render_gateway_config(_plan()))["goals"] == {
        "max_turns": 0,
    }
    assert contract["mcp_auto_discovery_enabled"] is False
    assert contract["gateway_event_hooks_enabled"] is False
    assert contract["shell_hooks_enabled"] is False
    assert contract["plugin_middleware_enabled"] is False
    assert contract["plugin_allowlist"] == [runtime.CAPABILITY_OBSERVER_PLUGIN]
    assert "mac_ops" in contract["toolsets"]
    assert contract["credential_bindings"] == list(
        runtime.CAPABILITY_CREDENTIAL_BINDINGS
    )
    assert contract["browser_identity"] == "dedicated_create_only_principal"
    assert contract["browser_gateway_access"] == (
        "authenticated_af_unix_controller_only"
    )
    assert contract["browser_sandbox"] == "unprivileged_user_namespace_required"
    assert contract["terminal_gateway_access"] == (
        "authenticated_af_unix_isolated_worker_only"
    )
    assert contract["terminal_network_access"] is False
    assert contract["workspace_policy"] == (
        "ephemeral_isolated_worker_lease_no_host_projection"
    )


def test_capability_effective_environment_is_exact_and_rejects_unknown():
    config = yaml.safe_load(runtime.render_gateway_config(_plan()))
    terminal = config["terminal"]
    env = {
        "HOME": str(runtime.DEFAULT_GATEWAY_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "hermes_gateway",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/usr/sbin/nologin",
        "TZ": "UTC",
        "USER": "hermes_gateway",
        "HERMES_CONFIG": str(runtime.DEFAULT_GATEWAY_CONFIG),
        "HERMES_HOME": str(runtime.DEFAULT_GATEWAY_PROFILE_HOME),
        "HERMES_EXEC_ASK": "1",
        "HERMES_MANAGED_DIR": str(runtime.DEFAULT_DISABLED_MANAGED_SCOPE),
        "HERMES_MAX_ITERATIONS": "90",
        "HERMES_QUIET": "1",
        "SSL_CERT_FILE": str(runtime.DEFAULT_GATEWAY_CA_BUNDLE),
        "GATEWAY_RELAY_URL": "unix:///run/muncho-discord-connector/connector.sock",
        "GATEWAY_RELAY_PLATFORMS": "discord",
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": terminal["isolated_worker_socket"],
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(
            terminal["isolated_worker_server_uid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(
            terminal["isolated_worker_server_gid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": str(
            terminal["isolated_worker_socket_uid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(
            terminal["isolated_worker_socket_gid"]
        ),
        "CREDENTIALS_DIRECTORY": "/run/credentials/hermes-cloud-gateway.service",
        "RUNTIME_DIRECTORY": str(runtime.DEFAULT_GATEWAY_RUNTIME),
        "STATE_DIRECTORY": str(runtime.DEFAULT_GATEWAY_HOME),
        "NOTIFY_SOCKET": "/run/systemd/notify",
        "SYSTEMD_EXEC_PID": str(os.getpid()),
        "_HERMES_GATEWAY": "1",
    }
    assert runtime.capability_gateway_effective_environment_is_sealed(env, config)
    env["DISCORD_BOT_TOKEN"] = "forbidden"
    assert not runtime.capability_gateway_effective_environment_is_sealed(env, config)


def test_plan_has_no_host_workspace_projection_or_legacy_execution_transport():
    plan = _plan()
    value = plan.to_mapping()
    assert "workspaces" not in value
    assert "writable_root" not in value
    assert value["execution_workspace"] == {
        "path": "/workspace",
        "host_projection_enabled": False,
        "read_only_binds": [],
        "ephemeral_across_worker_restart": True,
        "lease_quota_bytes": runtime.SERVICE_GLOBAL_QUOTA_BYTES,
        "lease_quota_entries": runtime.LEASE_QUOTA_ENTRIES,
    }
    serialized = json.dumps(value).lower()
    assert "/srv/" not in serialized
    assert "docker" not in serialized
    assert "cdp" not in serialized
    assert "9222" not in serialized


def test_execution_identity_foundation_is_distinct_and_create_only(monkeypatch):
    plan = _plan()
    worker_user = SimpleNamespace(
        pw_name=plan.identities.worker_user,
        pw_uid=plan.identities.worker_uid,
        pw_gid=plan.identities.worker_gid,
        pw_dir=runtime.DEFAULT_WORKER_HOME,
        pw_shell=runtime.DEFAULT_WORKER_SHELL,
    )
    worker_group = SimpleNamespace(
        gr_name=plan.identities.worker_group,
        gr_gid=plan.identities.worker_gid,
        gr_mem=[],
    )
    client_group = SimpleNamespace(
        gr_name=plan.identities.worker_client_group,
        gr_gid=plan.identities.worker_client_gid,
        gr_mem=[],
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_name",
        lambda name: worker_user if name == plan.identities.worker_user else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_passwd_by_uid",
        lambda uid: worker_user if uid == plan.identities.worker_uid else None,
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_name",
        lambda name: (
            worker_group
            if name == plan.identities.worker_group
            else client_group
            if name == plan.identities.worker_client_group
            else None
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_optional_group_by_gid",
        lambda gid: (
            worker_group
            if gid == plan.identities.worker_gid
            else client_group
            if gid == plan.identities.worker_client_gid
            else None
        ),
    )
    monkeypatch.setattr(
        runtime.os, "getgrouplist", lambda *_args: [plan.identities.worker_gid]
    )
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda *_args: (
            [
                (
                    plan.identities.worker_user,
                    plan.identities.worker_uid,
                    plan.identities.worker_gid,
                    runtime.DEFAULT_WORKER_HOME,
                    runtime.DEFAULT_WORKER_SHELL,
                )
            ],
            [plan.identities.worker_user],
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_capability_group_slot_inventory",
        lambda name, _gid: [
            (
                name,
                (
                    plan.identities.worker_gid
                    if name == plan.identities.worker_group
                    else plan.identities.worker_client_gid
                ),
                (),
            )
        ],
    )
    monkeypatch.setattr(
        runtime,
        "_capability_primary_group_user_names",
        lambda _gid: [],
    )
    receipt = runtime.execution_host_identity_receipt(
        plan, _full_plan(), allow_create_only_absence=False
    )
    assert receipt["worker"]["state"] == "present_exact"
    assert receipt["socket_client_group"]["state"] == "present_exact"
    assert receipt["worker"]["supplementary_group_ids"] == [plan.identities.worker_gid]

    exact_passwd_inventory = runtime._capability_passwd_slot_inventory
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        lambda *args: (
            exact_passwd_inventory(*args)[0],
            [plan.identities.worker_user, "unrelated-primary-user"],
        ),
    )
    with pytest.raises(RuntimeError, match="collides or drifted"):
        runtime.execution_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )
    monkeypatch.setattr(
        runtime,
        "_capability_passwd_slot_inventory",
        exact_passwd_inventory,
    )
    monkeypatch.setattr(
        runtime,
        "_capability_primary_group_user_names",
        lambda _gid: ["unrelated-primary-user"],
    )
    with pytest.raises(RuntimeError, match="collides or drifted"):
        runtime.execution_host_identity_receipt(
            plan, _full_plan(), allow_create_only_absence=False
        )
    monkeypatch.setattr(
        runtime,
        "_capability_primary_group_user_names",
        lambda _gid: [],
    )

    before = {
        "receipt_sha256": "1" * 64,
        "worker": {"state": "absent_create_only_slot"},
        "socket_client_group": {"state": "absent_create_only_slot"},
    }
    after = {
        "receipt_sha256": "2" * 64,
        "worker": {"state": "present_exact"},
        "socket_client_group": {"state": "present_exact"},
    }
    observations = iter((before, after))
    commands = []
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    foundation = runtime.ensure_execution_identities_create_only(
        plan,
        _full_plan(),
        observer=lambda *_args, **_kwargs: next(observations),
        runner=lambda command: (
            commands.append(command.argv)
            or __import__("subprocess").CompletedProcess(
                command.argv, 0, stdout=b"", stderr=b""
            )
        ),
    )
    assert foundation["created"] == [
        "worker_group",
        "worker_client_group",
        "worker_user",
    ]
    assert [command[0] for command in commands] == [
        runtime.GROUPADD,
        runtime.GROUPADD,
        runtime.USERADD,
    ]


def test_systemd252_and_live_worker_tmpfs_contract_are_exact(monkeypatch):
    plan = _plan()
    mountpoint = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_uid=0,
        st_gid=0,
    )
    monkeypatch.setattr(runtime.os, "lstat", lambda _path: mountpoint)
    proof = runtime.worker_systemd252_preflight(
        plan,
        runner=lambda command: __import__("subprocess").CompletedProcess(
            command.argv,
            0,
            stdout=b"systemd 252 (252.39-1~deb12u2)\n+PAM\n",
            stderr=b"",
        ),
    )
    assert proof["systemd_major"] == 252
    assert proof["contract"] == runtime.LEASE_TMPFS_PREFLIGHT_CONTRACT

    state = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "disabled",
        "MainPID": 4243,
        "FragmentPath": str(runtime.DEFAULT_WORKER_SERVICE_UNIT_PATH),
        "DropInPaths": "",
        "Type": "simple",
        "NotifyAccess": "none",
    }
    mountinfo = (
        b"101 99 0:55 / /var/lib/muncho-isolated-worker "
        b"rw,nosuid,nodev,relatime - tmpfs tmpfs "
        b"rw,size=4194304k,nr_inodes=200001,mode=700,uid=2107,gid=2208\n"
    )
    live = runtime.worker_tmpfs_runtime_preflight(
        plan,
        state,
        mountinfo_reader=lambda _path: mountinfo,
        path_lstat=lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=plan.identities.worker_uid,
            st_gid=plan.identities.worker_gid,
        ),
        path_statvfs=lambda _path: SimpleNamespace(
            f_blocks=runtime.SERVICE_GLOBAL_QUOTA_BYTES // 4096,
            f_frsize=4096,
            f_files=runtime.SERVICE_TMPFS_INODE_LIMIT,
        ),
    )
    assert live["filesystem"] == "tmpfs"
    assert live["mount_flags"] == ["nodev", "nosuid", "exec"]


def test_real_execution_readiness_helpers_are_both_required(monkeypatch):
    plan = _plan()
    observed = {}
    monkeypatch.setattr(runtime.os, "geteuid", lambda: plan.identities.gateway_uid)
    monkeypatch.setattr(runtime.os, "getegid", lambda: plan.identities.gateway_gid)

    def worker(**kwargs):
        observed["worker"] = kwargs
        return {
            "schema": runtime.WORKER_RECEIPT_SCHEMA,
            "lease_identity_sha256": hashlib.sha256(
                b"muncho-worker-readiness-v1\x00"
                + plan.revision.encode()
                + b"\x00"
                + plan.worker_config_sha256.encode()
            ).hexdigest(),
            "socket_path": str(runtime.DEFAULT_WORKER_SOCKET),
            "server_uid": plan.identities.worker_uid,
            "server_gid": plan.identities.worker_gid,
            "socket_uid": 0,
            "socket_gid": plan.identities.worker_client_gid,
            "execution_round_trip": True,
            "output_sha256": hashlib.sha256(
                b"MUNCHO_ISOLATED_WORKER_READY\n"
            ).hexdigest(),
            "secret_material_recorded": False,
        }

    def browser(**kwargs):
        observed["browser"] = kwargs
        return {
            "schema": runtime.BROWSER_RECEIPT_SCHEMA,
            "session_identity_sha256": hashlib.sha256(
                b"muncho-browser-readiness-v1\x00"
                + plan.revision.encode()
                + b"\x00"
                + plan.browser_config_sha256.encode()
            ).hexdigest(),
            "socket_path": str(runtime.DEFAULT_BROWSER_SOCKET),
            "server_uid": plan.identities.browser_uid,
            "command_round_trip": True,
            "secret_material_recorded": False,
        }

    monkeypatch.setattr(runtime, "attest_isolated_worker_execution", worker)
    monkeypatch.setattr(runtime, "attest_browser_controller_execution", browser)
    receipt = runtime.attest_capability_execution_readiness(plan)
    assert observed["worker"]["socket_path"] == runtime.DEFAULT_WORKER_SOCKET
    assert observed["worker"]["config_sha256"] == plan.worker_config_sha256
    assert observed["browser"]["config_sha256"] == plan.browser_config_sha256
    assert receipt["schema"] == runtime.CAPABILITY_EXECUTION_READINESS_SCHEMA
    assert receipt["isolated_worker"]["execution_round_trip"] is True
    assert receipt["browser_controller"]["command_round_trip"] is True

    monkeypatch.setattr(
        runtime,
        "attest_isolated_worker_execution",
        lambda **_kwargs: {**worker(**_kwargs), "unexpected": True},
    )
    with pytest.raises(RuntimeError, match="readiness receipt is invalid"):
        runtime.attest_capability_execution_readiness(plan)


def test_production_observation_marker_wait_is_exact_and_bounded(
    monkeypatch,
    tmp_path,
):
    plan = _plan()
    request = _production_observation_wait_request(plan, phase="after")
    raw = runtime._canonical_bytes(request)
    assert (
        runtime.read_production_observation_wait_request(io.BytesIO(raw), plan=plan)
        == request
    )

    marker_path = tmp_path / "awaiting-production-after.json"
    marker_path.write_bytes(b"marker-present")
    observed = {}
    monkeypatch.setattr(
        runtime,
        "_production_observation_marker_path",
        lambda **_kwargs: marker_path,
    )

    def load_marker(_plan, **kwargs):
        observed.update(kwargs)
        return {"marker_sha256": "f" * 64}

    monkeypatch.setattr(
        runtime,
        "load_capability_production_observation_marker",
        load_marker,
    )
    receipt = runtime.wait_for_capability_production_observation_marker(
        plan,
        request,
        observer_gid=2200,
    )
    assert receipt["phase"] == "after"
    assert receipt["observer_live_verified"] is True
    assert observed["require_current_observer"] is True

    tampered = {**request, "timeout_seconds": 301}
    with pytest.raises(PermissionError, match="wait request is invalid"):
        runtime.read_production_observation_wait_request(
            io.BytesIO(runtime._canonical_bytes(tampered)), plan=plan
        )


def test_after_owner_observation_stages_then_publishes_exact_diff(monkeypatch):
    plan = _plan()
    envelope = {
        "phase": "after",
        "fixture_sha256": "c" * 64,
        "run_id": "capability-run-observed",
        "owner_subject_sha256": "d" * 64,
        "observation_sha256": "e" * 64,
    }
    calls = []
    before_envelope = {
        "phase": "before",
        "signed_at_unix_ms": 10,
    }
    monkeypatch.setattr(
        runtime,
        "_read_exact_file",
        lambda *_args, **_kwargs: (
            runtime._canonical_bytes(before_envelope),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "stage_owner_signed_production_observation",
        lambda value, **_kwargs: {
            "envelope_sha256": "1" * 64,
            "marker_sha256": "2" * 64,
        },
    )

    def load_observation(**kwargs):
        calls.append(("load", kwargs["phase"], kwargs.get("now_unix_ms")))
        return {"phase": kwargs["phase"], "signed_at_unix_ms": 10}

    monkeypatch.setattr(
        runtime,
        "load_staged_owner_signed_production_observation",
        load_observation,
    )
    monkeypatch.setattr(
        runtime,
        "build_capability_production_diff",
        lambda before, after, **_kwargs: {
            "schema": runtime.CAPABILITY_PRODUCTION_DIFF_SCHEMA,
            "diff_sha256": "3" * 64,
        },
    )

    def publish_diff(value, **kwargs):
        calls.append(("publish", value["diff_sha256"]))
        assert kwargs["run_id"] == envelope["run_id"]
        assert kwargs["observer_gid"] == 2200
        return {"diff_sha256": value["diff_sha256"]}

    monkeypatch.setattr(
        runtime,
        "publish_capability_production_diff",
        publish_diff,
    )
    receipt = runtime.stage_and_publish_owner_signed_production_observation(
        envelope,
        plan=plan,
        observer_gid=2200,
    )
    assert calls == [
        ("load", "before", 10),
        ("load", "after", None),
        ("publish", "3" * 64),
    ]
    assert receipt["observation_sha256"] == "e" * 64
    assert receipt["production_diff_sha256"] == "3" * 64
    assert receipt["schema"] == (
        runtime.CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA
    )


def test_public_api_exports_only_live_v2_symbols():
    assert runtime.__all__
    assert all(hasattr(runtime, name) for name in runtime.__all__)
    assert "WorkspaceBinding" not in runtime.__all__
    assert "browser_runtime_preflight" not in runtime.__all__
    namespace = {}
    exec(
        "from gateway.canonical_capability_canary_runtime import *",
        namespace,
    )
    assert "attest_capability_execution_readiness" in namespace
    assert "render_worker_service_unit" in namespace
