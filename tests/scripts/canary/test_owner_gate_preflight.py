from __future__ import annotations

import base64
import copy
import hashlib
import json
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_production_ingress_contract as ingress
from scripts.canary import owner_gate_trust as trust
from tests.scripts.canary.test_owner_gate_foundation import (
    IMAGE,
    NOW,
    REVISION,
    _signed_network_evidence,
)


def _key_id(key: Ed25519PrivateKey) -> str:
    return hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()


RELEASE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x17" * 32)
RELEASE_KEY_ID = _key_id(RELEASE_KEY)


@pytest.fixture(autouse=True)
def _pin_release_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        RELEASE_KEY_ID,
    )
    monkeypatch.setattr(
        ingress,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        RELEASE_KEY_ID,
    )


def _production_ingress_envelope(plan, *, iam: bool) -> dict:
    phase = "post_iam" if iam else "inert"
    collected_at_unix = NOW - 1
    fresh_through_unix = collected_at_unix + ingress.FRESHNESS_SECONDS
    observation_body = {
        "schema": ingress.OBSERVATION_SCHEMA,
        "phase": phase,
        "release_revision": plan.spec.release_revision,
        "plan_sha256": plan.sha256,
        "target": {
            "project": ingress.PROJECT,
            "zone": ingress.ZONE,
            "vm": ingress.VM_NAME,
            "instance_id": ingress.INSTANCE_ID,
        },
        "collected_at_unix": collected_at_unix,
        "completed_at_unix": collected_at_unix,
        "fresh_through_unix": fresh_through_unix,
        "old_v1": {
            "unit": ingress.OLD_V1_UNIT,
            "load_state": "masked",
            "active_state": "inactive",
            "sub_state": "dead",
            "unit_file_state": "masked",
            "fragment_path": ingress.OLD_V1_MASK_TARGET,
            "drop_in_paths": [],
            "permanent_mask_path": str(ingress.OLD_V1_MASK_PATH),
            "permanent_mask_target": ingress.OLD_V1_MASK_TARGET,
            "mask_uid": ingress.EXPECTED_ROOT_UID,
            "mask_gid": ingress.EXPECTED_ROOT_GID,
            "trusted_for_v2": False,
        },
        "caddy": {
            "unit": ingress.CADDY_UNIT,
            "load_state": "loaded",
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "enabled",
            "fragment_path": ingress.CADDY_UNIT_FRAGMENT,
            "drop_in_paths": [],
            "main_pid": 4242,
            "exec_start_argv": [
                "/usr/bin/caddy",
                "run",
                "--environ",
                "--config",
                str(ingress.CADDYFILE_PATH),
            ],
            "config_path": str(ingress.CADDYFILE_PATH),
            "config_uid": ingress.EXPECTED_ROOT_UID,
            "config_gid": ingress.EXPECTED_ROOT_GID,
            "config_mode": f"{ingress.CADDYFILE_MODE:04o}",
            "config_size": 512,
            "public_origin": ingress.PUBLIC_ORIGIN,
            "auth_host_route_count": 1,
            "reverse_proxy_handler_count": 1,
            "reverse_proxy_upstream_count": 1,
            "still_on_current_host": True,
            "private_v2_upstream_active": False,
            "process_executable": "/usr/bin/caddy",
            "process_cmdline": [
                "/usr/bin/caddy",
                "run",
                "--environ",
                "--config",
                str(ingress.CADDYFILE_PATH),
            ],
            "admin_endpoint": "127.0.0.1:2019",
            "live_route_projection_sha256": foundation.sha256_json({
                "auth_host_route_count": 1,
                "reverse_proxy_handler_count": 1,
                "reverse_proxy_upstream_count": 1,
                "still_on_current_host": True,
                "private_v2_upstream_active": False,
            }),
            "effective_unit_inventory_closed": True,
            "active_process_stable": True,
            "admin_listener_owned_by_main_pid": True,
            "live_config_matches_adapted_config": True,
            "double_live_config_projection_identical": True,
            "config_validated": True,
            "stable_nofollow_config_verified": True,
            "double_adapt_projection_identical": True,
            "rollback_mode": "pre_migration_v1_only",
        },
        "collector_authority": "production_root_read_only_fixed_projection",
        "caller_selected_input_accepted": False,
        "cloud_mutation_performed": False,
        "service_mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    observation = {
        **observation_body,
        "report_sha256": foundation.sha256_json(observation_body),
    }
    envelope_unsigned = {
        "schema": ingress.ENVELOPE_SCHEMA,
        "phase": phase,
        "release_revision": plan.spec.release_revision,
        "plan_sha256": plan.sha256,
        "observation": observation,
        "observer_report_sha256": observation["report_sha256"],
        "transport_authority": {
            "kind": "pinned_owner_gcloud_iap_ssh_read_only",
            "project": ingress.PROJECT,
            "zone": ingress.ZONE,
            "vm": ingress.VM_NAME,
            "instance_id": ingress.INSTANCE_ID,
            "known_hosts_file_sha256": "1" * 64,
            "observer_source_sha256": "2" * 64,
            "instance_authorization_sha256": "3" * 64,
            "project_authorization_sha256": "4" * 64,
            "oslogin_authorization_sha256": "5" * 64,
        },
        "signed_at_unix": collected_at_unix,
        "fresh_through_unix": fresh_through_unix,
        "signer_key_id": RELEASE_KEY_ID,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    signed = {
        **envelope_unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            RELEASE_KEY.sign(
                ingress.SIGNATURE_DOMAIN
                + ingress._canonical(envelope_unsigned)
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    }
    return {
        **signed,
        "envelope_sha256": hashlib.sha256(
            ingress._canonical(signed)
        ).hexdigest(),
    }


def _production_ingress_kwargs(plan, *, iam: bool) -> dict:
    return {
        "production_ingress_observation": _production_ingress_envelope(
            plan,
            iam=iam,
        ),
        "release_public_key": RELEASE_KEY.public_key(),
    }


def _resign_production_ingress_envelope(
    envelope: dict,
    signing_key: Ed25519PrivateKey,
) -> dict:
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in envelope.items()
        if key not in {"signature_ed25519_b64url", "envelope_sha256"}
    }
    signed = {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            signing_key.sign(
                ingress.SIGNATURE_DOMAIN + ingress._canonical(unsigned)
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    }
    return {
        **signed,
        "envelope_sha256": hashlib.sha256(
            ingress._canonical(signed)
        ).hexdigest(),
    }


def _substitute_production_ingress_binding(
    envelope: dict,
    *,
    field: str,
    value: str,
) -> dict:
    substituted = copy.deepcopy(envelope)
    substituted[field] = value
    substituted["observation"][field] = value
    observation_unsigned = {
        key: item
        for key, item in substituted["observation"].items()
        if key != "report_sha256"
    }
    substituted["observation"]["report_sha256"] = foundation.sha256_json(
        observation_unsigned
    )
    substituted["observer_report_sha256"] = substituted["observation"][
        "report_sha256"
    ]
    return _resign_production_ingress_envelope(substituted, RELEASE_KEY)


def _build_with_production_ingress(
    plan,
    cloud_key: Ed25519PrivateKey,
    host_key: Ed25519PrivateKey,
    *,
    iam: bool,
    envelope: dict,
    release_public_key=None,
    now_unix: int = NOW,
):
    build = (
        preflight.build_post_iam_preflight_report
        if iam
        else preflight.build_preflight_report
    )
    return build(
        plan=plan,
        cloud_observation=_cloud(plan, cloud_key, iam=iam),
        host_observation=_host(plan, host_key, iam=iam),
        production_ingress_observation=envelope,
        cloud_collector_public_key=cloud_key.public_key(),
        host_collector_public_key=host_key.public_key(),
        release_public_key=(
            RELEASE_KEY.public_key()
            if release_public_key is None
            else release_public_key
        ),
        now_unix=now_unix,
    )


def _plan(
    network_key: Ed25519PrivateKey,
    cloud_key: Ed25519PrivateKey,
    host_key: Ed25519PrivateKey,
) -> foundation.OwnerGateFoundationPlan:
    network_id = _key_id(network_key)
    evidence = foundation.ProductionNetworkEvidence.from_mapping(
        _signed_network_evidence(network_key),
        public_key=network_key.public_key(),
        expected_public_key_id=network_id,
        now_unix=NOW,
    )
    return foundation.build_plan(
        spec=foundation.OwnerGateSpec(
            release_revision=REVISION,
            boot_image_self_link=IMAGE,
            package_inventory_sha256="5" * 64,
            interpreter_sha256="6" * 64,
            network_collector_public_key_id=network_id,
            organization_id="123456789012",
            ancestry_evidence_sha256="9" * 64,
            cloud_collector_public_key_id=_key_id(cloud_key),
            host_collector_public_key_id=_key_id(host_key),
        ),
        network_evidence=evidence,
        network_collector_public_key=network_key.public_key(),
        now_unix=NOW,
    )


def _attest(body: dict, key: Ed25519PrivateKey) -> dict:
    report = {**body, "report_sha256": foundation.sha256_json(body)}
    signature = key.sign(foundation.canonical_json_bytes(report))
    return {
        **report,
        "attestation": {
            "schema": "muncho-owner-gate-observation-attestation.v1",
            "public_key_id": _key_id(key),
            "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii"),
        },
    }


def _cloud(plan, key, *, iam: bool) -> dict:
    executor_sa = f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
    body = {
        "schema": preflight.CLOUD_OBSERVATION_SCHEMA,
        "collected_at_unix": NOW - 1,
        "plan_sha256": plan.sha256,
        "project": foundation.PROJECT,
        "zone": foundation.ZONE,
        "source": {
            "name": foundation.PRODUCTION_SOURCE_VM,
            "numeric_id": foundation.PRODUCTION_SOURCE_VM_ID,
            "internal_ip": "10.80.0.2",
            "service_account": foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
            "network": (
                f"https://www.googleapis.com/compute/v1/projects/{foundation.PROJECT}/"
                f"global/networks/{foundation.NETWORK_NAME}"
            ),
            "subnetwork": (
                f"https://www.googleapis.com/compute/v1/projects/{foundation.PROJECT}/"
                f"regions/{foundation.REGION}/subnetworks/"
                f"{foundation.PRODUCTION_SUBNET_NAME}"
            ),
        },
        "subnet": {
            "name": foundation.OWNER_GATE_SUBNET_NAME,
            "network": (
                f"https://www.googleapis.com/compute/v1/projects/{foundation.PROJECT}/"
                f"global/networks/{foundation.NETWORK_NAME}"
            ),
            "cidr": foundation.OWNER_GATE_SUBNET_CIDR,
            "private_google_access": True,
            "stack_type": "IPV4_ONLY",
            "overlap_count": 0,
            "route_inventory_sha256": "1" * 64,
        },
        "instance": {
            "name": foundation.VM_NAME,
            "numeric_id": "1234567890123456789",
            "status": "RUNNING",
            "network": f"projects/p/global/networks/{foundation.NETWORK_NAME}",
            "subnetwork": (
                f"projects/p/regions/{foundation.REGION}/subnetworks/"
                f"{foundation.OWNER_GATE_SUBNET_NAME}"
            ),
            "internal_ip": foundation.OWNER_GATE_PRIVATE_IP,
            "access_config_count": 0,
            "service_accounts": [executor_sa],
            "oauth_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
            "tags": [foundation.IAP_NETWORK_TAG, foundation.OWNER_GATE_NETWORK_TAG],
            "shielded_secure_boot": True,
            "shielded_vtpm": True,
            "shielded_integrity_monitoring": True,
            "serial_port_enabled": False,
            "project_ssh_keys_blocked": True,
            "os_login_enabled": True,
            "startup_script_present": False,
        },
        "service_account": {
            "email": executor_sa,
            "disabled": False,
            "user_managed_key_count": 0,
            "project_roles": sorted([
                f"projects/{foundation.PROJECT}/roles/{foundation.READ_ONLY_IAM_ROLE_ID}",
                *(
                    [f"projects/{foundation.PROJECT}/roles/{foundation.MUTATION_ROLE_ID}"]
                    if iam else []
                ),
            ]),
            "effective_sensitive_permissions": (
                sorted(foundation.EXECUTION_PERMISSIONS) if iam else []
            ),
            "effective_permissions_probe_verified": True,
            "effective_permission_probe": (
                preflight.expected_effective_permission_probe(iam)
            ),
        },
        "iam": {
            "custom_role_permissions": sorted(foundation.EXECUTION_PERMISSIONS),
            "mutation_binding_present": iam,
            "forbidden_roles": [],
            "condition_expression": foundation._condition_expression(),
            "read_only_role_permissions": list(
                foundation.READ_ONLY_IAM_PERMISSIONS
            ),
            "read_only_binding_present": True,
            "ancestor_read_only_permissions": list(
                foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS
            ),
        },
        "firewalls": {
            "iap": {
                "name": "allow-iap-ssh",
                "source_ranges": [foundation.IAP_SOURCE_RANGE],
                "target_tags": [foundation.IAP_NETWORK_TAG],
                "tcp_ports": [22],
                "enabled": True,
            },
            "private_web": {
                "name": "muncho-owner-gate-web-from-production",
                "source_service_accounts": [
                    foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT
                ],
                "target_service_accounts": [executor_sa],
                "tcp_ports": [foundation.WEB_LISTEN_PORT],
                "enabled": True,
                "logging": True,
            },
            "public_owner_gate_rules": [],
            "effective_inventory_sha256": "2" * 64,
            "effective_firewall_probe_verified": True,
        },
        "targets": {
            "instance_name": foundation.TARGET_INSTANCE,
            "instance_numeric_id": foundation.TARGET_INSTANCE_ID,
            "disk_name": foundation.TARGET_DISK,
            "disk_numeric_id": foundation.TARGET_DISK_ID,
            "boot_device_name": foundation.TARGET_BOOT_DEVICE,
        },
        "release_binding": {
            "phase": "post_iam" if iam else "inert",
            "release_revision": REVISION,
            "source_tree_oid": "a" * 40,
            "package_sha256": "3" * 64,
            "package_inventory_sha256": "5" * 64,
            "pre_foundation_authority_sha256": "7" * 64,
            "foundation_apply_receipt_sha256": "8" * 64,
            "project_ancestry_evidence_sha256": "9" * 64,
            "project_ancestry_chain_sha256": "a" * 64,
            "resource_ancestor_chain": [plan.spec.organization_resource],
            "terminal_receipt_sha256": "e" * 64,
            "host_observation_report_sha256": _host(
                plan, Ed25519PrivateKey.generate(), iam=iam
            )["report_sha256"],
            "host_observation_binding_sha256": "4" * 64,
            "production_ingress_observation_sha256": (
                _production_ingress_envelope(plan, iam=iam)["envelope_sha256"]
            ),
            "attached_sa_permission_probe_report_sha256": "3" * 64,
            "cloud_signer_provisioning_receipt_sha256": "f" * 64,
            "cloud_signer_readiness_sha256": "0" * 64,
            "host_signer_provisioning_receipt_sha256": "1" * 64,
            "host_signer_readiness_sha256": "2" * 64,
            "effective_permission_probe_sha256": foundation.sha256_json(
                preflight.expected_effective_permission_probe(iam)
            ),
        },
        "collector": "owner_read_only_rest_remote_executor_attested",
        "credential_values_read": False,
    }
    return _attest(body, key)


def _host(plan, key, *, iam: bool) -> dict:
    body = {
        "schema": preflight.HOST_OBSERVATION_SCHEMA,
        "phase": "post_iam" if iam else "inert",
        "collected_at_unix": NOW - 1,
        "completed_at_unix": NOW - 1,
        "fresh_through_unix": NOW + 59,
        "plan_sha256": plan.sha256,
        "production_ingress_observation_sha256": (
            _production_ingress_envelope(plan, iam=iam)["envelope_sha256"]
        ),
        "observation_binding_sha256": "4" * 64,
        "release": {
            "revision": REVISION,
            "source_tree_oid": "a" * 40,
            "root": f"/opt/muncho-owner-gate/releases/{REVISION}",
            "uid": 0,
            "gid": 0,
            "mode": "0555",
            "immutable": True,
            "package_sha256": "3" * 64,
            "package_inventory_sha256": "5" * 64,
            "pre_foundation_authority_sha256": "7" * 64,
            "foundation_apply_receipt_sha256": "8" * 64,
            "project_ancestry_evidence_sha256": "9" * 64,
            "project_ancestry_chain_sha256": "a" * 64,
            "resource_ancestor_chain": [plan.spec.organization_resource],
            "install_receipt_sha256": "c" * 64,
            "install_receipt_file_sha256": "d" * 64,
            "cloud_signer_provisioning_receipt_sha256": "f" * 64,
            "cloud_signer_readiness_sha256": "0" * 64,
            "host_signer_provisioning_receipt_sha256": "1" * 64,
            "host_signer_readiness_sha256": "2" * 64,
            "attached_sa_permission_probe_report_sha256": "3" * 64,
            "offline_wheelhouse_verified": True,
            "network_install_performed": False,
            "entrypoints": [
                "muncho-owner-gate-intake",
                "muncho-passkey-v2-web",
                "muncho-passkey-v2-authority",
                "muncho-passkey-v2-executor",
                "muncho-owner-gate-cloud-observation-signer",
                "muncho-host-observation-attestor",
            ],
            "observation_dispatcher_schemas": [
                "muncho-storage-growth-trusted-attestation-request.v1",
                "muncho-owner-gate-attached-sa-permission-probe-request.v1",
                "muncho-owner-gate-host-observation-frame.v1",
            ],
            "python_version": "3.11.2",
            "python_executable": (
                f"/opt/muncho-owner-gate/releases/{REVISION}/venv/bin/python"
            ),
            "python_executable_sha256": "6" * 64,
            "python_hash_revalidated_by_sha256sum": True,
        },
        "identities": {
            "web": {
                "name": "muncho-passkey-web",
                "uid": 29101,
                "gid": 29101,
                "shell": "/usr/sbin/nologin",
            },
            "authority": {
                "name": "muncho-passkey-authority",
                "uid": 29102,
                "gid": 29102,
                "shell": "/usr/sbin/nologin",
            },
            "executor": {
                "name": "muncho-storage-executor",
                "uid": 29103,
                "gid": 29103,
                "shell": "/usr/sbin/nologin",
            },
        },
        "sockets": {
            "web_authority": {
                "path": str(foundation.PASSKEY_AUTHORITY_SOCKET),
                "uid": 29102,
                "gid": 29101,
                "mode": "0660",
            },
            "authority_executor": {
                "path": str(foundation.PRIVILEGED_EXECUTOR_SOCKET),
                "uid": 29103,
                "gid": 29102,
                "mode": "0660",
            },
        },
        "units": copy.deepcopy(preflight.EXPECTED_UNIT_PROPERTIES),
        "filesystem_boundaries": {
            "web_reads_authority_db": False,
            "web_writes_authority_db": False,
            "web_reads_mutation_journal": False,
            "authority_reads_mutation_journal": False,
            "executor_reads_authority_db": False,
            "authority_database_owner_uid": 29102,
            "mutation_journal_owner_uid": 29103,
        },
        "metadata_firewall": {
            "web_blocked": True,
            "authority_blocked": True,
            "executor_metadata_allowed": True,
            "executor_private_google_api_allowed": True,
            "other_unprivileged_uids_blocked": True,
            "root_admin_metadata_allowed": True,
            "nft_ruleset_verified": True,
            "root_readiness_receipt_verified": True,
        },
        "sqlite": {
            "runtime_module": "scripts.canary.passkey_v2_sqlite",
            "authority_schema": "muncho-passkey-v2-authority-sqlite.v1",
            "executor_schema": "muncho-passkey-v2-executor-sqlite.v1",
            "authority_db": preflight.AUTHORITY_DB,
            "executor_db": preflight.EXECUTOR_DB,
            "directory_mode": "0700",
            "database_mode": "0600",
            "journal_mode": "DELETE",
            "synchronous": "FULL",
            "begin_immediate": True,
            "append_only_triggers": True,
            "runtime_eligible": True,
        },
        "migration": {
            "credential_count": 1,
            "enabled_owner_count": 1,
            "owner_discord_user_id": "1279454038731264061",
            "credential_id_sha256": preflight.EXPECTED_CREDENTIAL_ID_SHA256,
            "public_key_sha256": preflight.EXPECTED_PUBLIC_KEY_SHA256,
            "user_handle_sha256": preflight.EXPECTED_USER_HANDLE_SHA256,
            "credential_id_b64url_source_receipt_bound": True,
            "public_key_byte_count": 77,
            "sign_count": 0,
            "backed_up": True,
            "active_request_count": 0,
            "active_challenge_count": 0,
            "active_grant_count": 0,
            "totp_seed_migrated": False,
            "source_receipt_verified": True,
            "target_receipt_verified": True,
            "public_key_only": True,
        },
        "webauthn": {
            "rp_id": "lomliev.com",
            "origin": "https://auth.lomliev.com",
            "user_verification_required": True,
            "forged_assertion_blocked": True,
            "wrong_challenge_blocked": True,
            "wrong_origin_blocked": True,
            "wrong_rp_blocked": True,
            "no_uv_blocked": True,
            "replay_blocked": True,
            "concurrent_exactly_one_authorized": True,
            "web_raw_grant_api_absent": True,
        },
        "request_intake": {
            "public_web_can_author_envelope": False,
            "iap_fixed_command_only": True,
            "signed_release_verified": True,
            "signed_source_preflight_verified": True,
            "signed_host_identity_verified": True,
            "signed_external_iam_verified": True,
            "release_plan_transaction_evidence_bound": True,
        },
        "executor": {
            "uid": 29103,
            "mutation_iam_binding_present": iam,
            "activation_seal_present": False,
            "authorization_receipt_signature_self_verified": True,
            "receipt_action_binding_self_verified": True,
            "local_gcloud_present": False,
            "generic_shell_fallback_present": False,
            "compute_api_connectivity_verified": iam,
            "numeric_targets_reverified": iam,
            "target_instance_id": foundation.TARGET_INSTANCE_ID,
            "target_disk_id": foundation.TARGET_DISK_ID,
            "receipt_public_key_sha256": "6" * 64,
            "receipt_public_key_owner": "root:root",
            "receipt_public_key_mode": "0444",
        },
        "effective_permission_probe": (
            preflight.expected_effective_permission_probe(iam)
        ),
        "secret_material_recorded": False,
    }
    return _attest(body, key)


def test_signed_inert_and_post_iam_preflights_are_distinct() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    inert = preflight.build_preflight_report(
        plan=plan,
        cloud_observation=_cloud(plan, cloud_key, iam=False),
        host_observation=_host(plan, host_key, iam=False),
        cloud_collector_public_key=cloud_key.public_key(),
        host_collector_public_key=host_key.public_key(),
        **_production_ingress_kwargs(plan, iam=False),
        now_unix=NOW,
    )
    post = preflight.build_post_iam_preflight_report(
        plan=plan,
        cloud_observation=_cloud(plan, cloud_key, iam=True),
        host_observation=_host(plan, host_key, iam=True),
        cloud_collector_public_key=cloud_key.public_key(),
        host_collector_public_key=host_key.public_key(),
        **_production_ingress_kwargs(plan, iam=True),
        now_unix=NOW,
    )
    assert inert["schema"] == preflight.PREFLIGHT_SCHEMA
    assert inert["mutation_iam_binding_present"] is False
    assert inert["old_v1_masked"] is True
    assert inert["caddy_cutover_performed"] is False
    assert inert["rollback_mode"] == "pre_migration_v1_only"
    assert inert["production_ingress_observation_sha256"] == (
        _production_ingress_envelope(plan, iam=False)["envelope_sha256"]
    )
    assert post["schema"] == preflight.POST_IAM_PREFLIGHT_SCHEMA
    assert post["old_v1_masked"] is True
    assert post["caddy_cutover_performed"] is False
    assert post["rollback_mode"] == "pre_migration_v1_only"
    assert post["production_ingress_observation_sha256"] == (
        _production_ingress_envelope(plan, iam=True)["envelope_sha256"]
    )
    assert post["effective_permissions_exact_for_fixed_probe_set"] is True
    assert len(post["effective_permission_probe_sha256"]) == 64
    assert post["operation_permission_absent"] is True
    assert post["executor_activation_seal_present"] is False


@pytest.mark.parametrize("iam", (False, True))
@pytest.mark.parametrize("field", ("phase", "release_revision", "plan_sha256"))
def test_preflight_rejects_substituted_production_ingress_binding(
    iam: bool,
    field: str,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    values = {
        "phase": "inert" if iam else "post_iam",
        "release_revision": "f" * 40,
        "plan_sha256": "e" * 64,
    }
    envelope = _substitute_production_ingress_binding(
        _production_ingress_envelope(plan, iam=iam),
        field=field,
        value=values[field],
    )

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=envelope,
        )


@pytest.mark.parametrize("iam", (False, True))
def test_preflight_rejects_invalid_production_ingress_signature_and_key(
    iam: bool,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    envelope = _production_ingress_envelope(plan, iam=iam)
    forged = _resign_production_ingress_envelope(envelope, attacker)

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=forged,
        )
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=envelope,
            release_public_key=attacker.public_key(),
        )


@pytest.mark.parametrize("iam", (False, True))
def test_preflight_rejects_stale_or_incomplete_production_ingress_envelope(
    iam: bool,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    envelope = _production_ingress_envelope(plan, iam=iam)

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=envelope,
            now_unix=NOW + ingress.FRESHNESS_SECONDS,
        )
    incomplete = copy.deepcopy(envelope)
    incomplete.pop("observation")
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_observation_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=incomplete,
        )


@pytest.mark.parametrize("iam", (False, True))
def test_preflight_rejects_valid_envelope_not_cross_bound_to_host_and_cloud(
    iam: bool,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    substituted = _production_ingress_envelope(plan, iam=iam)
    substituted["transport_authority"]["known_hosts_file_sha256"] = "9" * 64
    substituted = _resign_production_ingress_envelope(substituted, RELEASE_KEY)

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_production_ingress_cross_binding_invalid",
    ):
        _build_with_production_ingress(
            plan,
            cloud_key,
            host_key,
            iam=iam,
            envelope=substituted,
        )


def test_attacker_collector_key_is_rejected() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    attacker = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    with pytest.raises(preflight.OwnerGatePreflightError):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_cloud(plan, attacker, iam=False),
            host_observation=_host(plan, host_key, iam=False),
            cloud_collector_public_key=attacker.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


def test_individually_valid_cloud_and_host_from_different_runs_are_rejected() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    cloud = _cloud(plan, cloud_key, iam=False)
    matching_host = _host(plan, host_key, iam=False)
    other_host_body = {
        name: copy.deepcopy(item)
        for name, item in matching_host.items()
        if name not in {"report_sha256", "attestation"}
    }
    other_host_body["observation_binding_sha256"] = "b" * 64
    other_host = _attest(other_host_body, host_key)

    preflight._validate_cloud(
        cloud,
        plan_sha256=plan.sha256,
        public_key=cloud_key.public_key(),
        expected_public_key_id=cast(
            str, plan.spec.cloud_collector_public_key_id
        ),
        mutation_binding_present=False,
    )
    preflight._validate_host(
        other_host,
        spec=plan.spec,
        plan_sha256=plan.sha256,
        public_key=host_key.public_key(),
        expected_public_key_id=cast(str, plan.spec.host_collector_public_key_id),
        mutation_binding_present=False,
    )
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_observation_cross_binding_invalid",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=cloud,
            host_observation=other_host,
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


@pytest.mark.parametrize("iam", (False, True))
def test_cross_binding_rejects_different_production_ingress_observations(
    iam: bool,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    cloud = _cloud(plan, cloud_key, iam=iam)
    host = _host(plan, host_key, iam=iam)
    host_body = {
        name: copy.deepcopy(item)
        for name, item in host.items()
        if name not in {"report_sha256", "attestation"}
    }
    host_body["production_ingress_observation_sha256"] = "c" * 64
    substituted_host = _attest(host_body, host_key)

    preflight._validate_cloud(
        cloud,
        plan_sha256=plan.sha256,
        public_key=cloud_key.public_key(),
        expected_public_key_id=cast(
            str, plan.spec.cloud_collector_public_key_id
        ),
        mutation_binding_present=iam,
    )
    preflight._validate_host(
        substituted_host,
        spec=plan.spec,
        plan_sha256=plan.sha256,
        public_key=host_key.public_key(),
        expected_public_key_id=cast(str, plan.spec.host_collector_public_key_id),
        mutation_binding_present=iam,
    )
    build = (
        preflight.build_post_iam_preflight_report
        if iam
        else preflight.build_preflight_report
    )
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_observation_cross_binding_invalid",
    ):
        build(
            plan=plan,
            cloud_observation=cloud,
            host_observation=substituted_host,
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=iam),
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    "surface",
    ("cloud-missing", "cloud-malformed", "host-missing", "host-malformed"),
)
def test_production_ingress_observation_digest_is_required_and_strict(
    surface: str,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    if surface.startswith("cloud"):
        cloud = _cloud(plan, cloud_key, iam=False)
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in cloud.items()
            if name not in {"report_sha256", "attestation"}
        }
        if surface.endswith("missing"):
            unsigned["release_binding"].pop(
                "production_ingress_observation_sha256"
            )
        else:
            unsigned["release_binding"][
                "production_ingress_observation_sha256"
            ] = "not-a-digest"
        with pytest.raises(preflight.OwnerGatePreflightError):
            preflight._validate_cloud_unsigned(
                unsigned,
                plan_sha256=plan.sha256,
                mutation_binding_present=False,
            )
        return

    host = _host(plan, host_key, iam=False)
    host_body = {
        name: copy.deepcopy(item)
        for name, item in host.items()
        if name not in {"report_sha256", "attestation"}
    }
    if surface.endswith("missing"):
        host_body.pop("production_ingress_observation_sha256")
    else:
        host_body["production_ingress_observation_sha256"] = "not-a-digest"
    changed = _attest(host_body, host_key)
    with pytest.raises(preflight.OwnerGatePreflightError):
        preflight._validate_host(
            changed,
            spec=plan.spec,
            plan_sha256=plan.sha256,
            public_key=host_key.public_key(),
            expected_public_key_id=cast(
                str, plan.spec.host_collector_public_key_id
            ),
            mutation_binding_present=False,
        )


def test_public_ingress_or_wrong_disk_id_is_rejected() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    cloud = _cloud(plan, cloud_key, iam=False)
    body = {
        key: value for key, value in cloud.items()
        if key not in {"report_sha256", "attestation"}
    }
    body["firewalls"]["public_owner_gate_rules"] = ["broad-rule"]
    cloud = _attest(body, cloud_key)
    with pytest.raises(preflight.OwnerGatePreflightError):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=cloud,
            host_observation=_host(plan, host_key, iam=False),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


def test_read_only_inventory_uses_post_for_iam_and_effective_firewalls() -> None:
    requests = preflight.read_only_cloud_requests()
    assert requests[-1]["method"] == "POST"
    assert requests[-1]["url"].endswith(":getIamPolicy")
    assert requests[-1]["body"] == '{"options":{"requestedPolicyVersion":3}}'
    assert not any("testIamPermissions" in item["url"] for item in requests)
    assert any("getEffectiveFirewalls" in item["url"] for item in requests)
    assert all("gcloud" not in repr(item) for item in requests)


def test_effective_permission_claim_requires_inherited_probe_proof() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    body = _observation_body(_cloud(plan, cloud_key, iam=False))
    body["service_account"]["effective_permission_probe"][
        "inherited_bindings_evaluated"
    ] = False

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_cloud_service_account_invalid",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_attest(body, cloud_key),
            host_observation=_host(plan, host_key, iam=False),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


def _observation_body(observation: dict) -> dict:
    return copy.deepcopy({
        key: value
        for key, value in observation.items()
        if key not in {"report_sha256", "attestation"}
    })


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "b" * 40),
        ("root", "/opt/muncho-owner-gate/releases/" + "b" * 40),
        ("package_inventory_sha256", "a" * 64),
        ("python_executable_sha256", "b" * 64),
        (
            "python_executable",
            "/opt/muncho-owner-gate/current/venv/bin/python",
        ),
    ],
)
def test_host_release_is_bound_to_exact_signed_plan_authority(
    field: str,
    value: str,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    body = _observation_body(_host(plan, host_key, iam=False))
    body["release"][field] = value

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_host_release_invalid",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_cloud(plan, cloud_key, iam=False),
            host_observation=_attest(body, host_key),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


@pytest.mark.parametrize(
    ("unit_name", "field", "value"),
    [
        ("web", "User", "root"),
        ("authority", "Group", "muncho-passkey-web"),
        ("executor", "ExecStart", "/bin/true"),
        ("web", "PrivateNetwork", "yes"),
        ("authority", "ActiveState", "active"),
        ("executor", "UnitFileState", "enabled"),
    ],
)
def test_host_unit_exact_identity_command_and_network_are_required(
    unit_name: str,
    field: str,
    value: str,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    body = _observation_body(_host(plan, host_key, iam=False))
    body["units"][unit_name][field] = value

    with pytest.raises(preflight.OwnerGatePreflightError):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_cloud(plan, cloud_key, iam=False),
            host_observation=_attest(body, host_key),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


def test_host_unit_rejects_unrequested_extra_property() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    body = _observation_body(_host(plan, host_key, iam=False))
    body["units"]["web"]["Environment"] = "BYPASS=1"

    with pytest.raises(preflight.OwnerGatePreflightError):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_cloud(plan, cloud_key, iam=False),
            host_observation=_attest(body, host_key),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )


def test_host_observation_must_be_consumed_before_signed_freshness_deadline() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_preflight_stale",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=_cloud(plan, cloud_key, iam=False),
            host_observation=_host(plan, host_key, iam=False),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW + 60,
        )


def test_observation_rejects_noncanonical_ed25519_encoding() -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    cloud = _cloud(plan, cloud_key, iam=False)
    encoded = cloud["attestation"]["signature_ed25519_b64url"]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    index = alphabet.index(encoded[-1])
    assert index % 16 == 0
    cloud["attestation"]["signature_ed25519_b64url"] = (
        encoded[:-1] + alphabet[index + 1]
    )

    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_cloud_observation_attestation_invalid",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=cloud,
            host_observation=_host(plan, host_key, iam=False),
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            **_production_ingress_kwargs(plan, iam=False),
            now_unix=NOW,
        )
