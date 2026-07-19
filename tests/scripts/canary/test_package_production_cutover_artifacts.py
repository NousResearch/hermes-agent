from __future__ import annotations

import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.api_verifier_credentials import (
    build_api_approval_scrypt_verifier,
    build_api_bearer_verifier,
)
from gateway.operational_edge_catalog import asset_catalog
from scripts.canary import package_production_cutover_artifacts as package
from tests.gateway.test_canonical_writer_production_cutover import (
    _cron_authority,
    _database_recovery_receipt,
    _runtime_attestation,
)
from tests.gateway.test_production_capability_prerequisites import _topology


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40
RELEASE_OWNER_UID = os.geteuid()
RELEASE_OWNER_GID = os.getegid()
PRODUCTION_RELEASE = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases/hermes-agent-aaaaaaaaaaaa"
)


def _operational_receipt_key_ids() -> dict[str, str]:
    domains = (
        "adventico_email",
        "bitrix",
        "canonical",
        "github",
        "infrastructure",
        "skyvision_db",
        "skyvision_email",
        "skyvision_gitlab",
        "skyvision_panel",
    )
    return {
        domain: f"{index:x}" * 64
        for index, domain in enumerate(domains, start=1)
    }


def _operational_asset_verification() -> dict:
    files = [
        {
            "asset_id": asset_id,
            "path": str(PRODUCTION_RELEASE / asset.packaged_relative),
            "uid": RELEASE_OWNER_UID,
            "gid": RELEASE_OWNER_GID,
            "mode": "0555",
            "size": 1,
            "sha256": hashlib.sha256(asset_id.encode("ascii")).hexdigest(),
        }
        for asset_id, asset in asset_catalog().items()
    ]
    unsigned = {
        "schema": "muncho-operational-edge-assets-verification.v1",
        "release_revision": REVISION,
        "manifest_sha256": "f" * 64,
        "expected_uid": RELEASE_OWNER_UID,
        "expected_gid": RELEASE_OWNER_GID,
        "files": files,
        "file_count": len(files),
        "all_payloads_verified": True,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "verification_sha256": _sha_json(unsigned)}


def _operational_key_foundation() -> dict:
    root = Path(
        "/var/lib/muncho-production-legacy-cutover/staged/keys"
    )
    keys = [
        {
            "domain": domain,
            "private_path": str(
                root / f"operational-edge-{domain}-receipt-private.pem"
            ),
            "private_uid": 0,
            "private_gid": 0,
            "private_mode": "0400",
            "public_path": str(root / f"{domain}-receipt-public.pem"),
            "public_uid": 0,
            "public_gid": 0,
            "public_mode": "0444",
            "public_key_id": key_id,
            "created": True,
        }
        for domain, key_id in _operational_receipt_key_ids().items()
    ]
    unsigned = {
        "schema": "muncho-operational-edge-key-foundation.v1",
        "writer_public_key_id": "c" * 64,
        "keys": keys,
        "key_count": len(keys),
        "keys_distinct": True,
        "retain_created_keys_on_rollback": True,
        "private_content_or_digest_recorded": False,
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha_json(unsigned)}


def _set_test_identity(path: Path) -> None:
    """Make the fixture match the strict claimed process UID/GID on macOS."""

    os.chown(path, os.geteuid(), os.getegid())


def _unit_inputs() -> dict:
    domains = sorted(_operational_receipt_key_ids())
    operational_identities = {
        domain: {
            "user": f"muncho-edge-{domain}",
            "group": f"muncho-edge-{domain}",
            "uid": 2100 + index,
            "gid": 2100 + index,
        }
        for index, domain in enumerate(domains)
    }
    operational_socket_groups = {
        domain: {
            "group": f"muncho-edge-{domain}-c",
            "gid": 2200 + index,
        }
        for index, domain in enumerate(domains)
    }
    return {
        "schema": package.UNIT_INPUT_SCHEMA,
        "release_revision": REVISION,
        "authority_plan_sha256": "d" * 64,
        "authority_approval_sha256": "e" * 64,
        "database_ip": "10.20.30.40",
        "target": {
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "vm": "ai-platform-runtime-01",
            "database": "ai_platform_brain",
            "sql_instance": "production-pg18",
            "sql_host": "10.20.30.40",
            "tls_server_name": "production.example.internal",
            "port": 5432,
            "writer_login": "muncho_production_writer_login",
        },
        "gateway": {
            "user": "ai-platform-brain", "group": "ai-platform-brain",
            "uid": RELEASE_OWNER_UID, "gid": RELEASE_OWNER_GID,
        },
        "writer": {
            "user": "muncho-canonical-writer",
            "group": "muncho-canonical-writer", "uid": 2000, "gid": 2000,
        },
        "projector": {
            "user": "muncho-projector", "group": "muncho-projector",
            "uid": 2004, "gid": 2004,
        },
        "routeback": {
            "user": "muncho-discord-egress", "group": "muncho-discord-egress",
            "uid": 2002, "gid": 2002,
        },
        "connector": {
            "user": "muncho-discord-connector",
            "group": "muncho-discord-connector", "uid": 2001, "gid": 2001,
        },
        "mac_ops": {
            "user": "muncho-mac-ops-edge", "group": "muncho-mac-ops-edge",
            "uid": 2003, "gid": 2003,
        },
        "browser": {
            "user": "muncho-capability-browser", "group": "muncho-capability-browser",
            "uid": 2006, "gid": 2006,
        },
        "worker": {
            "user": "muncho-worker", "group": "muncho-worker",
            "uid": 2007, "gid": 2007,
        },
        "writer_client_group": {"group": "muncho-writer-client", "gid": 2005},
        "worker_client_group": {"group": "muncho-worker-clients", "gid": 2008},
        "operational_edge_identities": operational_identities,
        "operational_edge_socket_groups": operational_socket_groups,
        "writer_capability_public_key_id": "c" * 64,
        "discord_edge_receipt_public_key_id": "a" * 64,
        "operational_edge_key_foundation_sha256": (
            _operational_key_foundation()["receipt_sha256"]
        ),
        "operational_edge_receipt_public_key_ids": (
            _operational_receipt_key_ids()
        ),
        "discord_reconciliation_intent": {
            "schema": package.DISCORD_RECONCILIATION_INTENT_SCHEMA,
            "purpose": package.DISCORD_RECONCILIATION_INTENT_PURPOSE,
            "release_revision": REVISION,
            "legacy_public_policy_sha256": "1" * 64,
            "target_public_policy_sha256": "2" * 64,
            "reviewed_reconciliation": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        },
        "release_owner_uid": RELEASE_OWNER_UID,
        "release_owner_gid": RELEASE_OWNER_GID,
        "bwrap_sha256": "6" * 64,
        "shell_sha256": "7" * 64,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _unit_input_payload() -> dict:
    value = _unit_inputs()
    return {
        "schema": package.UNIT_INPUT_PAYLOAD_SCHEMA,
        **{
            key: item
            for key, item in value.items()
            if key
            not in {
                "schema",
                "release_revision",
                "authority_plan_sha256",
                "authority_approval_sha256",
            }
        },
    }


def _unit_input_authority(now_unix: int = 1_800_000_000):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    plan = package.build_unit_input_plan(
        release_revision=REVISION,
        unit_inputs=_unit_input_payload(),
        owner_subject_sha256="a" * 64,
        owner_public_key_ed25519_hex=public,
        owner_runtime_attestation=_runtime_attestation(),
        created_at_unix=now_unix - 10,
    )
    approval = {
        "schema": package.UNIT_INPUT_APPROVAL_SCHEMA,
        "purpose": "production_cutover_unit_inputs",
        "plan_sha256": plan["plan_sha256"],
        "release_revision": REVISION,
        "owner_subject_sha256": plan["owner_subject_sha256"],
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": plan["owner_key_id"],
        "nonce_sha256": "b" * 64,
        "issued_at_unix": now_unix - 1,
        "expires_at_unix": now_unix + 600,
        "approved": True,
        "signature_ed25519_hex": "0" * 128,
        "approval_sha256": "0" * 64,
    }
    approval["signature_ed25519_hex"] = private.sign(
        package.unit_input_approval_signature_payload(approval)
    ).hex()
    approval["approval_sha256"] = _sha_json(
        {
            key: item
            for key, item in approval.items()
            if key != "approval_sha256"
        }
    )
    return plan, approval


def _runtime_dependency_for_units() -> dict:
    browser_root = PRODUCTION_RELEASE
    return {
        "agent_browser": {
            "version": "0.26.0",
            "config_path": str(
                browser_root
                / "ops/muncho/runtime/dependencies/agent-browser.json"
            ),
            "config_sha256": "8" * 64,
            "wrapper_path": str(
                browser_root
                / "node_modules/agent-browser/bin/agent-browser.js"
            ),
            "wrapper_sha256": "9" * 64,
            "native_path": str(
                browser_root
                / "node_modules/agent-browser/bin/agent-browser-linux-x64"
            ),
            "native_sha256": "a" * 64,
            "package_tree": {},
            "node_path": str(
                browser_root
                / "ops/muncho/runtime/dependencies/node-linux-x64/bin/node"
            ),
            "node_version": "v24.18.0",
            "node_sha256": "b" * 64,
            "npm_path": str(browser_root / "unused/npm"),
            "npm_version": "10.0.0",
            "npm_target_sha256": "c" * 64,
            "node_tree": {},
        },
        "chrome": {
            "version": "150.0.7871.114",
            "executable_path": str(
                browser_root
                / "ops/muncho/runtime/dependencies/chrome-linux64/chrome"
            ),
            "executable_sha256": "d" * 64,
            "tree": {},
        },
    }


def _release(tmp_path: Path) -> Path:
    release = (tmp_path / f"hermes-agent-{REVISION[:12]}").resolve()
    release.mkdir(parents=True, exist_ok=True)
    os.chown(release, os.geteuid(), os.getegid())
    for relative in (
        "ops/muncho/cutover/production_cutover_artifact_runtime.py.in",
        "scripts/sql/canonical_writer_legacy_reconcile_v1.sql",
        "scripts/sql/canonical_writer_v1.sql",
        "ops/muncho/systemd/muncho-discord-connector.service.in",
        "ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf",
        "ops/muncho/systemd/discord-public-connector.json.in",
    ):
        target = release / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (release / ".codex-source-commit").write_text(REVISION + "\n", encoding="ascii")
    browser_root = PRODUCTION_RELEASE
    dependency_unsigned = {
        "schema": "muncho-production-runtime-dependencies.v1",
        "release_revision": REVISION,
        "release_address": str(browser_root),
        "platform": {},
        "source": {},
        **_runtime_dependency_for_units(),
        "python": {},
        "secret_material_recorded": False,
    }
    dependency_manifest = {
        **dependency_unsigned,
        "manifest_sha256": _sha_json(dependency_unsigned),
    }
    dependency_path = (
        release / "ops/muncho/runtime/dependencies/manifest.json"
    )
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    dependency_path.write_bytes(
        json.dumps(
            dependency_manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    hermes_home = release / "fixture-hermes-home"
    canonical_brain = release / "fixture-canonical-brain"
    roots = {
        "hermes": hermes_home,
        "canonical": canonical_brain,
        "release": release,
    }
    for asset_id, asset in asset_catalog().items():
        source = roots[asset.source_root] / asset.source_relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"fixture:{asset_id}\n".encode("ascii"))
        source.chmod(0o555)
    package.package_operational_assets(
        release_root=release,
        revision=REVISION,
        hermes_home=hermes_home,
        canonical_brain=canonical_brain,
    )
    return release


def _load_artifact(path: Path, name: str):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _sha_json(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _identity_foundation() -> dict:
    unit_inputs = _unit_inputs()
    operational_identities = unit_inputs["operational_edge_identities"]
    operational_socket_groups = unit_inputs[
        "operational_edge_socket_groups"
    ]
    group_specs = {
        "gateway": ("ai-platform-brain", RELEASE_OWNER_GID, []),
        "writer": ("muncho-canonical-writer", 2000, []),
        "projector": (
            "muncho-projector", 2004, ["muncho-canonical-writer"]
        ),
        "writer_client": (
            "muncho-writer-client", 2005, ["ai-platform-brain"]
        ),
        "routeback": (
            "muncho-discord-egress", 2002, ["ai-platform-brain"]
        ),
        "connector": (
            "muncho-discord-connector", 2001, ["ai-platform-brain"]
        ),
        "mac_ops": (
            "muncho-mac-ops-edge", 2003, ["ai-platform-brain"]
        ),
        "browser": (
            "muncho-capability-browser", 2006, ["ai-platform-brain"]
        ),
        "worker": ("muncho-worker", 2007, []),
        "worker_client": (
            "muncho-worker-clients", 2008, ["ai-platform-brain"]
        ),
        **{
            f"operational_edge_{domain}": (
                identity["group"], identity["gid"], []
            )
            for domain, identity in operational_identities.items()
        },
        **{
            f"operational_edge_{domain}_client": (
                operational_socket_groups[domain]["group"],
                operational_socket_groups[domain]["gid"],
                ["ai-platform-brain", operational_identities[domain]["user"]],
            )
            for domain in operational_identities
        },
    }
    groups = {}
    for role, (name, gid, members) in group_specs.items():
        present = role == "gateway"
        groups[role] = {
            "name": name,
            "gid": gid,
            "members": sorted(members),
            "pre": {
                "state": "present" if present else "absent",
                "gid": gid if present else None,
                "members": [] if present else None,
            },
        }
    user_specs = {
        "gateway": (
            "ai-platform-brain",
            RELEASE_OWNER_UID,
            "/opt/adventico-ai-platform/canonical-brain",
            [
                "muncho-discord-connector",
                "muncho-discord-egress",
                "muncho-mac-ops-edge",
                "muncho-capability-browser",
                *[
                    operational_socket_groups[domain]["group"]
                    for domain in sorted(operational_socket_groups)
                ],
                "muncho-worker-clients",
                "muncho-writer-client",
            ],
            ["google-sudoers"],
        ),
        "writer": (
            "muncho-canonical-writer", 2000, "/nonexistent",
            ["muncho-projector"], None,
        ),
        "projector": ("muncho-projector", 2004, "/nonexistent", [], None),
        "routeback": (
            "muncho-discord-egress", 2002, "/nonexistent", [], None
        ),
        "connector": (
            "muncho-discord-connector", 2001, "/nonexistent", [], None
        ),
        "mac_ops": (
            "muncho-mac-ops-edge", 2003, "/nonexistent",
            [], None,
        ),
        "browser": (
            "muncho-capability-browser", 2006, "/nonexistent", [], None
        ),
        "worker": ("muncho-worker", 2007, "/nonexistent", [], None),
        **{
            f"operational_edge_{domain}": (
                identity["user"],
                identity["uid"],
                "/nonexistent",
                [operational_socket_groups[domain]["group"]],
                None,
            )
            for domain, identity in operational_identities.items()
        },
    }
    users = {}
    for role, (name, uid, home, supplementary, before) in user_specs.items():
        present = role == "gateway"
        users[role] = {
            "name": name,
            "uid": uid,
            "primary_group": role,
            "home": home,
            "shell": "/usr/sbin/nologin",
            "supplementary_groups": sorted(supplementary),
            "pre": {
                "state": "present" if present else "absent",
                "uid": uid if present else None,
                "gid": group_specs[role][1] if present else None,
                "home": home if present else None,
                "shell": "/usr/sbin/nologin" if present else None,
                "supplementary_group_names": before,
            },
        }
    unsigned = {
        "schema": "muncho-production-host-identity-foundation.v1",
        "users": users,
        "groups": groups,
        "retain_created_dormant_on_rollback": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "foundation_sha256": _sha_json(unsigned)}


def _discord_key_foundation() -> dict:
    writer = {
        "staged_private_path": (
            "/var/lib/muncho-production-legacy-cutover/staged/keys/"
            "writer-capability-private.pem"
        ),
        "private_path": "/etc/muncho/keys/writer-capability-private.pem",
        "private_uid": 2000,
        "private_gid": 2000,
        "private_mode": 0o400,
        "public_path": "/etc/muncho/keys/writer-capability-public.pem",
        "public_uid": 0,
        "public_gid": 2002,
        "public_mode": 0o440,
        "public_key_id": "c" * 64,
    }
    edge = {
        "staged_private_path": (
            "/var/lib/muncho-production-legacy-cutover/staged/keys/"
            "discord-edge-receipt-private.pem"
        ),
        "private_path": "/etc/muncho/keys/discord-edge-receipt-private.pem",
        "private_uid": 0,
        "private_gid": 0,
        "private_mode": 0o400,
        "public_path": "/etc/muncho/keys/discord-edge-receipt-public.pem",
        "public_uid": 0,
        "public_gid": 2000,
        "public_mode": 0o440,
        "public_key_id": "d" * 64,
    }
    unsigned = {
        "schema": "muncho-production-discord-key-foundation.v1",
        "writer": writer,
        "edge": edge,
        "pre_state": "absent",
        "keys_distinct": True,
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
    }
    return {**unsigned, "foundation_sha256": _sha_json(unsigned)}


def _cutover_plan(
    artifact_sha: str,
    *,
    accepted_event_receipts: list[dict[str, str]] | None = None,
) -> dict:
    snapshot_unsigned = {
        "schema": "muncho-production-legacy-snapshot.v1",
        "database": "ai_platform_brain",
        "shape": "legacy19",
        "source_owner": "legacy_event_owner",
        "relation_oid": "16421",
        "source_row_count": 14081,
        "canonical14_sha256": "1" * 64,
        "extended19_sha256": "2" * 64,
        "occurred_at_cutoff": "2026-07-14T09:00:00+00:00",
        "inserted_at_cutoff": "2026-07-14T09:00:01+00:00",
        "relation_identity_sha256": "3" * 64,
        "acl_identity_sha256": "4" * 64,
        "index_identity_sha256": "5" * 64,
        "observed_at_unix": 1_800_000_000,
    }
    snapshot = {**snapshot_unsigned, "snapshot_sha256": _sha_json(snapshot_unsigned)}
    base = "/opt/adventico-ai-platform/hermes-agent-releases/hermes-agent-aaaaaaaaaaaa/ops/muncho/cutover/artifacts"
    artifacts = {
        "observe": {"path": f"{base}/production-observe", "sha256": "8" * 64},
        "database_apply": {"path": f"{base}/production-database-apply", "sha256": "9" * 64},
        "database_rollback": {"path": f"{base}/production-database-rollback", "sha256": "b" * 64},
        "database_postflight": {"path": f"{base}/production-database-postflight", "sha256": "c" * 64},
        "host_activation": {"path": f"{base}/production-host-activation", "sha256": "d" * 64},
        "host_rollback": {"path": f"{base}/production-host-rollback", "sha256": artifact_sha},
    }
    public = "11" * 32
    _sealed_request, sealed = package._sealed_runtime_artifact_request(
        revision=REVISION,
        runtime_dependency=_runtime_dependency_for_units(),
        unit_inputs=_unit_inputs(),
        operational_asset_verification=(
            _operational_asset_verification()
        ),
    )
    topology = _topology(REVISION)
    topology["isolated_worker"] = copy.deepcopy(
        sealed["topology_fragments"]["isolated_worker"]
    )
    topology["browser"] = copy.deepcopy(
        sealed["topology_fragments"]["browser"]
    )
    topology["phase_b"]["fragment_sha256"] = sealed["files"][
        "phase_b_unit"
    ]["sha256"]
    topology["routeback_edge"]["fragment_sha256"] = sealed["files"][
        "routeback_unit"
    ]["sha256"]
    topology["mac_ops"]["fragment_sha256"] = sealed["files"][
        "mac_ops_unit"
    ]["sha256"]
    topology["gateway_identity"] = {
        "uid": RELEASE_OWNER_UID,
        "gid": RELEASE_OWNER_GID,
    }
    drop_in = (
        "/etc/systemd/system/hermes-cloud-gateway.service.d/"
        "20-discord-connector.conf"
    )

    def service(name: str, fragment: str, digest: str | None, *, drop=False):
        absent = digest is None
        raw = {
            "schema": "muncho-production-service-observation.v1",
            "name": name,
            "fragment_path": "" if absent else fragment,
            "fragment_sha256": digest,
            "load_state": "not-found" if absent else "loaded",
            "active_state": "inactive",
            "sub_state": "dead",
            "unit_file_state": "" if absent else "enabled",
            "main_pid": 0,
            "drop_in_paths": [drop_in] if drop else [],
            "drop_in_sha256": {drop_in: "f" * 64} if drop else {},
            "need_daemon_reload": False,
            "triggered_by": [],
            "triggers": [],
            "observed_at_unix": 1_800_000_000,
        }
        return {**raw, "observation_sha256": _sha_json(raw)}

    def stable(value: dict) -> dict:
        return {
            key: value[key]
            for key in (
                "name", "fragment_path", "fragment_sha256", "load_state",
                "unit_file_state", "drop_in_paths", "drop_in_sha256",
                "need_daemon_reload", "triggered_by", "triggers",
            )
        }

    gateway_pre = service(
        "hermes-cloud-gateway.service",
        "/etc/systemd/system/hermes-cloud-gateway.service",
        "1" * 64,
    )
    writer_pre = service(
        "muncho-canonical-writer.service",
        "/etc/systemd/system/muncho-canonical-writer.service",
        None,
    )
    connector_pre = service(
        "muncho-discord-connector.service",
        "/etc/systemd/system/muncho-discord-connector.service",
        None,
    )
    gateway_target_observation = service(
        "hermes-cloud-gateway.service",
        "/etc/systemd/system/hermes-cloud-gateway.service",
        "2" * 64,
        drop=True,
    )
    writer_target_observation = service(
        "muncho-canonical-writer.service",
        "/etc/systemd/system/muncho-canonical-writer.service",
        "3" * 64,
    )
    connector_target_observation = service(
        "muncho-discord-connector.service",
        "/etc/systemd/system/muncho-discord-connector.service",
        "4" * 64,
    )
    host_paths = {
        "gateway_unit": "/etc/systemd/system/hermes-cloud-gateway.service",
        "writer_unit": "/etc/systemd/system/muncho-canonical-writer.service",
        "connector_unit": "/etc/systemd/system/muncho-discord-connector.service",
        "phase_b_unit": (
            "/etc/systemd/system/"
            "muncho-canonical-writer-phase-b-readiness.service"
        ),
        "routeback_unit": "/etc/systemd/system/muncho-discord-egress.service",
        "mac_ops_unit": "/etc/systemd/system/muncho-mac-ops-edge.service",
        "browser_unit": "/etc/systemd/system/muncho-capability-browser.service",
        "browser_config": "/etc/muncho/browser-controller.json",
        "isolated_worker_socket_unit": (
            "/etc/systemd/system/muncho-isolated-worker.socket"
        ),
        "isolated_worker_service_unit": (
            "/etc/systemd/system/muncho-isolated-worker.service"
        ),
        "isolated_worker_config": "/etc/muncho/isolated-worker.json",
        "gateway_connector_drop_in": drop_in,
        "gateway_config": "/opt/adventico-ai-platform/hermes-home/config.yaml",
        "writer_config": "/etc/muncho-canonical-writer/writer.json",
        "connector_config": "/etc/muncho/discord-public-connector.json",
        "routeback_config": "/etc/muncho/discord-edge.json",
        "mac_ops_config": "/etc/muncho/mac-ops-edge/config.json",
        "api_bearer_verifier": (
            "/etc/muncho/keys/api-server-bearer-sha256.json"
        ),
        "api_approval_verifier": (
            "/etc/muncho/keys/api-approval-passkey-scrypt.json"
        ),
    }
    host_digests = {
        "gateway_unit": "2" * 64,
        "writer_unit": "3" * 64,
        "connector_unit": "4" * 64,
        "phase_b_unit": topology["phase_b"]["fragment_sha256"],
        "routeback_unit": topology["routeback_edge"]["fragment_sha256"],
        "mac_ops_unit": topology["mac_ops"]["fragment_sha256"],
        "browser_unit": topology["browser"]["fragment_sha256"],
        "browser_config": topology["browser"]["config_sha256"],
        "isolated_worker_socket_unit": topology["isolated_worker"][
            "socket_fragment_sha256"
        ],
        "isolated_worker_service_unit": topology["isolated_worker"][
            "service_fragment_sha256"
        ],
        "isolated_worker_config": topology["isolated_worker"][
            "config_sha256"
        ],
        "gateway_connector_drop_in": "f" * 64,
        "gateway_config": "a" * 64,
        "writer_config": "e" * 64,
        "connector_config": "b" * 64,
        "routeback_config": topology["routeback_edge"]["config_sha256"],
        "mac_ops_config": topology["mac_ops"]["config_sha256"],
        "api_bearer_verifier": "8" * 64,
        "api_approval_verifier": "9" * 64,
    }
    operational_file_names = {
        name
        for name in sealed["files"]
        if name.startswith("operational_edge_")
    }
    for name in operational_file_names:
        host_paths[name] = sealed["files"][name]["target_path"]
        host_digests[name] = sealed["files"][name]["sha256"]
    absent = {
        "state": "absent", "sha256": None, "uid": None, "gid": None,
        "mode": None,
    }
    files = {}
    for name, path in host_paths.items():
        uid, gid, mode = (0, 0, 0o644)
        if name == "gateway_config":
            uid, gid, mode = (RELEASE_OWNER_UID, RELEASE_OWNER_GID, 0o640)
        elif name == "writer_config":
            uid, gid, mode = (0, 2000, 0o440)
        elif name == "connector_config":
            uid, gid, mode = (0, 2001, 0o440)
        elif name == "routeback_config":
            uid, gid, mode = (0, 2002, 0o440)
        elif name == "mac_ops_config":
            uid, gid, mode = (0, 2003, 0o440)
        elif name == "browser_config":
            uid, gid, mode = (0, 2006, 0o440)
        elif name == "isolated_worker_config":
            uid, gid, mode = (0, 2007, 0o440)
        elif name in {"api_bearer_verifier", "api_approval_verifier"}:
            uid, gid, mode = (0, 0, 0o400)
        elif name in operational_file_names:
            uid, gid, mode = (
                sealed["files"][name]["uid"],
                sealed["files"][name]["gid"],
                sealed["files"][name]["mode"],
            )
        pre = copy.deepcopy(absent)
        if name == "gateway_unit":
            pre = {
                "state": "present", "sha256": "1" * 64, "uid": 0,
                "gid": 0, "mode": 0o644,
            }
        elif name == "gateway_config":
            pre = {
                "state": "present", "sha256": "c" * 64,
                "uid": RELEASE_OWNER_UID,
                "gid": RELEASE_OWNER_GID, "mode": 0o640,
            }
        files[name] = {
            "staged_path": (
                "/var/lib/muncho-production-legacy-cutover/staged/host/"
                + Path(path).name
            ),
            "target_path": path,
            "sha256": host_digests[name],
            "uid": uid,
            "gid": gid,
            "mode": mode,
            "pre": pre,
        }
    legacy_discord_policy = {
        "allowed_guild_ids": ["1282725267068157972"],
        "allowed_channel_ids": [
            "1504852355588423801",
            "1504852408227069993",
            "1504852444407140402",
            "1504852485083496561",
            "1504852553031221391",
            "1504852628373373028",
            "1505499746939174993",
            "1507239177350283274",
            "1507239385010016308",
            "1507239516409167942",
            "1510888721614901358",
        ],
        "allowed_user_ids": [
            "1279454038731264061",
            "1282938967888498720",
        ],
        "allowed_role_ids": [
            "1282725267068157972",
            "1505077218374586468",
        ],
        "allow_all_users": False,
        "allow_bot_authors": False,
        "require_mention": True,
        "auto_thread": True,
        "thread_require_mention": False,
        "discord_dm_allowed": False,
        "free_response_channel_ids": [
            "1504852355588423801",
            "1505499746939174993",
        ],
        "public_only": False,
        "author_policy": "exact_ids_or_roles",
    }
    target_discord_policy = {
        **copy.deepcopy(legacy_discord_policy),
        "allowed_channel_ids": [
            "1504852355588423801",
            "1504852408227069993",
            "1504852444407140402",
            "1504852485083496561",
            "1504852553031221391",
            "1504852628373373028",
            "1505499746939174993",
            "1507239177350283274",
            "1507239385010016308",
            "1510888721614901358",
        ],
        "allowed_user_ids": [],
        "allowed_role_ids": [],
        "author_policy": "guild_acl",
    }
    continuity_unsigned = {
        "schema": "muncho-production-discord-policy-continuity.v2",
        "source_evidence_sha256": "6" * 64,
        "legacy_policy": legacy_discord_policy,
        "target_policy": target_discord_policy,
        "exact_membership_preserved": False,
        "reviewed_reconciliation": True,
        "secret_material_recorded": False,
    }
    continuity = {
        **continuity_unsigned,
        "continuity_sha256": _sha_json(continuity_unsigned),
    }
    operational_key_foundation = _operational_key_foundation()
    transition_unsigned = {
        "schema": "muncho-production-host-transition-manifest.v1",
        "files": files,
        "identity_foundation": _identity_foundation(),
        "discord_key_foundation": _discord_key_foundation(),
        "operational_edge_key_foundation": operational_key_foundation,
        "operational_edge_key_foundation_sha256": (
            operational_key_foundation["receipt_sha256"]
        ),
        "operational_edge_receipt_public_key_ids": (
            _operational_receipt_key_ids()
        ),
        "release_owner_uid": RELEASE_OWNER_UID,
        "release_owner_gid": RELEASE_OWNER_GID,
        "isolated_worker_lease_mountpoint": {
            "target_path": "/var/lib/muncho-isolated-worker",
            "uid": 0,
            "gid": 0,
            "mode": 0o700,
            "pre": {"state": "absent", "uid": None, "gid": None, "mode": None},
        },
        "connector_token": {
            "path": "/etc/muncho/discord-connector-credentials/bot-token",
            "uid": 2001,
            "gid": 2001,
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": "/opt/adventico-ai-platform/hermes-home/.discord-token",
            "source_uid": RELEASE_OWNER_UID,
            "source_gid": RELEASE_OWNER_GID,
            "source_mode": 0o400,
        },
        "gateway_retired_token_paths": [
            "/opt/adventico-ai-platform/hermes-home/.discord-token"
        ],
        "routeback_token_paths": [
            "/etc/muncho/discord-edge-credentials/bot-token"
        ],
        "approval_passkey": {
            "path": "/etc/muncho/keys/api-approval-passkey-scrypt.json",
            "uid": 0,
            "gid": 0,
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": (
                "/var/lib/muncho-production-legacy-cutover/staged/"
                "api-approval-passkey"
            ),
            "source_uid": 0,
            "source_gid": 0,
            "source_mode": 0o400,
        },
        "retired_approval_passkey_paths": [
            "/var/lib/muncho-production-legacy-cutover/staged/"
            "api-approval-passkey"
        ],
        "gateway_direct_discord_enabled": False,
        "gateway_relay_platforms": ["discord"],
        "connector_operation_class": "ordinary_guild_acl_session_only",
        "routeback_operation_class": "canonical_guild_acl_routeback_rest_only",
        "discord_dm_allowed": False,
        "discord_policy_continuity": continuity,
        "secret_material_recorded": False,
    }
    host_transition = {
        **transition_unsigned,
        "manifest_sha256": _sha_json(transition_unsigned),
    }
    target = {
        "project": "measured-project",
        "zone": "measured-zone",
        "vm": "measured-vm",
        "database": "ai_platform_brain",
        "sql_instance": "measured-instance",
        "sql_host": "10.20.30.40",
        "tls_server_name": "measured.internal",
        "port": 5432,
        "writer_login": "muncho_production_writer_login",
    }
    rollback = {
        "database_rollback_sha256": artifacts["database_rollback"]["sha256"],
        "host_rollback_sha256": artifacts["host_rollback"]["sha256"],
        "requires_gateway_stopped": True,
        "requires_writer_stopped": True,
        "requires_connector_stopped": True,
        "requires_zero_canonical_writer_writes": True,
        "restart_legacy_gateway": True,
    }
    initial_unsigned = {
        **snapshot_unsigned,
        "source_row_count": 14073,
    }
    initial_snapshot = {
        **initial_unsigned,
        "snapshot_sha256": _sha_json(initial_unsigned),
    }
    receipts = copy.deepcopy(accepted_event_receipts)
    event_ids = (
        [] if receipts is None else [item["event_id"] for item in receipts]
    )
    reseed_manifest_sha256 = None
    truth_epoch_id = None
    truth_epoch_sha256 = None
    if receipts is None:
        truth_epoch_id = (
            "truth-epoch:33333333-3333-4333-8333-333333333333"
        )
        truth_epoch_sha256 = _sha_json({
            "schema": "muncho-production-new-truth-epoch.v1",
            "reviewed_snapshot_sha256": initial_snapshot["snapshot_sha256"],
            "truth_epoch_id": truth_epoch_id,
        })
        receipts = []
    else:
        reseed_manifest_sha256 = _sha_json({
            "schema": "muncho-production-legacy-reseed-manifest.v1",
            "reviewed_snapshot_sha256": initial_snapshot["snapshot_sha256"],
            "accepted_event_ids": event_ids,
            "accepted_event_receipts": receipts,
        })
    decision_unsigned = {
        "schema": "muncho-production-legacy-truth-decision.v1",
        "mode": (
            "start_new_truth_epoch"
            if not event_ids
            else "reseed_accepted_events"
        ),
        "decision_id": (
            "legacy-truth-decision:11111111-1111-4111-8111-111111111111"
        ),
        "decision_event_id": "22222222-2222-4222-8222-222222222222",
        "owner_subject_sha256": "e" * 64,
        "reviewed_snapshot_sha256": initial_snapshot["snapshot_sha256"],
        "accepted_event_ids": event_ids,
        "accepted_event_receipts": receipts,
        "reseed_manifest_sha256": reseed_manifest_sha256,
        "truth_epoch_id": truth_epoch_id,
        "truth_epoch_sha256": truth_epoch_sha256,
    }
    legacy_truth_decision = {
        **decision_unsigned,
        "decision_sha256": _sha_json(decision_unsigned),
    }
    cron_inventory, cron_plan, host_facts, mechanical_package = _cron_authority()
    isolated_canary_unsigned = {
        "schema": "muncho-production-isolated-canary-goal-prerequisite.v2",
        "fixture": {},
        "fixture_sha256": "1" * 64,
        "workspace_gateway": {},
        "workspace_gateway_receipt_sha256": "2" * 64,
        "cleanup_receipt": {},
        "cleanup_receipt_sha256": _sha_json({}),
        "goal_continuation_terminal_schema": (
            "muncho-production-capability-goal-continuation-terminal.v2"
        ),
        "goal_continuation_terminal_sha256": "3" * 64,
        "isolation_equivalence_projection": {},
        "isolation_equivalence_projection_sha256": "4" * 64,
        "production_diff_sha256": "5" * 64,
        "production_diff": {"diff_sha256": "5" * 64},
        "production_diff_file_sha256": _sha_json(
            {"diff_sha256": "5" * 64}
        ),
        "run_id": "capability-run-package-test",
        "release_revision": REVISION,
        "capability_plan_sha256": "6" * 64,
        "full_canary_plan_sha256": "7" * 64,
        "canary_owner_approval_receipt_sha256": "8" * 64,
        "canary_production_mutation_observed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    isolated_canary = {
        **isolated_canary_unsigned,
        "evidence_sha256": _sha_json(isolated_canary_unsigned),
    }
    authority_unsigned = {
        "schema": "muncho-production-cutover-authority.v4",
        "release_revision": REVISION,
        "artifacts": artifacts,
        "gateway_target_identity": stable(gateway_target_observation),
        "writer_target_identity": stable(writer_target_observation),
        "connector_target_identity": stable(connector_target_observation),
        "host_transition": host_transition,
        "capability_topology": topology,
        "cron_inventory": cron_inventory,
        "cron_continuity_plan": cron_plan,
        "mechanical_job_host_facts": host_facts,
        "mechanical_job_package": mechanical_package,
        "isolated_canary_goal_prerequisite": isolated_canary,
        "database_recovery_receipt": _database_recovery_receipt(),
        "legacy_truth_decision": legacy_truth_decision,
        "final_tail_bounds": {
            "max_appended_rows": 10_000,
            "max_capture_delay_seconds": 900,
        },
        "rollback_contract": rollback,
        "secret_material_recorded": False,
    }
    authority = {
        **authority_unsigned,
        "authority_sha256": _sha_json(authority_unsigned),
    }
    freeze_unsigned = {
        "schema": "muncho-production-legacy-freeze-plan.v3",
        "release_revision": REVISION,
        "target": target,
        "owner_subject_sha256": "e" * 64,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
        "gateway_before": gateway_pre,
        "writer_before": writer_pre,
        "connector_before": connector_pre,
        "initial_snapshot": initial_snapshot,
        "owner_runtime_attestation": _runtime_attestation(),
        "observe_artifact": artifacts["observe"],
        "cutover_authority": authority,
        "database_recovery_receipt_sha256": authority[
            "database_recovery_receipt"
        ]["receipt_sha256"],
        "states": ["authority", "gateway_stopped", "final_tail_captured"],
        "secret_material_recorded": False,
    }
    freeze = {
        **freeze_unsigned,
        "plan_sha256": _sha_json(freeze_unsigned),
    }
    tail_unsigned = {
        "schema": "muncho-production-final-tail-receipt.v1",
        "freeze_plan_sha256": freeze["plan_sha256"],
        "approval_sha256": "7" * 64,
        "gateway_stopped": True,
        "writer_stopped": True,
        "final_snapshot": snapshot,
        "initial_row_count": 14073,
        "final_row_count": 14081,
        "append_only_row_floor_preserved": True,
        "captured_at_unix": 1_800_000_001,
    }
    tail = {**tail_unsigned, "receipt_sha256": _sha_json(tail_unsigned)}
    unsigned = {
        "schema": "muncho-production-legacy-cutover-plan.v3",
        "release_revision": REVISION,
        "target": target,
        "owner_subject_sha256": "e" * 64,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
        "freeze_plan": freeze,
        "freeze_plan_sha256": freeze["plan_sha256"],
        "freeze_approval_sha256": tail["approval_sha256"],
        "final_tail_receipt": tail,
        "final_tail_receipt_sha256": tail["receipt_sha256"],
        "artifacts": artifacts,
        "gateway_legacy_identity": gateway_pre,
        "writer_pre_identity": writer_pre,
        "connector_pre_identity": connector_pre,
        "gateway_target_identity": stable(gateway_target_observation),
        "writer_target_identity": stable(writer_target_observation),
        "connector_target_identity": stable(connector_target_observation),
        "host_transition": host_transition,
        "capability_topology": authority["capability_topology"],
        "cron_inventory": authority["cron_inventory"],
        "cron_continuity_plan": authority["cron_continuity_plan"],
        "mechanical_job_host_facts": authority["mechanical_job_host_facts"],
        "mechanical_job_package": authority["mechanical_job_package"],
        "legacy_truth_decision": legacy_truth_decision,
        "database_recovery_receipt_sha256": freeze[
            "database_recovery_receipt_sha256"
        ],
        "owner_runtime_attestation": freeze["owner_runtime_attestation"],
        "rollback_contract": rollback,
        "states": [
            "authority",
            "host_applied",
            "prerequisites_started",
            "capability_prerequisites_validated",
            "final_tail_reobserved",
            "preflight",
            "database_applied",
            "writer_started",
            "database_terminal_validated",
            "activation_commit_intent",
            "boot_committed",
            "gateway_started",
            "terminal",
        ],
        "secret_material_recorded": False,
    }
    return {**unsigned, "plan_sha256": _sha_json(unsigned)}


def test_clean_host_bootstraps_owner_approved_inputs_before_package_and_freeze(
    tmp_path,
):
    staged = (tmp_path / "staged").resolve()
    staged.mkdir(mode=0o700)
    staged.chmod(0o700)
    _set_test_identity(staged)
    plan_path = staged / "unit-input-plan.json"
    approval_path = staged / "unit-input-approval.json"
    inputs_path = staged / "production-unit-inputs.json"
    plan, approval = _unit_input_authority()
    plan_path.write_bytes(package._canonical_bytes(plan))
    approval_path.write_bytes(package._canonical_bytes(approval))
    plan_path.chmod(0o400)
    approval_path.chmod(0o400)

    receipt = package.bootstrap_fixed_unit_inputs(
        authority_plan_path=plan_path,
        authority_approval_path=approval_path,
        unit_inputs_path=inputs_path,
        require_root=False,
        now_unix=1_800_000_000,
    )
    inputs = package.load_fixed_unit_inputs(
        inputs_path,
        expected_uid=inputs_path.stat().st_uid,
        expected_gid=inputs_path.stat().st_gid,
    )
    assert receipt["created"] is True
    assert inputs["authority_plan_sha256"] == plan["plan_sha256"]
    assert inputs["authority_approval_sha256"] == approval["approval_sha256"]
    assert stat.S_IMODE(inputs_path.stat().st_mode) == 0o444
    assert not (staged / "freeze-plan.json").exists()

    release = _release(tmp_path / "release")
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=inputs,
    )
    assert package.verify_release_artifacts(
        release,
        REVISION,
        unit_inputs=inputs,
    ) == manifest

    # Freeze authority is deliberately constructed only after all artifact
    # digests exist; the pre-package unit-input approval has no FreezePlan
    # dependency and therefore cannot form a circular bootstrap.
    from gateway import canonical_writer_production_cutover as cutover
    from tests.gateway.test_canonical_writer_production_cutover import (
        Services,
        _approval,
        _freeze,
    )

    freeze_key = Ed25519PrivateKey.generate()
    freeze = _freeze(freeze_key, Services())
    assert cutover.CutoverApproval.from_mapping(
        _approval(freeze_key, freeze),
        plan=freeze,
        now_unix=1_800_000_000,
    ).sha256


def test_bootstrap_rejects_unapproved_unit_input_authority(tmp_path):
    staged = tmp_path.resolve()
    staged.chmod(0o700)
    _set_test_identity(staged)
    plan_path = staged / "unit-input-plan.json"
    approval_path = staged / "unit-input-approval.json"
    inputs_path = staged / "production-unit-inputs.json"
    plan, approval = _unit_input_authority()
    approval["signature_ed25519_hex"] = "f" * 128
    approval["approval_sha256"] = _sha_json(
        {
            key: item
            for key, item in approval.items()
            if key != "approval_sha256"
        }
    )
    plan_path.write_bytes(package._canonical_bytes(plan))
    approval_path.write_bytes(package._canonical_bytes(approval))
    plan_path.chmod(0o400)
    approval_path.chmod(0o400)

    with pytest.raises(
        package.PackagingError,
        match="unit_input_approval_invalid",
    ):
        package.bootstrap_fixed_unit_inputs(
            authority_plan_path=plan_path,
            authority_approval_path=approval_path,
            unit_inputs_path=inputs_path,
            require_root=False,
            now_unix=1_800_000_000,
        )
    assert not inputs_path.exists()


def test_standalone_target_bootstrap_verifies_signature_without_repo_imports(
    tmp_path,
):
    bootstrap = _load_artifact(
        ROOT / "ops/muncho/cutover/production_unit_input_bootstrap.py",
        "standalone_production_unit_input_bootstrap",
    )
    staged = (tmp_path / "staged").resolve()
    staged.mkdir(mode=0o700)
    _set_test_identity(staged)
    plan_path = staged / "unit-input-plan.json"
    approval_path = staged / "unit-input-approval.json"
    output_path = staged / "production-unit-inputs.json"
    plan, approval = _unit_input_authority()
    plan_path.write_bytes(package._canonical_bytes(plan))
    approval_path.write_bytes(package._canonical_bytes(approval))
    plan_path.chmod(0o400)
    approval_path.chmod(0o400)

    receipt = bootstrap.bootstrap(
        plan_path=plan_path,
        approval_path=approval_path,
        output_path=output_path,
        openssl=Path(shutil.which("openssl") or "/usr/bin/openssl"),
        now_unix=1_800_000_000,
        require_root=False,
    )

    assert receipt["created"] is True
    assert receipt["release_revision"] == REVISION
    assert output_path.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444


def test_packager_binds_gateway_imports_to_its_target_release_under_isolation(
    tmp_path,
):
    release = (tmp_path / "target-release").resolve()
    script = release / "scripts/canary/package_production_cutover_artifacts.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(
        (ROOT / "scripts/canary/package_production_cutover_artifacts.py").read_bytes()
    )
    gateway = release / "gateway"
    gateway.mkdir()
    (gateway / "__init__.py").write_text("", encoding="utf-8")
    (gateway / "isolated_worker_units.py").write_text(
        "from pathlib import Path\n"
        "print('TARGET_RELEASE_GATEWAY_IMPORT', file=__import__('sys').stderr)\n"
        "BWRAP_PATH=Path('/usr/bin/bwrap')\n"
        "CONFIG_MODE=0o440\n"
        "ISOLATED_WORKER_CLIENT_GROUP='clients'\n"
        "ISOLATED_WORKER_CONFIG=Path('/etc/worker.json')\n"
        "ISOLATED_WORKER_GROUP='worker'\n"
        "ISOLATED_WORKER_LEASE_BASE=Path('/var/lib/worker')\n"
        "ISOLATED_WORKER_SERVICE_UNIT='worker.service'\n"
        "ISOLATED_WORKER_SOCKET=Path('/run/worker.sock')\n"
        "ISOLATED_WORKER_SOCKET_UNIT='worker.socket'\n"
        "ISOLATED_WORKER_USER='worker'\n"
        "SHELL_PATH=Path('/bin/sh')\n"
        "def render_isolated_worker_units(**kwargs): raise AssertionError\n",
        encoding="utf-8",
    )
    (gateway / "production_capability_prerequisites.py").write_text(
        "from pathlib import Path\n"
        "BROWSER_CONFIG_PATH=Path('/etc/browser.json')\n"
        "BROWSER_SOCKET_PATH=Path('/run/browser.sock')\n"
        "BROWSER_UNIT='browser.service'\n"
        "MAC_OPS_UNIT='mac.service'\n"
        "PHASE_B_UNIT='phase.service'\n"
        "ROUTEBACK_EDGE_UNIT='route.service'\n"
        "def packaged_prerequisite_contract(): return {}\n",
        encoding="utf-8",
    )
    (gateway / "production_capability_units.py").write_text(
        "BROWSER_CONFIG_MODE=0o440\n"
        "def render_production_capability_units(**kwargs): raise AssertionError\n",
        encoding="utf-8",
    )
    (gateway / "production_cron_continuity_package.py").write_text(
        "PLAN_SCHEMA='muncho-production-cron-packaged-continuity-plan.v4'\n",
        encoding="utf-8",
    )
    (gateway / "operational_edge_catalog.py").write_text(
        "CREDENTIALS_BY_DOMAIN={name:() for name in ("
        "'adventico_email','bitrix','canonical','github','infrastructure',"
        "'skyvision_db','skyvision_email','skyvision_gitlab',"
        "'skyvision_panel')}\n",
        encoding="utf-8",
    )
    (gateway / "operational_edge_assets.py").write_text(
        "from pathlib import Path\n"
        "ASSET_MANIFEST_RELATIVE=Path('ops/manifest.json')\n"
        "class OperationalEdgeAssetError(RuntimeError): pass\n"
        "def package_operational_assets(**kwargs): raise AssertionError\n"
        "def validate_packaged_operational_asset_verification(*args, **kwargs): "
        "raise AssertionError\n"
        "def verify_packaged_operational_assets(**kwargs): raise AssertionError\n",
        encoding="utf-8",
    )
    (gateway / "operational_edge_units.py").write_text(
        "from pathlib import Path\n"
        "CLIENT_CONFIG_PATH=Path('/etc/operational-edge-client.json')\n"
        "class OperationalEdgeUnitError(ValueError): pass\n"
        "def render_operational_edge_units(**kwargs): raise AssertionError\n"
        "def service_config_path(domain): return Path('/etc') / (domain + '.json')\n"
        "def service_identity_name(domain): return 'muncho-edge-' + domain\n"
        "def service_unit(domain): return 'edge-' + domain + '.service'\n"
        "def socket_group_name(domain): return 'muncho-edge-' + domain + '-c'\n",
        encoding="utf-8",
    )
    (gateway / "production_owner_runtime.py").write_text(
        "class ProductionOwnerRuntimeError(RuntimeError): pass\n"
        "def validate_owner_runtime_attestation(value, *, revision): "
        "return value\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "TARGET_RELEASE_GATEWAY_IMPORT" in completed.stderr


def test_verify_rejects_symlinked_release_root(tmp_path: Path) -> None:
    release = _release((tmp_path / "release").resolve())
    package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )
    alias = tmp_path / "release-alias"
    alias.symlink_to(release, target_is_directory=True)

    with pytest.raises(
        package.PackagingError,
        match="cutover_packaging_release_invalid",
    ):
        package.verify_release_artifacts(
            alias,
            REVISION,
            unit_inputs=_unit_inputs(),
        )


def test_build_emits_six_distinct_self_contained_action_sealed_artifacts(tmp_path):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    verified = package.verify_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )

    assert verified == manifest
    assert manifest["schema"] == package.MANIFEST_SCHEMA
    assert manifest["release_revision"] == REVISION
    assert manifest["secret_material_recorded"] is False
    assert set(manifest["artifacts"]) == set(package.ARTIFACTS)
    assert set(manifest["plan_bindings"]) == set(package.PLAN_BINDINGS)
    assert len({item["sha256"] for item in manifest["artifacts"].values()}) == 6

    for name, actions in package.ARTIFACTS.items():
        item = manifest["artifacts"][name]
        path = Path(item["path"])
        payload = path.read_bytes()
        source = payload.decode("utf-8")
        compile(payload, str(path), "exec")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert "gateway" not in imported_roots
        assert "scripts" not in imported_roots
        assert f"ALLOWED_ACTIONS = {actions!r}" in source
        assert (
            "PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA = "
            f"{package.PRODUCTION_CRON_CONTINUITY_PLAN_SCHEMA!r}"
        ) in source
        assert "__MUNCHO_" not in source
        assert "muncho_canary_brain" not in source
        assert "legacy reconciliation refuses the production database" not in source
        assert "owner_approved_cutover" in source
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        assert len(payload) == item["size"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o500

    for binding, name in package.PLAN_BINDINGS.items():
        artifact = manifest["artifacts"][name]
        assert manifest["plan_bindings"][binding] == {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
        }

    manifest_path = release / "ops/muncho/cutover/artifacts/manifest.json"
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444
    assert manifest_path.read_bytes() == package._canonical_bytes(manifest) + b"\n"


def test_verify_rejects_artifact_drift(tmp_path):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    path = Path(manifest["artifacts"]["production-observe"]["path"])
    path.chmod(0o700)
    path.write_bytes(path.read_bytes() + b"\n")
    path.chmod(0o500)

    with pytest.raises(package.PackagingError, match="artifact_drifted"):
        package.verify_release_artifacts(
            release,
            REVISION,
            unit_inputs=_unit_inputs(),
        )


def test_staged_release_manifest_binds_final_release_address(tmp_path):
    staged = _release(tmp_path)
    final = (tmp_path / "final" / f"hermes-agent-{REVISION[:12]}").resolve()
    manifest = package.build_release_artifacts(
        staged,
        REVISION,
        release_address=final,
        unit_inputs=_unit_inputs(),
    )
    package.verify_release_artifacts(
        staged,
        REVISION,
        release_address=final,
        unit_inputs=_unit_inputs(),
    )
    assert all(
        item["path"].startswith(str(final) + "/")
        for item in manifest["artifacts"].values()
    )

    final.parent.mkdir(parents=True)
    staged.rename(final)
    assert package.verify_release_artifacts(
        final,
        REVISION,
        unit_inputs=_unit_inputs(),
    ) == manifest


def test_production_render_fails_closed_if_reviewed_canary_contract_changes():
    source = (ROOT / "scripts/sql/canonical_writer_legacy_reconcile_v1.sql").read_text()
    changed = source.replace("isolated_canary_copy", "different_scope", 1)

    with pytest.raises(package.PackagingError, match="scope_contract_changed"):
        package._production_reconcile(changed)


def test_generated_runtime_rejects_cross_artifact_action_and_plan_extensions(tmp_path, monkeypatch):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    item = manifest["artifacts"]["production-host-rollback"]
    runtime = _load_artifact(Path(item["path"]), "production_host_rollback_artifact")
    monkeypatch.setattr(runtime, "_self_sha256", lambda: item["sha256"])
    plan = _cutover_plan(item["sha256"])
    request_unsigned = {
        "schema": runtime.REQUEST_SCHEMA,
        "action": "host_rollback",
        "plan": plan,
        "apply_receipt": None,
        "secret_material_recorded": False,
    }
    request = {**request_unsigned, "request_sha256": _sha_json(request_unsigned)}

    action, validated_plan, receipt = runtime._validate_request(request)
    assert action == "host_rollback"
    assert validated_plan["plan_sha256"] == plan["plan_sha256"]
    assert receipt is None


    wrong_action = dict(request_unsigned)
    wrong_action["action"] = "host_apply_stopped"
    with pytest.raises(runtime.ArtifactError, match="request_invalid"):
        runtime._validate_request(
            {**wrong_action, "request_sha256": _sha_json(wrong_action)}
        )

    extended_plan = {**plan, "unreviewed": True}
    extended_plan["plan_sha256"] = _sha_json(
        {key: value for key, value in extended_plan.items() if key != "plan_sha256"}
    )
    extended_request = {**request_unsigned, "plan": extended_plan}
    with pytest.raises(runtime.ArtifactError, match="plan_invalid"):
        runtime._validate_request(
            {**extended_request, "request_sha256": _sha_json(extended_request)}
        )


def test_generated_runtime_accepts_only_exact_pidless_worker_socket_shape(
    tmp_path,
    monkeypatch,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    item = manifest["artifacts"]["production-host-rollback"]
    runtime = _load_artifact(
        Path(item["path"]),
        "production_host_rollback_pidless_socket",
    )

    def observation(unit, *, include_main_pid):
        values = {
            "Names": unit,
            "FragmentPath": "",
            "LoadState": "not-found",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "",
            "MainPID": "0",
            "DropInPaths": "",
            "NeedDaemonReload": "no",
            "TriggeredBy": "",
            "Triggers": "",
        }
        if not include_main_pid:
            values.pop("MainPID")
        return "".join(
            f"{name}={value}\n" for name, value in values.items()
        ).encode()

    socket = runtime.ISOLATED_WORKER_SOCKET_UNIT
    monkeypatch.setattr(
        runtime,
        "_systemctl",
        lambda *_args, **_kwargs: observation(socket, include_main_pid=False),
    )
    assert runtime._service(socket)["main_pid"] == 0

    monkeypatch.setattr(
        runtime,
        "_systemctl",
        lambda *_args, **_kwargs: observation(
            runtime.ISOLATED_WORKER_SERVICE_UNIT,
            include_main_pid=False,
        ),
    )
    with pytest.raises(runtime.ArtifactError, match="service_observation_invalid"):
        runtime._service(runtime.ISOLATED_WORKER_SERVICE_UNIT)


def test_host_rollback_recovers_every_missing_pre_mutation_backup_and_fails_closed_after_mutation(
    tmp_path,
    monkeypatch,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-rollback"]["path"]),
        "production_host_backup_fault_injection_artifact",
    )
    plan = {"plan_sha256": "a" * 64}
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    exact_pre = {name: True for name in runtime.HOST_FILE_NAMES}
    bootstrapped: list[str] = []

    def backup_path(_plan, name):
        return backup_root / f"{name}.json"

    def backup_file(_plan, name, item):
        unsigned = {
            "schema": "muncho-production-host-file-backup.v1",
            "name": name,
            "pre_identity": item["pre"],
            "bytes_hex": None,
        }
        backup_path(_plan, name).write_bytes(
            runtime._canonical_bytes(
                {**unsigned, "backup_sha256": runtime._sha_json(unsigned)}
            )
        )
        bootstrapped.append(name)

    monkeypatch.setattr(runtime, "_backup_path", backup_path)
    monkeypatch.setattr(
        runtime,
        "_file_matches",
        lambda _path, _identity: exact_pre[_path.name],
    )
    monkeypatch.setattr(runtime, "_backup_file", backup_file)
    monkeypatch.setattr(
        runtime,
        "_read_exact_file",
        lambda path, **_kwargs: path.read_bytes(),
    )

    items = {
        name: {
            "target_path": str(tmp_path / "targets" / name),
            "pre": {
                "state": "absent",
                "sha256": None,
                "uid": None,
                "gid": None,
                "mode": None,
            },
        }
        for name in runtime.HOST_FILE_NAMES
    }
    for name in sorted(runtime.HOST_FILE_NAMES):
        assert runtime._load_backup(plan, name, items[name]) is None
    assert sorted(bootstrapped) == sorted(runtime.HOST_FILE_NAMES)

    for name in sorted(runtime.HOST_FILE_NAMES):
        backup_path(plan, name).unlink()
        exact_pre[name] = False
        with pytest.raises(
            runtime.ArtifactError,
            match="artifact_host_backup_missing_after_mutation",
        ):
            runtime._load_backup(plan, name, items[name])


def test_identity_rollback_accepts_every_partial_create_only_group_and_user_boundary(
    tmp_path,
    monkeypatch,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-rollback"]["path"]),
        "production_host_identity_create_fault_injection_artifact",
    )
    host_manifest = {"identity_foundation": _identity_foundation()}
    foundation = runtime._identity_foundation(host_manifest)
    expected_pre = runtime._expected_identity_pre(foundation)
    expected_target = runtime._expected_identity_target(foundation)
    absent_group_roles = [
        role
        for role in runtime.IDENTITY_GROUP_ROLES
        if foundation["groups"][role]["pre"]["state"] == "absent"
    ]
    absent_user_roles = [
        role
        for role in runtime.IDENTITY_USER_ROLES
        if foundation["users"][role]["pre"]["state"] == "absent"
    ]
    monkeypatch.setattr(
        runtime,
        "_identity_command",
        lambda _arguments: pytest.fail("partial create rollback mutated NSS"),
    )

    for created_count in range(len(absent_group_roles) + 1):
        current = copy.deepcopy(expected_pre)
        for role in absent_group_roles[:created_count]:
            current["groups"][role] = {
                **expected_target["groups"][role],
                "members": [],
            }
        monkeypatch.setattr(
            runtime,
            "_identity_snapshot",
            lambda _foundation, snapshot=current: copy.deepcopy(snapshot),
        )
        receipt = runtime._rollback_identity_foundation(host_manifest)
        assert receipt["retained_dormant_groups"] == sorted(
            foundation["groups"][role]["name"]
            for role in absent_group_roles[:created_count]
        )
        assert receipt["retained_dormant_users"] == []

    all_groups_created = copy.deepcopy(expected_pre)
    for role in absent_group_roles:
        all_groups_created["groups"][role] = {
            **expected_target["groups"][role],
            "members": [],
        }
    for created_count in range(len(absent_user_roles) + 1):
        current = copy.deepcopy(all_groups_created)
        for role in absent_user_roles[:created_count]:
            current["users"][role] = {
                **expected_target["users"][role],
                "supplementary_group_names": [],
            }
        monkeypatch.setattr(
            runtime,
            "_identity_snapshot",
            lambda _foundation, snapshot=current: copy.deepcopy(snapshot),
        )
        receipt = runtime._rollback_identity_foundation(host_manifest)
        assert receipt["retained_dormant_groups"] == sorted(
            foundation["groups"][role]["name"]
            for role in absent_group_roles
        )
        assert receipt["retained_dormant_users"] == sorted(
            foundation["users"][role]["name"]
            for role in absent_user_roles[:created_count]
        )


def test_identity_rollback_reverses_every_partial_membership_convergence_boundary(
    tmp_path,
    monkeypatch,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-rollback"]["path"]),
        "production_host_identity_membership_fault_injection_artifact",
    )
    host_manifest = {"identity_foundation": _identity_foundation()}
    foundation = runtime._identity_foundation(host_manifest)
    expected_pre = runtime._expected_identity_pre(foundation)
    expected_target = runtime._expected_identity_target(foundation)
    absent_group_roles = [
        role
        for role in runtime.IDENTITY_GROUP_ROLES
        if foundation["groups"][role]["pre"]["state"] == "absent"
    ]
    absent_user_roles = [
        role
        for role in runtime.IDENTITY_USER_ROLES
        if foundation["users"][role]["pre"]["state"] == "absent"
    ]
    group_role_by_name = {
        item["name"]: role for role, item in foundation["groups"].items()
    }
    operations: list[tuple[str, str, str]] = []
    for role in runtime.IDENTITY_USER_ROLES:
        pre_groups = set(
            foundation["users"][role]["pre"][
                "supplementary_group_names"
            ]
            or []
        )
        target_groups = set(foundation["users"][role]["supplementary_groups"])
        username = foundation["users"][role]["name"]
        operations.extend(
            ("delete", username, group_name)
            for group_name in sorted(pre_groups - target_groups)
        )
        operations.extend(
            ("add", username, group_name)
            for group_name in sorted(target_groups - pre_groups)
        )

    def created_state():
        state = copy.deepcopy(expected_pre)
        for role in absent_group_roles:
            state["groups"][role] = {
                **expected_target["groups"][role],
                "members": [],
            }
        for role in absent_user_roles:
            state["users"][role] = {
                **expected_target["users"][role],
                "supplementary_group_names": [],
            }
        return state

    def apply_membership(state, action, username, group_name):
        user_role = next(
            role
            for role, item in foundation["users"].items()
            if item["name"] == username
        )
        groups = state["users"][user_role]["supplementary_group_names"]
        if action == "add":
            groups.append(group_name)
        else:
            groups.remove(group_name)
        groups.sort()
        managed_role = group_role_by_name.get(group_name)
        if managed_role is not None:
            members = state["groups"][managed_role]["members"]
            if action == "add":
                members.append(username)
            else:
                members.remove(username)
            members.sort()

    for completed_count in range(len(operations) + 1):
        current = created_state()
        for operation in operations[:completed_count]:
            apply_membership(current, *operation)

        monkeypatch.setattr(
            runtime,
            "_identity_snapshot",
            lambda _foundation, state=current: copy.deepcopy(state),
        )

        def identity_command(arguments, state=current):
            assert arguments[0] == runtime.GPASSWD
            apply_membership(
                state,
                "add" if arguments[1] == "--add" else "delete",
                arguments[2],
                arguments[3],
            )

        monkeypatch.setattr(runtime, "_identity_command", identity_command)
        receipt = runtime._rollback_identity_foundation(host_manifest)
        assert receipt["retained_dormant_users"] == sorted(
            foundation["users"][role]["name"] for role in absent_user_roles
        )
        assert receipt["retained_dormant_groups"] == sorted(
            foundation["groups"][role]["name"] for role in absent_group_roles
        )


def test_database_state_distinguishes_reconciled_migrated_and_terminal(tmp_path, monkeypatch):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    path = Path(manifest["artifacts"]["production-database-apply"]["path"])
    runtime = _load_artifact(path, "production_database_apply_artifact")
    plan = _cutover_plan("f" * 64)
    base = {
        "public_oid": 17000,
        "archive_oid": 16421,
        "public_columns": 14,
        "archive_columns": 19,
        "canonical_schema": False,
        "writer_contract_ready": False,
    }
    monkeypatch.setattr(runtime, "_psql_json", lambda *_args, **_kwargs: dict(base))
    assert runtime._database_state(plan) == "reconciled"

    base["canonical_schema"] = True
    base["writer_contract_ready"] = True
    monkeypatch.setattr(runtime, "_writer_membership_ready", lambda _plan: False)
    assert runtime._database_state(plan) == "migrated"
    monkeypatch.setattr(runtime, "_writer_membership_ready", lambda _plan: True)
    assert runtime._database_state(plan) == "target"


def test_legacy_truth_modes_are_exact_and_selected_continuity_is_mechanical(
    tmp_path,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-database-apply"]["path"]),
        "production_database_legacy_truth_artifact",
    )
    epoch_plan = _cutover_plan("f" * 64)
    receipts = [
        {
            "event_id": "00000000-0000-4000-8000-00000000000a",
            "canonical14_row_sha256": "a" * 64,
        },
        {
            "event_id": "00000000-0000-4000-8000-00000000000b",
            "canonical14_row_sha256": "b" * 64,
        },
    ]
    reseed_plan = _cutover_plan(
        "f" * 64,
        accepted_event_receipts=receipts,
    )

    runtime._plan_digest(epoch_plan)
    runtime._plan_digest(reseed_plan)
    assert epoch_plan["legacy_truth_decision"]["mode"] == (
        "start_new_truth_epoch"
    )
    assert reseed_plan["legacy_truth_decision"]["mode"] == (
        "reseed_accepted_events"
    )

    sql = runtime._legacy_truth_decision_sql(reseed_plan)
    projection = runtime._legacy_truth_projection_sql(reseed_plan)
    for receipt in receipts:
        assert receipt["event_id"] in sql
        assert receipt["canonical14_row_sha256"] in sql
        assert receipt["event_id"] in projection
    assert "00000000-0000-4000-8000-000000000099" not in sql
    assert "owner_approved_legacy_reseed" in sql
    assert "observed_session,thread_id" in sql
    assert "observed_session,chat_id" in sql
    assert "observed_session,session_key_sha256" in sql


def test_legacy_truth_decision_rejects_absence_tamper_and_cross_plan_replay(
    tmp_path,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-database-apply"]["path"]),
        "production_database_legacy_truth_replay_artifact",
    )
    receipts = [{
        "event_id": "00000000-0000-4000-8000-00000000000a",
        "canonical14_row_sha256": "a" * 64,
    }]
    first = _cutover_plan("f" * 64, accepted_event_receipts=receipts)
    second = _cutover_plan("e" * 64, accepted_event_receipts=receipts)
    assert first["legacy_truth_decision"]["decision_event_id"] == (
        second["legacy_truth_decision"]["decision_event_id"]
    )
    first_sql = runtime._legacy_truth_decision_sql(first)
    second_sql = runtime._legacy_truth_decision_sql(second)
    assert first["plan_sha256"] in first_sql
    assert second["plan_sha256"] not in first_sql
    assert second["plan_sha256"] in second_sql
    assert "replay or tamper detected" in first_sql

    absent = copy.deepcopy(first)
    absent.pop("legacy_truth_decision")
    absent["plan_sha256"] = _sha_json({
        key: value
        for key, value in absent.items()
        if key != "plan_sha256"
    })
    with pytest.raises(runtime.ArtifactError, match="plan_invalid"):
        runtime._plan_digest(absent)

    tampered = copy.deepcopy(first["legacy_truth_decision"])
    tampered["accepted_event_receipts"][0]["canonical14_row_sha256"] = (
        "c" * 64
    )
    tampered["decision_sha256"] = _sha_json({
        key: value
        for key, value in tampered.items()
        if key != "decision_sha256"
    })
    with pytest.raises(
        runtime.ArtifactError,
        match="legacy_truth_decision_invalid",
    ):
        runtime._validate_legacy_truth_decision(
            tampered,
            snapshot=first["freeze_plan"]["initial_snapshot"],
            owner_subject_sha256=first["owner_subject_sha256"],
        )


def test_database_rollback_receipt_is_bound_to_exact_legacy_truth_decision(
    tmp_path,
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-database-rollback"]["path"]),
        "production_database_legacy_truth_rollback_artifact",
    )
    plan = _cutover_plan("f" * 64)
    snapshot = plan["final_tail_receipt"]["final_snapshot"]
    decision = plan["legacy_truth_decision"]
    unsigned = {
        "schema": "muncho-production-legacy-database-apply.v1",
        "plan_sha256": plan["plan_sha256"],
        "artifact_sha256": plan["artifacts"]["database_apply"]["sha256"],
        "final_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_row_count": snapshot["source_row_count"],
        "archive_row_count": snapshot["source_row_count"],
        "canonical_row_count": snapshot["source_row_count"] + 1,
        "archive_extended19_sha256": snapshot["extended19_sha256"],
        "canonical14_sha256": snapshot["canonical14_sha256"],
        "relation_identity_sha256": snapshot["relation_identity_sha256"],
        "acl_identity_sha256": snapshot["acl_identity_sha256"],
        "index_identity_sha256": snapshot["index_identity_sha256"],
        "roles_acl_sha256": "d" * 64,
        "zero_canonical_writer_writes": True,
        "trusted_legacy_event_count": 0,
        "legacy_truth_mode": decision["mode"],
        "legacy_truth_decision_sha256": decision["decision_sha256"],
        "legacy_truth_decision_event_id": decision["decision_event_id"],
        "accepted_event_set_sha256": _sha_json(
            decision["accepted_event_ids"]
        ),
        "truth_epoch_sha256": decision["truth_epoch_sha256"],
        "legacy_shape_restored": False,
        "ok": True,
        "secret_material_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha_json(unsigned)}
    runtime._validate_apply_receipt(receipt, plan, "database_apply")

    tampered = copy.deepcopy(receipt)
    tampered["legacy_truth_decision_sha256"] = "c" * 64
    tampered["receipt_sha256"] = _sha_json({
        key: value
        for key, value in tampered.items()
        if key != "receipt_sha256"
    })
    with pytest.raises(runtime.ArtifactError, match="apply_receipt_invalid"):
        runtime._validate_apply_receipt(tampered, plan, "database_apply")


def test_host_token_transition_is_secret_free_resumable_and_reversible(
    tmp_path, monkeypatch
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-activation"]["path"]),
        "production_host_token_transition_artifact",
    )
    uid = tmp_path.stat().st_uid
    gid = tmp_path.stat().st_gid
    gateway_uid = uid + 1
    source = tmp_path / "gateway" / "discord-token"
    target = tmp_path / "connector" / "bot-token"
    routeback = tmp_path / "routeback" / "bot-token"
    for parent in (source.parent, target.parent, routeback.parent):
        parent.mkdir(mode=0o700)
        parent.chmod(0o700)
    secret = b"opaque-discord-token-value-never-in-a-receipt"
    source.write_bytes(secret)
    source.chmod(0o400)
    routeback.write_bytes(b"separate-routeback-only-token-value")
    routeback.chmod(0o400)
    monkeypatch.setattr(runtime, "CONNECTOR_TOKEN", target)
    transition = {
        "connector_token": {
            "path": str(target),
            "uid": uid,
            "gid": gid,
            "mode": 0o400,
            "source_path": str(source),
            "source_uid": uid,
            "source_gid": gid,
            "source_mode": 0o400,
        },
        "gateway_retired_token_paths": [str(source)],
        "routeback_token_paths": [str(routeback)],
        "files": {"gateway_config": {"uid": gateway_uid}},
    }

    runtime._move_token_to_connector(transition)
    assert not source.exists()
    assert target.read_bytes() == secret
    runtime._move_token_to_connector(transition)
    runtime._restore_token_to_source(transition)
    assert source.read_bytes() == secret
    assert not target.exists()
    runtime._restore_token_to_source(transition)


def test_host_api_verifier_foundation_is_secret_free_resumable_and_reversible(
    tmp_path, monkeypatch
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-activation"]["path"]),
        "production_host_approval_passkey_artifact",
    )
    uid = tmp_path.stat().st_uid
    gid = tmp_path.stat().st_gid
    source = tmp_path / "staged" / "api-approval-passkey"
    target = tmp_path / "keys" / "api-approval-passkey-scrypt.json"
    bearer_source = tmp_path / "legacy" / "api-server-control.key"
    bearer_target = tmp_path / "keys" / "api-server-bearer-sha256.json"
    legacy = tmp_path / "legacy" / "api-approval-passkey"
    for parent in {source.parent, target.parent, legacy.parent}:
        parent.mkdir(mode=0o700, exist_ok=True)
        parent.chmod(0o700)
    secret = "distinct-owner-approval-passkey-value-0001"
    bearer_secret = "distinct-api-bearer-value-for-tests-00000001"
    source.write_bytes(secret.encode())
    source.chmod(0o400)
    bearer_source.write_bytes(bearer_secret.encode())
    bearer_source.chmod(0o400)
    target.write_bytes(
        build_api_approval_scrypt_verifier(secret, salt=b"s" * 32)
    )
    target.chmod(0o400)
    bearer_target.write_bytes(build_api_bearer_verifier(bearer_secret))
    bearer_target.chmod(0o400)
    legacy.write_bytes(b"retired-owner-passkey-must-not-survive-0001")
    legacy.chmod(0o400)
    monkeypatch.setattr(runtime, "API_APPROVAL_VERIFIER", target)
    monkeypatch.setattr(runtime, "API_APPROVAL_PASSKEY", target)
    monkeypatch.setattr(runtime, "API_BEARER_VERIFIER", bearer_target)
    monkeypatch.setattr(runtime, "LEGACY_API_BEARER", bearer_source)
    monkeypatch.setattr(runtime, "API_SECRET_SOURCE_UID", uid)
    monkeypatch.setattr(runtime, "API_SECRET_SOURCE_GID", gid)
    monkeypatch.setattr(runtime, "API_VERIFIER_UID", uid)
    monkeypatch.setattr(runtime, "API_VERIFIER_GID", gid)
    transition = {
        "approval_passkey": {
            "path": str(target),
            "uid": uid,
            "gid": gid,
            "mode": 0o400,
            "source_path": str(source),
            "source_uid": uid,
            "source_gid": gid,
            "source_mode": 0o400,
        },
        "retired_approval_passkey_paths": sorted([str(source), str(legacy)]),
        "files": {
            "api_bearer_verifier": {
                "sha256": hashlib.sha256(bearer_target.read_bytes()).hexdigest()
            },
            "api_approval_verifier": {
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()
            },
        },
    }

    with pytest.raises(
        runtime.ArtifactError, match="approval_passkey_lease_survived"
    ):
        runtime._validate_api_verifier_foundation(transition)
    assert source.exists()
    assert target.exists()
    legacy.unlink()

    receipt = runtime._validate_api_verifier_foundation(transition)
    assert receipt["source_secrets_loaded_by_gateway"] is False
    assert source.read_bytes() == secret.encode()
    assert bearer_source.read_bytes() == bearer_secret.encode()
    runtime._validate_api_verifier_foundation(transition)
    target.unlink()
    bearer_target.unlink()
    runtime._restore_approval_passkey(transition)
    assert source.read_bytes() == secret.encode()
    assert not target.exists()
    assert not bearer_target.exists()

    source.chmod(0o600)
    source.write_bytes(b"short")
    source.chmod(0o400)
    with pytest.raises(runtime.ArtifactError, match="approval_passkey_invalid"):
        runtime._move_approval_passkey(transition)
    assert source.read_bytes() == b"short"
    assert not target.exists()


def test_host_boundary_embeds_exact_connector_unit_and_gateway_drop_in(
    tmp_path, monkeypatch
):
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release, REVISION, unit_inputs=_unit_inputs()
    )
    runtime = _load_artifact(
        Path(manifest["artifacts"]["production-host-activation"]["path"]),
        "production_host_reviewed_boundary_artifact",
    )
    plan = _cutover_plan("f" * 64)
    transition = plan["host_transition"]
    files = transition["files"]
    target_policy = transition["discord_policy_continuity"]["target_policy"]
    connector_config = {
        "service": {
            "socket_path": "/run/muncho-discord-connector/connector.sock",
            "gateway_unit": "hermes-cloud-gateway.service",
            "connector_unit": "muncho-discord-connector.service",
            "gateway_uid": files["gateway_config"]["uid"],
            "connector_uid": transition["connector_token"]["uid"],
            "connector_gid": transition["connector_token"]["gid"],
            "connection_timeout_seconds": 10,
        },
        "discord": {
            "token_file": "/etc/muncho/discord-connector-credentials/bot-token",
            "credentials_directory": "/etc/muncho/discord-connector-credentials",
            "allowed_guild_ids": target_policy["allowed_guild_ids"],
            "allowed_channel_ids": target_policy["allowed_channel_ids"],
            "allowed_user_ids": target_policy["allowed_user_ids"],
            "allowed_role_ids": target_policy["allowed_role_ids"],
            "free_response_channel_ids": target_policy[
                "free_response_channel_ids"
            ],
            "public_only": False,
            "author_policy": "guild_acl",
            "allow_bot_authors": False,
            "require_mention": True,
            "auto_thread": True,
            "thread_require_mention": False,
            "reviewed_cron_history_targets": {
                "06ef64d72891": ["1504852355588423801"],
                "e62f55ca93ca": ["1524321461714681976"],
            },
            "ready_timeout_seconds": 30,
            "request_timeout_seconds": 15,
        },
        "journal": {
            "path": "/var/lib/muncho-discord-connector/connector.sqlite3",
            "busy_timeout_ms": 5000,
        },
    }
    by_target = {
        "/etc/systemd/system/muncho-discord-connector.service": (
            runtime.CONNECTOR_UNIT_TEMPLATE.replace(
                "@EXACT_12_CHAR_SHA@", REVISION[:12]
            ).encode()
        ),
        str(runtime.GATEWAY_CONNECTOR_DROP_IN): (
            runtime.GATEWAY_CONNECTOR_DROP_IN_BYTES
        ),
        "/etc/systemd/system/hermes-cloud-gateway.service": (
            b"[Service]\n"
            b"LoadCredential=api-server-bearer-sha256:"
            b"/etc/muncho/keys/api-server-bearer-sha256.json\n"
            b"LoadCredential=api-approval-passkey-scrypt:"
            b"/etc/muncho/keys/api-approval-passkey-scrypt.json\n"
            b"InaccessiblePaths=-/etc/muncho/keys/api-server-control.key\n"
            b"InaccessiblePaths=-/var/lib/muncho-production-legacy-cutover/"
            b"staged/api-approval-passkey\n"
        ),
        "/opt/adventico-ai-platform/hermes-home/config.yaml": b"model: gpt-5.6-sol\n",
        "/etc/muncho-canonical-writer/writer.json": json.dumps({
            "discord_edge_authority": {
                "enabled": True,
                "capability_private_key_file": (
                    "/etc/muncho/keys/writer-capability-private.pem"
                ),
                "edge_receipt_public_key_file": (
                    "/etc/muncho/keys/discord-edge-receipt-public.pem"
                ),
                "edge_receipt_public_key_id": "d" * 64,
                "request_timeout_seconds": 15,
            }
        }).encode(),
        "/etc/muncho/discord-public-connector.json": json.dumps(
            connector_config
        ).encode(),
        "/etc/muncho/discord-edge.json": json.dumps({
            "keys": {
                "writer_capability_public_key_file": (
                    "/etc/muncho/keys/writer-capability-public.pem"
                ),
                "writer_capability_public_key_id": "c" * 64,
                "edge_receipt_private_key_file": (
                    "/run/credentials/muncho-discord-egress.service/"
                    "discord-edge-receipt-private-key"
                ),
                "edge_receipt_public_key_id": "d" * 64,
            }
        }).encode(),
    }
    monkeypatch.setattr(
        runtime,
        "_staged_file",
        lambda item: by_target[item["target_path"]],
    )
    runtime._assert_reviewed_connector_boundary(plan, transition)

    by_target["/etc/systemd/system/hermes-cloud-gateway.service"] = (
        b"Environment=DISCORD_BOT_TOKEN=forbidden\n"
    )
    with pytest.raises(
        runtime.ArtifactError, match="gateway_discord_credential_survived"
    ):
        runtime._assert_reviewed_connector_boundary(plan, transition)

    by_target["/etc/systemd/system/hermes-cloud-gateway.service"] = (
        b"[Service]\n"
        b"LoadCredential=api-server-bearer-sha256:"
        b"/etc/muncho/keys/api-server-bearer-sha256.json\n"
        b"LoadCredential=api-approval-passkey-scrypt:"
        b"/etc/muncho/keys/api-approval-passkey-scrypt.json\n"
        b"Environment=API_SERVER_APPROVAL_PASSKEY=forbidden\n"
    )
    with pytest.raises(
        runtime.ArtifactError, match="gateway_credential_boundary_invalid"
    ):
        runtime._assert_reviewed_connector_boundary(plan, transition)


def test_deploy_packages_and_verifies_before_release_activation():
    source = (ROOT / "ops/muncho/runtime/muncho-auto-deploy-release").read_text()
    run_deploy = source[source.index("run_deploy() {") : source.index("main() {")]
    copy_venv = run_deploy.index('cp -a "$active/.venv" "$tmp/.venv"')
    install_wheel = run_deploy.index(
        'install_target_release_wheel "$tmp" "$active"'
    )
    build = run_deploy.index("package_production_cutover_artifacts.py\" build")
    verify = run_deploy.index("package_production_cutover_artifacts.py\" verify")
    publish = run_deploy.index('mv "$tmp" "$new"')
    release_identity = run_deploy.index(
        'release_identity_matches "$new" "$sha"'
    )
    attest_venv = run_deploy.index(
        'attest_target_release_venv "$new" "$new"'
    )
    cutover_attest = run_deploy.index('cutover_artifacts_match "$new" "$sha"')
    activate = run_deploy.index('ln -sfn "$new" "$ACTIVE_LINK.next"')
    bootstrap = run_deploy.index(
        'bootstrap_cutover_unit_inputs_from_target "$tmp" "$sha"'
    )
    require_inputs = run_deploy.index(
        'require_cutover_unit_inputs "$sha" "$pr"',
        bootstrap,
    )
    dependency_prepare = run_deploy.index(
        'package_production_runtime_dependencies.py" prepare'
    )
    config_seal = run_deploy.index(
        'seal_agent_browser_config "$tmp" "$sha"'
    )
    dependency_manifest = run_deploy.index(
        'package_production_runtime_dependencies.py" build-manifest'
    )

    assert (
        copy_venv
        < bootstrap
        < require_inputs
        < install_wheel
        < dependency_prepare
        < config_seal
        < dependency_manifest
        < build
        < verify
        < publish
        < release_identity
        < attest_venv
        < cutover_attest
        < activate
    )
    assert run_deploy.count('--unit-inputs "$CUTOVER_UNIT_INPUTS_PATH"') >= 2
    assert 'cutover_artifacts_match "$new" "$sha"' in run_deploy
    assert 'blocked_target_cutover_artifacts_invalid' in run_deploy
    install = source[
        source.index("install_target_release_wheel() {") : source.index(
            "release_identity_matches() {"
        )
    ]
    for flag in (
        "--isolated install",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--no-cache-dir",
        "--force-reinstall",
    ):
        assert flag in install
    assert '"$release" >/dev/null' in install
