from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_capability_canary_e2e as e2e
from gateway import canonical_capability_canary_producers as producers
from gateway.operational_edge_protocol import sign_envelope


NOW_MS = 1_800_000_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _full_canary_terminal_receipt() -> dict[str, Any]:
    unsigned = {
        "schema": "muncho-full-canary-session-bound-owner-receipt.v1",
        "ok": True,
        "state": "verified_stopped_and_credentials_retired",
        "release_sha": "a" * 40,
        "coordinator_input_sha256": "1" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "owner_approval_sha256": "3" * 64,
        "phase_b_readiness_anchor_sha256": "4" * 64,
        "api_session_key_sha256": "5" * 64,
        "fixture_sha256": "6" * 64,
        "discord_token_install_receipt_sha256": "7" * 64,
        "coordinator_receipt_sha256": "8" * 64,
        "live_driver_receipt_sha256": "9" * 64,
        "services_stopped": True,
        "discord_token_retired": True,
        "temporary_admin_created": False,
        "bootstrap_credential_created": False,
        "completed_at_unix": 1_900_000_000,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _sshsig(
    private_key: Ed25519PrivateKey,
    message: bytes,
    *,
    namespace: str,
) -> str:
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public)
    namespace_bytes = namespace.encode("ascii")
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(hashlib.sha512(message).digest())
    )
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(private_key.sign(signed))
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + _ssh_string(public_blob)
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(signature_blob)
    )
    encoded = base64.b64encode(envelope).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _foundation(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    owner_private = Ed25519PrivateKey.generate()
    role_private = {
        role: Ed25519PrivateKey.generate() for role in producers.ENDPOINT_ROLES
    }
    public = {
        role: role_private[role]
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
        for role in producers.ENDPOINT_ROLES
    }
    owner_public = (
        owner_private
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )
    authorities = {
        role: {
            "key_id": _sha(bytes.fromhex(public[role])),
            "algorithm": "ed25519",
            "public_key_ed25519_hex": public[role],
        }
        for role in producers.ENDPOINT_ROLES
    }
    authorities["owner"] = {
        "key_id": _sha(bytes.fromhex(owner_public)),
        "algorithm": "sshsig-ed25519-sha512",
        "public_key_ed25519_hex": owner_public,
    }
    source = {
        "kind": "skyvision_mac_ops_ed25519",
        "path": "/Users/emillomliev/.ssh/skyvision_mac_ops_ed25519.pub",
        "comment": "skyvision-mac-ops-emil-20260710",
        "fingerprint": "SHA256:7Ea5WNys9ui7FL/p0FlOnL1ZLr6NPFuewekwqRw/rdw",
        "file_sha256": "9" * 64,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "mode": 0o600,
        "size": 128,
    }
    source_sha256 = _sha(_canonical(source))
    endpoints: dict[str, Any] = {}
    if os.getuid() == 0 or os.getgid() == 0:
        pytest.skip("producer identity tests require an unprivileged test user")
    for index, role in enumerate(producers.ENDPOINT_ROLES):
        endpoints[role] = {
            "service_unit": f"muncho-capability-producer-{role}.service",
            "service_identity_sha256": _sha(f"identity:{role}".encode()),
            "uid": os.getuid() + index,
            "gid": os.getgid() + index,
            "socket_path": str(tmp_path / f"{role}.sock"),
            "private_key_path": str(tmp_path / f"{role}.private.pem"),
            "public_key_path": str(tmp_path / f"{role}.public.pem"),
            "public_key_file_sha256": _sha(f"public-file:{role}".encode()),
            "allowed_slots": [
                slot
                for slot in producers.RECEIPT_SLOTS
                if producers.SLOT_ROLE[slot] == role
            ],
            "key_id": authorities[role]["key_id"],
            "algorithm": "ed25519",
            "public_key_ed25519_hex": public[role],
        }
    terminal = _full_canary_terminal_receipt()
    unsigned = {
        "schema": producers.PRODUCER_FOUNDATION_SCHEMA,
        "release_sha": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "service_identity_foundation_receipt_sha256": "8" * 64,
        "producer_identity_foundation_receipt_sha256": "7" * 64,
        "owner_id": producers.PRODUCTION_OWNER_ID,
        "owner_authority": {
            "owner_id": producers.PRODUCTION_OWNER_ID,
            "key_id": authorities["owner"]["key_id"],
            "algorithm": "sshsig-ed25519-sha512",
            "public_key_ed25519_hex": owner_public,
            "public_key_source": source,
            "public_key_source_sha256": source_sha256,
        },
        "authority_keys": authorities,
        "endpoints": endpoints,
        "bitrix_operational_edge_contract": {
            "revision": "a" * 40,
            "service_unit": producers.BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT,
            "service_identity_sha256": "1" * 64,
            "asset_manifest_sha256": "2" * 64,
            "asset_names": list(producers.BITRIX_OPERATIONAL_EDGE_ASSET_NAMES),
            "asset_manifest_path": (
                f"/opt/muncho-canary-releases/{'a' * 40}/ops/muncho/runtime/"
                "operational-assets/manifest.json"
            ),
            "rendered_unit_sha256": "3" * 64,
            "rendered_unit_path": producers.BITRIX_OPERATIONAL_EDGE_UNIT_PATH,
            "rendered_config_sha256": "4" * 64,
            "rendered_config_path": producers.BITRIX_OPERATIONAL_EDGE_CONFIG_PATH,
            "rendered_trust_sha256": "5" * 64,
            "rendered_trust_path": producers.BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
            "identity_bootstrap": {
                "service_user": producers.BITRIX_OPERATIONAL_EDGE_SERVICE_USER,
                "service_group": producers.BITRIX_OPERATIONAL_EDGE_SERVICE_GROUP,
                "service_uid": 2101,
                "service_gid": 2102,
                "socket_client_group": producers.BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP,
                "socket_client_gid": 2103,
                "receipt_sha256": "6" * 64,
            },
            "credential_projection": {
                "name": "bitrix-webhook-url",
                "source_path": (
                    "/opt/adventico-ai-platform/hermes-home/secrets/"
                    "bitrix_skyvision_crm_webhook.url"
                ),
                "projected_path": (
                    "/run/credentials/muncho-operational-edge-bitrix.service/"
                    "bitrix-webhook-url"
                ),
                "bind_target_path": (
                    "/opt/adventico-ai-platform/hermes-home/secrets/"
                    "bitrix_skyvision_crm_webhook.url"
                ),
                "source_owner_uid": 0,
                "source_owner_gid": 0,
                "source_mode": "0400",
                "service_reads_projection": True,
                "original_source_inaccessible": True,
                "value_or_digest_recorded": False,
            },
            "receipt_key_contract": {
                "private_credential_name": "receipt-private-key",
                "private_source_path": (
                    "/etc/muncho/keys/operational-edge-bitrix-receipt-private.pem"
                ),
                "private_projection_path": (
                    "/run/credentials/muncho-operational-edge-bitrix.service/"
                    "receipt-private-key"
                ),
                "private_owner_uid": 0,
                "private_owner_gid": 0,
                "private_mode": "0400",
                "public_path": producers.BITRIX_OPERATIONAL_EDGE_TRUST_PATH,
                "public_key_id": "7" * 64,
                "public_trust_sha256": "5" * 64,
                "writer_public_key_credential_name": "writer-public-key",
                "writer_public_key_source_path": (
                    "/etc/muncho/keys/writer-capability-public.pem"
                ),
                "writer_public_key_projection_path": (
                    "/run/credentials/muncho-operational-edge-bitrix.service/"
                    "writer-public-key"
                ),
                "key_bootstrap_receipt_sha256": "8" * 64,
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
            "staging_protocol": producers.BITRIX_OPERATIONAL_EDGE_STAGING_PROTOCOL,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        },
        "discord_edge_evidence_contract": {
            "edge_service_unit": "muncho-discord-egress.service",
            "edge_socket_path": "/run/muncho-discord-egress/edge.sock",
            "edge_service_uid": 2110,
            "edge_service_gid": 2210,
            "receipt_public_key_path": (
                "/etc/muncho/keys/discord-edge-receipt-public.pem"
            ),
            "receipt_public_key_id": "a" * 64,
            "receipt_public_key_file_sha256": "b" * 64,
            "connector_service_unit": "muncho-discord-connector.service",
            "connector_socket_path": ("/run/muncho-discord-connector/connector.sock"),
            "connector_service_uid": 2111,
            "connector_service_gid": 2211,
            "public_history_operation": "public.history.fetch",
            "direct_message_allowed": False,
            "token_or_token_digest_recorded": False,
        },
        "receipt_contract": {
            "base_root": str(tmp_path / "receipts"),
            "run_directory_uid": os.getuid(),
            "run_directory_gid": os.getgid(),
            "run_directory_mode": 0o770,
            "slot_filenames": dict(producers.SLOT_FILENAME),
            "slot_roles": dict(producers.SLOT_ROLE),
            "slot_native_binding_kinds": {
                slot: list(producers.SLOT_NATIVE_BINDING_KINDS[slot])
                for slot in producers.RECEIPT_SLOTS
            },
        },
        "producer_protocol": "role_local_native_evidence_v1",
        "root_can_sign_non_observer_roles": False,
        "token_or_token_digest_recorded": False,
        "signature_namespace": producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
        "signature_algorithm": "sshsig-ed25519-sha512",
    }
    signature = _sshsig(
        owner_private,
        producers.producer_foundation_signature_payload(unsigned),
        namespace=producers.PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
    )
    foundation = dict(
        producers.seal_producer_foundation(
            unsigned,
            owner_signature=signature,
            pinned_owner_public_key_ed25519_hex=owner_public,
            pinned_owner_public_key_source_sha256=source_sha256,
        )
    )
    context = {
        "owner_private": owner_private,
        "owner_public": owner_public,
        "source_sha256": source_sha256,
        "role_private": role_private,
    }
    return foundation, context


class _Endpoint:
    def __init__(
        self,
        *,
        role: str,
        foundation: Mapping[str, Any],
        foundation_sha256: str,
    ) -> None:
        endpoint = foundation["endpoints"][role]
        self.role = role
        self.socket_path = Path(endpoint["socket_path"])
        self.expected_peer = producers.PeerIdentity(
            os.getpid(), endpoint["uid"], endpoint["gid"]
        )
        unsigned = {
            "schema": producers.PRODUCER_ENDPOINT_READINESS_SCHEMA,
            "role": role,
            "foundation_sha256": foundation_sha256,
            "release_sha": foundation["release_sha"],
            "capability_plan_sha256": foundation["capability_plan_sha256"],
            "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
            "service_unit": endpoint["service_unit"],
            "service_identity_sha256": endpoint["service_identity_sha256"],
            "main_pid": self.expected_peer.pid,
            "uid": self.expected_peer.uid,
            "gid": self.expected_peer.gid,
            "socket_path": str(self.socket_path),
            "allowed_slots": endpoint["allowed_slots"],
            "key_id": endpoint["key_id"],
            "algorithm": "ed25519",
            "public_key_ed25519_hex": endpoint["public_key_ed25519_hex"],
            "public_key_file_sha256": endpoint["public_key_file_sha256"],
            "private_key_or_digest_present": False,
            "observed_at_unix_ms": NOW_MS,
        }
        self.readiness = {**unsigned, "readiness_sha256": _sha(_canonical(unsigned))}

    def call(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        assert value == {"action": "readiness"}
        return self.readiness


def _routeback_identity() -> dict[str, Any]:
    from gateway.canonical_capability_canary_runtime import (
        CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA,
    )

    unsigned = {
        "schema": CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA,
        "plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
        "live_bot_user_id": "1501976597455044802",
        "planned_routeback_bot_user_id": "1501976597455044802",
        "connector_bot_user_id": "1501976597455044803",
        "production_bot_user_id": producers.PRODUCTION_BOT_USER_ID,
        "provenance": {
            "source": "discord_rest_api_v10_current_user",
            "http_method": "GET",
            "resource": "/users/@me",
            "credential_boundary": "sealed_routeback_credential_file",
        },
        "pairwise_distinct": True,
        "observed_at_unix": NOW_MS // 1000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "credential_file_metadata_sha256": "d" * 64,
    }
    return {**unsigned, "attestation_sha256": _sha(_canonical(unsigned))}


def _fixture(
    foundation: Mapping[str, Any],
    *,
    run_id: str,
    start_ms: int,
) -> dict[str, Any]:
    return {
        "schema": "muncho-production-capability-canary-fixture.v1",
        "release_sha": foundation["release_sha"],
        "run_id": run_id,
        "owner_id": producers.PRODUCTION_OWNER_ID,
        "valid_from_unix_ms": start_ms,
        "valid_until_unix_ms": start_ms + 600_000,
        "producer_foundation_sha256": producers.producer_foundation_sha256(foundation),
        "discord_bot_identities": {
            "production_bot_user_id": producers.PRODUCTION_BOT_USER_ID,
            "connector_bot_user_id": "1501976597455044803",
            "routeback_bot_user_id": "1501976597455044802",
        },
        "authority_keys": foundation["authority_keys"],
    }


def _activation(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    foundation, context = _foundation(tmp_path)
    foundation_sha256 = producers.producer_foundation_sha256(foundation)
    fixture = _fixture(foundation, run_id="run-one", start_ms=NOW_MS - 1000)
    run_root = Path(foundation["receipt_contract"]["base_root"]) / "run-one"
    run_root.mkdir(parents=True)
    os.chown(run_root, os.getuid(), os.getgid())
    run_root.chmod(0o770)
    clients = {
        role: _Endpoint(
            role=role,
            foundation=foundation,
            foundation_sha256=foundation_sha256,
        )
        for role in producers.ENDPOINT_ROLES
    }
    activation = producers.build_fleet_activation(
        foundation=foundation,
        fixture=fixture,
        fixture_sha256=_sha(_canonical(fixture)),
        pinned_owner_public_key_ed25519_hex=context["owner_public"],
        pinned_owner_public_key_source_sha256=context["source_sha256"],
        owner_grant_sha256="e" * 64,
        owner_catalog_sha256="f" * 64,
        owner_authority_sha256="1" * 64,
        routeback_bot_identity=_routeback_identity(),
        endpoint_clients=clients,
        now_ms=lambda: NOW_MS,
    )
    return foundation, context, fixture, activation


def test_foundation_is_externally_pinned_and_run_independent(tmp_path: Path) -> None:
    foundation, context = _foundation(tmp_path)
    digest = producers.producer_foundation_sha256(foundation)
    first = _fixture(foundation, run_id="run-one", start_ms=NOW_MS)
    second = _fixture(foundation, run_id="run-two", start_ms=NOW_MS + 10_000)

    assert first["producer_foundation_sha256"] == digest
    assert second["producer_foundation_sha256"] == digest
    assert _sha(_canonical(first)) != _sha(_canonical(second))
    with pytest.raises(
        producers.CapabilityProducerError,
        match="producer_foundation_invalid",
    ):
        producers.validate_producer_foundation(
            foundation,
            pinned_owner_public_key_ed25519_hex=(
                Ed25519PrivateKey
                .generate()
                .public_key()
                .public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
                .hex()
            ),
            pinned_owner_public_key_source_sha256=context["source_sha256"],
        )


def test_activation_binds_fixture_exact_run_root_and_live_endpoints(
    tmp_path: Path,
) -> None:
    foundation, _context, fixture, activation = _activation(tmp_path)

    assert activation["foundation_sha256"] == producers.producer_foundation_sha256(
        foundation
    )
    assert activation["fixture_sha256"] == _sha(_canonical(fixture))
    assert activation["run_receipt_root"].endswith("/run-one")
    assert activation["slot_filenames"] == producers.SLOT_FILENAME
    assert (
        producers.validate_fleet_readiness(
            activation,
            now_ms=NOW_MS,
            expected_foundation_sha256=activation["foundation_sha256"],
        )
        == activation
    )


def test_expired_activation_can_be_exactly_retired_after_service_stop(
    tmp_path: Path,
) -> None:
    _foundation_value, _context, _fixture_value, activation = _activation(tmp_path)
    path = tmp_path / "activation.json"
    parent = tmp_path.lstat()
    producers.publish_fleet_readiness(
        activation,
        path=path,
        uid=parent.st_uid,
        gid=parent.st_gid,
        now_ms=NOW_MS,
    )
    retirement_directory = tmp_path / "activation-retirement"
    retirement_directory.mkdir(mode=0o700)
    retirement = producers.retire_fleet_readiness(
        expected_readiness_sha256=activation["readiness_sha256"],
        run_id=activation["run_id"],
        retirement_directory=retirement_directory,
        path=path,
        uid=parent.st_uid,
        gid=parent.st_gid,
        retirement_uid=parent.st_uid,
        retirement_gid=parent.st_gid,
        expected_foundation_sha256=activation["foundation_sha256"],
        expected_capability_plan_sha256=activation["capability_plan_sha256"],
        expected_full_canary_plan_sha256=activation["full_canary_plan_sha256"],
        retired_at_unix_ms=activation["valid_until_unix_ms"] + 1,
    )
    assert retirement["absence_verified"] is True
    assert not os.path.lexists(path)
    assert (
        producers.retire_fleet_readiness(
            expected_readiness_sha256=activation["readiness_sha256"],
            run_id=activation["run_id"],
            retirement_directory=retirement_directory,
            path=path,
            uid=parent.st_uid,
            gid=parent.st_gid,
            retirement_uid=parent.st_uid,
            retirement_gid=parent.st_gid,
            expected_foundation_sha256=activation["foundation_sha256"],
            expected_capability_plan_sha256=activation["capability_plan_sha256"],
            expected_full_canary_plan_sha256=(
                activation["full_canary_plan_sha256"]
            ),
            retired_at_unix_ms=activation["valid_until_unix_ms"] + 2,
        )
        == retirement
    )


class _ExactCollector:
    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> tuple[producers.NativeEvidenceBinding, ...]:
        artifact = _sha(_canonical(payload))
        return tuple(
            producers.NativeEvidenceBinding(
                kind=kind,
                source_identity_sha256=_sha(f"source:{kind}".encode()),
                artifact_sha256=artifact,
                verification_receipt_sha256=_sha(
                    f"verified:{slot}:{kind}:{artifact}".encode()
                ),
            )
            for kind in producers.SLOT_NATIVE_BINDING_KINDS[slot]
        )


def test_role_producer_cannot_sign_before_activation_and_publishes_exact_path(
    tmp_path: Path,
) -> None:
    foundation, context, fixture, activation = _activation(tmp_path)
    role = "business_edge"
    endpoint = foundation["endpoints"][role]
    receipt = foundation["receipt_contract"]
    config = producers.ProducerConfig.from_mapping({
        "schema": producers.PRODUCER_CONFIG_SCHEMA,
        "role": role,
        "foundation_sha256": producers.producer_foundation_sha256(foundation),
        "release_sha": foundation["release_sha"],
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "service_unit": endpoint["service_unit"],
        "service_identity_sha256": endpoint["service_identity_sha256"],
        "service_uid": os.getuid(),
        "service_gid": os.getgid(),
        "root_client_uid": 0,
        "socket_path": endpoint["socket_path"],
        "receipt_base_root": receipt["base_root"],
        "receipt_directory_uid": receipt["run_directory_uid"],
        "receipt_directory_gid": receipt["run_directory_gid"],
        "receipt_directory_mode": receipt["run_directory_mode"],
        "private_key_path": endpoint["private_key_path"],
        "public_key_path": endpoint["public_key_path"],
        "allowed_slots": endpoint["allowed_slots"],
    })
    producers.validate_producer_config_binding(config, foundation)
    private_key = context["role_private"][role]
    producer = producers.RoleReceiptProducer(
        config,
        native_collector=_ExactCollector(),
        private_key=private_key,
        public_key=private_key.public_key(),
        now_ms=lambda: NOW_MS,
    )
    request = {
        "schema": producers.PRODUCER_REQUEST_SCHEMA,
        "slot": "bitrix_edge",
        "role": role,
        "run_id": fixture["run_id"],
        "release_sha": foundation["release_sha"],
        "fixture_sha256": _sha(_canonical(fixture)),
        "producer_readiness_sha256": activation["readiness_sha256"],
        "payload": {"schema": "test", "observed_at_unix_ms": NOW_MS},
    }
    with pytest.raises(
        producers.CapabilityProducerError,
        match="producer_request_invalid",
    ):
        producer.produce(request)

    producer.activate(activation)
    signed = producer.produce(request)
    path = (
        Path(receipt["base_root"])
        / fixture["run_id"]
        / producers.SLOT_FILENAME["bitrix_edge"]
    )
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert (
        producers.verify_role_receipt(
            signed,
            role=role,
            slot="bitrix_edge",
            public_key=private_key.public_key(),
            producer_readiness_sha256=activation["readiness_sha256"],
        )
        == request["payload"]
    )


def test_production_pump_uses_action_produce_and_verifies_immutable_receipt(
    tmp_path: Path,
) -> None:
    foundation, context, fixture, activation = _activation(tmp_path)
    role = "business_edge"
    endpoint = foundation["endpoints"][role]
    receipt_contract = foundation["receipt_contract"]
    config = producers.ProducerConfig.from_mapping({
        "schema": producers.PRODUCER_CONFIG_SCHEMA,
        "role": role,
        "foundation_sha256": producers.producer_foundation_sha256(foundation),
        "release_sha": foundation["release_sha"],
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "service_unit": endpoint["service_unit"],
        "service_identity_sha256": endpoint["service_identity_sha256"],
        "service_uid": os.getuid(),
        "service_gid": os.getgid(),
        "root_client_uid": 0,
        "socket_path": endpoint["socket_path"],
        "receipt_base_root": receipt_contract["base_root"],
        "receipt_directory_uid": receipt_contract["run_directory_uid"],
        "receipt_directory_gid": receipt_contract["run_directory_gid"],
        "receipt_directory_mode": receipt_contract["run_directory_mode"],
        "private_key_path": endpoint["private_key_path"],
        "public_key_path": endpoint["public_key_path"],
        "allowed_slots": endpoint["allowed_slots"],
    })
    private_key = context["role_private"][role]
    producer = producers.RoleReceiptProducer(
        config,
        native_collector=_ExactCollector(),
        private_key=private_key,
        public_key=private_key.public_key(),
        now_ms=lambda: NOW_MS,
    )
    producer.activate(activation)

    clients = {
        item_role: _Endpoint(
            role=item_role,
            foundation=foundation,
            foundation_sha256=activation["foundation_sha256"],
        )
        for item_role in producers.ENDPOINT_ROLES
    }
    calls: list[Mapping[str, Any]] = []

    class ProducingEndpoint:
        socket_path = clients[role].socket_path
        expected_peer = clients[role].expected_peer

        def call(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            calls.append(value)
            assert value.get("action") == "produce"
            return producer.produce(value["request"])

    clients[role] = ProducingEndpoint()
    pump = producers.ProductionReceiptPump(
        installed_foundation=producers.InstalledProducerFoundation(
            value=foundation,
            pinned_owner_public_key_ed25519_hex=context["owner_public"],
            pinned_owner_public_key_source_sha256=context["source_sha256"],
        ),
        readiness=activation,
        endpoint_clients=clients,
    )
    payload = {
        "schema": "test-production-pump.v1",
        "run_id": fixture["run_id"],
        "release_sha": foundation["release_sha"],
        "fixture_sha256": activation["fixture_sha256"],
        "observed_at_unix_ms": NOW_MS,
    }

    signed = pump.produce(slot="bitrix_edge", payload=payload)

    assert len(calls) == 1
    assert (
        calls[0]["request"]["producer_readiness_sha256"]
        == activation["readiness_sha256"]
    )
    receipt_path = (
        Path(activation["run_receipt_root"]) / producers.SLOT_FILENAME["bitrix_edge"]
    )
    assert json.loads(receipt_path.read_text()) == signed
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400


def test_atomic_publication_is_idempotent_and_divergence_fails_closed(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "receipt.json"
    parent = tmp_path.stat()
    kwargs = {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "mode": 0o400,
        "parent_uid": parent.st_uid,
        "parent_gid": parent.st_gid,
    }
    producers._publish_no_replace(path, b'{"ok":true}', **kwargs)
    producers._publish_no_replace(path, b'{"ok":true}', **kwargs)
    with pytest.raises(
        producers.CapabilityProducerError,
        match="publication_collision_diverged",
    ):
        producers._publish_no_replace(path, b'{"ok":false}', **kwargs)
    assert path.read_bytes() == b'{"ok":true}'


def test_linux_atomic_publication_uses_noreplace_on_validated_directory_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    calls: list[tuple[Any, ...]] = []

    class RenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    renameat2 = RenameAt2()
    library = type("Library", (), {"renameat2": renameat2})()
    monkeypatch.setattr(producers.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: library)

    producers._rename_no_replace_at(
        ".receipt.tmp",
        "receipt.json",
        directory_fd=37,
    )

    assert calls == [(37, b".receipt.tmp", 37, b"receipt.json", 1)]


def test_atomic_publication_concurrent_same_payload_has_one_immutable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "receipt.json"
    parent = tmp_path.stat()

    publishers = 8
    barrier = threading.Barrier(publishers)
    real_publish = producers._rename_no_replace_at

    def synchronized_publish(*args: Any, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        real_publish(*args, **kwargs)

    monkeypatch.setattr(producers, "_rename_no_replace_at", synchronized_publish)

    def publish() -> None:
        producers._publish_no_replace(
            path,
            b'{"receipt":1}',
            uid=os.getuid(),
            gid=os.getgid(),
            mode=0o400,
            parent_uid=parent.st_uid,
            parent_gid=parent.st_gid,
        )

    with ThreadPoolExecutor(max_workers=publishers) as pool:
        list(pool.map(lambda _index: publish(), range(publishers)))
    assert path.read_bytes() == b'{"receipt":1}'
    assert stat.S_IMODE(path.stat().st_mode) == 0o400


def test_atomic_publication_failure_before_link_leaves_no_final_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path.stat()
    path = tmp_path / "receipt.json"
    real_publish = producers._rename_no_replace_at

    def fail_publish(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected-before-publication")

    monkeypatch.setattr(producers, "_rename_no_replace_at", fail_publish)
    with pytest.raises(OSError, match="injected-before-publication"):
        producers._publish_no_replace(
            path,
            b'{"receipt":1}',
            uid=os.getuid(),
            gid=os.getgid(),
            parent_uid=parent.st_uid,
            parent_gid=parent.st_gid,
        )
    assert not path.exists()
    assert not list(tmp_path.glob(".receipt.json.tmp.*"))

    monkeypatch.setattr(producers, "_rename_no_replace_at", real_publish)
    producers._publish_no_replace(
        path,
        b'{"receipt":1}',
        uid=os.getuid(),
        gid=os.getgid(),
        parent_uid=parent.st_uid,
        parent_gid=parent.st_gid,
    )
    assert path.read_bytes() == b'{"receipt":1}'


def test_atomic_publication_concurrent_divergence_never_replaces_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path.stat()
    path = tmp_path / "receipt.json"

    barrier = threading.Barrier(2)
    real_publish = producers._rename_no_replace_at

    def synchronized_publish(*args: Any, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        real_publish(*args, **kwargs)

    monkeypatch.setattr(producers, "_rename_no_replace_at", synchronized_publish)

    def publish(payload: bytes) -> str:
        try:
            producers._publish_no_replace(
                path,
                payload,
                uid=os.getuid(),
                gid=os.getgid(),
                parent_uid=parent.st_uid,
                parent_gid=parent.st_gid,
            )
            return "published"
        except producers.CapabilityProducerError as exc:
            return exc.code

    payloads = (b'{"receipt":1}', b'{"receipt":2}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, payloads))
    assert sorted(results) == ["publication_collision_diverged", "published"]
    assert path.read_bytes() in payloads


def test_atomic_publication_recovers_exact_legacy_two_link_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path.stat()
    path = tmp_path / "receipt.json"
    payload = b'{"receipt":"legacy-crash"}'
    temporary = tmp_path / f".{path.name}.tmp.4242.{'a' * 32}"
    temporary.write_bytes(payload)
    temporary.chmod(0o400)
    os.link(temporary, path)
    assert path.stat().st_nlink == 2
    monkeypatch.setattr(producers.time, "sleep", lambda _seconds: None)

    producers._publish_no_replace(
        path,
        payload,
        uid=os.getuid(),
        gid=path.lstat().st_gid,
        parent_uid=parent.st_uid,
        parent_gid=parent.st_gid,
    )

    assert path.read_bytes() == payload
    assert path.stat().st_nlink == 1
    assert not temporary.exists()


@pytest.mark.parametrize(
    "hazard",
    ("multiple_mismatched", "malformed_name", "link_count_above_two"),
)
def test_legacy_link_recovery_rejects_ambiguous_or_unbounded_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path.stat()
    path = tmp_path / "receipt.json"
    payload = b'{"receipt":"legacy-crash"}'
    temporary = tmp_path / f".{path.name}.tmp.4242.{'a' * 32}"
    if hazard == "malformed_name":
        temporary = tmp_path / f".{path.name}.tmp.malformed"
    temporary.write_bytes(payload)
    temporary.chmod(0o400)
    os.link(temporary, path)
    if hazard == "multiple_mismatched":
        extra = tmp_path / f".{path.name}.tmp.4343.{'b' * 32}"
        extra.write_bytes(b'{"receipt":"different"}')
        extra.chmod(0o400)
    elif hazard == "link_count_above_two":
        extra = tmp_path / f".{path.name}.tmp.4343.{'b' * 32}"
        os.link(path, extra)
    monkeypatch.setattr(producers.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        producers.CapabilityProducerError,
        match="artifact_identity_invalid",
    ):
        producers._publish_no_replace(
            path,
            payload,
            uid=os.getuid(),
            gid=os.getgid(),
            parent_uid=parent.st_uid,
            parent_gid=parent.st_gid,
        )

    assert path.read_bytes() == payload
    assert temporary.exists()


def test_legacy_link_recovery_rejects_unsafe_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    parent = tmp_path.stat()
    path = tmp_path / "receipt.json"
    payload = b'{"receipt":"legacy-crash"}'
    temporary = tmp_path / f".{path.name}.tmp.4242.{'a' * 32}"
    temporary.write_bytes(payload)
    temporary.chmod(0o400)
    os.link(temporary, path)
    tmp_path.chmod(0o755)
    monkeypatch.setattr(producers.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        producers.CapabilityProducerError,
        match="directory_identity_invalid",
    ):
        producers._publish_no_replace(
            path,
            payload,
            uid=os.getuid(),
            gid=os.getgid(),
            parent_uid=parent.st_uid,
            parent_gid=parent.st_gid,
        )

    assert path.read_bytes() == payload
    assert temporary.exists()


def test_cleanup_requires_all_six_retirement_journals_and_absence() -> None:
    assert producers.SLOT_NATIVE_BINDING_KINDS["cleanup"][:9] == (
        "systemd_non_observer_services_stopped_state",
        "gateway_observer_cleanup_signer_live_identity",
        "api_control_credential_retirement_journal",
        "routeback_credential_retirement_journal",
        "connector_credential_retirement_journal",
        "codex_credential_retirement_journal",
        "mac_ops_credential_retirement_journal",
        "bitrix_operational_edge_credential_retirement_journal",
        "all_six_credentials_absent_readback",
    )


class _BitrixClient:
    def __init__(
        self,
        *,
        tamper: str | None = None,
        unstable_readback: bool = False,
    ) -> None:
        self.receipt_private_key = Ed25519PrivateKey.generate()
        self.receipt_public_key = self.receipt_private_key.public_key()
        receipt_key_id = _sha(
            self.receipt_public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        )
        self.config = type(
            "Config",
            (),
            {
                "domain": "bitrix",
                "receipt_key_id": receipt_key_id,
                "service_unit": "muncho-operational-edge-bitrix.service",
            },
        )()
        self.calls: list[
            tuple[str, Mapping[str, Any], str, Mapping[str, Any] | None]
        ] = []
        self.tamper = tamper
        self.unstable_readback = unstable_readback

    def invoke_verified_evidence(
        self,
        operation_id: str,
        _arguments: Mapping[str, Any],
        *,
        idempotency_key: str,
        expected_release_revision: str,
        capability: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((operation_id, dict(_arguments), idempotency_key, capability))
        denial = operation_id == "bitrix.crm.lead_add"
        read_ordinal = sum(
            1
            for called_operation, _args, _key, _capability in self.calls
            if called_operation == "bitrix.crm.status_list"
        )
        status_value = (
            "CHANGED" if self.unstable_readback and read_ordinal == 2 else "NEW"
        )
        payload = {
            "operation_id": operation_id,
            "domain": "bitrix",
            "release_revision": expected_release_revision,
            "request_sha256": _sha(f"request:{idempotency_key}".encode()),
            "service_pid": 4321,
            "access": "mutation" if denial else "read",
            "outcome": "blocked" if denial else "succeeded",
            "blocker_code": "mutation_capability_required" if denial else None,
            "dispatched": not denial,
            "executable_started": not denial,
            "mutation_performed": False,
            "readback_verified": not denial,
            "secret_material_recorded": False,
            "stdout_b64": base64.b64encode(
                _canonical({
                    "status": "OK",
                    "generated_at_utc": f"2026-07-15T00:00:0{read_ordinal}Z",
                    "items": [{"ENTITY_ID": "STATUS", "STATUS_ID": status_value}],
                })
            ).decode(),
        }
        envelope = sign_envelope(
            payload,
            key_id=self.config.receipt_key_id,
            private_key=self.receipt_private_key,
        ).to_mapping()
        detached_payload = dict(payload)
        if self.tamper == "payload":
            detached_payload["outcome"] = "blocked"
        if self.tamper == "signature":
            envelope = dict(envelope)
            envelope["signature_b64"] = base64.b64encode(b"x" * 64).decode()
        envelope_sha = _sha(_canonical(envelope))
        if self.tamper == "envelope_digest":
            envelope_sha = "0" * 64
        return {
            "schema": "muncho-operational-edge-verified-evidence.v1",
            "payload": detached_payload,
            "signed_envelope": envelope,
            "signed_envelope_sha256": envelope_sha,
            "request_sha256": payload["request_sha256"],
            "peer": {
                "pid": 4321,
                "uid": 1001,
                "gid": 1002,
                "service_unit": "muncho-operational-edge-bitrix.service",
            },
        }


def test_bitrix_collector_uses_real_fixed_operations_and_signed_denial() -> None:
    client = _BitrixClient()
    collector = producers.BitrixOperationalEdgeNativeCollector(
        client,
        release_revision="a" * 40,
        receipt_key_id=client.config.receipt_key_id,
    )
    bindings = collector.collect(
        slot="bitrix_edge",
        payload={
            "read_probe": {
                "selected_edge_id": "operational-edge:bitrix",
                "read_operation_id": "bitrix.crm.status_list",
                "read_arguments": {"entity_id": "STATUS"},
                "initial_read_probe_id": "canary:bitrix:status-list:initial",
                "readback_probe_id": "canary:bitrix:status-list:readback",
                "normalized_equality_excluded_fields": ["generated_at_utc"],
                "stable_normalized_equality": True,
            }
        },
    )

    assert (
        tuple(item.kind for item in bindings)
        == producers.SLOT_NATIVE_BINDING_KINDS["bitrix_edge"]
    )
    assert client.calls == [
        (
            "bitrix.crm.status_list",
            {"entity_id": "STATUS"},
            "canary:bitrix:status-list:initial",
            None,
        ),
        (
            "bitrix.crm.status_list",
            {"entity_id": "STATUS"},
            "canary:bitrix:status-list:readback",
            None,
        ),
    ]


@pytest.mark.parametrize(
    "tamper",
    ("envelope_digest", "payload", "signature"),
)
def test_bitrix_collector_rejects_tampered_verified_envelope(
    tamper: str,
) -> None:
    client = _BitrixClient(tamper=tamper)
    collector = producers.BitrixOperationalEdgeNativeCollector(
        client,
        release_revision="a" * 40,
        receipt_key_id=client.config.receipt_key_id,
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="bitrix_native_evidence_invalid",
    ):
        collector.collect(
            slot="bitrix_edge",
            payload={
                "read_probe": {
                    "selected_edge_id": "operational-edge:bitrix",
                    "read_operation_id": "bitrix.crm.status_list",
                    "read_arguments": {"entity_id": "STATUS"},
                    "initial_read_probe_id": "canary:bitrix:initial",
                    "readback_probe_id": "canary:bitrix:readback",
                    "normalized_equality_excluded_fields": ["generated_at_utc"],
                    "stable_normalized_equality": True,
                }
            },
        )


def test_bitrix_collector_rejects_unstable_normalized_readback() -> None:
    client = _BitrixClient(unstable_readback=True)
    collector = producers.BitrixOperationalEdgeNativeCollector(
        client,
        release_revision="a" * 40,
        receipt_key_id=client.config.receipt_key_id,
    )
    with pytest.raises(
        producers.CapabilityProducerError,
        match="bitrix_native_evidence_invalid",
    ):
        collector.collect(
            slot="bitrix_edge",
            payload={
                "read_probe": {
                    "selected_edge_id": "operational-edge:bitrix",
                    "read_operation_id": "bitrix.crm.status_list",
                    "read_arguments": {"entity_id": "STATUS"},
                    "initial_read_probe_id": "canary:bitrix:initial",
                    "readback_probe_id": "canary:bitrix:readback",
                    "normalized_equality_excluded_fields": ["generated_at_utc"],
                    "stable_normalized_equality": True,
                }
            },
        )


def test_bitrix_collector_rejects_malformed_receipt_key_id() -> None:
    client = _BitrixClient()
    with pytest.raises(
        producers.CapabilityProducerError,
        match="bitrix_native_collector_config_invalid",
    ):
        producers.BitrixOperationalEdgeNativeCollector(
            client,
            release_revision="a" * 40,
            receipt_key_id="not-a-digest",
        )


class _WriterHandoffCollector:
    def collect(self, *, slot: str, payload: Mapping[str, Any]):
        assert slot == "bitrix_writer"
        assert payload["handoff_id"] == "handoff:bitrix"
        return (
            producers.NativeEvidenceBinding(
                kind="canonical_writer_handoff_events",
                source_identity_sha256="1" * 64,
                artifact_sha256="2" * 64,
                verification_receipt_sha256="3" * 64,
            ),
        )


def test_bitrix_writer_alone_collects_signed_predispatch_denial() -> None:
    client = _BitrixClient()
    collector = producers.BitrixWriterNativeCollector(
        client,
        release_revision="a" * 40,
        receipt_key_id=client.config.receipt_key_id,
        canonical_writer_collector=_WriterHandoffCollector(),
    )
    bindings = collector.collect(
        slot="bitrix_writer",
        payload={
            "handoff_id": "handoff:bitrix",
            "mutation_probe": {
                "selected_edge_id": "operational-edge:bitrix",
                "mutation_operation_id": "bitrix.crm.lead_add",
                "mutation_arguments": dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
                "mutation_probe_id": "canary:bitrix:denial",
            },
        },
    )

    assert tuple(item.kind for item in bindings) == (
        "canonical_writer_handoff_events",
        "operational_edge_bitrix_mutation_predispatch_denial",
    )
    assert client.calls == [
        (
            "bitrix.crm.lead_add",
            dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
            "canary:bitrix:denial",
            None,
        )
    ]


def _command(command_id: str, body: bytes) -> Mapping[str, Any]:
    return {
        "command_id": command_id,
        "command_b64": base64.b64encode(body).decode(),
        "command_sha256": _sha(body),
        "max_uses": 1,
    }


def _api_admission_bundle(tmp_path: Path) -> dict[str, Any]:
    foundation, context = _foundation(tmp_path / "foundation")
    fixture = {
        **_fixture(foundation, run_id="run-one", start_ms=NOW_MS - 1_000),
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
    }
    fixture_sha256 = _sha(_canonical(fixture))
    session_id = f"capability_{fixture['run_id']}"
    capability_epoch_sha256 = "d" * 64
    catalog = producers.build_probe_catalog(
        release_sha=fixture["release_sha"],
        capability_plan_sha256=fixture["capability_plan_sha256"],
        full_canary_plan_sha256=fixture["full_canary_plan_sha256"],
        fixture_sha256=fixture_sha256,
        run_id=fixture["run_id"],
        session_id=session_id,
        capability_epoch_sha256=capability_epoch_sha256,
        case_ids={
            name: f"case:{name}"
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
            "first_path_probe_id": "probe:first",
            "alternate_path_probe_id": "probe:alternate",
            "worker_restart_checkpoint_step_id": "step:restart",
        },
        commands={
            "allowed": [_command("allowed", b"printf allowed")],
            "denied": [
                {
                    "kind": kind,
                    "command": _command(
                        f"denied:{index}", f"deny-{index}".encode()
                    ),
                }
                for index, kind in enumerate(producers.DENIAL_KINDS)
            ],
        },
        database={
            "row_key": "row:one",
            "idempotency_key": "database:one",
            "read_probe_id": "database:read",
            "write_probe_id": "database:write",
            "lost_response_probe_id": "database:lost-response",
        },
        bitrix={
            "handoff_id": "handoff:bitrix",
            "selected_edge_id": "operational-edge:bitrix",
            "read_operation_id": "bitrix.crm.status_list",
            "read_arguments": {"entity_id": "STATUS"},
            "initial_read_probe_id": "canary:bitrix:initial",
            "readback_probe_id": "canary:bitrix:readback",
            "normalized_equality_excluded_fields": ["generated_at_utc"],
            "mutation_operation_id": "bitrix.crm.lead_add",
            "mutation_arguments": dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
            "mutation_probe_id": "canary:bitrix:denial",
        },
        discord={
            "public_target": {
                "target_type": "public_channel",
                "guild_id": "1282725267068157972",
                "channel_id": "1504852355588423801",
            },
            "public_idempotency_key": "discord:public",
            "private_target_kind": "dm",
            "private_probe_id": "discord:private",
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
    request = producers._api_admission_request(
        session_id=session_id,
        capability_epoch_sha256=capability_epoch_sha256,
        nonce="e" * 32,
    )
    challenge = producers.build_api_admission_owner_challenge(
        request=request,
        catalog=catalog,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        requested_at_unix_ms=NOW_MS,
        owner_response_deadline_unix_ms=NOW_MS + 60_000,
    )
    command_sha256s = sorted(
        item["command_sha256"] for item in catalog["commands"]["allowed"]
    )
    approval_payload = {
        "schema": e2e.PLAN_APPROVAL_SCHEMA,
        "run_id": fixture["run_id"],
        "release_sha": fixture["release_sha"],
        "fixture_sha256": fixture_sha256,
        "observed_at_unix_ms": NOW_MS,
        "approval_id": "approval-one",
        "owner_id": producers.PRODUCTION_OWNER_ID,
        "session_id": session_id,
        "capability_epoch_sha256": capability_epoch_sha256,
        "command_sha256s": command_sha256s,
        "ttl_seconds": 300,
        "max_uses": len(command_sha256s),
    }
    approval_unsigned = {
        "schema": producers.SIGNED_RECEIPT_SCHEMA,
        "authority_role": "owner",
        "key_id": fixture["authority_keys"]["owner"]["key_id"],
        "signature_algorithm": "sshsig-ed25519-sha512",
        "payload": approval_payload,
    }
    pregrant = {
        **approval_unsigned,
        "signature": _sshsig(
            context["owner_private"],
            _canonical(approval_unsigned),
            namespace=producers.OWNER_SSHSIG_NAMESPACE,
        ),
    }
    authority_unsigned = {
        "schema": producers.API_ADMISSION_OWNER_AUTHORITY_SCHEMA,
        "challenge_sha256": request["challenge_sha256"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "run_id": fixture["run_id"],
        "session_id": session_id,
        "capability_epoch_sha256": capability_epoch_sha256,
        "issued_at_unix_ms": NOW_MS,
        "valid_until_unix_ms": NOW_MS + 300_000,
        "catalog": catalog,
        "workspace_owner_receipt": pregrant,
    }
    authority = {
        **authority_unsigned,
        "owner_signature": _sshsig(
            context["owner_private"],
            _canonical(authority_unsigned),
            namespace=producers.API_ADMISSION_OWNER_SSHSIG_NAMESPACE,
        ),
    }
    installed = producers.InstalledProducerFoundation(
        value=foundation,
        pinned_owner_public_key_ed25519_hex=context["owner_public"],
        pinned_owner_public_key_source_sha256=context["source_sha256"],
    )
    return {
        "foundation": foundation,
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "challenge": challenge,
        "authority": authority,
        "catalog": catalog,
        "pregrant": pregrant,
        "installed": installed,
    }


def _install_rootless_recovery_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: producers.InstalledProducerFoundation,
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    real_require_directory = producers._require_directory
    real_stable_read = producers._stable_read
    real_read_expected = producers._read_expected_publication
    real_publish = producers._publish_no_replace

    def owner(value: int) -> int:
        return uid if value == 0 else value

    def group(value: int) -> int:
        return gid if value == 0 else value

    def require_directory(path, *, uid, gid, mode):
        return real_require_directory(
            path,
            uid=owner(uid),
            gid=group(gid),
            mode=mode,
        )

    def stable_read(path, *, maximum, uid, gid, mode):
        return real_stable_read(
            path,
            maximum=maximum,
            uid=owner(uid),
            gid=group(gid),
            mode=mode,
        )

    def read_expected(
        path,
        payload,
        *,
        uid,
        gid,
        mode,
        parent_uid,
        parent_gid,
        parent_mode,
    ):
        return real_read_expected(
            path,
            payload,
            uid=owner(uid),
            gid=group(gid),
            mode=mode,
            parent_uid=owner(parent_uid),
            parent_gid=group(parent_gid),
            parent_mode=parent_mode,
        )

    def publish(
        path,
        payload,
        *,
        uid,
        gid,
        mode=0o400,
        parent_mode=0o700,
        parent_uid=None,
        parent_gid=None,
    ):
        return real_publish(
            path,
            payload,
            uid=owner(uid),
            gid=group(gid),
            mode=mode,
            parent_mode=parent_mode,
            parent_uid=(None if parent_uid is None else owner(parent_uid)),
            parent_gid=(None if parent_gid is None else group(parent_gid)),
        )

    monkeypatch.setattr(producers, "_require_directory", require_directory)
    monkeypatch.setattr(producers, "_stable_read", stable_read)
    monkeypatch.setattr(producers, "_read_expected_publication", read_expected)
    monkeypatch.setattr(producers, "_publish_no_replace", publish)
    monkeypatch.setattr(
        producers,
        "load_installed_producer_foundation",
        lambda **_kwargs: installed,
    )
    monkeypatch.setattr(
        e2e,
        "_validate_fixture",
        lambda value, fixture_sha256: (
            dict(value)
            if _sha(_canonical(value)) == fixture_sha256
            else pytest.fail("fixture digest drifted")
        ),
    )


def test_probe_catalog_contains_real_bitrix_operational_edge_facts() -> None:
    catalog = producers.build_probe_catalog(
        release_sha="a" * 40,
        capability_plan_sha256="b" * 64,
        full_canary_plan_sha256="c" * 64,
        fixture_sha256="d" * 64,
        run_id="run-one",
        session_id="session-one",
        capability_epoch_sha256="e" * 64,
        case_ids={
            name: f"case:{name}"
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
            "first_path_probe_id": "probe:first",
            "alternate_path_probe_id": "probe:alternate",
            "worker_restart_checkpoint_step_id": "step:restart",
        },
        commands={
            "allowed": [_command("allowed", b"printf allowed")],
            "denied": [
                {
                    "kind": kind,
                    "command": _command(f"denied:{index}", f"deny-{index}".encode()),
                }
                for index, kind in enumerate(producers.DENIAL_KINDS)
            ],
        },
        database={
            "row_key": "row:one",
            "idempotency_key": "database:one",
            "read_probe_id": "database:read",
            "write_probe_id": "database:write",
            "lost_response_probe_id": "database:lost-response",
        },
        bitrix={
            "handoff_id": "handoff:bitrix",
            "selected_edge_id": "operational-edge:bitrix",
            "read_operation_id": "bitrix.crm.status_list",
            "read_arguments": {"entity_id": "STATUS"},
            "initial_read_probe_id": "canary:bitrix:status-list:initial",
            "readback_probe_id": "canary:bitrix:status-list:readback",
            "normalized_equality_excluded_fields": ["generated_at_utc"],
            "mutation_operation_id": "bitrix.crm.lead_add",
            "mutation_arguments": dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
            "mutation_probe_id": "canary:bitrix:denial",
        },
        discord={
            "public_target": {
                "target_type": "public_channel",
                "guild_id": "1282725267068157972",
                "channel_id": "1504852355588423801",
            },
            "public_idempotency_key": "discord:public",
            "private_target_kind": "dm",
            "private_probe_id": "discord:private",
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
    assert catalog["bitrix"] == {
        "handoff_id": "handoff:bitrix",
        "selected_edge_id": "operational-edge:bitrix",
        "read_operation_id": "bitrix.crm.status_list",
        "read_arguments": {"entity_id": "STATUS"},
        "initial_read_probe_id": "canary:bitrix:status-list:initial",
        "readback_probe_id": "canary:bitrix:status-list:readback",
        "normalized_equality_excluded_fields": ["generated_at_utc"],
        "mutation_operation_id": "bitrix.crm.lead_add",
        "mutation_arguments": dict(producers.BITRIX_CANARY_MUTATION_ARGUMENTS),
        "mutation_probe_id": "canary:bitrix:denial",
    }
    assert "issue_iid" not in _canonical(catalog).decode()
    assert "task.read" not in _canonical(catalog).decode()

    unsupported_thread = json.loads(json.dumps(catalog))
    unsupported_thread["discord"]["public_target"]["target_type"] = "public_thread"
    with pytest.raises(RuntimeError, match="probe_catalog_invalid"):
        producers.validate_probe_catalog(unsupported_thread)


def _stage_api_admission_for_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_after: str | None,
) -> dict[str, Any]:
    bundle = _api_admission_bundle(tmp_path)
    installed = bundle["installed"]
    _install_rootless_recovery_boundaries(monkeypatch, installed=installed)
    receipt_root = Path(bundle["foundation"]["receipt_contract"]["base_root"])
    receipt_root.mkdir(parents=True, mode=0o700)
    os.chown(receipt_root, os.getuid(), os.getgid())
    receipt_root.chmod(0o700)
    run_root = receipt_root / bundle["fixture"]["run_id"]
    run_root.mkdir(mode=0o770)
    os.chown(run_root, os.getuid(), os.getgid())
    run_root.chmod(0o770)
    inputs_root = tmp_path / "inputs"
    inputs_root.mkdir(mode=0o700)
    os.chown(inputs_root, os.getuid(), os.getgid())
    inputs_root.chmod(0o700)
    catalog_path = inputs_root / "probe-catalog.json"
    owner_grant_path = inputs_root / "owner-grant.json"
    readiness_path = tmp_path / "runtime/producer-activation.json"
    reviewed_fixture_root = tmp_path / "reviewed"
    reviewed_fixture_root.mkdir(mode=0o700)
    os.chown(reviewed_fixture_root, os.getuid(), os.getgid())
    reviewed_fixture_root.chmod(0o700)
    reviewed_fixture_path = reviewed_fixture_root / "reviewed-live-fixture.json"
    live_fixture_root = tmp_path / "live"
    live_fixture_root.mkdir(mode=0o700)
    os.chown(live_fixture_root, os.getuid(), os.getgid())
    live_fixture_root.chmod(0o700)
    live_run_directory = live_fixture_root / bundle["fixture"]["run_id"]
    live_run_directory.mkdir(mode=0o700)
    os.chown(live_run_directory, os.getuid(), os.getgid())
    live_run_directory.chmod(0o700)
    live_fixture_path = live_run_directory / "fixture.json"
    live_fixture_path.write_bytes(_canonical(bundle["fixture"]))
    os.chown(live_fixture_path, os.getuid(), os.getgid())
    live_fixture_path.chmod(0o400)
    producers.publish_api_admission_owner_challenge(
        bundle["challenge"],
        fixture=bundle["fixture"],
        fixture_sha256=bundle["fixture_sha256"],
        installed_foundation=installed,
    )
    selected = {
        "authority": run_root / "api-admission-owner-authority.json",
        "intent": run_root / "api-admission-install-intent.json",
        "catalog": catalog_path,
        "grant": owner_grant_path,
        "owner_receipt": run_root / producers.SLOT_FILENAME["workspace_owner"],
    }
    real_publish = producers._publish_or_identical
    if fail_after == "pre_authority":
        pass
    elif fail_after is not None:
        target = selected[fail_after]

        def publish_then_fail(path, *args, **kwargs):
            result = real_publish(path, *args, **kwargs)
            if path == target:
                raise RuntimeError(f"injected-after-{fail_after}")
            return result

        monkeypatch.setattr(producers, "_publish_or_identical", publish_then_fail)
        with pytest.raises(RuntimeError, match=f"injected-after-{fail_after}"):
            producers.provision_api_admission_owner_authority(
                bundle["authority"],
                challenge=bundle["challenge"]["request"],
                fixture=bundle["fixture"],
                fixture_sha256=bundle["fixture_sha256"],
                installed_foundation=installed,
                writer_gid=os.getgid(),
                now_ms=NOW_MS,
                catalog_path=catalog_path,
                owner_grant_path=owner_grant_path,
            )
        monkeypatch.setattr(producers, "_publish_or_identical", real_publish)
    else:
        bundle["publication"] = producers.provision_api_admission_owner_authority(
            bundle["authority"],
            challenge=bundle["challenge"]["request"],
            fixture=bundle["fixture"],
            fixture_sha256=bundle["fixture_sha256"],
            installed_foundation=installed,
            writer_gid=os.getgid(),
            now_ms=NOW_MS,
            catalog_path=catalog_path,
            owner_grant_path=owner_grant_path,
        )
    return {
        **bundle,
        "receipt_root": receipt_root,
        "run_root": run_root,
        "catalog_path": catalog_path,
        "owner_grant_path": owner_grant_path,
        "readiness_path": readiness_path,
        "reviewed_fixture_path": reviewed_fixture_path,
        "live_fixture_root": live_fixture_root,
        "live_fixture_path": live_fixture_path,
        "selected": selected,
    }


def test_post_publication_pre_authority_recovery_uses_durable_live_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = _stage_api_admission_for_recovery(
        monkeypatch,
        tmp_path,
        fail_after="pre_authority",
    )
    source = staged["reviewed_fixture_path"]
    source.write_bytes(_canonical(staged["fixture"]))
    os.chown(source, os.getuid(), os.getgid())
    source.chmod(0o400)

    recovered = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_000,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=source,
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )

    assert recovered["outcome"] == (
        "reconciled_published_run_without_admission"
    )
    assert recovered["run_id"] == staged["fixture"]["run_id"]
    assert recovered["fixture_sha256"] == staged["fixture_sha256"]
    assert recovered["admission_retirement"] is None
    assert recovered["fleet_retirement"] is None
    assert not source.exists()
    assert staged["live_fixture_path"].is_file()
    assert not staged["selected"]["authority"].exists()
    aggregate = staged["run_root"] / "active-api-admission-retirement.json"
    assert aggregate.is_file()
    repeated = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_001,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=source,
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )
    assert repeated == recovered


def test_recovery_and_live_driver_share_one_durable_fixture_root() -> None:
    from gateway import canonical_capability_canary_live_driver as live_driver

    assert producers.DEFAULT_LIVE_FIXTURE_ROOT == live_driver.DEFAULT_LIVE_ROOT


def test_post_publication_before_inbox_prepare_is_bound_not_no_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = _stage_api_admission_for_recovery(
        monkeypatch,
        tmp_path,
        fail_after="pre_authority",
    )
    challenge = staged["run_root"] / "api-admission-owner-challenge.json"
    challenge.unlink()
    staged["run_root"].rmdir()

    recovered = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_000,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=staged["reviewed_fixture_path"],
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )

    assert recovered["outcome"] == (
        "reconciled_published_run_without_admission"
    )
    assert recovered["run_id"] == staged["fixture"]["run_id"]
    assert recovered["fixture_sha256"] == staged["fixture_sha256"]
    assert staged["live_fixture_path"].is_file()
    assert not staged["run_root"].exists()


@pytest.mark.parametrize(
    ("fail_after", "state", "presence"),
    (
        ("authority", "authority_only_aborted", (False, False, False)),
        ("intent", "partial_install_retired", (False, False, False)),
        ("catalog", "partial_install_retired", (True, False, False)),
        ("grant", "partial_install_retired", (True, True, False)),
        ("owner_receipt", "partial_install_retired", (True, True, True)),
    ),
)
def test_api_admission_partial_publication_recovers_exactly_and_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_after: str,
    state: str,
    presence: tuple[bool, bool, bool],
) -> None:
    staged = _stage_api_admission_for_recovery(
        monkeypatch,
        tmp_path,
        fail_after=fail_after,
    )
    recovered = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_000,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=staged["reviewed_fixture_path"],
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )

    assert not staged["reviewed_fixture_path"].exists()
    assert staged["live_fixture_path"].is_file()
    assert recovered["outcome"] == "retired_partial_install"
    abort = recovered["admission_retirement"]
    assert abort["schema"] == producers.API_ADMISSION_INSTALL_ABORT_SCHEMA
    assert abort["state"] == state
    assert (
        abort["catalog_present_at_recovery"],
        abort["owner_grant_present_at_recovery"],
        abort["owner_receipt_present_at_recovery"],
    ) == presence
    assert not staged["catalog_path"].exists()
    assert not staged["owner_grant_path"].exists()
    assert not staged["selected"]["owner_receipt"].exists()
    assert (staged["run_root"] / "api-admission-install-abort.json").is_file()
    assert (
        staged["run_root"] / "active-api-admission-retirement.json"
    ).is_file()
    repeated = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_001,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=staged["reviewed_fixture_path"],
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )
    assert repeated == recovered
    with pytest.raises(
        producers.CapabilityProducerError,
        match="api_admission_owner_publication_aborted",
    ):
        producers.provision_api_admission_owner_authority(
            staged["authority"],
            challenge=staged["challenge"]["request"],
            fixture=staged["fixture"],
            fixture_sha256=staged["fixture_sha256"],
            installed_foundation=staged["installed"],
            writer_gid=os.getgid(),
            now_ms=NOW_MS + 1,
            catalog_path=staged["catalog_path"],
            owner_grant_path=staged["owner_grant_path"],
        )


def test_api_admission_completed_before_activation_retires_without_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staged = _stage_api_admission_for_recovery(
        monkeypatch,
        tmp_path,
        fail_after=None,
    )
    recovered = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=NOW_MS + 500_000,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=staged["reviewed_fixture_path"],
        live_fixture_root=staged["live_fixture_root"],
        receipt_root=staged["receipt_root"],
        readiness_path=staged["readiness_path"],
        catalog_path=staged["catalog_path"],
        owner_grant_path=staged["owner_grant_path"],
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )

    assert recovered["outcome"] == "retired_partial_install"
    abort = recovered["admission_retirement"]
    assert abort["state"] == "completed_install_retired_before_activation"
    assert abort["install_publication_sha256"] == staged["publication"][
        "receipt_sha256"
    ]
    assert abort["producer_activation_absent"] is True


@pytest.mark.parametrize(
    "target_name",
    ("catalog", "grant", "owner_receipt"),
)
@pytest.mark.parametrize("hazard", ("bytes", "symlink", "hardlink"))
def test_api_admission_partial_recovery_rejects_fixed_input_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hazard: str,
    target_name: str,
) -> None:
    staged = _stage_api_admission_for_recovery(
        monkeypatch,
        tmp_path,
        fail_after="owner_receipt",
    )
    targets = {
        "catalog": staged["catalog_path"],
        "grant": staged["owner_grant_path"],
        "owner_receipt": staged["selected"]["owner_receipt"],
    }
    modes = {"catalog": 0o440, "grant": 0o400, "owner_receipt": 0o400}
    target = targets[target_name]
    if hazard == "bytes":
        target.unlink()
        target.write_bytes(b'{"drifted":true}')
        target.chmod(modes[target_name])
    elif hazard == "symlink":
        target.unlink()
        decoy = tmp_path / f"{target_name}-decoy"
        decoy.write_bytes(b'{"decoy":true}')
        target.symlink_to(decoy)
    else:
        os.link(target, tmp_path / f"{target_name}-unexpected-hardlink")
        monkeypatch.setattr(producers.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        producers.CapabilityProducerError,
        match=(
            "publication_collision_diverged"
            if hazard == "bytes"
            else "artifact_identity_invalid"
        ),
    ):
        producers.recover_and_retire_active_api_admission(
            retired_at_unix_ms=NOW_MS + 500_000,
            foundation_path=tmp_path / "foundation.json",
            reviewed_fixture_path=staged["reviewed_fixture_path"],
            live_fixture_root=staged["live_fixture_root"],
            receipt_root=staged["receipt_root"],
            readiness_path=staged["readiness_path"],
            catalog_path=staged["catalog_path"],
            owner_grant_path=staged["owner_grant_path"],
            owner_public_key_hex_path=tmp_path / "owner.hex",
            owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
        )
    assert os.path.lexists(target)
    assert all(os.path.lexists(path) for path in targets.values())
    assert not (staged["run_root"] / "api-admission-install-abort.json").exists()


def test_recover_active_api_admission_records_exact_no_active_observation(
    tmp_path: Path,
) -> None:
    result = producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=1_700_000_000_000,
        foundation_path=tmp_path / "foundation.json",
        reviewed_fixture_path=tmp_path / "reviewed-live-fixture.json",
        live_fixture_root=tmp_path / "live",
        receipt_root=tmp_path / "receipts",
        readiness_path=tmp_path / "producer-activation.json",
        catalog_path=tmp_path / "probe-catalog.json",
        owner_grant_path=tmp_path / "owner-grant.json",
        owner_public_key_hex_path=tmp_path / "owner.hex",
        owner_public_key_source_sha256_path=tmp_path / "owner-source.sha256",
    )

    assert result["schema"] == producers.ACTIVE_API_ADMISSION_RETIREMENT_SCHEMA
    assert result["outcome"] == "confirmed_no_active_run"
    assert result["run_id"] is None
    assert result["catalog_absent"] is True
    assert result["owner_grant_absent"] is True
    assert result["producer_activation_absent"] is True
    assert producers.validate_active_api_admission_retirement(result) == result


def test_active_api_admission_retirement_rejects_self_digest_drift(
    tmp_path: Path,
) -> None:
    result = dict(producers.recover_and_retire_active_api_admission(
        retired_at_unix_ms=1_700_000_000_000,
        reviewed_fixture_path=tmp_path / "reviewed-live-fixture.json",
        live_fixture_root=tmp_path / "live",
        readiness_path=tmp_path / "producer-activation.json",
        catalog_path=tmp_path / "probe-catalog.json",
        owner_grant_path=tmp_path / "owner-grant.json",
    ))
    result["catalog_absent"] = False

    with pytest.raises(
        producers.CapabilityProducerError,
        match="active_api_admission_retirement_invalid",
    ):
        producers.validate_active_api_admission_retirement(result)
