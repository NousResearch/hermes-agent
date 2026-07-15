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
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(
        private_key.sign(signed)
    )
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
        role: role_private[role].public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        for role in producers.ENDPOINT_ROLES
    }
    owner_public = owner_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
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
    unsigned = {
        "schema": producers.PRODUCER_FOUNDATION_SCHEMA,
        "release_sha": "a" * 40,
        "capability_plan_sha256": "b" * 64,
        "full_canary_plan_sha256": "c" * 64,
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
                    "/etc/muncho/keys/"
                    "operational-edge-bitrix-receipt-private.pem"
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
            "connector_socket_path": (
                "/run/muncho-discord-connector/connector.sock"
            ),
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
        "producer_foundation_sha256": producers.producer_foundation_sha256(
            foundation
        ),
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
                Ed25519PrivateKey.generate()
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
    assert producers.validate_fleet_readiness(
        activation,
        now_ms=NOW_MS,
        expected_foundation_sha256=activation["foundation_sha256"],
    ) == activation


def test_expired_activation_can_be_exactly_retired_after_service_stop(
    tmp_path: Path,
) -> None:
    _foundation_value, _context, _fixture_value, activation = _activation(
        tmp_path
    )
    path = tmp_path / "activation.json"
    parent = tmp_path.lstat()
    producers.publish_fleet_readiness(
        activation,
        path=path,
        uid=parent.st_uid,
        gid=parent.st_gid,
        now_ms=NOW_MS,
    )
    retirement = producers.retire_fleet_readiness(
        expected_readiness_sha256=activation["readiness_sha256"],
        path=path,
        uid=parent.st_uid,
        gid=parent.st_gid,
        expected_foundation_sha256=activation["foundation_sha256"],
        expected_capability_plan_sha256=activation[
            "capability_plan_sha256"
        ],
        expected_full_canary_plan_sha256=activation[
            "full_canary_plan_sha256"
        ],
        retired_at_unix_ms=activation["valid_until_unix_ms"] + 1,
    )
    assert retirement["absence_verified"] is True
    assert not os.path.lexists(path)


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
    config = producers.ProducerConfig.from_mapping(
        {
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
        }
    )
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
    assert producers.verify_role_receipt(
        signed,
        role=role,
        slot="bitrix_edge",
        public_key=private_key.public_key(),
        producer_readiness_sha256=activation["readiness_sha256"],
    ) == request["payload"]


def test_production_pump_uses_action_produce_and_verifies_immutable_receipt(
    tmp_path: Path,
) -> None:
    foundation, context, fixture, activation = _activation(tmp_path)
    role = "business_edge"
    endpoint = foundation["endpoints"][role]
    receipt_contract = foundation["receipt_contract"]
    config = producers.ProducerConfig.from_mapping(
        {
            "schema": producers.PRODUCER_CONFIG_SCHEMA,
            "role": role,
            "foundation_sha256": producers.producer_foundation_sha256(
                foundation
            ),
            "release_sha": foundation["release_sha"],
            "capability_plan_sha256": foundation[
                "capability_plan_sha256"
            ],
            "full_canary_plan_sha256": foundation[
                "full_canary_plan_sha256"
            ],
            "service_unit": endpoint["service_unit"],
            "service_identity_sha256": endpoint[
                "service_identity_sha256"
            ],
            "service_uid": os.getuid(),
            "service_gid": os.getgid(),
            "root_client_uid": 0,
            "socket_path": endpoint["socket_path"],
            "receipt_base_root": receipt_contract["base_root"],
            "receipt_directory_uid": receipt_contract[
                "run_directory_uid"
            ],
            "receipt_directory_gid": receipt_contract[
                "run_directory_gid"
            ],
            "receipt_directory_mode": receipt_contract[
                "run_directory_mode"
            ],
            "private_key_path": endpoint["private_key_path"],
            "public_key_path": endpoint["public_key_path"],
            "allowed_slots": endpoint["allowed_slots"],
        }
    )
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
    assert calls[0]["request"]["producer_readiness_sha256"] == activation[
        "readiness_sha256"
    ]
    receipt_path = (
        Path(activation["run_receipt_root"])
        / producers.SLOT_FILENAME["bitrix_edge"]
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


def test_atomic_publication_concurrent_same_payload_has_one_immutable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "receipt.json"
    parent = tmp_path.stat()

    publishers = 8
    barrier = threading.Barrier(publishers)
    real_link = os.link

    def synchronized_link(*args: Any, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", synchronized_link)

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
    real_link = os.link

    def fail_link(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected-before-publication")

    monkeypatch.setattr(os, "link", fail_link)
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

    monkeypatch.setattr(os, "link", real_link)
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
    real_link = os.link

    def synchronized_link(*args: Any, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", synchronized_link)

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
        self.calls.append(
            (operation_id, dict(_arguments), idempotency_key, capability)
        )
        denial = operation_id == "bitrix.crm.lead_add"
        read_ordinal = sum(
            1
            for called_operation, _args, _key, _capability in self.calls
            if called_operation == "bitrix.crm.status_list"
        )
        status_value = "CHANGED" if self.unstable_readback and read_ordinal == 2 else "NEW"
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
                _canonical(
                    {
                        "status": "OK",
                        "generated_at_utc": f"2026-07-15T00:00:0{read_ordinal}Z",
                        "items": [{"ENTITY_ID": "STATUS", "STATUS_ID": status_value}],
                    }
                )
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

    assert tuple(item.kind for item in bindings) == producers.SLOT_NATIVE_BINDING_KINDS[
        "bitrix_edge"
    ]
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
                    "normalized_equality_excluded_fields": [
                        "generated_at_utc"
                    ],
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
                    "normalized_equality_excluded_fields": [
                        "generated_at_utc"
                    ],
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
                "mutation_arguments": dict(
                    producers.BITRIX_CANARY_MUTATION_ARGUMENTS
                ),
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
            "mutation_arguments": dict(
                producers.BITRIX_CANARY_MUTATION_ARGUMENTS
            ),
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
