from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pytest


ISOLATED_RUNTIME_ENV = "MUNCHO_OWNER_GATE_ISOLATED_TEST_RUNTIME"
if os.environ.get(ISOLATED_RUNTIME_ENV) != "1":
    pytest.skip(
        "runs through test_passkey_v2_isolated_runtime.py under the exact "
        "owner-gate WebAuthn dependency boundary",
        allow_module_level=True,
    )

import cbor2
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import passkey_v2_sqlite as database
from scripts.canary import passkey_v2_webauthn as webauthn
from scripts.canary.passkey_v2_signer import ReceiptSigner


NOW = 1_785_000_000
OWNER = "1279454038731264061"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _noncanonical_pad_bits(value: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    remainder = len(value) % 4
    if remainder not in {2, 3}:
        raise AssertionError("test vector has no unused base64url pad bits")
    index = alphabet.index(value[-1])
    alternate = index ^ 1
    assert alphabet[alternate] != value[-1]
    changed = value[:-1] + alphabet[alternate]
    assert base64.urlsafe_b64decode(changed + "=" * (-len(changed) % 4)) == (
        base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    )
    return changed


def _envelope(*, request_id: str = "R" * 32) -> Mapping[str, Any]:
    return protocol.build_action_envelope(
        request_id=request_id,
        requester_discord_user_id=OWNER,
        required_approver_discord_user_id=OWNER,
        scope="runtime_config_mutation",
        case_id="case:canary-storage-growth-p0",
        target_system="gce:muncho-canary-v2-01/disk",
        action_summary="Resize the exact canary boot disk from 40 GB to 80 GB.",
        risk="The bounded disk resize could require one conditional reboot.",
        rollback="Stop before mutation on drift; preserve the prior stopped release.",
        action_payload={
            "operation": "resize_boot_disk",
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "instance": "muncho-canary-v2-01",
            "disk_id": "4195397669213846393",
            "source_size_gb": 40,
            "target_size_gb": 80,
            "remaining_actions": ["resize", "conditional_stop_start", "postflight"],
        },
        executor_release_sha="a" * 40,
        executor_plan_sha256="b" * 64,
        transaction_id="c" * 64,
        stage="resize",
        webauthn_rp_id=protocol.PRODUCTION_RP_ID,
        webauthn_origin=protocol.PRODUCTION_ORIGIN,
        authority_release_sha="d" * 40,
        authority_manifest_sha256="e" * 64,
        authority_host_receipt_sha256="f" * 64,
        source_preflight_sha256="1" * 64,
        live_projection_sha256="2" * 64,
        external_iam_receipt_sha256="3" * 64,
        prior_authoritative_receipt_sha256="4" * 64,
        prior_event_head_sha256="5" * 64,
        issued_at_unix=NOW,
        approval_ttl_seconds=300,
    )


def _challenge(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    return protocol.build_challenge_record(
        envelope=envelope,
        challenge_id="C" * 32,
        challenge_b64url=_b64(b"challenge" * 4),
        rp_id=protocol.PRODUCTION_RP_ID,
        origin=protocol.PRODUCTION_ORIGIN,
        created_at_unix=NOW + 1,
    )


def _credential_and_assertion(
    envelope: Mapping[str, Any],
    challenge: Mapping[str, Any],
    *,
    flags: int = 0x1D,
    client_origin: str = protocol.PRODUCTION_ORIGIN,
    client_challenge: str | None = None,
    rp_id: str = protocol.PRODUCTION_RP_ID,
    sign_count: int = 0,
    client_extra: Mapping[str, Any] | None = None,
    user_handle: bytes | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    public_key = cbor2.dumps({
        1: 2,
        3: -7,
        -1: 1,
        -2: numbers.x.to_bytes(32, "big"),
        -3: numbers.y.to_bytes(32, "big"),
    })
    credential_id = b"credential-id-for-emil-passkey"
    credential = webauthn.build_migrated_credential(
        owner_discord_user_id=OWNER,
        credential_id=credential_id,
        public_key_cose=public_key,
        rp_id=protocol.PRODUCTION_RP_ID,
        origin=protocol.PRODUCTION_ORIGIN,
        imported_at_unix=NOW - 100,
        migration_receipt_sha256="6" * 64,
        initial_sign_count=0,
        initial_credential_backed_up=True,
        expected_user_handle=OWNER.encode("utf-8"),
    )
    client_object = {
        "type": "webauthn.get",
        "challenge": client_challenge or challenge["challenge_b64url"],
        "origin": client_origin,
        "crossOrigin": False,
    }
    client_object.update(dict(client_extra or {}))
    client_data = json.dumps(
        client_object,
        separators=(",", ":"),
    ).encode("utf-8")
    authenticator_data = (
        hashlib.sha256(rp_id.encode("ascii")).digest()
        + bytes([flags])
        + sign_count.to_bytes(4, "big")
    )
    signed = authenticator_data + hashlib.sha256(client_data).digest()
    signature = private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
    assertion = {
        "schema": webauthn.ASSERTION_SCHEMA,
        "credential": {
            "id": _b64(credential_id),
            "rawId": _b64(credential_id),
            "response": {
                "clientDataJSON": _b64(client_data),
                "authenticatorData": _b64(authenticator_data),
                "signature": _b64(signature),
                "userHandle": None if user_handle is None else _b64(user_handle),
            },
            "type": "public-key",
            "authenticatorAttachment": "platform",
            "clientExtensionResults": {},
        },
    }
    return credential, assertion


def _runtime() -> Mapping[str, Any]:
    return protocol.build_runtime_binding(
        executor_release_sha="a" * 40,
        executor_plan_sha256="b" * 64,
        executor_binary_sha256="7" * 64,
        mutation_wrapper_sha256="8" * 64,
        remote_transport_sha256="9" * 64,
    )


def _activation_seal(*, release_revision: str = "a" * 40) -> Mapping[str, Any]:
    lineage_unsigned = {
        "schema": service.ACTIVATION_RELEASE_LINEAGE_SCHEMA,
        "release_revision": release_revision,
        "source_tree_oid": "b" * 40,
        "package_inventory_sha256": "6" * 64,
        "release_trust_manifest_sha256": "7" * 64,
        "release_trust_public_key_sha256": "8" * 64,
        "direct_iam_identity_authority_sha256": "9" * 64,
        "pre_foundation_authority_sha256": "a" * 64,
        "foundation_apply_receipt_sha256": "b" * 64,
        "foundation_owner_reauthentication_receipt_sha256": "c" * 64,
        "activation_owner_reauthentication_receipt_sha256": "d" * 64,
        "project_ancestry_evidence_sha256": "e" * 64,
        "project_ancestry_chain_sha256": "f" * 64,
        "resource_ancestor_chain": [
            "organizations/123456789012",
            "projects/123456789012",
        ],
        "inert_preflight_receipt_sha256": "0" * 64,
        "post_iam_preflight_receipt_sha256": "5" * 64,
    }
    lineage = {
        **lineage_unsigned,
        "lineage_sha256": protocol.sha256_json(lineage_unsigned),
    }
    unsigned = {
        "schema": service.ACTIVATION_SEAL_SCHEMA,
        "release_revision": release_revision,
        "foundation_plan_sha256": "1" * 64,
        "package_sha256": "2" * 64,
        "cloud_topology_receipt_sha256": "3" * 64,
        "host_security_smoke_receipt_sha256": "4" * 64,
        "iam_repreflight_receipt_sha256": "5" * 64,
        "owner_gate_vm_numeric_id": "1234567890123456789",
        "target_instance_numeric_id": service.storage.VM_INSTANCE_ID,
        "target_disk_numeric_id": service.storage.DISK_ID,
        "created_at_unix": NOW,
        "authorization_record_complete": True,
        "verified_release_lineage": lineage,
        "evidence_file_sha256": {
            name: "f" * 64 for name in service._ACTIVATION_EVIDENCE_NAMES
        },
        "activation_installed": True,
        "cloud_mutation_performed": False,
    }
    return {**unsigned, "seal_sha256": protocol.sha256_json(unsigned)}


def _database_root(tmp_path: Path) -> tuple[Path, int, int]:
    root = tmp_path / "authority"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    state = root.stat()
    return root / "passkey-v2.sqlite3", state.st_uid, state.st_gid


def _ready_authority(
    tmp_path: Path,
) -> tuple[
    database.PasskeyV2AuthorityDatabase,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    path, uid, gid = _database_root(tmp_path)
    database.bootstrap_authority_database(
        path,
        authority_uid=uid,
        authority_gid=gid,
        now_unix=NOW - 200,
        require_root=False,
    )
    authority = database.PasskeyV2AuthorityDatabase(
        path, authority_uid=uid, authority_gid=gid
    )
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(envelope, challenge)
    authority.import_migrated_credential(credential)
    authority.create_request(envelope)
    authority.create_challenge(challenge, envelope=envelope)
    grant = authority.verify_assertion_and_record_grant(
        assertion=assertion,
        envelope=envelope,
        challenge=challenge,
        grant_id="G" * 32,
        now_unix=NOW + 2,
    )
    return authority, envelope, challenge, grant


@pytest.mark.parametrize(
    "raw,error",
    [
        (b'{"a":1,"a":2}', "duplicate"),
        (b'{"a":NaN}', "nonfinite"),
        (b'{"a":1.0}', "floating"),
        (b'{"a": 1}', "not_canonical"),
        (b'\xff', "json_invalid"),
    ],
)
def test_canonical_json_rejects_ambiguous_encodings(raw: bytes, error: str) -> None:
    with pytest.raises(protocol.PasskeyV2ProtocolError, match=error):
        protocol.decode_canonical_json(raw)


def test_challenge_rejects_noncanonical_base64url_pad_bits() -> None:
    envelope = _envelope()
    canonical = _b64(b"x" * 32)
    with pytest.raises(protocol.PasskeyV2ProtocolError, match="challenge_invalid"):
        protocol.build_challenge_record(
            envelope=envelope,
            challenge_id="C" * 32,
            challenge_b64url=_noncanonical_pad_bits(canonical),
            rp_id=protocol.PRODUCTION_RP_ID,
            origin=protocol.PRODUCTION_ORIGIN,
            created_at_unix=NOW + 1,
        )


def test_action_envelope_and_ui_render_every_exact_bound_value() -> None:
    envelope = _envelope()
    view = protocol.build_ui_view(envelope)
    assert view["exact_action_envelope_canonical_json"] == (
        protocol.canonical_json_bytes(envelope).decode("utf-8")
    )
    assert view["exact_action_payload_canonical_json"] == (
        protocol.canonical_json_bytes(envelope["action_payload"]).decode("utf-8")
    )
    assert len(view["full_action_envelope_sha256"]) == 64
    for value in envelope.values():
        if isinstance(value, str):
            assert value in view["exact_action_envelope_canonical_json"]
    protocol.validate_ui_headers(protocol.UI_SECURITY_HEADERS)
    assert protocol.UI_SECURITY_HEADERS["Cache-Control"].startswith("no-store")
    assert "frame-ancestors 'none'" in protocol.UI_SECURITY_HEADERS[
        "Content-Security-Policy"
    ]
    assert protocol.UI_SECURITY_HEADERS["X-Frame-Options"] == "DENY"


def test_production_boundary_rejects_wrong_rp_or_origin_and_totp() -> None:
    envelope = dict(_envelope())
    envelope["webauthn_origin"] = "https://lomliev.com"
    unsigned = {key: item for key, item in envelope.items() if key != "envelope_sha256"}
    envelope["envelope_sha256"] = protocol.sha256_json(unsigned)
    with pytest.raises(protocol.PasskeyV2ProtocolError, match="production_webauthn"):
        protocol.require_production_webauthn_identity(envelope)
    with pytest.raises(protocol.PasskeyV2ProtocolError, match="totp_disabled"):
        protocol.require_dangerous_approval_method("totp")


@pytest.mark.parametrize(
    "change,error",
    [
        ({"client_origin": "https://evil.example"}, "cryptographic_verification"),
        ({"client_challenge": _b64(b"wrong" * 8)}, "cryptographic_verification"),
        ({"rp_id": "evil.example"}, "cryptographic_verification"),
        ({"flags": 0x01}, "cryptographic_verification"),
    ],
)
def test_webauthn_rejects_wrong_origin_challenge_rp_and_missing_uv(
    change: Mapping[str, Any], error: str
) -> None:
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(
        envelope, challenge, **change
    )
    with pytest.raises(webauthn.PasskeyV2WebAuthnError, match=error):
        webauthn.verify_assertion(
            assertion,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )


def test_webauthn_rejects_forged_signature_and_accepts_zero_counter() -> None:
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(envelope, challenge)
    verified = webauthn.verify_assertion(
        assertion,
        credential=credential,
        challenge=challenge,
        envelope=envelope,
        prior_sign_count=0,
    )
    assert verified["credential_sign_count"] == 0
    forged = dict(assertion)
    forged["credential"] = dict(assertion["credential"])
    forged["credential"]["response"] = dict(assertion["credential"]["response"])
    encoded_signature = forged["credential"]["response"]["signature"]
    signature = bytearray(base64.urlsafe_b64decode(
        encoded_signature + "=" * (-len(encoded_signature) % 4)
    ))
    signature[-1] ^= 1
    forged["credential"]["response"]["signature"] = _b64(bytes(signature))
    with pytest.raises(
        webauthn.PasskeyV2WebAuthnError, match="cryptographic_verification_failed"
    ):
        webauthn.verify_assertion(
            forged,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )


@pytest.mark.parametrize(
    "client_extra,error",
    [
        ({"crossOrigin": True}, "cross_origin_forbidden"),
        ({"crossOrigin": "false"}, "cross_origin_forbidden"),
        ({"topOrigin": protocol.PRODUCTION_ORIGIN}, "cross_origin_forbidden"),
    ],
)
def test_webauthn_rejects_cross_origin_context(
    client_extra: Mapping[str, Any], error: str
) -> None:
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(
        envelope,
        challenge,
        client_extra=client_extra,
    )
    with pytest.raises(webauthn.PasskeyV2WebAuthnError, match=error):
        webauthn.verify_assertion(
            assertion,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )


def test_webauthn_rejects_duplicate_client_data_key() -> None:
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(envelope, challenge)
    modified = dict(assertion)
    modified["credential"] = dict(assertion["credential"])
    modified["credential"]["response"] = dict(
        assertion["credential"]["response"]
    )
    duplicate = (
        b'{"type":"webauthn.get","challenge":"'
        + challenge["challenge_b64url"].encode("ascii")
        + b'","origin":"https://auth.lomliev.com",'
        b'"crossOrigin":false,"crossOrigin":false}'
    )
    modified["credential"]["response"]["clientDataJSON"] = _b64(duplicate)
    with pytest.raises(
        webauthn.PasskeyV2WebAuthnError, match="client_data_duplicate_key"
    ):
        webauthn.verify_assertion(
            modified,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )


def test_webauthn_binds_non_null_user_handle_to_owner() -> None:
    envelope = _envelope()
    challenge = _challenge(envelope)
    credential, assertion = _credential_and_assertion(
        envelope,
        challenge,
        user_handle=OWNER.encode("utf-8"),
    )
    verified = webauthn.verify_assertion(
        assertion,
        credential=credential,
        challenge=challenge,
        envelope=envelope,
        prior_sign_count=0,
    )
    assert credential["expected_user_handle_sha256"] == hashlib.sha256(
        OWNER.encode("utf-8")
    ).hexdigest()
    assert credential["expected_user_handle_sha256"] == (
        "a72512de5fcd7fa3e679fcca570c9b4db6ff1e403b6329586ddad90c093ad983"
    )
    assert verified["approver_discord_user_id"] == OWNER

    _credential, wrong = _credential_and_assertion(
        envelope,
        challenge,
        user_handle=b"1279454038731264062",
    )
    with pytest.raises(
        webauthn.PasskeyV2WebAuthnError,
        match="assertion_user_handle_mismatch",
    ):
        webauthn.verify_assertion(
            wrong,
            credential=_credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )


def test_authority_never_accepts_self_asserted_grant(tmp_path: Path) -> None:
    authority, envelope, challenge, grant = _ready_authority(tmp_path)
    with pytest.raises(database.PasskeyV2SqliteDenied, match="untrusted_grant"):
        authority.record_passkey_grant(
            grant,
            envelope=envelope,
            challenge=challenge,
            now_unix=NOW + 3,
        )


@pytest.mark.parametrize(
    ("operation", "document"),
    [
        ("create_request", {"action_envelope": _envelope()}),
        (
            "consume",
            {
                "request_id": "R" * 32,
                "runtime_binding": _runtime(),
                "consume_attempt_id": "a" * 64,
            },
        ),
    ],
)
def test_public_web_peer_cannot_author_or_consume_owner_requests(
    tmp_path: Path,
    operation: str,
    document: Mapping[str, Any],
) -> None:
    path, uid, gid = _database_root(tmp_path)
    database.bootstrap_authority_database(
        path,
        authority_uid=uid,
        authority_gid=gid,
        now_unix=NOW - 1,
        require_root=False,
    )
    authority = database.PasskeyV2AuthorityDatabase(
        path,
        authority_uid=uid,
        authority_gid=gid,
    )
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_authority_privileged_peer_required",
    ):
        service.handle_authority_frame(
            service.build_service_frame(operation, document),
            authority=authority,
            signer=ReceiptSigner(ed25519.Ed25519PrivateKey.generate()),
            peer_uid=service.WEB_UID,
            now_unix=NOW,
        )
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM requests").fetchone() == (0,)
    assert connection.execute("SELECT COUNT(*) FROM consumptions").fetchone() == (0,)
    connection.close()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"release_revision": "b" * 40}, "stale"),
        ({"target_disk_numeric_id": "123456"}, "invalid"),
        ({"created_at_unix": NOW + service.MAX_FUTURE_SKEW_SECONDS + 1}, "invalid"),
        ({"seal_sha256": "0" * 64}, "tampered"),
    ],
)
def test_activation_seal_rejects_stale_tampered_or_wrong_target_authority(
    change: Mapping[str, Any],
    error: str,
) -> None:
    seal = {**_activation_seal(), **change}
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match=f"passkey_v2_activation_seal_{error}",
    ):
        service.validate_activation_seal(
            seal,
            expected_release_revision="a" * 40,
            now_unix=NOW,
        )


def test_activation_seal_rejects_incomplete_or_nested_tampered_truth() -> None:
    incomplete = {
        **_activation_seal(),
        "authorization_record_complete": False,
    }
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_activation_seal_invalid",
    ):
        service.validate_activation_seal(
            incomplete,
            expected_release_revision="a" * 40,
            now_unix=NOW,
        )

    seal = dict(_activation_seal())
    lineage = dict(seal["verified_release_lineage"])
    lineage["foundation_apply_receipt_sha256"] = "0" * 64
    seal["verified_release_lineage"] = lineage
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_activation_seal_tampered",
    ):
        service.validate_activation_seal(
            seal,
            expected_release_revision="a" * 40,
            now_unix=NOW,
        )

    seal = dict(_activation_seal())
    evidence = dict(seal["evidence_file_sha256"])
    evidence.pop("post-iam-preflight.json")
    seal["evidence_file_sha256"] = evidence
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_activation_seal_invalid",
    ):
        service.validate_activation_seal(
            seal,
            expected_release_revision="a" * 40,
            now_unix=NOW,
        )


def test_activation_seal_reader_pins_root_executor_identity_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = _activation_seal()
    observed: dict[str, Any] = {}

    def read_exact(path: Path, **requirements: Any) -> tuple[bytes, object]:
        observed["path"] = path
        observed.update(requirements)
        return protocol.canonical_json_bytes(seal), object()

    monkeypatch.setattr(service, "_read_regular_file", read_exact)
    assert service.read_activation_seal(
        expected_release_revision="a" * 40,
        now_unix=NOW,
        path=Path("/fixed/activation-seal"),
    ) == seal
    assert observed == {
        "path": Path("/fixed/activation-seal"),
        "maximum": service.MAX_SEAL_BYTES,
        "expected_uid": 0,
        "expected_gid": service.EXECUTOR_GID,
        "expected_mode": 0o440,
    }


def test_sqlite_consume_is_atomic_single_use_and_exact_replay(tmp_path: Path) -> None:
    authority, envelope, _challenge_record, _grant = _ready_authority(tmp_path)
    signer = ReceiptSigner(ed25519.Ed25519PrivateKey.generate())
    runtime = _runtime()
    barrier = threading.Barrier(20)

    def consume(index: int) -> tuple[str, str]:
        attempt = f"{index + 10:064x}"
        barrier.wait()
        try:
            result = authority.consume_or_replay(
                envelope=envelope,
                runtime_binding=runtime,
                consume_attempt_id=attempt,
                signer=signer,
                now_unix=NOW + 3,
            )
            return result.disposition, attempt
        except database.PasskeyV2SqliteDenied as exc:
            return str(exc), attempt

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(consume, range(20)))
    winners = [item for item in results if item[0] == "authorized_once"]
    assert len(winners) == 1
    winner_attempt = winners[0][1]
    replay = authority.consume_or_replay(
        envelope=envelope,
        runtime_binding=runtime,
        consume_attempt_id=winner_attempt,
        signer=signer,
        now_unix=NOW + 4,
    )
    assert replay.disposition == "receipt_replay"
    assert replay.receipt["consume_attempt_id"] == winner_attempt
    assert replay.receipt["authorization_disposition"] == "authorized_once"
    with pytest.raises(database.PasskeyV2SqliteDenied, match="different_attempt"):
        authority.consume_or_replay(
            envelope=envelope,
            runtime_binding=runtime,
            consume_attempt_id="f" * 64,
            signer=signer,
            now_unix=NOW + 4,
        )
    assert authority.assert_bijection()["authorization_count"] == 1


def test_authorization_receipt_rejects_noncanonical_signature_pad_bits(
    tmp_path: Path,
) -> None:
    authority, envelope, challenge, grant = _ready_authority(tmp_path)
    signer = ReceiptSigner(ed25519.Ed25519PrivateKey.generate())
    result = authority.consume_or_replay(
        envelope=envelope,
        runtime_binding=_runtime(),
        consume_attempt_id="d" * 64,
        signer=signer,
        now_unix=NOW + 3,
    )
    changed = dict(result.receipt)
    changed["signature_ed25519_b64url"] = _noncanonical_pad_bits(
        changed["signature_ed25519_b64url"]
    )
    changed["receipt_sha256"] = protocol.sha256_json({
        key: item for key, item in changed.items() if key != "receipt_sha256"
    })
    with pytest.raises(
        protocol.PasskeyV2ProtocolError, match="receipt_signature_invalid"
    ):
        protocol.validate_authorization_receipt(
            changed,
            envelope=envelope,
            grant=grant,
            challenge=challenge,
            receipt_public_key=signer.public_key,
        )


def test_sqlite_abort_cannot_leave_tombstone_without_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, envelope, _challenge_record, _grant = _ready_authority(tmp_path)
    original = authority._journal_state
    calls = 0

    def fail_after_atomic_inserts(
        connection: sqlite3.Connection,
    ) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise database.PasskeyV2SqliteError("injected_before_commit")
        return original(connection)

    monkeypatch.setattr(authority, "_journal_state", fail_after_atomic_inserts)
    with pytest.raises(database.PasskeyV2SqliteError, match="injected_before_commit"):
        authority.consume_or_replay(
            envelope=envelope,
            runtime_binding=_runtime(),
            consume_attempt_id="a" * 64,
            signer=ReceiptSigner(ed25519.Ed25519PrivateKey.generate()),
            now_unix=NOW + 3,
        )
    check = sqlite3.connect(authority.path)
    counts = check.execute(
        "SELECT (SELECT COUNT(*) FROM consumptions),"
        "(SELECT COUNT(*) FROM authorization_journal)"
    ).fetchone()
    check.close()
    assert counts == (0, 0)


def test_sqlite_exact_schema_attestation_rejects_removed_trigger(
    tmp_path: Path,
) -> None:
    authority, _envelope_value, _challenge_record, _grant = _ready_authority(
        tmp_path
    )
    raw = sqlite3.connect(authority.path)
    raw.execute("DROP TRIGGER authorization_journal_no_delete")
    raw.close()
    with pytest.raises(database.PasskeyV2SqliteError, match="schema_drift"):
        authority.preflight()


def test_sqlite_bootstrap_schema_and_metadata_are_one_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, uid, gid = _database_root(tmp_path)

    def injected_failure(_connection: sqlite3.Connection) -> str:
        raise RuntimeError("injected_schema_attestation_failure")

    monkeypatch.setattr(database, "_sqlite_master_sha256", injected_failure)
    with pytest.raises(RuntimeError, match="injected_schema_attestation_failure"):
        database.bootstrap_authority_database(
            path,
            authority_uid=uid,
            authority_gid=gid,
            now_unix=NOW,
            require_root=False,
        )
    raw = sqlite3.connect(path)
    objects = raw.execute(
        "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    raw.close()
    assert objects == []


def test_append_only_tables_reject_update_delete_and_reapproval(tmp_path: Path) -> None:
    authority, envelope, challenge, grant = _ready_authority(tmp_path)
    connection = sqlite3.connect(authority.path)
    with pytest.raises(sqlite3.IntegrityError, match="append_only"):
        connection.execute("DELETE FROM grants")
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="append_only"):
        connection.execute("UPDATE requests SET request_id='x'")
    connection.close()
    with pytest.raises(database.PasskeyV2SqliteDenied, match="untrusted_grant"):
        authority.record_passkey_grant(
            grant,
            envelope=envelope,
            challenge=challenge,
            now_unix=NOW + 4,
        )


def test_begin_immediate_maps_real_sqlite_lock_to_concurrent_attempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "locked.sqlite3"
    holder = sqlite3.connect(path, isolation_level=None)
    contender = sqlite3.connect(path, isolation_level=None)
    try:
        holder.execute("CREATE TABLE probe (value INTEGER NOT NULL)")
        holder.execute("BEGIN IMMEDIATE")
        contender.execute("PRAGMA busy_timeout=0")

        with pytest.raises(
            database.PasskeyV2SqliteDenied,
            match="^passkey_v2_concurrent_attempt$",
        ):
            database._Database._begin_immediate(contender)
    finally:
        holder.rollback()
        holder.close()
        contender.close()


def test_begin_immediate_does_not_route_on_misleading_error_text() -> None:
    error = sqlite3.OperationalError("no such table: busy_jobs")
    error.sqlite_errorcode = sqlite3.SQLITE_ERROR

    class MisleadingConnection:
        @staticmethod
        def execute(_sql: str) -> None:
            raise error

    with pytest.raises(
        database.PasskeyV2SqliteError,
        match="^passkey_v2_transaction_failed$",
    ):
        database._Database._begin_immediate(
            MisleadingConnection()  # type: ignore[arg-type]
        )


def test_sqlite_database_path_symlink_is_rejected(tmp_path: Path) -> None:
    path, uid, gid = _database_root(tmp_path)
    target = path.parent / "target.sqlite3"
    target.write_bytes(b"not-sqlite")
    target.chmod(0o600)
    path.symlink_to(target)
    with pytest.raises(database.PasskeyV2SqliteError, match="database_file_invalid"):
        database.PasskeyV2AuthorityDatabase(
            path, authority_uid=uid, authority_gid=gid
        )


def test_sqlite_storage_is_runtime_eligible_and_legacy_backend_is_unimportable() -> None:
    assert database.RUNTIME_ELIGIBLE is True
    assert database.PasskeyV2AuthorityDatabase.__module__ == database.__name__
    assert importlib.util.find_spec("scripts.canary.passkey_v2_store") is None
