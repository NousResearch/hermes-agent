from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from gateway import canonical_full_canary_coordinator as coordinator
from gateway import canonical_writer_foundation_phase_b as phase_b
from scripts.canary import full_canary_owner_launcher as launcher


def test_full_canary_fixture_targets_only_the_dedicated_public_channel() -> None:
    locked_private_channels = {
        "1504852355588423801",
        "1505499746939174993",
    }

    assert launcher.FIXTURE_PUBLICATION_CHANNEL_ID == "1526858760100909066"
    assert launcher.FIXTURE_PUBLICATION_CHANNEL_ID not in locked_private_channels


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _gate(*, now: int = 1_000) -> Mapping[str, Any]:
    unsigned = {
        "schema": launcher.PHASE_B_APPLY_GATE_SCHEMA,
        "ok": True,
        "state": "initial_apply_ready",
        "release_sha": "a" * 40,
        "coordinator_input_sha256": _digest("coordinator-input"),
        "owner_subject_sha256": _digest("owner"),
        "approval_source_sha256": _digest("approval-source"),
        "authority_present": False,
        "phase_b_plan_sha256": None,
        "phase_b_approval_sha256": None,
        "phase_b_approval_sequence": None,
        "phase_b_incomplete_state": None,
        "phase_b_inspection_sha256": None,
        "phase_b_terminal": False,
        "phase_b_requires_reapproval": False,
        "issued_at_unix": now,
        "expires_at_unix": now + 300,
    }
    return {
        **unsigned,
        "gate_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _context(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    release = str(gate["release_sha"])
    activation_plan = _digest("activation-plan")
    activation_approval = _digest("activation-approval")
    native_plan = _digest("native-plan")
    native_approval = _digest("native-approval")
    external_iam = _digest("external-iam")
    collector = _digest("collector")
    gateway_intent = _digest("gateway-intent")
    edge_intent = _digest("edge-intent")
    paths = {
        "coordinator_input": "/etc/muncho/full-canary/coordinator-input.json",
        "activation_plan": "/etc/muncho/writer-activation/activation-plan.json",
        "activation_receipt": (
            f"/var/lib/muncho-writer-activation/plans/{release}/"
            f"{activation_plan}/success/activation.json"
        ),
        "activation_owner_approval": (
            "/etc/muncho/writer-activation/approvals/activation/"
            f"{activation_plan}/{activation_approval}.json"
        ),
        "native_plan": (
            "/etc/muncho/writer-activation/native-observation-plan.json"
        ),
        "native_receipt": (
            f"/var/lib/muncho-writer-canary-evidence/{release}/{native_plan}/"
            "native-observation.json"
        ),
        "native_owner_approval": (
            "/etc/muncho/writer-activation/approvals/native_observation/"
            f"{native_plan}/{native_approval}.json"
        ),
        "external_iam_receipt": (
            f"/var/lib/muncho-writer-canary-evidence/{release}/{native_plan}/"
            f"external-iam/{external_iam}.json"
        ),
        "config_collector_receipt": (
            "/var/lib/muncho-writer-canary-evidence/config-collector/"
            f"{release}/{collector}.json"
        ),
        "gateway_config_intent": "/etc/muncho/full-canary/staged/gateway.yaml",
        "edge_config_intent": (
            "/etc/muncho/full-canary/staged/discord-edge.json"
        ),
        "fixture_intent": "/etc/muncho/full-canary/fixture.json",
        "host_identity_receipt": "/etc/muncho/full-canary/host-identity.json",
    }
    sources = {
        label: {
            "path": path,
            "file_sha256": (
                gateway_intent
                if label == "gateway_config_intent"
                else edge_intent
                if label == "edge_config_intent"
                else _digest(f"file:{label}")
            ),
            "device": 1,
            "inode": ordinal + 10,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "size": 100 + ordinal,
        }
        for ordinal, (label, path) in enumerate(sorted(paths.items()))
    }
    sources["owner_resume_public_key"] = None
    value = {
        "release_sha": release,
        "coordinator_input_sha256": gate["coordinator_input_sha256"],
        "activation_plan_sha256": activation_plan,
        "writer_activation_receipt_sha256": _digest("activation-receipt"),
        "activation_owner_approval_sha256": activation_approval,
        "activation_approval_issued_at_unix": 100,
        "activation_approval_expires_at_unix": 200,
        "native_observation_plan_sha256": native_plan,
        "native_observation_receipt_sha256": _digest("native-receipt"),
        "native_observation_approval_sha256": native_approval,
        "native_approval_issued_at_unix": 300,
        "native_approval_expires_at_unix": 400,
        "external_iam_policy_sha256": _digest("external-policy"),
        "external_iam_receipt_sha256": external_iam,
        "config_collector_receipt_sha256": collector,
        "gateway_config_intent_sha256": gateway_intent,
        "edge_config_intent_sha256": edge_intent,
        "fixture_intent_sha256": _digest("fixture"),
        "host_identity_receipt_sha256": _digest("host"),
        "authority_sources": dict(sorted(sources.items())),
        "authority_sources_sha256": hashlib.sha256(
            _canonical(dict(sorted(sources.items())))
        ).hexdigest(),
        "approval_source_sha256": _digest("approval-source"),
        "owner_subject_sha256": gate["owner_subject_sha256"],
        "owner_resume_public_key_ed25519_hex": None,
        "owner_resume_key_id": None,
        "owner_resume_public_key_file_sha256": None,
        "owner_resume_public_fingerprint": None,
    }
    assert set(value) == launcher._PHASE_B_AUTHORITY_CONTEXT_KEYS
    return value


def _request(
    gate: Mapping[str, Any],
    *,
    operation: str = "authority_observe_initial",
    payload: Mapping[str, Any] | None = None,
    plan_sha256: str | None = None,
    approval_sha256: str | None = None,
    boundary_kind: str | None = None,
    boundary_ordinal: int | None = None,
    credential_expected: bool = False,
) -> Mapping[str, Any]:
    context = _context(gate)
    context_sha = hashlib.sha256(_canonical(context)).hexdigest()
    body = {} if payload is None else copy.deepcopy(dict(payload))
    idempotency_projection = {
        "authority_context_sha256": context_sha,
        "phase_b_plan_sha256": plan_sha256,
        "phase_b_approval_sha256": approval_sha256,
        "operation": operation,
        "sequence": 0,
        "previous_response_sha256": None,
        "payload": body,
    }
    unsigned = {
        "schema": launcher.PHASE_B_OWNER_REQUEST_SCHEMA,
        "frame_schema": launcher.PHASE_B_OWNER_FRAME_SCHEMA,
        "operation": operation,
        "sequence": 0,
        "previous_response_sha256": None,
        "authority_context_sha256": context_sha,
        "authority_context": context,
        "phase_b_plan_sha256": plan_sha256,
        "phase_b_approval_sha256": approval_sha256,
        "boundary_kind": boundary_kind,
        "boundary_ordinal": boundary_ordinal,
        "credential_expected": credential_expected,
        "payload": body,
        "issued_at_unix": 1_000,
        "expires_at_unix": gate["expires_at_unix"],
        "idempotency_key": hashlib.sha256(
            _canonical(idempotency_projection)
        ).hexdigest(),
    }
    return {
        **unsigned,
        "request_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }


def _protocol(gate: Mapping[str, Any]) -> launcher._PhaseBOwnerProtocol:
    return launcher._PhaseBOwnerProtocol(
        gate=gate,
        cloud_sql_client=object(),  # validation tests never cross the Cloud edge
        password_factory=lambda: bytearray(b"A" * 64),
        clock=lambda: 1_000.0,
    )


def _test_signer(tmp_path: Path) -> launcher._PhaseBOwnerExternalSigner:
    key = tmp_path / "owner-key"
    comment = "phase-b-owner-test"
    subprocess.run(
        [
            str(launcher.PHASE_B_SSH_KEYGEN),
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            comment,
            "-f",
            str(key),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    public = key.with_suffix(".pub")
    os.chmod(public, 0o600)
    encoded = public.read_bytes().split(b" ", 2)[1]
    public_blob = base64.b64decode(encoded, validate=True)
    fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(public_blob).digest()
    ).decode("ascii").rstrip("=")
    return launcher._PhaseBOwnerExternalSigner(
        private_key_path=key,
        public_key_path=public,
        expected_comment=comment,
        expected_fingerprint=fingerprint,
    )


def test_owner_external_signer_uses_strict_sshsig_and_pins_public_source(
    tmp_path: Path,
) -> None:
    signer = _test_signer(tmp_path)
    authority = signer.inspect()
    message = _canonical({"approved": True, "plan_sha256": _digest("plan")})

    signature = signer.sign(
        message,
        namespace=launcher.PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
        expected_authority=authority,
    )

    phase_b.verify_phase_b_sshsig(
        signature,
        message=message,
        public_key_ed25519_hex=authority.public_key_ed25519_hex,
        namespace=launcher.PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
    )
    public_source = authority.to_mapping()["public_key_source"]
    assert public_source["path"].endswith("owner-key.pub")
    assert public_source["mode"] == "0600"
    assert public_source["file_sha256"] == authority.public_key_file_sha256
    assert authority.key_id == hashlib.sha256(
        bytes.fromhex(authority.public_key_ed25519_hex)
    ).hexdigest()
    assert "PRIVATE" not in json.dumps(authority.to_mapping())


def test_owner_external_signer_rejects_private_key_metadata_drift(
    tmp_path: Path,
) -> None:
    signer = _test_signer(tmp_path)
    authority = signer.inspect()
    os.chmod(tmp_path / "owner-key", 0o640)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="phase_b_owner_private_key_untrusted",
    ):
        signer.sign(
            b"fixed-message",
            namespace=launcher.PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
            expected_authority=authority,
        )


def test_owner_request_binds_exact_context_sequence_and_rejects_replay() -> None:
    gate = _gate()
    protocol = _protocol(gate)
    request = _request(gate)

    assert protocol.validate_request(request) == request
    with pytest.raises(launcher.OwnerLauncherError, match="invalid_phase_b_request"):
        protocol.validate_request(request)


def test_owner_request_rejects_forged_root_source_even_with_fresh_hashes() -> None:
    gate = _gate()
    request = copy.deepcopy(dict(_request(gate)))
    context = dict(request["authority_context"])
    sources = copy.deepcopy(dict(context["authority_sources"]))
    sources["coordinator_input"]["path"] = "/tmp/forged.json"
    context["authority_sources"] = sources
    context["authority_sources_sha256"] = hashlib.sha256(
        _canonical(dict(sorted(sources.items())))
    ).hexdigest()
    request["authority_context"] = context
    request["authority_context_sha256"] = hashlib.sha256(
        _canonical(context)
    ).hexdigest()
    projection = {
        "authority_context_sha256": request["authority_context_sha256"],
        "phase_b_plan_sha256": request["phase_b_plan_sha256"],
        "phase_b_approval_sha256": request["phase_b_approval_sha256"],
        "operation": request["operation"],
        "sequence": request["sequence"],
        "previous_response_sha256": request["previous_response_sha256"],
        "payload": request["payload"],
    }
    request["idempotency_key"] = hashlib.sha256(_canonical(projection)).hexdigest()
    unsigned = dict(request)
    del unsigned["request_sha256"]
    request["request_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="invalid_phase_b_authority_context",
    ):
        _protocol(gate).validate_request(request)


def test_owner_context_allows_only_exact_public_authority_monotonic_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _gate()
    signer = _test_signer(tmp_path)
    test_authority = signer.inspect()
    public_source = test_authority.to_mapping()["public_key_source"]
    monkeypatch.setattr(
        launcher,
        "PHASE_B_OWNER_PUBLIC_KEY_PATH",
        Path(public_source["path"]),
    )
    monkeypatch.setattr(
        launcher,
        "PHASE_B_OWNER_PUBLIC_KEY_UID",
        public_source["uid"],
    )
    monkeypatch.setattr(
        launcher,
        "PHASE_B_OWNER_PUBLIC_KEY_GID",
        public_source["gid"],
    )
    monkeypatch.setattr(
        launcher,
        "PHASE_B_OWNER_PUBLIC_KEY_FINGERPRINT",
        test_authority.public_fingerprint,
    )
    protocol = launcher._PhaseBOwnerProtocol(
        gate=gate,
        cloud_sql_client=object(),
        password_factory=lambda: bytearray(b"A" * 64),
        clock=lambda: 1_000.0,
        owner_signer=signer,
    )
    monkeypatch.setattr(
        launcher,
        "_phase_b_initial_cloud_observation",
        lambda _boundary: {"project": launcher.PROJECT},
    )
    first = _request(
        gate,
        payload={"local_preflight": {"release_revision": gate["release_sha"]}},
    )
    validated = protocol.validate_request(first)
    frame, _terminal = protocol.response_frame(validated)
    try:
        receipt_size = struct.unpack(">I", frame[4:8])[0]
        receipt = json.loads(bytes(frame[12 : 12 + receipt_size]))
    finally:
        launcher._wipe(frame)
    authority = protocol._owner_resume_authority
    assert authority is not None
    context = copy.deepcopy(dict(first["authority_context"]))
    evidence = authority.to_mapping()
    sources = copy.deepcopy(dict(context["authority_sources"]))
    sources["owner_resume_public_key"] = evidence["public_key_source"]
    context.update({
        "owner_resume_public_key_ed25519_hex": evidence[
            "public_key_ed25519_hex"
        ],
        "owner_resume_key_id": evidence["key_id"],
        "owner_resume_public_key_file_sha256": evidence[
            "public_key_file_sha256"
        ],
        "owner_resume_public_fingerprint": evidence["public_fingerprint"],
        "authority_sources": dict(sorted(sources.items())),
    })
    context["authority_sources_sha256"] = hashlib.sha256(
        _canonical(context["authority_sources"])
    ).hexdigest()

    def second_request(bound_context: Mapping[str, Any]) -> Mapping[str, Any]:
        context_sha = hashlib.sha256(_canonical(bound_context)).hexdigest()
        payload = {"phase_b_plan": {}}
        projection = {
            "authority_context_sha256": context_sha,
            "phase_b_plan_sha256": _digest("plan"),
            "phase_b_approval_sha256": None,
            "operation": "authority_approve",
            "sequence": 1,
            "previous_response_sha256": receipt["response_sha256"],
            "payload": payload,
        }
        unsigned = {
            "schema": launcher.PHASE_B_OWNER_REQUEST_SCHEMA,
            "frame_schema": launcher.PHASE_B_OWNER_FRAME_SCHEMA,
            "operation": "authority_approve",
            "sequence": 1,
            "previous_response_sha256": receipt["response_sha256"],
            "authority_context_sha256": context_sha,
            "authority_context": copy.deepcopy(dict(bound_context)),
            "phase_b_plan_sha256": _digest("plan"),
            "phase_b_approval_sha256": None,
            "boundary_kind": None,
            "boundary_ordinal": None,
            "credential_expected": False,
            "payload": payload,
            "issued_at_unix": 1_000,
            "expires_at_unix": gate["expires_at_unix"],
            "idempotency_key": hashlib.sha256(
                _canonical(projection)
            ).hexdigest(),
        }
        return {
            **unsigned,
            "request_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        }

    exact = second_request(context)
    assert protocol.validate_request(exact) == exact

    tampered = copy.deepcopy(context)
    tampered["authority_sources"]["owner_resume_public_key"]["inode"] += 1
    tampered["authority_sources_sha256"] = hashlib.sha256(
        _canonical(tampered["authority_sources"])
    ).hexdigest()
    other = launcher._PhaseBOwnerProtocol(
        gate=gate,
        cloud_sql_client=object(),
        clock=lambda: 1_000.0,
        owner_signer=signer,
    )
    other._context = copy.deepcopy(dict(first["authority_context"]))
    other._context_sha256 = first["authority_context_sha256"]
    other._sequence = 1
    other._previous_response_sha256 = receipt["response_sha256"]
    other._previous_operation = "authority_observe_initial"
    other._owner_resume_authority = authority
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="phase_b_authority_context_changed",
    ):
        other.validate_request(second_request(tampered))


def test_owner_response_frame_contains_only_exact_opaque_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    protocol = _protocol(gate)
    request = _request(
        gate,
        operation="temporary_create_or_rotate",
        plan_sha256=_digest("plan"),
        approval_sha256=_digest("approval"),
        boundary_kind="temporary",
        boundary_ordinal=0,
        credential_expected=True,
        payload={
            "username": launcher.ADMIN_USERNAME_PREFIX + _digest("plan")[:16],
            "expected_owner_subject_sha256": gate["owner_subject_sha256"],
            "expected_mutation_context_sha256": _digest("plan"),
        },
    )
    # Frame construction is tested independently from the transition graph;
    # transition admissibility is covered by the request tests above.
    validated = request
    monkeypatch.setattr(
        protocol,
        "execute",
        lambda _request: ({"authority_receipt": {"receipt_sha256": _digest("r")}}, bytearray(b"A" * 64), False),
    )

    frame, terminal = protocol.response_frame(validated)
    try:
        magic, receipt_size, credential_size = struct.unpack(">4sII", frame[:12])
        receipt = json.loads(bytes(frame[12 : 12 + receipt_size]))
        opaque = bytes(frame[12 + receipt_size :])
        assert magic == launcher.PHASE_B_OWNER_FRAME_MAGIC
        assert credential_size == launcher.PHASE_B_CREDENTIAL_BYTES == len(opaque)
        assert opaque == b"A" * 64
        assert receipt["credential_present"] is True
        assert receipt["credential_length"] == 64
        assert "password" not in json.dumps(receipt).casefold()
        assert terminal is False
    finally:
        launcher._wipe(frame)
    assert frame == bytearray(len(frame))


def test_vm_accepts_secret_free_error_without_credential_for_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue: list[bytearray] = []
    now = int(time.time())

    def emit(request: Mapping[str, Any]) -> None:
        unsigned = {
            "schema": coordinator.PHASE_B_OWNER_RESPONSE_SCHEMA,
            "frame_schema": coordinator.PHASE_B_OWNER_FRAME_SCHEMA,
            "ok": False,
            "operation": request["operation"],
            "sequence": request["sequence"],
            "request_sha256": request["request_sha256"],
            "idempotency_key": request["idempotency_key"],
            "authority_context_sha256": request["authority_context_sha256"],
            "phase_b_plan_sha256": request["phase_b_plan_sha256"],
            "phase_b_approval_sha256": request["phase_b_approval_sha256"],
            "credential_present": False,
            "credential_length": 0,
            "result": {"mutation_reconciliation_required": True},
            "error_code": "cloud_sql_mutation_ambiguous",
            "completed_at_unix": request["issued_at_unix"],
        }
        receipt = {
            **unsigned,
            "response_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        }
        raw = bytearray(_canonical(receipt))
        queue.extend([
            bytearray(struct.pack(">4sII", b"MPB1", len(raw), 0)),
            raw,
        ])

    monkeypatch.setattr(coordinator, "_ACTIVE_PHASE_B_FRAME_EMITTER", emit)
    monkeypatch.setattr(
        coordinator,
        "_read_phase_b_exact",
        lambda _size, *, maximum: queue.pop(0),
    )
    protocol = coordinator._FixedPhaseBVMProtocol(
        provenance={"release_sha": "a" * 40},
        phase_b_plan_sha256=_digest("plan"),
        phase_b_approval_sha256=_digest("approval"),
        approval_expires_at_unix=now + 60,
    )

    with pytest.raises(coordinator._PhaseBOwnerOperationError) as caught:
        protocol.exchange(
            "temporary_create_or_rotate",
            payload={},
            boundary_kind="temporary",
            boundary_ordinal=0,
            credential_expected=True,
        )
    assert caught.value.reconciliation_required is True
    assert protocol.sequence == 1
    assert queue == []


def test_protocol_caps_are_fixed_and_representative_request_has_headroom() -> None:
    gate = _gate()
    request = _request(gate)
    size = len(_canonical(request))

    assert launcher.PHASE_B_MAX_ROUNDS == coordinator.PHASE_B_MAX_ROUNDS == 16
    assert launcher.PHASE_B_MAX_REQUEST_BYTES == 512 * 1024
    assert launcher.PHASE_B_MAX_RESPONSE_BYTES == 512 * 1024
    assert launcher.PHASE_B_CREDENTIAL_BYTES == 64
    assert size < 32 * 1024
    assert launcher.PHASE_B_MAX_REQUEST_BYTES >= size * 16


def test_resume_protocol_allows_only_historical_repair_then_fresh_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coordinator.time, "time", lambda: 1_000)
    initial = _digest("approval-initial")
    historical = _digest("approval-historical")
    fresh = _digest("approval-fresh")
    protocol = coordinator._FixedPhaseBVMProtocol(
        provenance={"release_sha": "a" * 40},
        phase_b_plan_sha256=_digest("plan"),
        phase_b_approval_sha256=initial,
        approval_expires_at_unix=1_300,
    )
    protocol._sequence = 1

    protocol.bind_historical_resume_approval(
        expected_previous_approval_sha256=initial,
        approval_sha256=historical,
        approval_expires_at_unix=999,
    )
    assert protocol._approval_sha256 == historical
    assert protocol._expires == 1_300
    with pytest.raises(
        coordinator.CoordinatorError,
        match="phase_b_owner_protocol_rebind_forbidden",
    ):
        protocol.bind_historical_resume_approval(
            expected_previous_approval_sha256=historical,
            approval_sha256=_digest("second-historical"),
            approval_expires_at_unix=999,
        )

    protocol._sequence = 2
    protocol.bind_resume_approval(
        expected_previous_approval_sha256=historical,
        approval_sha256=fresh,
        approval_expires_at_unix=1_200,
    )
    assert protocol._approval_sha256 == fresh
    assert protocol._expires == 1_200
    assert launcher._PhaseBOwnerProtocol._MAX_COUNTS[
        "authority_resume_approve"
    ] == 2
    assert "authority_resume_approve" in launcher._PhaseBOwnerProtocol._ALLOWED_AFTER[
        "authority_resume_approve"
    ]


def test_vm_rejects_oversized_request_before_owner_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, Any]] = []
    monkeypatch.setattr(coordinator, "_ACTIVE_PHASE_B_FRAME_EMITTER", calls.append)
    protocol = coordinator._FixedPhaseBVMProtocol(
        provenance={"padding": "X" * coordinator.PHASE_B_MAX_REQUEST_BYTES},
        phase_b_plan_sha256=None,
        phase_b_approval_sha256=None,
        approval_expires_at_unix=int(time.time()) + 60,
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="phase_b_owner_request_oversized",
    ):
        protocol.exchange("authority_observe_initial", payload={})
    assert calls == []
    assert protocol.sequence == 0


def test_vm_read_bound_rejects_size_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int] = []
    monkeypatch.setattr(
        coordinator.os,
        "read",
        lambda _descriptor, size: reads.append(size) or b"X" * size,
    )

    with pytest.raises(
        coordinator.CoordinatorError,
        match="phase_b_owner_frame_bound_invalid",
    ):
        coordinator._read_phase_b_exact(65, maximum=64)
    assert reads == []


def test_owner_phase_b_terminal_replay_uses_no_cloud_or_secret_frame() -> None:
    gate = _gate()
    unsigned = {
        "schema": launcher.PHASE_B_APPLY_RECEIPT_SCHEMA,
        "ok": True,
        "state": "terminal_ready",
        "release_sha": gate["release_sha"],
        "coordinator_input_sha256": gate["coordinator_input_sha256"],
        "phase_b_plan_sha256": _digest("plan"),
        "phase_b_approval_sha256": _digest("approval"),
        "phase_b_terminal_receipt_sha256": _digest("terminal"),
        "phase_b_readiness_receipt_sha256": _digest("readiness"),
        "safe_to_start": True,
        "completed_at_unix": 1_000,
    }
    terminal = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }

    class Session:
        def __init__(self) -> None:
            self.marked: Mapping[str, Any] | None = None
            self.closed = False

        def read_gate(self) -> Mapping[str, Any]:
            return terminal

        def mark_validated(self, value: Mapping[str, Any]) -> None:
            self.marked = value

        def close(self) -> None:
            self.closed = True

    class Transport:
        def __init__(self) -> None:
            self.session = Session()

        def preflight_phase_b_apply(self, release_sha: str) -> Mapping[str, Any]:
            assert release_sha == gate["release_sha"]
            return gate

        def open_phase_b_apply(self, release_sha: str) -> Session:
            assert release_sha == gate["release_sha"]
            return self.session

    class Identity:
        def __init__(self) -> None:
            self.bound: str | None = None
            self.stability_checks = 0

        def bind_approved_subject(self, value: str) -> None:
            self.bound = value

        def require_stable(self) -> None:
            self.stability_checks += 1

    transport = Transport()
    identity = Identity()
    cloud_calls: list[str] = []

    receipt = launcher.apply_phase_b_foundation(
        release_sha=str(gate["release_sha"]),
        transport=transport,
        cloud_sql_client=lambda: cloud_calls.append("cloud"),
        owner_identity=identity,
        now=lambda: 1_000,
        secret_hardener=lambda: None,
        provenance_guard=lambda _release: None,
    )

    assert receipt == terminal
    assert transport.session.marked == terminal
    assert transport.session.closed is True
    assert identity.bound == gate["owner_subject_sha256"]
    assert identity.stability_checks == 2
    assert cloud_calls == []


def test_owner_cli_exposes_only_explicit_stopped_phase_b_apply(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    release_sha = "a" * 40
    events: list[object] = []

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            events.append("runtime")
            return ("/trusted/python",)

    class Identity:
        pass

    identity = Identity()
    transport = object()
    cloud_client = object()
    receipt = {
        "schema": launcher.PHASE_B_APPLY_RECEIPT_SCHEMA,
        "ok": True,
        "state": "terminal_ready",
    }

    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda exact_release: events.append(("trusted", exact_release)) or Runtime(),
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda exact_release: events.append(("provenance", exact_release)) or _digest(
            "launcher"
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda path: events.append(("interpreter", path)),
    )
    configuration = object()
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: configuration)
    monkeypatch.setattr(
        launcher,
        "GcloudOwnerAccessToken",
        lambda **kwargs: events.append(("identity", kwargs)) or identity,
    )
    monkeypatch.setattr(
        launcher,
        "IapCoordinatorTransport",
        lambda *args, **kwargs: events.append(("transport", args, kwargs)) or transport,
    )
    monkeypatch.setattr(
        launcher,
        "GoogleRestClient",
        lambda token_provider: events.append(("cloud", token_provider)) or cloud_client,
    )

    def apply(**kwargs: object) -> Mapping[str, Any]:
        events.append(("apply", kwargs))
        return receipt

    monkeypatch.setattr(launcher, "apply_phase_b_foundation", apply)
    monkeypatch.setattr(
        launcher,
        "launch_full_canary",
        lambda **_kwargs: pytest.fail("live launch must not run during Phase-B apply"),
    )
    monkeypatch.setattr(
        launcher,
        "OwnerDiscordTokenReader",
        lambda: pytest.fail("Discord token must not be read during Phase-B apply"),
    )

    assert launcher.main((
        "--release-sha",
        release_sha,
        "--apply-phase-b-foundation",
    )) == 0
    assert json.loads(capfd.readouterr().out) == receipt

    apply_event = next(item for item in events if isinstance(item, tuple) and item[0] == "apply")
    assert apply_event[1]["release_sha"] == release_sha
    assert apply_event[1]["transport"] is transport
    assert apply_event[1]["cloud_sql_client"] is cloud_client
    assert apply_event[1]["owner_identity"] is identity
    assert "password_factory" not in apply_event[1]
    assert events.count(("cloud", identity)) == 1


def test_owner_cli_rejects_phase_b_secret_or_unrelated_authority_inputs(
    capfd: pytest.CaptureFixture[str],
) -> None:
    release_sha = "a" * 40
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args((
            "--release-sha",
            release_sha,
            "--apply-phase-b-foundation",
            "--admin-password",
            "must-never-be-accepted",
        ))
    capfd.readouterr()

    # The only shared digest option is not authority for the stopped Phase-B
    # protocol and must fail before trusted runtime or Cloud mutation begins.
    result = launcher.main((
        "--release-sha",
        release_sha,
        "--apply-phase-b-foundation",
        "--external-iam-policy-sha256",
        _digest("unrelated-policy"),
    ))
    failure = json.loads(capfd.readouterr().out)
    assert result == 2
    assert failure["ok"] is False
    assert failure["error_code"] == "phase_b_owner_cli_invalid"


def test_pinned_transport_identity_can_be_rechecked_without_gaining_cloud_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = object.__new__(launcher.GcloudOwnerAccessToken)
    identity._pinned_account = None
    identity.owner_subject_sha256 = None
    identity._approved = False
    account = "owner@example.com"
    monkeypatch.setattr(identity, "_active_account", lambda: account)

    assert identity.account_for_read_only_preflight() == account
    identity.require_stable()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="approved_owner_identity_unbound",
    ):
        _ = identity.approved_account
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="approved_owner_identity_unbound",
    ):
        identity()


def test_owner_transport_publishes_only_exact_fixed_coordinator_input() -> None:
    release_sha = "a" * 40
    unsigned = {
        "schema": launcher.COORDINATOR_INPUT_PUBLICATION_SCHEMA,
        "ok": True,
        "state": "published",
        "release_sha": release_sha,
        "coordinator_input_sha256": _digest("coordinator-input"),
        "coordinator_input_path": launcher.COORDINATOR_INPUT_PATH,
        "coordinator_input_file_sha256": _digest("coordinator-input-file"),
        "publication_receipt_path": (
            launcher.COORDINATOR_INPUT_PUBLICATION_RECEIPT_PATH
        ),
        "owner_uid": 0,
        "group_gid": 0,
        "mode": "0400",
        "published_at_unix": 1_000,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }

    class Session:
        def __init__(self) -> None:
            self.validated: Mapping[str, Any] | None = None
            self.closed = False

        def read_gate(self) -> Mapping[str, Any]:
            return receipt

        def mark_validated(self, value: Mapping[str, Any]) -> None:
            self.validated = value

        def close(self) -> None:
            self.closed = True

    class Identity:
        def __init__(self) -> None:
            self.checks = 0

        def require_stable(self) -> None:
            self.checks += 1

    session = Session()
    identity = Identity()
    transport = object.__new__(launcher.IapCoordinatorTransport)
    transport._owner_identity = identity
    transport._open = lambda exact_release, command, *, approved: (
        session
        if (
            exact_release == release_sha
            and command == "publish-coordinator-input"
            and approved is False
        )
        else pytest.fail("coordinator publication command drifted")
    )

    assert transport.publish_coordinator_input(release_sha) == receipt
    assert session.validated == receipt
    assert session.closed is True
    assert identity.checks == 1


def test_owner_cli_routes_explicit_coordinator_publication_without_live_or_cloud_sql(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    release_sha = "a" * 40
    receipt = {
        "schema": launcher.COORDINATOR_INPUT_PUBLICATION_SCHEMA,
        "ok": True,
        "state": "published",
    }

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            return ("/trusted/python",)

    identity = object()

    class Transport:
        def publish_coordinator_input(self, exact_release: str) -> Mapping[str, Any]:
            assert exact_release == release_sha
            return receipt

    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: Runtime(),
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda _release: _digest("launcher"),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda _path: None,
    )
    configuration = object()
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: configuration)
    monkeypatch.setattr(
        launcher,
        "GcloudOwnerAccessToken",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        launcher,
        "IapCoordinatorTransport",
        lambda *args, **kwargs: (
            Transport()
            if args == (identity,)
            and isinstance(kwargs.get("gcloud_executable"), Runtime)
            and kwargs.get("gcloud_configuration") is configuration
            and set(kwargs) == {"gcloud_executable", "gcloud_configuration"}
            else pytest.fail("coordinator transport inputs drifted")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "GoogleRestClient",
        lambda *_args: pytest.fail("Cloud SQL must not be opened"),
    )
    monkeypatch.setattr(
        launcher,
        "launch_full_canary",
        lambda **_kwargs: pytest.fail("live launch must not run"),
    )
    monkeypatch.setattr(
        launcher,
        "OwnerDiscordTokenReader",
        lambda: pytest.fail("Discord token must not be read"),
    )

    assert launcher.main((
        "--release-sha",
        release_sha,
        "--publish-coordinator-input",
    )) == 0
    assert json.loads(capfd.readouterr().out) == receipt


def test_trusted_canary_iam_runner_allows_only_exact_read_only_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = "owner@example.com"
    calls: list[tuple[str, ...]] = []

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            return (
                "/trusted/python",
                *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
                "/trusted/google-cloud-sdk/lib/gcloud.py",
            )

    class Configuration:
        def assert_stable(self) -> None:
            return None

    def runner(argv, **_kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=b"[]", stderr=b"")

    identity = object.__new__(launcher.GcloudOwnerAccessToken)
    identity._runner = runner
    identity._gcloud_executable = Runtime()
    identity._gcloud_configuration = Configuration()
    identity._timeout_seconds = 20.0
    identity._pinned_account = account
    identity.owner_subject_sha256 = hashlib.sha256(account.encode()).hexdigest()
    identity._approved = False
    monkeypatch.setattr(identity, "_active_account", lambda: account)
    monkeypatch.setattr(launcher, "_owner_gcloud_environment", lambda *_args: {})
    logical = (
        "gcloud",
        "compute",
        "instances",
        "list",
        "--project=adventico-ai-platform",
        "--format=json",
    )

    assert identity.run_canary_iam_read_only_json(logical) == []
    assert calls and f"--account={account}" in calls[0]
    assert calls[0][-1] == "--quiet"

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="canary_iam_command_forbidden",
    ):
        identity.run_canary_iam_read_only_json((
            "gcloud",
            "compute",
            "instances",
            "delete",
            launcher.VM_NAME,
        ))
    assert len(calls) == 1


def test_owner_cli_routes_explicit_stopped_writer_activation(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    release_sha = "a" * 40
    policy_sha = _digest("policy")
    receipt = {
        "schema": launcher.WRITER_ACTIVATION_OWNER_RECEIPT_SCHEMA,
        "ok": True,
        "state": "writer_activation_verified_stopped",
    }

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            return ("/trusted/python",)

    class Transport:
        def activate(self, exact_release, *, external_iam_policy_sha256):
            assert exact_release == release_sha
            assert external_iam_policy_sha256 == policy_sha
            return receipt

    identity = object()
    configuration = object()
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _release: Runtime(),
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda _runtime, *, release_sha: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda _release: _digest("launcher"),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda _path: None,
    )
    monkeypatch.setattr(launcher, "PinnedGcloudConfiguration", lambda: configuration)
    monkeypatch.setattr(
        launcher,
        "GcloudOwnerAccessToken",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        launcher,
        "IapWriterActivationBridgeTransport",
        lambda *args, **kwargs: (
            Transport()
            if args == (identity,)
            and isinstance(kwargs.get("gcloud_executable"), Runtime)
            and kwargs.get("gcloud_configuration") is configuration
            else pytest.fail("writer activation transport inputs drifted")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "launch_full_canary",
        lambda **_kwargs: pytest.fail("live launch must not run"),
    )
    monkeypatch.setattr(
        launcher,
        "OwnerDiscordTokenReader",
        lambda: pytest.fail("Discord token must not be read"),
    )

    assert launcher.main((
        "--release-sha",
        release_sha,
        "--activate-writer-stopped",
        "--external-iam-policy-sha256",
        policy_sha,
    )) == 0
    assert json.loads(capfd.readouterr().out) == receipt


def test_writer_activation_bridge_invokes_only_packaged_sealed_module() -> None:
    release_sha = "a" * 40
    frame = launcher.WRITER_AUTHORITY_FRAME_MAGIC + struct.pack(">I", 2) + b"{}"
    observed: dict[str, object] = {}
    transport = object.__new__(launcher.IapWriterActivationBridgeTransport)

    def run_remote_input(remote, **kwargs):
        observed["remote"] = tuple(remote)
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            remote,
            0,
            stdout=b'{"ok":true}\n',
            stderr=b"",
        )

    transport._run_remote_input = run_remote_input

    assert transport._run_packaged_json(
        release_sha,
        module=launcher.WRITER_ACTIVATION_BRIDGE_MODULE,
        arguments=("stage-native-authority",),
        account="owner@example.com",
        stdin_frame=frame,
    ) == {"ok": True}
    remote = observed["remote"]
    assert isinstance(remote, tuple)
    assert (
        f"/opt/muncho-canary-releases/{release_sha}/venv/bin/python"
        in remote
    )
    assert launcher.WRITER_ACTIVATION_BRIDGE_MODULE in remote
    assert "-B" in remote and "-I" in remote
    assert not any("muncho-canary-source" in item for item in remote)
    assert observed["kwargs"]["input_bytes"] == frame


def test_writer_activation_bridge_rejects_noncanonical_remote_json() -> None:
    release_sha = "a" * 40
    transport = object.__new__(launcher.IapWriterActivationBridgeTransport)
    transport._run_remote = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        (),
        0,
        stdout=b'{ "ok": true }\n',
        stderr=b"",
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="writer_activation_output_invalid",
    ):
        transport._run_packaged_json(
            release_sha,
            module=launcher.WRITER_ACTIVATION_MODULE,
            arguments=("install-native-plan",),
            account="owner@example.com",
        )


def test_stopped_writer_activation_orchestrates_only_fixed_packaged_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_sha = "a" * 40
    policy_sha = _digest("policy")
    native_plan_sha = _digest("native-plan")
    final_plan_sha = _digest("final-plan")
    native_approval_sha = _digest("native-approval")
    final_approval_sha = _digest("final-approval")
    native_iam_sha = _digest("native-iam")
    final_iam_sha = _digest("final-iam")
    native_observation_sha = _digest("native-observation")
    owner_sha = _digest("owner")
    calls: list[tuple[str, tuple[str, ...], bytes | None]] = []

    class Identity:
        def __init__(self) -> None:
            self.checks = 0

        def account_for_read_only_preflight(self) -> str:
            return "owner@example.com"

        def require_stable(self) -> None:
            self.checks += 1

    identity = Identity()
    transport = object.__new__(launcher.IapWriterActivationBridgeTransport)
    transport._owner_identity = identity

    def run_packaged(
        _release_sha,
        *,
        module,
        arguments,
        account,
        stdin_frame=None,
        **_kwargs,
    ):
        assert _release_sha == release_sha
        assert account == "owner@example.com"
        calls.append((module, tuple(arguments), stdin_frame))
        return {"call": len(calls)}

    transport._run_packaged_json = run_packaged
    transport._validate_install_plan = (
        lambda _value, *, final, release_sha: (
            final_plan_sha if final else native_plan_sha
        )
    )

    def approval_receipt(value):
        scope = value["scope"]
        return value, (
            final_approval_sha if scope == "activation" else native_approval_sha
        )

    transport._approval_receipt = approval_receipt

    def external_receipt(value):
        final = value["source"] == final_approval_sha
        return value, final_iam_sha if final else native_iam_sha, policy_sha

    transport._external_iam_receipt = external_receipt
    transport._validate_authority_stage = lambda *_args, action, **_kwargs: {
        "receipt_sha256": _digest(action)
    }
    transport._validate_install_approval = (
        lambda *_args, scope, **_kwargs: f"/fixed/{scope}/approval.json"
    )
    transport._validate_install_iam = lambda *_args, scope, **_kwargs: {
        "scope": scope
    }
    transport._validate_native_observation = (
        lambda *_args, **_kwargs: native_observation_sha
    )
    transport._validate_final_plan_build = (
        lambda *_args, **_kwargs: final_plan_sha
    )
    transport._validate_read_only_preflight = lambda *_args, **_kwargs: {
        "report_sha256": _digest("preflight")
    }
    transport._validate_terminal_activation = lambda *_args, **_kwargs: {
        "receipt_sha256": _digest("terminal"),
        "completed_at_unix": 1_800_000_000,
    }

    monkeypatch.setattr(
        launcher,
        "build_writer_owner_approval",
        lambda *, scope, plan_sha256, **_kwargs: {
            "scope": scope,
            "plan_sha256": plan_sha256,
            "owner_subject_sha256": owner_sha,
        },
    )
    monkeypatch.setattr(
        launcher,
        "collect_fresh_writer_external_iam",
        lambda *, source_approval_sha256, **_kwargs: {
            "source": source_approval_sha256
        },
    )
    monkeypatch.setattr(
        launcher,
        "build_writer_authority_frame",
        lambda *, action, **_kwargs: action.encode("ascii"),
    )
    monkeypatch.setattr(
        launcher,
        "_writer_authority_frame_sha256",
        lambda frame: _digest(frame.decode("ascii")),
    )

    receipt = transport.activate(
        release_sha,
        external_iam_policy_sha256=policy_sha,
        now=lambda: 1_800_000_000,
    )

    assert receipt["state"] == "writer_activation_verified_stopped"
    assert receipt["services_stopped"] is True
    assert receipt["discord_started"] is False
    assert identity.checks == 2
    assert [(module, arguments[0]) for module, arguments, _frame in calls] == [
        (launcher.WRITER_ACTIVATION_MODULE, "install-native-plan"),
        (launcher.WRITER_ACTIVATION_BRIDGE_MODULE, "stage-native-authority"),
        (launcher.WRITER_ACTIVATION_MODULE, "install-approval"),
        (launcher.WRITER_ACTIVATION_MODULE, "install-external-iam"),
        (launcher.WRITER_ACTIVATION_MODULE, "observe-native"),
        (launcher.WRITER_PLANNER_MODULE, "build-final-plan"),
        (launcher.WRITER_ACTIVATION_MODULE, "install-plan"),
        (launcher.WRITER_ACTIVATION_BRIDGE_MODULE, "replace-final-authority"),
        (launcher.WRITER_ACTIVATION_MODULE, "install-approval"),
        (launcher.WRITER_ACTIVATION_MODULE, "install-external-iam"),
        (launcher.WRITER_ACTIVATION_MODULE, "validate-plan"),
        (launcher.WRITER_ACTIVATION_MODULE, "apply"),
    ]
    assert calls[1][2] == b"stage-native-authority"
    assert calls[7][2] == b"replace-final-authority"
    assert all(
        module
        in {
            launcher.WRITER_ACTIVATION_MODULE,
            launcher.WRITER_ACTIVATION_BRIDGE_MODULE,
            launcher.WRITER_PLANNER_MODULE,
        }
        for module, _arguments, _frame in calls
    )
