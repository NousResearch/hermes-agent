from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.operational_edge_catalog import operation_catalog
from gateway.operational_edge_client import _predispatch_capability_truth_matches
from gateway.operational_edge_protocol import (
    OperationalIntent,
    OperationalCapability,
    OperationalProtocolError,
    OperationalRequest,
    canonical_json_bytes,
    decode_json_object,
    sha256_json,
    sign_envelope,
    verify_envelope,
)
from gateway.operational_edge_service import OperationalEdgePeer, OperationalEdgeService


class _Journal:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, bytes]] = {}
        self.denials: dict[tuple[str, str, str], tuple[str, bytes]] = {}
        self.store_calls = 0
        self.denial_store_calls = 0

    def read(self, key: str, request_sha256: str) -> bytes | None:
        row = self.rows.get(key)
        if row is None:
            return None
        assert row[0] == request_sha256
        return row[1]

    def store(self, key: str, request_sha256: str, response: bytes) -> None:
        self.store_calls += 1
        existing = self.rows.setdefault(key, (request_sha256, response))
        assert existing == (request_sha256, response)

    def read_predispatch_denial(
        self,
        key: str,
        intent_sha256: str,
        capability_state: str,
        request_sha256: str,
    ) -> bytes | None:
        row = self.denials.get((key, intent_sha256, capability_state))
        if row is None:
            return None
        assert row[0] == request_sha256
        return row[1]

    def store_predispatch_denial(
        self,
        key: str,
        intent_sha256: str,
        capability_state: str,
        request_sha256: str,
        response: bytes,
    ) -> None:
        self.denial_store_calls += 1
        existing = self.denials.setdefault(
            (key, intent_sha256, capability_state),
            (request_sha256, response),
        )
        assert existing == (request_sha256, response)


def _service() -> tuple[
    OperationalEdgeService,
    Ed25519PrivateKey,
    Ed25519PrivateKey,
    _Journal,
]:
    receipt_key = Ed25519PrivateKey.generate()
    writer_key = Ed25519PrivateKey.generate()
    journal = _Journal()
    service = object.__new__(OperationalEdgeService)
    service.config = SimpleNamespace(
        domain="bitrix",
        release_revision="a" * 40,
        allowed_read_peer_uids=frozenset({1001}),
        mutation_peer_uid=1001,
        writer_key_id="b" * 64,
        receipt_key_id="c" * 64,
        release_root=Path("/tmp"),
        subprocess_home=Path("/tmp"),
        maximum_output_bytes=1024 * 1024,
    )
    service.operations = {
        "bitrix.crm.lead_add": operation_catalog()["bitrix.crm.lead_add"]
    }
    service.writer_public_key = writer_key.public_key()
    service.receipt_private_key = receipt_key
    service.journal = journal
    return service, receipt_key, writer_key, journal


def _request(*, capability=None) -> OperationalRequest:
    arguments: dict[str, object] = {}
    intent = OperationalIntent(
        operation_id="bitrix.crm.lead_add",
        arguments=arguments,
        arguments_sha256=sha256_json(arguments),
        idempotency_key="canary:bitrix:mutation-denial",
    )
    return OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=1,
        deadline_unix_ms=9_999_999_999_999,
        intent=intent,
        capability=capability,
    )


def test_missing_capability_is_signed_journaled_predispatch_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, receipt_key, _writer_key, journal = _service()
    monkeypatch.setattr(
        "gateway.operational_edge_service.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )
    request = _request()
    peer = OperationalEdgePeer(pid=4321, uid=1001, gid=1002)

    first = service.dispatch(request, peer)
    second = service.dispatch(request, peer)

    assert first == second
    assert journal.denial_store_calls == 1
    assert journal.store_calls == 0
    envelope = decode_json_object(first, maximum=2 * 1024 * 1024)
    payload = verify_envelope(
        envelope,
        key_id="c" * 64,
        public_key=receipt_key.public_key(),
        code="test_signature_invalid",
    )
    assert payload == {
        **payload,
        "schema": "muncho-operational-edge-receipt.v2",
        "operation_id": "bitrix.crm.lead_add",
        "domain": "bitrix",
        "access": "mutation",
        "outcome": "blocked",
        "service_unit": "muncho-operational-edge-bitrix.service",
        "release_revision": "a" * 40,
        "blocker_code": "mutation_capability_required",
        "dispatched": False,
        "executable_started": False,
        "mutation_performed": False,
        "readback_verified": False,
        "secret_material_recorded": False,
    }
    assert payload["request_sha256"] == sha256_json(request.to_mapping())
    assert payload["executable_sha256"] == "0" * 64
    assert payload["return_code"] is None


def test_tampered_predispatch_envelope_never_verifies() -> None:
    service, receipt_key, _writer_key, _journal = _service()
    encoded = service.dispatch(
        _request(),
        OperationalEdgePeer(pid=4321, uid=1001, gid=1002),
    )
    envelope = dict(decode_json_object(encoded, maximum=2 * 1024 * 1024))
    payload = dict(envelope["payload"])
    payload["dispatched"] = True
    envelope["payload"] = payload
    tampered = json.loads(canonical_json_bytes(envelope))

    with pytest.raises(OperationalProtocolError, match="test_signature_invalid"):
        verify_envelope(
            tampered,
            key_id="c" * 64,
            public_key=receipt_key.public_key(),
            code="test_signature_invalid",
        )


def test_present_invalid_capability_has_distinct_signed_predispatch_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, receipt_key, _writer_key, journal = _service()
    monkeypatch.setattr(
        "gateway.operational_edge_service.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not start"),
    )
    invalid = sign_envelope(
        {"not": "an operational capability"},
        key_id="b" * 64,
        private_key=Ed25519PrivateKey.generate(),
    )
    request = _request(capability=invalid)

    encoded = service.dispatch(
        request,
        OperationalEdgePeer(pid=4321, uid=1001, gid=1002),
    )

    assert journal.denial_store_calls == 1
    assert journal.store_calls == 0
    payload = verify_envelope(
        decode_json_object(encoded, maximum=2 * 1024 * 1024),
        key_id="c" * 64,
        public_key=receipt_key.public_key(),
        code="test_signature_invalid",
    )
    assert payload["blocker_code"] == "mutation_capability_invalid"
    assert payload["request_sha256"] == sha256_json(request.to_mapping())
    assert payload["dispatched"] is False
    assert payload["executable_started"] is False
    assert payload["mutation_performed"] is False
    assert payload["executable_sha256"] == "0" * 64


@pytest.mark.parametrize(
    ("blocker_code", "capability_present", "expected"),
    (
        ("mutation_capability_required", False, True),
        ("mutation_capability_required", True, False),
        ("mutation_capability_invalid", True, True),
        ("mutation_capability_invalid", False, False),
        ("some_other_blocker", False, False),
    ),
)
def test_client_preserves_absent_vs_invalid_capability_truth(
    blocker_code: str,
    capability_present: bool,
    expected: bool,
) -> None:
    assert (
        _predispatch_capability_truth_matches(
            blocker_code,
            capability_present=capability_present,
        )
        is expected
    )


def test_denials_do_not_consume_execution_idempotency_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, receipt_key, writer_key, journal = _service()
    peer = OperationalEdgePeer(pid=4321, uid=1001, gid=1002)
    missing = _request()
    invalid = _request(
        capability=sign_envelope(
            {"not": "an operational capability"},
            key_id="b" * 64,
            private_key=Ed25519PrivateKey.generate(),
        )
    )
    now_ms = int(time.time() * 1000)
    capability = OperationalCapability(
        authority_kind="canonical_owner_plan",
        authority_ref="plan:capability-canary:1",
        operation_id=missing.intent.operation_id,
        arguments_sha256=missing.intent.arguments_sha256,
        idempotency_key=missing.intent.idempotency_key,
        issued_at_unix_ms=now_ms - 1_000,
        expires_at_unix_ms=now_ms + 60_000,
    )
    valid = OperationalRequest(
        request_id=str(uuid.uuid4()),
        sequence=2,
        deadline_unix_ms=now_ms + 30_000,
        intent=missing.intent,
        capability=sign_envelope(
            capability.to_mapping(),
            key_id="b" * 64,
            private_key=writer_key,
        ),
    )
    calls: list[list[str]] = []

    def run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            stdout=b'{"status":"OK"}', stderr=b"", returncode=0
        )

    service._argv = lambda _operation, _arguments: (["/sealed/helper"], "d" * 64)
    monkeypatch.setattr("gateway.operational_edge_service.subprocess.run", run)

    service.dispatch(missing, peer)
    service.dispatch(invalid, peer)
    first_execution = service.dispatch(valid, peer)
    replayed_execution = service.dispatch(valid, peer)

    assert first_execution == replayed_execution
    assert calls == [["/sealed/helper"]]
    assert journal.denial_store_calls == 2
    assert journal.store_calls == 1
    payload = verify_envelope(
        decode_json_object(first_execution, maximum=2 * 1024 * 1024),
        key_id="c" * 64,
        public_key=receipt_key.public_key(),
        code="test_signature_invalid",
    )
    assert payload["outcome"] == "succeeded"
    assert payload["dispatched"] is True
    assert payload["mutation_performed"] is True
