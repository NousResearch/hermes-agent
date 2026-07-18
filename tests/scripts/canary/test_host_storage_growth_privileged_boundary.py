from __future__ import annotations

import base64
import copy
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import host_storage_growth as growth
from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_storage_growth as passkey_v2
from scripts.canary import storage_growth_contract as contract
from scripts.canary import storage_growth_evidence as evidence
from scripts.canary import storage_growth_trusted_collector as collector
from tests.scripts.canary.test_host_storage_growth import (
    _pending_report,
    _source_report,
)
from tests.scripts.canary.test_storage_growth_trusted_collector import (
    CloudReader,
    HostReader,
    _config,
)


RELEASE = "a" * 40
REQUEST = "R" * 32
NOW = 1_800_000_001


def _response(operation: str, document: Mapping[str, Any]) -> bytes:
    unsigned = {
        "schema": passkey_v2.REMOTE_RESPONSE_SCHEMA,
        "operation": operation,
        "release_sha": RELEASE,
        "ok": True,
        "document": dict(document),
    }
    return protocol.canonical_json_bytes({
        **unsigned,
        "response_sha256": protocol.sha256_json(unsigned),
    })


def _request(
    transaction_id: str,
    *,
    checkpoint: str = "source",
    prior_head: str = protocol.GENESIS_JOURNAL_HEAD_SHA256,
    sequence: int = 1,
    issued_at: int = NOW,
) -> Mapping[str, Any]:
    binding = evidence.observation_request_binding_sha256(
        transaction_id=transaction_id,
        checkpoint=checkpoint,
        prior_event_head_sha256=prior_head,
        release_sha=RELEASE,
        plan_sha256=contract.canonical_plan_sha256(),
    )
    context = {
        "schema": "muncho-storage-growth-collection-context.v1",
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
        "prior_event_head_sha256": prior_head,
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    identity = {
        "schema": "muncho-storage-growth-collection-attempt-id.v1",
        "context_sha256": protocol.sha256_json(context),
        "context_sequence": sequence,
        "issued_at_unix": issued_at,
    }
    unsigned = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
        "canonical_state": f"await_{checkpoint}",
        "prior_event_head_sha256": prior_head,
        "request_binding_sha256": binding,
        "observation_nonce_sha256": evidence.observation_nonce_sha256(
            request_binding_sha256=binding,
            transaction_id=transaction_id,
            checkpoint=checkpoint,
        ),
        "collection_attempt_id": protocol.sha256_json(identity),
        "collection_attempt_sequence": sequence,
        "collection_attempt_issued_at_unix": issued_at,
        "collection_attempt_expires_at_unix": (
            issued_at + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
        ),
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    return {
        **unsigned,
        "observation_request_sha256": protocol.sha256_json(unsigned),
    }


def _terminal_request(transaction_id: str) -> Mapping[str, Any]:
    unsigned = {
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": transaction_id,
        "checkpoint": None,
        "canonical_state": "terminal",
        "prior_event_head_sha256": "9" * 64,
        "request_binding_sha256": None,
        "observation_nonce_sha256": None,
        "collection_attempt_id": None,
        "collection_attempt_sequence": None,
        "collection_attempt_issued_at_unix": None,
        "collection_attempt_expires_at_unix": None,
        "release_sha": RELEASE,
        "plan_sha256": contract.canonical_plan_sha256(),
    }
    return {
        **unsigned,
        "observation_request_sha256": protocol.sha256_json(unsigned),
    }


class _HostAttestor:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        report: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.report = report
        self.calls = 0

    def attest_host_observation(
        self, canonical_frame: bytes
    ) -> Mapping[str, Any]:
        self.calls += 1
        frame = protocol.decode_canonical_json(canonical_frame)
        return collector.run_host_attestor(
            frame,
            config=self.config,
            facts_reader=HostReader(self.report),
            now_unix=NOW,
        )


class _Transport:
    def __init__(
        self,
        tmp_path: Path,
        *,
        transaction_id: str,
        report: Mapping[str, Any],
        checkpoint: str = "source",
        terminal: bool = False,
        rotate_request_after_cloud: bool = False,
    ) -> None:
        self.transaction_id = transaction_id
        self.report = report
        self.terminal = terminal
        self.rotate_request_after_cloud = rotate_request_after_cloud
        self.cloud_key = Ed25519PrivateKey.generate()
        self.host_key = Ed25519PrivateKey.generate()
        self.cloud_config = _config(tmp_path, "cloud", self.cloud_key)
        self.host_config = _config(tmp_path, "host", self.host_key)
        self.request = (
            _terminal_request(transaction_id)
            if terminal
            else _request(transaction_id, checkpoint=checkpoint)
        )
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.cloud_signed = False

    def _ready(self) -> Mapping[str, Any]:
        def key(private: Ed25519PrivateKey) -> Mapping[str, Any]:
            raw = private.public_key().public_bytes_raw()
            return {
                "public_key_id": protocol.sha256_bytes(raw),
                "public_key_b64url": base64.urlsafe_b64encode(raw).rstrip(
                    b"="
                ).decode("ascii"),
            }

        return {
            "owner_gate_vm_name": passkey_v2.OWNER_GATE_VM_NAME,
            "web_uid": passkey_v2.OWNER_GATE_WEB_UID,
            "authority_uid": passkey_v2.OWNER_GATE_AUTHORITY_UID,
            "executor_uid": passkey_v2.OWNER_GATE_EXECUTOR_UID,
            "authority_socket": passkey_v2.AUTHORITY_SOCKET,
            "executor_socket": passkey_v2.EXECUTOR_SOCKET,
            "authority_db": passkey_v2.AUTHORITY_DB,
            "executor_db": passkey_v2.EXECUTOR_DB,
            "rp_id": protocol.PRODUCTION_RP_ID,
            "origin": protocol.PRODUCTION_ORIGIN,
            "iap_only": True,
            "local_compute_mutation_available": False,
            "sqlite_synchronous": "FULL",
            "sqlite_begin_immediate": True,
            "totp_dangerous_actions": False,
            "observation_attestors": {
                "cloud": key(self.cloud_key),
                "host": key(self.host_key),
            },
        }

    def invoke_owner_gate(self, canonical_frame: bytes) -> bytes:
        frame = protocol.decode_canonical_json(canonical_frame)
        operation = str(frame["operation"])
        document = dict(frame["document"])
        self.calls.append((operation, copy.deepcopy(document)))
        if operation == "observation_request":
            if self.rotate_request_after_cloud and self.cloud_signed:
                return _response(
                    operation,
                    _request(
                        self.transaction_id,
                        checkpoint=str(self.request["checkpoint"]),
                        sequence=2,
                        issued_at=NOW + 1,
                    ),
                )
            return _response(operation, self.request)
        if operation == "preflight":
            return _response(operation, self._ready())
        if operation == "attest_cloud_observation":
            attestation_request = document["attestation_request"]
            response = collector.run_cloud_attestor(
                attestation_request,
                config=self.cloud_config,
                facts_reader=CloudReader(self.report),
                now_unix=NOW,
            )
            self.cloud_signed = True
            return _response(operation, {
                "terminal": False,
                "terminal_receipt": None,
                "observation_request": self.request,
                "attestation_response": response,
            })
        if operation in {"request_initial", "request_resume"}:
            bundle_name = (
                "source_preflight"
                if operation == "request_initial"
                else "continuation_preflight"
            )
            bundle = document[bundle_name]
            return _response(operation, {
                "schema": "test-storage-passkey-request.v2",
                "state": "step_up_required",
                "release_sha": RELEASE,
                "plan_sha256": contract.canonical_plan_sha256(),
                "transaction_id": self.transaction_id,
                "attested_observation_bundle_sha256": bundle[
                    "bundle_sha256"
                ],
                "request_id": REQUEST,
                "passkey_only": True,
                "local_mutation_authority": False,
                "opens_runtime_gate": False,
            })
        if operation == "execute_or_recover":
            return _response(operation, {
                "schema": "test-storage-privileged-terminal.v2",
                "state": "terminal_verified",
                "release_sha": RELEASE,
                "plan_sha256": contract.canonical_plan_sha256(),
                "transaction_id": self.transaction_id,
                "request_id": document["request_id"],
                "authoritative_remote_executor": True,
                "local_mutation_performed": False,
                "passkey_method": "passkey",
            })
        if operation == "verify_terminal":
            receipt = {
                "terminal": True,
                "transaction_id": self.transaction_id,
                "release_sha": RELEASE,
                "plan_sha256": contract.canonical_plan_sha256(),
            }
            return _response(operation, {
                "terminal": True,
                "transaction_id": self.transaction_id,
                "release_sha": RELEASE,
                "plan_sha256": contract.canonical_plan_sha256(),
                "terminal_receipt": receipt,
            })
        raise AssertionError(f"unexpected operation: {operation}")


def _boundary(transport: _Transport) -> passkey_v2.ProductionStorageGrowthBoundary:
    return passkey_v2.ProductionStorageGrowthBoundary(RELEASE, transport)


def _forbid_local_subprocess(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError(
        "storage growth must never mutate through local subprocess"
    )


def test_initial_request_dual_attests_exact_source_with_real_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_report(collected_at=NOW - 1)
    transaction = passkey_v2.transaction_id_for_observation(source)
    transport = _Transport(
        tmp_path, transaction_id=transaction, report=source
    )
    host = _HostAttestor(config=transport.host_config, report=source)
    monkeypatch.setattr(launcher.subprocess, "run", _forbid_local_subprocess)

    receipt = launcher.author_storage_growth_owner_approval(
        release_sha=RELEASE,
        gcloud_executable=object(),
        gcloud_configuration=object(),
        owner_identity=object(),
        now_unix=NOW,
        privileged_boundary=_boundary(transport),
        source_observer=lambda: source,
        host_attestor=host,
    )

    assert receipt["request_id"] == REQUEST
    assert host.calls == 1
    assert [name for name, _ in transport.calls] == [
        "observation_request",
        "preflight",
        "attest_cloud_observation",
        "observation_request",
        "request_initial",
    ]
    initial = transport.calls[-1][1]
    assert initial["transaction_id"] == transaction
    assert initial["source_preflight"]["cloud_attestation"] is not None
    assert initial["source_preflight"]["host_attestation"] is not None


def test_apply_supplies_stable_consume_id_and_attested_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = "b" * 64
    source = _source_report(collected_at=NOW - 1)
    transport = _Transport(
        tmp_path, transaction_id=transaction, report=source
    )
    host = _HostAttestor(config=transport.host_config, report=source)
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda *_args, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(launcher.subprocess, "run", _forbid_local_subprocess)

    receipt = launcher.apply_storage_growth_owner_gate(
        release_sha=RELEASE,
        transaction_id=transaction,
        request_id=REQUEST,
        gcloud_executable=object(),
        gcloud_configuration=object(),
        owner_identity=object(),
        privileged_boundary=_boundary(transport),
        now_unix=NOW,
        continuation_observer=lambda: source,
        host_attestor=host,
    )

    assert receipt["authoritative_remote_executor"] is True
    execution = transport.calls[-1][1]
    assert execution["consume_attempt_id"] == (
        launcher._storage_growth_consume_attempt_id(
            transaction_id=transaction, request_id=REQUEST
        )
    )
    assert execution["continuation_preflight"]["bundle_sha256"]
    assert [name for name, _ in transport.calls][-1] == "execute_or_recover"


def test_terminal_short_circuit_never_touches_readiness_or_attestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = "b" * 64
    source = _source_report(collected_at=NOW - 1)
    transport = _Transport(
        tmp_path,
        transaction_id=transaction,
        report=source,
        terminal=True,
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda *_args, **_kwargs: "d" * 64,
    )

    receipt = launcher.apply_storage_growth_owner_gate(
        release_sha=RELEASE,
        transaction_id=transaction,
        request_id=REQUEST,
        gcloud_executable=object(),
        gcloud_configuration=object(),
        owner_identity=object(),
        privileged_boundary=_boundary(transport),
        continuation_observer=lambda: (_ for _ in ()).throw(
            AssertionError("terminal path collected observation")
        ),
        host_attestor=object(),
    )

    assert receipt["terminal"] is True
    assert [name for name, _ in transport.calls] == [
        "observation_request",
        "verify_terminal",
    ]


def test_resume_request_carries_dual_attested_continuation(
    tmp_path: Path,
) -> None:
    transaction = "b" * 64
    pending = _pending_report(collected_at=NOW - 1)
    transport = _Transport(
        tmp_path,
        transaction_id=transaction,
        report=pending,
        checkpoint="post_resize",
    )
    host = _HostAttestor(config=transport.host_config, report=pending)

    receipt = launcher.author_storage_growth_resume_approval(
        release_sha=RELEASE,
        transaction_id=transaction,
        gcloud_executable=object(),
        gcloud_configuration=object(),
        owner_identity=object(),
        now_unix=NOW,
        privileged_boundary=_boundary(transport),
        continuation_observer=lambda: pending,
        host_attestor=host,
    )

    assert receipt["request_id"] == REQUEST
    request = transport.calls[-1][1]
    assert request["continuation_preflight"]["checkpoint"] == "post_resize"
    assert request["continuation_preflight"]["host_attestation"] is not None


def test_refreshed_request_rotation_after_signing_fails_closed(
    tmp_path: Path,
) -> None:
    source = _source_report(collected_at=NOW - 1)
    transaction = passkey_v2.transaction_id_for_observation(source)
    transport = _Transport(
        tmp_path,
        transaction_id=transaction,
        report=source,
        rotate_request_after_cloud=True,
    )
    host = _HostAttestor(config=transport.host_config, report=source)
    with pytest.raises(
        launcher.OwnerLauncherError, match="trusted_observation_failed"
    ):
        launcher.author_storage_growth_owner_approval(
            release_sha=RELEASE,
            gcloud_executable=object(),
            gcloud_configuration=object(),
            owner_identity=object(),
            now_unix=NOW,
            privileged_boundary=_boundary(transport),
            source_observer=lambda: source,
            host_attestor=host,
        )


@pytest.mark.parametrize(
    "request_id",
    ["../escape", "short", "A" * 65, "A" * 31, "A" * 32 + "/"],
)
def test_apply_rejects_non_token_request_ids_before_boundary(
    monkeypatch: pytest.MonkeyPatch,
    request_id: str,
) -> None:
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda *_args, **_kwargs: "d" * 64,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="storage_growth_passkey_request_invalid",
    ):
        launcher.apply_storage_growth_owner_gate(
            release_sha=RELEASE,
            transaction_id="b" * 64,
            request_id=request_id,
            gcloud_executable=object(),
            gcloud_configuration=object(),
            owner_identity=object(),
            privileged_boundary=object(),
        )


def test_cli_requires_request_id_only_for_apply() -> None:
    parser = launcher._cli_parser()
    apply_missing = parser.parse_args([
        "--release-sha", RELEASE,
        "--apply-storage-growth",
        "--storage-growth-transaction-id", "b" * 64,
    ])
    author_with_request = parser.parse_args([
        "--release-sha", RELEASE,
        "--author-storage-growth",
        "--storage-growth-passkey-request-id", REQUEST,
    ])
    apply_exact = parser.parse_args([
        "--release-sha", RELEASE,
        "--apply-storage-growth",
        "--storage-growth-transaction-id", "b" * 64,
        "--storage-growth-passkey-request-id", REQUEST,
    ])

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="storage_growth_passkey_request_invalid",
    ):
        launcher._validate_storage_growth_cli_arguments(apply_missing)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="storage_growth_passkey_request_invalid",
    ):
        launcher._validate_storage_growth_cli_arguments(author_with_request)
    launcher._validate_storage_growth_cli_arguments(apply_exact)
