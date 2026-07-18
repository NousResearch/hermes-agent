from __future__ import annotations

import base64
import json
import os
import socket
import sqlite3
import sys
import tempfile
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

from cryptography.hazmat.primitives.asymmetric import ed25519

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import passkey_v2_sqlite as database
from scripts.canary import passkey_v2_storage_growth as storage
from scripts.canary import storage_growth_evidence as evidence
from scripts.canary import storage_growth_trusted_collector as trusted_collector
from scripts.canary.passkey_v2_signer import ReceiptSigner
from tests.scripts.canary.test_host_storage_growth import (
    _pending_report,
    _source_report,
    _target_report,
)
from tests.scripts.canary.test_storage_growth_trusted_collector import (
    _trusted_iam_projection,
)
from tests.scripts.canary.test_passkey_v2_security import (
    NOW,
    _challenge,
    _credential_and_assertion,
)


RELEASE = "a" * 40
SHA = "b" * 64


def _bundle(
    report: Mapping[str, Any],
    *,
    transaction_id: str,
    checkpoint: str,
    prior_event_head_sha256: str,
    cloud_key: ed25519.Ed25519PrivateKey,
    host_key: ed25519.Ed25519PrivateKey,
    observation_request: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    request_binding = evidence.observation_request_binding_sha256(
        transaction_id=transaction_id,
        checkpoint=checkpoint,
        prior_event_head_sha256=prior_event_head_sha256,
        release_sha=RELEASE,
        plan_sha256=storage.exact_storage_plan()["plan_sha256"],
    )
    if observation_request is None:
        context = {
            "schema": "muncho-storage-growth-collection-context.v1",
            "transaction_id": transaction_id,
            "checkpoint": checkpoint,
            "prior_event_head_sha256": prior_event_head_sha256,
            "release_sha": RELEASE,
            "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
        }
        identity = {
            "schema": "muncho-storage-growth-collection-attempt-id.v1",
            "context_sha256": protocol.sha256_json(context),
            "context_sequence": 1,
            "issued_at_unix": NOW + 3,
        }
        unsigned_request = {
            "schema": "muncho-storage-growth-observation-request.v1",
            "transaction_id": transaction_id,
            "checkpoint": checkpoint,
            "canonical_state": f"await_{checkpoint}",
            "prior_event_head_sha256": prior_event_head_sha256,
            "request_binding_sha256": request_binding,
            "observation_nonce_sha256": evidence.observation_nonce_sha256(
                request_binding_sha256=request_binding,
                transaction_id=transaction_id,
                checkpoint=checkpoint,
            ),
            "collection_attempt_id": protocol.sha256_json(identity),
            "collection_attempt_sequence": 1,
            "collection_attempt_issued_at_unix": NOW + 3,
            "collection_attempt_expires_at_unix": (
                NOW + 3 + evidence.OBSERVATION_BUNDLE_TTL_SECONDS
            ),
            "release_sha": RELEASE,
            "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
        }
        observation_request = {
            **unsigned_request,
            "observation_request_sha256": protocol.sha256_json(
                unsigned_request
            ),
        }
    assert observation_request["transaction_id"] == transaction_id
    assert observation_request["checkpoint"] == checkpoint
    assert (
        observation_request["prior_event_head_sha256"]
        == prior_event_head_sha256
    )
    return evidence.build_attested_observation(
        observation=report,
        transaction_id=transaction_id,
        checkpoint=checkpoint,
        prior_event_head_sha256=prior_event_head_sha256,
        request_binding_sha256=request_binding,
        observation_nonce_sha256_value=evidence.observation_nonce_sha256(
            request_binding_sha256=request_binding,
            transaction_id=transaction_id,
            checkpoint=checkpoint,
        ),
        observation_request_sha256=observation_request[
            "observation_request_sha256"
        ],
        collection_attempt_id=observation_request[
            "collection_attempt_id"
        ],
        collection_attempt_sequence=observation_request[
            "collection_attempt_sequence"
        ],
        collection_attempt_issued_at_unix=observation_request[
            "collection_attempt_issued_at_unix"
        ],
        collection_attempt_expires_at_unix=observation_request[
            "collection_attempt_expires_at_unix"
        ],
        trusted_iam_projection=_trusted_iam_projection(
            report,
            observation_request,
            now_unix=max(
                observation_request[
                    "collection_attempt_issued_at_unix"
                ],
                report["collected_at_unix"],
            ),
        ),
        cloud_public_key_id=protocol.sha256_bytes(
            cloud_key.public_key().public_bytes_raw()
        ),
        cloud_signer=cloud_key.sign,
        host_public_key_id=protocol.sha256_bytes(
            host_key.public_key().public_bytes_raw()
        ),
        host_signer=host_key.sign,
    )


def _authority_and_executor(
    tmp_path: Path,
    *,
    cloud_key: ed25519.Ed25519PrivateKey,
    host_key: ed25519.Ed25519PrivateKey,
) -> tuple[
    database.PasskeyV2ExecutorDatabase,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    source = _source_report(collected_at=NOW)
    transaction_id = storage.transaction_id_for_observation(source)
    source_bundle = _bundle(
        source,
        transaction_id=transaction_id,
        checkpoint="source",
        prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        cloud_key=cloud_key,
        host_key=host_key,
    )
    action = storage.build_storage_growth_envelope(
        source_preflight=source_bundle,
        transaction_id=transaction_id,
        stage="intent",
        release_sha=RELEASE,
        authority_manifest_sha256="c" * 64,
        authority_host_receipt_sha256="d" * 64,
        prior_authoritative_receipt_sha256=(
            protocol.GENESIS_JOURNAL_HEAD_SHA256
        ),
        prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        issued_at_unix=NOW,
    )

    authority_root = tmp_path / "authority"
    authority_root.mkdir(mode=0o700)
    authority_root.chmod(0o700)
    authority_state = authority_root.stat()
    authority_path = authority_root / "passkey-v2.sqlite3"
    database.bootstrap_authority_database(
        authority_path,
        authority_uid=authority_state.st_uid,
        authority_gid=authority_state.st_gid,
        now_unix=NOW - 1,
        require_root=False,
    )
    authority = database.PasskeyV2AuthorityDatabase(
        authority_path,
        authority_uid=authority_state.st_uid,
        authority_gid=authority_state.st_gid,
    )
    challenge = _challenge(action)
    credential, assertion = _credential_and_assertion(action, challenge)
    authority.import_migrated_credential(credential)
    authority.create_request(action)
    authority.create_challenge(challenge, envelope=action)
    grant = authority.verify_assertion_and_record_grant(
        assertion=assertion,
        envelope=action,
        challenge=challenge,
        grant_id="G" * 32,
        now_unix=NOW + 1,
    )
    receipt_signer = ReceiptSigner(ed25519.Ed25519PrivateKey.generate())
    consumed = authority.consume_or_replay(
        envelope=action,
        runtime_binding=protocol.build_runtime_binding(
            executor_release_sha=RELEASE,
            executor_plan_sha256=storage.exact_storage_plan()["plan_sha256"],
            executor_binary_sha256="1" * 64,
            mutation_wrapper_sha256="2" * 64,
            remote_transport_sha256="3" * 64,
        ),
        consume_attempt_id="4" * 64,
        signer=receipt_signer,
        now_unix=NOW + 2,
    )

    executor_root = tmp_path / "executor"
    executor_root.mkdir(mode=0o700)
    executor_root.chmod(0o700)
    executor_state = executor_root.stat()
    executor_path = executor_root / "execution-v2.sqlite3"
    database.bootstrap_executor_database(
        executor_path,
        executor_uid=executor_state.st_uid,
        executor_gid=executor_state.st_gid,
        now_unix=NOW - 1,
        require_root=False,
    )
    executor = database.PasskeyV2ExecutorDatabase(
        executor_path,
        executor_uid=executor_state.st_uid,
        executor_gid=executor_state.st_gid,
        pinned_authority_receipt_public_key=receipt_signer.public_key,
        pinned_authority_receipt_key_id=receipt_signer.key_id,
    )
    return executor, action, challenge, grant, consumed.receipt


def _projection(*, disk_size: int, instance_status: str = "RUNNING") -> Mapping[str, Any]:
    unsigned = {
        "schema": "muncho-passkey-v2-compute-live-projection.v1",
        "project": storage.PROJECT,
        "zone": storage.ZONE,
        "disk": {
            "id": storage.DISK_ID,
            "name": storage.DISK_NAME,
            "size_gb": disk_size,
            "status": "READY",
            "zone": storage.ZONE,
            "type": "pd-balanced",
        },
        "instance": {
            "id": storage.VM_INSTANCE_ID,
            "name": storage.VM_NAME,
            "status": instance_status,
            "zone": storage.ZONE,
            "boot_device_name": storage.BOOT_DEVICE_NAME,
            "disk_name": storage.DISK_NAME,
        },
    }
    return {**unsigned, "projection_sha256": protocol.sha256_json(unsigned)}


class InjectedCrash(RuntimeError):
    pass


class StaticOwnerGateTransport:
    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = dict(document)

    def invoke_owner_gate(self, raw: bytes) -> bytes:
        request = protocol.decode_canonical_json(raw)
        assert isinstance(request, Mapping)
        unsigned = {
            "schema": storage.REMOTE_RESPONSE_SCHEMA,
            "operation": request["operation"],
            "release_sha": request["release_sha"],
            "ok": True,
            "document": self.document,
        }
        return protocol.canonical_json_bytes({
            **unsigned,
            "response_sha256": protocol.sha256_json(unsigned),
        })


class CrashAfterResizeClient:
    def __init__(self) -> None:
        self.disk_size = storage.SOURCE_SIZE_GB
        self.resize_calls: list[str] = []

    def observe(self) -> Mapping[str, Any]:
        return _projection(disk_size=self.disk_size)

    def resize(self, request_id: str) -> None:
        self.resize_calls.append(request_id)
        self.disk_size = storage.TARGET_SIZE_GB
        raise InjectedCrash("after provider accepted resize")

    def wait_disk_80(self) -> Mapping[str, Any]:
        return self.observe()

    def stop(self, _request_id: str) -> None:
        raise AssertionError("online completion must not stop")

    def start(self, _request_id: str) -> None:
        raise AssertionError("online completion must not start")

    def wait_terminated(self) -> Mapping[str, Any]:
        raise AssertionError("online completion must not stop")

    def wait_running(self) -> Mapping[str, Any]:
        return self.observe()


class RebootRequiredClient:
    def __init__(self) -> None:
        self.disk_size = storage.SOURCE_SIZE_GB
        self.instance_status = "RUNNING"

    def observe(self) -> Mapping[str, Any]:
        return _projection(
            disk_size=self.disk_size,
            instance_status=self.instance_status,
        )

    def resize(self, _request_id: str) -> None:
        self.disk_size = storage.TARGET_SIZE_GB

    def wait_disk_80(self) -> Mapping[str, Any]:
        return self.observe()

    def stop(self, _request_id: str) -> None:
        self.instance_status = "TERMINATED"

    def wait_terminated(self) -> Mapping[str, Any]:
        return self.observe()

    def start(self, _request_id: str) -> None:
        self.instance_status = "RUNNING"

    def wait_running(self) -> Mapping[str, Any]:
        return self.observe()


class DirectExecutorClient:
    def __init__(
        self,
        *,
        executor: database.PasskeyV2ExecutorDatabase,
        mutation: service.StorageGrowthComputeExecutor,
        verifier: Any,
        now_unix: int,
    ) -> None:
        self.executor = executor
        self.mutation = mutation
        self.verifier = verifier
        self.now_unix = now_unix
        self.operations: list[str] = []

    def call(
        self, operation: str, document: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.operations.append(operation)
        return service.handle_executor_frame(
            service.build_service_frame(operation, document),
            executor=self.executor,
            peer_uid=service.AUTHORITY_UID,
            release_revision=RELEASE,
            now_unix=self.now_unix,
            mutation_handler=self.mutation,
            readiness_handler=lambda: (_ for _ in ()).throw(
                AssertionError("read-only reconciliation touched readiness")
            ),
            observation_verifier=self.verifier,
        )["document"]


class ForbiddenAuthorityClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(
        self, operation: str, _document: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append(operation)
        raise AssertionError("read-only reconciliation touched authority")


class RecordingAuthorityClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def call(
        self, operation: str, document: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append((operation, dict(document)))
        return {"request_created": True}


def _remote_execute_frame(
    *,
    action: Mapping[str, Any],
    continuation: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": storage.REMOTE_FRAME_SCHEMA,
        "operation": "execute_or_recover",
        "release_sha": RELEASE,
        "document": {
            "request_id": action["request_id"],
            "consume_attempt_id": "4" * 64,
            "transaction_id": action["transaction_id"],
            "continuation_preflight": dict(continuation),
        },
    }
    return {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}


def _observation_verifier(
    cloud_key: ed25519.Ed25519PrivateKey,
    host_key: ed25519.Ed25519PrivateKey,
):
    def verify(
        bundle: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return evidence.validate_attested_observation(
            bundle,
            cloud_public_key=cloud_key.public_key(),
            cloud_public_key_id=protocol.sha256_bytes(
                cloud_key.public_key().public_bytes_raw()
            ),
            host_public_key=host_key.public_key(),
            host_public_key_id=protocol.sha256_bytes(
                host_key.public_key().public_bytes_raw()
            ),
            now_unix=int(bundle["observed_at_unix"]) + 1,
            allowed_states=frozenset({
                "source_ready", "resize_complete_boot_required",
                "terminated_after_growth_intent", "target_ready",
            }),
            expected_transaction_id=expected["transaction_id"],
            expected_checkpoint=expected["checkpoint"],
            expected_request_binding_sha256=expected[
                "request_binding_sha256"
            ],
            expected_prior_event_head_sha256=expected[
                "prior_event_head_sha256"
            ],
            expected_observation_request_sha256=expected[
                "observation_request_sha256"
            ],
            expected_collection_attempt_id=expected[
                "collection_attempt_id"
            ],
            expected_collection_attempt_sequence=expected[
                "collection_attempt_sequence"
            ],
            expected_collection_attempt_issued_at_unix=expected[
                "collection_attempt_issued_at_unix"
            ],
            expected_collection_attempt_expires_at_unix=expected[
                "collection_attempt_expires_at_unix"
            ],
        )

    return verify


def test_attested_observation_binds_checkpoint_nonce_and_both_signatures() -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    source = _source_report(collected_at=NOW)
    transaction_id = storage.transaction_id_for_observation(source)
    bundle = _bundle(
        source,
        transaction_id=transaction_id,
        checkpoint="source",
        prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        cloud_key=cloud,
        host_key=host,
    )
    expected_binding = evidence.observation_request_binding_sha256(
        transaction_id=transaction_id,
        checkpoint="source",
        prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        release_sha=RELEASE,
        plan_sha256=storage.exact_storage_plan()["plan_sha256"],
    )
    checked = evidence.validate_attested_observation(
        bundle,
        cloud_public_key=cloud.public_key(),
        cloud_public_key_id=protocol.sha256_bytes(
            cloud.public_key().public_bytes_raw()
        ),
        host_public_key=host.public_key(),
        host_public_key_id=protocol.sha256_bytes(
            host.public_key().public_bytes_raw()
        ),
        now_unix=NOW + 1,
        allowed_states=frozenset({"source_ready"}),
        expected_transaction_id=transaction_id,
        expected_checkpoint="source",
        expected_request_binding_sha256=expected_binding,
        expected_prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        expected_observation_request_sha256=bundle[
            "observation_request_sha256"
        ],
        expected_collection_attempt_id=bundle[
            "collection_attempt_id"
        ],
        expected_collection_attempt_sequence=bundle[
            "collection_attempt_sequence"
        ],
        expected_collection_attempt_issued_at_unix=bundle[
            "collection_attempt_issued_at_unix"
        ],
        expected_collection_attempt_expires_at_unix=bundle[
            "collection_attempt_expires_at_unix"
        ],
    )
    assert checked["state"] == "source_ready"
    tampered = dict(bundle)
    tampered["checkpoint"] = "post_resize"
    tampered["bundle_sha256"] = protocol.sha256_json({
        key: item for key, item in tampered.items() if key != "bundle_sha256"
    })
    with pytest.raises(evidence.StorageGrowthEvidenceError):
        evidence.validate_attested_observation(
            tampered,
            cloud_public_key=cloud.public_key(),
            cloud_public_key_id=protocol.sha256_bytes(
                cloud.public_key().public_bytes_raw()
            ),
            host_public_key=host.public_key(),
            host_public_key_id=protocol.sha256_bytes(
                host.public_key().public_bytes_raw()
            ),
            now_unix=NOW + 1,
            allowed_states=frozenset({"source_ready"}),
            expected_transaction_id=transaction_id,
            expected_checkpoint="source",
            expected_request_binding_sha256=expected_binding,
            expected_prior_event_head_sha256=(
                protocol.GENESIS_JOURNAL_HEAD_SHA256
            ),
            expected_observation_request_sha256=bundle[
                "observation_request_sha256"
            ],
            expected_collection_attempt_id=bundle[
                "collection_attempt_id"
            ],
            expected_collection_attempt_sequence=bundle[
                "collection_attempt_sequence"
            ],
            expected_collection_attempt_issued_at_unix=bundle[
                "collection_attempt_issued_at_unix"
            ],
            expected_collection_attempt_expires_at_unix=bundle[
                "collection_attempt_expires_at_unix"
            ],
        )


def test_compute_rest_uses_metadata_header_and_durable_request_id() -> None:
    calls: list[tuple[str, str, str]] = []

    def requester(
        host: str,
        method: str,
        path: str,
        _headers: Mapping[str, str],
        _body: bytes | None,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        calls.append((host, method, path))
        if host == "169.254.169.254":
            return 200, json.dumps({
                "access_token": "x" * 32,
                "expires_in": 300,
                "token_type": "Bearer",
            }, separators=(",", ":"), sort_keys=True).encode(), {
                "metadata-flavor": "Google"
            }
        return 200, b'{"status":"PENDING"}', {}

    client = service.FixedComputeRestClient(requester=requester)
    request_id = service.StorageGrowthComputeExecutor._gce_request_id(
        "c" * 64, "resize"
    )
    client.resize(request_id)
    assert calls[-1][2].endswith(
        f"/disks/{storage.DISK_NAME}/resize?requestId={request_id}"
    )

    def missing_header(*args: Any) -> tuple[int, bytes, Mapping[str, str]]:
        return requester(*args)[:2] + ({},)

    with pytest.raises(service.PasskeyV2ServiceError, match="token_unavailable"):
        service.FixedComputeRestClient(requester=missing_header)._token()


def test_compute_https_uses_only_fixed_descriptor_loaded_debian_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    captured: dict[str, Any] = {}

    class Response:
        status = 200

        @staticmethod
        def read(_maximum: int) -> bytes:
            return b"{}"

        @staticmethod
        def getheaders() -> list[tuple[str, str]]:
            return []

    class Connection:
        def __init__(
            self,
            host: str,
            port: int,
            *,
            timeout: int,
            context: object,
        ) -> None:
            captured.update({
                "host": host,
                "port": port,
                "timeout": timeout,
                "context": context,
            })

        @staticmethod
        def request(
            method: str,
            path: str,
            *,
            body: bytes | None,
            headers: Mapping[str, str],
        ) -> None:
            captured.update({
                "method": method,
                "path": path,
                "body": body,
                "headers": dict(headers),
            })

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(
        service.trusted_collector,
        "fixed_debian_tls_context",
        lambda: marker,
    )
    monkeypatch.setattr(service.http.client, "HTTPSConnection", Connection)
    prefix = (
        f"/compute/v1/projects/{storage.PROJECT}/zones/{storage.ZONE}/"
    )
    status, raw, headers = service.FixedComputeRestClient._request(
        "compute.googleapis.com",
        "GET",
        prefix + f"disks/{storage.DISK_NAME}",
        {"Authorization": "Bearer redacted"},
        None,
    )
    assert (status, raw, headers) == (200, b"{}", {})
    assert captured["context"] is marker
    assert captured["host"] == "compute.googleapis.com"


def test_compute_https_maps_fixed_trust_failure_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> Any:
        raise trusted_collector.TrustedObservationError("injected")

    monkeypatch.setattr(
        service.trusted_collector, "fixed_debian_tls_context", fail
    )
    prefix = (
        f"/compute/v1/projects/{storage.PROJECT}/zones/{storage.ZONE}/"
    )
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_compute_tls_invalid",
    ):
        service.FixedComputeRestClient._request(
            "compute.googleapis.com",
            "GET",
            prefix + f"disks/{storage.DISK_NAME}",
            {},
            None,
        )


def test_executor_cloud_attestor_binds_canonical_request_without_readiness(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    request = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=NOW + 3,
    )
    attestation_request = trusted_collector.build_attestation_request(
        request,
        _source_report(collected_at=NOW + 3),
        role="cloud",
        now_unix=NOW + 3,
    )
    calls: list[Mapping[str, Any]] = []

    def attest(
        frame: Mapping[str, Any], *, now_unix: int
    ) -> Mapping[str, Any]:
        assert now_unix == NOW + 3
        calls.append(frame)
        return {
            "role": "cloud",
            "observation_request_sha256": request[
                "observation_request_sha256"
            ],
        }

    before = executor.read_transaction_state(action["transaction_id"])
    result = service.handle_executor_frame(
        service.build_service_frame(
            "attest_cloud_observation",
            {
                "transaction_id": action["transaction_id"],
                "attestation_request": attestation_request,
            },
        ),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 3,
        mutation_handler=lambda *_args: (_ for _ in ()).throw(
            AssertionError("attestor touched mutation")
        ),
        readiness_handler=lambda: (_ for _ in ()).throw(
            AssertionError("attestor touched readiness")
        ),
        observation_verifier=lambda *_args: (_ for _ in ()).throw(
            AssertionError("attestor used bundle verifier")
        ),
        cloud_attestor=attest,
    )["document"]
    after = executor.read_transaction_state(action["transaction_id"])
    assert result["terminal"] is False
    assert result["attestation_response"]["role"] == "cloud"
    assert calls == [attestation_request]
    assert before["events"] == after["events"]


def test_production_boundary_requires_exact_pinned_attestor_keys() -> None:
    cloud = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    host = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()

    def record(raw: bytes) -> Mapping[str, str]:
        return {
            "public_key_id": protocol.sha256_bytes(raw),
            "public_key_b64url": base64.urlsafe_b64encode(raw)
            .rstrip(b"=")
            .decode("ascii"),
        }

    ready = {
        "owner_gate_vm_name": storage.OWNER_GATE_VM_NAME,
        "web_uid": storage.OWNER_GATE_WEB_UID,
        "authority_uid": storage.OWNER_GATE_AUTHORITY_UID,
        "executor_uid": storage.OWNER_GATE_EXECUTOR_UID,
        "authority_socket": storage.AUTHORITY_SOCKET,
        "executor_socket": storage.EXECUTOR_SOCKET,
        "authority_db": storage.AUTHORITY_DB,
        "executor_db": storage.EXECUTOR_DB,
        "rp_id": protocol.PRODUCTION_RP_ID,
        "origin": protocol.PRODUCTION_ORIGIN,
        "iap_only": True,
        "local_compute_mutation_available": False,
        "sqlite_synchronous": "FULL",
        "sqlite_begin_immediate": True,
        "totp_dangerous_actions": False,
        "observation_attestors": {
            "cloud": record(cloud),
            "host": record(host),
        },
    }
    checked = storage.ProductionStorageGrowthBoundary(
        RELEASE, StaticOwnerGateTransport(ready)
    ).require_ready()
    assert checked["observation_attestors"] == ready["observation_attestors"]

    corrupted = {
        **ready,
        "observation_attestors": {
            **ready["observation_attestors"],
            "cloud": {
                **ready["observation_attestors"]["cloud"],
                "public_key_id": "0" * 64,
            },
        },
    }
    with pytest.raises(
        storage.PasskeyV2StorageBoundaryError,
        match="observation_attestor_trust_invalid",
    ):
        storage.ProductionStorageGrowthBoundary(
            RELEASE, StaticOwnerGateTransport(corrupted)
        ).require_ready()


def test_post_stop_cloud_attestation_is_bound_to_dual_signed_source_snapshot(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    mutation = service.StorageGrowthComputeExecutor(
        executor, RebootRequiredClient(), clock=lambda: NOW + 3
    )

    def context(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "authorization_request_id": action["request_id"],
            "observation_bundle": dict(bundle),
            "observation_bundle_sha256": bundle["bundle_sha256"],
            "observation_nonce_sha256": bundle["observation_nonce_sha256"],
            "activation_seal_sha256": "5" * 64,
            "firewall_readiness_receipt_sha256": "6" * 64,
        }

    source_bundle = action["action_payload"]["source_preflight"]
    after_resize = mutation(
        action,
        source_bundle["observation"],
        context(source_bundle),
    )
    resize_request = after_resize["observation_request"]
    resize_bundle = _bundle(
        _pending_report(collected_at=NOW + 4),
        transaction_id=action["transaction_id"],
        checkpoint="post_resize",
        prior_event_head_sha256=resize_request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=resize_request,
    )
    after_stop = mutation(
        action,
        resize_bundle["observation"],
        context(resize_bundle),
    )
    assert after_stop["observation_request"]["checkpoint"] == "post_stop"
    canonical_request = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=NOW + 5,
    )
    attestation_request = trusted_collector.build_attestation_request(
        canonical_request,
        _pending_report(collected_at=NOW + 5, status="TERMINATED"),
        role="cloud",
        now_unix=NOW + 5,
    )
    callback_calls = 0

    def attest(
        _frame: Mapping[str, Any], *, now_unix: int
    ) -> Mapping[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        assert now_unix == NOW + 5
        return {
            "role": "cloud",
            "observation_request_sha256": canonical_request[
                "observation_request_sha256"
            ],
        }

    accepted = service.handle_executor_frame(
        service.build_service_frame(
            "attest_cloud_observation",
            {
                "transaction_id": action["transaction_id"],
                "attestation_request": attestation_request,
            },
        ),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 5,
        mutation_handler=lambda *_args: (_ for _ in ()).throw(
            AssertionError("attestation touched mutation")
        ),
        readiness_handler=lambda: (_ for _ in ()).throw(
            AssertionError("attestation touched readiness")
        ),
        observation_verifier=lambda *_args: (_ for _ in ()).throw(
            AssertionError("attestation used bundle verifier")
        ),
        cloud_attestor=attest,
    )["document"]
    assert accepted["terminal"] is False
    assert callback_calls == 1

    mismatched = json.loads(json.dumps(attestation_request))
    mismatched["candidate_observation"][
        "current_host_receipt_sha256"
    ] = "0" * 64
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="stopped_snapshot_binding_invalid",
    ):
        service.handle_executor_frame(
            service.build_service_frame(
                "attest_cloud_observation",
                {
                    "transaction_id": action["transaction_id"],
                    "attestation_request": mismatched,
                },
            ),
            executor=executor,
            peer_uid=service.AUTHORITY_UID,
            release_revision=RELEASE,
            now_unix=NOW + 5,
            mutation_handler=lambda *_args: {},
            readiness_handler=lambda: {},
            observation_verifier=lambda *_args: {},
            cloud_attestor=attest,
        )
    assert callback_calls == 1


def _seed_crashed_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cloud: ed25519.Ed25519PrivateKey,
    host: ed25519.Ed25519PrivateKey,
) -> tuple[
    database.PasskeyV2ExecutorDatabase,
    Mapping[str, Any],
    service.StorageGrowthComputeExecutor,
    CrashAfterResizeClient,
    Any,
]:
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    compute = CrashAfterResizeClient()
    mutation = service.StorageGrowthComputeExecutor(
        executor, compute, clock=lambda: NOW + 3
    )
    monkeypatch.setattr(
        service,
        "read_activation_seal",
        lambda **_kwargs: {"seal_sha256": "5" * 64},
    )
    with pytest.raises(service.PasskeyV2ServiceError, match="mutation_failed"):
        service.handle_executor_frame(
            service.build_service_frame("execute", {
                "authorization_receipt": receipt,
                "action_envelope": action,
                "challenge_record": challenge,
                "grant_record": grant,
                "continuation_observation": action["action_payload"][
                    "source_preflight"
                ],
            }),
            executor=executor,
            peer_uid=service.AUTHORITY_UID,
            release_revision=RELEASE,
            now_unix=NOW + 3,
            mutation_handler=mutation,
            readiness_handler=lambda: {"receipt_sha256": "6" * 64},
            observation_verifier=_observation_verifier(cloud, host),
        )
    assert len(compute.resize_calls) == 1
    return (
        executor,
        action,
        mutation,
        compute,
        _observation_verifier(cloud, host),
    )


def test_expired_crash_reconciles_online_target_without_authority_or_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, mutation, compute, verifier = _seed_crashed_resize(
        tmp_path, monkeypatch, cloud=cloud, host=host
    )
    fresh_at = NOW + 10_000
    request = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=fresh_at,
    )
    assert request["checkpoint"] == "post_resize"
    target = _bundle(
        _target_report(collected_at=fresh_at),
        transaction_id=action["transaction_id"],
        checkpoint=request["checkpoint"],
        prior_event_head_sha256=request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=request,
    )
    executor_client = DirectExecutorClient(
        executor=executor,
        mutation=mutation,
        verifier=verifier,
        now_unix=fresh_at + 1,
    )
    mutation.clock = lambda: fresh_at + 1
    authority_client = ForbiddenAuthorityClient()
    response = service.handle_intake_frame(
        _remote_execute_frame(action=action, continuation=target),
        authority_client=authority_client,
        executor_client=executor_client,
        release_revision=RELEASE,
        now_unix=fresh_at + 1,
        binding_loader=lambda _release: (_ for _ in ()).throw(
            AssertionError("read-only reconciliation loaded runtime binding")
        ),
    )
    result = response["document"]
    assert result["read_only_reconciliation"] is True
    assert result["mutation_receipt"]["terminal"] is True
    assert authority_client.calls == []
    assert executor_client.operations == ["terminal", "reconcile_read_only"]
    assert len(compute.resize_calls) == 1
    assert executor.read_terminal_receipt(action["transaction_id"]) == result[
        "mutation_receipt"
    ]
    terminal_attestation = service.handle_executor_frame(
        service.build_service_frame(
            "attest_cloud_observation",
            {
                "transaction_id": action["transaction_id"],
                "attestation_request": {},
            },
        ),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=fresh_at + 2,
        mutation_handler=lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal attestation touched mutation")
        ),
        readiness_handler=lambda: (_ for _ in ()).throw(
            AssertionError("terminal attestation touched readiness")
        ),
        observation_verifier=lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal attestation verified observation")
        ),
        cloud_attestor=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal attestation touched metadata")
        ),
    )["document"]
    assert terminal_attestation["terminal"] is True
    assert terminal_attestation["terminal_receipt"] == result[
        "mutation_receipt"
    ]


def test_expired_crash_reconciles_resize_but_blocks_remaining_reboot_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, mutation, compute, verifier = _seed_crashed_resize(
        tmp_path, monkeypatch, cloud=cloud, host=host
    )
    fresh_at = NOW + 10_000
    request = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=fresh_at,
    )
    boot_needed = _bundle(
        _pending_report(collected_at=fresh_at),
        transaction_id=action["transaction_id"],
        checkpoint=request["checkpoint"],
        prior_event_head_sha256=request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=request,
    )
    executor_client = DirectExecutorClient(
        executor=executor,
        mutation=mutation,
        verifier=verifier,
        now_unix=fresh_at + 1,
    )
    mutation.clock = lambda: fresh_at + 1
    authority_client = ForbiddenAuthorityClient()
    first = service.handle_intake_frame(
        _remote_execute_frame(action=action, continuation=boot_needed),
        authority_client=authority_client,
        executor_client=executor_client,
        release_revision=RELEASE,
        now_unix=fresh_at + 1,
        binding_loader=lambda _release: (_ for _ in ()).throw(
            AssertionError("expired reconciliation loaded runtime binding")
        ),
    )["document"]["mutation_receipt"]
    assert first["remaining_stage"] == "stop"
    assert first["fresh_observation_required"] is True
    next_request = first["observation_request"]
    refreshed = _bundle(
        _pending_report(collected_at=fresh_at + 2),
        transaction_id=action["transaction_id"],
        checkpoint=next_request["checkpoint"],
        prior_event_head_sha256=next_request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=next_request,
    )
    second = service.handle_intake_frame(
        _remote_execute_frame(action=action, continuation=refreshed),
        authority_client=authority_client,
        executor_client=executor_client,
        release_revision=RELEASE,
        now_unix=fresh_at + 3,
        binding_loader=lambda _release: (_ for _ in ()).throw(
            AssertionError("expired reconciliation loaded runtime binding")
        ),
    )["document"]["mutation_receipt"]
    assert second["remaining_stage"] == "stop"
    assert second["fresh_observation_required"] is False
    assert second["passkey_resume_required"] is True
    assert authority_client.calls == []
    assert len(compute.resize_calls) == 1
    state = executor.read_transaction_state(action["transaction_id"])
    kinds = [event["event_kind"] for event in state["events"]]
    assert "resize_complete" in kinds
    assert "post_resize_observation_accepted" in kinds
    assert "stop_intent" not in kinds


def test_resume_request_is_model_free_and_exactly_bound_to_remaining_stop_stage(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    mutation = service.StorageGrowthComputeExecutor(
        executor, RebootRequiredClient(), clock=lambda: NOW + 3
    )

    def context(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "authorization_request_id": action["request_id"],
            "observation_bundle": dict(bundle),
            "observation_bundle_sha256": bundle["bundle_sha256"],
            "observation_nonce_sha256": bundle["observation_nonce_sha256"],
            "activation_seal_sha256": "5" * 64,
            "firewall_readiness_receipt_sha256": "6" * 64,
        }

    source_bundle = action["action_payload"]["source_preflight"]
    progress = mutation(
        action,
        source_bundle["observation"],
        context(source_bundle),
    )
    resize_request = progress["observation_request"]
    resize_bundle = _bundle(
        _pending_report(collected_at=NOW + 4),
        transaction_id=action["transaction_id"],
        checkpoint="post_resize",
        prior_event_head_sha256=resize_request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=resize_request,
    )
    state = executor.read_execution_state(action["request_id"])
    mutation._accept_observation(
        request_id=action["request_id"],
        stage="resize",
        observation=resize_bundle["observation"],
        context=context(resize_bundle),
        kinds={event["event_kind"] for event in state["events"]},
    )
    resume_request = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=NOW + 5,
    )
    continuation = _bundle(
        _pending_report(collected_at=NOW + 5),
        transaction_id=action["transaction_id"],
        checkpoint="post_resize",
        prior_event_head_sha256=resume_request[
            "prior_event_head_sha256"
        ],
        cloud_key=cloud,
        host_key=host,
        observation_request=resume_request,
    )
    unsigned = {
        "schema": storage.REMOTE_FRAME_SCHEMA,
        "operation": "request_resume",
        "release_sha": RELEASE,
        "document": {
            "plan_sha256": storage.exact_storage_plan()["plan_sha256"],
            "transaction_id": action["transaction_id"],
            "continuation_preflight": continuation,
        },
    }
    authority_client = RecordingAuthorityClient()
    executor_client = DirectExecutorClient(
        executor=executor,
        mutation=mutation,
        verifier=_observation_verifier(cloud, host),
        now_unix=NOW + 6,
    )
    result = service.handle_intake_frame(
        {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)},
        authority_client=authority_client,
        executor_client=executor_client,
        release_revision=RELEASE,
        now_unix=NOW + 6,
        binding_loader=lambda _release: (
            {"unused_for_authoring": True}, "c" * 64, "d" * 64
        ),
    )["document"]
    assert result["passkey_only"] is True
    assert result["local_mutation_authority"] is False
    assert executor_client.operations == [
        "observation_request", "context", "verify_observation"
    ]
    assert len(authority_client.calls) == 1
    operation, authority_document = authority_client.calls[0]
    assert operation == "create_request"
    resume_action = authority_document["action_envelope"]
    assert resume_action["stage"] == "stop"
    assert resume_action["transaction_id"] == action["transaction_id"]
    assert resume_action["prior_event_head_sha256"] == resume_request[
        "prior_event_head_sha256"
    ]
    assert resume_action["action_payload"]["source_preflight"] == continuation
    assert resume_action["action_payload"]["allowed_operations"] == [
        "conditional_stop_exact_instance",
        "conditional_start_exact_instance",
        "read_only_postflight",
    ]


def test_crash_reconcile_terminal_replay_needs_no_live_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    compute = CrashAfterResizeClient()
    mutation = service.StorageGrowthComputeExecutor(
        executor, compute, clock=lambda: NOW + 3
    )
    monkeypatch.setattr(
        service,
        "read_activation_seal",
        lambda **_kwargs: {"seal_sha256": "5" * 64},
    )
    readiness_calls = 0

    def readiness() -> Mapping[str, Any]:
        nonlocal readiness_calls
        readiness_calls += 1
        return {"receipt_sha256": "6" * 64}

    verify = _observation_verifier(cloud, host)
    source_bundle = action["action_payload"]["source_preflight"]
    base_document = {
        "authorization_receipt": receipt,
        "action_envelope": action,
        "challenge_record": challenge,
        "grant_record": grant,
    }
    with pytest.raises(service.PasskeyV2ServiceError, match="mutation_failed"):
        service.handle_executor_frame(
            service.build_service_frame("execute", {
                **base_document,
                "continuation_observation": source_bundle,
            }),
            executor=executor,
            peer_uid=service.AUTHORITY_UID,
            release_revision=RELEASE,
            now_unix=NOW + 3,
            mutation_handler=mutation,
            readiness_handler=readiness,
            observation_verifier=verify,
        )
    state = executor.read_transaction_state(action["transaction_id"])
    assert [event["event_kind"] for event in state["events"]][-2:] == [
        "resize_intent", "attempt_failed"
    ]
    request = service.handle_executor_frame(
        service.build_service_frame(
            "observation_request", {"transaction_id": action["transaction_id"]}
        ),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 3,
        mutation_handler=mutation,
        readiness_handler=readiness,
        observation_verifier=verify,
    )["document"]
    recovery_source = _bundle(
        _pending_report(collected_at=NOW),
        transaction_id=action["transaction_id"],
        checkpoint=request["checkpoint"],
        prior_event_head_sha256=request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=request,
    )
    progress = service.handle_executor_frame(
        service.build_service_frame("execute", {
            **base_document,
            "continuation_observation": recovery_source,
        }),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 4,
        mutation_handler=mutation,
        readiness_handler=readiness,
        observation_verifier=verify,
    )["document"]["mutation_receipt"]
    assert progress["state"] == "observation_required"
    assert len(compute.resize_calls) == 1
    next_request = progress["observation_request"]
    target = _bundle(
        _target_report(collected_at=NOW + 5),
        transaction_id=action["transaction_id"],
        checkpoint=next_request["checkpoint"],
        prior_event_head_sha256=next_request["prior_event_head_sha256"],
        cloud_key=cloud,
        host_key=host,
        observation_request=next_request,
    )
    terminal = service.handle_executor_frame(
        service.build_service_frame("execute", {
            **base_document,
            "continuation_observation": target,
        }),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 6,
        mutation_handler=mutation,
        readiness_handler=readiness,
        observation_verifier=verify,
    )["document"]["mutation_receipt"]
    assert terminal["terminal"] is True
    assert executor.read_terminal_receipt(action["transaction_id"]) == terminal

    def forbidden() -> Mapping[str, Any]:
        raise AssertionError("terminal replay touched readiness")

    replay = service.handle_executor_frame(
        service.build_service_frame("execute", {
            **base_document,
            "continuation_observation": {},
        }),
        executor=executor,
        peer_uid=service.AUTHORITY_UID,
        release_revision=RELEASE,
        now_unix=NOW + 10_000,
        mutation_handler=lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal replay touched Compute")
        ),
        readiness_handler=forbidden,
        observation_verifier=lambda *_args: (_ for _ in ()).throw(
            AssertionError("terminal replay touched collector")
        ),
    )["document"]
    assert replay["terminal_replay"] is True
    assert replay["mutation_receipt"] == terminal


def test_repeated_attempt_cannot_precede_or_escape_matching_intent(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    opened = executor.read_transaction_state(action["transaction_id"])["events"][-1]
    request_id = service.StorageGrowthComputeExecutor._gce_request_id(
        action["transaction_id"], "resize"
    )
    attempt_id = protocol.sha256_json({
        "schema": "muncho-passkey-v2-stage-attempt-id.v1",
        "transaction_id": action["transaction_id"],
        "stage": "resize",
        "attempt_anchor_event_head_sha256": opened["event_head_sha256"],
    })
    with pytest.raises(database.PasskeyV2SqliteDenied, match="attempt_binding"):
        executor.append_execution_event(
            request_id=action["request_id"],
            event_kind="attempt_failed",
            event_payload={
                "stage": "resize",
                "attempt_id": attempt_id,
                "attempt_anchor_event_head_sha256": opened[
                    "event_head_sha256"
                ],
                "gce_request_id": request_id,
                "failure": "InjectedCrash",
                "live_resource_sha256": SHA,
            },
            now_unix=NOW + 4,
        )


def test_real_unix_socket_fragmentation_bounds_peer_uid_and_isolation(
) -> None:
    short_directory = tempfile.TemporaryDirectory(
        prefix="pv2-", dir="/tmp"
    )
    path = Path(short_directory.name) / "s"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(4)
    peer_uids: list[int] = []
    failures: list[BaseException] = []

    def handler(
        value: Mapping[str, Any], peer_uid: int
    ) -> Mapping[str, Any]:
        frame = service.validate_service_frame(value)
        peer_uids.append(peer_uid)
        return service.build_service_response(
            frame["operation"], {"peer_uid": peer_uid}
        )

    def serve_three() -> None:
        try:
            for _index in range(3):
                connection, _address = listener.accept()
                with connection:
                    service._handle_service_connection(connection, handler)
        except BaseException as exc:
            failures.append(exc)
        finally:
            listener.close()

    thread = threading.Thread(target=serve_three, daemon=True)
    thread.start()

    def exchange(parts: list[bytes]) -> Mapping[str, Any]:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(5.0)
        try:
            connection.connect(str(path))
            for part in parts:
                connection.sendall(part)
            connection.shutdown(socket.SHUT_WR)
            response = bytearray()
            while True:
                chunk = connection.recv(64 * 1024)
                if not chunk:
                    break
                response.extend(chunk)
        finally:
            connection.close()
        assert response.endswith(b"\n")
        decoded = protocol.decode_canonical_json(bytes(response[:-1]))
        assert isinstance(decoded, Mapping)
        return decoded

    frame = protocol.canonical_json_bytes(
        service.build_service_frame("health", {})
    ) + b"\n"
    fragmented = exchange([frame[:3], frame[3:17], frame[17:]])
    assert service.validate_service_response(
        fragmented, expected_operation="health"
    )["peer_uid"] == os.getuid()

    oversized = exchange([b"x" * (service.MAX_FRAME_BYTES + 2)])
    assert oversized["ok"] is False
    assert oversized["document"] == {"error": "request_rejected"}

    client = service.UnixServiceClient(path)
    assert client.call("health", {})["peer_uid"] == os.getuid()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert failures == []
    assert peer_uids == [os.getuid(), os.getuid()]
    short_directory.cleanup()


def test_unix_client_uses_fixed_long_mutation_deadline() -> None:
    assert service.SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS["execute"] > (
        service.COMPUTE_POLL_ATTEMPTS * 2.0
    )
    assert service.SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS["terminal"] < 30.0
    assert set(service.SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS) == {
        "health",
        "render",
        "options",
        "verify",
        "create_request",
        "consume",
        "preflight",
        "execute",
        "terminal",
        "context",
        "verify_observation",
        "observation_request",
        "reconcile_read_only",
        "attest_cloud_observation",
    }


def test_response_timeout_is_unknown_then_terminal_replay_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_directory = tempfile.TemporaryDirectory(prefix="pv2-", dir="/tmp")
    path = Path(short_directory.name) / "s"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(2)
    first_finished = threading.Event()
    failures: list[BaseException] = []

    def serve_two() -> None:
        try:
            for index in range(2):
                connection, _address = listener.accept()
                with connection:
                    raw = bytearray()
                    # Match the production service framing contract exactly:
                    # the client half-closes its write side after the single
                    # newline-delimited frame, and the server does not answer
                    # until it observes EOF.  Reading only through the newline
                    # lets the test server race the client's shutdown() and can
                    # produce Darwin ENOTCONN before the replay is exercised.
                    while True:
                        chunk = connection.recv(64 * 1024)
                        if not chunk:
                            break
                        raw.extend(chunk)
                    frame = service.validate_service_frame(
                        protocol.decode_canonical_json(bytes(raw[:-1]))
                    )
                    assert frame["operation"] == "terminal"
                    response = service.build_service_response(
                        "terminal",
                        {
                            "transaction_id": SHA,
                            "terminal": True,
                            "terminal_receipt": {"receipt_sha256": SHA},
                        },
                    )
                    if index == 0:
                        threading.Event().wait(0.12)
                    try:
                        connection.sendall(
                            protocol.canonical_json_bytes(response) + b"\n"
                        )
                    except OSError:
                        if index != 0:
                            raise
                    if index == 0:
                        first_finished.set()
        except BaseException as exc:
            failures.append(exc)
            first_finished.set()
        finally:
            listener.close()

    timeouts = dict(service.SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS)
    timeouts["terminal"] = 0.03
    monkeypatch.setattr(
        service, "SERVICE_OPERATION_RESPONSE_TIMEOUT_SECONDS", timeouts
    )
    thread = threading.Thread(target=serve_two, daemon=True)
    thread.start()
    client = service.UnixServiceClient(path)
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_service_operation_outcome_unknown",
    ):
        client.call("terminal", {"transaction_id": SHA})
    assert first_finished.wait(timeout=1.0)
    replay = client.call("terminal", {"transaction_id": SHA})
    assert replay["terminal"] is True
    assert replay["terminal_receipt"] == {"receipt_sha256": SHA}
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert failures == []
    short_directory.cleanup()


def test_cached_readiness_is_single_flight_fresh_and_fail_closed() -> None:
    now = [100.0]
    started = threading.Event()
    release = threading.Event()

    class AuthorityClient:
        def __init__(self) -> None:
            self.calls = 0
            self.healthy = True

        def call(
            self, operation: str, document: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            assert operation == "health"
            assert document == {}
            self.calls += 1
            started.set()
            assert release.wait(timeout=1.0)
            if not self.healthy:
                raise service.PasskeyV2ServiceError("injected")
            return {"healthy": True, "preflight": {"ok": True}}

    client = AuthorityClient()
    readiness = service._CachedAuthorityReadiness(
        client, clock=lambda: now[0]  # type: ignore[arg-type]
    )
    assert readiness.status() is False
    assert started.wait(timeout=1.0)
    assert readiness.status() is False
    assert client.calls == 1
    release.set()
    for _index in range(100):
        if readiness.status():
            break
        threading.Event().wait(0.01)
    assert readiness.status() is True
    assert client.calls == 1

    now[0] += service.WEB_READINESS_REFRESH_SECONDS - 1.0
    assert readiness.status() is True
    assert client.calls == 1
    now[0] += 2.0
    client.healthy = False
    assert readiness.status() is True
    for _index in range(100):
        if client.calls == 2:
            break
        threading.Event().wait(0.01)
    for _index in range(100):
        with readiness._lock:
            refreshing = readiness._refreshing
        if not refreshing:
            break
        threading.Event().wait(0.01)
    assert client.calls == 2
    assert readiness.status() is False


def test_web_health_and_readiness_are_distinct_exact_routes() -> None:
    headers = {"host": "auth.lomliev.com"}
    assert service.validate_web_request(
        method="GET",
        path="/healthz",
        headers=headers,
        body=b"",
        csrf_cookie=None,
    ) == ("health", None, None)
    assert service.validate_web_request(
        method="GET",
        path="/readyz",
        headers=headers,
        body=b"",
        csrf_cookie=None,
    ) == ("readiness", None, None)


def test_socket_activation_descriptor_requires_exact_listening_unix_path(
) -> None:
    directory = tempfile.TemporaryDirectory(prefix="pv2-", dir="/tmp")
    root = Path(directory.name)
    path = root / "authority.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    duplicate = service._validate_activated_listener_descriptor(
        listener.fileno(), expected_path=path
    )
    try:
        assert duplicate.family == socket.AF_UNIX
        assert duplicate.getsockname() == str(path)
        if sys.platform.startswith("linux"):
            assert duplicate.getsockopt(
                socket.SOL_SOCKET, socket.SO_ACCEPTCONN
            ) == 1
    finally:
        duplicate.close()
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_socket_activation_descriptor_invalid",
    ):
        service._validate_activated_listener_descriptor(
            listener.fileno(), expected_path=root / "other.sock"
        )
    listener.close()

    unready_path = root / "unready.sock"
    unready = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unready.bind(str(unready_path))
    try:
        if sys.platform.startswith("linux"):
            with pytest.raises(
                service.PasskeyV2ServiceError,
                match="passkey_v2_socket_activation_descriptor_invalid",
            ):
                service._validate_activated_listener_descriptor(
                    unready.fileno(), expected_path=unready_path
                )
    finally:
        unready.close()

    regular = root / "not-a-socket"
    regular.write_bytes(b"x")
    descriptor = os.open(regular, os.O_RDONLY)
    try:
        with pytest.raises(
            service.PasskeyV2ServiceError,
            match="passkey_v2_socket_activation_descriptor_invalid",
        ):
            service._validate_activated_listener_descriptor(
                descriptor, expected_path=path
            )
    finally:
        os.close(descriptor)
    directory.cleanup()


def test_socket_activation_requires_exact_systemd_descriptor_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISTEN_FDS", "1")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDNAMES", "wrong-service")
    with pytest.raises(
        service.PasskeyV2ServiceError,
        match="passkey_v2_socket_activation_required",
    ):
        service._validated_activated_listener(
            expected_path=service.AUTHORITY_SOCKET,
            expected_name="passkey-authority",
        )


def test_service_connection_gate_bounds_concurrency_and_per_uid_rate() -> None:
    now = [100.0]
    gate = service._ServiceConnectionGate(clock=lambda: now[0])

    for _index in range(service.SERVICE_MAX_CONNECTIONS_PER_UID):
        assert gate.try_acquire(1001) is True
    assert gate.try_acquire(1001) is False
    assert gate.try_acquire(1002) is True
    for _index in range(service.SERVICE_MAX_CONNECTIONS_PER_UID):
        gate.release(1001)
    gate.release(1002)

    fresh = service._ServiceConnectionGate(clock=lambda: now[0])
    for _index in range(int(service.SERVICE_UID_BURST)):
        assert fresh.try_acquire(1001) is True
        fresh.release(1001)
    assert fresh.try_acquire(1001) is False
    now[0] += 1.0 / service.SERVICE_UID_RATE_PER_SECOND
    assert fresh.try_acquire(1001) is True
    fresh.release(1001)


def test_slow_local_frame_cannot_block_second_health_frame() -> None:
    gate = service._ServiceConnectionGate()
    slow_service, slow_client = socket.socketpair()
    fast_service, fast_client = socket.socketpair()

    def handler(
        value: Mapping[str, Any], peer_uid: int
    ) -> Mapping[str, Any]:
        frame = service.validate_service_frame(value)
        return service.build_service_response(
            frame["operation"], {"peer_uid": peer_uid}
        )

    slow_worker = service._dispatch_service_connection(
        slow_service,
        handler,
        gate,
    )
    fast_worker = service._dispatch_service_connection(
        fast_service,
        handler,
        gate,
    )
    assert slow_worker is not None
    assert fast_worker is not None
    fast_client.settimeout(1.0)
    frame = protocol.canonical_json_bytes(
        service.build_service_frame("health", {})
    ) + b"\n"
    try:
        fast_client.sendall(frame)
        fast_client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while not response.endswith(b"\n"):
            response.extend(fast_client.recv(64 * 1024))
        decoded = protocol.decode_canonical_json(bytes(response[:-1]))
        assert service.validate_service_response(
            decoded,
            expected_operation="health",
        )["peer_uid"] == os.getuid()
    finally:
        fast_client.close()
        slow_client.close()
    fast_worker.join(timeout=1.0)
    slow_worker.join(timeout=1.0)
    assert not fast_worker.is_alive()
    assert not slow_worker.is_alive()


@pytest.mark.parametrize("_stress_round", range(12))
def test_concurrent_singleton_intent_is_byte_identical_and_conflict_denied(
    tmp_path: Path,
    _stress_round: int,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    payload = {
        "stage": "resize",
        "requested_operation": "resize_exact_disk_40_to_80",
        "observation_bundle_sha256": action["action_payload"][
            "source_preflight"
        ]["bundle_sha256"],
        "live_before_sha256": SHA,
        "gce_request_id": service.StorageGrowthComputeExecutor._gce_request_id(
            action["transaction_id"], "resize"
        ),
        "activation_seal_sha256": "5" * 64,
        "firewall_readiness_receipt_sha256": "6" * 64,
    }

    def append() -> Mapping[str, Any]:
        return executor.append_execution_event(
            request_id=action["request_id"],
            event_kind="resize_intent",
            event_payload=payload,
            now_unix=NOW + 4,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: append(), range(16)))
    assert all(result == results[0] for result in results)
    state = executor.read_transaction_state(action["transaction_id"])
    assert [event["event_kind"] for event in state["events"]].count(
        "resize_intent"
    ) == 1
    with pytest.raises(database.PasskeyV2SqliteError, match="event_invalid"):
        executor.append_execution_event(
            request_id=action["request_id"],
            event_kind="resize_intent",
            event_payload={**payload, "activation_seal_sha256": "7" * 64},
            now_unix=NOW + 5,
        )


def test_concurrent_collection_attempt_is_singleton_and_rotates_at_expiry(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, _challenge_value, _grant, _receipt = (
        _authority_and_executor(tmp_path, cloud_key=cloud, host_key=host)
    )

    def issue() -> Mapping[str, Any]:
        return service._canonical_observation_request(
            executor,
            transaction_id=action["transaction_id"],
            release_revision=RELEASE,
            now_unix=NOW + 3,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = list(pool.map(lambda _index: issue(), range(24)))
    encoded = protocol.canonical_json_bytes(requests[0])
    assert all(protocol.canonical_json_bytes(item) == encoded for item in requests)
    assert requests[0]["collection_attempt_sequence"] == 1
    assert executor.preflight()["observation_collection_attempt_count"] == 1

    rotated = service._canonical_observation_request(
        executor,
        transaction_id=action["transaction_id"],
        release_revision=RELEASE,
        now_unix=requests[0]["collection_attempt_expires_at_unix"],
    )
    assert rotated["collection_attempt_sequence"] == 2
    assert rotated["collection_attempt_id"] != requests[0][
        "collection_attempt_id"
    ]
    assert rotated["observation_request_sha256"] != requests[0][
        "observation_request_sha256"
    ]
    assert executor.preflight()["observation_collection_attempt_count"] == 2


def test_corrupted_executor_event_blob_is_detected_after_trigger_restoration(
    tmp_path: Path,
) -> None:
    cloud = ed25519.Ed25519PrivateKey.generate()
    host = ed25519.Ed25519PrivateKey.generate()
    executor, action, challenge, grant, receipt = _authority_and_executor(
        tmp_path, cloud_key=cloud, host_key=host
    )
    executor.claim_execution(
        receipt=receipt,
        envelope=action,
        grant=grant,
        challenge=challenge,
        now_unix=NOW + 3,
    )
    connection = sqlite3.connect(str(executor.path))
    try:
        connection.execute("DROP TRIGGER execution_events_no_update")
        connection.execute("DROP TRIGGER execution_events_no_delete")
        row = connection.execute(
            "SELECT event FROM execution_events WHERE event_kind='opened'"
        ).fetchone()
        assert row is not None
        event = protocol.decode_canonical_json(bytes(row[0]))
        assert isinstance(event, Mapping)
        corrupted = dict(event)
        corrupted["event_payload"] = {
            **event["event_payload"],
            "source_preflight_sha256": "0" * 64,
        }
        connection.execute(
            "UPDATE execution_events SET event=? WHERE event_kind='opened'",
            (protocol.canonical_json_bytes(corrupted),),
        )
        connection.executescript(
            database._append_only_triggers("execution_events")
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(database.PasskeyV2SqliteError):
        executor.preflight()
