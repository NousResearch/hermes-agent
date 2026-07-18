from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_host_observation as producer
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_preflight as preflight


REVISION = "a" * 40
FOUNDATION_REVISION = "b" * 40
NOW = 1_785_000_000
DIRECT_IAM_RAW = b"direct"
PRODUCTION_INGRESS_OBSERVATION_SHA256 = "2" * 64


def _lineage_request(*, phase: str = "inert") -> dict[str, Any]:
    request = {
        "schema": producer.ATTACHED_SA_REQUEST_SCHEMA,
        "phase": phase,
        "collected_at_unix": NOW,
        "plan_sha256": "1" * 64,
        "production_ingress_observation_sha256": (
            PRODUCTION_INGRESS_OBSERVATION_SHA256
        ),
        "cloud_install_receipt": {"fixture": "signed-install-receipt"},
        "cloud_signer_provisioning_receipt_sha256": "3" * 64,
        "cloud_signer_readiness_sha256": "4" * 64,
        "host_signer_provisioning_receipt_sha256": "5" * 64,
        "host_signer_readiness_sha256": "6" * 64,
    }
    request["observation_binding_sha256"] = foundation.sha256_json({
        name: item for name, item in request.items() if name != "schema"
    })
    request["request_sha256"] = foundation.sha256_json(request)
    return request


def _package() -> dict[str, Any]:
    return {
        "release_revision": REVISION,
        "source_tree_oid": "c" * 40,
        "foundation_source_revision": FOUNDATION_REVISION,
        "foundation_source_tree_oid": "d" * 40,
        "package_sha256": "7" * 64,
        "package_inventory_sha256": "8" * 64,
        "pre_foundation_authority_sha256": "9" * 64,
        "foundation_apply_receipt_sha256": "a" * 64,
        "project_ancestry_evidence_sha256": "b" * 64,
        "project_ancestry_chain_sha256": "c" * 64,
        "resource_ancestor_chain": [
            "organizations/123456789012",
            "projects/123456789012",
        ],
        "direct_iam_identity_authority_sha256": hashlib.sha256(
            DIRECT_IAM_RAW
        ).hexdigest(),
    }


def _direct_identity() -> dict[str, Any]:
    return {
        "release_revision": FOUNDATION_REVISION,
        "pre_foundation_authority_sha256": "9" * 64,
        "foundation_apply_receipt_sha256": "a" * 64,
        "resource_ancestor_chain": [
            "organizations/123456789012",
            "projects/123456789012",
        ],
        "owner_gate_vm_numeric_id": _facts()[
            "runtime_instance_numeric_id"
        ],
        "owner_gate_service_account_email": (
            direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL
        ),
        "metadata_oauth_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
        "owner_gate_service_account_unique_id": "123456789012345678901",
    }


def _facts(*, post_iam: bool = False) -> dict[str, Any]:
    return {
        "runtime_instance_numeric_id": "1234567890123456789",
        "runtime_service_account_email": direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL,
        "metadata_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
        "effective_permission_probe": preflight.expected_effective_permission_probe(
            post_iam
        ),
        "target_instance_numeric_id": foundation.TARGET_INSTANCE_ID,
        "target_disk_numeric_id": foundation.TARGET_DISK_ID,
        "numeric_targets_reverified": post_iam,
    }


def test_embedded_foundation_identity_is_bound_to_final_release_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_manifest = _package()
    reads: list[tuple[Path, frozenset[int]]] = []
    decode_revisions: list[str | None] = []

    def read(path: Path, **kwargs: Any) -> bytes:
        reads.append((path, kwargs["modes"]))
        return DIRECT_IAM_RAW

    def decode(
        raw: bytes,
        *,
        release_revision: str | None = None,
    ) -> dict[str, Any]:
        assert raw == DIRECT_IAM_RAW
        decode_revisions.append(release_revision)
        identity = _direct_identity()
        if (
            release_revision is not None
            and identity["release_revision"] != release_revision
        ):
            raise direct_iam.DirectIamIdentityAuthorityError("mismatch")
        return identity

    monkeypatch.setattr(producer, "_read_regular", read)
    monkeypatch.setattr(producer.direct_iam, "decode_canonical", decode)

    assert producer._load_direct_iam_identity(package_manifest) == (
        _direct_identity()
    )
    assert reads == [(
        producer.OWNER_RELEASE_BASE
        / REVISION
        / "trust/direct-iam-identity-authority.json",
        frozenset({0o444}),
    )]
    assert decode_revisions == [None, FOUNDATION_REVISION]


@pytest.mark.parametrize(
    "corruption",
    ("same-release", "foundation-revision", "digest", "lineage"),
)
def test_embedded_foundation_identity_rejects_cross_release_drift(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    package_manifest = _package()
    identity = _direct_identity()
    if corruption == "same-release":
        identity["release_revision"] = REVISION
    elif corruption == "foundation-revision":
        package_manifest["foundation_source_revision"] = "e" * 40
    elif corruption == "digest":
        package_manifest["direct_iam_identity_authority_sha256"] = "0" * 64
    else:
        identity["foundation_apply_receipt_sha256"] = "0" * 64
    monkeypatch.setattr(
        producer,
        "_read_regular",
        lambda *_args, **_kwargs: DIRECT_IAM_RAW,
    )
    monkeypatch.setattr(
        producer.direct_iam,
        "decode_canonical",
        lambda *_args, **_kwargs: identity,
    )

    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_attached_sa_direct_identity_invalid",
    ):
        producer._load_direct_iam_identity(package_manifest)


def test_api_parser_accepts_bounded_content_length_and_chunked_json() -> None:
    ordinary = b'{ "permissions" : ["compute.instances.get"] }'
    content_length = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(ordinary)}\r\n\r\n".encode("ascii")
        + ordinary
    )
    assert producer._parse_api_response(content_length) == {
        "permissions": ["compute.instances.get"]
    }
    first = b'{"permissions":['
    second = b'"compute.instances.get"]}'
    chunked = (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        + f"{len(first):x}\r\n".encode("ascii")
        + first
        + b"\r\n"
        + f"{len(second):x}\r\n".encode("ascii")
        + second
        + b"\r\n0\r\n\r\n"
    )
    assert producer._parse_api_response(chunked) == {
        "permissions": ["compute.instances.get"]
    }
    repeated_vary = (
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
        b"Vary: Origin\r\nVary: X-Origin\r\nVary: Referer\r\n\r\n{}"
    )
    assert producer._parse_api_response(repeated_vary) == {}


@pytest.mark.parametrize(
    "raw",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}",
        b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"a\":1,\"a\":2}",
    ],
)
def test_api_parser_rejects_duplicate_partial_or_ambiguous_response(
    raw: bytes,
) -> None:
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_attached_sa_api_response_invalid",
    ):
        producer._parse_api_response(raw)


def test_https_request_buffer_is_wiped_on_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawSocket:
        def close(self) -> None:
            return None

    class TlsSocket:
        request: bytearray | None = None

        def sendall(self, raw: bytearray) -> None:
            self.request = raw
            raise KeyboardInterrupt

        def close(self) -> None:
            return None

    tls = TlsSocket()

    class Context:
        def wrap_socket(self, _raw: object, *, server_hostname: str) -> TlsSocket:
            assert server_hostname == "compute.googleapis.com"
            return tls

    monkeypatch.setattr(producer.socket, "create_connection", lambda *_a, **_k: RawSocket())
    monkeypatch.setattr(producer.trusted, "fixed_debian_tls_context", Context)
    token_raw = bytearray(b"not-a-real-token")
    token = memoryview(token_raw)
    path = (
        f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
        f"instances/{foundation.TARGET_INSTANCE}"
    )
    with pytest.raises(KeyboardInterrupt):
        producer.MetadataSaProbe._https_json(
            "compute.googleapis.com", path, b"", token
        )
    token.release()
    assert tls.request is not None
    assert set(tls.request) == {0}


def test_metadata_token_is_wiped_when_api_raises_base_exception() -> None:
    class Probe(producer.MetadataSaProbe):
        token_raw: bytearray | None = None

        def _metadata_get(self, path: str, *, maximum: int = 64 * 1024) -> bytearray:
            del maximum
            if path == producer.METADATA_INSTANCE_ID_PATH:
                return bytearray(b"1234567890123456789")
            if path == producer.METADATA_SERVICE_ACCOUNT_EMAIL_PATH:
                return bytearray(direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL.encode("ascii"))
            if path == producer.METADATA_SCOPES_PATH:
                return bytearray("\n".join(foundation.OWNER_GATE_OAUTH_SCOPES).encode("ascii"))
            self.token_raw = bytearray(
                b'{"access_token":"not-a-real-token","expires_in":300,"token_type":"Bearer"}'
            )
            return self.token_raw

        @staticmethod
        def _https_json(
            host: str, path: str, body: bytes, token: memoryview
        ) -> Mapping[str, Any]:
            del host, path, body, token
            raise KeyboardInterrupt

    probe = Probe()
    with pytest.raises(KeyboardInterrupt):
        probe.collect(post_iam=False)
    assert probe.token_raw is not None
    assert set(probe.token_raw) == {0}


@pytest.mark.parametrize("failure", ["oversize", "base_exception"])
def test_metadata_get_wipes_owned_body_on_every_nonreturn(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 200
        retained: bytearray | None = None
        calls = 0

        def getheader(self, name: str) -> str:
            assert name == "Metadata-Flavor"
            return "Google"

        def readinto(self, view: memoryview) -> int:
            self.retained = cast(bytearray, view.obj)
            self.calls += 1
            payload = b"secret123"
            view[: min(len(view), len(payload))] = payload[: len(view)]
            if failure == "base_exception":
                raise KeyboardInterrupt
            return min(len(view), len(payload)) if self.calls == 1 else 0

    response = Response()

    class Connection:
        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> Response:
            return response

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        producer.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: Connection(),
    )
    expected = KeyboardInterrupt if failure == "base_exception" else (
        producer.OwnerGateHostObservationError
    )
    with pytest.raises(expected):
        producer.MetadataSaProbe()._metadata_get(
            producer.METADATA_TOKEN_PATH, maximum=8
        )
    assert response.retained is not None
    assert set(response.retained) == {0}


def _scope_probe(scopes: list[str]) -> producer.MetadataSaProbe:
    class Probe(producer.MetadataSaProbe):
        def _metadata_get(
            self, path: str, *, maximum: int = 64 * 1024
        ) -> bytearray:
            del maximum
            values = {
                producer.METADATA_INSTANCE_ID_PATH: b"1234567890123456789",
                producer.METADATA_SERVICE_ACCOUNT_EMAIL_PATH: (
                    direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL.encode("ascii")
                ),
                producer.METADATA_SCOPES_PATH: "\n".join(scopes).encode("ascii"),
                producer.METADATA_TOKEN_PATH: (
                    b'{"access_token":"test-token","expires_in":300,'
                    b'"token_type":"Bearer"}'
                ),
            }
            return bytearray(values[path])

        def _https_json(
            self, host: str, path: str, body: bytes, token: memoryview
        ) -> Mapping[str, Any]:
            del body, token
            if host == "iam.googleapis.com":
                name = "service_account"
            elif host == "cloudresourcemanager.googleapis.com":
                name = "project"
            elif "/disks/" in path:
                name = "disk"
            else:
                name = "instance"
            return {
                "permissions": preflight.expected_effective_permission_probe(
                    False
                )[name]["granted_permissions"]
            }

    return Probe()


def test_metadata_scope_set_is_normalized_to_canonical_order() -> None:
    expected = list(foundation.OWNER_GATE_OAUTH_SCOPES)
    observed = _scope_probe(list(reversed(expected))).collect(post_iam=False)
    assert observed["metadata_scopes"] == expected


@pytest.mark.parametrize("variant", ["duplicate", "extra", "missing"])
def test_metadata_scope_set_rejects_duplicate_extra_or_missing(
    variant: str,
) -> None:
    scopes = list(foundation.OWNER_GATE_OAUTH_SCOPES)
    if variant == "duplicate":
        scopes[-1] = scopes[0]
    elif variant == "extra":
        scopes.append("https://www.googleapis.com/auth/cloud-platform")
    else:
        scopes.pop()
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_attached_sa_metadata_invalid",
    ):
        _scope_probe(scopes).collect(post_iam=False)


def test_executor_child_payload_refuses_to_probe_as_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class Collector:
        def collect(self, *, post_iam: bool) -> Mapping[str, Any]:
            calls.append(post_iam)
            return _facts(post_iam=post_iam)

    monkeypatch.setattr(producer.os, "getuid", lambda: 0)
    monkeypatch.setattr(producer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(producer.os, "getgid", lambda: 0)
    monkeypatch.setattr(producer.os, "getegid", lambda: 0)
    monkeypatch.setattr(producer.os, "getgroups", lambda: [])
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_attached_sa_child_identity_invalid",
    ):
        producer._executor_child_payload(
            post_iam=False,
            collector=cast(producer.MetadataSaProbe, Collector()),
        )
    assert calls == []
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(
            producer.os, name, lambda: preflight.EXECUTOR_UID
        )
    payload = producer._executor_child_payload(
        post_iam=True,
        collector=cast(producer.MetadataSaProbe, Collector()),
    )
    assert payload["uid"] == preflight.EXECUTOR_UID
    assert payload["groups"] == []
    assert calls == [True]


def test_identity_facts_rejects_cross_boundary_supplementary_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {
        "muncho-passkey-web": preflight.WEB_UID,
        "muncho-passkey-authority": preflight.AUTHORITY_UID,
        "muncho-storage-executor": preflight.EXECUTOR_UID,
    }
    monkeypatch.setattr(
        producer.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(
            pw_uid=identities[name],
            pw_gid=identities[name],
            pw_shell="/usr/sbin/nologin",
        ),
    )
    monkeypatch.setattr(
        producer.grp,
        "getgrnam",
        lambda name: SimpleNamespace(gr_gid=identities[name]),
    )
    monkeypatch.setattr(
        producer.os,
        "getgrouplist",
        lambda name, primary: (
            [primary, preflight.EXECUTOR_UID]
            if name == "muncho-passkey-web"
            else [primary]
        ),
    )
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_identity_invalid",
    ):
        producer._identity_facts()


def test_host_frame_digest_binds_request_and_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        producer,
        "build_host_observation",
        lambda request, **_kwargs: {"request": request},
    )
    unsigned = {
        "schema": producer.HOST_FRAME_SCHEMA,
        "request": {"bound": "request"},
        "attached_sa_probe": {"bound": "sidecar"},
    }
    frame = {**unsigned, "frame_sha256": foundation.sha256_json(unsigned)}
    assert producer._run_host_frame(
        frame, release_revision=REVISION, now_unix=NOW
    ) == {"request": {"bound": "request"}}
    tampered = {**frame, "attached_sa_probe": {"bound": "substituted"}}
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_frame_invalid",
    ):
        producer._run_host_frame(
            tampered, release_revision=REVISION, now_unix=NOW
        )


def test_request_accepts_schema_neutral_binding_but_rejects_terminal_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package()
    install = {"receipt_sha256": "d" * 64}
    request = _lineage_request()
    monkeypatch.setattr(producer, "_load_release_package", lambda _revision: package)
    monkeypatch.setattr(
        producer,
        "_validate_install_receipt",
        lambda *_a, **_k: install,
    )
    assert producer._validate_request(
        request,
        schema=producer.ATTACHED_SA_REQUEST_SCHEMA,
        release_revision=REVISION,
        now_unix=NOW,
    ) == (request, package, install)
    host_request = {**request, "schema": producer.HOST_REQUEST_SCHEMA}
    host_request["request_sha256"] = foundation.sha256_json({
        name: item for name, item in host_request.items()
        if name != "request_sha256"
    })
    assert producer._validate_request(
        host_request,
        schema=producer.HOST_REQUEST_SCHEMA,
        release_revision=REVISION,
        now_unix=NOW,
    ) == (host_request, package, install)
    changed = {**host_request, "inert_terminal": {"caller": "untrusted"}}
    changed["observation_binding_sha256"] = foundation.sha256_json({
        name: item for name, item in changed.items()
        if name not in {"schema", "observation_binding_sha256", "request_sha256"}
    })
    changed["request_sha256"] = foundation.sha256_json({
        name: item for name, item in changed.items() if name != "request_sha256"
    })
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_request_invalid",
    ):
        producer._validate_request(
            changed,
            schema=producer.HOST_REQUEST_SCHEMA,
            release_revision=REVISION,
            now_unix=NOW,
        )


@pytest.mark.parametrize("mutation", ("missing", "malformed"))
def test_request_requires_valid_production_ingress_observation_digest(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    request = _lineage_request()
    if mutation == "missing":
        request.pop("production_ingress_observation_sha256")
    else:
        request["production_ingress_observation_sha256"] = "not-a-digest"
    request["observation_binding_sha256"] = foundation.sha256_json({
        name: item
        for name, item in request.items()
        if name not in {
            "schema", "observation_binding_sha256", "request_sha256"
        }
    })
    request["request_sha256"] = foundation.sha256_json({
        name: item for name, item in request.items() if name != "request_sha256"
    })
    monkeypatch.setattr(producer, "_load_release_package", lambda _revision: _package())
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_request_invalid",
    ):
        producer._validate_request(
            request,
            schema=producer.ATTACHED_SA_REQUEST_SCHEMA,
            release_revision=REVISION,
            now_unix=NOW,
        )


def test_attached_sa_probe_is_byte_stable_and_binds_all_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _lineage_request()
    package = _package()
    install = {"receipt_sha256": "e" * 64}
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    ordering: list[str] = []
    direct_identity = _direct_identity()
    monkeypatch.setattr(
        producer,
        "_validate_request",
        lambda *_a, **_k: (request, package, install),
    )
    monkeypatch.setattr(
        producer,
        "_read_regular",
        lambda *_a, **_k: DIRECT_IAM_RAW,
    )
    monkeypatch.setattr(
        producer.direct_iam,
        "decode_canonical",
        lambda *_a, **_k: direct_identity,
    )
    def collect_child(**_kwargs: Any) -> Mapping[str, Any]:
        ordering.append("executor_uid_child")
        return _facts()

    monkeypatch.setattr(producer, "_collect_attached_sa_child", collect_child)
    def load_signer(_revision: str) -> tuple[Any, ...]:
        ordering.append("host_private_signer")
        return (
            key,
            key.public_key(),
            key_id,
            {
                "provisioning_receipt_sha256": request[
                    "host_signer_provisioning_receipt_sha256"
                ],
                "readiness_sha256": request["host_signer_readiness_sha256"],
            },
        )

    monkeypatch.setattr(producer, "_load_host_signer", load_signer)
    monkeypatch.setattr(
        producer.provisioning,
        "verify_cloud_signer_inert_readiness",
        lambda _revision: {
            "provisioning_receipt_sha256": request[
                "cloud_signer_provisioning_receipt_sha256"
            ],
            "readiness_sha256": request["cloud_signer_readiness_sha256"],
        },
    )
    first = producer.build_attached_sa_permission_probe(
        request, release_revision=REVISION, now_unix=NOW
    )
    second = producer.build_attached_sa_permission_probe(
        request, release_revision=REVISION, now_unix=NOW
    )
    assert ordering == [
        "executor_uid_child", "host_private_signer",
        "executor_uid_child", "host_private_signer",
    ]
    assert producer._canonical(first) == producer._canonical(second)
    for name in producer._SIGNER_LINEAGE_FIELDS:
        assert first[name] == request[name]
    host_request = dict(request)
    host_request["schema"] = producer.HOST_REQUEST_SCHEMA
    host_request["request_sha256"] = foundation.sha256_json({
        name: item
        for name, item in host_request.items()
        if name != "request_sha256"
    })
    assert producer._validate_attached_probe(
        first,
        request=host_request,
        package=package,
        install=install,
        direct_identity=direct_identity,
        host_public_key=key.public_key(),
        host_key_id=key_id,
    ) == first
    tamper_cases = {
        "install_receipt_sha256": "f" * 64,
        "runtime_instance_numeric_id": "2234567890123456789",
        "runtime_service_account_email": (
            "substituted@adventico-ai-platform.iam.gserviceaccount.com"
        ),
        "runtime_service_account_unique_id": "223456789012345678901",
        "metadata_scopes": list(reversed(foundation.OWNER_GATE_OAUTH_SCOPES)),
    }
    for field, value in tamper_cases.items():
        unsigned = {
            name: item
            for name, item in first.items()
            if name not in {"report_sha256", "attestation"}
        }
        unsigned[field] = value
        re_signed = producer._attest(
            unsigned, private_key=key, public_key_id=key_id
        )
        with pytest.raises(
            producer.OwnerGateHostObservationError,
            match="owner_gate_attached_sa_probe_invalid",
        ):
            producer._validate_attached_probe(
                re_signed,
                request=host_request,
                package=package,
                install=install,
                direct_identity=direct_identity,
                host_public_key=key.public_key(),
                host_key_id=key_id,
            )
    substituted = dict(host_request)
    substituted["plan_sha256"] = "f" * 64
    substituted["observation_binding_sha256"] = foundation.sha256_json({
        name: item
        for name, item in substituted.items()
        if name not in {
            "schema", "observation_binding_sha256", "request_sha256"
        }
    })
    substituted["request_sha256"] = foundation.sha256_json({
        name: item
        for name, item in substituted.items()
        if name != "request_sha256"
    })
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_attached_sa_probe_invalid",
    ):
        producer._validate_attached_probe(
            first,
            request=substituted,
            package=package,
            install=install,
            direct_identity=direct_identity,
            host_public_key=key.public_key(),
            host_key_id=key_id,
        )


def test_host_observation_is_byte_stable_and_binds_verified_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _lineage_request()
    request["schema"] = producer.HOST_REQUEST_SCHEMA
    package = _package()
    install = {"receipt_sha256": "e" * 64}
    sidecar = {
        "report_sha256": "f" * 64,
        "effective_permission_probe": preflight.expected_effective_permission_probe(
            False
        ),
        "numeric_targets_reverified": False,
    }
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    monkeypatch.setattr(
        producer,
        "_validate_request",
        lambda *_a, **_k: (request, package, install),
    )
    monkeypatch.setattr(
        producer,
        "_load_host_signer",
        lambda _revision: (
            key,
            key.public_key(),
            key_id,
            {
                "provisioning_receipt_sha256": request[
                    "host_signer_provisioning_receipt_sha256"
                ],
                "readiness_sha256": request["host_signer_readiness_sha256"],
            },
        ),
    )
    monkeypatch.setattr(
        producer.provisioning,
        "verify_cloud_signer_inert_readiness",
        lambda _revision: {
            "provisioning_receipt_sha256": request[
                "cloud_signer_provisioning_receipt_sha256"
            ],
            "readiness_sha256": request["cloud_signer_readiness_sha256"],
        },
    )
    monkeypatch.setattr(
        producer, "_load_direct_iam_identity", lambda _revision: {}
    )
    monkeypatch.setattr(
        producer, "_validate_attached_probe", lambda *_a, **_k: sidecar
    )
    monkeypatch.setattr(producer, "_sqlite_facts", lambda: ({"sqlite": True}, {"migration": True}))
    security = {
        "webauthn": {"selftest": True},
        "public_web_can_author_envelope": False,
        "authorization_receipt_signature_self_verified": True,
        "receipt_action_binding_self_verified": True,
    }
    monkeypatch.setattr(producer, "_runtime_security_selftest", lambda: security)
    monkeypatch.setattr(
        producer,
        "_executor_receipt_key_facts",
        lambda: {
            "receipt_public_key_sha256": "0" * 64,
            "receipt_public_key_owner": "root:root",
            "receipt_public_key_mode": "0444",
            "receipt_public_key_id": "1" * 64,
        },
    )
    monkeypatch.setattr(producer.os.path, "lexists", lambda _path: False)
    seal_checks: list[bool] = []
    monkeypatch.setattr(
        producer,
        "_require_executor_activation_seal_absent",
        lambda: seal_checks.append(True),
    )
    monkeypatch.setattr(
        producer,
        "_host_release_facts",
        lambda *_a: {
            "attached_sa_permission_probe_report_sha256": sidecar[
                "report_sha256"
            ]
        },
    )
    monkeypatch.setattr(producer, "_identity_facts", lambda: {"identity": True})
    monkeypatch.setattr(producer, "_socket_facts", lambda _release: {"socket": True})
    monkeypatch.setattr(producer, "_unit_properties", lambda _release: {"unit": True})
    monkeypatch.setattr(
        producer,
        "_firewall_facts",
        lambda **_kwargs: {"firewall": True},
    )
    first = producer.build_host_observation(
        request,
        release_revision=REVISION,
        attached_sa_probe=sidecar,
        now_unix=NOW,
    )
    second = producer.build_host_observation(
        request,
        release_revision=REVISION,
        attached_sa_probe=sidecar,
        now_unix=NOW,
    )
    assert producer._canonical(first) == producer._canonical(second)
    assert first["effective_permission_probe"] == sidecar[
        "effective_permission_probe"
    ]
    assert first["release"][
        "attached_sa_permission_probe_report_sha256"
    ] == sidecar["report_sha256"]
    assert (
        first["production_ingress_observation_sha256"]
        == PRODUCTION_INGRESS_OBSERVATION_SHA256
    )
    assert seal_checks == [True] * 6


@pytest.mark.parametrize("kind", ("regular", "broken-symlink"))
def test_executor_activation_seal_presence_is_never_reported_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    seal = tmp_path / "storage-executor-enabled"
    if kind == "regular":
        seal.write_bytes(b"enabled\n")
    else:
        seal.symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(producer.service, "ACTIVATION_SEAL", seal)

    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_executor_activation_seal_present",
    ):
        producer._require_executor_activation_seal_absent()


def test_late_host_fact_cannot_run_after_signed_completion_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _lineage_request()
    request["schema"] = producer.HOST_REQUEST_SCHEMA
    package_manifest = _package()
    install = {"receipt_sha256": "e" * 64}
    sidecar = {
        "report_sha256": "f" * 64,
        "effective_permission_probe": preflight.expected_effective_permission_probe(
            False
        ),
        "numeric_targets_reverified": False,
    }
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    monkeypatch.setattr(
        producer,
        "_validate_request",
        lambda *_a, **_k: (request, package_manifest, install),
    )
    monkeypatch.setattr(
        producer,
        "_load_host_signer",
        lambda _revision: (
            key,
            key.public_key(),
            key_id,
            {
                "provisioning_receipt_sha256": request[
                    "host_signer_provisioning_receipt_sha256"
                ],
                "readiness_sha256": request["host_signer_readiness_sha256"],
            },
        ),
    )
    monkeypatch.setattr(
        producer.provisioning,
        "verify_cloud_signer_inert_readiness",
        lambda _revision: {
            "provisioning_receipt_sha256": request[
                "cloud_signer_provisioning_receipt_sha256"
            ],
            "readiness_sha256": request["cloud_signer_readiness_sha256"],
        },
    )
    monkeypatch.setattr(
        producer, "_load_direct_iam_identity", lambda _revision: {}
    )
    monkeypatch.setattr(
        producer, "_validate_attached_probe", lambda *_a, **_k: sidecar
    )
    monkeypatch.setattr(
        producer, "_sqlite_facts", lambda: ({"sqlite": True}, {"migration": True})
    )
    monkeypatch.setattr(
        producer,
        "_runtime_security_selftest",
        lambda: {
            "webauthn": {"selftest": True},
            "public_web_can_author_envelope": False,
            "authorization_receipt_signature_self_verified": True,
            "receipt_action_binding_self_verified": True,
        },
    )
    monkeypatch.setattr(
        producer,
        "_executor_receipt_key_facts",
        lambda: {
            "receipt_public_key_sha256": "0" * 64,
            "receipt_public_key_owner": "root:root",
            "receipt_public_key_mode": "0444",
            "receipt_public_key_id": "1" * 64,
        },
    )
    monkeypatch.setattr(producer.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(producer, "_host_release_facts", lambda *_a: {})
    monkeypatch.setattr(producer, "_identity_facts", lambda: {})
    monkeypatch.setattr(producer, "_socket_facts", lambda _release: {})
    monkeypatch.setattr(producer, "_unit_properties", lambda _release: {})
    late_fact_collected: list[bool] = []
    monkeypatch.setattr(
        producer,
        "_firewall_facts",
        lambda **_kwargs: late_fact_collected.append(True) or {},
    )
    attest_called: list[bool] = []
    monkeypatch.setattr(
        producer,
        "_attest",
        lambda *_args, **_kwargs: attest_called.append(True) or {},
    )
    ticks = iter((100.0, 161.0))
    monkeypatch.setattr(producer.time, "monotonic", lambda: next(ticks))
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_clock_invalid",
    ):
        producer.build_host_observation(
            request,
            release_revision=REVISION,
            attached_sa_probe=sidecar,
            now_unix=NOW,
        )
    assert late_fact_collected == [True]
    assert attest_called == []


def test_real_release_projection_matches_preflight_entrypoint_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases = tmp_path / "releases"
    root = releases / REVISION
    (root / "venv/bin").mkdir(parents=True)
    (root / "bin").mkdir()
    python = root / "venv/bin/python"
    python.write_bytes(b"fixed-test-python")
    python.chmod(0o555)
    for name in preflight.HOST_RELEASE_ENTRYPOINTS:
        entrypoint = root / "bin" / name
        entrypoint.write_bytes(f"fixed:{name}".encode("ascii"))
        entrypoint.chmod(0o555)
    root.chmod(0o555)
    monkeypatch.setattr(producer, "OWNER_RELEASE_BASE", releases)
    package = _package()
    request = _lineage_request()
    install = {"receipt_sha256": "e" * 64}
    sidecar = {"report_sha256": "f" * 64}
    root_state = root.lstat()
    projection = producer._host_release_facts(
        package,
        install,
        request,
        sidecar,
        expected_uid=root_state.st_uid,
        expected_gid=root_state.st_gid,
    )
    assert projection["entrypoints"] == list(
        preflight.HOST_RELEASE_ENTRYPOINTS
    )
    assert projection["observation_dispatcher_schemas"] == list(
        preflight.HOST_OBSERVATION_DISPATCHER_SCHEMAS
    )
    assert projection[
        "attached_sa_permission_probe_report_sha256"
    ] == sidecar["report_sha256"]


def test_existing_wrapper_dispatches_schema_without_sudoers_expansion() -> None:
    repository = Path(__file__).resolve().parents[3]
    wrapper = (
        repository
        / "ops/muncho/owner-gate/bin/muncho-host-observation-attestor"
    ).read_text(encoding="utf-8")
    sudoers = (
        repository
        / "ops/muncho/owner-gate/muncho-host-observation-attestor.sudoers.in"
    ).read_text(encoding="utf-8")
    assert "owner_gate_host_observation_producer" in wrapper
    assert "observation_dispatcher_main" in wrapper
    assert "verify_host_signer_runtime_readiness" not in wrapper
    assert sudoers.count("MUNCHO_HOST_OBSERVATION_ATTESTOR") == 2
    assert "ATTACHED_SA" not in sudoers
    assert "HOST_OBSERVATION_PRODUCER" not in sudoers
    assert (
        "scripts/canary/owner_gate_host_observation_producer.py"
        in package.ROOT_RUNTIME_FILES
    )
    assert (
        "ops/muncho/owner-gate/bin/muncho-host-observation-attestor"
        in package.REQUIRED_ASSET_FILES
    )


def test_stalled_attached_sa_collection_fails_before_private_signer_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _lineage_request()
    package_manifest = _package()
    install = {"receipt_sha256": "e" * 64}
    monkeypatch.setattr(
        producer,
        "_validate_request",
        lambda *_args, **_kwargs: (request, package_manifest, install),
    )
    monkeypatch.setattr(
        producer,
        "_read_regular",
        lambda *_a, **_k: DIRECT_IAM_RAW,
    )
    monkeypatch.setattr(
        producer.direct_iam,
        "decode_canonical",
        lambda *_a, **_k: _direct_identity(),
    )
    monkeypatch.setattr(
        producer, "_collect_attached_sa_child", lambda **_kwargs: _facts()
    )
    signer_loaded: list[bool] = []
    monkeypatch.setattr(
        producer,
        "_load_host_signer",
        lambda _revision: signer_loaded.append(True),
    )
    ticks = iter((100.0, 161.0))
    monkeypatch.setattr(producer.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(producer.time, "time", lambda: float(NOW))
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_clock_invalid",
    ):
        producer.build_attached_sa_permission_probe(
            request, release_revision=REVISION, now_unix=None
        )
    assert signer_loaded == []


def test_delayed_signer_readiness_cannot_escape_final_completion_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _lineage_request()
    package_manifest = _package()
    install = {"receipt_sha256": "e" * 64}
    key = Ed25519PrivateKey.generate()
    key_id = hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
    monkeypatch.setattr(
        producer,
        "_validate_request",
        lambda *_args, **_kwargs: (request, package_manifest, install),
    )
    monkeypatch.setattr(
        producer,
        "_read_regular",
        lambda *_a, **_k: DIRECT_IAM_RAW,
    )
    monkeypatch.setattr(
        producer.direct_iam,
        "decode_canonical",
        lambda *_a, **_k: _direct_identity(),
    )
    monkeypatch.setattr(
        producer, "_collect_attached_sa_child", lambda **_kwargs: _facts()
    )
    monkeypatch.setattr(
        producer,
        "_load_host_signer",
        lambda _revision: (
            key,
            key.public_key(),
            key_id,
            {
                "provisioning_receipt_sha256": request[
                    "host_signer_provisioning_receipt_sha256"
                ],
                "readiness_sha256": request["host_signer_readiness_sha256"],
            },
        ),
    )
    monkeypatch.setattr(
        producer.provisioning,
        "verify_cloud_signer_inert_readiness",
        lambda _revision: {
            "provisioning_receipt_sha256": request[
                "cloud_signer_provisioning_receipt_sha256"
            ],
            "readiness_sha256": request["cloud_signer_readiness_sha256"],
        },
    )
    ticks = iter((100.0, 100.0, 161.0))
    monkeypatch.setattr(producer.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(producer.time, "time", lambda: float(NOW))
    with pytest.raises(
        producer.OwnerGateHostObservationError,
        match="owner_gate_host_observation_clock_invalid",
    ):
        producer.build_attached_sa_permission_probe(
            request, release_revision=REVISION, now_unix=None
        )


def test_executor_config_key_claims_match_fixed_root_owned_pem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = Ed25519PrivateKey.generate().public_key()
    pem = key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    digest = hashlib.sha256(pem).hexdigest()
    config = dict.fromkeys(producer.service._EXECUTOR_CONFIG_FIELDS)
    config.update({
        "schema": "muncho-owner-gate-executor-config.v1",
        "api_host": "compute.googleapis.com",
        "api_private_vip_range": "199.36.153.8/30",
        "expected_disk_id": foundation.TARGET_DISK_ID,
        "expected_instance_id": foundation.TARGET_INSTANCE_ID,
        "executor_database": str(producer.EXECUTOR_DB),
        "journal_root": str(producer.EXECUTOR_DB.parent),
        "mutation_enable_seal": str(producer.service.ACTIVATION_SEAL),
        "mutation_enable_seal_uid": 0,
        "mutation_enable_seal_gid": preflight.EXECUTOR_UID,
        "mutation_enable_seal_mode": "0440",
        "metadata_host": producer.METADATA_HOST,
        "firewall_readiness_receipt": str(producer.FIREWALL_RECEIPT),
        "firewall_readiness_receipt_uid": 0,
        "firewall_readiness_receipt_gid": preflight.EXECUTOR_UID,
        "firewall_readiness_receipt_mode": "0440",
        "firewall_readiness_max_age_seconds": 60,
        "firewall_readiness_requires_current_boot_id": True,
        "firewall_readiness_requires_rules_source_sha256": True,
        "project": foundation.PROJECT,
        "target_disk": foundation.TARGET_DISK,
        "target_instance": foundation.TARGET_INSTANCE,
        "target_boot_device": foundation.TARGET_BOOT_DEVICE,
        "zone": foundation.ZONE,
        "cloud_observation_public_key": str(
            producer.service.CLOUD_OBSERVATION_PUBLIC_KEY
        ),
        "host_observation_public_key": str(
            producer.service.HOST_OBSERVATION_PUBLIC_KEY
        ),
        "receipt_public_key": str(producer.AUTHORITY_RECEIPT_PUBLIC_KEY),
        "receipt_public_key_owner": "root:root",
        "receipt_public_key_mode": "0444",
        "receipt_public_key_sha256": digest,
        "signed_authorization_receipt_required": True,
        "topology_iam_readiness_seal_required_for_mutation_only": True,
    })
    monkeypatch.setattr(producer, "_config_mapping", lambda _path: config)
    monkeypatch.setattr(
        producer.service, "_direct_iam_pins", lambda _config: {"valid": True}
    )
    monkeypatch.setattr(
        producer, "_load_authority_receipt_key", lambda: (key, pem)
    )
    facts = producer._executor_receipt_key_facts()
    assert facts == {
        "receipt_public_key_sha256": digest,
        "receipt_public_key_owner": "root:root",
        "receipt_public_key_mode": "0444",
        "receipt_public_key_id": hashlib.sha256(
            key.public_bytes_raw()
        ).hexdigest(),
    }
    for field, changed in (
        ("receipt_public_key", "/tmp/substituted.pem"),
        ("receipt_public_key_sha256", "f" * 64),
        ("receipt_public_key_owner", "root:29103"),
        ("receipt_public_key_mode", "0440"),
    ):
        tampered = {**config, field: changed}
        monkeypatch.setattr(
            producer, "_config_mapping", lambda _path, value=tampered: value
        )
        with pytest.raises(
            producer.OwnerGateHostObservationError,
            match="owner_gate_host_executor_invalid",
        ):
            producer._executor_receipt_key_facts()
