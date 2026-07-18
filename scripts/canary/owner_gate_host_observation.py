#!/usr/bin/env python3
"""Fixed, release-bound owner-gate host and attached-SA observations.

The two public entrypoints in this module accept one canonical request on
stdin.  They have no caller-selected path, URL, command, credential, or
network surface.  Host facts are recollected from the installed immutable
release and fixed root-owned state.  Effective permissions are recollected in
an executor-UID child using only the VM metadata identity and four exact
``testIamPermissions`` requests.  Every response is self-hashed and signed by
the separately provisioned release host-observation key.

No function in this module installs firewall rules, selects a release, starts
or enables a service, mutates IAM, changes storage, or touches Caddy.
"""

from __future__ import annotations

import base64
import errno
import grp
import hashlib
import http.client
import json
import math
import os
import pwd
import re
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, Mapping, NoReturn, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import owner_gate_firewall_readiness as firewall
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from scripts.canary import passkey_v2_signer as signer_runtime
from scripts.canary import passkey_v2_sqlite as sqlite_backend
from scripts.canary import passkey_v2_webauthn as webauthn
from scripts.canary import storage_growth_trusted_collector as trusted
from scripts.canary import trusted_signer_provisioning as provisioning


HOST_REQUEST_SCHEMA = "muncho-owner-gate-host-observation-request.v1"
HOST_FRAME_SCHEMA = "muncho-owner-gate-host-observation-frame.v1"
ATTACHED_SA_REQUEST_SCHEMA = (
    "muncho-owner-gate-attached-sa-permission-probe-request.v1"
)
ATTACHED_SA_PROBE_SCHEMA = (
    "muncho-owner-gate-attached-sa-permission-probe.v1"
)
OBSERVATION_ATTESTATION_SCHEMA = (
    "muncho-owner-gate-observation-attestation.v1"
)

OWNER_RELEASE_BASE = Path("/opt/muncho-owner-gate/releases")
INSTALL_RECEIPT_BASE = Path(
    "/var/lib/muncho-owner-gate/bootstrap-receipts"
)
AUTHORITY_RECEIPT_PUBLIC_KEY = Path(
    "/etc/muncho-owner-gate/public/authority-receipt-public.pem"
)
FIREWALL_RECEIPT = Path(
    "/run/muncho-owner-gate/metadata-firewall-ready.json"
)
EXECUTOR_CONFIG = Path("/etc/muncho-owner-gate/executor.json")
AUTHORITY_DB = Path(preflight.AUTHORITY_DB)
EXECUTOR_DB = Path(preflight.EXECUTOR_DB)

METADATA_HOST = "169.254.169.254"
METADATA_INSTANCE_ID_PATH = "/computeMetadata/v1/instance/id"
METADATA_SERVICE_ACCOUNT_EMAIL_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/email"
)
METADATA_SCOPES_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/scopes"
)
METADATA_TOKEN_PATH = (
    "/computeMetadata/v1/instance/service-accounts/default/token"
)
PRIVATE_GOOGLE_API_VIP = "199.36.153.8"

MAX_REQUEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_HTTP_BYTES = 512 * 1024
MAX_COMMAND_BYTES = 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 20
HTTP_TIMEOUT_SECONDS = 10
FRESHNESS_SECONDS = preflight.HOST_OBSERVATION_FRESHNESS_SECONDS
SELFTEST_BASE = Path("/run")
ATTACHED_SA_CHILD_SCHEMA = "muncho-owner-gate-attached-sa-child.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_TOKEN_BYTE = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~+/="
)

_INSTALL_RECEIPT_FIELDS = frozenset({
    "schema", "release_revision", "package_sha256", "source_tree_oid",
    "pre_foundation_authority_sha256", "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256", "project_ancestry_chain_sha256",
    "resource_ancestor_chain", "installed_at_unix", "release_path",
    "release_tree_sha256", "transaction_prefix_sha256",
    "phase_evidence_sha256", "authority_receipt_public_key_sha256",
    "authority_receipt_public_key_id", "credential_id_sha256",
    "executor_hosts_receipt_sha256", "current_release_selected",
    "systemd_units_enabled", "activation_performed", "activation_seal_created",
    "iam_binding_created", "cloud_mutation_performed",
    "caddy_cutover_performed", "receipt_sha256", "signer_key_id",
    "signature_ed25519_b64url",
})

_LINEAGE_FIELDS = (
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
)
_SIGNER_LINEAGE_FIELDS = (
    "cloud_signer_provisioning_receipt_sha256",
    "cloud_signer_readiness_sha256",
    "host_signer_provisioning_receipt_sha256",
    "host_signer_readiness_sha256",
)

_UNIT_NAMES = {
    "web": "muncho-passkey-web.service",
    "authority": "muncho-passkey-authority.service",
    "executor": "muncho-privileged-executor.service",
}
_UNIT_ASSETS = {
    name: f"ops/muncho/owner-gate/{unit}"
    for name, unit in _UNIT_NAMES.items()
}
_SOCKET_ASSETS = {
    "web_authority": (
        "muncho-passkey-authority.socket",
        foundation.PASSKEY_AUTHORITY_SOCKET,
        preflight.AUTHORITY_UID,
        preflight.WEB_UID,
    ),
    "authority_executor": (
        "muncho-privileged-executor.socket",
        foundation.PRIVILEGED_EXECUTOR_SOCKET,
        preflight.EXECUTOR_UID,
        preflight.AUTHORITY_UID,
    ),
}


class OwnerGateHostObservationError(RuntimeError):
    """Stable, secret-free host observation failure."""


def _error(code: str, _exc: BaseException | None = None) -> NoReturn:
    raise OwnerGateHostObservationError(code) from None


def _require_executor_activation_seal_absent() -> None:
    if os.path.lexists(service.ACTIVATION_SEAL):
        _error("owner_gate_host_executor_activation_seal_present")


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError:
        _error("owner_gate_host_observation_json_invalid")


def _sha(value: Any) -> str:
    return foundation.sha256_json(value)


def _decode_canonical(raw: bytes, *, maximum: int, code: str) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _error(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise ValueError("duplicate")
            value[name] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
            parse_float=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _error(code)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error(code)
    return dict(value)


def _decode_json_response(
    raw: bytes, *, maximum: int, code: str
) -> Mapping[str, Any]:
    """Decode ordinary API JSON while rejecting ambiguous JSON domains."""

    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _error(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in items:
            if name in value:
                raise ValueError("duplicate")
            value[name] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
            parse_float=lambda _item: (_ for _ in ()).throw(ValueError()),
        )
        # Enforce the same bounded integer/string/container domain before any
        # response can be signed, without requiring Google to emit canonical
        # whitespace or key ordering on the wire.
        _canonical(value)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _error(code)
    if not isinstance(value, Mapping):
        _error(code)
    return dict(value)


def _decode_chunked_body(raw: bytes) -> bytes:
    """Decode one bounded RFC 9112 chunk stream; trailers are forbidden."""

    position = 0
    decoded = bytearray()
    while True:
        line_end = raw.find(b"\r\n", position)
        if line_end < position or line_end - position not in range(1, 17):
            _error("owner_gate_attached_sa_api_response_invalid")
        size_raw = raw[position:line_end]
        if re.fullmatch(rb"[0-9A-Fa-f]+", size_raw) is None:
            _error("owner_gate_attached_sa_api_response_invalid")
        size = int(size_raw, 16)
        position = line_end + 2
        if size > MAX_HTTP_BYTES - len(decoded):
            _error("owner_gate_attached_sa_api_response_invalid")
        data_end = position + size
        if data_end + 2 > len(raw) or raw[data_end:data_end + 2] != b"\r\n":
            _error("owner_gate_attached_sa_api_response_invalid")
        decoded.extend(raw[position:data_end])
        position = data_end + 2
        if size == 0:
            if position != len(raw) or not decoded:
                _error("owner_gate_attached_sa_api_response_invalid")
            return bytes(decoded)


def _parse_api_response(raw: bytes) -> Mapping[str, Any]:
    if not raw or len(raw) > MAX_HTTP_BYTES:
        _error("owner_gate_attached_sa_api_response_invalid")
    head_end = raw.find(b"\r\n\r\n")
    if head_end < 0:
        _error("owner_gate_attached_sa_api_response_invalid")
    try:
        head = raw[:head_end].decode("ascii", errors="strict")
    except UnicodeError:
        _error("owner_gate_attached_sa_api_response_invalid")
    lines = head.split("\r\n")
    if lines[0] != "HTTP/1.1 200 OK":
        _error("owner_gate_attached_sa_api_response_invalid")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            _error("owner_gate_attached_sa_api_response_invalid")
        name, item = line.split(":", 1)
        lowered = name.lower()
        if (
            re.fullmatch(r"[A-Za-z0-9-]+", name) is None
            or "\x00" in item
        ):
            _error("owner_gate_attached_sa_api_response_invalid")
        if lowered in headers:
            if lowered != "vary":
                _error("owner_gate_attached_sa_api_response_invalid")
            # Google emits multiple Vary fields.  It is explicitly repeatable
            # and does not affect framing; every other duplicate fails closed.
            continue
        headers[lowered] = item.strip()
    body_wire = raw[head_end + 4 :]
    transfer_encoding = headers.get("transfer-encoding")
    content_length = headers.get("content-length")
    if (transfer_encoding is None) == (content_length is None):
        _error("owner_gate_attached_sa_api_response_invalid")
    if transfer_encoding is not None:
        if transfer_encoding.lower() != "chunked":
            _error("owner_gate_attached_sa_api_response_invalid")
        body_raw = _decode_chunked_body(body_wire)
    else:
        if (
            re.fullmatch(r"0|[1-9][0-9]{0,6}", str(content_length))
            is None
            or int(str(content_length)) != len(body_wire)
            or not body_wire
        ):
            _error("owner_gate_attached_sa_api_response_invalid")
        body_raw = body_wire
    return _decode_json_response(
        body_raw,
        maximum=MAX_HTTP_BYTES,
        code="owner_gate_attached_sa_api_response_invalid",
    )


def _read_stdin(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(MAX_REQUEST_BYTES + 2)
    if (
        not raw
        or len(raw) > MAX_REQUEST_BYTES + 1
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        _error("owner_gate_host_observation_request_invalid")
    return _decode_canonical(
        raw[:-1],
        maximum=MAX_REQUEST_BYTES,
        code="owner_gate_host_observation_request_invalid",
    )


def _write_stdout(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > MAX_REQUEST_BYTES + 1:
        _error("owner_gate_host_observation_response_invalid")
    stream.write(raw)
    stream.flush()


def _identity(state: os.stat_result) -> tuple[int, ...]:
    return (
        state.st_mode, state.st_uid, state.st_gid, state.st_dev, state.st_ino,
        state.st_nlink, state.st_size, state.st_mtime_ns, state.st_ctime_ns,
    )


def _read_regular(
    path: Path,
    *,
    maximum: int = MAX_FILE_BYTES,
    uid: int = 0,
    gid: int | None = None,
    modes: frozenset[int] = frozenset({0o400, 0o440, 0o444}),
) -> bytes:
    descriptor = -1
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or _identity(before) != _identity(opened)
            or opened.st_uid != uid
            or (gid is not None and opened.st_gid != gid)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) not in modes
            or not 0 < opened.st_size <= maximum
        ):
            _error("owner_gate_host_observation_file_invalid")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        if (
            len(raw) != opened.st_size
            or len(raw) > maximum
            or _identity(after) != _identity(opened)
            or _identity(path_after) != _identity(before)
        ):
            _error("owner_gate_host_observation_file_changed")
        return bytes(raw)
    except OwnerGateHostObservationError:
        raise
    except OSError:
        _error("owner_gate_host_observation_file_unavailable")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) != 86 or "=" in value:
        _error("owner_gate_host_observation_signature_invalid")
    try:
        raw = base64.urlsafe_b64decode(value + "==")
    except (TypeError, ValueError):
        _error("owner_gate_host_observation_signature_invalid")
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        _error("owner_gate_host_observation_signature_invalid")
    return raw


def _load_authority_receipt_key() -> tuple[Ed25519PublicKey, bytes]:
    raw = _read_regular(
        AUTHORITY_RECEIPT_PUBLIC_KEY,
        maximum=4096,
        gid=0,
        modes=frozenset({0o444}),
    )
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError):
        _error("owner_gate_host_install_receipt_key_invalid")
    if not isinstance(key, Ed25519PublicKey):
        _error("owner_gate_host_install_receipt_key_invalid")
    return key, raw


def _validate_install_receipt(
    value: Any,
    *,
    package: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _INSTALL_RECEIPT_FIELDS:
        _error("owner_gate_host_install_receipt_invalid")
    receipt = dict(value)
    release = str(package["release_revision"])
    lineage = {name: package[name] for name in _LINEAGE_FIELDS}
    signed = {
        name: item
        for name, item in receipt.items()
        if name not in {"signer_key_id", "signature_ed25519_b64url"}
    }
    unsigned = {name: item for name, item in signed.items() if name != "receipt_sha256"}
    public_key, public_raw = _load_authority_receipt_key()
    signature = _decode_signature(receipt.get("signature_ed25519_b64url"))
    false_fields = (
        "current_release_selected", "activation_performed", "activation_seal_created",
        "iam_binding_created", "cloud_mutation_performed", "caddy_cutover_performed",
    )
    if (
        receipt.get("schema") != "muncho-owner-gate-offline-install-receipt.v1"
        or receipt.get("release_revision") != release
        or receipt.get("source_tree_oid") != package.get("source_tree_oid")
        or receipt.get("package_sha256") != package.get("package_sha256")
        or any(receipt.get(name) != expected for name, expected in lineage.items())
        or receipt.get("resource_ancestor_chain")
        != package.get("resource_ancestor_chain")
        or receipt.get("release_path") != str(OWNER_RELEASE_BASE / release)
        or receipt.get("systemd_units_enabled") != []
        or any(receipt.get(name) is not False for name in false_fields)
        or receipt.get("receipt_sha256") != _sha(unsigned)
        or receipt.get("signer_key_id")
        != receipt.get("authority_receipt_public_key_id")
        or receipt.get("authority_receipt_public_key_sha256")
        != hashlib.sha256(public_raw).hexdigest()
        or receipt.get("authority_receipt_public_key_id")
        != hashlib.sha256(public_key.public_bytes_raw()).hexdigest()
    ):
        _error("owner_gate_host_install_receipt_invalid")
    try:
        public_key.verify(signature, _canonical(signed))
    except InvalidSignature:
        _error("owner_gate_host_install_receipt_signature_invalid")
    path = INSTALL_RECEIPT_BASE / f"install-{release}.json"
    installed_raw = _read_regular(path, maximum=MAX_FILE_BYTES, modes=frozenset({0o400}))
    if installed_raw != _canonical(receipt):
        _error("owner_gate_host_install_receipt_file_mismatch")
    return receipt


def _load_release_package(release_revision: str) -> Mapping[str, Any]:
    if _REVISION.fullmatch(release_revision) is None:
        _error("owner_gate_host_release_invalid")
    root = OWNER_RELEASE_BASE / release_revision
    raw = _read_regular(root / "package-manifest.json", modes=frozenset({0o444}))
    package = _decode_canonical(
        raw, maximum=stage0.MAX_JSON_BYTES, code="owner_gate_host_release_invalid"
    )
    if (
        frozenset(package) != stage0.MANIFEST_FIELDS
        or package.get("schema") != stage0.PACKAGE_SCHEMA
        or package.get("release_revision") != release_revision
        or package.get("release_root") != str(root)
        or package.get("package_sha256")
        != _sha({name: item for name, item in package.items() if name != "package_sha256"})
        or package.get("activation_performed") is not False
        or package.get("cloud_mutation_performed") is not False
        or package.get("generic_shell_entrypoint") is not False
        or package.get("local_gcloud_runtime_fallback") is not False
    ):
        _error("owner_gate_host_release_invalid")
    return package


def _load_direct_iam_identity(
    package: Mapping[str, Any],
) -> Mapping[str, Any]:
    release_revision = str(package.get("release_revision", ""))
    if _REVISION.fullmatch(release_revision) is None:
        _error("owner_gate_attached_sa_direct_identity_invalid")
    raw = _read_regular(
        OWNER_RELEASE_BASE
        / release_revision
        / "trust/direct-iam-identity-authority.json",
        modes=frozenset({0o444}),
    )
    try:
        unbound = direct_iam.decode_canonical(raw)
        foundation_revision = str(unbound.get("release_revision", ""))
        authority = direct_iam.decode_canonical(
            raw,
            release_revision=foundation_revision,
        )
    except direct_iam.DirectIamIdentityAuthorityError:
        _error("owner_gate_attached_sa_direct_identity_invalid")
    if (
        _REVISION.fullmatch(foundation_revision) is None
        or foundation_revision == release_revision
        or foundation_revision
        != package.get("foundation_source_revision")
        or _REVISION.fullmatch(
            str(package.get("foundation_source_tree_oid", ""))
        )
        is None
        or hashlib.sha256(raw).hexdigest()
        != package.get("direct_iam_identity_authority_sha256")
        or authority.get("pre_foundation_authority_sha256")
        != package.get("pre_foundation_authority_sha256")
        or authority.get("foundation_apply_receipt_sha256")
        != package.get("foundation_apply_receipt_sha256")
        or authority.get("resource_ancestor_chain")
        != package.get("resource_ancestor_chain")
    ):
        _error("owner_gate_attached_sa_direct_identity_invalid")
    return authority


def _validate_request(
    value: Any,
    *,
    schema: str,
    release_revision: str,
    now_unix: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    fields = {
        "schema", "phase", "collected_at_unix", "plan_sha256",
        "production_ingress_observation_sha256",
        "cloud_install_receipt", *_SIGNER_LINEAGE_FIELDS,
        "observation_binding_sha256", "request_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_host_observation_request_invalid")
    request = dict(value)
    unsigned = {name: item for name, item in request.items() if name != "request_sha256"}
    binding = {
        name: item
        for name, item in request.items()
        if name not in {
            "schema", "observation_binding_sha256", "request_sha256"
        }
    }
    if (
        request.get("schema") != schema
        or request.get("phase") not in {"inert", "post_iam"}
        or type(request.get("collected_at_unix")) is not int
        or not now_unix - FRESHNESS_SECONDS
        <= request["collected_at_unix"]
        <= now_unix + 5
        or _SHA256.fullmatch(str(request.get("plan_sha256", ""))) is None
        or type(request.get("production_ingress_observation_sha256")) is not str
        or _SHA256.fullmatch(
            str(request.get("production_ingress_observation_sha256", ""))
        )
        is None
        or any(
            _SHA256.fullmatch(str(request.get(name, ""))) is None
            for name in _SIGNER_LINEAGE_FIELDS
        )
        or request.get("observation_binding_sha256") != _sha(binding)
        or request.get("request_sha256") != _sha(unsigned)
    ):
        _error("owner_gate_host_observation_request_invalid")
    package = _load_release_package(release_revision)
    install = _validate_install_receipt(
        request.get("cloud_install_receipt"), package=package
    )
    return request, package, install


def _attest(
    unsigned: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
    public_key_id: str,
) -> Mapping[str, Any]:
    report = {**unsigned, "report_sha256": _sha(unsigned)}
    signature = private_key.sign(_canonical(report))
    return {
        **report,
        "attestation": {
            "schema": OBSERVATION_ATTESTATION_SCHEMA,
            "public_key_id": public_key_id,
            "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii"),
        },
    }


def _completion_facts(
    request: Mapping[str, Any],
    *,
    observed_at_entry: int,
    started_monotonic: float,
    injected_now: int | None,
) -> tuple[int, int]:
    elapsed = time.monotonic() - started_monotonic
    if (
        not math.isfinite(elapsed)
        or elapsed < 0
        or elapsed > FRESHNESS_SECONDS
    ):
        _error("owner_gate_host_observation_clock_invalid")
    completed = (
        int(time.time())
        if injected_now is None
        else observed_at_entry + math.ceil(elapsed)
    )
    fresh_through = int(request["collected_at_unix"]) + FRESHNESS_SECONDS
    if completed < request["collected_at_unix"] or completed > fresh_through:
        _error("owner_gate_host_observation_collection_stale")
    return completed, fresh_through


def _load_host_signer(
    release_revision: str,
) -> tuple[Ed25519PrivateKey, Ed25519PublicKey, str, Mapping[str, Any]]:
    try:
        readiness = provisioning.verify_host_signer_runtime_readiness(
            release_revision
        )
        config = trusted._load_config(
            trusted.HOST_CONFIG_PATH,
            role="host",
            expected_uid=0,
            expected_path=trusted.HOST_CONFIG_PATH,
            expected_private_key_path=trusted.HOST_PRIVATE_KEY_PATH,
            expected_replay_directory=trusted.HOST_REPLAY_DIRECTORY,
        )
        private_key, public_key, key_id = trusted._load_private_key(config)
    except (provisioning.TrustedSignerProvisioningError, trusted.TrustedObservationError):
        _error("owner_gate_host_signer_not_ready")
    if (
        readiness.get("release_revision") != release_revision
        or readiness.get("public_key_id") != key_id
        or readiness.get("activation_seal_absent") is not True
        or readiness.get("current_link_absent") is not True
        or readiness.get("services_inactive_disabled") is not True
        or readiness.get("activation_performed") is not False
        or readiness.get("iam_mutation_performed") is not False
    ):
        _error("owner_gate_host_signer_not_ready")
    return private_key, public_key, key_id, readiness


def _fixed_command(argv: Sequence[str], *, allow_returncodes: frozenset[int] = frozenset({0})) -> bytes:
    try:
        completed = subprocess.run(
            tuple(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=COMMAND_TIMEOUT_SECONDS,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
    except (OSError, subprocess.SubprocessError):
        _error("owner_gate_host_command_failed")
    if (
        completed.returncode not in allow_returncodes
        or completed.stderr
        or len(completed.stdout) > MAX_COMMAND_BYTES
    ):
        _error("owner_gate_host_command_failed")
    return completed.stdout


def _unit_properties(release: Path) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for role, unit in _UNIT_NAMES.items():
        source = _read_regular(
            release / _UNIT_ASSETS[role], modes=frozenset({0o444})
        )
        installed = _read_regular(
            Path("/etc/systemd/system") / unit, modes=frozenset({0o444})
        )
        if source != installed:
            _error("owner_gate_host_unit_invalid")
        try:
            lines = source.decode("ascii", errors="strict").splitlines()
        except UnicodeError:
            _error("owner_gate_host_unit_invalid")
        properties: dict[str, str] = {}
        for name in (*preflight.REQUIRED_HARDENING, "User", "Group", "ExecStart", "PrivateNetwork"):
            values = [line.split("=", 1)[1] for line in lines if line.startswith(f"{name}=")]
            if name == "PrivateNetwork" and not values:
                values = ["no"]
            if len(values) != 1:
                _error("owner_gate_host_unit_invalid")
            properties[name] = values[0]
        live = _fixed_command((
            "/usr/bin/systemctl", "show", "--no-pager",
            "--property=ActiveState", "--property=UnitFileState", unit,
        ))
        try:
            live_lines = live.decode("ascii", errors="strict").splitlines()
            live_values = dict(line.split("=", 1) for line in live_lines)
        except (UnicodeError, ValueError):
            _error("owner_gate_host_unit_invalid")
        if live_values != {"ActiveState": "inactive", "UnitFileState": "disabled"}:
            _error("owner_gate_host_unit_invalid")
        properties.update(live_values)
        expected = preflight.EXPECTED_UNIT_PROPERTIES[role]
        if properties != expected:
            _error("owner_gate_host_unit_invalid")
        result[role] = properties
    return result


def _identity_facts() -> Mapping[str, Any]:
    identities = {
        "web": ("muncho-passkey-web", preflight.WEB_UID),
        "authority": ("muncho-passkey-authority", preflight.AUTHORITY_UID),
        "executor": ("muncho-storage-executor", preflight.EXECUTOR_UID),
    }
    result: dict[str, Any] = {}
    observed_ids: set[int] = set()
    for role, (name, numeric_id) in identities.items():
        try:
            user = pwd.getpwnam(name)
            group = grp.getgrnam(name)
            supplementary_groups = os.getgrouplist(name, numeric_id)
        except (KeyError, OSError):
            _error("owner_gate_host_identity_invalid")
        if (
            user.pw_uid != numeric_id
            or user.pw_gid != numeric_id
            or group.gr_gid != numeric_id
            or user.pw_shell != "/usr/sbin/nologin"
            or supplementary_groups != [numeric_id]
            or numeric_id in observed_ids
        ):
            _error("owner_gate_host_identity_invalid")
        observed_ids.add(numeric_id)
        result[role] = {
            "name": name, "uid": numeric_id, "gid": numeric_id,
            "shell": "/usr/sbin/nologin",
        }
    return result


def _socket_facts(release: Path) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for role, (unit, path, uid, gid) in _SOCKET_ASSETS.items():
        source = _read_regular(
            release / f"ops/muncho/owner-gate/{unit}", modes=frozenset({0o444})
        )
        installed = _read_regular(
            Path("/etc/systemd/system") / unit, modes=frozenset({0o444})
        )
        if source != installed or os.path.lexists(path):
            _error("owner_gate_host_socket_invalid")
        text = source.decode("ascii", errors="strict")
        expected_lines = {
            f"ListenStream={path}", "SocketMode=0660",
            f"SocketUser={pwd.getpwuid(uid).pw_name}",
            f"SocketGroup={grp.getgrgid(gid).gr_name}",
        }
        if not expected_lines <= set(text.splitlines()):
            _error("owner_gate_host_socket_invalid")
        result[role] = {"path": str(path), "uid": uid, "gid": gid, "mode": "0660"}
    return result


def _firewall_facts(*, collected_at_unix: int) -> Mapping[str, Any]:
    raw = _read_regular(
        FIREWALL_RECEIPT,
        maximum=MAX_FILE_BYTES,
        uid=0,
        gid=preflight.EXECUTOR_UID,
        modes=frozenset({0o440}),
    )
    stored = _decode_canonical(
        raw, maximum=MAX_FILE_BYTES, code="owner_gate_host_firewall_receipt_invalid"
    )
    unsigned = {name: item for name, item in stored.items() if name != "receipt_sha256"}
    if (
        set(stored) != {
            "schema", "backend", "boot_id", "rules_source_sha256",
            "live_projection_sha256", "executor_uid", "root_admin_metadata_allowed",
            "other_unprivileged_uids_blocked", "web_uid_blocked",
            "authority_uid_blocked", "observed_at_unix", "ready", "receipt_sha256",
        }
        or stored.get("schema") != firewall.READINESS_SCHEMA
        or stored.get("receipt_sha256") != _sha(unsigned)
        or stored.get("ready") is not True
        or not collected_at_unix - FRESHNESS_SECONDS
        <= stored.get("observed_at_unix", 0)
        <= collected_at_unix
    ):
        _error("owner_gate_host_firewall_receipt_invalid")
    live = firewall.collect_receipt(
        rules_path=Path("/etc/muncho-owner-gate/metadata-firewall.rules"),
        now_unix=stored["observed_at_unix"],
    )
    if live != stored:
        _error("owner_gate_host_firewall_receipt_invalid")
    return {
        "web_blocked": True,
        "authority_blocked": True,
        "executor_metadata_allowed": True,
        "executor_private_google_api_allowed": True,
        "other_unprivileged_uids_blocked": True,
        "root_admin_metadata_allowed": True,
        "nft_ruleset_verified": True,
        "root_readiness_receipt_verified": True,
    }


def _database_file(path: Path, *, uid: int) -> tuple[os.stat_result, str]:
    if any(os.path.lexists(Path(f"{path}{suffix}")) for suffix in ("-journal", "-wal", "-shm")):
        _error("owner_gate_host_database_sidecar_present")
    state = path.lstat()
    if (
        not stat.S_ISREG(state.st_mode)
        or stat.S_ISLNK(state.st_mode)
        or state.st_uid != uid
        or state.st_gid != uid
        or state.st_nlink != 1
        or stat.S_IMODE(state.st_mode) != 0o600
    ):
        _error("owner_gate_host_database_invalid")
    return state, hashlib.sha256(_read_regular(
        path, maximum=MAX_FILE_BYTES, uid=uid, gid=uid, modes=frozenset({0o600})
    )).hexdigest()


def _sqlite_facts() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    authority_before, authority_sha = _database_file(
        AUTHORITY_DB, uid=preflight.AUTHORITY_UID
    )
    executor_before, executor_sha = _database_file(
        EXECUTOR_DB, uid=preflight.EXECUTOR_UID
    )
    key, _raw = _load_authority_receipt_key()
    key_id = hashlib.sha256(key.public_bytes_raw()).hexdigest()
    authority = sqlite_backend.PasskeyV2AuthorityDatabase(
        AUTHORITY_DB,
        authority_uid=preflight.AUTHORITY_UID,
        authority_gid=preflight.AUTHORITY_UID,
    )
    executor = sqlite_backend.PasskeyV2ExecutorDatabase(
        EXECUTOR_DB,
        executor_uid=preflight.EXECUTOR_UID,
        executor_gid=preflight.EXECUTOR_UID,
        pinned_authority_receipt_public_key=key,
        pinned_authority_receipt_key_id=key_id,
    )
    authority_preflight = authority.preflight()
    executor_preflight = executor.preflight()
    credentials = authority.read_active_credentials()
    if len(credentials) != 1:
        _error("owner_gate_host_migration_invalid")
    credential = credentials[0]
    uri = f"file:{AUTHORITY_DB}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        counts = {
            name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in ("requests", "challenges", "grants", "credential_uses")
        }
    except sqlite3.Error:
        _error("owner_gate_host_database_invalid")
    finally:
        connection.close()
    authority_after, authority_after_sha = _database_file(
        AUTHORITY_DB, uid=preflight.AUTHORITY_UID
    )
    executor_after, executor_after_sha = _database_file(
        EXECUTOR_DB, uid=preflight.EXECUTOR_UID
    )
    if (
        _identity(authority_before) != _identity(authority_after)
        or _identity(executor_before) != _identity(executor_after)
        or authority_sha != authority_after_sha
        or executor_sha != executor_after_sha
        or counts != {"requests": 0, "challenges": 0, "grants": 0, "credential_uses": 0}
        or authority_preflight.get("runtime_eligible") is not True
        or executor_preflight.get("runtime_eligible") is not True
    ):
        _error("owner_gate_host_database_changed")
    sqlite_facts = {
        "runtime_module": "scripts.canary.passkey_v2_sqlite",
        "authority_schema": "muncho-passkey-v2-authority-sqlite.v1",
        "executor_schema": "muncho-passkey-v2-executor-sqlite.v1",
        "authority_db": str(AUTHORITY_DB),
        "executor_db": str(EXECUTOR_DB),
        "directory_mode": "0700",
        "database_mode": "0600",
        "journal_mode": "DELETE",
        "synchronous": "FULL",
        "begin_immediate": True,
        "append_only_triggers": True,
        "runtime_eligible": True,
    }
    migration = {
        "credential_count": 1,
        "enabled_owner_count": 1,
        "owner_discord_user_id": credential["owner_discord_user_id"],
        "credential_id_sha256": credential["credential_id_sha256"],
        "public_key_sha256": credential["source_public_key_sha256"],
        "public_key_byte_count": credential["public_key_byte_count"],
        "sign_count": credential["initial_sign_count"],
        "backed_up": credential["initial_credential_backed_up"],
        "active_request_count": counts["requests"],
        "active_challenge_count": counts["challenges"],
        "active_grant_count": counts["grants"],
        "totp_seed_migrated": False,
        "source_receipt_verified": True,
        "target_receipt_verified": True,
        "public_key_only": True,
        "user_handle_sha256": credential["expected_user_handle_sha256"],
        "credential_id_b64url_source_receipt_bound": True,
    }
    return sqlite_facts, migration


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _selftest_envelope(*, request_id: str) -> Mapping[str, Any]:
    return protocol.build_action_envelope(
        request_id=request_id,
        requester_discord_user_id="1279454038731264061",
        required_approver_discord_user_id="1279454038731264061",
        scope="runtime_config_mutation",
        case_id="case:owner-gate-runtime-selftest",
        target_system="gce:muncho-canary-v2-01/disk",
        action_summary="Bounded immutable-runtime authorization self-test.",
        risk="No production state is reachable from the isolated temporary database.",
        rollback="The temporary database is destroyed after the self-test.",
        action_payload={
            "operation": "selftest_only",
            "project": foundation.PROJECT,
            "zone": foundation.ZONE,
            "instance": foundation.TARGET_INSTANCE,
            "disk_id": foundation.TARGET_DISK_ID,
        },
        executor_release_sha="a" * 40,
        executor_plan_sha256="b" * 64,
        transaction_id="c" * 64,
        stage="selftest",
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
        issued_at_unix=1_785_000_000,
        approval_ttl_seconds=300,
    )


def _selftest_challenge(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    return protocol.build_challenge_record(
        envelope=envelope,
        challenge_id="C" * 32,
        challenge_b64url=_b64url(b"owner-gate-selftest-challenge-32"),
        rp_id=protocol.PRODUCTION_RP_ID,
        origin=protocol.PRODUCTION_ORIGIN,
        created_at_unix=1_785_000_001,
    )


def _selftest_credential_assertion(
    envelope: Mapping[str, Any],
    challenge: Mapping[str, Any],
    *,
    flags: int = 0x1D,
    client_origin: str = protocol.PRODUCTION_ORIGIN,
    client_challenge: str | None = None,
    rp_id: str = protocol.PRODUCTION_RP_ID,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    # Canonical CBOR encoding of {1:2, 3:-7, -1:1, -2:x, -3:y}.
    public_key_cose = (
        b"\xa5\x01\x02\x03\x26\x20\x01\x21\x58\x20"
        + numbers.x.to_bytes(32, "big")
        + b"\x22\x58\x20"
        + numbers.y.to_bytes(32, "big")
    )
    owner = "1279454038731264061"
    credential_id = b"owner-gate-selftest-credential"
    credential = webauthn.build_migrated_credential(
        owner_discord_user_id=owner,
        credential_id=credential_id,
        public_key_cose=public_key_cose,
        rp_id=protocol.PRODUCTION_RP_ID,
        origin=protocol.PRODUCTION_ORIGIN,
        imported_at_unix=1_784_999_900,
        migration_receipt_sha256="6" * 64,
        initial_sign_count=0,
        initial_credential_backed_up=True,
        expected_user_handle=owner.encode("ascii"),
    )
    client_data = json.dumps(
        {
            "type": "webauthn.get",
            "challenge": client_challenge or challenge["challenge_b64url"],
            "origin": client_origin,
            "crossOrigin": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    authenticator_data = (
        hashlib.sha256(rp_id.encode("ascii")).digest()
        + bytes([flags])
        + (0).to_bytes(4, "big")
    )
    signed = authenticator_data + hashlib.sha256(client_data).digest()
    signature = private_key.sign(signed, ec.ECDSA(hashes.SHA256()))
    assertion = {
        "schema": webauthn.ASSERTION_SCHEMA,
        "credential": {
            "id": _b64url(credential_id),
            "rawId": _b64url(credential_id),
            "response": {
                "clientDataJSON": _b64url(client_data),
                "authenticatorData": _b64url(authenticator_data),
                "signature": _b64url(signature),
                "userHandle": None,
            },
            "type": "public-key",
            "authenticatorAttachment": "platform",
            "clientExtensionResults": {},
        },
    }
    return credential, assertion


def _selftest_denied(
    envelope: Mapping[str, Any],
    challenge: Mapping[str, Any],
    **changes: Any,
) -> bool:
    credential, assertion = _selftest_credential_assertion(
        envelope, challenge, **changes
    )
    try:
        webauthn.verify_assertion(
            assertion,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )
    except webauthn.PasskeyV2WebAuthnError:
        return True
    return False


def _runtime_security_selftest() -> Mapping[str, Any]:
    """Exercise immutable WebAuthn, atomic consume, and receipt binding code."""

    envelope = _selftest_envelope(request_id="R" * 32)
    challenge = _selftest_challenge(envelope)
    credential, assertion = _selftest_credential_assertion(envelope, challenge)
    verified = webauthn.verify_assertion(
        assertion,
        credential=credential,
        challenge=challenge,
        envelope=envelope,
        prior_sign_count=0,
    )
    if verified.get("credential_sign_count") != 0:
        _error("owner_gate_host_webauthn_selftest_failed")
    forged = json.loads(json.dumps(assertion))
    encoded = forged["credential"]["response"]["signature"]
    forged_raw = bytearray(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    forged_raw[-1] ^= 1
    forged["credential"]["response"]["signature"] = _b64url(bytes(forged_raw))
    forged_blocked = False
    try:
        webauthn.verify_assertion(
            forged,
            credential=credential,
            challenge=challenge,
            envelope=envelope,
            prior_sign_count=0,
        )
    except webauthn.PasskeyV2WebAuthnError:
        forged_blocked = True
    negative = {
        "forged_assertion_blocked": forged_blocked,
        "wrong_challenge_blocked": _selftest_denied(
            envelope,
            challenge,
            client_challenge=_b64url(b"wrong-owner-gate-challenge-value"),
        ),
        "wrong_origin_blocked": _selftest_denied(
            envelope, challenge, client_origin="https://invalid.example"
        ),
        "wrong_rp_blocked": _selftest_denied(
            envelope, challenge, rp_id="invalid.example"
        ),
        "no_uv_blocked": _selftest_denied(envelope, challenge, flags=0x01),
    }
    if any(value is not True for value in negative.values()):
        _error("owner_gate_host_webauthn_selftest_failed")

    try:
        temporary = tempfile.TemporaryDirectory(
            prefix="muncho-owner-gate-selftest-", dir=SELFTEST_BASE
        )
    except OSError:
        _error("owner_gate_host_selftest_directory_unavailable")
    with temporary as directory:
        root = Path(directory) / "authority"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        state = root.stat()
        database_path = root / "passkey-v2.sqlite3"
        sqlite_backend.bootstrap_authority_database(
            database_path,
            authority_uid=state.st_uid,
            authority_gid=state.st_gid,
            now_unix=1_784_999_800,
            require_root=False,
        )
        authority = sqlite_backend.PasskeyV2AuthorityDatabase(
            database_path,
            authority_uid=state.st_uid,
            authority_gid=state.st_gid,
        )
        authority.import_migrated_credential(credential)
        authority.create_request(envelope)
        authority.create_challenge(challenge, envelope=envelope)
        grant = authority.verify_assertion_and_record_grant(
            assertion=assertion,
            envelope=envelope,
            challenge=challenge,
            grant_id="G" * 32,
            now_unix=1_785_000_002,
        )
        receipt_signer = signer_runtime.ReceiptSigner(
            Ed25519PrivateKey.generate()
        )
        runtime_binding = protocol.build_runtime_binding(
            executor_release_sha="a" * 40,
            executor_plan_sha256="b" * 64,
            executor_binary_sha256="7" * 64,
            mutation_wrapper_sha256="8" * 64,
            remote_transport_sha256="9" * 64,
        )
        barrier = threading.Barrier(8)

        def consume(index: int) -> tuple[str, str, Mapping[str, Any] | None]:
            attempt = f"{index + 10:064x}"
            barrier.wait()
            try:
                result = authority.consume_or_replay(
                    envelope=envelope,
                    runtime_binding=runtime_binding,
                    consume_attempt_id=attempt,
                    signer=receipt_signer,
                    now_unix=1_785_000_003,
                )
                return result.disposition, attempt, result.receipt
            except sqlite_backend.PasskeyV2SqliteDenied:
                return "denied", attempt, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(consume, range(8)))
        winners = [item for item in outcomes if item[0] == "authorized_once"]
        if len(winners) != 1 or winners[0][2] is None:
            _error("owner_gate_host_concurrency_selftest_failed")
        _disposition, winner_attempt, winner_receipt = winners[0]
        replay = authority.consume_or_replay(
            envelope=envelope,
            runtime_binding=runtime_binding,
            consume_attempt_id=winner_attempt,
            signer=receipt_signer,
            now_unix=1_785_000_004,
        )
        if replay.disposition != "receipt_replay":
            _error("owner_gate_host_replay_selftest_failed")
        try:
            authority.consume_or_replay(
                envelope=envelope,
                runtime_binding=runtime_binding,
                consume_attempt_id="f" * 64,
                signer=receipt_signer,
                now_unix=1_785_000_004,
            )
        except sqlite_backend.PasskeyV2SqliteDenied:
            replay_blocked = True
        else:
            replay_blocked = False
        protocol.validate_authorization_receipt(
            winner_receipt,
            envelope=envelope,
            grant=grant,
            challenge=challenge,
            receipt_public_key=receipt_signer.public_key,
        )
        wrong_envelope = _selftest_envelope(request_id="S" * 32)
        try:
            protocol.validate_authorization_receipt(
                winner_receipt,
                envelope=wrong_envelope,
                grant=grant,
                challenge=challenge,
                receipt_public_key=receipt_signer.public_key,
            )
        except protocol.PasskeyV2ProtocolError:
            receipt_action_binding = True
        else:
            receipt_action_binding = False
        web_privilege_blocked = False
        try:
            service.handle_authority_frame(
                service.build_service_frame(
                    "create_request", {"action_envelope": wrong_envelope}
                ),
                authority=authority,
                signer=receipt_signer,
                peer_uid=service.WEB_UID,
                now_unix=1_785_000_004,
            )
        except service.PasskeyV2ServiceError:
            web_privilege_blocked = True
        raw_grant_api_absent = False
        try:
            service.build_service_frame("grant", {"grant": grant})
        except service.PasskeyV2ServiceError:
            raw_grant_api_absent = True
        if not all((replay_blocked, receipt_action_binding, web_privilege_blocked, raw_grant_api_absent)):
            _error("owner_gate_host_authorization_selftest_failed")

    return {
        "webauthn": {
            "rp_id": protocol.PRODUCTION_RP_ID,
            "origin": protocol.PRODUCTION_ORIGIN,
            "user_verification_required": True,
            **negative,
            "replay_blocked": True,
            "concurrent_exactly_one_authorized": True,
            "web_raw_grant_api_absent": True,
        },
        "public_web_can_author_envelope": False,
        "authorization_receipt_signature_self_verified": True,
        "receipt_action_binding_self_verified": True,
    }


def _config_mapping(path: Path) -> Mapping[str, Any]:
    return _decode_canonical(
        _read_regular(path, gid=0, modes=frozenset({0o444})),
        maximum=MAX_FILE_BYTES,
        code="owner_gate_host_config_invalid",
    )


def _executor_receipt_key_facts() -> Mapping[str, str]:
    """Bind executor key claims to the fixed root-owned PEM on this host."""

    config = _config_mapping(EXECUTOR_CONFIG)
    expected = {
        "api_host": "compute.googleapis.com",
        "api_private_vip_range": "199.36.153.8/30",
        "expected_disk_id": foundation.TARGET_DISK_ID,
        "expected_instance_id": foundation.TARGET_INSTANCE_ID,
        "executor_database": str(EXECUTOR_DB),
        "journal_root": str(EXECUTOR_DB.parent),
        "mutation_enable_seal": str(service.ACTIVATION_SEAL),
        "mutation_enable_seal_uid": 0,
        "mutation_enable_seal_gid": preflight.EXECUTOR_UID,
        "mutation_enable_seal_mode": "0440",
        "metadata_host": METADATA_HOST,
        "firewall_readiness_receipt": str(FIREWALL_RECEIPT),
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
            service.CLOUD_OBSERVATION_PUBLIC_KEY
        ),
        "host_observation_public_key": str(
            service.HOST_OBSERVATION_PUBLIC_KEY
        ),
        "receipt_public_key": str(AUTHORITY_RECEIPT_PUBLIC_KEY),
        "receipt_public_key_owner": "root:root",
        "receipt_public_key_mode": "0444",
        "signed_authorization_receipt_required": True,
        "topology_iam_readiness_seal_required_for_mutation_only": True,
    }
    try:
        direct_iam_pins = service._direct_iam_pins(config)
    except service.PasskeyV2ServiceError:
        _error("owner_gate_host_executor_invalid")
    if (
        set(config) != service._EXECUTOR_CONFIG_FIELDS
        or config.get("schema") != "muncho-owner-gate-executor-config.v1"
        or any(config.get(name) != value for name, value in expected.items())
        or not isinstance(direct_iam_pins, Mapping)
    ):
        _error("owner_gate_host_executor_invalid")
    public_key, raw = _load_authority_receipt_key()
    digest = hashlib.sha256(raw).hexdigest()
    if config.get("receipt_public_key_sha256") != digest:
        _error("owner_gate_host_executor_invalid")
    return {
        "receipt_public_key_sha256": digest,
        "receipt_public_key_owner": "root:root",
        "receipt_public_key_mode": "0444",
        "receipt_public_key_id": hashlib.sha256(
            public_key.public_bytes_raw()
        ).hexdigest(),
    }


def _host_release_facts(
    package: Mapping[str, Any],
    install: Mapping[str, Any],
    request: Mapping[str, Any],
    attached_sa_probe: Mapping[str, Any],
    expected_uid: int = 0,
    expected_gid: int = 0,
) -> Mapping[str, Any]:
    revision = str(package["release_revision"])
    root = OWNER_RELEASE_BASE / revision
    root_state = root.lstat()
    if (
        not stat.S_ISDIR(root_state.st_mode)
        or root_state.st_uid != expected_uid
        or root_state.st_gid != expected_gid
        or stat.S_IMODE(root_state.st_mode) != 0o555
    ):
        _error("owner_gate_host_release_invalid")
    python = root / "venv/bin/python"
    python_raw = _read_regular(
        python, maximum=128 * 1024 * 1024, uid=expected_uid,
        modes=frozenset({0o555})
    )
    entrypoints = preflight.HOST_RELEASE_ENTRYPOINTS
    for name in entrypoints:
        _read_regular(
            root / "bin" / name,
            uid=expected_uid,
            modes=frozenset({0o555}),
        )
    return {
        "revision": revision,
        "source_tree_oid": package["source_tree_oid"],
        "root": str(root),
        "uid": expected_uid,
        "gid": expected_gid,
        "mode": "0555",
        "immutable": True,
        "package_sha256": package["package_sha256"],
        "package_inventory_sha256": package["package_inventory_sha256"],
        **{name: package[name] for name in _LINEAGE_FIELDS},
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "install_receipt_sha256": install["receipt_sha256"],
        "install_receipt_file_sha256": hashlib.sha256(_canonical(install)).hexdigest(),
        **{name: request[name] for name in _SIGNER_LINEAGE_FIELDS},
        "attached_sa_permission_probe_report_sha256": attached_sa_probe[
            "report_sha256"
        ],
        "offline_wheelhouse_verified": True,
        "network_install_performed": False,
        "entrypoints": list(entrypoints),
        "observation_dispatcher_schemas": list(
            preflight.HOST_OBSERVATION_DISPATCHER_SCHEMAS
        ),
        "python_version": "3.11.2",
        "python_executable": str(python),
        "python_executable_sha256": hashlib.sha256(python_raw).hexdigest(),
        "python_hash_revalidated_by_sha256sum": True,
    }


def _validate_attached_probe(
    probe: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    package: Mapping[str, Any],
    install: Mapping[str, Any],
    direct_identity: Mapping[str, Any],
    host_public_key: Ed25519PublicKey,
    host_key_id: str,
) -> Mapping[str, Any]:
    fields = {
        "schema", "phase", "collected_at_unix", "completed_at_unix",
        "fresh_through_unix", "release_revision",
        "plan_sha256", "observation_binding_sha256",
        "attached_request_sha256",
        "source_tree_oid", "package_sha256", "package_inventory_sha256",
        *_LINEAGE_FIELDS, "resource_ancestor_chain",
        "install_receipt_sha256", *_SIGNER_LINEAGE_FIELDS,
        "runtime_instance_numeric_id",
        "runtime_service_account_email", "runtime_service_account_unique_id",
        "metadata_scopes", "effective_permission_probe",
        "target_instance_numeric_id", "target_disk_numeric_id",
        "numeric_targets_reverified", "inherited_bindings_evaluated",
        "conditional_bindings_evaluated", "metadata_token_acquired",
        "metadata_token_wiped", "owner_credential_values_read",
        "report_sha256", "attestation",
    }
    if not isinstance(probe, Mapping) or set(probe) != fields:
        _error("owner_gate_attached_sa_probe_invalid")
    unsigned = {
        name: item for name, item in probe.items()
        if name not in {"report_sha256", "attestation"}
    }
    signed = {**unsigned, "report_sha256": probe.get("report_sha256")}
    attestation = probe.get("attestation")
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != {"schema", "public_key_id", "signature_ed25519_b64url"}
        or attestation.get("schema") != OBSERVATION_ATTESTATION_SCHEMA
        or attestation.get("public_key_id") != host_key_id
        or probe.get("report_sha256") != _sha(unsigned)
    ):
        _error("owner_gate_attached_sa_probe_invalid")
    try:
        host_public_key.verify(
            _decode_signature(attestation["signature_ed25519_b64url"]),
            _canonical(signed),
        )
    except InvalidSignature:
        _error("owner_gate_attached_sa_probe_invalid")
    expected_permissions = preflight.expected_effective_permission_probe(
        request["phase"] == "post_iam"
    )
    if (
        probe.get("schema") != ATTACHED_SA_PROBE_SCHEMA
        or probe.get("phase") != request["phase"]
        or probe.get("collected_at_unix") != request["collected_at_unix"]
        or type(probe.get("completed_at_unix")) is not int
        or probe["completed_at_unix"] < probe["collected_at_unix"]
        or probe["completed_at_unix"] > probe.get("fresh_through_unix", -1)
        or probe.get("fresh_through_unix")
        != request["collected_at_unix"] + FRESHNESS_SECONDS
        or probe.get("plan_sha256") != request["plan_sha256"]
        or probe.get("observation_binding_sha256")
        != request["observation_binding_sha256"]
        or probe.get("attached_request_sha256")
        != _sha({
            **{
                name: item
                for name, item in request.items()
                if name not in {"schema", "request_sha256"}
            },
            "schema": ATTACHED_SA_REQUEST_SCHEMA,
        })
        or probe.get("release_revision") != package["release_revision"]
        or probe.get("source_tree_oid") != package["source_tree_oid"]
        or probe.get("package_sha256") != package["package_sha256"]
        or probe.get("package_inventory_sha256") != package["package_inventory_sha256"]
        or any(probe.get(name) != package[name] for name in _LINEAGE_FIELDS)
        or probe.get("resource_ancestor_chain") != package["resource_ancestor_chain"]
        or probe.get("install_receipt_sha256") != install["receipt_sha256"]
        or any(probe.get(name) != request[name] for name in _SIGNER_LINEAGE_FIELDS)
        or probe.get("runtime_instance_numeric_id")
        != direct_identity["owner_gate_vm_numeric_id"]
        or probe.get("runtime_service_account_email")
        != direct_identity["owner_gate_service_account_email"]
        or probe.get("runtime_service_account_unique_id")
        != direct_identity["owner_gate_service_account_unique_id"]
        or probe.get("metadata_scopes")
        != direct_identity["metadata_oauth_scopes"]
        or probe.get("effective_permission_probe") != expected_permissions
        or probe.get("target_instance_numeric_id") != foundation.TARGET_INSTANCE_ID
        or probe.get("target_disk_numeric_id") != foundation.TARGET_DISK_ID
        or probe.get("numeric_targets_reverified")
        is not (request["phase"] == "post_iam")
        or any(probe.get(name) is not True for name in (
            "inherited_bindings_evaluated", "conditional_bindings_evaluated",
            "metadata_token_acquired", "metadata_token_wiped",
        ))
        or probe.get("owner_credential_values_read") is not False
    ):
        _error("owner_gate_attached_sa_probe_invalid")
    return probe


def build_host_observation(
    request_value: Mapping[str, Any],
    *,
    release_revision: str,
    attached_sa_probe: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Recollect and sign the exact HOST_OBSERVATION_SCHEMA document."""

    started_monotonic = time.monotonic()
    observed = int(time.time()) if now_unix is None else now_unix
    request, package, install = _validate_request(
        request_value,
        schema=HOST_REQUEST_SCHEMA,
        release_revision=release_revision,
        now_unix=observed,
    )
    _require_executor_activation_seal_absent()
    private_key, public_key, key_id, host_readiness = _load_host_signer(
        release_revision
    )
    try:
        cloud_readiness = provisioning.verify_cloud_signer_inert_readiness(
            release_revision
        )
    except provisioning.TrustedSignerProvisioningError:
        _error("owner_gate_cloud_signer_not_ready")
    if (
        request["host_signer_provisioning_receipt_sha256"]
        != host_readiness.get("provisioning_receipt_sha256")
        or request["host_signer_readiness_sha256"]
        != host_readiness.get("readiness_sha256")
        or request["cloud_signer_provisioning_receipt_sha256"]
        != cloud_readiness.get("provisioning_receipt_sha256")
        or request["cloud_signer_readiness_sha256"]
        != cloud_readiness.get("readiness_sha256")
    ):
        _error("owner_gate_signer_lineage_invalid")
    direct_identity = _load_direct_iam_identity(package)
    checked_probe = _validate_attached_probe(
        attached_sa_probe,
        request=request,
        package=package,
        install=install,
        direct_identity=direct_identity,
        host_public_key=public_key,
        host_key_id=key_id,
    )
    release = OWNER_RELEASE_BASE / release_revision
    sqlite_facts, migration = _sqlite_facts()
    security_selftest = _runtime_security_selftest()
    executor_key = _executor_receipt_key_facts()
    if any(os.path.lexists(path) for path in (
        Path("/usr/bin/gcloud"), Path("/usr/local/bin/gcloud"), Path("/snap/bin/gcloud")
    )):
        _error("owner_gate_host_local_gcloud_present")
    release_facts = _host_release_facts(
        package, install, request, checked_probe
    )
    identity_facts = _identity_facts()
    socket_facts = _socket_facts(release)
    unit_facts = _unit_properties(release)
    firewall_facts = _firewall_facts(
        collected_at_unix=request["collected_at_unix"]
    )
    _require_executor_activation_seal_absent()
    completed_at_unix, fresh_through_unix = _completion_facts(
        request,
        observed_at_entry=observed,
        started_monotonic=started_monotonic,
        injected_now=now_unix,
    )
    unsigned = {
        "schema": preflight.HOST_OBSERVATION_SCHEMA,
        "phase": request["phase"],
        "collected_at_unix": request["collected_at_unix"],
        "completed_at_unix": completed_at_unix,
        "fresh_through_unix": fresh_through_unix,
        "plan_sha256": request["plan_sha256"],
        "production_ingress_observation_sha256": request[
            "production_ingress_observation_sha256"
        ],
        "observation_binding_sha256": request[
            "observation_binding_sha256"
        ],
        "release": release_facts,
        "identities": identity_facts,
        "sockets": socket_facts,
        "units": unit_facts,
        "filesystem_boundaries": {
            "web_reads_authority_db": False,
            "web_writes_authority_db": False,
            "web_reads_mutation_journal": False,
            "authority_reads_mutation_journal": False,
            "executor_reads_authority_db": False,
            "authority_database_owner_uid": preflight.AUTHORITY_UID,
            "mutation_journal_owner_uid": preflight.EXECUTOR_UID,
        },
        "metadata_firewall": firewall_facts,
        "sqlite": sqlite_facts,
        "migration": migration,
        "webauthn": security_selftest["webauthn"],
        "request_intake": {
            "public_web_can_author_envelope": security_selftest[
                "public_web_can_author_envelope"
            ],
            "iap_fixed_command_only": True,
            "signed_release_verified": True,
            "signed_source_preflight_verified": True,
            "signed_host_identity_verified": True,
            "signed_external_iam_verified": True,
            "release_plan_transaction_evidence_bound": True,
        },
        "executor": {
            "uid": preflight.EXECUTOR_UID,
            "mutation_iam_binding_present": request["phase"] == "post_iam",
            "activation_seal_present": False,
            "authorization_receipt_signature_self_verified": security_selftest[
                "authorization_receipt_signature_self_verified"
            ],
            "receipt_action_binding_self_verified": security_selftest[
                "receipt_action_binding_self_verified"
            ],
            "local_gcloud_present": False,
            "generic_shell_fallback_present": False,
            "compute_api_connectivity_verified": request["phase"] == "post_iam",
            "numeric_targets_reverified": checked_probe["numeric_targets_reverified"],
            "target_instance_id": foundation.TARGET_INSTANCE_ID,
            "target_disk_id": foundation.TARGET_DISK_ID,
            "receipt_public_key_sha256": executor_key[
                "receipt_public_key_sha256"
            ],
            "receipt_public_key_owner": executor_key[
                "receipt_public_key_owner"
            ],
            "receipt_public_key_mode": executor_key[
                "receipt_public_key_mode"
            ],
        },
        "effective_permission_probe": checked_probe[
            "effective_permission_probe"
        ],
        "secret_material_recorded": False,
    }
    _require_executor_activation_seal_absent()
    return _attest(unsigned, private_key=private_key, public_key_id=key_id)


class MetadataSaProbe:
    """Exact metadata and private-Google-API observer for the attached SA."""

    def _metadata_get(self, path: str, *, maximum: int = 64 * 1024) -> bytearray:
        if path not in {
            METADATA_INSTANCE_ID_PATH,
            METADATA_SERVICE_ACCOUNT_EMAIL_PATH,
            METADATA_SCOPES_PATH,
            METADATA_TOKEN_PATH,
        }:
            _error("owner_gate_attached_sa_metadata_path_invalid")
        connection = http.client.HTTPConnection(
            METADATA_HOST, 80, timeout=HTTP_TIMEOUT_SECONDS
        )
        body: bytearray | None = None
        returned = False
        try:
            connection.request(
                "GET", path,
                headers={"Metadata-Flavor": "Google", "Accept": "application/json"},
            )
            response = connection.getresponse()
            body = bytearray(maximum + 1)
            view = memoryview(body)
            position = 0
            try:
                while position < len(view):
                    read = response.readinto(view[position:])
                    if read is None or read < 0:
                        _error("owner_gate_attached_sa_metadata_invalid")
                    if read == 0:
                        break
                    position += read
            finally:
                view.release()
            del body[position:]
            if (
                response.status != 200
                or response.getheader("Metadata-Flavor") != "Google"
                or not body
                or len(body) > maximum
            ):
                _error("owner_gate_attached_sa_metadata_invalid")
            returned = True
            return body
        except (OSError, http.client.HTTPException):
            _error("owner_gate_attached_sa_metadata_unavailable")
        finally:
            if body is not None and not returned:
                for index in range(len(body)):
                    body[index] = 0
            connection.close()

    @staticmethod
    def _ascii_text(raw: bytearray, *, pattern: re.Pattern[str]) -> str:
        try:
            value = bytes(raw).decode("ascii", errors="strict").strip()
        finally:
            for index in range(len(raw)):
                raw[index] = 0
        if pattern.fullmatch(value) is None:
            _error("owner_gate_attached_sa_metadata_invalid")
        return value

    @staticmethod
    def _token_view(raw: bytearray) -> tuple[memoryview, Mapping[str, Any]]:
        marker = b'"access_token":"'
        start = raw.find(marker)
        if start < 0:
            _error("owner_gate_attached_sa_token_invalid")
        start += len(marker)
        end = raw.find(b'"', start)
        if end <= start or end - start > 16 * 1024:
            _error("owner_gate_attached_sa_token_invalid")
        if any(byte not in _TOKEN_BYTE for byte in raw[start:end]):
            _error("owner_gate_attached_sa_token_invalid")
        redacted = bytearray(raw)
        redacted[start:end] = b"0" * (end - start)
        try:
            value = json.loads(bytes(redacted).decode("ascii", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError):
            _error("owner_gate_attached_sa_token_invalid")
        if (
            not isinstance(value, Mapping)
            or set(value) != {"access_token", "expires_in", "token_type"}
            or value.get("access_token") != "0" * (end - start)
            or type(value.get("expires_in")) is not int
            or not 1 <= value["expires_in"] <= 7200
            or value.get("token_type") != "Bearer"
        ):
            _error("owner_gate_attached_sa_token_invalid")
        return memoryview(raw)[start:end], value

    @staticmethod
    def _https_json(host: str, path: str, body: bytes, token: memoryview) -> Mapping[str, Any]:
        allowed = {
            "compute.googleapis.com": (
                f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                f"instances/{foundation.TARGET_INSTANCE}/testIamPermissions",
                f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                f"disks/{foundation.TARGET_DISK}/testIamPermissions",
                f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                f"instances/{foundation.TARGET_INSTANCE}",
                f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                f"disks/{foundation.TARGET_DISK}",
            ),
            "iam.googleapis.com": (
                f"/v1/projects/{foundation.PROJECT}/serviceAccounts/"
                f"{foundation.SERVICE_ACCOUNT_NAME}%40{foundation.PROJECT}.iam.gserviceaccount.com"
                ":testIamPermissions",
            ),
            "cloudresourcemanager.googleapis.com": (
                f"/v1/projects/{foundation.PROJECT}:testIamPermissions",
            ),
        }
        if host not in allowed or path not in allowed[host]:
            _error("owner_gate_attached_sa_api_target_invalid")
        method = b"GET" if not body else b"POST"
        request = bytearray()
        request.extend(method + b" " + path.encode("ascii") + b" HTTP/1.1\r\n")
        request.extend(b"Host: " + host.encode("ascii") + b"\r\n")
        request.extend(b"Authorization: Bearer ")
        request.extend(token)
        request.extend(b"\r\nAccept: application/json\r\nConnection: close\r\n")
        if body:
            request.extend(b"Content-Type: application/json\r\nContent-Length: ")
            request.extend(str(len(body)).encode("ascii"))
            request.extend(b"\r\n")
        request.extend(b"\r\n")
        request.extend(body)
        response = bytearray()
        raw_socket: socket.socket | None = None
        tls_socket: Any = None
        try:
            raw_socket = socket.create_connection(
                (PRIVATE_GOOGLE_API_VIP, 443), timeout=HTTP_TIMEOUT_SECONDS
            )
            tls_socket = trusted.fixed_debian_tls_context().wrap_socket(
                raw_socket, server_hostname=host
            )
            raw_socket = None
            tls_socket.sendall(request)
            while len(response) <= MAX_HTTP_BYTES:
                chunk = tls_socket.recv(min(64 * 1024, MAX_HTTP_BYTES + 1 - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
        except (OSError, ValueError):
            _error("owner_gate_attached_sa_api_unavailable")
        finally:
            for index in range(len(request)):
                request[index] = 0
            if tls_socket is not None:
                tls_socket.close()
            if raw_socket is not None:
                raw_socket.close()
        return _parse_api_response(bytes(response))

    def collect(self, *, post_iam: bool) -> Mapping[str, Any]:
        instance_id = self._ascii_text(
            self._metadata_get(METADATA_INSTANCE_ID_PATH), pattern=_NUMERIC_ID
        )
        service_account = self._ascii_text(
            self._metadata_get(METADATA_SERVICE_ACCOUNT_EMAIL_PATH),
            pattern=re.compile(re.escape(direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL)),
        )
        scopes_raw = self._metadata_get(METADATA_SCOPES_PATH)
        try:
            scopes_text = bytes(scopes_raw).decode("ascii", errors="strict")
        finally:
            for index in range(len(scopes_raw)):
                scopes_raw[index] = 0
        scopes = scopes_text.strip().splitlines()
        expected_scopes = list(foundation.OWNER_GATE_OAUTH_SCOPES)
        if (
            len(scopes) != len(expected_scopes)
            or len(scopes) != len(set(scopes))
            or set(scopes) != set(expected_scopes)
        ):
            _error("owner_gate_attached_sa_metadata_invalid")
        scopes = expected_scopes
        token_raw = self._metadata_get(METADATA_TOKEN_PATH)
        token: memoryview | None = None
        try:
            token, _token_meta = self._token_view(token_raw)
            probes = (
                ("instance", "compute.googleapis.com", (
                    f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                    f"instances/{foundation.TARGET_INSTANCE}/testIamPermissions"
                ), preflight.INSTANCE_PERMISSION_PROBE),
                ("disk", "compute.googleapis.com", (
                    f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                    f"disks/{foundation.TARGET_DISK}/testIamPermissions"
                ), preflight.DISK_PERMISSION_PROBE),
                ("service_account", "iam.googleapis.com", (
                    f"/v1/projects/{foundation.PROJECT}/serviceAccounts/"
                    f"{foundation.SERVICE_ACCOUNT_NAME}%40{foundation.PROJECT}.iam.gserviceaccount.com"
                    ":testIamPermissions"
                ), preflight.SERVICE_ACCOUNT_PERMISSION_PROBE),
                ("project", "cloudresourcemanager.googleapis.com", (
                    f"/v1/projects/{foundation.PROJECT}:testIamPermissions"
                ), preflight.PROJECT_PERMISSION_PROBE),
            )
            effective = preflight.expected_effective_permission_probe(post_iam)
            normalized: dict[str, list[str]] = {}
            for name, host, path, permissions in probes:
                response = self._https_json(
                    host, path,
                    _canonical({"permissions": list(permissions)}), token,
                )
                if set(response) - {"permissions"}:
                    _error("owner_gate_attached_sa_permission_response_invalid")
                granted = response.get("permissions", [])
                if (
                    not isinstance(granted, list)
                    or any(not isinstance(item, str) for item in granted)
                    or len(granted) != len(set(granted))
                ):
                    _error("owner_gate_attached_sa_permission_response_invalid")
                normalized[name] = sorted(granted)
                if normalized[name] != effective[name]["granted_permissions"]:
                    _error("owner_gate_attached_sa_permission_mismatch")
            target_instance_id: str | None = None
            target_disk_id: str | None = None
            if post_iam:
                instance = self._https_json(
                    "compute.googleapis.com",
                    f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                    f"instances/{foundation.TARGET_INSTANCE}", b"", token,
                )
                disk = self._https_json(
                    "compute.googleapis.com",
                    f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
                    f"disks/{foundation.TARGET_DISK}", b"", token,
                )
                target_instance_id = str(instance.get("id", ""))
                target_disk_id = str(disk.get("id", ""))
                if (
                    target_instance_id != foundation.TARGET_INSTANCE_ID
                    or target_disk_id != foundation.TARGET_DISK_ID
                ):
                    _error("owner_gate_attached_sa_numeric_target_mismatch")
            return {
                "runtime_instance_numeric_id": instance_id,
                "runtime_service_account_email": service_account,
                "metadata_scopes": scopes,
                "effective_permission_probe": effective,
                "target_instance_numeric_id": foundation.TARGET_INSTANCE_ID,
                "target_disk_numeric_id": foundation.TARGET_DISK_ID,
                "numeric_targets_reverified": post_iam,
            }
        finally:
            if token is not None:
                token.release()
            for index in range(len(token_raw)):
                token_raw[index] = 0


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    try:
        position = 0
        while position < len(view):
            try:
                written = os.write(descriptor, view[position:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            position += written
    finally:
        view.release()


def _executor_child_payload(
    *, post_iam: bool, collector: MetadataSaProbe | None = None
) -> Mapping[str, Any]:
    identity = {
        "uid": os.getuid(),  # windows-footgun: ok — Linux-only UID-drop child
        "euid": os.geteuid(),  # windows-footgun: ok — Linux-only UID-drop child
        "gid": os.getgid(),  # windows-footgun: ok — Linux-only UID-drop child
        "egid": os.getegid(),  # windows-footgun: ok — Linux-only UID-drop child
        "groups": os.getgroups(),
    }
    if identity != {
        "uid": preflight.EXECUTOR_UID,
        "euid": preflight.EXECUTOR_UID,
        "gid": preflight.EXECUTOR_UID,
        "egid": preflight.EXECUTOR_UID,
        "groups": [],
    }:
        _error("owner_gate_attached_sa_child_identity_invalid")
    return {
        "schema": ATTACHED_SA_CHILD_SCHEMA,
        **identity,
        "facts": (collector or MetadataSaProbe()).collect(post_iam=post_iam),
    }


def _collect_attached_sa_child(*, post_iam: bool) -> Mapping[str, Any]:
    """Collect metadata/API facts only after a one-way executor UID drop."""

    if os.getuid() != 0 or os.geteuid() != 0:  # windows-footgun: ok — Linux-only UID-drop parent
        _error("owner_gate_attached_sa_parent_identity_invalid")
    read_descriptor, write_descriptor = os.pipe()
    try:
        pid = os.fork()  # windows-footgun: ok — Linux-only one-way UID-drop child
    except OSError:
        os.close(read_descriptor)
        os.close(write_descriptor)
        _error("owner_gate_attached_sa_child_unavailable")
    if pid == 0:  # pragma: no branch - exercised by subprocess integration
        try:
            os.close(read_descriptor)
            os.setgroups([])
            os.setgid(preflight.EXECUTOR_UID)
            os.setuid(preflight.EXECUTOR_UID)
            frame = _executor_child_payload(post_iam=post_iam)
            raw = _canonical(frame) + b"\n"
            if len(raw) > MAX_REQUEST_BYTES + 1:
                os._exit(113)
            _write_all(write_descriptor, raw)
            os.close(write_descriptor)
            os._exit(0)
        except BaseException:
            try:
                os.close(write_descriptor)
            except OSError:
                pass
            os._exit(111)

    os.close(write_descriptor)
    raw = bytearray()
    try:
        while len(raw) <= MAX_REQUEST_BYTES:
            try:
                chunk = os.read(
                    read_descriptor,
                    min(64 * 1024, MAX_REQUEST_BYTES + 1 - len(raw)),
                )
            except InterruptedError:
                continue
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(read_descriptor)
    while True:
        try:
            waited_pid, status = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
        except OSError:
            _error("owner_gate_attached_sa_child_invalid")
    if (
        waited_pid != pid
        or not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
        or len(raw) > MAX_REQUEST_BYTES + 1
    ):
        _error("owner_gate_attached_sa_child_invalid")
    frame = _decode_canonical(
        bytes(raw[:-1]),
        maximum=MAX_REQUEST_BYTES,
        code="owner_gate_attached_sa_child_invalid",
    )
    if (
        set(frame) != {
            "schema", "uid", "euid", "gid", "egid", "groups", "facts"
        }
        or frame.get("schema") != ATTACHED_SA_CHILD_SCHEMA
        or frame.get("uid") != preflight.EXECUTOR_UID
        or frame.get("euid") != preflight.EXECUTOR_UID
        or frame.get("gid") != preflight.EXECUTOR_UID
        or frame.get("egid") != preflight.EXECUTOR_UID
        or frame.get("groups") != []
        or not isinstance(frame.get("facts"), Mapping)
    ):
        _error("owner_gate_attached_sa_child_invalid")
    facts = dict(frame["facts"])
    if set(facts) != {
        "runtime_instance_numeric_id",
        "runtime_service_account_email",
        "metadata_scopes",
        "effective_permission_probe",
        "target_instance_numeric_id",
        "target_disk_numeric_id",
        "numeric_targets_reverified",
    }:
        _error("owner_gate_attached_sa_child_invalid")
    return facts


def build_attached_sa_permission_probe(
    request_value: Mapping[str, Any],
    *,
    release_revision: str,
    collector: MetadataSaProbe | None = None,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    started_monotonic = time.monotonic()
    observed = int(time.time()) if now_unix is None else now_unix
    request, package, install = _validate_request(
        request_value,
        schema=ATTACHED_SA_REQUEST_SCHEMA,
        release_revision=release_revision,
        now_unix=observed,
    )
    direct = _load_direct_iam_identity(package)
    post_iam = request["phase"] == "post_iam"
    facts = (
        collector.collect(post_iam=post_iam)
        if collector is not None
        else _collect_attached_sa_child(post_iam=post_iam)
    )
    if (
        facts.get("runtime_instance_numeric_id")
        != direct["owner_gate_vm_numeric_id"]
        or facts.get("runtime_service_account_email")
        != direct["owner_gate_service_account_email"]
        or facts.get("metadata_scopes") != direct["metadata_oauth_scopes"]
        or facts.get("effective_permission_probe")
        != preflight.expected_effective_permission_probe(post_iam)
        or facts.get("target_instance_numeric_id")
        != foundation.TARGET_INSTANCE_ID
        or facts.get("target_disk_numeric_id") != foundation.TARGET_DISK_ID
        or facts.get("numeric_targets_reverified") is not post_iam
    ):
        _error("owner_gate_attached_sa_runtime_identity_mismatch")
    # Reject a stalled unprivileged collection before reading the private
    # signer.  A second check below stamps the final signed completion time.
    _completion_facts(
        request,
        observed_at_entry=observed,
        started_monotonic=started_monotonic,
        injected_now=now_unix,
    )
    private_key, _public_key, key_id, host_readiness = _load_host_signer(
        release_revision
    )
    try:
        cloud_readiness = provisioning.verify_cloud_signer_inert_readiness(
            release_revision
        )
    except provisioning.TrustedSignerProvisioningError:
        _error("owner_gate_cloud_signer_not_ready")
    if (
        request["host_signer_provisioning_receipt_sha256"]
        != host_readiness.get("provisioning_receipt_sha256")
        or request["host_signer_readiness_sha256"]
        != host_readiness.get("readiness_sha256")
        or request["cloud_signer_provisioning_receipt_sha256"]
        != cloud_readiness.get("provisioning_receipt_sha256")
        or request["cloud_signer_readiness_sha256"]
        != cloud_readiness.get("readiness_sha256")
    ):
        _error("owner_gate_signer_lineage_invalid")
    completed_at_unix, fresh_through_unix = _completion_facts(
        request,
        observed_at_entry=observed,
        started_monotonic=started_monotonic,
        injected_now=now_unix,
    )
    unsigned = {
        "schema": ATTACHED_SA_PROBE_SCHEMA,
        "phase": request["phase"],
        "collected_at_unix": request["collected_at_unix"],
        "completed_at_unix": completed_at_unix,
        "fresh_through_unix": fresh_through_unix,
        "plan_sha256": request["plan_sha256"],
        "observation_binding_sha256": request[
            "observation_binding_sha256"
        ],
        "attached_request_sha256": request["request_sha256"],
        "release_revision": release_revision,
        "source_tree_oid": package["source_tree_oid"],
        "package_sha256": package["package_sha256"],
        "package_inventory_sha256": package["package_inventory_sha256"],
        **{name: package[name] for name in _LINEAGE_FIELDS},
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "install_receipt_sha256": install["receipt_sha256"],
        **{name: request[name] for name in _SIGNER_LINEAGE_FIELDS},
        "runtime_instance_numeric_id": facts["runtime_instance_numeric_id"],
        "runtime_service_account_email": facts["runtime_service_account_email"],
        "runtime_service_account_unique_id": direct[
            "owner_gate_service_account_unique_id"
        ],
        "metadata_scopes": facts["metadata_scopes"],
        "effective_permission_probe": facts["effective_permission_probe"],
        "target_instance_numeric_id": facts["target_instance_numeric_id"],
        "target_disk_numeric_id": facts["target_disk_numeric_id"],
        "numeric_targets_reverified": facts["numeric_targets_reverified"],
        "inherited_bindings_evaluated": True,
        "conditional_bindings_evaluated": True,
        "metadata_token_acquired": True,
        "metadata_token_wiped": True,
        "owner_credential_values_read": False,
    }
    return _attest(unsigned, private_key=private_key, public_key_id=key_id)


def _runtime_revision() -> str:
    executable = Path(sys.executable).resolve(strict=True)
    candidates = [
        parent
        for parent in executable.parents
        if parent.parent == stage0.HOST_TRUSTED_OBSERVATION_RELEASE_BASE
    ]
    if len(candidates) != 1 or _REVISION.fullmatch(candidates[0].name) is None:
        _error("owner_gate_host_observation_runtime_invalid")
    return candidates[0].name


def attached_sa_main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != 0:  # windows-footgun: ok — Linux-only host entrypoint
        _error("owner_gate_attached_sa_entrypoint_invalid")
    revision = _runtime_revision()
    request = _read_stdin(sys.stdin.buffer)
    response = build_attached_sa_permission_probe(
        request, release_revision=revision
    )
    _write_stdout(sys.stdout.buffer, response)
    return 0


def _run_host_frame(
    frame_value: Mapping[str, Any],
    *,
    release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(frame_value, Mapping) or set(frame_value) != {
        "schema", "request", "attached_sa_probe", "frame_sha256"
    }:
        _error("owner_gate_host_observation_frame_invalid")
    frame = dict(frame_value)
    unsigned = {
        name: item for name, item in frame.items() if name != "frame_sha256"
    }
    if (
        frame.get("schema") != HOST_FRAME_SCHEMA
        or frame.get("frame_sha256") != _sha(unsigned)
        or not isinstance(frame.get("request"), Mapping)
        or not isinstance(frame.get("attached_sa_probe"), Mapping)
    ):
        _error("owner_gate_host_observation_frame_invalid")
    return build_host_observation(
        frame["request"],
        release_revision=release_revision,
        attached_sa_probe=frame["attached_sa_probe"],
        now_unix=now_unix,
    )


def host_observation_main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != 0:  # windows-footgun: ok — Linux-only host entrypoint
        _error("owner_gate_host_observation_entrypoint_invalid")
    revision = _runtime_revision()
    frame = _read_stdin(sys.stdin.buffer)
    response = _run_host_frame(
        frame,
        release_revision=revision,
        now_unix=int(time.time()),
    )
    _write_stdout(sys.stdout.buffer, response)
    return 0


def observation_dispatcher_main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the existing exact sudo command by canonical frame schema."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments or os.geteuid() != 0:  # windows-footgun: ok — Linux-only host entrypoint
        _error("owner_gate_host_observation_entrypoint_invalid")
    revision = _runtime_revision()
    frame = _read_stdin(sys.stdin.buffer)
    schema = frame.get("schema")
    now_unix = int(time.time())
    if schema == trusted.ATTESTATION_REQUEST_SCHEMA:
        try:
            provisioning.verify_host_signer_runtime_readiness(revision)
            config = trusted._load_config(
                trusted.HOST_CONFIG_PATH,
                role="host",
                expected_uid=0,
                expected_path=trusted.HOST_CONFIG_PATH,
                expected_private_key_path=trusted.HOST_PRIVATE_KEY_PATH,
                expected_replay_directory=trusted.HOST_REPLAY_DIRECTORY,
            )
            response = trusted.run_host_attestor(
                frame,
                config=config,
                facts_reader=trusted.FixedHostFactsReader(),
                now_unix=now_unix,
            )
        except (
            provisioning.TrustedSignerProvisioningError,
            trusted.TrustedObservationError,
        ):
            _error("owner_gate_host_observation_storage_frame_invalid")
    elif schema == ATTACHED_SA_REQUEST_SCHEMA:
        response = build_attached_sa_permission_probe(
            frame,
            release_revision=revision,
            now_unix=now_unix,
        )
    elif schema == HOST_FRAME_SCHEMA:
        response = _run_host_frame(
            frame,
            release_revision=revision,
            now_unix=now_unix,
        )
    else:
        _error("owner_gate_host_observation_frame_schema_invalid")
    _write_stdout(sys.stdout.buffer, response)
    return 0


__all__ = [
    "ATTACHED_SA_PROBE_SCHEMA",
    "ATTACHED_SA_REQUEST_SCHEMA",
    "HOST_REQUEST_SCHEMA",
    "HOST_FRAME_SCHEMA",
    "MetadataSaProbe",
    "OwnerGateHostObservationError",
    "attached_sa_main",
    "build_attached_sa_permission_probe",
    "build_host_observation",
    "host_observation_main",
    "observation_dispatcher_main",
]
