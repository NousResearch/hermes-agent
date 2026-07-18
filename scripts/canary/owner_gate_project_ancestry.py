#!/usr/bin/env python3
"""Collect and validate signed project-to-organization ancestry evidence.

The live producer accepts only the sealed owner gcloud runtime and its exact
pinned configuration.  It obtains a short-lived owner bearer token without
recording it, reads the Cloud Resource Manager v3 chain twice, requires the two
complete normalized chains to be identical, and signs the result with the
release-bound bootstrap observation key.  It never mutates project, folder,
organization, policy, or IAM state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Never, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as signer_author


EVIDENCE_SCHEMA = "muncho-owner-gate-project-ancestry-evidence.v1"
EVIDENCE_PURPOSE = "muncho_owner_gate_stable_project_ancestry_observation"
SIGNATURE_DOMAIN = b"muncho-owner-gate/project-ancestry-evidence/v1\x00"
MAX_EVIDENCE_TTL_SECONDS = foundation.PREFLIGHT_MAX_AGE_SECONDS
MAX_CHAIN_DEPTH = 32
MAX_JSON_BYTES = 4 * 1024 * 1024
_RESOURCE_MANAGER_BASE = "https://cloudresourcemanager.googleapis.com/v3/"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_RESOURCE = re.compile(r"^(projects|folders|organizations)/([1-9][0-9]{5,30})$")
_OPAQUE = re.compile(r"^[^\x00\r\n]{1,1024}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]{86}$")
_FORBIDDEN_NETWORK_ENVIRONMENT = frozenset({
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
    "CURL_CA_BUNDLE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
})

_NODE_FIELDS = frozenset({
    "resource_type",
    "resource_name",
    "numeric_id",
    "display_name",
    "state",
    "etag",
    "parent_resource_name",
})
_STABLE_READ_FIELDS = frozenset({
    "ordinal",
    "chain_sha256",
    "provider_consistency_token_sha256",
})
_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "release_revision",
    "project_id",
    "project_number",
    "project_resource_name",
    "organization_id",
    "organization_resource_name",
    "ordered_chain",
    "stable_chain_sha256",
    "stable_reads",
    "collected_at_unix",
    "expires_at_unix",
    "owner_reauthentication_receipt_sha256",
    "collector_public_key_id",
})
_FIELDS = _BODY_FIELDS | frozenset({
    "evidence_sha256",
    "signature_ed25519_b64url",
})


class OwnerGateProjectAncestryError(RuntimeError):
    """Stable, secret-free project ancestry evidence failure."""


def _error(code: str, exc: BaseException | None = None) -> Never:
    del exc
    raise OwnerGateProjectAncestryError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_project_ancestry_json_invalid", exc)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        _error(code)
    return value


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or _B64URL.fullmatch(value) is None:
        _error("owner_gate_project_ancestry_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_project_ancestry_signature_invalid", exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        != value
    ):
        _error("owner_gate_project_ancestry_signature_invalid")
    return raw


def _key_id(public_key: Ed25519PublicKey) -> str:
    if not isinstance(public_key, Ed25519PublicKey):
        _error("owner_gate_project_ancestry_collector_key_invalid")
    return _sha256_bytes(public_key.public_bytes_raw())


def _normalize_resource_name(value: Any, *, expected_type: str) -> tuple[str, str]:
    if not isinstance(value, str):
        _error("owner_gate_project_ancestry_chain_invalid")
    match = _RESOURCE.fullmatch(value)
    if match is None or match.group(1) != expected_type:
        _error("owner_gate_project_ancestry_chain_invalid")
    return value, match.group(2)


def _normalized_node(
    value: Any,
    *,
    expected_resource_name: str,
    expected_type: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_project_ancestry_chain_invalid")
    resource_name, numeric_id = _normalize_resource_name(
        value.get("name"),
        expected_type=expected_type,
    )
    display_name = value.get("displayName")
    state = value.get("state")
    etag = value.get("etag")
    parent = value.get("parent")
    if expected_type == "organizations":
        if parent is not None:
            _error("owner_gate_project_ancestry_chain_invalid")
        normalized_parent = None
    else:
        if not isinstance(parent, str) or _RESOURCE.fullmatch(parent) is None:
            _error("owner_gate_project_ancestry_chain_invalid")
        normalized_parent = parent
    if (
        resource_name != expected_resource_name
        or not isinstance(display_name, str)
        or _OPAQUE.fullmatch(display_name) is None
        or state != "ACTIVE"
        or not isinstance(etag, str)
        or _OPAQUE.fullmatch(etag) is None
    ):
        _error("owner_gate_project_ancestry_chain_invalid")
    return {
        "resource_type": expected_type.removesuffix("s"),
        "resource_name": resource_name,
        "numeric_id": numeric_id,
        "display_name": display_name,
        "state": state,
        "etag": etag,
        "parent_resource_name": normalized_parent,
    }


ReadResource = Callable[[str], Mapping[str, Any]]


class _StableOwnerTokenProvider(Protocol):
    """Private capability shape used only after the exact runtime is sealed."""

    def __call__(self) -> str: ...

    def require_stable(self) -> None: ...


def _read_chain(
    read_resource: ReadResource,
    *,
    project_id: str,
    project_number: str,
) -> list[Mapping[str, Any]]:
    if not callable(read_resource):
        _error("owner_gate_project_ancestry_reader_invalid")
    current = f"projects/{project_number}"
    expected_type = "projects"
    chain: list[Mapping[str, Any]] = []
    visited: set[str] = set()
    for _depth in range(MAX_CHAIN_DEPTH):
        if current in visited:
            _error("owner_gate_project_ancestry_cycle")
        visited.add(current)
        node_raw = read_resource(current)
        node = _normalized_node(
            node_raw,
            expected_resource_name=current,
            expected_type=expected_type,
        )
        if expected_type == "projects" and node_raw.get("projectId") != project_id:
            _error("owner_gate_project_ancestry_project_mismatch")
        chain.append(node)
        parent = node["parent_resource_name"]
        if expected_type == "organizations":
            if parent is not None:
                _error("owner_gate_project_ancestry_chain_invalid")
            return chain
        if not isinstance(parent, str):
            _error("owner_gate_project_ancestry_chain_invalid")
        parent_match = _RESOURCE.fullmatch(parent)
        if parent_match is None or parent_match.group(1) not in {
            "folders",
            "organizations",
        }:
            _error("owner_gate_project_ancestry_chain_invalid")
        expected_type = parent_match.group(1)
        current = parent
    _error("owner_gate_project_ancestry_chain_too_deep")


def _consistency_token(chain: list[Mapping[str, Any]]) -> str:
    return _sha256_json([
        {
            "resource_name": item["resource_name"],
            "state": item["state"],
            "etag": item["etag"],
            "parent_resource_name": item["parent_resource_name"],
        }
        for item in chain
    ])


def _validate_body(value: Any, *, now_unix: int | None) -> Mapping[str, Any]:
    body = _strict_mapping(
        value,
        _BODY_FIELDS,
        code="owner_gate_project_ancestry_evidence_invalid",
    )
    chain_value = body.get("ordered_chain")
    reads_value = body.get("stable_reads")
    if (
        not isinstance(chain_value, list)
        or not 2 <= len(chain_value) <= MAX_CHAIN_DEPTH
        or not isinstance(reads_value, list)
        or len(reads_value) != 2
    ):
        _error("owner_gate_project_ancestry_evidence_invalid")
    chain: list[Mapping[str, Any]] = []
    for index, raw_node in enumerate(chain_value):
        node = _strict_mapping(
            raw_node,
            _NODE_FIELDS,
            code="owner_gate_project_ancestry_evidence_invalid",
        )
        expected_type = (
            "project"
            if index == 0
            else "organization"
            if index == len(chain_value) - 1
            else "folder"
        )
        resource_name = node.get("resource_name")
        match = _RESOURCE.fullmatch(str(resource_name or ""))
        parent = node.get("parent_resource_name")
        if (
            node.get("resource_type") != expected_type
            or match is None
            or match.group(1) != expected_type + "s"
            or node.get("numeric_id") != match.group(2)
            or not isinstance(node.get("display_name"), str)
            or _OPAQUE.fullmatch(str(node.get("display_name", ""))) is None
            or node.get("state") != "ACTIVE"
            or not isinstance(node.get("etag"), str)
            or _OPAQUE.fullmatch(str(node.get("etag", ""))) is None
            or (
                index == len(chain_value) - 1
                and parent is not None
            )
            or (
                index < len(chain_value) - 1
                and parent != chain_value[index + 1].get("resource_name")
            )
        ):
            _error("owner_gate_project_ancestry_evidence_invalid")
        chain.append(dict(node))
    chain_sha = _sha256_json(chain)
    token = _consistency_token(chain)
    reads: list[Mapping[str, Any]] = []
    for index, raw_read in enumerate(reads_value, start=1):
        item = _strict_mapping(
            raw_read,
            _STABLE_READ_FIELDS,
            code="owner_gate_project_ancestry_evidence_invalid",
        )
        if item != {
            "ordinal": index,
            "chain_sha256": chain_sha,
            "provider_consistency_token_sha256": token,
        }:
            _error("owner_gate_project_ancestry_evidence_invalid")
        reads.append(dict(item))
    first = chain[0]
    last = chain[-1]
    collected = body.get("collected_at_unix")
    expires = body.get("expires_at_unix")
    if (
        body.get("schema") != EVIDENCE_SCHEMA
        or body.get("purpose") != EVIDENCE_PURPOSE
        or _REVISION.fullmatch(str(body.get("release_revision", ""))) is None
        or body.get("project_id") != foundation.PROJECT
        or _PROJECT_ID.fullmatch(str(body.get("project_id", ""))) is None
        or body.get("project_number") != first["numeric_id"]
        or body.get("project_resource_name") != first["resource_name"]
        or body.get("organization_id") != last["numeric_id"]
        or body.get("organization_resource_name") != last["resource_name"]
        or body.get("stable_chain_sha256") != chain_sha
        or type(collected) is not int
        or type(expires) is not int
        or collected <= 0
        or expires <= collected
        or expires - collected > MAX_EVIDENCE_TTL_SECONDS
        or _SHA256.fullmatch(
            str(body.get("owner_reauthentication_receipt_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(str(body.get("collector_public_key_id", "")))
        is None
    ):
        _error("owner_gate_project_ancestry_evidence_invalid")
    if now_unix is not None and (
        type(now_unix) is not int
        or now_unix < collected
        or now_unix > expires
    ):
        _error("owner_gate_project_ancestry_evidence_expired")
    return {
        **dict(body),
        "ordered_chain": chain,
        "stable_reads": reads,
    }


@dataclass(frozen=True)
class ProjectAncestryEvidence:
    """Validated, signed immutable project ancestry observation."""

    value: Mapping[str, Any]
    canonical_bytes: bytes

    @property
    def signed_evidence_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes)

    @property
    def organization_id(self) -> str:
        return str(self.value["organization_id"])

    @property
    def project_number(self) -> str:
        return str(self.value["project_number"])

    @property
    def ordered_chain(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self.value["ordered_chain"])

    @property
    def collector_public_key_id(self) -> str:
        return str(self.value["collector_public_key_id"])


def validate_project_ancestry_evidence(
    value: Any,
    *,
    collector_public_key: Ed25519PublicKey,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    expected_release_revision: str,
    now_unix: int | None = None,
    canonical_bytes: bytes | None = None,
) -> ProjectAncestryEvidence:
    receipt = _strict_mapping(
        value,
        _FIELDS,
        code="owner_gate_project_ancestry_evidence_invalid",
    )
    body = {
        key: item
        for key, item in receipt.items()
        if key not in {"evidence_sha256", "signature_ed25519_b64url"}
    }
    checked = _validate_body(body, now_unix=now_unix)
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=owner_reauthentication_public_key,
            now_unix=checked["collected_at_unix"],
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_project_ancestry_owner_reauth_invalid", exc)
    signed_payload = {
        **checked,
        "evidence_sha256": receipt.get("evidence_sha256"),
    }
    collector_key_id = _key_id(collector_public_key)
    if (
        expected_release_revision != checked["release_revision"]
        or checked_reauth["trusted_runtime_identity"]["release_revision"]
        != expected_release_revision
        or checked["project_number"]
        != checked_reauth["authenticated_probe"]["project_number"]
        or checked["owner_reauthentication_receipt_sha256"]
        != checked_reauth["owner_reauthentication_receipt_sha256"]
        or checked["expires_at_unix"] > checked_reauth["expires_at_unix"]
        or checked["collector_public_key_id"] != collector_key_id
        or receipt.get("evidence_sha256") != _sha256_json(checked)
    ):
        _error("owner_gate_project_ancestry_evidence_invalid")
    try:
        collector_public_key.verify(
            _decode_signature(receipt.get("signature_ed25519_b64url")),
            SIGNATURE_DOMAIN + _canonical(signed_payload),
        )
    except InvalidSignature as exc:
        _error("owner_gate_project_ancestry_signature_invalid", exc)
    raw = _canonical(receipt) if canonical_bytes is None else canonical_bytes
    if _canonical(receipt) != raw:
        _error("owner_gate_project_ancestry_canonical_invalid")
    return ProjectAncestryEvidence(dict(receipt), raw)


def decode_canonical_project_ancestry_evidence(
    raw: bytes,
    **validation: Any,
) -> ProjectAncestryEvidence:
    if type(raw) is not bytes or not raw or len(raw) > MAX_JSON_BYTES:
        _error("owner_gate_project_ancestry_canonical_invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_project_ancestry_canonical_invalid", exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error("owner_gate_project_ancestry_canonical_invalid")
    return validate_project_ancestry_evidence(
        value,
        canonical_bytes=raw,
        **validation,
    )


def _load_collector_private_key(
    release_revision: str,
) -> Ed25519PrivateKey:
    try:
        private_raw = release_author._read_exact_regular(
            signer_author._private_path(release_revision, "network"),
            size=32,
            modes=frozenset({0o600}),
            code="owner_gate_project_ancestry_collector_key_invalid",
        )
        public_raw = release_author._read_exact_regular(
            signer_author._public_path(release_revision, "network"),
            size=32,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_project_ancestry_collector_key_invalid",
        )
    except release_author.OwnerGateTrustAuthorError as exc:
        _error("owner_gate_project_ancestry_collector_key_invalid", exc)
    key = Ed25519PrivateKey.from_private_bytes(private_raw)
    if key.public_key().public_bytes_raw() != public_raw:
        _error("owner_gate_project_ancestry_collector_key_invalid")
    return key


def _collect_and_author_with_reader(
    *,
    release_revision: str,
    read_resource: ReadResource,
    collected_at_unix: int,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Private deterministic seam after the exact live capability is bound."""

    if type(collected_at_unix) is not int or collected_at_unix <= 0:
        _error("owner_gate_project_ancestry_time_invalid")
    signer_author._require_authority_directories(release_revision, create=False)
    try:
        checked_reauth = owner_reauth.validate_owner_reauth_receipt(
            owner_reauthentication_receipt,
            public_key=owner_reauthentication_public_key,
            now_unix=collected_at_unix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_project_ancestry_owner_reauth_invalid", exc)
    project_number = str(
        checked_reauth["authenticated_probe"]["project_number"]
    )
    first = _read_chain(
        read_resource,
        project_id=foundation.PROJECT,
        project_number=project_number,
    )
    second = _read_chain(
        read_resource,
        project_id=foundation.PROJECT,
        project_number=project_number,
    )
    if first != second:
        _error("owner_gate_project_ancestry_unstable")
    chain_sha = _sha256_json(first)
    token = _consistency_token(first)
    key = _load_collector_private_key(release_revision)
    key_id = _key_id(key.public_key())
    last = first[-1]
    expires = min(
        collected_at_unix + MAX_EVIDENCE_TTL_SECONDS,
        int(checked_reauth["expires_at_unix"]),
    )
    if expires <= collected_at_unix:
        _error("owner_gate_project_ancestry_owner_reauth_invalid")
    body = {
        "schema": EVIDENCE_SCHEMA,
        "purpose": EVIDENCE_PURPOSE,
        "release_revision": release_revision,
        "project_id": foundation.PROJECT,
        "project_number": project_number,
        "project_resource_name": first[0]["resource_name"],
        "organization_id": last["numeric_id"],
        "organization_resource_name": last["resource_name"],
        "ordered_chain": first,
        "stable_chain_sha256": chain_sha,
        "stable_reads": [
            {
                "ordinal": ordinal,
                "chain_sha256": chain_sha,
                "provider_consistency_token_sha256": token,
            }
            for ordinal in (1, 2)
        ],
        "collected_at_unix": collected_at_unix,
        "expires_at_unix": expires,
        "owner_reauthentication_receipt_sha256": checked_reauth[
            "owner_reauthentication_receipt_sha256"
        ],
        "collector_public_key_id": key_id,
    }
    checked = _validate_body(body, now_unix=collected_at_unix)
    signed_payload = {**checked, "evidence_sha256": _sha256_json(checked)}
    result = {
        **signed_payload,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            key.sign(SIGNATURE_DOMAIN + _canonical(signed_payload))
        )
        .rstrip(b"=")
        .decode("ascii"),
    }
    validate_project_ancestry_evidence(
        result,
        collector_public_key=key.public_key(),
        owner_reauthentication_receipt=owner_reauthentication_receipt,
        owner_reauthentication_public_key=owner_reauthentication_public_key,
        expected_release_revision=release_revision,
        now_unix=collected_at_unix,
    )
    return result


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _json_content_type(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = tuple(item.strip() for item in value.split(";"))
    if not parts or parts[0].casefold() != "application/json":
        return False
    return len(parts) == 1 or (
        len(parts) == 2
        and parts[1].casefold().replace(" ", "") == "charset=utf-8"
    )


def _strict_rest_json(raw: bytes) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in result:
                _error("owner_gate_project_ancestry_rest_invalid")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _error("owner_gate_project_ancestry_rest_invalid")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except OwnerGateProjectAncestryError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_project_ancestry_rest_invalid", exc)
    if not isinstance(value, Mapping):
        _error("owner_gate_project_ancestry_rest_invalid")
    return dict(value)


def _resource_manager_get(token: str, resource_name: str) -> Mapping[str, Any]:
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 16 * 1024
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in token)
        or _RESOURCE.fullmatch(resource_name) is None
    ):
        _error("owner_gate_project_ancestry_rest_invalid")
    url = _RESOURCE_MANAGER_BASE + resource_name
    try:
        if any(os.environ.get(name) for name in _FORBIDDEN_NETWORK_ENVIRONMENT):
            _error("owner_gate_project_ancestry_rest_invalid")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="GET",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(
                context=launcher._pinned_system_tls_context()
            ),
            _NoRedirectHandler(),
        )
        with opener.open(request, timeout=30.0) as response:
            content_length = response.headers.get("Content-Length")
            declared_length: int | None = None
            if content_length is not None:
                if not content_length.isdecimal():
                    _error("owner_gate_project_ancestry_rest_invalid")
                declared_length = int(content_length)
            if (
                type(response.status) is not int
                or response.status < 200
                or response.status >= 300
                or response.geturl() != url
                or response.headers.get("Location") is not None
                or not _json_content_type(response.headers.get("Content-Type"))
                or (
                    declared_length is not None
                    and not 0 < declared_length <= MAX_JSON_BYTES
                )
            ):
                _error("owner_gate_project_ancestry_rest_invalid")
            raw = response.read(MAX_JSON_BYTES + 1)
            if declared_length is not None and len(raw) != declared_length:
                _error("owner_gate_project_ancestry_rest_invalid")
    except OwnerGateProjectAncestryError:
        raise
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        launcher.OwnerLauncherError,
    ) as exc:
        _error("owner_gate_project_ancestry_rest_failed", exc)
    if not raw or len(raw) > MAX_JSON_BYTES:
        _error("owner_gate_project_ancestry_rest_invalid")
    return _strict_rest_json(raw)


def _collect_and_author_with_token_provider(
    *,
    release_revision: str,
    collected_at_unix: int,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    token_provider: _StableOwnerTokenProvider,
) -> Mapping[str, Any]:
    """Private seam that binds both full reads to one bearer capability.

    A token rotation between nodes or between the two reads would make the
    purportedly stable ancestry observation span two different credentials.
    Acquire once, close over that exact string for the complete collection,
    and check the provider's sealed runtime both before and after use.
    """

    token = ""
    try:
        token_provider.require_stable()
        token = token_provider()
        if not isinstance(token, str) or not token or len(token) > 16 * 1024:
            _error("owner_gate_project_ancestry_token_invalid")

        def read_resource(resource_name: str) -> Mapping[str, Any]:
            return _resource_manager_get(token, resource_name)

        result = _collect_and_author_with_reader(
            release_revision=release_revision,
            read_resource=read_resource,
            collected_at_unix=collected_at_unix,
            owner_reauthentication_receipt=owner_reauthentication_receipt,
            owner_reauthentication_public_key=(
                owner_reauthentication_public_key
            ),
        )
        token_provider.require_stable()
        return result
    finally:
        token = ""  # Drop the only local bearer reference on every path.
        token_provider.require_stable()


def collect_and_author_project_ancestry(
    *,
    release_revision: str,
    collected_at_unix: int,
    owner_reauthentication_receipt: Mapping[str, Any],
    owner_reauthentication_public_key: Ed25519PublicKey,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
) -> Mapping[str, Any]:
    """Collect live ancestry only through exact sealed owner capability."""

    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or _REVISION.fullmatch(release_revision or "") is None
    ):
        _error("owner_gate_project_ancestry_runtime_invalid")
    try:
        before = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=release_revision,
        )
        gcloud_configuration.assert_stable()
        token_provider = launcher.GcloudOwnerAccessToken(
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
        )
        token_provider.bind_approved_subject(
            _sha256_bytes(owner_reauth.OWNER_ACCOUNT.encode("utf-8"))
        )

        result = _collect_and_author_with_token_provider(
            release_revision=release_revision,
            collected_at_unix=collected_at_unix,
            owner_reauthentication_receipt=owner_reauthentication_receipt,
            owner_reauthentication_public_key=owner_reauthentication_public_key,
            token_provider=token_provider,
        )
        after = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=release_revision,
        )
        gcloud_configuration.assert_stable()
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_project_ancestry_runtime_invalid", exc)
    if before != after:
        _error("owner_gate_project_ancestry_runtime_changed")
    return result


def live_chain_equals_evidence(
    evidence: ProjectAncestryEvidence,
    live_chain: Any,
) -> bool:
    """Exact mechanical comparison used by apply/trust/direct-IAM gates."""

    if not isinstance(evidence, ProjectAncestryEvidence):
        _error("owner_gate_project_ancestry_evidence_invalid")
    try:
        return _canonical(list(evidence.ordered_chain)) == _canonical(live_chain)
    except OwnerGateProjectAncestryError:
        return False


__all__ = [
    "EVIDENCE_SCHEMA",
    "MAX_EVIDENCE_TTL_SECONDS",
    "OwnerGateProjectAncestryError",
    "ProjectAncestryEvidence",
    "SIGNATURE_DOMAIN",
    "collect_and_author_project_ancestry",
    "decode_canonical_project_ancestry_evidence",
    "live_chain_equals_evidence",
    "validate_project_ancestry_evidence",
]
