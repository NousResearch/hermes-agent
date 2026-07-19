#!/usr/bin/env python3
"""Owner-side, read-only author for the direct-IAM identity authority.

The author accepts only the already-approved owner launcher's pinned gcloud
token provider, then reads a bounded allow-list of Compute, Cloud Resource
Manager, and IAM REST resources directly.  It never creates a second weaker
gcloud trust path, never invokes a Cloud mutation command, and never emits the
bearer token.  The resulting asset is validated by
:mod:`direct_iam_identity_authority` and serialized only through the repository
canonical JSON encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import direct_iam_identity_authority as authority_schema
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import source_artifact_publication as source_publication


COMPUTE_HOST = "compute.googleapis.com"
RESOURCE_MANAGER_HOST = "cloudresourcemanager.googleapis.com"
IAM_HOST = "iam.googleapis.com"
HTTP_TIMEOUT_SECONDS = 10
MAX_HTTP_BODY_BYTES = 1024 * 1024
MAX_POLICY_ROLE_DEFINITIONS = 256
MAX_ANCESTOR_DEPTH = 16
MAX_OWNER_INPUT_BYTES = 16 * 1024 * 1024
MAX_OWNER_PUBLIC_KEY_BYTES = 16 * 1024
OWNER_ACCOUNT = "lomliev@adventico.com"
DIRECT_IAM_AUTHORITY_RELATIVE = (
    ".hermes/trusted/owner-gate-direct-iam-identity-authority-v1.json"
)
OWNER_GATE_INSTANCE_PATH = (
    f"/compute/v1/projects/{foundation.PROJECT}/zones/{foundation.ZONE}/"
    f"instances/{foundation.VM_NAME}"
)
PROJECT_RESOURCE = f"projects/{authority_schema.PROJECT_NUMBER}"
PROJECT_POLICY_RESOURCE = f"projects/{foundation.PROJECT}"
OWNER_GATE_SERVICE_ACCOUNT_PATH = (
    f"/v1/projects/{foundation.PROJECT}/serviceAccounts/"
    f"{authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL}"
)
TARGET_SERVICE_ACCOUNT_PATH = (
    f"/v1/projects/{foundation.PROJECT}/serviceAccounts/"
    f"{authority_schema.TARGET_SERVICE_ACCOUNT_EMAIL}"
)
OWNER_GATE_SERVICE_ACCOUNT_POLICY_PATH = (
    f"{OWNER_GATE_SERVICE_ACCOUNT_PATH}:getIamPolicy"
    "?options.requestedPolicyVersion=3"
)
OWNER_GATE_USER_KEYS_PATH = (
    f"{OWNER_GATE_SERVICE_ACCOUNT_PATH}/keys?keyTypes=USER_MANAGED"
)
PROJECT_READ_ROLE = (
    f"projects/{foundation.PROJECT}/roles/"
    f"{foundation.READ_ONLY_IAM_ROLE_ID}"
)
MUTATION_ROLE = (
    f"projects/{foundation.PROJECT}/roles/{foundation.MUTATION_ROLE_ID}"
)
MUTATION_CONDITION = {
    "title": foundation.MUTATION_CONDITION_TITLE,
    "description": foundation.MUTATION_CONDITION_DESCRIPTION,
    "expression": foundation._condition_expression(),
}
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_FOLDER = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION = re.compile(r"^organizations/[1-9][0-9]{5,30}$")
_BUILTIN_ROLE = re.compile(r"^roles/[A-Za-z0-9_.]{1,128}$")
_PROJECT_ROLE = re.compile(
    rf"^projects/{re.escape(foundation.PROJECT)}/roles/"
    r"[A-Za-z0-9_.]{1,128}$"
)
_ORGANIZATION_ROLE = re.compile(
    r"^organizations/([1-9][0-9]{5,30})/roles/"
    r"[A-Za-z0-9_.]{1,128}$"
)
_ROLE_STAGES = {"ALPHA", "BETA", "GA", "DEPRECATED", "DISABLED", "EAP"}


class DirectIamIdentityAuthorError(RuntimeError):
    """Stable, credential-free owner-side authoring failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    # These failures cross an owner-support CLI/journald boundary.  Preserve
    # only the stable code: provider URLs, TLS diagnostics and OS paths from a
    # nested exception must never become part of an uncaught traceback.
    del exc
    raise DirectIamIdentityAuthorError(code) from None


_TOKEN_MARKER = object()
_AUTHORITY_MARKER = object()


class _GcloudAccessToken:
    """Opaque token capability created only by the fixed gcloud command."""

    __slots__ = ("_raw",)

    def __init__(self, raw: bytearray, *, marker: object) -> None:
        if (
            marker is not _TOKEN_MARKER
            or not isinstance(raw, bytearray)
            or len(raw) < 20
            or any(byte < 0x21 or byte > 0x7E for byte in raw)
            or b":" in raw
        ):
            _error("direct_iam_identity_author_gcloud_token_invalid")
        self._raw = raw


@dataclass(frozen=True)
class _FoundationChainProjection:
    release_revision: str
    owner_reauthentication_receipt_sha256: str
    owner_reauthentication_expires_at_unix: int
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str
    foundation_apply_completed_at_unix: int
    project_number: str
    resource_ancestor_chain: tuple[str, ...]
    owner_gate_vm_numeric_id: str
    owner_gate_service_account_unique_id: str

    def validate(self) -> None:
        if (
            _REVISION.fullmatch(self.release_revision or "") is None
            or any(
                _SHA256.fullmatch(value or "") is None
                for value in (
                    self.owner_reauthentication_receipt_sha256,
                    self.pre_foundation_authority_sha256,
                    self.foundation_apply_receipt_sha256,
                )
            )
            or type(self.owner_reauthentication_expires_at_unix) is not int
            or self.owner_reauthentication_expires_at_unix <= 0
            or type(self.foundation_apply_completed_at_unix) is not int
            or self.foundation_apply_completed_at_unix <= 0
            or self.foundation_apply_completed_at_unix
            > self.owner_reauthentication_expires_at_unix
            or self.project_number != authority_schema.PROJECT_NUMBER
            or not self.resource_ancestor_chain
            or len(self.resource_ancestor_chain) > MAX_ANCESTOR_DEPTH
            or _ORGANIZATION.fullmatch(self.resource_ancestor_chain[-1]) is None
            or any(
                _FOLDER.fullmatch(item) is None
                for item in self.resource_ancestor_chain[:-1]
            )
            or _NUMERIC_ID.fullmatch(self.owner_gate_vm_numeric_id or "")
            is None
            or _NUMERIC_ID.fullmatch(
                self.owner_gate_service_account_unique_id or ""
            )
            is None
        ):
            _error("direct_iam_identity_author_foundation_chain_invalid")


@dataclass(frozen=True, init=False)
class CanonicalDirectIamAuthority:
    """Opaque canonical authority validated against one signed apply chain."""

    value: Mapping[str, Any]
    raw: bytes
    raw_sha256: str
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "CanonicalDirectIamAuthority":
        _error("direct_iam_identity_author_authority_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        value: Mapping[str, Any],
        raw: bytes,
    ) -> "CanonicalDirectIamAuthority":
        instance = object.__new__(cls)
        object.__setattr__(instance, "value", dict(value))
        object.__setattr__(instance, "raw", raw)
        object.__setattr__(instance, "raw_sha256", hashlib.sha256(raw).hexdigest())
        object.__setattr__(instance, "_marker", _AUTHORITY_MARKER)
        return instance

    def __post_init__(self) -> None:
        if self._marker is not _AUTHORITY_MARKER:
            _error("direct_iam_identity_author_authority_invalid")


def _projection_from_validated_apply_chain(
    foundation_chain: Any,
    *,
    now_unix: int,
    require_fresh_owner: bool,
) -> _FoundationChainProjection:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    if (
        type(foundation_chain)
        is not foundation_apply.ValidatedFoundationApplyChain
        or getattr(foundation_chain, "_marker", None)
        is not foundation_apply._CHAIN_MARKER
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error("direct_iam_identity_author_foundation_chain_invalid")
    try:
        foundation_a = foundation_chain.foundation_a
        owner_receipt = foundation_a.owner_reauthentication_receipt
        vm = foundation_chain.owner_gate_vm_identity
        service_account = foundation_chain.service_account_identity
        ancestry = tuple(
            str(item["resource_name"])
            for item in foundation_a.ancestry_evidence.ordered_chain[1:]
        )
        projection = _FoundationChainProjection(
            release_revision=foundation_chain.foundation_source_revision,
            owner_reauthentication_receipt_sha256=(
                foundation_chain.owner_reauthentication_receipt_sha256
            ),
            owner_reauthentication_expires_at_unix=int(
                owner_receipt["expires_at_unix"]
            ),
            pre_foundation_authority_sha256=(
                foundation_chain.pre_foundation_authority_sha256
            ),
            foundation_apply_receipt_sha256=(
                foundation_chain.foundation_apply_receipt_sha256
            ),
            foundation_apply_completed_at_unix=int(
                foundation_chain.apply_receipt["completed_at_unix"]
            ),
            project_number=str(foundation_a.ancestry_evidence.project_number),
            resource_ancestor_chain=ancestry,
            owner_gate_vm_numeric_id=str(vm["numeric_id"]),
            owner_gate_service_account_unique_id=str(
                service_account["unique_id"]
            ),
        )
    except DirectIamIdentityAuthorError:
        raise
    except (
        KeyError,
        AttributeError,
        TypeError,
        ValueError,
        foundation_apply.OwnerGateFoundationApplyError,
    ) as exc:
        _error("direct_iam_identity_author_foundation_chain_invalid", exc)
    projection.validate()
    if (
        require_fresh_owner
        and now_unix > projection.owner_reauthentication_expires_at_unix
    ):
        _error("direct_iam_identity_author_owner_reauth_expired")
    return projection


def _revalidated_foundation_projection(
    foundation_chain: Any,
    *,
    now_unix: int,
) -> _FoundationChainProjection:
    """Re-decode the raw signed lineage; never trust a copied marker alone."""

    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    try:
        original_a = foundation_chain.foundation_a
        replay_a = foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=(
                original_a.pre_foundation_authority_raw
            ),
            owner_reauthentication_receipt_raw=(
                original_a.owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=original_a.network_evidence_raw,
            project_ancestry_evidence_raw=original_a.ancestry_evidence_raw,
            release_public_key=original_a.release_public_key,
            network_collector_public_key=(
                original_a.network_collector_public_key
            ),
            project_ancestry_collector_public_key=(
                original_a.ancestry_collector_public_key
            ),
            now_unix=now_unix,
        )
        replay = foundation_apply.load_validated_foundation_apply_chain(replay_a)
        if (
            original_a.authority != replay_a.authority
            or original_a.owner_reauthentication_receipt
            != replay_a.owner_reauthentication_receipt
            or original_a.plan != replay_a.plan
            or original_a.network_evidence != replay_a.network_evidence
            or original_a.ancestry_evidence != replay_a.ancestry_evidence
            or foundation_chain.apply_receipt != replay.apply_receipt
        ):
            _error("direct_iam_identity_author_foundation_chain_invalid")
    except DirectIamIdentityAuthorError:
        raise
    except (AttributeError, foundation_apply.OwnerGateFoundationApplyError) as exc:
        _error("direct_iam_identity_author_foundation_chain_invalid", exc)
    return _projection_from_validated_apply_chain(
        replay,
        now_unix=now_unix,
        require_fresh_owner=True,
    )


def _decode_canonical_authority_for_projection(
    raw: bytes,
    *,
    projection: _FoundationChainProjection,
    now_unix: int,
) -> CanonicalDirectIamAuthority:
    try:
        value = authority_schema.decode_canonical(
            raw,
            release_revision=projection.release_revision,
        )
    except authority_schema.DirectIamIdentityAuthorityError as exc:
        _error("direct_iam_identity_author_authority_invalid", exc)
    if (
        value["project_number"] != projection.project_number
        or value["owner_reauthentication_receipt_sha256"]
        != projection.owner_reauthentication_receipt_sha256
        or value["pre_foundation_authority_sha256"]
        != projection.pre_foundation_authority_sha256
        or value["foundation_apply_receipt_sha256"]
        != projection.foundation_apply_receipt_sha256
        or tuple(value["resource_ancestor_chain"])
        != projection.resource_ancestor_chain
        or value["owner_gate_vm_numeric_id"]
        != projection.owner_gate_vm_numeric_id
        or value["owner_gate_service_account_unique_id"]
        != projection.owner_gate_service_account_unique_id
        or type(value["collected_at_unix"]) is not int
        or value["collected_at_unix"]
        < projection.foundation_apply_completed_at_unix
        or value["collected_at_unix"] > now_unix
        or value["collected_at_unix"]
        > projection.owner_reauthentication_expires_at_unix
    ):
        _error("direct_iam_identity_author_authority_chain_mismatch")
    return CanonicalDirectIamAuthority._create(value=value, raw=raw)


def decode_canonical_authority_for_validated_chain(
    raw: bytes,
    *,
    foundation_chain: Any,
    now_unix: int,
) -> CanonicalDirectIamAuthority:
    """Bind one canonical direct-IAM asset to its revalidated signed chain."""

    projection = _revalidated_foundation_projection(
        foundation_chain,
        now_unix=now_unix,
    )
    return _decode_canonical_authority_for_projection(
        raw,
        projection=projection,
        now_unix=now_unix,
    )


def _decode_canonical_authority_for_recovery_chain(
    raw: bytes,
    *,
    foundation_chain: Any,
) -> CanonicalDirectIamAuthority:
    """Validate persisted direct-IAM evidence at its signed collection time."""

    projection = _projection_from_validated_apply_chain(
        foundation_chain,
        now_unix=int(foundation_chain.apply_receipt["completed_at_unix"]),
        require_fresh_owner=False,
    )
    try:
        value = authority_schema.decode_canonical(
            raw,
            release_revision=projection.release_revision,
        )
        collected_at = value["collected_at_unix"]
    except (KeyError, authority_schema.DirectIamIdentityAuthorityError) as exc:
        _error("direct_iam_identity_author_authority_invalid", exc)
    if type(collected_at) is not int or collected_at <= 0:
        _error("direct_iam_identity_author_authority_invalid")
    return _decode_canonical_authority_for_projection(
        raw,
        projection=projection,
        now_unix=collected_at,
    )


def _json_no_duplicates(raw: bytes) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if not isinstance(key, str) or key in value:
                _error("direct_iam_identity_author_cloud_json_invalid")
            value[key] = item
        return value

    def reject_constant(_: str) -> None:
        _error("direct_iam_identity_author_cloud_json_invalid")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except DirectIamIdentityAuthorError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("direct_iam_identity_author_cloud_json_invalid", exc)
    if not isinstance(value, Mapping):
        _error("direct_iam_identity_author_cloud_json_invalid")
    return dict(value)


def _bounded_http_body(response: Any) -> bytes:
    raw_length = response.getheader("Content-Length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            _error("direct_iam_identity_author_cloud_http_invalid", exc)
        if length < 1 or length > MAX_HTTP_BODY_BYTES:
            _error("direct_iam_identity_author_cloud_http_invalid")
    body = response.read(MAX_HTTP_BODY_BYTES + 1)
    if not isinstance(body, bytes) or not body or len(body) > MAX_HTTP_BODY_BYTES:
        _error("direct_iam_identity_author_cloud_http_invalid")
    return body


def _json_content_type(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = tuple(item.strip() for item in value.split(";"))
    if not parts or parts[0].lower() != "application/json":
        return False
    return len(parts) == 1 or (
        len(parts) == 2
        and parts[1].lower().replace(" ", "") == "charset=utf-8"
    )


def acquire_access_token(
    *,
    token_provider: Any,
) -> _GcloudAccessToken:
    """Copy one token from the already-approved owner-launcher capability."""

    # The import stays local so this small schema/author module does not load
    # the full launcher unless the live owner collection path is requested.
    from scripts.canary import full_canary_owner_launcher as owner_launcher

    if type(token_provider) is not owner_launcher.GcloudOwnerAccessToken:
        _error("direct_iam_identity_author_gcloud_provider_untrusted")
    return _acquire_access_token_with_provider(token_provider)


def _acquire_access_token_with_provider(
    token_provider: Any,
) -> _GcloudAccessToken:
    """Private deterministic seam; the live entry enforces the exact type."""

    token_text: Any = None
    try:
        if token_provider.approved_account != OWNER_ACCOUNT:
            _error("direct_iam_identity_author_owner_identity_invalid")
        token_text = token_provider()
        token_provider.require_stable()
        if token_provider.approved_account != OWNER_ACCOUNT:
            _error("direct_iam_identity_author_owner_identity_invalid")
    except DirectIamIdentityAuthorError:
        raise
    except Exception:
        # Never preserve the provider exception (which could contain command
        # output) as a cause of this credential-boundary failure.
        raise DirectIamIdentityAuthorError(
            "direct_iam_identity_author_gcloud_unavailable"
        ) from None
    try:
        raw = bytearray(
            token_text.encode("ascii", errors="strict")
            if isinstance(token_text, str)
            else b""
        )
    except UnicodeError:
        raw = bytearray()
    try:
        if (
            not raw
            or len(raw) > 16 * 1024 + 1
        ):
            _error("direct_iam_identity_author_gcloud_token_invalid")
        return _GcloudAccessToken(bytearray(raw), marker=_TOKEN_MARKER)
    finally:
        for index in range(len(raw)):
            raw[index] = 0


def wipe_access_token(token: _GcloudAccessToken) -> None:
    if type(token) is not _GcloudAccessToken:
        _error("direct_iam_identity_author_gcloud_token_invalid")
    for index in range(len(token._raw)):
        token._raw[index] = 0


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


def _reject_ambient_network_environment() -> None:
    if any(os.environ.get(name) for name in _FORBIDDEN_NETWORK_ENVIRONMENT):
        _error("direct_iam_identity_author_cloud_tls_invalid")
    try:
        from scripts.canary import full_canary_owner_launcher as owner_launcher

        owner_launcher._reject_custom_ca_environment()
    except DirectIamIdentityAuthorError:
        raise
    except Exception as exc:
        _error("direct_iam_identity_author_cloud_tls_invalid", exc)


def _default_connection(host: str) -> Any:
    if host not in {COMPUTE_HOST, RESOURCE_MANAGER_HOST, IAM_HOST}:
        _error("direct_iam_identity_author_cloud_resource_forbidden")
    _reject_ambient_network_environment()
    try:
        from scripts.canary import full_canary_owner_launcher as owner_launcher

        context = owner_launcher._pinned_system_tls_context()
    except DirectIamIdentityAuthorError:
        raise
    except Exception as exc:
        _error("direct_iam_identity_author_cloud_tls_invalid", exc)
    return http.client.HTTPSConnection(
        host,
        443,
        timeout=HTTP_TIMEOUT_SECONDS,
        context=context,
    )


class DirectIamOwnerFactsReader:
    """No-proxy reader with a closed set of REST methods and resources."""

    def __init__(
        self,
        *,
        token: _GcloudAccessToken,
        compute_connection_factory: Callable[[], Any] | None = None,
        resource_manager_connection_factory: Callable[[], Any] | None = None,
        iam_connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        if (
            type(token) is not _GcloudAccessToken
            or len(token._raw) < 20
            or any(byte < 0x21 or byte > 0x7E for byte in token._raw)
            or b":" in token._raw
        ):
            _error("direct_iam_identity_author_gcloud_token_invalid")
        self._token = token
        self._compute_factory = compute_connection_factory or (
            lambda: _default_connection(COMPUTE_HOST)
        )
        self._resource_manager_factory = resource_manager_connection_factory or (
            lambda: _default_connection(RESOURCE_MANAGER_HOST)
        )
        self._iam_factory = iam_connection_factory or (
            lambda: _default_connection(IAM_HOST)
        )

    def _request_json(
        self,
        *,
        connection_factory: Callable[[], Any],
        method: str,
        path: str,
        body: bytes | None = None,
    ) -> Mapping[str, Any]:
        if method not in {"GET", "POST"} or not path.startswith("/"):
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        if method == "GET" and body is not None:
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        try:
            bearer = self._token._raw.decode("ascii", errors="strict")
        except UnicodeError as exc:
            _error("direct_iam_identity_author_gcloud_token_invalid", exc)
        headers = {
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/json",
            "Connection": "close",
        }
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        connection = connection_factory()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = _bounded_http_body(response)
            if (
                response.status != 200
                or response.getheader("Location") is not None
                or not _json_content_type(response.getheader("Content-Type"))
            ):
                _error("direct_iam_identity_author_cloud_resource_unavailable")
        except DirectIamIdentityAuthorError:
            raise
        except (
            OSError,
            socket.timeout,
            ssl.SSLError,
            http.client.HTTPException,
        ) as exc:
            _error("direct_iam_identity_author_cloud_resource_unavailable", exc)
        finally:
            bearer = ""
            connection.close()
        return _json_no_duplicates(response_body)

    def _compute_instance(self) -> Mapping[str, Any]:
        return self._request_json(
            connection_factory=self._compute_factory,
            method="GET",
            path=OWNER_GATE_INSTANCE_PATH,
        )

    def _resource(self, resource: str) -> Mapping[str, Any]:
        if resource != PROJECT_RESOURCE and not (
            _FOLDER.fullmatch(resource) or _ORGANIZATION.fullmatch(resource)
        ):
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        return self._request_json(
            connection_factory=self._resource_manager_factory,
            method="GET",
            path=f"/v3/{resource}",
        )

    def _resource_policy(self, resource: str) -> Mapping[str, Any]:
        if resource != PROJECT_RESOURCE and not (
            _FOLDER.fullmatch(resource) or _ORGANIZATION.fullmatch(resource)
        ):
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        return self._request_json(
            connection_factory=self._resource_manager_factory,
            method="POST",
            path=f"/v3/{resource}:getIamPolicy",
            body=foundation.canonical_json_bytes({
                "options": {"requestedPolicyVersion": 3}
            }),
        )

    def _iam_get(self, path: str) -> Mapping[str, Any]:
        if path not in {
            OWNER_GATE_SERVICE_ACCOUNT_PATH,
            TARGET_SERVICE_ACCOUNT_PATH,
            OWNER_GATE_USER_KEYS_PATH,
        }:
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        return self._request_json(
            connection_factory=self._iam_factory,
            method="GET",
            path=path,
        )

    def _service_account_policy(self) -> Mapping[str, Any]:
        return self._request_json(
            connection_factory=self._iam_factory,
            method="POST",
            path=OWNER_GATE_SERVICE_ACCOUNT_POLICY_PATH,
            body=None,
        )

    def _role(self, role: str, *, ancestor_chain: Sequence[str]) -> Mapping[str, Any]:
        if not _role_allowed(role, ancestor_chain=ancestor_chain):
            _error("direct_iam_identity_author_cloud_resource_forbidden")
        return self._request_json(
            connection_factory=self._iam_factory,
            method="GET",
            path=f"/v1/{role}",
        )

    def _collect_once(self) -> Mapping[str, Any]:
        project = self._resource(PROJECT_RESOURCE)
        resources: list[Mapping[str, Any]] = [project]
        resource_names: list[str] = [PROJECT_RESOURCE]
        parent = project.get("parent")
        for _ in range(MAX_ANCESTOR_DEPTH):
            if not isinstance(parent, str) or not (
                _FOLDER.fullmatch(parent) or _ORGANIZATION.fullmatch(parent)
            ):
                _error("direct_iam_identity_author_hierarchy_invalid")
            if parent in resource_names:
                _error("direct_iam_identity_author_hierarchy_invalid")
            resource = self._resource(parent)
            resource_names.append(parent)
            resources.append(resource)
            if _ORGANIZATION.fullmatch(parent):
                if resource.get("parent") not in {None, ""}:
                    _error("direct_iam_identity_author_hierarchy_invalid")
                break
            parent = resource.get("parent")
        else:
            _error("direct_iam_identity_author_hierarchy_invalid")
        ancestor_chain = tuple(resource_names[1:])
        policies = [self._resource_policy(name) for name in resource_names]
        policy_roles = _policy_role_names(
            policies,
            ancestor_chain=ancestor_chain,
        )
        exact_roles = (
            PROJECT_READ_ROLE,
            _ancestor_read_role(ancestor_chain),
            MUTATION_ROLE,
        )
        roles = {
            role: self._role(role, ancestor_chain=ancestor_chain)
            for role in sorted(set((*policy_roles, *exact_roles)))
        }
        return {
            "instance": self._compute_instance(),
            "resources": resources,
            "resource_names": resource_names,
            "policies": policies,
            "roles": roles,
            "owner_gate_service_account": self._iam_get(
                OWNER_GATE_SERVICE_ACCOUNT_PATH
            ),
            "target_service_account": self._iam_get(TARGET_SERVICE_ACCOUNT_PATH),
            "owner_gate_service_account_policy": self._service_account_policy(),
            "owner_gate_user_managed_keys": self._iam_get(
                OWNER_GATE_USER_KEYS_PATH
            ),
        }

    def collect(self) -> Mapping[str, Any]:
        """Require two complete same-token snapshots to canonical-match."""

        first = self._collect_once()
        second = self._collect_once()
        try:
            stable = (
                foundation.canonical_json_bytes(first)
                == foundation.canonical_json_bytes(second)
            )
        except foundation.OwnerGateFoundationError as exc:
            _error("direct_iam_identity_author_facts_unstable", exc)
        if not stable:
            _error("direct_iam_identity_author_facts_unstable")
        return second


def _strings(value: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _error("direct_iam_identity_author_cloud_fact_invalid")
    normalized = tuple(sorted(set(value)))
    if len(normalized) != len(value) or (not normalized and not allow_empty):
        _error("direct_iam_identity_author_cloud_fact_invalid")
    return normalized


def _condition(value: Any) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not {"title", "expression"} <= set(value) or (
        set(value) - {"title", "description", "expression"}
    ):
        _error("direct_iam_identity_author_policy_invalid")
    normalized = {
        "title": value.get("title"),
        "description": value.get("description", ""),
        "expression": value.get("expression"),
    }
    if any(
        not isinstance(item, str)
        or (key != "description" and not item)
        or len(item) > 4096
        or any(ord(character) < 0x20 for character in item)
        for key, item in normalized.items()
    ):
        _error("direct_iam_identity_author_policy_invalid")
    return normalized


def _role_allowed(role: Any, *, ancestor_chain: Sequence[str]) -> bool:
    if not isinstance(role, str):
        return False
    if _BUILTIN_ROLE.fullmatch(role) or _PROJECT_ROLE.fullmatch(role):
        return True
    match = _ORGANIZATION_ROLE.fullmatch(role)
    return bool(
        match
        and ancestor_chain
        and ancestor_chain[-1] == f"organizations/{match.group(1)}"
    )


def _policy_role_names(
    policies: Sequence[Mapping[str, Any]],
    *,
    ancestor_chain: Sequence[str],
) -> tuple[str, ...]:
    names: set[str] = set()
    for policy in policies:
        bindings = policy.get("bindings") if isinstance(policy, Mapping) else None
        if not isinstance(bindings, list):
            _error("direct_iam_identity_author_policy_invalid")
        for binding in bindings:
            role = binding.get("role") if isinstance(binding, Mapping) else None
            if not _role_allowed(role, ancestor_chain=ancestor_chain):
                _error("direct_iam_identity_author_policy_invalid")
            names.add(str(role))
            if len(names) > MAX_POLICY_ROLE_DEFINITIONS:
                _error("direct_iam_identity_author_policy_invalid")
    return tuple(sorted(names))


def _ancestor_read_role(ancestor_chain: Sequence[str]) -> str:
    if not ancestor_chain or _ORGANIZATION.fullmatch(str(ancestor_chain[-1])) is None:
        _error("direct_iam_identity_author_hierarchy_invalid")
    organization = str(ancestor_chain[-1]).split("/", 1)[1]
    return (
        f"organizations/{organization}/roles/"
        f"{foundation.ANCESTOR_READ_ONLY_IAM_ROLE_ID}"
    )


def _normalize_policy(
    policy: Mapping[str, Any],
    *,
    resource: str,
    ancestor_chain: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(policy, Mapping) or set(policy) - {
        "version", "bindings", "etag", "auditConfigs"
    }:
        _error("direct_iam_identity_author_policy_invalid")
    version = policy.get("version", 1)
    etag = policy.get("etag")
    bindings = policy.get("bindings")
    audit_configs = policy.get("auditConfigs", [])
    if (
        type(version) is not int
        or version not in {1, 3}
        or not isinstance(etag, str)
        or not etag
        or len(etag) > 4096
        or not isinstance(bindings, list)
        or not isinstance(audit_configs, list)
    ):
        _error("direct_iam_identity_author_policy_invalid")
    normalized_bindings: list[Mapping[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) - {
            "role", "members", "condition"
        } or not _role_allowed(binding.get("role"), ancestor_chain=ancestor_chain):
            _error("direct_iam_identity_author_policy_invalid")
        members = _strings(binding.get("members"))
        if any(
            not member
            or len(member) > 1024
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in member
            )
            for member in members
        ):
            _error("direct_iam_identity_author_policy_invalid")
        normalized_bindings.append({
            "resource": resource,
            "role": binding["role"],
            "members": list(members),
            "condition": _condition(binding.get("condition")),
        })
    normalized_bindings.sort(key=foundation.canonical_json_bytes)
    canonical = [foundation.canonical_json_bytes(item) for item in normalized_bindings]
    if canonical != sorted(set(canonical)):
        _error("direct_iam_identity_author_policy_invalid")
    return {
        "resource": resource,
        "version": version,
        "etag": etag,
        "audit_configs": json.loads(
            foundation.canonical_json_bytes(audit_configs).decode("ascii")
        ),
        "bindings": normalized_bindings,
    }


def _normalize_role(
    value: Mapping[str, Any],
    *,
    expected_role: str,
    ancestor_chain: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not _role_allowed(
        expected_role, ancestor_chain=ancestor_chain
    ):
        _error("direct_iam_identity_author_role_invalid")
    permissions = _strings(value.get("includedPermissions"))
    title = value.get("title")
    description = value.get("description", "")
    deleted = value.get("deleted", False)
    etag = value.get("etag")
    if (
        value.get("name") != expected_role
        or not isinstance(title, str)
        or not title
        or len(title) > 4096
        or not isinstance(description, str)
        or len(description) > 16 * 1024
        or value.get("stage") not in _ROLE_STAGES
        or deleted is not False
        or (
            etag is not None
            and (
                not isinstance(etag, str)
                or not etag
                or len(etag) > 4096
            )
        )
    ):
        _error("direct_iam_identity_author_role_invalid")
    return {
        "name": expected_role,
        "title": title,
        "description": description,
        "included_permissions": list(permissions),
        "stage": value["stage"],
        "deleted": False,
        "etag": etag,
    }


def _validate_service_account(
    value: Any,
    *,
    email: str,
) -> str:
    name = f"projects/{foundation.PROJECT}/serviceAccounts/{email}"
    if (
        not isinstance(value, Mapping)
        or value.get("name") != name
        or value.get("projectId") != foundation.PROJECT
        or value.get("email") != email
        or _NUMERIC_ID.fullmatch(str(value.get("uniqueId", ""))) is None
        or value.get("disabled", False) is not False
    ):
        _error("direct_iam_identity_author_service_account_invalid")
    return str(value["uniqueId"])


def _validate_instance(value: Any) -> str:
    accounts = value.get("serviceAccounts") if isinstance(value, Mapping) else None
    scopes = (
        _strings(accounts[0].get("scopes"))
        if isinstance(accounts, list)
        and len(accounts) == 1
        and isinstance(accounts[0], Mapping)
        else ()
    )
    if (
        not isinstance(value, Mapping)
        or value.get("name") != foundation.VM_NAME
        or _NUMERIC_ID.fullmatch(str(value.get("id", ""))) is None
        or value.get("status") != "RUNNING"
        or not isinstance(value.get("zone"), str)
        or not value["zone"].endswith(f"/zones/{foundation.ZONE}")
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or accounts[0].get("email")
        != authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL
        or scopes != tuple(sorted(foundation.OWNER_GATE_OAUTH_SCOPES))
    ):
        _error("direct_iam_identity_author_instance_invalid")
    return str(value["id"])


def _validate_local_service_account_authority(
    policy: Any,
    keys: Any,
) -> None:
    if not isinstance(policy, Mapping) or set(policy) - {
        "version", "bindings", "etag", "auditConfigs"
    }:
        _error("direct_iam_identity_author_impersonation_invalid")
    if (
        type(policy.get("version", 1)) is not int
        or policy.get("version", 1) not in {1, 3}
        or policy.get("bindings", []) != []
        or policy.get("auditConfigs", []) != []
        or not isinstance(policy.get("etag", ""), str)
    ):
        _error("direct_iam_identity_author_impersonation_invalid")
    if (
        not isinstance(keys, Mapping)
        or set(keys) - {"keys"}
        or keys.get("keys", []) != []
    ):
        _error("direct_iam_identity_author_user_keys_invalid")


def _validate_chain_digests(
    *,
    pre_foundation_authority_sha256: str,
    foundation_apply_receipt_sha256: str,
) -> tuple[str, str]:
    if (
        _SHA256.fullmatch(pre_foundation_authority_sha256 or "") is None
        or _SHA256.fullmatch(foundation_apply_receipt_sha256 or "") is None
    ):
        _error("direct_iam_identity_author_foundation_chain_invalid")
    return pre_foundation_authority_sha256, foundation_apply_receipt_sha256


def _build_authority_bytes(
    *,
    live_facts: Mapping[str, Any],
    release_revision: str,
    owner_reauthentication_receipt_sha256: str,
    pre_foundation_authority_sha256: str,
    foundation_apply_receipt_sha256: str,
    collected_at_unix: int,
) -> bytes:
    """Normalize live facts into the strict canonical signed-package asset."""

    if (
        _REVISION.fullmatch(release_revision or "") is None
        or _SHA256.fullmatch(owner_reauthentication_receipt_sha256 or "") is None
        or type(collected_at_unix) is not int
        or collected_at_unix <= 0
        or not isinstance(live_facts, Mapping)
    ):
        _error("direct_iam_identity_author_input_invalid")
    pre_sha, apply_sha = _validate_chain_digests(
        pre_foundation_authority_sha256=pre_foundation_authority_sha256,
        foundation_apply_receipt_sha256=foundation_apply_receipt_sha256,
    )
    resources = live_facts.get("resources")
    resource_names = live_facts.get("resource_names")
    policies = live_facts.get("policies")
    roles = live_facts.get("roles")
    if (
        not isinstance(resources, list)
        or not isinstance(resource_names, list)
        or not isinstance(policies, list)
        or not isinstance(roles, Mapping)
        or len(resources) != len(resource_names)
        or len(policies) != len(resource_names)
        or len(resource_names) < 2
        or resource_names[0] != PROJECT_RESOURCE
    ):
        _error("direct_iam_identity_author_hierarchy_invalid")
    ancestor_chain: list[Any] = list(resource_names[1:])
    if (
        not ancestor_chain
        or _ORGANIZATION.fullmatch(str(ancestor_chain[-1])) is None
        or any(_FOLDER.fullmatch(str(item)) is None for item in ancestor_chain[:-1])
        or len(ancestor_chain) > MAX_ANCESTOR_DEPTH
    ):
        _error("direct_iam_identity_author_hierarchy_invalid")
    for index, (name, resource) in enumerate(zip(resource_names, resources, strict=True)):
        expected_parent = resource_names[index + 1] if index + 1 < len(resource_names) else None
        if (
            not isinstance(resource, Mapping)
            or resource.get("name") != name
            or resource.get("state") != "ACTIVE"
            or not isinstance(resource.get("etag"), str)
            or not resource["etag"]
            or resource.get("parent") not in ({None, ""} if expected_parent is None else {expected_parent})
            or (index == 0 and resource.get("projectId") != foundation.PROJECT)
        ):
            _error("direct_iam_identity_author_hierarchy_invalid")
    normalized_policies = [
        _normalize_policy(
            policy,
            resource=(
                PROJECT_POLICY_RESOURCE if name == PROJECT_RESOURCE else str(name)
            ),
            ancestor_chain=ancestor_chain,
        )
        for name, policy in zip(resource_names, policies, strict=True)
    ]
    policy_role_names = _policy_role_names(
        policies,
        ancestor_chain=ancestor_chain,
    )
    ancestor_role = _ancestor_read_role(ancestor_chain)
    required_role_names = set((*policy_role_names, PROJECT_READ_ROLE, ancestor_role, MUTATION_ROLE))
    if set(roles) != required_role_names or len(roles) > MAX_POLICY_ROLE_DEFINITIONS:
        _error("direct_iam_identity_author_role_invalid")
    normalized_roles = {
        role: _normalize_role(
            roles[role],
            expected_role=role,
            ancestor_chain=ancestor_chain,
        )
        for role in sorted(roles)
    }
    exact_role_contracts = {
        PROJECT_READ_ROLE: (
            foundation.PROJECT_READ_ROLE_TITLE,
            foundation.PROJECT_READ_ROLE_DESCRIPTION,
            tuple(sorted(foundation.READ_ONLY_IAM_PERMISSIONS)),
        ),
        ancestor_role: (
            foundation.ANCESTOR_READ_ROLE_TITLE,
            foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
            tuple(sorted(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS)),
        ),
        MUTATION_ROLE: (
            foundation.MUTATION_ROLE_TITLE,
            foundation.MUTATION_ROLE_DESCRIPTION,
            tuple(sorted(foundation.MUTATION_PERMISSIONS)),
        ),
    }
    for role, (title, description, permissions) in exact_role_contracts.items():
        definition = normalized_roles[role]
        if (
            definition["title"] != title
            or definition["description"] != description
            or tuple(definition["included_permissions"]) != permissions
            or definition["stage"] != "GA"
        ):
            _error("direct_iam_identity_author_role_drift")
    member = f"serviceAccount:{authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL}"
    project_binding_count = 0
    ancestor_binding_count = 0
    residual_bindings: list[Mapping[str, Any]] = []
    exact_mutation_binding = {
        "resource": PROJECT_POLICY_RESOURCE,
        "role": MUTATION_ROLE,
        "members": [member],
        "condition": dict(MUTATION_CONDITION),
    }
    for policy in normalized_policies:
        for binding in policy["bindings"]:
            if binding == {
                "resource": PROJECT_POLICY_RESOURCE,
                "role": PROJECT_READ_ROLE,
                "members": [member],
                "condition": None,
            }:
                project_binding_count += 1
            elif binding == {
                "resource": ancestor_chain[-1],
                "role": ancestor_role,
                "members": [member],
                "condition": None,
            }:
                ancestor_binding_count += 1
            elif binding == exact_mutation_binding:
                _error("direct_iam_identity_author_binding_drift")
            elif member in binding["members"]:
                _error("direct_iam_identity_author_binding_drift")
            else:
                residual_bindings.append(binding)
    if project_binding_count != 1 or ancestor_binding_count != 1:
        _error("direct_iam_identity_author_binding_drift")
    residual_bindings.sort(key=foundation.canonical_json_bytes)
    residual_role_names = sorted({binding["role"] for binding in residual_bindings})
    residual_definitions = [normalized_roles[role] for role in residual_role_names]
    policy_generations = [
        {
            "resource": policy["resource"],
            "version": policy["version"],
            "etag": policy["etag"],
            "audit_configs": policy["audit_configs"],
        }
        for policy in normalized_policies
    ]
    _validate_local_service_account_authority(
        live_facts.get("owner_gate_service_account_policy"),
        live_facts.get("owner_gate_user_managed_keys"),
    )
    owner_gate_unique_id = _validate_service_account(
        live_facts.get("owner_gate_service_account"),
        email=authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL,
    )
    target_unique_id = _validate_service_account(
        live_facts.get("target_service_account"),
        email=authority_schema.TARGET_SERVICE_ACCOUNT_EMAIL,
    )
    instance_id = _validate_instance(live_facts.get("instance"))
    unsigned = {
        "schema": authority_schema.SCHEMA,
        "release_revision": release_revision,
        "project_id": foundation.PROJECT,
        "project_number": authority_schema.PROJECT_NUMBER,
        "owner_gate_vm_name": foundation.VM_NAME,
        "owner_gate_vm_numeric_id": instance_id,
        "owner_gate_service_account_email": authority_schema.OWNER_GATE_SERVICE_ACCOUNT_EMAIL,
        "owner_gate_service_account_unique_id": owner_gate_unique_id,
        "target_service_account_email": authority_schema.TARGET_SERVICE_ACCOUNT_EMAIL,
        "target_service_account_unique_id": target_unique_id,
        "resource_ancestor_chain": list(ancestor_chain),
        "project_read_role": PROJECT_READ_ROLE,
        "project_read_role_title": foundation.PROJECT_READ_ROLE_TITLE,
        "project_read_role_description": foundation.PROJECT_READ_ROLE_DESCRIPTION,
        "project_read_role_etag": normalized_roles[PROJECT_READ_ROLE]["etag"],
        "project_read_permissions": list(foundation.READ_ONLY_IAM_PERMISSIONS),
        "project_read_binding_member": member,
        "project_read_binding_present": True,
        "ancestor_read_role": ancestor_role,
        "ancestor_read_role_title": foundation.ANCESTOR_READ_ROLE_TITLE,
        "ancestor_read_role_description": foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
        "ancestor_read_role_etag": normalized_roles[ancestor_role]["etag"],
        "ancestor_read_permissions": list(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS),
        "ancestor_binding_member": member,
        "ancestor_binding_present": True,
        "mutation_role": MUTATION_ROLE,
        "mutation_role_title": foundation.MUTATION_ROLE_TITLE,
        "mutation_role_description": foundation.MUTATION_ROLE_DESCRIPTION,
        "mutation_role_etag": normalized_roles[MUTATION_ROLE]["etag"],
        "mutation_permissions": list(foundation.MUTATION_PERMISSIONS),
        "mutation_condition": dict(MUTATION_CONDITION),
        "mutation_binding_member": member,
        "mutation_binding_present": False,
        "mutation_activation_seal": str(foundation.MUTATION_ENABLE_SEAL),
        "mutation_activation_seal_present": False,
        "allowed_owner_gate_impersonators": [],
        "owner_gate_user_managed_key_inventory": {
            "requested_key_types": ["USER_MANAGED"],
            "allowed_key_names": [],
        },
        "external_gcp_admin_trust_root": {
            "inventory_complete": True,
            "structural_partition_complete": True,
            "passkey_protects_against_external_gcp_admins": False,
            "passkey_protects_against_pinned_external_roots": False,
            "google_provider_control_plane_outside_passkey": True,
            "collected_under_owner_reauthentication_receipt_sha256": (
                owner_reauthentication_receipt_sha256
            ),
            "resource_policy_generations": policy_generations,
            "allowed_residual_bindings": residual_bindings,
            "allowed_residual_role_definitions": residual_definitions,
        },
        "metadata_oauth_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
        "private_google_api_hosts": list(foundation.PRIVATE_GOOGLE_API_HOSTS),
        "private_google_api_vip_range": foundation.PRIVATE_GOOGLE_API_VIP_RANGE,
        "owner_reauthentication_receipt_sha256": (
            owner_reauthentication_receipt_sha256
        ),
        "pre_foundation_authority_sha256": pre_sha,
        "foundation_apply_receipt_sha256": apply_sha,
        "collected_at_unix": collected_at_unix,
    }
    value = {**unsigned, "authority_sha256": foundation.sha256_json(unsigned)}
    try:
        checked = authority_schema.validate(value, release_revision=release_revision)
        raw = foundation.canonical_json_bytes(checked)
        decoded = authority_schema.decode_canonical(
            raw,
            release_revision=release_revision,
        )
    except (
        authority_schema.DirectIamIdentityAuthorityError,
        foundation.OwnerGateFoundationError,
    ) as exc:
        _error("direct_iam_identity_author_output_invalid", exc)
    if decoded != checked:
        _error("direct_iam_identity_author_output_invalid")
    return raw


def _collect_and_build_authority_with_provider(
    *,
    release_revision: str,
    owner_reauthentication_receipt_sha256: str,
    pre_foundation_authority_sha256: str,
    foundation_apply_receipt_sha256: str,
    token_provider: Any,
    collected_at_unix: int | None = None,
) -> bytes:
    token = acquire_access_token(token_provider=token_provider)
    try:
        facts = DirectIamOwnerFactsReader(token=token).collect()
        return _build_authority_bytes(
            live_facts=facts,
            release_revision=release_revision,
            owner_reauthentication_receipt_sha256=(
                owner_reauthentication_receipt_sha256
            ),
            pre_foundation_authority_sha256=pre_foundation_authority_sha256,
            foundation_apply_receipt_sha256=foundation_apply_receipt_sha256,
            collected_at_unix=(
                int(time.time()) if collected_at_unix is None else collected_at_unix
            ),
        )
    finally:
        wipe_access_token(token)


def _collect_canonical_authority_with_capabilities(
    *,
    foundation_chain: Any,
    runtime: Any,
    configuration: Any,
    owner_identity: Any,
    facts_reader_factory: Callable[..., DirectIamOwnerFactsReader],
    clock: Callable[[], float],
) -> CanonicalDirectIamAuthority:
    """Private seam after the live boundary constructed every capability."""

    from scripts.canary import full_canary_owner_launcher as owner_launcher

    if (
        type(runtime) is not owner_launcher.TrustedGcloudExecutable
        or type(configuration) is not owner_launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not owner_launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not configuration
        or getattr(owner_identity, "_gcloud_executable", None) is not runtime
        or not callable(facts_reader_factory)
        or not callable(clock)
    ):
        _error("direct_iam_identity_author_capability_invalid")
    before_unix = int(clock())
    projection = _revalidated_foundation_projection(
        foundation_chain,
        now_unix=before_unix,
    )
    _reject_ambient_network_environment()
    try:
        runtime_before = runtime.sealed_runtime_identity(
            expected_release_sha=projection.release_revision,
        )
        configuration.assert_stable()
        if configuration.account != OWNER_ACCOUNT:
            _error("direct_iam_identity_author_owner_identity_invalid")
    except DirectIamIdentityAuthorError:
        raise
    except owner_launcher.OwnerLauncherError as exc:
        _error("direct_iam_identity_author_capability_invalid", exc)
    if (
        not isinstance(runtime_before, Mapping)
        or _SHA256.fullmatch(
            str(runtime_before.get("identity_sha256", ""))
        )
        is None
    ):
        _error("direct_iam_identity_author_capability_invalid")

    token: _GcloudAccessToken | None = None
    owner_bound = False
    try:
        owner_identity.bind_approved_subject(
            hashlib.sha256(OWNER_ACCOUNT.encode("ascii")).hexdigest()
        )
        owner_bound = True
        owner_identity.require_stable()
        if owner_identity.approved_account != OWNER_ACCOUNT:
            _error("direct_iam_identity_author_owner_identity_invalid")
        token = acquire_access_token(token_provider=owner_identity)
        reader = facts_reader_factory(token=token)
        if not isinstance(reader, DirectIamOwnerFactsReader):
            _error("direct_iam_identity_author_capability_invalid")
        facts = reader.collect()
        collected_at_unix = int(clock())
        if (
            collected_at_unix < before_unix
            or collected_at_unix
            > projection.owner_reauthentication_expires_at_unix
        ):
            _error("direct_iam_identity_author_owner_reauth_expired")
        raw = _build_authority_bytes(
            live_facts=facts,
            release_revision=projection.release_revision,
            owner_reauthentication_receipt_sha256=(
                projection.owner_reauthentication_receipt_sha256
            ),
            pre_foundation_authority_sha256=(
                projection.pre_foundation_authority_sha256
            ),
            foundation_apply_receipt_sha256=(
                projection.foundation_apply_receipt_sha256
            ),
            collected_at_unix=collected_at_unix,
        )
        return decode_canonical_authority_for_validated_chain(
            raw,
            foundation_chain=foundation_chain,
            now_unix=collected_at_unix,
        )
    finally:
        if token is not None:
            wipe_access_token(token)
        stability_errors: list[BaseException] = []
        runtime_after: Any = None
        account_after: Any = None
        checks: list[Callable[[], Any]] = [
            _reject_ambient_network_environment,
            configuration.assert_stable,
        ]
        if owner_bound:
            checks.append(owner_identity.require_stable)
        for check in checks:
            try:
                check()
            except BaseException as exc:
                stability_errors.append(exc)
        try:
            runtime_after = runtime.sealed_runtime_identity(
                expected_release_sha=projection.release_revision,
            )
        except BaseException as exc:
            stability_errors.append(exc)
        try:
            account_after = configuration.account
        except BaseException as exc:
            stability_errors.append(exc)
        if (
            stability_errors
            or runtime_after != runtime_before
            or account_after != OWNER_ACCOUNT
            or (token is not None and any(token._raw))
        ):
            _error("direct_iam_identity_author_capability_changed")


def _validated_foundation_apply_chain_from_raw(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> tuple[Any, _FoundationChainProjection]:
    """Validate canonical A and load its one fixed-journal apply success."""

    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    if (
        any(
            type(value) is not bytes or not value
            for value in (
                pre_foundation_authority_raw,
                owner_reauthentication_receipt_raw,
                network_evidence_raw,
                project_ancestry_evidence_raw,
            )
        )
        or not isinstance(release_public_key, Ed25519PublicKey)
        or not isinstance(network_collector_public_key, Ed25519PublicKey)
        or not isinstance(
            project_ancestry_collector_public_key,
            Ed25519PublicKey,
        )
    ):
        _error("direct_iam_identity_author_foundation_chain_invalid")
    now_unix = int(time.time())
    try:
        foundation_a = foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=network_evidence_raw,
            project_ancestry_evidence_raw=project_ancestry_evidence_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_collector_public_key,
            project_ancestry_collector_public_key=(
                project_ancestry_collector_public_key
            ),
            now_unix=now_unix,
        )
        apply_chain = foundation_apply.load_validated_foundation_apply_chain(
            foundation_a
        )
        projection = _revalidated_foundation_projection(
            apply_chain,
            now_unix=now_unix,
        )
    except foundation_apply.OwnerGateFoundationApplyError as exc:
        _error("direct_iam_identity_author_foundation_chain_invalid", exc)
    return apply_chain, projection


def _direct_publication_chain(
    projection: _FoundationChainProjection,
) -> Mapping[str, Any]:
    projection.validate()
    return {
        "foundation_source_revision": projection.release_revision,
        "owner_reauthentication_receipt_sha256": (
            projection.owner_reauthentication_receipt_sha256
        ),
        "pre_foundation_authority_sha256": (
            projection.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": (
            projection.foundation_apply_receipt_sha256
        ),
        "project_number": projection.project_number,
        "resource_ancestor_chain": list(projection.resource_ancestor_chain),
        "owner_gate_vm_numeric_id": projection.owner_gate_vm_numeric_id,
        "owner_gate_service_account_unique_id": (
            projection.owner_gate_service_account_unique_id
        ),
    }


def _recovery_foundation_apply_chain_from_raw(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> tuple[Any, _FoundationChainProjection]:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply

    try:
        apply_chain = (
            foundation_apply._load_validated_foundation_apply_chain_for_source_recovery(
                pre_foundation_authority_raw=pre_foundation_authority_raw,
                owner_reauthentication_receipt_raw=(
                    owner_reauthentication_receipt_raw
                ),
                network_evidence_raw=network_evidence_raw,
                project_ancestry_evidence_raw=project_ancestry_evidence_raw,
                release_public_key=release_public_key,
                network_collector_public_key=network_collector_public_key,
                project_ancestry_collector_public_key=(
                    project_ancestry_collector_public_key
                ),
            )
        )
        projection = _projection_from_validated_apply_chain(
            apply_chain,
            now_unix=int(apply_chain.apply_receipt["completed_at_unix"]),
            require_fresh_owner=False,
        )
    except foundation_apply.OwnerGateFoundationApplyError as exc:
        _error("direct_iam_identity_author_foundation_chain_invalid", exc)
    return apply_chain, projection


def collect_and_publish_canonical_authority(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    from scripts.canary import full_canary_owner_launcher as owner_launcher

    recovery_only = False
    try:
        apply_chain, projection = _validated_foundation_apply_chain_from_raw(
            pre_foundation_authority_raw=pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=network_evidence_raw,
            project_ancestry_evidence_raw=project_ancestry_evidence_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_collector_public_key,
            project_ancestry_collector_public_key=(
                project_ancestry_collector_public_key
            ),
        )
    except DirectIamIdentityAuthorError as fresh_error:
        try:
            apply_chain, projection = (
                _recovery_foundation_apply_chain_from_raw(
                    pre_foundation_authority_raw=pre_foundation_authority_raw,
                    owner_reauthentication_receipt_raw=(
                        owner_reauthentication_receipt_raw
                    ),
                    network_evidence_raw=network_evidence_raw,
                    project_ancestry_evidence_raw=(
                        project_ancestry_evidence_raw
                    ),
                    release_public_key=release_public_key,
                    network_collector_public_key=(
                        network_collector_public_key
                    ),
                    project_ancestry_collector_public_key=(
                        project_ancestry_collector_public_key
                    ),
                )
            )
        except DirectIamIdentityAuthorError:
            raise fresh_error
        recovery_only = True
    owner_home = Path(owner_launcher._canonical_owner_home())

    def validate(raw: bytes) -> source_publication._ValidatedArtifact:
        canonical = (
            _decode_canonical_authority_for_recovery_chain(
                raw,
                foundation_chain=apply_chain,
            )
            if recovery_only
            else decode_canonical_authority_for_validated_chain(
                raw,
                foundation_chain=apply_chain,
                now_unix=int(time.time()),
            )
        )
        return source_publication._ValidatedArtifact(
            value=canonical,
            logical_sha256=str(canonical.value["authority_sha256"]),
        )

    def collect() -> bytes:
        try:
            runtime = owner_launcher.TrustedGcloudExecutable(
                release_sha=projection.release_revision,
            )
            configuration = owner_launcher.PinnedGcloudConfiguration()
            owner_identity = owner_launcher.GcloudOwnerAccessToken(
                gcloud_executable=runtime,
                gcloud_configuration=configuration,
            )
        except owner_launcher.OwnerLauncherError as exc:
            _error("direct_iam_identity_author_capability_invalid", exc)
        authority = _collect_canonical_authority_with_capabilities(
            foundation_chain=apply_chain,
            runtime=runtime,
            configuration=configuration,
            owner_identity=owner_identity,
            facts_reader_factory=DirectIamOwnerFactsReader,
            clock=time.time,
        )
        return authority.raw

    try:
        result = source_publication._run_direct_iam(
            owner_home=owner_home,
            chain=_direct_publication_chain(projection),
            maximum=authority_schema.MAX_BYTES,
            validator=validate,
            collector=collect,
            _recovery_only=recovery_only,
        )
    except source_publication._SourceArtifactPublicationError as exc:
        _error("direct_iam_identity_author_publication_failed", exc)
    authority = result.value
    if (
        type(authority) is not CanonicalDirectIamAuthority
        or getattr(authority, "_marker", None) is not _AUTHORITY_MARKER
    ):
        _error("direct_iam_identity_author_publication_failed")
    publication = {
        "path": result.path,
        "authority_sha256": result.logical_sha256,
        "authority_file_sha256": result.file_sha256,
    }
    return {
        "authority": authority,
        "publication": publication,
    }


def _read_owner_input(path: Path, *, maximum: int) -> bytes:
    from scripts.canary import owner_gate_trust as release_trust

    if (
        type(path) is not type(Path())
        or not path.is_absolute()
        or ".." in path.parts
        or os.path.realpath(path) != str(path)
        or type(maximum) is not int
        or not 0 < maximum <= MAX_OWNER_INPUT_BYTES
    ):
        _error("direct_iam_identity_author_owner_input_invalid")
    try:
        return release_trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except release_trust.OwnerGateTrustError as exc:
        _error("direct_iam_identity_author_owner_input_invalid", exc)


def _load_owner_collector_public_key(path: Path) -> Ed25519PublicKey:
    raw = _read_owner_input(path, maximum=MAX_OWNER_PUBLIC_KEY_BYTES)
    try:
        if len(raw) == 32:
            key = Ed25519PublicKey.from_public_bytes(raw)
        else:
            key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        _error("direct_iam_identity_author_owner_public_key_invalid", exc)
    if not isinstance(key, Ed25519PublicKey):
        _error("direct_iam_identity_author_owner_public_key_invalid")
    return key


def main(argv: Sequence[str] | None = None) -> int:
    """Fixed owner-only live author; no caller can select output or chain pins."""

    from scripts.canary import full_canary_owner_launcher as owner_launcher
    from scripts.canary import owner_gate_pre_foundation as pre_foundation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-foundation-authority", type=Path, required=True)
    parser.add_argument("--owner-reauth-receipt", type=Path, required=True)
    parser.add_argument("--release-trust-public-key", type=Path, required=True)
    parser.add_argument("--network-collector-public-key", type=Path, required=True)
    parser.add_argument("--network-evidence", type=Path, required=True)
    parser.add_argument("--project-ancestry-evidence", type=Path, required=True)
    parser.add_argument(
        "--project-ancestry-collector-public-key",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args(argv)
    try:
        release_public_key = pre_foundation.load_pinned_public_key(
            arguments.release_trust_public_key,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("direct_iam_identity_author_owner_public_key_invalid", exc)
    network_public_key = _load_owner_collector_public_key(
        arguments.network_collector_public_key
    )
    ancestry_public_key = _load_owner_collector_public_key(
        arguments.project_ancestry_collector_public_key
    )
    result = collect_and_publish_canonical_authority(
        pre_foundation_authority_raw=_read_owner_input(
            arguments.pre_foundation_authority,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        owner_reauthentication_receipt_raw=_read_owner_input(
            arguments.owner_reauth_receipt,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        network_evidence_raw=_read_owner_input(
            arguments.network_evidence,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        project_ancestry_evidence_raw=_read_owner_input(
            arguments.project_ancestry_evidence,
            maximum=MAX_OWNER_INPUT_BYTES,
        ),
        release_public_key=release_public_key,
        network_collector_public_key=network_public_key,
        project_ancestry_collector_public_key=ancestry_public_key,
    )
    publication = result.get("publication") if isinstance(result, Mapping) else None
    expected_output = str(
        Path(owner_launcher._canonical_owner_home())
        / DIRECT_IAM_AUTHORITY_RELATIVE
    )
    if (
        not isinstance(publication, Mapping)
        or set(publication)
        != {"path", "authority_sha256", "authority_file_sha256"}
        or publication.get("path") != expected_output
        or _SHA256.fullmatch(str(publication.get("authority_sha256", "")))
        is None
        or _SHA256.fullmatch(
            str(publication.get("authority_file_sha256", ""))
        )
        is None
    ):
        _error("direct_iam_identity_author_publication_invalid")
    summary = {
        "schema": "muncho-owner-gate-direct-iam-authority-publication.v1",
        "authority_published": True,
        **dict(publication),
    }
    sys.stdout.write(foundation.canonical_json_bytes(summary).decode("utf-8") + "\n")
    return 0


__all__ = [
    "CanonicalDirectIamAuthority",
    "DirectIamIdentityAuthorError",
    "collect_and_publish_canonical_authority",
    "decode_canonical_authority_for_validated_chain",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
