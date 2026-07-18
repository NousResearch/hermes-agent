#!/usr/bin/env python3
"""Bounded executor for the signed owner-gate foundation plan.

The public boundary accepts only canonical signed foundation-A artifacts and
the exact sealed owner gcloud runtime/configuration classes.  The executor can
run only the nine signed foundation operations.  Every resource is observed
twice before use; exact pre-existing state is reused, absence may be created,
and drift or unknown state fails closed.  A failure rolls back only resources
whose exact postcondition was first confirmed in this invocation, in reverse
order, and emits a separately domain-signed failure/rollback receipt.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import ssl
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Never, Protocol, Sequence


_DIRECT_ENTRYPOINT_RELATIVE = Path(
    "source/scripts/canary/owner_gate_foundation_apply.py"
)
_OWNER_SUPPORT_ROOT = re.compile(r"^owner-support-([0-9a-f]{40})$")
_OWNER_SUPPORT_MAX_ENTRIES = 50_000
_OWNER_SUPPORT_MAX_BYTES = 512 * 1024 * 1024


def _activate_direct_owner_support() -> str:
    """Admit only the absolute, sealed owner-support script entrypoint."""

    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError("owner_gate_foundation_direct_isolation_required")
    module_path = Path(__file__)
    invoked_path = Path(sys.argv[0])
    if (
        not module_path.is_absolute()
        or not invoked_path.is_absolute()
        or invoked_path != module_path
        or ".." in module_path.parts
    ):
        raise RuntimeError("owner_gate_foundation_direct_path_invalid")
    try:
        source_root = module_path.parents[2]
        support_root = module_path.parents[3]
    except IndexError:
        raise RuntimeError(
            "owner_gate_foundation_direct_path_invalid"
        ) from None
    match = _OWNER_SUPPORT_ROOT.fullmatch(support_root.name)
    if (
        match is None
        or module_path != support_root / _DIRECT_ENTRYPOINT_RELATIVE
        or not support_root.is_absolute()
    ):
        raise RuntimeError("owner_gate_foundation_direct_path_invalid")
    site_root = support_root / "site-packages"
    try:
        if os.path.realpath(module_path, strict=True) != str(module_path):
            raise RuntimeError("owner_gate_foundation_direct_path_invalid")
    except OSError:
        raise RuntimeError(
            "owner_gate_foundation_direct_path_invalid"
        ) from None

    entries = 0
    total_bytes = 0
    pending = [support_root]
    root_children: set[str] | None = None
    while pending:
        path = pending.pop()
        try:
            metadata = path.lstat()
        except OSError:
            raise RuntimeError(
                "owner_gate_foundation_direct_tree_invalid"
            ) from None
        if metadata.st_uid != os.getuid():  # windows-footgun: ok
            raise RuntimeError("owner_gate_foundation_direct_tree_invalid")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o500:
                raise RuntimeError(
                    "owner_gate_foundation_direct_tree_invalid"
                )
            try:
                children = tuple(path.iterdir())
            except OSError:
                raise RuntimeError(
                    "owner_gate_foundation_direct_tree_invalid"
                ) from None
            if path == support_root:
                root_children = {item.name for item in children}
            pending.extend(children)
        elif stat.S_ISREG(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
                raise RuntimeError(
                    "owner_gate_foundation_direct_tree_invalid"
                )
            total_bytes += metadata.st_size
        else:
            raise RuntimeError("owner_gate_foundation_direct_tree_invalid")
        entries += 1
        if (
            entries > _OWNER_SUPPORT_MAX_ENTRIES
            or total_bytes > _OWNER_SUPPORT_MAX_BYTES
        ):
            raise RuntimeError("owner_gate_foundation_direct_tree_invalid")
    if root_children != {"owner-support.json", "source", "site-packages"}:
        raise RuntimeError("owner_gate_foundation_direct_tree_invalid")
    for required in (
        source_root / "scripts/__init__.py",
        source_root / "scripts/canary/__init__.py",
        module_path,
        site_root / "cryptography/__init__.py",
    ):
        try:
            metadata = required.lstat()
        except OSError:
            raise RuntimeError(
                "owner_gate_foundation_direct_tree_invalid"
            ) from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("owner_gate_foundation_direct_tree_invalid")

    standard_paths = tuple(sys.path)
    if any(
        not isinstance(item, str)
        or not item
        or not os.path.isabs(item)
        or "site-packages" in Path(item).parts
        or "dist-packages" in Path(item).parts
        for item in standard_paths
    ):
        raise RuntimeError("owner_gate_foundation_direct_sys_path_invalid")
    sys.path[:] = [str(source_root), str(site_root), *standard_paths]
    return match.group(1)


_OWNER_SUPPORT_BOOTSTRAP_RELEASE_SHA = (
    _activate_direct_owner_support() if __package__ is None else None
)

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_journal as foundation_journal
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_project_ancestry as project_ancestry
from scripts.canary import owner_gate_trust as release_trust
from scripts.canary import owner_gate_trust_author as trust_author


FAILURE_RECEIPT_SCHEMA = "muncho-owner-gate-foundation-apply-failure.v1"
FAILURE_RECEIPT_PURPOSE = "muncho_owner_gate_bounded_foundation_apply_failure"
FAILURE_SIGNATURE_DOMAIN = b"muncho-owner-gate/foundation-apply-failure/v1\x00"
JOURNAL_TRANSITION_SCHEMA = "muncho-owner-gate-foundation-transition.v1"
JOURNAL_TRANSITION_PURPOSE = "muncho_owner_gate_crash_safe_foundation_transition"
JOURNAL_SIGNATURE_DOMAIN = b"muncho-owner-gate/foundation-transition/v1\x00"
MAX_JSON_BYTES = 16 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 180.0
_POSTCONDITION_VISIBILITY_ATTEMPTS = 12
_POSTCONDITION_VISIBILITY_DELAY_SECONDS = 1.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]{86}$")
_IAM_RESOURCE = re.compile(
    r"^(?:projects/[a-z][a-z0-9-]{4,28}[a-z0-9]"
    r"|organizations/[1-9][0-9]{5,30})$"
)
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
_PROVIDER_STATES = frozenset({"absent", "exact", "drift", "unknown"})
_OPERATION_STATES = frozenset({"completed", "failed", "unknown"})
_ROLLBACK_DISPOSITIONS = frozenset({
    "rolled_back",
    "rollback_failed",
    "rollback_unknown",
    "not_attempted_manual",
})


def _no_postcondition_wait(_seconds: float) -> None:
    """Keep the private test seam fast; production supplies ``time.sleep``."""

_FAILURE_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "transaction_id",
    "pre_foundation_authority_sha256",
    "inert_plan_sha256",
    "foundation_source_revision",
    "foundation_source_tree_oid",
    "owner_reauthentication_receipt_sha256",
    "ancestry_evidence_sha256",
    "ancestry_chain_sha256",
    "started_at_unix",
    "failed_at_unix",
    "failed_step_name",
    "failure_code",
    "completed_step_receipts",
    "rollback_step_receipts",
    "terminal_state",
    "partial_unknown_state",
    "mutation_iam_binding_created",
    "package_deployed",
    "service_started",
    "signer_key_id",
})
_FAILURE_FIELDS = _FAILURE_BODY_FIELDS | frozenset({
    "foundation_apply_failure_receipt_sha256",
    "signature_ed25519_b64url",
})
_ROLLBACK_RECEIPT_FIELDS = frozenset({
    "original_step_name",
    "rollback_step_name",
    "rollback_argv_sha256",
    "disposition",
    "operation_receipt_sha256",
    "postcondition_receipt_sha256",
})
_FAILURE_INTENT_FIELDS = frozenset({
    "schema",
    "purpose",
    "transaction_id",
    "phase",
    "pre_foundation_authority_sha256",
    "inert_plan_sha256",
    "started_at_unix",
    "failed_at_unix",
    "failed_step_name",
    "failure_code",
    "completed_step_receipts",
    "created_step_names",
    "inherently_unknown",
})
_SUCCESS_TRANSITION_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "transaction_id",
    "phase",
    "pre_foundation_authority_sha256",
    "inert_plan_sha256",
    "receipt",
})


class OwnerGateFoundationApplyError(RuntimeError):
    """Stable, secret-free foundation apply error."""


class FoundationApplyFailed(OwnerGateFoundationApplyError):
    """Terminal failure carrying the signed rollback/failure receipt."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__("owner_gate_foundation_apply_failed")
        self.receipt = dict(receipt)


def _error(code: str, exc: BaseException | None = None) -> Never:
    del exc
    raise OwnerGateFoundationApplyError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_foundation_apply_json_invalid", exc)


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
        _error("owner_gate_foundation_apply_failure_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_foundation_apply_failure_signature_invalid", exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        != value
    ):
        _error("owner_gate_foundation_apply_failure_signature_invalid")
    return raw


def _sign_journal_transition(
    body: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
) -> Mapping[str, Any]:
    if (
        not isinstance(private_key, Ed25519PrivateKey)
        or body.get("schema") != JOURNAL_TRANSITION_SCHEMA
        or body.get("purpose") != JOURNAL_TRANSITION_PURPOSE
        or _SHA256.fullmatch(str(body.get("transaction_id", ""))) is None
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    try:
        key_id = pre_foundation._require_pinned_public_key(
            private_key.public_key()
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_journal_transition_invalid", exc)
    digest = _sha256_json(body)
    signed = {
        **dict(body),
        "transition_sha256": digest,
        "signer_key_id": key_id,
    }
    signature = private_key.sign(
        JOURNAL_SIGNATURE_DOMAIN + _canonical(signed)
    )
    return {
        **signed,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def _verify_journal_transition(
    value: Any,
    *,
    public_key: Ed25519PublicKey,
    transaction_id: str,
    expected_phase: str,
    expected_step_index: int | None = None,
    expected_step_name: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_foundation_journal_transition_invalid")
    body = {
        key: item
        for key, item in value.items()
        if key not in {
            "transition_sha256",
            "signer_key_id",
            "signature_ed25519_b64url",
        }
    }
    try:
        key_id = pre_foundation._require_pinned_public_key(public_key)
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_journal_transition_invalid", exc)
    signed = {
        **body,
        "transition_sha256": value.get("transition_sha256"),
        "signer_key_id": value.get("signer_key_id"),
    }
    if (
        body.get("schema") != JOURNAL_TRANSITION_SCHEMA
        or body.get("purpose") != JOURNAL_TRANSITION_PURPOSE
        or body.get("transaction_id") != transaction_id
        or body.get("phase") != expected_phase
        or (
            expected_step_index is not None
            and body.get("step_index") != expected_step_index
        )
        or (
            expected_step_name is not None
            and body.get("step_name") != expected_step_name
        )
        or value.get("transition_sha256") != _sha256_json(body)
        or value.get("signer_key_id") != key_id
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    try:
        public_key.verify(
            _decode_signature(value.get("signature_ed25519_b64url")),
            JOURNAL_SIGNATURE_DOMAIN + _canonical(signed),
        )
    except InvalidSignature as exc:
        _error("owner_gate_foundation_journal_transition_invalid", exc)
    return dict(body)


@dataclass(frozen=True)
class ResourceObservation:
    state: str
    receipt_sha256: str
    resource_identity: Mapping[str, Any] | None = None
    precondition: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if (
            self.state not in _PROVIDER_STATES
            or _SHA256.fullmatch(self.receipt_sha256 or "") is None
            or (self.state == "exact") is not isinstance(
                self.resource_identity, Mapping
            )
            or (
                self.state != "exact"
                and self.resource_identity is not None
            )
            or (
                self.precondition is not None
                and not isinstance(self.precondition, Mapping)
            )
        ):
            _error("owner_gate_foundation_provider_observation_invalid")


@dataclass(frozen=True)
class OperationObservation:
    state: str
    receipt_sha256: str
    attempt_id: str
    provider_result_binding_sha256: str | None = None
    cas_precondition_etag: str | None = None
    cas_postcondition_etag: str | None = None

    def validate(self) -> None:
        if (
            self.state not in _OPERATION_STATES
            or _SHA256.fullmatch(self.receipt_sha256 or "") is None
            or _SHA256.fullmatch(self.attempt_id or "") is None
            or (
                self.state == "completed"
                and _SHA256.fullmatch(
                    self.provider_result_binding_sha256 or ""
                )
                is None
            )
            or any(
                value is not None
                and (not isinstance(value, str) or not value)
                for value in (
                    self.cas_precondition_etag,
                    self.cas_postcondition_etag,
                )
            )
        ):
            _error("owner_gate_foundation_provider_operation_invalid")


class FoundationApplyProvider(Protocol):
    def assert_stable(self) -> None: ...

    def observe_ancestry_chain(self) -> Sequence[Mapping[str, Any]]: ...

    def inspect_resource(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
    ) -> ResourceObservation: ...

    def execute_step(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> OperationObservation: ...

    def rollback_step(
        self,
        original_step: foundation.PlanStep,
        rollback_step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> OperationObservation: ...


@dataclass(frozen=True)
class _ProviderExecutionResult:
    """Exact provider-collected success input accepted by the receipt signer."""

    step_receipts: tuple[Mapping[str, Any], ...]
    started_at_unix: int
    completed_at_unix: int


_CHAIN_MARKER = object()


@dataclass(frozen=True, init=False)
class ValidatedFoundationAChain:
    """Validated signed foundation-A lineage; no caller raw IDs survive."""

    authority: Mapping[str, Any]
    owner_reauthentication_receipt: Mapping[str, Any]
    plan: foundation.OwnerGateFoundationPlan
    network_evidence: foundation.ProductionNetworkEvidence
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence
    network_evidence_raw: bytes
    ancestry_evidence_raw: bytes
    pre_foundation_authority_raw: bytes
    owner_reauthentication_receipt_raw: bytes
    release_public_key: Ed25519PublicKey
    network_collector_public_key: Ed25519PublicKey
    ancestry_collector_public_key: Ed25519PublicKey
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "ValidatedFoundationAChain":
        _error("owner_gate_foundation_chain_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        authority: Mapping[str, Any],
        owner_reauthentication_receipt: Mapping[str, Any],
        plan: foundation.OwnerGateFoundationPlan,
        network_evidence: foundation.ProductionNetworkEvidence,
        ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
        network_evidence_raw: bytes,
        ancestry_evidence_raw: bytes,
        pre_foundation_authority_raw: bytes,
        owner_reauthentication_receipt_raw: bytes,
        release_public_key: Ed25519PublicKey,
        network_collector_public_key: Ed25519PublicKey,
        ancestry_collector_public_key: Ed25519PublicKey,
    ) -> "ValidatedFoundationAChain":
        value = object.__new__(cls)
        for name, item in locals().copy().items():
            if name not in {"cls", "value"}:
                object.__setattr__(value, name, item)
        object.__setattr__(value, "_marker", _CHAIN_MARKER)
        return value

    def __post_init__(self) -> None:
        if self._marker is not _CHAIN_MARKER:
            _error("owner_gate_foundation_chain_invalid")

    @property
    def foundation_source_revision(self) -> str:
        return str(self.authority["foundation_source_revision"])

    @property
    def foundation_source_tree_oid(self) -> str:
        return str(self.authority["foundation_source_tree_oid"])

    @property
    def pre_foundation_authority_sha256(self) -> str:
        return str(self.authority["pre_foundation_authority_sha256"])

    @property
    def owner_reauthentication_receipt_sha256(self) -> str:
        return str(
            self.owner_reauthentication_receipt[
                "owner_reauthentication_receipt_sha256"
            ]
        )

    @property
    def ancestry_evidence_sha256(self) -> str:
        return self.ancestry_evidence.signed_evidence_sha256

    @property
    def network_evidence_sha256(self) -> str:
        return self.network_evidence.evidence_sha256

    @property
    def signed_network_evidence_sha256(self) -> str:
        return _sha256_bytes(self.network_evidence_raw)


@dataclass(frozen=True, init=False)
class ValidatedFoundationApplyChain:
    """Opaque post-apply capability created only from canonical signed input."""

    foundation_a: ValidatedFoundationAChain
    apply_receipt: Mapping[str, Any]
    apply_receipt_raw: bytes
    _marker: object

    def __new__(
        cls, *_args: Any, **_kwargs: Any
    ) -> "ValidatedFoundationApplyChain":
        _error("owner_gate_foundation_apply_chain_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        foundation_a: ValidatedFoundationAChain,
        apply_receipt: Mapping[str, Any],
        apply_receipt_raw: bytes,
    ) -> "ValidatedFoundationApplyChain":
        if type(foundation_a) is not ValidatedFoundationAChain:
            _error("owner_gate_foundation_apply_chain_invalid")
        value = object.__new__(cls)
        object.__setattr__(value, "foundation_a", foundation_a)
        object.__setattr__(value, "apply_receipt", dict(apply_receipt))
        object.__setattr__(value, "apply_receipt_raw", apply_receipt_raw)
        object.__setattr__(value, "_marker", _CHAIN_MARKER)
        return value

    def __post_init__(self) -> None:
        if self._marker is not _CHAIN_MARKER:
            _error("owner_gate_foundation_apply_chain_invalid")

    @property
    def foundation_source_revision(self) -> str:
        return self.foundation_a.foundation_source_revision

    @property
    def foundation_source_tree_oid(self) -> str:
        return self.foundation_a.foundation_source_tree_oid

    @property
    def pre_foundation_authority_sha256(self) -> str:
        return self.foundation_a.pre_foundation_authority_sha256

    @property
    def foundation_apply_receipt_sha256(self) -> str:
        return str(self.apply_receipt["foundation_apply_receipt_sha256"])

    @property
    def owner_reauthentication_receipt_sha256(self) -> str:
        return self.foundation_a.owner_reauthentication_receipt_sha256

    @property
    def step_resource_identities(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            str(item["step_name"]): dict(item["resource_identity"])
            for item in self.apply_receipt["applied_steps"]
        }

    def resource_identity(self, step_name: str) -> Mapping[str, Any]:
        try:
            return dict(self.step_resource_identities[step_name])
        except KeyError as exc:
            _error("owner_gate_foundation_apply_chain_step_invalid", exc)

    @property
    def owner_gate_vm_identity(self) -> Mapping[str, Any]:
        return self.resource_identity("create_private_owner_gate_vm")

    @property
    def service_account_identity(self) -> Mapping[str, Any]:
        return self.resource_identity("create_dedicated_service_account")

    @property
    def subnet_identity(self) -> Mapping[str, Any]:
        return self.resource_identity(
            "create_dedicated_private_owner_gate_subnet"
        )


def _decode_canonical_mapping(raw: bytes, *, code: str) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_JSON_BYTES:
        _error(code)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error(code, exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error(code)
    return dict(value)


def decode_validated_foundation_a_chain(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
    now_unix: int,
) -> ValidatedFoundationAChain:
    """Decode the exact canonical signed foundation-A lineage."""

    if (
        not isinstance(release_public_key, Ed25519PublicKey)
        or not isinstance(network_collector_public_key, Ed25519PublicKey)
        or not isinstance(
            project_ancestry_collector_public_key, Ed25519PublicKey
        )
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error("owner_gate_foundation_chain_invalid")
    try:
        reauth_receipt = owner_reauth.decode_canonical_owner_reauth_receipt(
            owner_reauthentication_receipt_raw,
            public_key=release_public_key,
            now_unix=now_unix,
        )
        network_value = _decode_canonical_mapping(
            network_evidence_raw,
            code="owner_gate_foundation_network_evidence_invalid",
        )
        network_key_id = _sha256_bytes(
            network_collector_public_key.public_bytes_raw()
        )
        network = foundation.ProductionNetworkEvidence.from_mapping(
            network_value,
            public_key=network_collector_public_key,
            expected_public_key_id=network_key_id,
            now_unix=now_unix,
        )
        ancestry = (
            project_ancestry.decode_canonical_project_ancestry_evidence(
                project_ancestry_evidence_raw,
                collector_public_key=(
                    project_ancestry_collector_public_key
                ),
                owner_reauthentication_receipt=reauth_receipt,
                owner_reauthentication_public_key=release_public_key,
                expected_release_revision=str(
                    reauth_receipt["trusted_runtime_identity"][
                        "release_revision"
                    ]
                ),
                now_unix=now_unix,
            )
        )
        authority_value = _decode_canonical_mapping(
            pre_foundation_authority_raw,
            code="owner_gate_foundation_authority_invalid",
        )
        spec = pre_foundation.spec_from_authority(authority_value)
        plan = foundation.build_plan(
            spec=spec,
            network_evidence=network,
            network_collector_public_key=network_collector_public_key,
            now_unix=now_unix,
        )
        authority = pre_foundation.decode_canonical_authority(
            pre_foundation_authority_raw,
            public_key=release_public_key,
            owner_reauthentication_receipt=reauth_receipt,
            now_unix=now_unix,
            expected_plan=plan,
            network_evidence=network,
            network_collector_public_key=network_collector_public_key,
            project_ancestry_evidence_raw=(
                project_ancestry_evidence_raw
            ),
            project_ancestry_collector_public_key=(
                project_ancestry_collector_public_key
            ),
        )
    except (
        foundation.OwnerGateFoundationError,
        owner_reauth.OwnerGateOwnerReauthError,
        pre_foundation.OwnerGatePreFoundationError,
        project_ancestry.OwnerGateProjectAncestryError,
    ) as exc:
        _error("owner_gate_foundation_chain_invalid", exc)
    if (
        network_collector_public_key.public_bytes_raw()
        != project_ancestry_collector_public_key.public_bytes_raw()
        or authority["foundation_source_revision"]
        != reauth_receipt["trusted_runtime_identity"]["release_revision"]
        or authority["ancestry_evidence_sha256"]
        != ancestry.signed_evidence_sha256
    ):
        _error("owner_gate_foundation_chain_invalid")
    return ValidatedFoundationAChain._create(
        authority=authority,
        owner_reauthentication_receipt=reauth_receipt,
        plan=plan,
        network_evidence=network,
        ancestry_evidence=ancestry,
        network_evidence_raw=network_evidence_raw,
        ancestry_evidence_raw=project_ancestry_evidence_raw,
        pre_foundation_authority_raw=pre_foundation_authority_raw,
        owner_reauthentication_receipt_raw=(
            owner_reauthentication_receipt_raw
        ),
        release_public_key=release_public_key,
        network_collector_public_key=network_collector_public_key,
        ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
    )


def _decode_validated_foundation_apply_chain(
    *,
    foundation_a: ValidatedFoundationAChain,
    apply_receipt_raw: bytes,
    now_unix: int,
) -> ValidatedFoundationApplyChain:
    """Decode a canonical signed apply receipt over an exact A capability."""

    if (
        type(foundation_a) is not ValidatedFoundationAChain
        or foundation_a._marker is not _CHAIN_MARKER
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error("owner_gate_foundation_apply_chain_invalid")
    try:
        receipt = pre_foundation.decode_canonical_apply_receipt(
            apply_receipt_raw,
            public_key=foundation_a.release_public_key,
            authority=foundation_a.authority,
            owner_reauthentication_receipt=(
                foundation_a.owner_reauthentication_receipt
            ),
            project_ancestry_evidence_raw=(
                foundation_a.ancestry_evidence_raw
            ),
            project_ancestry_collector_public_key=(
                foundation_a.ancestry_collector_public_key
            ),
            plan=foundation_a.plan,
            now_unix=now_unix,
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_apply_chain_invalid", exc)
    return ValidatedFoundationApplyChain._create(
        foundation_a=foundation_a,
        apply_receipt=receipt,
        apply_receipt_raw=apply_receipt_raw,
    )


_ROLLBACK_BY_STEP = {
    "create_dedicated_service_account": (
        "delete_dedicated_service_account_if_created"
    ),
    "create_narrow_iam_observation_reader_role": (
        "delete_read_only_iam_custom_role_if_created"
    ),
    "bind_narrow_iam_observation_reader_to_owner_gate_service_account": (
        "remove_exact_read_only_iam_observation_binding_if_present"
    ),
    "create_narrow_storage_executor_role": (
        "delete_mutation_custom_role_if_created"
    ),
    "create_narrow_organization_iam_observation_reader_role": (
        "delete_organization_iam_observation_role_if_created"
    ),
    "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account": (
        "remove_exact_organization_iam_observation_binding_if_present"
    ),
    "create_dedicated_private_owner_gate_subnet": (
        "delete_dedicated_owner_gate_subnet_if_created"
    ),
    "create_private_owner_gate_vm": (
        "delete_private_owner_gate_vm_if_created"
    ),
    "allow_private_web_upstream_from_current_caddy_host": (
        "delete_private_web_firewall_if_created"
    ),
}


def _rollback_step_for(
    step: foundation.PlanStep,
    *,
    plan: foundation.OwnerGateFoundationPlan,
) -> foundation.PlanStep:
    expected = _ROLLBACK_BY_STEP.get(step.name)
    matches = [item for item in plan.rollback_steps if item.name == expected]
    if expected is None or len(matches) != 1:
        _error("owner_gate_foundation_rollback_plan_invalid")
    return matches[0]


def _validate_live_ancestry(
    provider: FoundationApplyProvider,
    chain: ValidatedFoundationAChain,
) -> None:
    try:
        live = list(provider.observe_ancestry_chain())
    except OwnerGateFoundationApplyError:
        raise
    except Exception as exc:
        _error("owner_gate_foundation_live_ancestry_unknown", exc)
    if not project_ancestry.live_chain_equals_evidence(
        chain.ancestry_evidence,
        live,
    ):
        _error("owner_gate_foundation_live_ancestry_mismatch")


def _fresh_reauth(
    chain: ValidatedFoundationAChain,
    *,
    now_unix: int,
) -> None:
    try:
        owner_reauth.validate_owner_reauth_receipt(
            chain.owner_reauthentication_receipt,
            public_key=chain.release_public_key,
            now_unix=now_unix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        _error("owner_gate_foundation_owner_reauth_expired", exc)


def _sign_failure_receipt(
    body: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
) -> Mapping[str, Any]:
    checked = _validate_failure_body(body)
    try:
        key_id = pre_foundation._require_pinned_public_key(
            private_key.public_key()
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_failure_signer_invalid", exc)
    if checked["signer_key_id"] != key_id:
        _error("owner_gate_foundation_failure_signer_invalid")
    signed = {
        **checked,
        "foundation_apply_failure_receipt_sha256": _sha256_json(checked),
    }
    signature = private_key.sign(
        FAILURE_SIGNATURE_DOMAIN + _canonical(signed)
    )
    return {
        **signed,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def _validate_failure_body(value: Any) -> Mapping[str, Any]:
    body = _strict_mapping(
        value,
        _FAILURE_BODY_FIELDS,
        code="owner_gate_foundation_failure_receipt_invalid",
    )
    completed = body.get("completed_step_receipts")
    rollbacks = body.get("rollback_step_receipts")
    if not isinstance(completed, list) or not isinstance(rollbacks, list):
        _error("owner_gate_foundation_failure_receipt_invalid")
    checked_rollbacks: list[Mapping[str, Any]] = []
    for raw in rollbacks:
        item = _strict_mapping(
            raw,
            _ROLLBACK_RECEIPT_FIELDS,
            code="owner_gate_foundation_failure_receipt_invalid",
        )
        if (
            not isinstance(item.get("original_step_name"), str)
            or item.get("original_step_name") not in _ROLLBACK_BY_STEP
            or not isinstance(item.get("rollback_step_name"), str)
            or item.get("rollback_step_name")
            != _ROLLBACK_BY_STEP[item["original_step_name"]]
            or item.get("disposition") not in _ROLLBACK_DISPOSITIONS
            or any(
                _SHA256.fullmatch(str(item.get(field, ""))) is None
                for field in (
                    "rollback_argv_sha256",
                    "operation_receipt_sha256",
                    "postcondition_receipt_sha256",
                )
            )
        ):
            _error("owner_gate_foundation_failure_receipt_invalid")
        checked_rollbacks.append(dict(item))
    started = body.get("started_at_unix")
    failed = body.get("failed_at_unix")
    partial = body.get("partial_unknown_state")
    terminal = body.get("terminal_state")
    if (
        body.get("schema") != FAILURE_RECEIPT_SCHEMA
        or body.get("purpose") != FAILURE_RECEIPT_PURPOSE
        or any(
            _SHA256.fullmatch(str(body.get(field, ""))) is None
            for field in (
                "pre_foundation_authority_sha256",
                "inert_plan_sha256",
                "owner_reauthentication_receipt_sha256",
                "ancestry_evidence_sha256",
                "ancestry_chain_sha256",
                "signer_key_id",
            )
        )
        or _SHA256.fullmatch(str(body.get("transaction_id", ""))) is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(body.get("foundation_source_revision", "")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(body.get("foundation_source_tree_oid", "")),
        )
        is None
        or type(started) is not int
        or type(failed) is not int
        or started <= 0
        or failed < started
        or not isinstance(body.get("failed_step_name"), str)
        or not isinstance(body.get("failure_code"), str)
        or re.fullmatch(
            r"owner_gate_foundation_[a-z0-9_]{1,120}",
            body["failure_code"],
        )
        is None
        or type(partial) is not bool
        or terminal
        not in {"rolled_back_clean", "manual_reconciliation_required"}
        or (terminal == "manual_reconciliation_required") is not partial
        or (
            any(
                item["disposition"] != "rolled_back"
                for item in checked_rollbacks
            )
            and not partial
        )
        or body.get("mutation_iam_binding_created") is not False
        or body.get("package_deployed") is not False
        or body.get("service_started") is not False
    ):
        _error("owner_gate_foundation_failure_receipt_invalid")
    return {
        **dict(body),
        "completed_step_receipts": [dict(item) for item in completed],
        "rollback_step_receipts": checked_rollbacks,
    }


def validate_failure_receipt(
    value: Any,
    *,
    public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    receipt = _strict_mapping(
        value,
        _FAILURE_FIELDS,
        code="owner_gate_foundation_failure_receipt_invalid",
    )
    body = {
        key: item
        for key, item in receipt.items()
        if key
        not in {
            "foundation_apply_failure_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    checked = _validate_failure_body(body)
    try:
        key_id = pre_foundation._require_pinned_public_key(public_key)
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_failure_signer_invalid", exc)
    signed = {
        **checked,
        "foundation_apply_failure_receipt_sha256": receipt.get(
            "foundation_apply_failure_receipt_sha256"
        ),
    }
    if (
        checked["signer_key_id"] != key_id
        or receipt.get("foundation_apply_failure_receipt_sha256")
        != _sha256_json(checked)
    ):
        _error("owner_gate_foundation_failure_receipt_invalid")
    try:
        public_key.verify(
            _decode_signature(receipt.get("signature_ed25519_b64url")),
            FAILURE_SIGNATURE_DOMAIN + _canonical(signed),
        )
    except InvalidSignature as exc:
        _error("owner_gate_foundation_apply_failure_signature_invalid", exc)
    return dict(receipt)


def _transaction_id(chain: ValidatedFoundationAChain) -> str:
    return _sha256_json({
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "inert_plan_sha256": pre_foundation.inert_plan_sha256(chain.plan),
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "signed_network_evidence_sha256": (
            chain.signed_network_evidence_sha256
        ),
    })


def _transition_body(
    *,
    chain: ValidatedFoundationAChain,
    transaction_id: str,
    phase: str,
    step_index: int | None = None,
    step: foundation.PlanStep | None = None,
    payload: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    body: dict[str, Any] = {
        "schema": JOURNAL_TRANSITION_SCHEMA,
        "purpose": JOURNAL_TRANSITION_PURPOSE,
        "transaction_id": transaction_id,
        "phase": phase,
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "inert_plan_sha256": pre_foundation.inert_plan_sha256(chain.plan),
    }
    if step_index is not None and step is not None:
        body.update({
            "step_index": step_index,
            "step_name": step.name,
            "argv_sha256": _sha256_json(list(step.argv)),
        })
    if payload:
        body.update(dict(payload))
    return body


def _publish_transition(
    *,
    journal: foundation_journal.FoundationApplyJournal,
    transaction_id: str,
    name: str,
    body: Mapping[str, Any],
    private_key: Ed25519PrivateKey,
) -> Mapping[str, Any]:
    try:
        return journal.publish(
            transaction_id,
            name,
            _sign_journal_transition(body, private_key=private_key),
        )
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_foundation_journal_write_failed", exc)


def _read_transition(
    *,
    journal: foundation_journal.FoundationApplyJournal,
    chain: ValidatedFoundationAChain,
    transaction_id: str,
    name: str,
    phase: str,
    step_index: int | None = None,
    step: foundation.PlanStep | None = None,
    strict_read_only: bool = False,
) -> Mapping[str, Any] | None:
    try:
        value = (
            journal.read_strict(transaction_id, name)
            if strict_read_only
            else journal.read(transaction_id, name)
        )
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_foundation_journal_read_failed", exc)
    if value is None:
        return None
    body = _verify_journal_transition(
        value,
        public_key=chain.release_public_key,
        transaction_id=transaction_id,
        expected_phase=phase,
        expected_step_index=step_index,
        expected_step_name=(step.name if step is not None else None),
    )
    if (
        body.get("pre_foundation_authority_sha256")
        != chain.pre_foundation_authority_sha256
        or body.get("inert_plan_sha256")
        != pre_foundation.inert_plan_sha256(chain.plan)
        or (
            step is not None
            and body.get("argv_sha256")
            != _sha256_json(list(step.argv))
        )
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    return body


def _load_fixed_success_for_validated_a(
    foundation_a: ValidatedFoundationAChain,
    *,
    now_unix: int | None,
) -> ValidatedFoundationApplyChain:
    if (
        type(foundation_a) is not ValidatedFoundationAChain
        or foundation_a._marker is not _CHAIN_MARKER
    ):
        _error("owner_gate_foundation_apply_chain_invalid")
    if now_unix is not None and (type(now_unix) is not int or now_unix <= 0):
        _error("owner_gate_foundation_apply_time_invalid")
    transaction_id = _transaction_id(foundation_a)
    store = foundation_journal.FoundationApplyJournal()
    success = _read_transition(
        journal=store,
        chain=foundation_a,
        transaction_id=transaction_id,
        name="success",
        phase="success",
        strict_read_only=True,
    )
    failure_intent = _read_transition(
        journal=store,
        chain=foundation_a,
        transaction_id=transaction_id,
        name="failure-intent",
        phase="failure_intent",
        strict_read_only=True,
    )
    failure = _read_transition(
        journal=store,
        chain=foundation_a,
        transaction_id=transaction_id,
        name="failure",
        phase="failure",
        strict_read_only=True,
    )
    if (
        success is None
        or failure_intent is not None
        or failure is not None
        or frozenset(success) != _SUCCESS_TRANSITION_BODY_FIELDS
        or not isinstance(success.get("receipt"), Mapping)
    ):
        _error("owner_gate_foundation_success_journal_invalid")
    receipt = success["receipt"]
    decode_time = now_unix
    if decode_time is None:
        completed = receipt.get("completed_at_unix")
        if type(completed) is not int or completed <= 0:
            _error("owner_gate_foundation_success_journal_invalid")
        decode_time = completed
    return _decode_validated_foundation_apply_chain(
        foundation_a=foundation_a,
        apply_receipt_raw=_canonical(receipt),
        now_unix=decode_time,
    )


def load_validated_foundation_apply_chain(
    foundation_a: ValidatedFoundationAChain,
) -> ValidatedFoundationApplyChain:
    """Load the exact signed success capability from the fixed journal.

    This boundary performs no recovery and accepts no journal path, raw apply
    receipt, output path, or caller-authored identity.  The signed transition
    and its nested signed receipt must both validate against ``foundation_a``.
    """

    now_unix = int(time.time())
    if now_unix <= 0:
        _error("owner_gate_foundation_apply_time_invalid")
    return _load_fixed_success_for_validated_a(
        foundation_a,
        now_unix=now_unix,
    )


def _load_validated_foundation_apply_chain_for_source_recovery(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> ValidatedFoundationApplyChain:
    """Historical A validation solely for persisted source-artifact recovery.

    The caller cannot select a journal, receipt, output, or validation time.
    A is validated at its own signed issue time and the apply time is obtained
    only from the nested receipt in the fixed, conflict-free success journal.
    """

    authority_value = _decode_canonical_mapping(
        pre_foundation_authority_raw,
        code="owner_gate_foundation_authority_invalid",
    )
    issued_at = authority_value.get("issued_at_unix")
    if type(issued_at) is not int or issued_at <= 0:
        _error("owner_gate_foundation_authority_invalid")
    foundation_a = decode_validated_foundation_a_chain(
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
        now_unix=issued_at,
    )
    return _load_fixed_success_for_validated_a(
        foundation_a,
        now_unix=None,
    )


def load_foundation_terminal_for_source_recovery(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Read one exact historical terminal from the fixed apply journal only.

    No caller can select a journal, validation time, provider, path, or output.
    The result distinguishes durable success/failure from absence and an
    incomplete transaction, so an outer recovery layer cannot manufacture a
    false failure after the original owner reauthentication expires.
    """

    authority_value = _decode_canonical_mapping(
        pre_foundation_authority_raw,
        code="owner_gate_foundation_authority_invalid",
    )
    issued_at = authority_value.get("issued_at_unix")
    if type(issued_at) is not int or issued_at <= 0:
        _error("owner_gate_foundation_authority_invalid")
    chain = decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=pre_foundation_authority_raw,
        owner_reauthentication_receipt_raw=owner_reauthentication_receipt_raw,
        network_evidence_raw=network_evidence_raw,
        project_ancestry_evidence_raw=project_ancestry_evidence_raw,
        release_public_key=release_public_key,
        network_collector_public_key=network_collector_public_key,
        project_ancestry_collector_public_key=(
            project_ancestry_collector_public_key
        ),
        now_unix=issued_at,
    )
    transaction_id = _transaction_id(chain)
    store = foundation_journal.FoundationApplyJournal()
    manifest = _read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="manifest",
        phase="manifest",
        strict_read_only=True,
    )
    success = _read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="success",
        phase="success",
        strict_read_only=True,
    )
    failure_intent = _read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="failure-intent",
        phase="failure_intent",
        strict_read_only=True,
    )
    failure = _read_transition(
        journal=store,
        chain=chain,
        transaction_id=transaction_id,
        name="failure",
        phase="failure",
        strict_read_only=True,
    )
    if manifest is None:
        if any(item is not None for item in (success, failure_intent, failure)):
            _error("owner_gate_foundation_terminal_journal_invalid")
        return {"state": "absent", "transaction_id": transaction_id}
    started = manifest.get("started_at_unix")
    if type(started) is not int or started <= 0:
        _error("owner_gate_foundation_journal_transition_invalid")
    if success is not None:
        if failure_intent is not None or failure is not None:
            _error("owner_gate_foundation_terminal_journal_invalid")
        validated = _load_fixed_success_for_validated_a(chain, now_unix=None)
        return {
            "state": "succeeded",
            "transaction_id": transaction_id,
            "receipt": dict(validated.apply_receipt),
        }
    if failure is not None:
        if failure_intent is None or not isinstance(failure.get("receipt"), Mapping):
            _error("owner_gate_foundation_failure_intent_missing")
        intent_values = _validate_failure_intent_transition(
            failure_intent,
            chain=chain,
            transaction_id=transaction_id,
            started_at_unix=started,
        )
        receipt = validate_failure_receipt(
            failure["receipt"],
            public_key=chain.release_public_key,
        )
        (
            intent_completed,
            intent_created,
            intent_failed_at,
            intent_failed_step,
            intent_failure_code,
            _intent_unknown,
        ) = intent_values
        if (
            receipt.get("transaction_id") != transaction_id
            or receipt.get("pre_foundation_authority_sha256")
            != chain.pre_foundation_authority_sha256
            or receipt.get("inert_plan_sha256")
            != pre_foundation.inert_plan_sha256(chain.plan)
            or receipt.get("foundation_source_revision")
            != chain.foundation_source_revision
            or receipt.get("foundation_source_tree_oid")
            != chain.foundation_source_tree_oid
            or receipt.get("owner_reauthentication_receipt_sha256")
            != chain.owner_reauthentication_receipt_sha256
            or receipt.get("ancestry_evidence_sha256")
            != chain.ancestry_evidence_sha256
            or receipt.get("ancestry_chain_sha256")
            != chain.ancestry_evidence.value["stable_chain_sha256"]
            or receipt.get("started_at_unix") != started
            or receipt.get("failed_at_unix") != intent_failed_at
            or receipt.get("failed_step_name") != intent_failed_step
            or receipt.get("failure_code") != intent_failure_code
            or receipt.get("completed_step_receipts") != intent_completed
            or [
                item["step_name"]
                for item in intent_completed
                if item["disposition"] == "created"
            ]
            != [step.name for step in intent_created]
        ):
            _error("owner_gate_foundation_failure_receipt_invalid")
        return {
            "state": "failed",
            "transaction_id": transaction_id,
            "receipt": dict(receipt),
        }
    return {
        "state": "in_progress",
        "transaction_id": transaction_id,
        "failure_intent_present": failure_intent is not None,
    }


def _step_receipt_from_transition(
    body: Mapping[str, Any],
    *,
    step: foundation.PlanStep,
    plan: foundation.OwnerGateFoundationPlan,
    disposition: str,
) -> Mapping[str, Any]:
    identity = pre_foundation._validate_resource_identity(
        step.name,
        body.get("resource_identity"),
        plan=plan,
    )
    operation_sha = body.get("operation_receipt_sha256")
    post_sha = body.get("postcondition_receipt_sha256")
    if (
        disposition == "preexisting_exact"
        and _SHA256.fullmatch(str(post_sha or "")) is not None
    ):
        operation_sha = _sha256_json({
            "step_name": step.name,
            "disposition": disposition,
            "precondition_receipt_sha256": post_sha,
        })
    if (
        _SHA256.fullmatch(str(operation_sha or "")) is None
        or _SHA256.fullmatch(str(post_sha or "")) is None
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    return {
        "step_name": step.name,
        "argv_sha256": _sha256_json(list(step.argv)),
        "disposition": disposition,
        "operation_receipt_sha256": operation_sha,
        "postcondition_receipt_sha256": post_sha,
        "resource_identity": identity,
    }


def _validate_partial_step_receipts(
    value: Any,
    *,
    plan: foundation.OwnerGateFoundationPlan,
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) > len(plan.foundation_steps)
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    checked: list[Mapping[str, Any]] = []
    fields = frozenset({
        "step_name",
        "argv_sha256",
        "disposition",
        "operation_receipt_sha256",
        "postcondition_receipt_sha256",
        "resource_identity",
    })
    for raw, step in zip(value, plan.foundation_steps, strict=False):
        item = _strict_mapping(
            raw,
            fields,
            code="owner_gate_foundation_journal_transition_invalid",
        )
        if (
            item.get("step_name") != step.name
            or item.get("argv_sha256") != _sha256_json(list(step.argv))
            or item.get("disposition")
            not in {"created", "preexisting_exact"}
            or _SHA256.fullmatch(
                str(item.get("operation_receipt_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(item.get("postcondition_receipt_sha256", ""))
            )
            is None
        ):
            _error("owner_gate_foundation_journal_transition_invalid")
        try:
            identity = pre_foundation._validate_resource_identity(
                step.name,
                item.get("resource_identity"),
                plan=plan,
            )
        except pre_foundation.OwnerGatePreFoundationError as exc:
            _error("owner_gate_foundation_journal_transition_invalid", exc)
        checked.append({**dict(item), "resource_identity": identity})
    return checked


def _validate_failure_intent_transition(
    value: Any,
    *,
    chain: ValidatedFoundationAChain,
    transaction_id: str,
    started_at_unix: int,
) -> tuple[
    list[Mapping[str, Any]],
    list[foundation.PlanStep],
    int,
    str,
    str,
    bool,
]:
    body = _strict_mapping(
        value,
        _FAILURE_INTENT_FIELDS,
        code="owner_gate_foundation_journal_transition_invalid",
    )
    completed = _validate_partial_step_receipts(
        body.get("completed_step_receipts"),
        plan=chain.plan,
    )
    created_names = body.get("created_step_names")
    failed_at = body.get("failed_at_unix")
    failed_step = body.get("failed_step_name")
    failure_code = body.get("failure_code")
    inherently_unknown = body.get("inherently_unknown")
    expected_created = [
        str(item["step_name"])
        for item in completed
        if item["disposition"] == "created"
    ]
    steps_by_name = {step.name: step for step in chain.plan.foundation_steps}
    if (
        body.get("transaction_id") != transaction_id
        or body.get("phase") != "failure_intent"
        or body.get("started_at_unix") != started_at_unix
        or not isinstance(created_names, list)
        or created_names != expected_created
        or any(name not in steps_by_name for name in created_names)
        or type(failed_at) is not int
        or failed_at < started_at_unix
        or not isinstance(failed_step, str)
        or not failed_step
        or not isinstance(failure_code, str)
        or re.fullmatch(
            r"owner_gate_foundation_[a-z0-9_]{1,120}",
            failure_code,
        )
        is None
        or type(inherently_unknown) is not bool
    ):
        _error("owner_gate_foundation_journal_transition_invalid")
    return (
        completed,
        [steps_by_name[name] for name in created_names],
        failed_at,
        failed_step,
        failure_code,
        inherently_unknown,
    )


def _require_live_journaled_exact(
    *,
    provider: FoundationApplyProvider,
    chain: ValidatedFoundationAChain,
    step: foundation.PlanStep,
    transition: Mapping[str, Any],
) -> ResourceObservation:
    try:
        expected = pre_foundation._validate_resource_identity(
            step.name,
            transition.get("resource_identity"),
            plan=chain.plan,
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_foundation_journal_transition_invalid", exc)
    observed = provider.inspect_resource(step, plan=chain.plan)
    observed.validate()
    provider.assert_stable()
    if (
        observed.state != "exact"
        or dict(observed.resource_identity or {}) != dict(expected)
    ):
        _error("owner_gate_foundation_journaled_resource_not_exact")
    return observed


def _operation_from_transition(body: Mapping[str, Any]) -> OperationObservation:
    operation = OperationObservation(
        state=str(body.get("operation_state", "")),
        receipt_sha256=str(body.get("operation_receipt_sha256", "")),
        attempt_id=str(body.get("attempt_id", "")),
        provider_result_binding_sha256=(
            str(body["provider_result_binding_sha256"])
            if body.get("provider_result_binding_sha256") is not None
            else None
        ),
        cas_precondition_etag=(
            str(body["cas_precondition_etag"])
            if body.get("cas_precondition_etag") is not None
            else None
        ),
        cas_postcondition_etag=(
            str(body["cas_postcondition_etag"])
            if body.get("cas_postcondition_etag") is not None
            else None
        ),
    )
    operation.validate()
    return operation


def _binding_cas_exact(
    step: foundation.PlanStep,
    *,
    before_precondition: Mapping[str, Any] | None,
    operation: OperationObservation,
    post: ResourceObservation,
) -> bool:
    if not step.name.startswith("bind_narrow_"):
        return True
    pre_etag = (
        before_precondition.get("policy_etag")
        if isinstance(before_precondition, Mapping)
        else None
    )
    post_etag = (
        post.resource_identity.get("policy_etag")
        if isinstance(post.resource_identity, Mapping)
        else None
    )
    return (
        isinstance(pre_etag, str)
        and bool(pre_etag)
        and isinstance(post_etag, str)
        and bool(post_etag)
        and pre_etag != post_etag
        and operation.cas_precondition_etag == pre_etag
        and operation.cas_postcondition_etag == post_etag
    )


def _observe_completed_postcondition(
    *,
    provider: FoundationApplyProvider,
    chain: ValidatedFoundationAChain,
    step: foundation.PlanStep,
    operation: OperationObservation,
    now_unix: Callable[[], int],
    wait: Callable[[float], None],
) -> ResourceObservation:
    """Bound only provider visibility lag after a confirmed create call.

    Drift is never retried.  Only an absent or unstable read after a provider
    reported ``completed`` may be observed again, and every retry rechecks the
    sealed runtime and owner-reauthentication lifetime.
    """

    if operation.state != "completed":
        post = provider.inspect_resource(step, plan=chain.plan)
        post.validate()
        return post
    for attempt in range(_POSTCONDITION_VISIBILITY_ATTEMPTS):
        provider.assert_stable()
        _fresh_reauth(chain, now_unix=now_unix())
        post = provider.inspect_resource(step, plan=chain.plan)
        post.validate()
        if (
            post.state not in {"absent", "unknown"}
            or attempt + 1 == _POSTCONDITION_VISIBILITY_ATTEMPTS
        ):
            return post
        wait(_POSTCONDITION_VISIBILITY_DELAY_SECONDS)
    raise AssertionError("bounded postcondition observation exhausted")


def _rollback_binding_cas_exact(
    step: foundation.PlanStep,
    *,
    before_precondition: Mapping[str, Any] | None,
    operation: OperationObservation,
    post: ResourceObservation,
) -> bool:
    if not step.name.startswith("bind_narrow_"):
        return True
    pre_etag = (
        before_precondition.get("policy_etag")
        if isinstance(before_precondition, Mapping)
        else None
    )
    post_etag = (
        post.precondition.get("policy_etag")
        if isinstance(post.precondition, Mapping)
        else None
    )
    return (
        isinstance(pre_etag, str)
        and bool(pre_etag)
        and isinstance(post_etag, str)
        and bool(post_etag)
        and pre_etag != post_etag
        and operation.cas_precondition_etag == pre_etag
        and operation.cas_postcondition_etag == post_etag
    )


def _rollback_created(
    *,
    provider: FoundationApplyProvider,
    chain: ValidatedFoundationAChain,
    private_key: Ed25519PrivateKey,
    journal: foundation_journal.FoundationApplyJournal,
    transaction_id: str,
    created_steps: Sequence[foundation.PlanStep],
    now_unix: Callable[[], int],
) -> tuple[list[Mapping[str, Any]], bool]:
    receipts: list[Mapping[str, Any]] = []
    partial_unknown = False
    stop_dispatch = False
    for step in reversed(created_steps):
        index = chain.plan.foundation_steps.index(step)
        rollback = _rollback_step_for(step, plan=chain.plan)
        if stop_dispatch:
            receipts.append({
                "original_step_name": step.name,
                "rollback_step_name": rollback.name,
                "rollback_argv_sha256": _sha256_json(list(rollback.argv)),
                "disposition": "not_attempted_manual",
                "operation_receipt_sha256": _sha256_json({
                    "rollback": rollback.name,
                    "state": "not_attempted_manual",
                }),
                "postcondition_receipt_sha256": _sha256_json({
                    "step": step.name,
                    "state": "preserved_for_manual_reconciliation",
                }),
            })
            continue
        post_body = _read_transition(
            journal=journal,
            chain=chain,
            transaction_id=transaction_id,
            name=f"s{index}-rollback-post",
            phase="rollback_post",
            step_index=index,
            step=step,
        )
        operation_body = _read_transition(
            journal=journal,
            chain=chain,
            transaction_id=transaction_id,
            name=f"s{index}-rollback-operation",
            phase="rollback_operation",
            step_index=index,
            step=step,
        )
        intent_body = _read_transition(
            journal=journal,
            chain=chain,
            transaction_id=transaction_id,
            name=f"s{index}-rollback-intent",
            phase="rollback_intent",
            step_index=index,
            step=step,
        )
        operation: OperationObservation
        post: ResourceObservation
        if post_body is not None:
            disposition = str(post_body.get("disposition", ""))
            operation_sha = str(
                post_body.get("operation_receipt_sha256", "")
            )
            post_sha = str(
                post_body.get("postcondition_receipt_sha256", "")
            )
        else:
            try:
                rollback_time = now_unix()
                _fresh_reauth(chain, now_unix=rollback_time)
                _validate_live_ancestry(provider, chain)
                provider.assert_stable()
                before_precondition: Mapping[str, Any] | None = None
                if operation_body is None:
                    if intent_body is not None:
                        raise OwnerGateFoundationApplyError(
                            "owner_gate_foundation_rollback_interrupted_unknown"
                        )
                    before = provider.inspect_resource(step, plan=chain.plan)
                    before.validate()
                    if before.state != "exact":
                        raise OwnerGateFoundationApplyError(
                            "owner_gate_foundation_rollback_precondition_unknown"
                        )
                    before_precondition = before.precondition
                    attempt_id = _sha256_json({
                        "transaction_id": transaction_id,
                        "step_index": index,
                        "phase": "rollback",
                    })
                    _publish_transition(
                        journal=journal,
                        transaction_id=transaction_id,
                        name=f"s{index}-rollback-intent",
                        body=_transition_body(
                            chain=chain,
                            transaction_id=transaction_id,
                            phase="rollback_intent",
                            step_index=index,
                            step=step,
                            payload={
                                "attempt_id": attempt_id,
                                "rollback_step_name": rollback.name,
                                "rollback_argv_sha256": _sha256_json(
                                    list(rollback.argv)
                                ),
                                "precondition_receipt_sha256": (
                                    before.receipt_sha256
                                ),
                                "precondition": dict(
                                    before_precondition or {}
                                ),
                            },
                        ),
                        private_key=private_key,
                    )
                    operation = provider.rollback_step(
                        step,
                        rollback,
                        plan=chain.plan,
                        attempt_id=attempt_id,
                        precondition=before_precondition,
                    )
                    operation.validate()
                    operation_body = _publish_transition(
                        journal=journal,
                        transaction_id=transaction_id,
                        name=f"s{index}-rollback-operation",
                        body=_transition_body(
                            chain=chain,
                            transaction_id=transaction_id,
                            phase="rollback_operation",
                            step_index=index,
                            step=step,
                            payload={
                                "attempt_id": operation.attempt_id,
                                "operation_state": operation.state,
                                "operation_receipt_sha256": (
                                    operation.receipt_sha256
                                ),
                                "provider_result_binding_sha256": (
                                    operation.provider_result_binding_sha256
                                ),
                                "cas_precondition_etag": (
                                    operation.cas_precondition_etag
                                ),
                                "cas_postcondition_etag": (
                                    operation.cas_postcondition_etag
                                ),
                            },
                        ),
                        private_key=private_key,
                    )
                    operation_body = _verify_journal_transition(
                        operation_body,
                        public_key=chain.release_public_key,
                        transaction_id=transaction_id,
                        expected_phase="rollback_operation",
                        expected_step_index=index,
                        expected_step_name=step.name,
                    )
                elif intent_body is None:
                    raise OwnerGateFoundationApplyError(
                        "owner_gate_foundation_rollback_journal_invalid"
                    )
                else:
                    raw_precondition = intent_body.get("precondition", {})
                    if not isinstance(raw_precondition, Mapping):
                        raise OwnerGateFoundationApplyError(
                            "owner_gate_foundation_rollback_journal_invalid"
                        )
                    before_precondition = dict(raw_precondition)
                operation = _operation_from_transition(operation_body)
                post = provider.inspect_resource(step, plan=chain.plan)
                post.validate()
                disposition = (
                    "rolled_back"
                    if (
                        operation.state == "completed"
                        and post.state == "absent"
                        and _rollback_binding_cas_exact(
                            step,
                            before_precondition=before_precondition,
                            operation=operation,
                            post=post,
                        )
                    )
                    else "rollback_failed"
                    if operation.state == "failed"
                    else "rollback_unknown"
                )
                post_body = _publish_transition(
                    journal=journal,
                    transaction_id=transaction_id,
                    name=f"s{index}-rollback-post",
                    body=_transition_body(
                        chain=chain,
                        transaction_id=transaction_id,
                        phase="rollback_post",
                        step_index=index,
                        step=step,
                        payload={
                            "disposition": disposition,
                            "operation_receipt_sha256": (
                                operation.receipt_sha256
                            ),
                            "postcondition_state": post.state,
                            "postcondition_receipt_sha256": (
                                post.receipt_sha256
                            ),
                        },
                    ),
                    private_key=private_key,
                )
                operation_sha = operation.receipt_sha256
                post_sha = post.receipt_sha256
            except Exception:
                disposition = "rollback_unknown"
                operation_sha = _sha256_json({
                    "rollback": rollback.name,
                    "state": "unknown",
                })
                post_sha = _sha256_json({
                    "step": step.name,
                    "state": "unknown",
                })
        if (
            disposition not in _ROLLBACK_DISPOSITIONS
            or _SHA256.fullmatch(operation_sha) is None
            or _SHA256.fullmatch(post_sha) is None
        ):
            disposition = "rollback_unknown"
            partial_unknown = True
        if disposition != "rolled_back":
            partial_unknown = True
            stop_dispatch = True
        receipts.append({
            "original_step_name": step.name,
            "rollback_step_name": rollback.name,
            "rollback_argv_sha256": _sha256_json(list(rollback.argv)),
            "disposition": disposition,
            "operation_receipt_sha256": operation_sha,
            "postcondition_receipt_sha256": post_sha,
        })
    return receipts, partial_unknown


def _failure(
    *,
    chain: ValidatedFoundationAChain,
    private_key: Ed25519PrivateKey,
    provider: FoundationApplyProvider,
    journal: foundation_journal.FoundationApplyJournal,
    transaction_id: str,
    started_at_unix: int,
    failed_at_unix: int,
    failed_step_name: str,
    failure_code: str,
    step_receipts: Sequence[Mapping[str, Any]],
    created_steps: Sequence[foundation.PlanStep],
    inherently_unknown: bool,
    now_unix: Callable[[], int],
) -> FoundationApplyFailed:
    existing = _read_transition(
        journal=journal,
        chain=chain,
        transaction_id=transaction_id,
        name="failure",
        phase="failure",
    )
    if existing is not None and isinstance(existing.get("receipt"), Mapping):
        checked = validate_failure_receipt(
            existing["receipt"],
            public_key=chain.release_public_key,
        )
        if checked.get("transaction_id") != transaction_id:
            _error("owner_gate_foundation_failure_receipt_invalid")
        return FoundationApplyFailed(checked)
    failure_intent_body = _transition_body(
        chain=chain,
        transaction_id=transaction_id,
        phase="failure_intent",
        payload={
            "started_at_unix": started_at_unix,
            "failed_at_unix": max(failed_at_unix, started_at_unix),
            "failed_step_name": failed_step_name,
            "failure_code": failure_code,
            "completed_step_receipts": [
                dict(item) for item in step_receipts
            ],
            "created_step_names": [step.name for step in created_steps],
            "inherently_unknown": inherently_unknown,
        },
    )
    try:
        _publish_transition(
            journal=journal,
            transaction_id=transaction_id,
            name="failure-intent",
            body=failure_intent_body,
            private_key=private_key,
        )
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_foundation_journal_write_failed", exc)
    if inherently_unknown:
        rollbacks = []
        for step in reversed(created_steps):
            rollback = _rollback_step_for(step, plan=chain.plan)
            rollbacks.append({
                "original_step_name": step.name,
                "rollback_step_name": rollback.name,
                "rollback_argv_sha256": _sha256_json(list(rollback.argv)),
                "disposition": "not_attempted_manual",
                "operation_receipt_sha256": _sha256_json({
                    "rollback": rollback.name,
                    "state": "not_attempted_manual",
                }),
                "postcondition_receipt_sha256": _sha256_json({
                    "step": step.name,
                    "state": "preserved_for_manual_reconciliation",
                }),
            })
        rollback_unknown = bool(created_steps)
    else:
        rollbacks, rollback_unknown = _rollback_created(
            provider=provider,
            chain=chain,
            private_key=private_key,
            journal=journal,
            transaction_id=transaction_id,
            created_steps=created_steps,
            now_unix=now_unix,
        )
    partial = inherently_unknown or rollback_unknown
    body = {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "purpose": FAILURE_RECEIPT_PURPOSE,
        "transaction_id": transaction_id,
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "inert_plan_sha256": pre_foundation.inert_plan_sha256(chain.plan),
        "foundation_source_revision": chain.foundation_source_revision,
        "foundation_source_tree_oid": chain.foundation_source_tree_oid,
        "owner_reauthentication_receipt_sha256": (
            chain.owner_reauthentication_receipt_sha256
        ),
        "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
        "ancestry_chain_sha256": chain.authority["ancestry_chain_sha256"],
        "started_at_unix": started_at_unix,
        "failed_at_unix": max(failed_at_unix, started_at_unix),
        "failed_step_name": failed_step_name,
        "failure_code": failure_code,
        "completed_step_receipts": [dict(item) for item in step_receipts],
        "rollback_step_receipts": rollbacks,
        "terminal_state": (
            "manual_reconciliation_required"
            if partial
            else "rolled_back_clean"
        ),
        "partial_unknown_state": partial,
        "mutation_iam_binding_created": False,
        "package_deployed": False,
        "service_started": False,
        "signer_key_id": chain.authority["signer_key_id"],
    }
    receipt = _sign_failure_receipt(body, private_key=private_key)
    try:
        _publish_transition(
            journal=journal,
            transaction_id=transaction_id,
            name="failure",
            body=_transition_body(
                chain=chain,
                transaction_id=transaction_id,
                phase="failure",
                payload={"receipt": receipt},
            ),
            private_key=private_key,
        )
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_foundation_journal_write_failed", exc)
    return FoundationApplyFailed(receipt)


def _preflight_failure_is_proven_nonmutating(
    *,
    journal: foundation_journal.FoundationApplyJournal,
    transaction_id: str,
    failed_step_name: str,
    step_receipts: Sequence[Mapping[str, Any]],
    created_steps: Sequence[foundation.PlanStep],
) -> bool:
    """Admit a clean failure only before any step artifact can exist."""

    if (
        failed_step_name != "preflight_live_ancestry"
        or step_receipts
        or created_steps
    ):
        return False
    try:
        artifacts = journal.list(transaction_id)
    except (OSError, RuntimeError, PermissionError):
        return False
    return frozenset(artifacts) == {"manifest"}


def _apply_with_provider(
    *,
    chain: ValidatedFoundationAChain,
    private_key: Ed25519PrivateKey,
    provider: FoundationApplyProvider,
    journal: foundation_journal.FoundationApplyJournal,
    now_unix: Callable[[], int],
    postcondition_wait: Callable[[float], None] = _no_postcondition_wait,
) -> Mapping[str, Any]:
    """Private seam holding one OS lease for the complete state machine."""
    if (
        type(chain) is not ValidatedFoundationAChain
        or chain._marker is not _CHAIN_MARKER
        or not isinstance(private_key, Ed25519PrivateKey)
        or not isinstance(journal, foundation_journal.FoundationApplyJournal)
        or not callable(postcondition_wait)
    ):
        _error("owner_gate_foundation_apply_boundary_invalid")
    transaction_id = _transaction_id(chain)
    try:
        with journal.transaction_lease(transaction_id):
            return _apply_with_leased_provider(
                chain=chain,
                private_key=private_key,
                provider=provider,
                journal=journal,
                transaction_id=transaction_id,
                now_unix=now_unix,
                postcondition_wait=postcondition_wait,
            )
    except FoundationApplyFailed:
        raise
    except OwnerGateFoundationApplyError:
        raise
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_foundation_journal_lease_failed", exc)


def _apply_with_leased_provider(
    *,
    chain: ValidatedFoundationAChain,
    private_key: Ed25519PrivateKey,
    provider: FoundationApplyProvider,
    journal: foundation_journal.FoundationApplyJournal,
    transaction_id: str,
    now_unix: Callable[[], int],
    postcondition_wait: Callable[[float], None],
) -> Mapping[str, Any]:
    started_now = now_unix()
    if type(started_now) is not int or started_now <= 0:
        _error("owner_gate_foundation_apply_time_invalid")
    manifest = _read_transition(
        journal=journal,
        chain=chain,
        transaction_id=transaction_id,
        name="manifest",
        phase="manifest",
    )
    if manifest is None:
        manifest = _transition_body(
            chain=chain,
            transaction_id=transaction_id,
            phase="manifest",
            payload={
                "started_at_unix": started_now,
                "owner_reauthentication_receipt_sha256": (
                    chain.owner_reauthentication_receipt_sha256
                ),
                "ancestry_evidence_sha256": chain.ancestry_evidence_sha256,
                "signed_network_evidence_sha256": (
                    chain.signed_network_evidence_sha256
                ),
                "steps": [
                    {
                        "step_index": index,
                        "step_name": step.name,
                        "argv_sha256": _sha256_json(list(step.argv)),
                    }
                    for index, step in enumerate(chain.plan.foundation_steps)
                ],
            },
        )
        _publish_transition(
            journal=journal,
            transaction_id=transaction_id,
            name="manifest",
            body=manifest,
            private_key=private_key,
        )
    started = manifest.get("started_at_unix")
    if type(started) is not int or started <= 0:
        _error("owner_gate_foundation_journal_transition_invalid")
    success = _read_transition(
        journal=journal,
        chain=chain,
        transaction_id=transaction_id,
        name="success",
        phase="success",
    )
    failed = _read_transition(
        journal=journal,
        chain=chain,
        transaction_id=transaction_id,
        name="failure",
        phase="failure",
    )
    failure_intent = _read_transition(
        journal=journal,
        chain=chain,
        transaction_id=transaction_id,
        name="failure-intent",
        phase="failure_intent",
    )
    if success is not None and (
        failed is not None or failure_intent is not None
    ):
        _error("owner_gate_foundation_terminal_journal_invalid")
    intent_values = (
        _validate_failure_intent_transition(
            failure_intent,
            chain=chain,
            transaction_id=transaction_id,
            started_at_unix=started,
        )
        if failure_intent is not None
        else None
    )
    if failed is not None and isinstance(failed.get("receipt"), Mapping):
        if intent_values is None:
            _error("owner_gate_foundation_failure_intent_missing")
        checked_failure = validate_failure_receipt(
            failed["receipt"],
            public_key=chain.release_public_key,
        )
        (
            intent_completed,
            intent_created,
            intent_failed_at,
            intent_failed_step,
            intent_failure_code,
            _intent_unknown,
        ) = intent_values
        if (
            checked_failure.get("transaction_id") != transaction_id
            or checked_failure.get("started_at_unix") != started
            or checked_failure.get("failed_at_unix") != intent_failed_at
            or checked_failure.get("failed_step_name") != intent_failed_step
            or checked_failure.get("failure_code") != intent_failure_code
            or checked_failure.get("completed_step_receipts")
            != intent_completed
            or [
                item["step_name"]
                for item in intent_completed
                if item["disposition"] == "created"
            ]
            != [step.name for step in intent_created]
        ):
            _error("owner_gate_foundation_failure_receipt_invalid")
        raise FoundationApplyFailed(checked_failure)
    artifacts = journal.list(transaction_id)
    rollback_artifacts = sorted(
        name for name in artifacts if "-rollback-" in name
    )
    if failure_intent is None and rollback_artifacts:
        _error("owner_gate_foundation_rollback_journal_orphaned")
    if intent_values is not None:
        (
            completed,
            created_steps_from_intent,
            failed_intent,
            failed_step,
            failure_code,
            inherently_unknown,
        ) = intent_values
        raise _failure(
            chain=chain,
            private_key=private_key,
            provider=provider,
            journal=journal,
            transaction_id=transaction_id,
            started_at_unix=started,
            failed_at_unix=failed_intent,
            failed_step_name=failed_step,
            failure_code=failure_code,
            step_receipts=completed,
            created_steps=created_steps_from_intent,
            inherently_unknown=inherently_unknown,
            now_unix=now_unix,
        )
    if success is not None and isinstance(success.get("receipt"), Mapping):
        provider.assert_stable()
        _fresh_reauth(chain, now_unix=started_now)
        _validate_live_ancestry(provider, chain)
        for index, step in enumerate(chain.plan.foundation_steps):
            preexisting = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-preexisting",
                phase="preexisting_exact",
                step_index=index,
                step=step,
            )
            post_body = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-post",
                phase="postcondition_exact",
                step_index=index,
                step=step,
            )
            if (preexisting is None) == (post_body is None):
                _error("owner_gate_foundation_terminal_journal_invalid")
            terminal_transition = (
                preexisting if preexisting is not None else post_body
            )
            if terminal_transition is None:
                _error("owner_gate_foundation_terminal_journal_invalid")
            _require_live_journaled_exact(
                provider=provider,
                chain=chain,
                step=step,
                transition=terminal_transition,
            )
        return pre_foundation.validate_foundation_apply_receipt(
            success["receipt"],
            public_key=chain.release_public_key,
            authority=chain.authority,
            owner_reauthentication_receipt=chain.owner_reauthentication_receipt,
            project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
            project_ancestry_collector_public_key=(
                chain.ancestry_collector_public_key
            ),
            plan=chain.plan,
            now_unix=started_now,
        )
    created_steps: list[foundation.PlanStep] = []
    step_receipts: list[Mapping[str, Any]] = []
    current_step = "preflight_live_ancestry"
    try:
        provider.assert_stable()
        _fresh_reauth(chain, now_unix=started_now)
        _validate_live_ancestry(provider, chain)
        for index, step in enumerate(chain.plan.foundation_steps):
            current_step = step.name
            current_time = now_unix()
            _fresh_reauth(chain, now_unix=current_time)
            provider.assert_stable()
            preexisting = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-preexisting",
                phase="preexisting_exact",
                step_index=index,
                step=step,
            )
            post_body = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-post",
                phase="postcondition_exact",
                step_index=index,
                step=step,
            )
            if preexisting is not None:
                _require_live_journaled_exact(
                    provider=provider,
                    chain=chain,
                    step=step,
                    transition=preexisting,
                )
                step_receipts.append(_step_receipt_from_transition(
                    preexisting,
                    step=step,
                    plan=chain.plan,
                    disposition="preexisting_exact",
                ))
                continue
            if post_body is not None:
                _require_live_journaled_exact(
                    provider=provider,
                    chain=chain,
                    step=step,
                    transition=post_body,
                )
                step_receipts.append(_step_receipt_from_transition(
                    post_body,
                    step=step,
                    plan=chain.plan,
                    disposition="created",
                ))
                created_steps.append(step)
                continue
            intent = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-intent",
                phase="mutation_intent",
                step_index=index,
                step=step,
            )
            operation_body = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-operation",
                phase="operation_observed",
                step_index=index,
                step=step,
            )
            pre_body = _read_transition(
                journal=journal,
                chain=chain,
                transaction_id=transaction_id,
                name=f"s{index}-pre",
                phase="precondition_absent",
                step_index=index,
                step=step,
            )
            before_precondition: Mapping[str, Any] | None = None
            if pre_body is None and intent is None and operation_body is None:
                before = provider.inspect_resource(step, plan=chain.plan)
                before.validate()
                if before.state in {"drift", "unknown"}:
                    raise _failure(
                        chain=chain,
                        private_key=private_key,
                        provider=provider,
                        journal=journal,
                        transaction_id=transaction_id,
                        started_at_unix=started,
                        failed_at_unix=current_time,
                        failed_step_name=step.name,
                        failure_code=(
                            "owner_gate_foundation_preexisting_drift"
                            if before.state == "drift"
                            else "owner_gate_foundation_preexisting_unknown"
                        ),
                        step_receipts=step_receipts,
                        created_steps=created_steps,
                        inherently_unknown=True,
                        now_unix=now_unix,
                    )
                if before.state == "exact":
                    body = _transition_body(
                        chain=chain,
                        transaction_id=transaction_id,
                        phase="preexisting_exact",
                        step_index=index,
                        step=step,
                        payload={
                            "postcondition_receipt_sha256": (
                                before.receipt_sha256
                            ),
                            "resource_identity": dict(
                                before.resource_identity or {}
                            ),
                        },
                    )
                    signed = _publish_transition(
                        journal=journal,
                        transaction_id=transaction_id,
                        name=f"s{index}-preexisting",
                        body=body,
                        private_key=private_key,
                    )
                    checked = _verify_journal_transition(
                        signed,
                        public_key=chain.release_public_key,
                        transaction_id=transaction_id,
                        expected_phase="preexisting_exact",
                        expected_step_index=index,
                        expected_step_name=step.name,
                    )
                    step_receipts.append(_step_receipt_from_transition(
                        checked,
                        step=step,
                        plan=chain.plan,
                        disposition="preexisting_exact",
                    ))
                    continue
                before_precondition = before.precondition
                pre_body = _transition_body(
                    chain=chain,
                    transaction_id=transaction_id,
                    phase="precondition_absent",
                    step_index=index,
                    step=step,
                    payload={
                        "precondition_receipt_sha256": before.receipt_sha256,
                        "precondition": dict(before.precondition or {}),
                    },
                )
                _publish_transition(
                    journal=journal,
                    transaction_id=transaction_id,
                    name=f"s{index}-pre",
                    body=pre_body,
                    private_key=private_key,
                )
            elif pre_body is not None:
                raw_precondition = pre_body.get("precondition", {})
                if not isinstance(raw_precondition, Mapping):
                    _error("owner_gate_foundation_journal_transition_invalid")
                before_precondition = dict(raw_precondition)
            if intent is not None and operation_body is None:
                provider.inspect_resource(step, plan=chain.plan).validate()
                raise _failure(
                    chain=chain,
                    private_key=private_key,
                    provider=provider,
                    journal=journal,
                    transaction_id=transaction_id,
                    started_at_unix=started,
                    failed_at_unix=now_unix(),
                    failed_step_name=step.name,
                    failure_code=(
                        "owner_gate_foundation_interrupted_operation_unknown"
                    ),
                    step_receipts=step_receipts,
                    created_steps=created_steps,
                    inherently_unknown=True,
                    now_unix=now_unix,
                )
            if operation_body is None:
                if pre_body is None:
                    _error("owner_gate_foundation_journal_transition_invalid")
                attempt_id = _sha256_json({
                    "transaction_id": transaction_id,
                    "step_index": index,
                    "phase": "create",
                })
                _publish_transition(
                    journal=journal,
                    transaction_id=transaction_id,
                    name=f"s{index}-intent",
                    body=_transition_body(
                        chain=chain,
                        transaction_id=transaction_id,
                        phase="mutation_intent",
                        step_index=index,
                        step=step,
                        payload={
                            "attempt_id": attempt_id,
                            "precondition_receipt_sha256": pre_body.get(
                                "precondition_receipt_sha256"
                            ),
                            "precondition": dict(
                                before_precondition or {}
                            ),
                        },
                    ),
                    private_key=private_key,
                )
                operation = provider.execute_step(
                    step,
                    plan=chain.plan,
                    attempt_id=attempt_id,
                    precondition=before_precondition,
                )
                operation.validate()
                operation_body = _transition_body(
                    chain=chain,
                    transaction_id=transaction_id,
                    phase="operation_observed",
                    step_index=index,
                    step=step,
                    payload={
                        "attempt_id": operation.attempt_id,
                        "operation_state": operation.state,
                        "operation_receipt_sha256": (
                            operation.receipt_sha256
                        ),
                        "provider_result_binding_sha256": (
                            operation.provider_result_binding_sha256
                        ),
                        "cas_precondition_etag": (
                            operation.cas_precondition_etag
                        ),
                        "cas_postcondition_etag": (
                            operation.cas_postcondition_etag
                        ),
                    },
                )
                _publish_transition(
                    journal=journal,
                    transaction_id=transaction_id,
                    name=f"s{index}-operation",
                    body=operation_body,
                    private_key=private_key,
                )
            operation = _operation_from_transition(operation_body)
            post = _observe_completed_postcondition(
                provider=provider,
                chain=chain,
                step=step,
                operation=operation,
                now_unix=now_unix,
                wait=postcondition_wait,
            )
            if (
                operation.state != "completed"
                or post.state != "exact"
                or not _binding_cas_exact(
                    step,
                    before_precondition=before_precondition,
                    operation=operation,
                    post=post,
                )
            ):
                raise _failure(
                    chain=chain,
                    private_key=private_key,
                    provider=provider,
                    journal=journal,
                    transaction_id=transaction_id,
                    started_at_unix=started,
                    failed_at_unix=now_unix(),
                    failed_step_name=step.name,
                    failure_code=(
                        "owner_gate_foundation_operation_not_completed"
                        if operation.state != "completed"
                        else "owner_gate_foundation_binding_cas_unproven"
                        if step.name.startswith("bind_narrow_")
                        else "owner_gate_foundation_postcondition_not_exact"
                    ),
                    step_receipts=step_receipts,
                    created_steps=created_steps,
                    inherently_unknown=(
                        operation.state == "unknown"
                        or operation.state == "completed"
                        or post.state != "absent"
                        or step.name.startswith("bind_narrow_")
                    ),
                    now_unix=now_unix,
                )
            post_body = _transition_body(
                chain=chain,
                transaction_id=transaction_id,
                phase="postcondition_exact",
                step_index=index,
                step=step,
                payload={
                    "attempt_id": operation.attempt_id,
                    "provider_result_binding_sha256": (
                        operation.provider_result_binding_sha256
                    ),
                    "operation_receipt_sha256": operation.receipt_sha256,
                    "postcondition_receipt_sha256": post.receipt_sha256,
                    "resource_identity": dict(post.resource_identity or {}),
                },
            )
            _publish_transition(
                journal=journal,
                transaction_id=transaction_id,
                name=f"s{index}-post",
                body=post_body,
                private_key=private_key,
            )
            step_receipts.append(_step_receipt_from_transition(
                post_body,
                step=step,
                plan=chain.plan,
                disposition="created",
            ))
            created_steps.append(step)
        current_step = "terminal_live_ancestry"
        completed = now_unix()
        _fresh_reauth(chain, now_unix=completed)
        _validate_live_ancestry(provider, chain)
        provider.assert_stable()
        execution = _ProviderExecutionResult(
            step_receipts=tuple(step_receipts),
            started_at_unix=started,
            completed_at_unix=completed,
        )
        receipt = pre_foundation._sign_foundation_apply_execution(
            execution,
            private_key=private_key,
            authority=chain.authority,
            owner_reauthentication_receipt=(
                chain.owner_reauthentication_receipt
            ),
            project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
            project_ancestry_collector_public_key=(
                chain.ancestry_collector_public_key
            ),
            plan=chain.plan,
        )
        _publish_transition(
            journal=journal,
            transaction_id=transaction_id,
            name="success",
            body=_transition_body(
                chain=chain,
                transaction_id=transaction_id,
                phase="success",
                payload={"receipt": receipt},
            ),
            private_key=private_key,
        )
        return receipt
    except FoundationApplyFailed:
        raise
    except OwnerGateFoundationApplyError as exc:
        raise _failure(
            chain=chain,
            private_key=private_key,
            provider=provider,
            journal=journal,
            transaction_id=transaction_id,
            started_at_unix=started,
            failed_at_unix=now_unix(),
            failed_step_name=current_step,
            failure_code=str(exc),
            step_receipts=step_receipts,
            created_steps=created_steps,
            inherently_unknown=not _preflight_failure_is_proven_nonmutating(
                journal=journal,
                transaction_id=transaction_id,
                failed_step_name=current_step,
                step_receipts=step_receipts,
                created_steps=created_steps,
            ),
            now_unix=now_unix,
        ) from None
    except Exception as exc:
        raise _failure(
            chain=chain,
            private_key=private_key,
            provider=provider,
            journal=journal,
            transaction_id=transaction_id,
            started_at_unix=started,
            failed_at_unix=now_unix(),
            failed_step_name=current_step,
            failure_code="owner_gate_foundation_provider_unknown",
            step_receipts=step_receipts,
            created_steps=created_steps,
            inherently_unknown=not _preflight_failure_is_proven_nonmutating(
                journal=journal,
                transaction_id=transaction_id,
                failed_step_name=current_step,
                step_receipts=step_receipts,
                created_steps=created_steps,
            ),
            now_unix=now_unix,
        ) from None


@dataclass(frozen=True)
class _CapturedCommand:
    returncode: int
    stdout: bytes
    stderr: bytes
    transport_unknown: bool = False


class _SubprocessFoundationRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> _CapturedCommand:
        try:
            completed = subprocess.run(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return _CapturedCommand(-1, b"", b"", transport_unknown=True)
        return _CapturedCommand(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )


def _canonical_inventory(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_inventory(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        normalized = [_canonical_inventory(item) for item in value]
        return sorted(normalized, key=_canonical)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    _error("owner_gate_foundation_provider_json_invalid")


@dataclass(frozen=True)
class _IamPolicyHttpResponse:
    status: int | None
    body: bytes
    transport_unknown: bool = False


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


def _strict_provider_json(raw: bytes) -> Any:
    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate json key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite json constant")

    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _resource_manager_set_iam_policy(
    token: str,
    resource_name: str,
    policy: Mapping[str, Any],
) -> _IamPolicyHttpResponse:
    """Send one no-retry Resource Manager v3 IAM CAS request."""

    if (
        not isinstance(token, str)
        or not token
        or len(token) > 16 * 1024
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in token)
        or _IAM_RESOURCE.fullmatch(resource_name) is None
        or not isinstance(policy, Mapping)
    ):
        _error("owner_gate_foundation_iam_cas_request_invalid")
    encoded = _canonical({
        "policy": dict(policy),
        "updateMask": "bindings,etag",
    })
    if len(encoded) > MAX_JSON_BYTES:
        _error("owner_gate_foundation_iam_cas_request_invalid")
    url = (
        "https://cloudresourcemanager.googleapis.com/v3/"
        f"{resource_name}:setIamPolicy"
    )
    try:
        if any(
            os.environ.get(name) for name in _FORBIDDEN_NETWORK_ENVIRONMENT
        ):
            _error("owner_gate_foundation_iam_cas_tls_invalid")
        launcher._reject_custom_ca_environment()
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(
                context=launcher._pinned_system_tls_context()
            ),
            _NoRedirectHandler(),
        )
        with opener.open(request, timeout=30.0) as response:
            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")
            if (
                response.geturl() != url
                or content_type.split(";", 1)[0].strip().casefold()
                != "application/json"
                or (
                    content_length is not None
                    and (
                        not content_length.isdecimal()
                        or int(content_length) > MAX_JSON_BYTES
                    )
                )
            ):
                return _IamPolicyHttpResponse(
                    int(response.status),
                    b"",
                    transport_unknown=True,
                )
            raw = response.read(MAX_JSON_BYTES + 1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        try:
            return _IamPolicyHttpResponse(int(exc.code), b"")
        finally:
            exc.close()
    except (OSError, TimeoutError, urllib.error.URLError, launcher.OwnerLauncherError):
        return _IamPolicyHttpResponse(None, b"", transport_unknown=True)
    if not raw or len(raw) > MAX_JSON_BYTES:
        return _IamPolicyHttpResponse(status, b"", transport_unknown=True)
    return _IamPolicyHttpResponse(status, raw)


def _normalize_iam_policy(
    value: Any,
    *,
    resource_name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or _IAM_RESOURCE.fullmatch(resource_name) is None
        or not set(value).issubset({"bindings", "auditConfigs", "etag", "version"})
    ):
        _error("owner_gate_foundation_iam_policy_invalid")
    etag = value.get("etag")
    version = value.get("version", 1)
    bindings_raw = value.get("bindings", [])
    audit_raw = value.get("auditConfigs", [])
    if (
        not isinstance(etag, str)
        or not etag
        or len(etag) > 4096
        or type(version) is not int
        or version not in {1, 3}
        or not isinstance(bindings_raw, list)
        or len(bindings_raw) > 20_000
        or not isinstance(audit_raw, list)
        or len(audit_raw) > 20_000
    ):
        _error("owner_gate_foundation_iam_policy_invalid")
    for binding in bindings_raw:
        if (
            not isinstance(binding, Mapping)
            or not set(binding).issubset({"role", "members", "condition"})
            or not isinstance(binding.get("role"), str)
            or not binding.get("role")
            or len(str(binding.get("role"))) > 1024
            or not isinstance(binding.get("members", []), list)
            or len(binding.get("members", [])) > 20_000
            or any(
                not isinstance(member, str)
                or not member
                or len(member) > 4096
                for member in binding.get("members", [])
            )
            or (
                "condition" in binding
                and not isinstance(binding.get("condition"), Mapping)
            )
        ):
            _error("owner_gate_foundation_iam_policy_invalid")
    bindings = _canonical_inventory(bindings_raw)
    audit_configs = _canonical_inventory(audit_raw)
    if not isinstance(bindings, list) or not isinstance(audit_configs, list):
        _error("owner_gate_foundation_iam_policy_invalid")
    return {
        "resource_name": resource_name,
        "policy_etag": etag,
        "policy_version": version,
        "policy_bindings": bindings,
        "policy_audit_configs": audit_configs,
    }


def _policy_from_precondition(
    value: Mapping[str, Any] | None,
    *,
    resource_name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or frozenset(value)
        != frozenset({
            "resource_name",
            "policy_etag",
            "policy_version",
            "policy_bindings",
            "policy_audit_configs",
        })
        or value.get("resource_name") != resource_name
    ):
        _error("owner_gate_foundation_iam_cas_precondition_invalid")
    normalized = _normalize_iam_policy(
        {
            "etag": value.get("policy_etag"),
            "version": value.get("policy_version"),
            "bindings": value.get("policy_bindings"),
            "auditConfigs": value.get("policy_audit_configs"),
        },
        resource_name=resource_name,
    )
    if dict(normalized) != dict(value):
        _error("owner_gate_foundation_iam_cas_precondition_invalid")
    return {
        "etag": normalized["policy_etag"],
        "version": normalized["policy_version"],
        "bindings": normalized["policy_bindings"],
        "auditConfigs": normalized["policy_audit_configs"],
    }


def _iam_binding_contract(
    step: foundation.PlanStep,
    *,
    plan: foundation.OwnerGateFoundationPlan,
) -> tuple[str, str, str] | None:
    member = f"serviceAccount:{plan.spec.service_account_email}"
    contracts = {
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account": (
            f"projects/{plan.spec.project}",
            plan.spec.read_only_iam_role,
            member,
        ),
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account": (
            plan.spec.organization_resource,
            plan.spec.ancestor_read_only_iam_role,
            member,
        ),
    }
    contract = contracts.get(step.name)
    if contract is None:
        return None
    resource_name, role, expected_member = contract
    if step.argv != (
        "owner-gate-provider",
        "set-iam-binding-cas",
        resource_name,
        f"--member={expected_member}",
        f"--role={role}",
        "--condition=None",
    ):
        _error("owner_gate_foundation_provider_step_forbidden")
    return contract


def _edited_iam_policy(
    precondition: Mapping[str, Any] | None,
    *,
    resource_name: str,
    role: str,
    member: str,
    add: bool,
) -> Mapping[str, Any]:
    policy = _policy_from_precondition(
        precondition,
        resource_name=resource_name,
    )
    bindings = [dict(item) for item in policy["bindings"]]
    matching = [item for item in bindings if item.get("role") == role]
    if add:
        if matching:
            _error("owner_gate_foundation_iam_cas_precondition_invalid")
        bindings.append({"role": role, "members": [member]})
    else:
        if (
            len(matching) != 1
            or matching[0].get("members") != [member]
            or matching[0].get("condition") is not None
        ):
            _error("owner_gate_foundation_iam_cas_precondition_invalid")
        bindings.remove(matching[0])
    normalized_bindings = _canonical_inventory(bindings)
    if not isinstance(normalized_bindings, list):
        _error("owner_gate_foundation_iam_cas_precondition_invalid")
    return {
        "etag": policy["etag"],
        "version": policy["version"],
        "bindings": normalized_bindings,
        "auditConfigs": policy["auditConfigs"],
    }


def _provider_link(value: Any) -> str:
    if not isinstance(value, str) or not value:
        _error("owner_gate_foundation_provider_resource_invalid")
    if value.startswith("https://www.googleapis.com/compute/v1/"):
        return value
    if value.startswith("projects/"):
        return "https://www.googleapis.com/compute/v1/" + value
    _error("owner_gate_foundation_provider_resource_invalid")


def _provider_link_equals(value: Any, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value
    if value.startswith("projects/"):
        normalized = "https://www.googleapis.com/compute/v1/" + value
    return normalized == expected


def _provider_tag(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        _error("owner_gate_foundation_provider_resource_invalid")
    return value


def _list(value: Any) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) > 20_000
        or any(not isinstance(item, Mapping) for item in value)
    ):
        _error("owner_gate_foundation_provider_json_invalid")
    return [dict(item) for item in value]


class _TrustedGcloudFoundationProvider:
    """Concrete fixed-command provider behind the exact public boundary."""

    def __init__(
        self,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        gcloud_executable: launcher.TrustedGcloudExecutable,
        gcloud_configuration: launcher.PinnedGcloudConfiguration,
        expected_release_revision: str,
        runner: _SubprocessFoundationRunner,
    ) -> None:
        self._plan_sha256 = pre_foundation.inert_plan_sha256(plan)
        self._plan = plan
        self._gcloud_executable = gcloud_executable
        self._gcloud_configuration = gcloud_configuration
        self._release_revision = expected_release_revision
        self._runner = runner
        try:
            identity, prefix, environment = owner_reauth._trusted_snapshot(
                gcloud_executable,
                gcloud_configuration,
            )
            sealed = owner_reauth._validate_sealed_runtime_identity(
                gcloud_executable.sealed_runtime_identity(
                    expected_release_sha=expected_release_revision,
                ),
                expected_release_revision=expected_release_revision,
                prefix=prefix,
            )
        except (launcher.OwnerLauncherError, owner_reauth.OwnerGateOwnerReauthError) as exc:
            _error("owner_gate_foundation_provider_runtime_invalid", exc)
        environment = dict(environment)
        environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        self._identity = identity
        self._sealed = sealed
        self._prefix = prefix
        self._environment = environment
        self._token_provider = launcher.GcloudOwnerAccessToken(
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
        )
        try:
            self._token_provider.bind_approved_subject(
                _sha256_bytes(owner_reauth.OWNER_ACCOUNT.encode("utf-8"))
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_foundation_provider_runtime_invalid", exc)

    def assert_stable(self) -> None:
        try:
            identity, prefix, environment = owner_reauth._trusted_snapshot(
                self._gcloud_executable,
                self._gcloud_configuration,
            )
            sealed = owner_reauth._validate_sealed_runtime_identity(
                self._gcloud_executable.sealed_runtime_identity(
                    expected_release_sha=self._release_revision,
                ),
                expected_release_revision=self._release_revision,
                prefix=prefix,
            )
            self._token_provider.require_stable()
        except (launcher.OwnerLauncherError, owner_reauth.OwnerGateOwnerReauthError) as exc:
            _error("owner_gate_foundation_provider_runtime_changed", exc)
        environment = dict(environment)
        environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        if (
            identity != self._identity
            or sealed != self._sealed
            or prefix != self._prefix
            or environment != self._environment
            or pre_foundation.inert_plan_sha256(self._plan)
            != self._plan_sha256
        ):
            _error("owner_gate_foundation_provider_runtime_changed")

    def _full_argv(self, logical: Sequence[str]) -> tuple[str, ...]:
        if not logical or logical[0] != "gcloud":
            _error("owner_gate_foundation_provider_command_forbidden")
        return (
            *self._prefix,
            *tuple(logical)[1:],
            f"--account={owner_reauth.OWNER_ACCOUNT}",
            f"--configuration={owner_reauth.GCLOUD_CONFIGURATION}",
        )

    @staticmethod
    def _capture_receipt(
        logical: Sequence[str],
        result: _CapturedCommand,
    ) -> str:
        return _sha256_json({
            "logical_argv_sha256": _sha256_json(list(logical)),
            "returncode": result.returncode,
            "stdout_sha256": _sha256_bytes(result.stdout),
            "stderr_sha256": _sha256_bytes(result.stderr),
            "transport_unknown": result.transport_unknown,
        })

    def _run(self, logical: Sequence[str]) -> _CapturedCommand:
        result = self._runner.run(
            self._full_argv(logical),
            env=self._environment,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        )
        if (
            type(result.stdout) is not bytes
            or type(result.stderr) is not bytes
            or len(result.stdout) > MAX_JSON_BYTES
            or len(result.stderr) > MAX_JSON_BYTES
        ):
            return _CapturedCommand(-1, b"", b"", transport_unknown=True)
        return result

    def _read_json(self, logical: Sequence[str]) -> tuple[Any, str]:
        result = self._run(logical)
        receipt = self._capture_receipt(logical, result)
        if result.transport_unknown or result.returncode != 0 or not result.stdout:
            _error("owner_gate_foundation_provider_read_unknown")
        try:
            value = _strict_provider_json(result.stdout)
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            _error("owner_gate_foundation_provider_read_unknown", exc)
        return value, receipt

    def _request_iam_policy_cas(
        self,
        *,
        resource_name: str,
        policy: Mapping[str, Any],
    ) -> _IamPolicyHttpResponse:
        self.assert_stable()
        token = self._token_provider()
        try:
            self._token_provider.require_stable()
            return _resource_manager_set_iam_policy(
                token,
                resource_name,
                policy,
            )
        finally:
            token = ""
            self._token_provider.require_stable()

    def _iam_binding_operation(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
        add: bool,
        logical_argv: Sequence[str],
    ) -> OperationObservation:
        contract = _iam_binding_contract(step, plan=plan)
        if contract is None:
            _error("owner_gate_foundation_provider_step_forbidden")
        resource_name, role, member = contract
        policy = _edited_iam_policy(
            precondition,
            resource_name=resource_name,
            role=role,
            member=member,
            add=add,
        )
        response = self._request_iam_policy_cas(
            resource_name=resource_name,
            policy=policy,
        )
        receipt = _sha256_json({
            "logical_argv_sha256": _sha256_json(list(logical_argv)),
            "resource_name": resource_name,
            "attempt_id": attempt_id,
            "request_policy_sha256": _sha256_json(policy),
            "update_mask": "bindings,etag",
            "http_status": response.status,
            "response_sha256": _sha256_bytes(response.body),
            "transport_unknown": response.transport_unknown,
        })
        pre_etag = str(policy["etag"])
        if (
            response.transport_unknown
            or response.status is None
            or response.status in {409, 412}
            or 300 <= response.status < 400
            or response.status < 200
            or response.status >= 500
            or response.status in {408, 425, 429}
        ):
            return OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        if response.status < 200 or response.status >= 300:
            return OperationObservation(
                "failed",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        try:
            output = _strict_provider_json(response.body)
            normalized = _normalize_iam_policy(
                output,
                resource_name=resource_name,
            )
        except (
            OwnerGateFoundationApplyError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        post_etag = normalized["policy_etag"]
        exact = (
            isinstance(post_etag, str)
            and bool(post_etag)
            and post_etag != pre_etag
            and normalized["policy_version"] == policy["version"]
            and normalized["policy_bindings"] == policy["bindings"]
            and normalized["policy_audit_configs"] == policy["auditConfigs"]
        )
        if not exact:
            return OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
                cas_postcondition_etag=(
                    str(post_etag) if isinstance(post_etag, str) else None
                ),
            )
        result_binding = _sha256_json({
            "attempt_id": attempt_id,
            "resource_name": resource_name,
            "request_policy_sha256": _sha256_json(policy),
            "response_policy_sha256": _sha256_json(normalized),
            "operation_receipt_sha256": receipt,
        })
        return OperationObservation(
            "completed",
            receipt,
            attempt_id,
            result_binding,
            pre_etag,
            str(post_etag),
        )

    def observe_ancestry_chain(self) -> Sequence[Mapping[str, Any]]:
        token = self._token_provider()
        try:
            project_number = str(self._plan.spec.organization_id)
            # The real project number is not a plan field; recover it from the
            # validated authority-bound token probe through the exact project
            # resource read.  The caller compares the full resulting chain.
            project_raw, _receipt = self._read_json((
                "gcloud",
                "projects",
                "describe",
                foundation.PROJECT,
                "--format=json",
            ))
            project_number = str(project_raw.get("projectNumber", ""))
            if _NUMERIC_ID.fullmatch(project_number) is None:
                _error("owner_gate_foundation_live_ancestry_unknown")

            def read_resource(resource_name: str) -> Mapping[str, Any]:
                return project_ancestry._resource_manager_get(
                    token,
                    resource_name,
                )

            first = project_ancestry._read_chain(
                read_resource,
                project_id=foundation.PROJECT,
                project_number=project_number,
            )
            second = project_ancestry._read_chain(
                read_resource,
                project_id=foundation.PROJECT,
                project_number=project_number,
            )
            if first != second:
                _error("owner_gate_foundation_live_ancestry_unknown")
            return first
        finally:
            token = ""
            self._token_provider.require_stable()

    def _read_once(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
    ) -> tuple[
        str,
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        list[str],
    ]:
        spec = plan.spec
        project = f"--project={spec.project}"
        receipts: list[str] = []
        precondition: Mapping[str, Any] | None = None

        def read(command: tuple[str, ...]) -> Any:
            value, receipt = self._read_json(command)
            receipts.append(receipt)
            return value

        try:
            if step.name == "create_dedicated_service_account":
                accounts = _list(read((
                    "gcloud",
                    "iam",
                    "service-accounts",
                    "list",
                    project,
                    "--format=json",
                )))
                matches = [
                    item
                    for item in accounts
                    if item.get("email") == spec.service_account_email
                ]
                if not matches:
                    return "absent", None, None, receipts
                if len(matches) != 1:
                    return "drift", None, None, receipts
                account = matches[0]
                keys = _list(read((
                    "gcloud",
                    "iam",
                    "service-accounts",
                    "keys",
                    "list",
                    f"--iam-account={spec.service_account_email}",
                    "--managed-by=user",
                    project,
                    "--format=json",
                )))
                identity: Mapping[str, Any] = {
                    "resource_type": "iam_service_account",
                    "resource_name": account.get("name"),
                    "email": account.get("email"),
                    "unique_id": str(account.get("uniqueId", "")),
                    "etag": account.get("etag"),
                    "disabled": account.get("disabled", False),
                    "user_managed_key_count": len(keys),
                    "user_managed_keys": sorted(
                        str(item.get("name")) for item in keys
                    ),
                }
            elif step.name in {
                "create_narrow_iam_observation_reader_role",
                "create_narrow_storage_executor_role",
                "create_narrow_organization_iam_observation_reader_role",
            }:
                org = step.name.startswith("create_narrow_organization")
                role_id = (
                    foundation.ANCESTOR_READ_ONLY_IAM_ROLE_ID
                    if org
                    else foundation.READ_ONLY_IAM_ROLE_ID
                    if step.name == "create_narrow_iam_observation_reader_role"
                    else foundation.MUTATION_ROLE_ID
                )
                scope = (
                    f"--organization={spec.organization_id}"
                    if org
                    else project
                )
                roles = _list(read((
                    "gcloud",
                    "iam",
                    "roles",
                    "list",
                    scope,
                    "--show-deleted",
                    "--format=json",
                )))
                role_name = (
                    spec.ancestor_read_only_iam_role
                    if org
                    else spec.read_only_iam_role
                    if role_id == foundation.READ_ONLY_IAM_ROLE_ID
                    else spec.custom_role
                )
                matches = [item for item in roles if item.get("name") == role_name]
                if not matches:
                    return "absent", None, None, receipts
                if len(matches) != 1:
                    return "drift", None, None, receipts
                role = read((
                    "gcloud",
                    "iam",
                    "roles",
                    "describe",
                    role_id,
                    scope,
                    "--format=json",
                ))
                if not isinstance(role, Mapping):
                    return "unknown", None, None, receipts
                identity = {
                    "resource_type": (
                        "organization_custom_role" if org else "project_custom_role"
                    ),
                    "name": role.get("name"),
                    "etag": role.get("etag"),
                    "title": role.get("title"),
                    "description": role.get("description"),
                    "stage": role.get("stage"),
                    "included_permissions": role.get("includedPermissions"),
                    "deleted": role.get("deleted", False),
                }
            elif step.name in {
                "bind_narrow_iam_observation_reader_to_owner_gate_service_account",
                "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account",
            }:
                org = step.name.startswith("bind_narrow_organization")
                role = (
                    spec.ancestor_read_only_iam_role
                    if org
                    else spec.read_only_iam_role
                )
                command = (
                    (
                        "gcloud",
                        "organizations",
                        "get-iam-policy",
                        spec.organization_id,
                        "--format=json",
                    )
                    if org
                    else (
                        "gcloud",
                        "projects",
                        "get-iam-policy",
                        spec.project,
                        "--format=json",
                    )
                )
                policy = read(command)
                if not isinstance(policy, Mapping):
                    return "unknown", None, None, receipts
                bindings = policy.get("bindings", [])
                if not isinstance(bindings, list):
                    return "unknown", None, None, receipts
                resource_name = (
                    spec.organization_resource
                    if org
                    else f"projects/{spec.project}"
                )
                precondition = _normalize_iam_policy(
                    policy,
                    resource_name=resource_name,
                )
                matches = [
                    item
                    for item in bindings
                    if isinstance(item, Mapping) and item.get("role") == role
                ]
                if not matches:
                    return "absent", None, precondition, receipts
                member = f"serviceAccount:{spec.service_account_email}"
                member_occurrences = sum(
                    list(item.get("members", [])).count(member)
                    for item in matches
                    if isinstance(item.get("members", []), list)
                )
                binding_members = (
                    sorted(str(value) for value in matches[0].get("members", []))
                    if len(matches) == 1
                    and isinstance(matches[0].get("members", []), list)
                    else []
                )
                identity = {
                    "resource_type": (
                        "organization_iam_binding" if org else "project_iam_binding"
                    ),
                    "resource_name": (
                        resource_name
                    ),
                    "role": role,
                    "member": member,
                    "condition": (
                        matches[0].get("condition")
                        if len(matches) == 1
                        else "ambiguous"
                    ),
                    "policy_etag": policy.get("etag"),
                    "policy_version": policy.get("version", 1),
                    "matching_binding_count": len(matches),
                    "matching_member_occurrences": member_occurrences,
                    "binding_members": binding_members,
                }
            elif step.name == "create_dedicated_private_owner_gate_subnet":
                subnets = _list(read((
                    "gcloud",
                    "compute",
                    "networks",
                    "subnets",
                    "list",
                    project,
                    "--format=json",
                )))
                matches = [
                    item
                    for item in subnets
                    if item.get("name") == foundation.OWNER_GATE_SUBNET_NAME
                    and _provider_link_equals(
                        item.get("selfLink"),
                        "https://www.googleapis.com/compute/v1/"
                        f"projects/{spec.project}/regions/{spec.region}/"
                        "subnetworks/"
                        f"{foundation.OWNER_GATE_SUBNET_NAME}",
                    )
                ]
                if not matches:
                    return "absent", None, None, receipts
                if len(matches) != 1:
                    return "drift", None, None, receipts
                subnet = matches[0]
                identity = {
                    "resource_type": "compute_subnetwork",
                    "name": subnet.get("name"),
                    "self_link": _provider_link(subnet.get("selfLink")),
                    "numeric_id": str(subnet.get("id", "")),
                    "fingerprint": subnet.get("fingerprint"),
                    "network_self_link": _provider_link(subnet.get("network")),
                    "region_self_link": _provider_link(subnet.get("region")),
                    "ip_cidr_range": subnet.get("ipCidrRange"),
                    "private_ip_google_access": subnet.get(
                        "privateIpGoogleAccess"
                    ),
                    "stack_type": subnet.get("stackType"),
                    "purpose": subnet.get("purpose", "PRIVATE"),
                    "secondary_ip_ranges": subnet.get("secondaryIpRanges", []),
                }
            elif step.name == "create_private_owner_gate_vm":
                instances = _list(read((
                    "gcloud",
                    "compute",
                    "instances",
                    "list",
                    project,
                    "--format=json",
                )))
                expected_instance = (
                    "https://www.googleapis.com/compute/v1/"
                    f"projects/{spec.project}/zones/{spec.zone}/instances/"
                    f"{spec.vm_name}"
                )
                matches = [
                    item
                    for item in instances
                    if item.get("name") == spec.vm_name
                    and _provider_link_equals(
                        item.get("selfLink"), expected_instance
                    )
                ]
                if not matches:
                    return "absent", None, None, receipts
                if len(matches) != 1:
                    return "drift", None, None, receipts
                instance = matches[0]
                disks = _list(read((
                    "gcloud",
                    "compute",
                    "disks",
                    "list",
                    project,
                    "--format=json",
                )))
                network = read((
                    "gcloud",
                    "compute",
                    "networks",
                    "describe",
                    foundation.NETWORK_NAME,
                    project,
                    "--format=json",
                ))
                subnet = read((
                    "gcloud",
                    "compute",
                    "networks",
                    "subnets",
                    "describe",
                    foundation.OWNER_GATE_SUBNET_NAME,
                    project,
                    f"--region={spec.region}",
                    "--format=json",
                ))
                expected_disk = (
                    "https://www.googleapis.com/compute/v1/"
                    f"projects/{spec.project}/zones/{spec.zone}/disks/"
                    f"{spec.vm_name}"
                )
                disk_matches = [
                    item
                    for item in disks
                    if item.get("name") == spec.vm_name
                    and _provider_link_equals(
                        item.get("selfLink"), expected_disk
                    )
                ]
                interfaces = instance.get("networkInterfaces", [])
                accounts = instance.get("serviceAccounts", [])
                attached = instance.get("disks", [])
                if (
                    len(disk_matches) != 1
                    or not isinstance(interfaces, list)
                    or len(interfaces) != 1
                    or not isinstance(interfaces[0], Mapping)
                    or not isinstance(accounts, list)
                    or len(accounts) != 1
                    or not isinstance(accounts[0], Mapping)
                    or not isinstance(attached, list)
                    or len(attached) != 1
                    or not isinstance(attached[0], Mapping)
                    or not isinstance(network, Mapping)
                    or not isinstance(subnet, Mapping)
                ):
                    return "drift", None, None, receipts
                interface = interfaces[0]
                account = accounts[0]
                attachment = attached[0]
                disk = disk_matches[0]
                metadata_items = instance.get("metadata", {}).get("items", [])
                if not isinstance(metadata_items, list):
                    return "drift", None, None, receipts
                metadata = {
                    str(item.get("key")): str(item.get("value"))
                    for item in metadata_items
                    if isinstance(item, Mapping)
                }
                shielded = instance.get("shieldedInstanceConfig", {})
                scheduling = instance.get("scheduling", {})
                tags = instance.get("tags", {}).get("items", [])
                identity = {
                    "resource_type": "compute_instance",
                    "name": instance.get("name"),
                    "self_link": _provider_link(instance.get("selfLink")),
                    "numeric_id": str(instance.get("id", "")),
                    "metadata_fingerprint": instance.get("metadata", {}).get(
                        "fingerprint"
                    ),
                    "machine_type": _provider_link(instance.get("machineType")),
                    "network_self_link": _provider_link(interface.get("network")),
                    "network_numeric_id": str(network.get("id", "")),
                    "internal_ip": interface.get("networkIP"),
                    "subnetwork_self_link": _provider_link(
                        interface.get("subnetwork")
                    ),
                    "subnetwork_numeric_id": str(subnet.get("id", "")),
                    "network_stack_type": interface.get(
                        "stackType", "IPV4_ONLY"
                    ),
                    "access_configs": interface.get("accessConfigs", []),
                    "service_account_email": account.get("email"),
                    "deletion_protection": instance.get(
                        "deletionProtection", False
                    ),
                    "boot_image_numeric_id": str(disk.get("sourceImageId", "")),
                    "boot_image_self_link": _provider_link(
                        disk.get("sourceImage")
                    ),
                    "boot_image_architecture": disk.get(
                        "architecture", "X86_64"
                    ),
                    "boot_image_license_self_links": sorted(
                        _provider_link(item)
                        for item in disk.get("licenses", [])
                    ),
                    "tags": sorted(str(item) for item in (tags or [])),
                    "metadata": metadata,
                    "shielded_instance_config": {
                        "enable_integrity_monitoring": shielded.get(
                            "enableIntegrityMonitoring"
                        ),
                        "enable_secure_boot": shielded.get("enableSecureBoot"),
                        "enable_vtpm": shielded.get("enableVtpm"),
                    },
                    "oauth_scopes": sorted(
                        str(item) for item in account.get("scopes", [])
                    ),
                    "can_ip_forward": instance.get("canIpForward", False),
                    "maintenance_policy": scheduling.get("onHostMaintenance"),
                    "provisioning_model": scheduling.get("provisioningModel"),
                    "automatic_restart": scheduling.get(
                        "automaticRestart", True
                    ),
                    "preemptible": scheduling.get("preemptible", False),
                    "instance_termination_action": scheduling.get(
                        "instanceTerminationAction", "DELETE"
                    ),
                    "network_interface_count": len(interfaces),
                    "alias_ip_ranges": interface.get("aliasIpRanges", []),
                    "creation_timestamp": instance.get("creationTimestamp"),
                    "labels": instance.get("labels", {}),
                    "resource_policies": sorted(
                        _provider_link(item)
                        for item in instance.get("resourcePolicies", [])
                    ),
                    "min_cpu_platform": instance.get(
                        "minCpuPlatform", "Automatic"
                    ),
                    "confidential_instance_config": {
                        "enable_confidential_compute": instance.get(
                            "confidentialInstanceConfig", {}
                        ).get("enableConfidentialCompute", False)
                    },
                    "boot_disk_name": disk.get("name"),
                    "boot_disk_self_link": _provider_link(attachment.get("source")),
                    "boot_disk_numeric_id": str(disk.get("id", "")),
                    "boot_disk_size_gb": int(disk.get("sizeGb", -1)),
                    "boot_disk_type_self_link": _provider_link(disk.get("type")),
                    "boot_disk_auto_delete": attachment.get("autoDelete"),
                    "boot_disk_boot": attachment.get("boot"),
                    "boot_disk_mode": attachment.get("mode"),
                    "boot_disk_interface": attachment.get("interface"),
                    "boot_disk_attachment_type": attachment.get("type"),
                    "boot_disk_attachment_index": attachment.get("index"),
                }
            elif step.name == "allow_private_web_upstream_from_current_caddy_host":
                firewalls = _list(read((
                    "gcloud",
                    "compute",
                    "firewall-rules",
                    "list",
                    project,
                    "--format=json",
                )))
                name = "muncho-owner-gate-web-from-production"
                expected_firewall = (
                    "https://www.googleapis.com/compute/v1/"
                    f"projects/{spec.project}/global/firewalls/{name}"
                )
                matches = [
                    item
                    for item in firewalls
                    if item.get("name") == name
                    and _provider_link_equals(
                        item.get("selfLink"), expected_firewall
                    )
                ]
                if not matches:
                    return "absent", None, None, receipts
                if len(matches) != 1:
                    return "drift", None, None, receipts
                firewall = matches[0]
                identity = {
                    "resource_type": "compute_firewall",
                    "name": firewall.get("name"),
                    "self_link": _provider_link(firewall.get("selfLink")),
                    "numeric_id": str(firewall.get("id", "")),
                    "fingerprint": firewall.get("fingerprint"),
                    "network_self_link": _provider_link(firewall.get("network")),
                    "direction": firewall.get("direction"),
                    "priority": firewall.get("priority"),
                    "disabled": firewall.get("disabled", False),
                    "action": "ALLOW" if firewall.get("allowed") else "DENY",
                    "allowed": [
                        {
                            "ip_protocol": item.get("IPProtocol"),
                            "ports": sorted(str(port) for port in item.get("ports", [])),
                        }
                        for item in firewall.get("allowed", [])
                        if isinstance(item, Mapping)
                    ],
                    "denied": [
                        {
                            "ip_protocol": item.get("IPProtocol"),
                            "ports": sorted(str(port) for port in item.get("ports", [])),
                        }
                        for item in firewall.get("denied", [])
                        if isinstance(item, Mapping)
                    ],
                    "source_ranges": sorted(firewall.get("sourceRanges", [])),
                    "destination_ranges": sorted(
                        firewall.get("destinationRanges", [])
                    ),
                    "source_tags": sorted(firewall.get("sourceTags", [])),
                    "target_tags": sorted(firewall.get("targetTags", [])),
                    "source_service_accounts": sorted(
                        firewall.get("sourceServiceAccounts", [])
                    ),
                    "target_service_accounts": sorted(
                        firewall.get("targetServiceAccounts", [])
                    ),
                    "log_config": firewall.get("logConfig", {"enable": False}),
                }
            else:
                _error("owner_gate_foundation_provider_step_forbidden")
        except OwnerGateFoundationApplyError as exc:
            if str(exc) == "owner_gate_foundation_provider_read_unknown":
                return "unknown", None, None, receipts
            return "drift", None, None, receipts
        try:
            checked = pre_foundation._validate_resource_identity(
                step.name,
                identity,
                plan=plan,
            )
        except pre_foundation.OwnerGatePreFoundationError:
            return "drift", None, None, receipts
        return "exact", checked, precondition, receipts

    def inspect_resource(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
    ) -> ResourceObservation:
        if (
            pre_foundation.inert_plan_sha256(plan) != self._plan_sha256
            or step not in plan.foundation_steps
        ):
            _error("owner_gate_foundation_provider_step_forbidden")
        (
            first_state,
            first_identity,
            first_precondition,
            first_receipts,
        ) = self._read_once(
            step,
            plan=plan,
        )
        (
            second_state,
            second_identity,
            second_precondition,
            second_receipts,
        ) = self._read_once(
            step,
            plan=plan,
        )
        receipt = _sha256_json({
            "step_name": step.name,
            "first_state": first_state,
            "second_state": second_state,
            "first_identity": first_identity,
            "second_identity": second_identity,
            "first_precondition": first_precondition,
            "second_precondition": second_precondition,
            "first_read_receipts": first_receipts,
            "second_read_receipts": second_receipts,
        })
        if (
            first_state != second_state
            or first_identity != second_identity
            or first_precondition != second_precondition
        ):
            return ResourceObservation("unknown", receipt)
        if first_state == "exact":
            return ResourceObservation(
                first_state,
                receipt,
                resource_identity=first_identity,
                precondition=first_precondition,
            )
        return ResourceObservation(
            first_state,
            receipt,
            precondition=first_precondition,
        )

    def execute_step(
        self,
        step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> OperationObservation:
        if (
            pre_foundation.inert_plan_sha256(plan) != self._plan_sha256
            or step not in plan.foundation_steps
            or _SHA256.fullmatch(attempt_id or "") is None
        ):
            _error("owner_gate_foundation_provider_step_forbidden")
        if _iam_binding_contract(step, plan=plan) is not None:
            return self._iam_binding_operation(
                step,
                plan=plan,
                attempt_id=attempt_id,
                precondition=precondition,
                add=True,
                logical_argv=step.argv,
            )
        result = self._run(step.argv)
        state = (
            "unknown"
            if result.transport_unknown
            else "completed"
            if result.returncode == 0
            else "failed"
        )
        post_etag = None
        if state == "completed" and result.stdout:
            try:
                output = json.loads(result.stdout.decode("utf-8", errors="strict"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                output = None
            if isinstance(output, Mapping) and isinstance(output.get("etag"), str):
                post_etag = str(output["etag"])
        pre_etag = (
            str(precondition["policy_etag"])
            if isinstance(precondition, Mapping)
            and isinstance(precondition.get("policy_etag"), str)
            else None
        )
        receipt = self._capture_receipt(step.argv, result)
        return OperationObservation(
            state,
            receipt,
            attempt_id,
            (
                _sha256_json({
                    "attempt_id": attempt_id,
                    "operation_receipt_sha256": receipt,
                    "stdout_sha256": _sha256_bytes(result.stdout),
                })
                if state == "completed"
                else None
            ),
            pre_etag,
            post_etag,
        )

    def rollback_step(
        self,
        original_step: foundation.PlanStep,
        rollback_step: foundation.PlanStep,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        attempt_id: str,
        precondition: Mapping[str, Any] | None,
    ) -> OperationObservation:
        expected = _rollback_step_for(original_step, plan=plan)
        if (
            pre_foundation.inert_plan_sha256(plan) != self._plan_sha256
            or expected != rollback_step
            or _SHA256.fullmatch(attempt_id or "") is None
        ):
            _error("owner_gate_foundation_provider_step_forbidden")
        if _iam_binding_contract(original_step, plan=plan) is not None:
            return self._iam_binding_operation(
                original_step,
                plan=plan,
                attempt_id=attempt_id,
                precondition=precondition,
                add=False,
                logical_argv=rollback_step.argv,
            )
        result = self._run(rollback_step.argv)
        state = (
            "unknown"
            if result.transport_unknown
            else "completed"
            if result.returncode == 0
            else "failed"
        )
        receipt = self._capture_receipt(rollback_step.argv, result)
        return OperationObservation(
            state,
            receipt,
            attempt_id,
            (
                _sha256_json({
                    "attempt_id": attempt_id,
                    "operation_receipt_sha256": receipt,
                    "stdout_sha256": _sha256_bytes(result.stdout),
                })
                if state == "completed"
                else None
            ),
        )


def _load_fixed_release_private_key(
    *,
    expected_public_key: Ed25519PublicKey,
) -> Ed25519PrivateKey:
    if not isinstance(expected_public_key, Ed25519PublicKey):
        _error("owner_gate_foundation_release_signer_invalid")
    try:
        trust_author._require_owner_directory(
            trust_author.KEY_DIRECTORY,
            expected=trust_author.KEY_DIRECTORY,
            parent=trust_author.AUTHORITY_PARENT,
            create=False,
        )
        private_raw = trust_author._read_exact_regular(
            trust_author.KEY_DIRECTORY / trust_author.PRIVATE_KEY_NAME,
            size=trust_author.PRIVATE_KEY_BYTES,
            modes=frozenset({0o600}),
            code="owner_gate_trust_author_private_key_invalid",
        )
        public_raw = trust_author._read_exact_regular(
            trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME,
            size=trust_author.PUBLIC_KEY_BYTES,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_trust_author_public_key_invalid",
        )
        key = Ed25519PrivateKey.from_private_bytes(private_raw)
        expected_raw = expected_public_key.public_bytes_raw()
        if (
            key.public_key().public_bytes_raw() != public_raw
            or public_raw != expected_raw
            or _sha256_bytes(public_raw)
            != release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
        ):
            _error("owner_gate_foundation_release_signer_not_pinned")
        pre_foundation._require_pinned_public_key(key.public_key())
        return key
    except (
        OSError,
        ValueError,
        trust_author.OwnerGateTrustAuthorError,
        pre_foundation.OwnerGatePreFoundationError,
    ) as exc:
        _error("owner_gate_foundation_release_signer_invalid", exc)


def apply_foundation_from_canonical_artifacts(
    *,
    pre_foundation_authority_raw: bytes,
    owner_reauthentication_receipt_raw: bytes,
    network_evidence_raw: bytes,
    project_ancestry_evidence_raw: bytes,
    release_public_key: Ed25519PublicKey,
    network_collector_public_key: Ed25519PublicKey,
    project_ancestry_collector_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Apply only one canonical signed foundation-A chain.

    Provider, journal, release signer, time, network transport, and command
    runner are deliberately not caller-injectable at this public boundary.
    """

    now_unix = int(time.time())
    if now_unix <= 0:
        _error("owner_gate_foundation_apply_time_invalid")
    chain = decode_validated_foundation_a_chain(
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
    if (
        _OWNER_SUPPORT_BOOTSTRAP_RELEASE_SHA is not None
        and chain.foundation_source_revision
        != _OWNER_SUPPORT_BOOTSTRAP_RELEASE_SHA
    ):
        _error("owner_gate_foundation_direct_release_mismatch")
    private_key = _load_fixed_release_private_key(
        expected_public_key=release_public_key,
    )
    if (
        chain.authority.get("signer_key_id")
        != _sha256_bytes(private_key.public_key().public_bytes_raw())
    ):
        _error("owner_gate_foundation_release_signer_invalid")
    try:
        gcloud_executable = launcher.TrustedGcloudExecutable(
            release_sha=chain.foundation_source_revision,
        )
        gcloud_configuration = launcher.PinnedGcloudConfiguration()
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_foundation_provider_runtime_invalid", exc)
    provider = _TrustedGcloudFoundationProvider(
        plan=chain.plan,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        expected_release_revision=chain.foundation_source_revision,
        runner=_SubprocessFoundationRunner(),
    )
    return _apply_with_provider(
        chain=chain,
        private_key=private_key,
        provider=provider,
        journal=foundation_journal.FoundationApplyJournal(),
        now_unix=lambda: int(time.time()),
        postcondition_wait=time.sleep,
    )


def _read_owner_immutable(path: Path, *, maximum: int) -> bytes:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or ".." in path.parts
        or type(maximum) is not int
        or maximum <= 0
    ):
        _error("owner_gate_foundation_input_path_invalid")
    try:
        return release_trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except release_trust.OwnerGateTrustError as exc:
        _error("owner_gate_foundation_input_path_invalid", exc)


def _load_collector_public_key(path: Path) -> Ed25519PublicKey:
    raw = _read_owner_immutable(path, maximum=16 * 1024)
    try:
        if len(raw) == 32:
            key: Any = Ed25519PublicKey.from_public_bytes(raw)
        else:
            key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        _error("owner_gate_foundation_collector_key_invalid", exc)
    if not isinstance(key, Ed25519PublicKey):
        _error("owner_gate_foundation_collector_key_invalid")
    return key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-foundation-authority", type=Path, required=True)
    parser.add_argument("--owner-reauth-receipt", type=Path, required=True)
    parser.add_argument("--network-evidence", type=Path, required=True)
    parser.add_argument("--project-ancestry-evidence", type=Path, required=True)
    parser.add_argument(
        "--network-collector-public-key",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--project-ancestry-collector-public-key",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args(argv)
    release_public_key = pre_foundation.load_pinned_public_key(
        trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
    )
    network_public_key = _load_collector_public_key(
        arguments.network_collector_public_key
    )
    ancestry_public_key = _load_collector_public_key(
        arguments.project_ancestry_collector_public_key
    )
    authority_raw = _read_owner_immutable(
        arguments.pre_foundation_authority,
        maximum=pre_foundation.MAX_JSON_BYTES,
    )
    owner_raw = _read_owner_immutable(
        arguments.owner_reauth_receipt,
        maximum=owner_reauth.MAX_CAPTURE_BYTES,
    )
    network_raw = _read_owner_immutable(
        arguments.network_evidence,
        maximum=MAX_JSON_BYTES,
    )
    ancestry_raw = _read_owner_immutable(
        arguments.project_ancestry_evidence,
        maximum=MAX_JSON_BYTES,
    )
    chain = decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=authority_raw,
        owner_reauthentication_receipt_raw=owner_raw,
        network_evidence_raw=network_raw,
        project_ancestry_evidence_raw=ancestry_raw,
        release_public_key=release_public_key,
        network_collector_public_key=network_public_key,
        project_ancestry_collector_public_key=ancestry_public_key,
        now_unix=int(time.time()),
    )
    transaction_id = _transaction_id(chain)
    terminal_name = "success"
    try:
        receipt = apply_foundation_from_canonical_artifacts(
            pre_foundation_authority_raw=authority_raw,
            owner_reauthentication_receipt_raw=owner_raw,
            network_evidence_raw=network_raw,
            project_ancestry_evidence_raw=ancestry_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_public_key,
            project_ancestry_collector_public_key=ancestry_public_key,
        )
        receipt_sha256 = receipt["foundation_apply_receipt_sha256"]
        terminal_state = "completed"
        exit_code = 0
    except FoundationApplyFailed as exc:
        terminal_name = "failure"
        receipt_sha256 = exc.receipt[
            "foundation_apply_failure_receipt_sha256"
        ]
        terminal_state = exc.receipt["terminal_state"]
        exit_code = 2
    report = {
        "schema": "muncho-owner-gate-foundation-apply-cli.v1",
        "transaction_id": transaction_id,
        "terminal_state": terminal_state,
        "terminal_receipt_sha256": receipt_sha256,
        "terminal_journal_path": str(
            foundation_journal.DEFAULT_JOURNAL_ROOT
            / transaction_id
            / f"{terminal_name}.json"
        ),
    }
    print(_canonical(report).decode("ascii"))
    return exit_code


__all__ = [
    "FoundationApplyFailed",
    "OwnerGateFoundationApplyError",
    "ValidatedFoundationAChain",
    "ValidatedFoundationApplyChain",
    "apply_foundation_from_canonical_artifacts",
    "decode_validated_foundation_a_chain",
    "load_validated_foundation_apply_chain",
    "load_foundation_terminal_for_source_recovery",
    "validate_failure_receipt",
]


if __name__ == "__main__":
    if _OWNER_SUPPORT_BOOTSTRAP_RELEASE_SHA is None:
        raise SystemExit("owner_gate_foundation_direct_sealed_entrypoint_required")
    raise SystemExit(main())
