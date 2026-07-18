#!/usr/bin/env python3
"""Sealed one-operation owner entrypoint for foundation author-and-apply."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Protocol


_DIRECT_ENTRYPOINT_RELATIVE = Path(
    "source/scripts/canary/owner_gate_author_and_apply.py"
)
_OWNER_SUPPORT_ROOT = re.compile(r"^owner-support-([0-9a-f]{40})$")
_OWNER_SUPPORT_MAX_ENTRIES = 50_000
_OWNER_SUPPORT_MAX_BYTES = 512 * 1024 * 1024


def _activate_direct_owner_support() -> str:
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError("owner_gate_author_direct_isolation_required")
    module_path = Path(__file__)
    invoked_path = Path(sys.argv[0])
    if (
        not module_path.is_absolute()
        or not invoked_path.is_absolute()
        or module_path != invoked_path
        or ".." in module_path.parts
    ):
        raise RuntimeError("owner_gate_author_direct_path_invalid")
    try:
        source_root = module_path.parents[2]
        support_root = module_path.parents[3]
    except IndexError:
        raise RuntimeError("owner_gate_author_direct_path_invalid") from None
    match = _OWNER_SUPPORT_ROOT.fullmatch(support_root.name)
    if (
        match is None
        or module_path != support_root / _DIRECT_ENTRYPOINT_RELATIVE
        or not support_root.is_absolute()
    ):
        raise RuntimeError("owner_gate_author_direct_path_invalid")
    site_root = support_root / "site-packages"
    try:
        if os.path.realpath(module_path, strict=True) != str(module_path):
            raise RuntimeError("owner_gate_author_direct_path_invalid")
    except OSError:
        raise RuntimeError("owner_gate_author_direct_path_invalid") from None
    pending = [support_root]
    entries = 0
    total_bytes = 0
    root_children: set[str] | None = None
    while pending:
        path = pending.pop()
        try:
            metadata = path.lstat()
        except OSError:
            raise RuntimeError("owner_gate_author_direct_tree_invalid") from None
        if metadata.st_uid != os.getuid():  # windows-footgun: ok
            raise RuntimeError("owner_gate_author_direct_tree_invalid")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o500:
                raise RuntimeError("owner_gate_author_direct_tree_invalid")
            try:
                children = tuple(path.iterdir())
            except OSError:
                raise RuntimeError(
                    "owner_gate_author_direct_tree_invalid"
                ) from None
            if path == support_root:
                root_children = {item.name for item in children}
            pending.extend(children)
        elif stat.S_ISREG(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
                raise RuntimeError("owner_gate_author_direct_tree_invalid")
            total_bytes += metadata.st_size
        else:
            raise RuntimeError("owner_gate_author_direct_tree_invalid")
        entries += 1
        if entries > _OWNER_SUPPORT_MAX_ENTRIES or total_bytes > _OWNER_SUPPORT_MAX_BYTES:
            raise RuntimeError("owner_gate_author_direct_tree_invalid")
    if root_children != {"owner-support.json", "source", "site-packages"}:
        raise RuntimeError("owner_gate_author_direct_tree_invalid")
    for required in (
        source_root / "scripts/__init__.py",
        source_root / "scripts/canary/__init__.py",
        module_path,
        site_root / "cryptography/__init__.py",
    ):
        try:
            metadata = required.lstat()
        except OSError:
            raise RuntimeError("owner_gate_author_direct_tree_invalid") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("owner_gate_author_direct_tree_invalid")
    standard_paths = tuple(sys.path)
    if any(
        not isinstance(item, str)
        or not item
        or not os.path.isabs(item)
        or "site-packages" in Path(item).parts
        or "dist-packages" in Path(item).parts
        for item in standard_paths
    ):
        raise RuntimeError("owner_gate_author_direct_sys_path_invalid")
    sys.path[:] = [str(source_root), str(site_root), *standard_paths]
    return match.group(1)


_DIRECT_RELEASE_SHA = _activate_direct_owner_support() if __package__ is None else None

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_author_journal as author_journal
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_interpreter_provenance as interpreter_provenance
from scripts.canary import owner_gate_network_evidence_author as network_author
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_project_ancestry as project_ancestry
from scripts.canary import owner_gate_trust_author as trust_author
from scripts.canary import trusted_signer_author as signer_author


OPERATION = "author-and-apply"
INTENT_SCHEMA = "muncho-owner-gate-foundation-author-intent.v1"
TERMINAL_SCHEMA = "muncho-owner-gate-foundation-author-terminal.v1"
MAX_AUTHORITY_TTL_SECONDS = 600
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OwnerGateAuthorAndApplyError(RuntimeError):
    """Stable, secret-free author-and-apply failure."""


def _error(code: str, _cause: BaseException | None = None) -> NoReturn:
    raise OwnerGateAuthorAndApplyError(code) from None


def _canonical(value: Any) -> bytes:
    return author_journal.canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_release_private_key() -> Ed25519PrivateKey:
    try:
        private_raw = trust_author._read_exact_regular(
            trust_author.KEY_DIRECTORY / trust_author.PRIVATE_KEY_NAME,
            size=trust_author.PRIVATE_KEY_BYTES,
            modes=frozenset({0o600}),
            code="owner_gate_author_release_key_invalid",
        )
        public_raw = trust_author._read_exact_regular(
            trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME,
            size=trust_author.PUBLIC_KEY_BYTES,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_author_release_key_invalid",
        )
        key = Ed25519PrivateKey.from_private_bytes(private_raw)
    except (ValueError, trust_author.OwnerGateTrustAuthorError) as exc:
        _error("owner_gate_author_release_key_invalid", exc)
    if key.public_key().public_bytes_raw() != public_raw:
        _error("owner_gate_author_release_key_invalid")
    try:
        pre_foundation._require_pinned_public_key(key.public_key())
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_author_release_key_not_pinned", exc)
    return key


def _load_network_public_key(release_revision: str) -> Ed25519PublicKey:
    try:
        raw = trust_author._read_exact_regular(
            signer_author._public_path(release_revision, "network"),
            size=32,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_author_network_key_invalid",
        )
        return Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, trust_author.OwnerGateTrustAuthorError) as exc:
        _error("owner_gate_author_network_key_invalid", exc)


class Capabilities(Protocol):
    journal: author_journal.OwnerGateAuthorJournal

    def now_unix(self) -> int: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def release_private_key(self) -> Ed25519PrivateKey: ...

    def network_public_key(self) -> Ed25519PublicKey: ...

    def owner_reauthentication(
        self,
        private_key: Ed25519PrivateKey,
    ) -> Mapping[str, Any]: ...

    def interpreter_evidence(self, collected_at_unix: int) -> Mapping[str, Any]: ...

    def network_evidence(self, collected_at_unix: int) -> Mapping[str, Any]: ...

    def ancestry_evidence(
        self,
        collected_at_unix: int,
        receipt: Mapping[str, Any],
        release_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]: ...

    def apply(
        self,
        *,
        authority_raw: bytes,
        owner_reauth_raw: bytes,
        network_raw: bytes,
        ancestry_raw: bytes,
        release_public_key: Ed25519PublicKey,
        network_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]: ...

    def reconcile_foundation_terminal(
        self,
        *,
        authority_raw: bytes,
        owner_reauth_raw: bytes,
        network_raw: bytes,
        ancestry_raw: bytes,
        release_public_key: Ed25519PublicKey,
        network_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]: ...


class _ProductionCapabilities:
    def __init__(self, release_revision: str) -> None:
        if _REVISION.fullmatch(release_revision or "") is None:
            _error("owner_gate_author_release_invalid")
        self.release_revision = release_revision
        try:
            self.executable = launcher.TrustedGcloudExecutable(
                release_sha=release_revision
            )
            self.configuration = launcher.PinnedGcloudConfiguration()
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_author_runtime_invalid", exc)
        self.journal = author_journal.OwnerGateAuthorJournal()

    def now_unix(self) -> int:
        value = int(time.time())
        if value <= 0:
            _error("owner_gate_author_time_invalid")
        return value

    def manifest(self) -> Mapping[str, Any]:
        try:
            return self.executable.sealed_owner_support_manifest(
                expected_release_sha=self.release_revision
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_author_manifest_invalid", exc)

    def release_private_key(self) -> Ed25519PrivateKey:
        return _load_release_private_key()

    def network_public_key(self) -> Ed25519PublicKey:
        return _load_network_public_key(self.release_revision)

    def owner_reauthentication(
        self,
        private_key: Ed25519PrivateKey,
    ) -> Mapping[str, Any]:
        try:
            return owner_reauth.produce_owner_reauth_receipt(
                runner=owner_reauth.SubprocessOwnerReauthRunner(),
                private_key=private_key,
                now_unix=self.now_unix,
                gcloud_executable=self.executable,
                gcloud_configuration=self.configuration,
                expected_release_revision=self.release_revision,
            )
        except owner_reauth.OwnerGateOwnerReauthError as exc:
            _error("owner_gate_author_owner_reauthentication_failed", exc)

    def interpreter_evidence(self, collected_at_unix: int) -> Mapping[str, Any]:
        return interpreter_provenance.collect_interpreter_provenance(
            release_revision=self.release_revision,
            collected_at_unix=collected_at_unix,
            gcloud_executable=self.executable,
            gcloud_configuration=self.configuration,
        )

    def network_evidence(self, collected_at_unix: int) -> Mapping[str, Any]:
        return network_author.collect_and_author(
            release_revision=self.release_revision,
            collected_at_unix=collected_at_unix,
            gcloud_executable=self.executable,
            gcloud_configuration=self.configuration,
        )

    def ancestry_evidence(
        self,
        collected_at_unix: int,
        receipt: Mapping[str, Any],
        release_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]:
        return project_ancestry.collect_and_author_project_ancestry(
            release_revision=self.release_revision,
            collected_at_unix=collected_at_unix,
            owner_reauthentication_receipt=receipt,
            owner_reauthentication_public_key=release_public_key,
            gcloud_executable=self.executable,
            gcloud_configuration=self.configuration,
        )

    def apply(
        self,
        *,
        authority_raw: bytes,
        owner_reauth_raw: bytes,
        network_raw: bytes,
        ancestry_raw: bytes,
        release_public_key: Ed25519PublicKey,
        network_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]:
        return foundation_apply.apply_foundation_from_canonical_artifacts(
            pre_foundation_authority_raw=authority_raw,
            owner_reauthentication_receipt_raw=owner_reauth_raw,
            network_evidence_raw=network_raw,
            project_ancestry_evidence_raw=ancestry_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_public_key,
            project_ancestry_collector_public_key=network_public_key,
        )

    def reconcile_foundation_terminal(
        self,
        *,
        authority_raw: bytes,
        owner_reauth_raw: bytes,
        network_raw: bytes,
        ancestry_raw: bytes,
        release_public_key: Ed25519PublicKey,
        network_public_key: Ed25519PublicKey,
    ) -> Mapping[str, Any]:
        return foundation_apply.load_foundation_terminal_for_source_recovery(
            pre_foundation_authority_raw=authority_raw,
            owner_reauthentication_receipt_raw=owner_reauth_raw,
            network_evidence_raw=network_raw,
            project_ancestry_evidence_raw=ancestry_raw,
            release_public_key=release_public_key,
            network_collector_public_key=network_public_key,
            project_ancestry_collector_public_key=network_public_key,
        )


@dataclass(frozen=True)
class _AuthoredArtifacts:
    owner_reauth: Mapping[str, Any]
    interpreter: Mapping[str, Any]
    network: Mapping[str, Any]
    ancestry: Mapping[str, Any]
    authority: Mapping[str, Any]
    plan_sha256: str


def _author(
    release_revision: str,
    capabilities: Capabilities,
) -> tuple[_AuthoredArtifacts, Ed25519PublicKey, Ed25519PublicKey]:
    manifest = capabilities.manifest()
    source_tree_oid = manifest.get("source_tree_oid")
    if (
        manifest.get("release_sha") != release_revision
        or _REVISION.fullmatch(str(source_tree_oid or "")) is None
    ):
        _error("owner_gate_author_manifest_invalid")
    release_private = capabilities.release_private_key()
    release_public = release_private.public_key()
    network_public = capabilities.network_public_key()
    owner_receipt = capabilities.owner_reauthentication(release_private)
    collected = capabilities.now_unix()
    interpreter = capabilities.interpreter_evidence(collected)
    network_mapping = capabilities.network_evidence(collected)
    ancestry_mapping = capabilities.ancestry_evidence(
        collected,
        owner_receipt,
        release_public,
    )
    network_key_id = _sha256(network_public.public_bytes_raw())
    try:
        network = foundation.ProductionNetworkEvidence.from_mapping(
            network_mapping,
            public_key=network_public,
            expected_public_key_id=network_key_id,
            now_unix=collected,
        )
        ancestry_raw = _canonical(ancestry_mapping)
        ancestry = project_ancestry.decode_canonical_project_ancestry_evidence(
            ancestry_raw,
            collector_public_key=network_public,
            owner_reauthentication_receipt=owner_receipt,
            owner_reauthentication_public_key=release_public,
            expected_release_revision=release_revision,
            now_unix=collected,
        )
    except (
        foundation.OwnerGateFoundationError,
        project_ancestry.OwnerGateProjectAncestryError,
    ) as exc:
        _error("owner_gate_author_collected_evidence_invalid", exc)
    try:
        interpreter = interpreter_provenance.validate_interpreter_provenance(
            interpreter,
            expected_release_revision=release_revision,
            now_unix=collected,
        )
    except interpreter_provenance.OwnerGateInterpreterProvenanceError as exc:
        _error("owner_gate_author_interpreter_evidence_invalid", exc)
    image = interpreter.get("image")
    if (
        interpreter.get("release_revision") != release_revision
        or not isinstance(image, Mapping)
        or image.get("shortLink") != interpreter_provenance.DEBIAN_IMAGE_SHORT_LINK
        or re.fullmatch(r"[1-9][0-9]{0,31}", str(image.get("id", ""))) is None
        or _SHA256.fullmatch(str(interpreter.get("interpreter_sha256", ""))) is None
        or interpreter.get("python_version") != pre_foundation.PYTHON_VERSION
        or _SHA256.fullmatch(str(interpreter.get("evidence_sha256", ""))) is None
    ):
        _error("owner_gate_author_interpreter_evidence_invalid")
    spec = foundation.OwnerGateSpec(
        release_revision=release_revision,
        source_tree_oid=str(source_tree_oid),
        boot_image_self_link=str(image["shortLink"]),
        boot_image_numeric_id=str(image["id"]),
        interpreter_sha256=str(interpreter["interpreter_sha256"]),
        network_collector_public_key_id=network_key_id,
        organization_id=ancestry.organization_id,
        ancestry_evidence_sha256=ancestry.signed_evidence_sha256,
    )
    try:
        plan = foundation.build_plan(
            spec=spec,
            network_evidence=network,
            network_collector_public_key=network_public,
            now_unix=collected,
        )
        checked_receipt = owner_reauth.validate_owner_reauth_receipt(
            owner_receipt,
            public_key=release_public,
            now_unix=collected,
        )
    except (
        foundation.OwnerGateFoundationError,
        owner_reauth.OwnerGateOwnerReauthError,
    ) as exc:
        _error("owner_gate_author_plan_invalid", exc)
    expires = min(
        collected + MAX_AUTHORITY_TTL_SECONDS,
        int(checked_receipt["expires_at_unix"]),
        int(ancestry.value["expires_at_unix"]),
    )
    if expires <= collected:
        _error("owner_gate_author_evidence_expired")
    try:
        body = pre_foundation.build_authority_body(
            plan=plan,
            network_evidence=network,
            network_collector_public_key=network_public,
            project_ancestry_evidence_raw=ancestry_raw,
            project_ancestry_collector_public_key=network_public,
            owner_reauthentication_receipt=owner_receipt,
            owner_reauthentication_public_key=release_public,
            issued_at_unix=collected,
            expires_at_unix=expires,
            signer_key_id=_sha256(release_public.public_bytes_raw()),
        )
        authority = pre_foundation.sign_pre_foundation_authority(
            body,
            private_key=release_private,
            owner_reauthentication_receipt=owner_receipt,
            project_ancestry_evidence_raw=ancestry_raw,
            project_ancestry_collector_public_key=network_public,
        )
    except pre_foundation.OwnerGatePreFoundationError as exc:
        _error("owner_gate_author_authority_invalid", exc)
    return (
        _AuthoredArtifacts(
            owner_reauth=owner_receipt,
            interpreter=interpreter,
            network=network_mapping,
            ancestry=ancestry_mapping,
            authority=authority,
            plan_sha256=plan.sha256,
        ),
        release_public,
        network_public,
    )


_AUTHORED_ARTIFACTS = (
    "owner-reauth",
    "interpreter-evidence",
    "network-evidence",
    "ancestry-evidence",
    "authority",
)


def _validate_intent(
    *,
    release_revision: str,
    transaction_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    require_complete: bool = True,
) -> Mapping[str, Any]:
    intent = artifacts.get("intent")
    if not isinstance(intent, Mapping) or set(intent) != {
        "schema",
        "operation",
        "release_revision",
        "transaction_id",
        "plan_sha256",
        "artifact_sha256",
        "cloud_mutation_authorized",
        "published_at_unix",
    }:
        _error("owner_gate_author_intent_invalid")
    digests = intent.get("artifact_sha256")
    if (
        intent.get("schema") != INTENT_SCHEMA
        or intent.get("operation") != OPERATION
        or intent.get("release_revision") != release_revision
        or intent.get("transaction_id") != transaction_id
        or _SHA256.fullmatch(str(intent.get("plan_sha256", ""))) is None
        or intent.get("cloud_mutation_authorized")
        != "bounded_nine_operation_foundation_apply"
        or type(intent.get("published_at_unix")) is not int
        or not isinstance(digests, Mapping)
        or set(digests) != set(_AUTHORED_ARTIFACTS)
    ):
        _error("owner_gate_author_intent_invalid")
    for name in _AUTHORED_ARTIFACTS:
        value = artifacts.get(name)
        if value is None and not require_complete:
            continue
        if not isinstance(value, Mapping) or digests.get(name) != _sha256(
            _canonical(value)
        ):
            _error("owner_gate_author_intent_artifact_mismatch")
    expected_transaction = _sha256(_canonical({
        "operation": OPERATION,
        "release_revision": release_revision,
        "authority_sha256": digests["authority"],
    }))
    if transaction_id != expected_transaction:
        _error("owner_gate_author_intent_invalid")
    return dict(intent)


def _historical_foundation_chain(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> foundation_apply.ValidatedFoundationAChain:
    """Decode one exact persisted A chain at its signed issue time."""

    try:
        authority = artifacts["authority"]
        issued = authority.get("issued_at_unix")
        if type(issued) is not int or issued <= 0:
            _error("owner_gate_author_authority_issue_time_invalid")
        return foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=_canonical(authority),
            owner_reauthentication_receipt_raw=_canonical(
                artifacts["owner-reauth"]
            ),
            network_evidence_raw=_canonical(artifacts["network-evidence"]),
            project_ancestry_evidence_raw=_canonical(
                artifacts["ancestry-evidence"]
            ),
            release_public_key=release_public,
            network_collector_public_key=network_public,
            project_ancestry_collector_public_key=network_public,
            now_unix=issued,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        foundation_apply.OwnerGateFoundationApplyError,
    ) as exc:
        _error("owner_gate_author_foundation_chain_invalid", exc)


def _validate_failure_receipt_for_artifacts(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Bind a signed failure receipt to this exact persisted A chain."""

    try:
        receipt = foundation_apply.validate_failure_receipt(
            value,
            public_key=release_public,
        )
        chain = _historical_foundation_chain(
            artifacts=artifacts,
            release_public=release_public,
            network_public=network_public,
        )
        ancestry = chain.ancestry_evidence.value
        expected = {
            "transaction_id": foundation_apply._transaction_id(chain),
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
            "ancestry_chain_sha256": ancestry["stable_chain_sha256"],
            "signer_key_id": _sha256(release_public.public_bytes_raw()),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        foundation_apply.OwnerGateFoundationApplyError,
        pre_foundation.OwnerGatePreFoundationError,
    ) as exc:
        _error("owner_gate_author_apply_failure_receipt_invalid", exc)
    if any(receipt.get(name) != expected_value for name, expected_value in expected.items()):
        _error("owner_gate_author_apply_failure_receipt_mismatch")
    return dict(receipt)


def _validate_success_receipt_for_artifacts(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> Mapping[str, Any]:
    """Bind a signed success receipt to this exact persisted A chain."""

    try:
        completed = value.get("completed_at_unix")
        if type(completed) is not int or completed <= 0:
            _error("owner_gate_author_apply_receipt_invalid")
        chain = _historical_foundation_chain(
            artifacts=artifacts,
            release_public=release_public,
            network_public=network_public,
        )
        validated = foundation_apply._decode_validated_foundation_apply_chain(
            foundation_a=chain,
            apply_receipt_raw=_canonical(value),
            now_unix=completed,
        )
    except (
        TypeError,
        ValueError,
        OwnerGateAuthorAndApplyError,
        foundation_apply.OwnerGateFoundationApplyError,
    ) as exc:
        _error("owner_gate_author_apply_receipt_invalid", exc)
    return dict(validated.apply_receipt)


def _validate_terminal(
    *,
    release_revision: str,
    transaction_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> Mapping[str, Any] | None:
    terminal = artifacts.get("terminal")
    if terminal is None:
        return None
    intent = _validate_intent(
        release_revision=release_revision,
        transaction_id=transaction_id,
        artifacts=artifacts,
        require_complete=False,
    )
    fields = {
        "schema",
        "operation",
        "release_revision",
        "transaction_id",
        "state",
        "error_code",
        "intent_sha256",
        "artifact_sha256",
        "authority_sha256",
        "terminal_receipt_kind",
        "terminal_receipt_sha256",
        "completed_at_unix",
        "terminal_sha256",
    }
    unsigned = {key: value for key, value in terminal.items() if key != "terminal_sha256"}
    if (
        set(terminal) != fields
        or terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("operation") != OPERATION
        or terminal.get("release_revision") != release_revision
        or terminal.get("transaction_id") != transaction_id
        or terminal.get("state")
        not in {"succeeded", "failed", "manual_reconciliation_required"}
        or terminal.get("intent_sha256") != _sha256(_canonical(intent))
        or terminal.get("artifact_sha256") != intent["artifact_sha256"]
        or terminal.get("authority_sha256")
        != intent["artifact_sha256"]["authority"]
        or type(terminal.get("completed_at_unix")) is not int
        or terminal.get("terminal_sha256") != _sha256(_canonical(unsigned))
    ):
        _error("owner_gate_author_terminal_invalid")
    kind = terminal.get("terminal_receipt_kind")
    receipt_sha = terminal.get("terminal_receipt_sha256")
    if terminal["state"] == "succeeded":
        _validate_intent(
            release_revision=release_revision,
            transaction_id=transaction_id,
            artifacts=artifacts,
            require_complete=True,
        )
        receipt = artifacts.get("apply-receipt")
        if (
            terminal.get("error_code") is not None
            or kind != "foundation_apply_success"
            or not isinstance(receipt, Mapping)
            or receipt_sha != _sha256(_canonical(receipt))
            or "apply-failure-receipt" in artifacts
        ):
            _error("owner_gate_author_terminal_invalid")
        try:
            _validate_success_receipt_for_artifacts(
                receipt,
                artifacts=artifacts,
                release_public=release_public,
                network_public=network_public,
            )
        except (
            TypeError,
            ValueError,
            OwnerGateAuthorAndApplyError,
            foundation_apply.OwnerGateFoundationApplyError,
        ) as exc:
            _error("owner_gate_author_terminal_receipt_invalid", exc)
        return receipt
    failure = artifacts.get("apply-failure-receipt")
    if kind == "foundation_apply_failure":
        _validate_intent(
            release_revision=release_revision,
            transaction_id=transaction_id,
            artifacts=artifacts,
            require_complete=True,
        )
        if (
            not isinstance(failure, Mapping)
            or "apply-receipt" in artifacts
            or receipt_sha != _sha256(_canonical(failure))
            or terminal.get("error_code")
            not in {
                "owner_gate_foundation_apply_failed",
                "owner_gate_author_manual_reconciliation_required",
            }
        ):
            _error("owner_gate_author_terminal_invalid")
        checked_failure = _validate_failure_receipt_for_artifacts(
            failure,
            artifacts=artifacts,
            release_public=release_public,
            network_public=network_public,
        )
        expected_state = (
            "manual_reconciliation_required"
            if checked_failure.get("terminal_state")
            == "manual_reconciliation_required"
            else "failed"
        )
        if terminal.get("state") != expected_state:
            _error("owner_gate_author_terminal_invalid")
        expected_code = (
            "owner_gate_author_manual_reconciliation_required"
            if expected_state == "manual_reconciliation_required"
            else "owner_gate_foundation_apply_failed"
        )
        if terminal.get("error_code") != expected_code:
            _error("owner_gate_author_terminal_invalid")
    elif (
        kind is not None
        or receipt_sha is not None
        or failure is not None
        or not isinstance(terminal.get("error_code"), str)
    ):
        _error("owner_gate_author_terminal_invalid")
    if terminal["state"] == "manual_reconciliation_required":
        if (
            terminal.get("error_code")
            != "owner_gate_author_manual_reconciliation_required"
            or "apply-receipt" in artifacts
            or (
                kind is None
                and "apply-failure-receipt" in artifacts
            )
        ):
            _error("owner_gate_author_terminal_invalid")
        _error("owner_gate_author_manual_reconciliation_required")
    return None


def _terminal(
    *,
    release_revision: str,
    transaction_id: str,
    state: str,
    code: str | None,
    artifacts: Mapping[str, Mapping[str, Any]],
    receipt_kind: str | None,
    receipt: Mapping[str, Any] | None,
    now_unix: int,
) -> Mapping[str, Any]:
    intent = _validate_intent(
        release_revision=release_revision,
        transaction_id=transaction_id,
        artifacts=artifacts,
        require_complete=False,
    )
    unsigned = {
        "schema": TERMINAL_SCHEMA,
        "operation": OPERATION,
        "release_revision": release_revision,
        "transaction_id": transaction_id,
        "state": state,
        "error_code": code,
        "intent_sha256": _sha256(_canonical(intent)),
        "artifact_sha256": intent["artifact_sha256"],
        "authority_sha256": intent["artifact_sha256"]["authority"],
        "terminal_receipt_kind": receipt_kind,
        "terminal_receipt_sha256": (
            None if receipt is None else _sha256(_canonical(receipt))
        ),
        "completed_at_unix": now_unix,
    }
    return {**unsigned, "terminal_sha256": _sha256(_canonical(unsigned))}


def _apply_published(
    *,
    release_revision: str,
    transaction_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    capabilities: Capabilities,
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> Mapping[str, Any]:
    required = {
        "intent",
        "owner-reauth",
        "interpreter-evidence",
        "network-evidence",
        "ancestry-evidence",
        "authority",
    }
    if not required.issubset(artifacts):
        _error("owner_gate_author_publication_incomplete")
    try:
        receipt = capabilities.apply(
            authority_raw=_canonical(artifacts["authority"]),
            owner_reauth_raw=_canonical(artifacts["owner-reauth"]),
            network_raw=_canonical(artifacts["network-evidence"]),
            ancestry_raw=_canonical(artifacts["ancestry-evidence"]),
            release_public_key=release_public,
            network_public_key=network_public,
        )
    except foundation_apply.FoundationApplyFailed as exc:
        failure = _validate_failure_receipt_for_artifacts(
            exc.receipt,
            artifacts=artifacts,
            release_public=release_public,
            network_public=network_public,
        )
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "apply-failure-receipt",
            failure,
        )
        failed_artifacts = capabilities.journal.list_artifacts(
            release_revision,
            transaction_id,
        )
        manual = (
            failure.get("terminal_state")
            == "manual_reconciliation_required"
        )
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "terminal",
            _terminal(
                release_revision=release_revision,
                transaction_id=transaction_id,
                state=(
                    "manual_reconciliation_required" if manual else "failed"
                ),
                code=(
                    "owner_gate_author_manual_reconciliation_required"
                    if manual
                    else "owner_gate_foundation_apply_failed"
                ),
                artifacts=failed_artifacts,
                receipt_kind="foundation_apply_failure",
                receipt=failure,
                now_unix=capabilities.now_unix(),
            ),
        )
        _error(
            "owner_gate_author_manual_reconciliation_required"
            if manual
            else "owner_gate_foundation_apply_failed",
            exc,
        )
    except Exception as exc:
        # Once apply has been entered, an ordinary exception cannot prove
        # whether the fixed inner journal already contains mutation intent.
        # Leave the outer transaction nonterminal so the next invocation must
        # reconcile that journal before any new authoring is possible.
        _error("owner_gate_author_apply_outcome_unknown", exc)
    receipt = _validate_success_receipt_for_artifacts(
        receipt,
        artifacts=artifacts,
        release_public=release_public,
        network_public=network_public,
    )
    capabilities.journal.publish(
        release_revision,
        transaction_id,
        "apply-receipt",
        receipt,
    )
    successful_artifacts = capabilities.journal.list_artifacts(
        release_revision,
        transaction_id,
    )
    capabilities.journal.publish(
        release_revision,
        transaction_id,
        "terminal",
        _terminal(
            release_revision=release_revision,
            transaction_id=transaction_id,
            state="succeeded",
            code=None,
            artifacts=successful_artifacts,
            receipt_kind="foundation_apply_success",
            receipt=receipt,
            now_unix=capabilities.now_unix(),
        ),
    )
    return receipt


def _reconcile_published_foundation_terminal(
    *,
    release_revision: str,
    transaction_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    capabilities: Capabilities,
    release_public: Ed25519PublicKey,
    network_public: Ed25519PublicKey,
) -> tuple[str, Mapping[str, Any] | None]:
    """Adopt only an exact terminal from the fixed inner apply journal."""

    try:
        observed = capabilities.reconcile_foundation_terminal(
            authority_raw=_canonical(artifacts["authority"]),
            owner_reauth_raw=_canonical(artifacts["owner-reauth"]),
            network_raw=_canonical(artifacts["network-evidence"]),
            ancestry_raw=_canonical(artifacts["ancestry-evidence"]),
            release_public_key=release_public,
            network_public_key=network_public,
        )
    except foundation_apply.OwnerGateFoundationApplyError as exc:
        _error("owner_gate_author_foundation_reconciliation_invalid", exc)
    if not isinstance(observed, Mapping) or observed.get("state") not in {
        "absent",
        "in_progress",
        "succeeded",
        "failed",
    }:
        _error("owner_gate_author_foundation_reconciliation_invalid")
    state = str(observed["state"])
    if (
        state in {"absent", "in_progress"}
        and (
            "apply-receipt" in artifacts
            or "apply-failure-receipt" in artifacts
        )
    ) or (
        state == "succeeded" and "apply-failure-receipt" in artifacts
    ) or (
        state == "failed" and "apply-receipt" in artifacts
    ):
        _error("owner_gate_author_foundation_reconciliation_conflict")
    if state == "absent":
        return state, None
    if state == "in_progress":
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "terminal",
            _terminal(
                release_revision=release_revision,
                transaction_id=transaction_id,
                state="manual_reconciliation_required",
                code="owner_gate_author_manual_reconciliation_required",
                artifacts=artifacts,
                receipt_kind=None,
                receipt=None,
                now_unix=capabilities.now_unix(),
            ),
        )
        _error("owner_gate_author_manual_reconciliation_required")
    receipt = observed.get("receipt")
    if not isinstance(receipt, Mapping):
        _error("owner_gate_author_foundation_reconciliation_invalid")
    if state == "succeeded":
        receipt = _validate_success_receipt_for_artifacts(
            receipt,
            artifacts=artifacts,
            release_public=release_public,
            network_public=network_public,
        )
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "apply-receipt",
            receipt,
        )
        linked = capabilities.journal.list_artifacts(
            release_revision,
            transaction_id,
        )
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "terminal",
            _terminal(
                release_revision=release_revision,
                transaction_id=transaction_id,
                state="succeeded",
                code=None,
                artifacts=linked,
                receipt_kind="foundation_apply_success",
                receipt=receipt,
                now_unix=capabilities.now_unix(),
            ),
        )
        return state, receipt
    failure = _validate_failure_receipt_for_artifacts(
        receipt,
        artifacts=artifacts,
        release_public=release_public,
        network_public=network_public,
    )
    capabilities.journal.publish(
        release_revision,
        transaction_id,
        "apply-failure-receipt",
        failure,
    )
    linked = capabilities.journal.list_artifacts(release_revision, transaction_id)
    manual = failure.get("terminal_state") == "manual_reconciliation_required"
    capabilities.journal.publish(
        release_revision,
        transaction_id,
        "terminal",
        _terminal(
            release_revision=release_revision,
            transaction_id=transaction_id,
            state="manual_reconciliation_required" if manual else "failed",
            code=(
                "owner_gate_author_manual_reconciliation_required"
                if manual
                else "owner_gate_foundation_apply_failed"
            ),
            artifacts=linked,
            receipt_kind="foundation_apply_failure",
            receipt=failure,
            now_unix=capabilities.now_unix(),
        ),
    )
    if manual:
        _error("owner_gate_author_manual_reconciliation_required")
    return state, failure


def _author_and_apply_with_capabilities(
    release_revision: str,
    capabilities: Capabilities,
) -> Mapping[str, Any]:
    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_author_release_invalid")
    with capabilities.journal.release_lease(release_revision):
        release_public = capabilities.release_private_key().public_key()
        network_public = capabilities.network_public_key()
        transactions = capabilities.journal.list_transactions(release_revision)
        for transaction_id, artifacts in transactions.items():
            replay = _validate_terminal(
                release_revision=release_revision,
                transaction_id=transaction_id,
                artifacts=artifacts,
                release_public=release_public,
                network_public=network_public,
            )
            if replay is not None:
                return replay
            if "terminal" not in artifacts:
                if {
                    "intent",
                    "owner-reauth",
                    "interpreter-evidence",
                    "network-evidence",
                    "ancestry-evidence",
                    "authority",
                }.issubset(artifacts):
                    _validate_intent(
                        release_revision=release_revision,
                        transaction_id=transaction_id,
                        artifacts=artifacts,
                    )
                    state, reconciled = (
                        _reconcile_published_foundation_terminal(
                            release_revision=release_revision,
                            transaction_id=transaction_id,
                            artifacts=artifacts,
                            capabilities=capabilities,
                            release_public=release_public,
                            network_public=network_public,
                        )
                    )
                    if state == "succeeded":
                        assert reconciled is not None
                        return reconciled
                    if state == "failed":
                        continue
                    if state != "absent":
                        _error(
                            "owner_gate_author_foundation_reconciliation_invalid"
                        )
                    capabilities.journal.publish(
                        release_revision,
                        transaction_id,
                        "terminal",
                        _terminal(
                            release_revision=release_revision,
                            transaction_id=transaction_id,
                            state="failed",
                            code=(
                                "owner_gate_author_foundation_absent_after_"
                                "interruption"
                            ),
                            artifacts=artifacts,
                            receipt_kind=None,
                            receipt=None,
                            now_unix=capabilities.now_unix(),
                        ),
                    )
                    continue
                if "apply-receipt" in artifacts or "apply-failure-receipt" in artifacts:
                    _error("owner_gate_author_publication_invalid")
                capabilities.journal.publish(
                    release_revision,
                    transaction_id,
                    "terminal",
                    _terminal(
                        release_revision=release_revision,
                        transaction_id=transaction_id,
                        state="failed",
                        code="owner_gate_author_publication_interrupted",
                        artifacts=artifacts,
                        receipt_kind=None,
                        receipt=None,
                        now_unix=capabilities.now_unix(),
                    ),
                )
        authored, release_public, network_public = _author(
            release_revision,
            capabilities,
        )
        raw = {
            "owner-reauth": _canonical(authored.owner_reauth),
            "interpreter-evidence": _canonical(authored.interpreter),
            "network-evidence": _canonical(authored.network),
            "ancestry-evidence": _canonical(authored.ancestry),
            "authority": _canonical(authored.authority),
        }
        authority_sha = _sha256(raw["authority"])
        transaction_id = _sha256(_canonical({
            "operation": OPERATION,
            "release_revision": release_revision,
            "authority_sha256": authority_sha,
        }))
        intent = {
            "schema": INTENT_SCHEMA,
            "operation": OPERATION,
            "release_revision": release_revision,
            "transaction_id": transaction_id,
            "plan_sha256": authored.plan_sha256,
            "artifact_sha256": {
                name: _sha256(payload) for name, payload in sorted(raw.items())
            },
            "cloud_mutation_authorized": "bounded_nine_operation_foundation_apply",
            "published_at_unix": capabilities.now_unix(),
        }
        capabilities.journal.publish(
            release_revision,
            transaction_id,
            "intent",
            intent,
        )
        mapping = {
            "owner-reauth": authored.owner_reauth,
            "interpreter-evidence": authored.interpreter,
            "network-evidence": authored.network,
            "ancestry-evidence": authored.ancestry,
            "authority": authored.authority,
        }
        for name in (
            "owner-reauth",
            "interpreter-evidence",
            "network-evidence",
            "ancestry-evidence",
            "authority",
        ):
            capabilities.journal.publish(
                release_revision,
                transaction_id,
                name,
                mapping[name],
            )
        artifacts = capabilities.journal.list_artifacts(
            release_revision,
            transaction_id,
        )
        return _apply_published(
            release_revision=release_revision,
            transaction_id=transaction_id,
            artifacts=artifacts,
            capabilities=capabilities,
            release_public=release_public,
            network_public=network_public,
        )


def author_and_apply() -> Mapping[str, Any]:
    """Run the sole production operation from the sealed direct entrypoint."""

    if _DIRECT_RELEASE_SHA is None:
        _error("owner_gate_author_direct_entrypoint_required")
    return _author_and_apply_with_capabilities(
        _DIRECT_RELEASE_SHA,
        _ProductionCapabilities(_DIRECT_RELEASE_SHA),
    )


def main() -> int:
    if sys.argv != [str(Path(__file__)), OPERATION]:
        print(_canonical({
            "ok": False,
            "error": "owner_gate_author_operation_invalid",
        }).decode("ascii"))
        return 2
    try:
        receipt = author_and_apply()
    except BaseException as exc:
        code = (
            str(exc)
            if type(exc) is OwnerGateAuthorAndApplyError
            else "owner_gate_author_and_apply_failed"
        )
        print(_canonical({"ok": False, "error": code}).decode("ascii"))
        return 1
    print(_canonical({"ok": True, "receipt": receipt}).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OPERATION",
    "OwnerGateAuthorAndApplyError",
    "author_and_apply",
]
