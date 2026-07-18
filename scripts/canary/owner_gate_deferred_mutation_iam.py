#!/usr/bin/env python3
"""Owner-authenticated CAS lifecycle for the deferred mutation IAM binding.

This module deliberately does not extend Foundation F's executable step set.
It recognizes the single immutable deferred step already present in the final
release plan, binds it to the validated Foundation F apply lineage, and owns a
separate append-only activate/remove transaction.  The public API accepts no
project, role, member, condition, command, path, provider, or journal input.
"""

from __future__ import annotations

import hashlib
import fcntl
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Never, Protocol, Sequence, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_foundation_journal as foundation_journal
from scripts.canary import owner_gate_inert_observation as inert_observation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_pre_foundation as pre_foundation


INTENT_SCHEMA = "muncho-owner-gate-deferred-mutation-iam-intent.v1"
OPERATION_SCHEMA = "muncho-owner-gate-deferred-mutation-iam-operation.v1"
SUCCESS_SCHEMA = "muncho-owner-gate-deferred-mutation-iam-success.v1"
FAILURE_SCHEMA = "muncho-owner-gate-deferred-mutation-iam-failure.v1"
TRANSACTION_SCHEMA = "muncho-owner-gate-deferred-mutation-iam-transaction.v1"

ACTION_ACTIVATE = "activate"
ACTION_REMOVE = "remove"
_ACTIONS = frozenset({ACTION_ACTIVATE, ACTION_REMOVE})
_OBSERVATION_STATES = frozenset({"absent", "exact", "drift", "unknown"})
_OPERATION_STATES = frozenset({"completed", "failed", "unknown"})
MAX_ACTIVATION_ATTEMPTS = 4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")

CONDITION_TITLE = "muncho_owner_gate_exact_storage_v1"
CONDITION_DESCRIPTION = "Exact canary disk and instance resources only"
DEFAULT_JOURNAL_ROOT = foundation_journal.DEFAULT_JOURNAL_ROOT.parent / (
    "owner-gate-deferred-mutation-iam"
)
_JOURNAL_ARTIFACTS = frozenset({
    f"{action}-{phase}"
    for action in _ACTIONS
    for phase in ("intent", "operation", "success", "failure")
}) | frozenset({
    f"activate-retry-{index}-{phase}"
    for index in range(1, MAX_ACTIVATION_ATTEMPTS)
    for phase in ("intent", "operation")
})


def _intent_artifact_name(action: str, activation_attempt_index: int | None) -> str:
    if action == ACTION_REMOVE and activation_attempt_index is None:
        return "remove-intent"
    if action != ACTION_ACTIVATE or type(activation_attempt_index) is not int:
        _error("owner_gate_deferred_mutation_iam_attempt_invalid")
    if not 0 <= activation_attempt_index < MAX_ACTIVATION_ATTEMPTS:
        _error("owner_gate_deferred_mutation_iam_attempt_invalid")
    return (
        "activate-intent"
        if activation_attempt_index == 0
        else f"activate-retry-{activation_attempt_index}-intent"
    )


def _operation_artifact_name(
    action: str,
    activation_attempt_index: int | None,
) -> str:
    return _intent_artifact_name(action, activation_attempt_index).replace(
        "-intent",
        "-operation",
    )


class OwnerGateDeferredMutationIamError(RuntimeError):
    """Stable fail-closed deferred-IAM boundary error."""


class OwnerGateDeferredMutationIamFailed(OwnerGateDeferredMutationIamError):
    """A terminal, append-only provider failure."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = dict(receipt)
        super().__init__("owner_gate_deferred_mutation_iam_failed")


def _error(code: str, _cause: BaseException | None = None) -> Never:
    del _cause
    raise OwnerGateDeferredMutationIamError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_deferred_mutation_iam_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical(value))


class DeferredMutationIamJournal(foundation_journal.FoundationApplyJournal):
    """Closed owner-only journal for one fixed activate/remove lifecycle."""

    def __init__(
        self,
        *,
        _root: Path = DEFAULT_JOURNAL_ROOT,
        _owner_uid: int = foundation_journal.OWNER_UID,
        _owner_gid: int = foundation_journal.OWNER_GID,
    ) -> None:
        super().__init__(
            _root=_root,
            _owner_uid=_owner_uid,
            _owner_gid=_owner_gid,
            _artifact_names=_JOURNAL_ARTIFACTS,
        )
        self._active_contract_lease: (
            tuple[int, Path, tuple[int, int]] | None
        ) = None

    def _validated_contract_inventory(self) -> tuple[str, ...]:
        active = self._active_contract_lease
        if active is None or active[0] != threading.get_ident():
            raise RuntimeError(
                "owner_gate_deferred_mutation_iam_contract_lease_required"
            )
        _owner_thread, active_root, active_identity = active
        descriptor = os.open(
            active_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            current = os.lstat(active_root)
        finally:
            os.close(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != active_identity
            or (current.st_dev, current.st_ino) != active_identity
        ):
            raise RuntimeError(
                "owner_gate_deferred_mutation_iam_contract_root_changed"
            )
        self._validate_directory(active_root)
        entries_before = tuple(sorted(os.listdir(active_root)))
        identities: dict[str, tuple[int, int, int, int, int]] = {}
        for entry in entries_before:
            if foundation_journal._TRANSACTION_ID.fullmatch(entry) is None:
                raise RuntimeError(
                    "owner_gate_deferred_mutation_iam_contract_inventory_invalid"
                )
            path = active_root / entry
            self._validate_directory(path)
            item = os.lstat(path)
            identities[entry] = (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_gid,
            )
        entries_after = tuple(sorted(os.listdir(active_root)))
        if entries_after != entries_before:
            raise RuntimeError(
                "owner_gate_deferred_mutation_iam_contract_inventory_changed"
            )
        for entry, identity in identities.items():
            item = os.lstat(active_root / entry)
            if (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_uid,
                item.st_gid,
            ) != identity:
                raise RuntimeError(
                    "owner_gate_deferred_mutation_iam_contract_inventory_changed"
                )
        return entries_before

    def require_contract_lease(self) -> None:
        """Require the fixed-contract root lease on the calling thread."""

        self._validated_contract_inventory()

    def transaction_ids(self) -> tuple[str, ...]:
        """Return a hardened inventory while the fixed-contract lease is held."""

        return self._validated_contract_inventory()

    @contextmanager
    def contract_lease(self) -> Iterator[None]:
        """Serialize all releases that target the one immutable IAM contract."""

        if self._active_contract_lease is not None or self._active_lease is not None:
            raise RuntimeError(
                "owner_gate_deferred_mutation_iam_contract_lease_conflict"
            )
        self._require_owner_process()
        if not os.path.lexists(self.root):
            self._ensure_root()
        else:
            self._validate_directory(self.root)
        descriptor = os.open(
            self.root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            current = os.lstat(self.root)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
            ) != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
            ):
                raise RuntimeError(
                    "owner_gate_deferred_mutation_iam_contract_root_changed"
                )
            self._validate_directory(self.root)
            identity = (opened.st_dev, opened.st_ino)
            self._active_contract_lease = (
                threading.get_ident(),
                self.root,
                identity,
            )
            self._validated_contract_inventory()
            try:
                yield
            except BaseException:
                raise
            else:
                self._validated_contract_inventory()
            finally:
                self._active_contract_lease = None
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@dataclass(frozen=True)
class _ActivationAttemptDescriptor:
    index: int
    intent_name: str
    intent: Mapping[str, Any]
    success: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None
    attempts: tuple[tuple[int, str, Mapping[str, Any]], ...]


def _basic_activation_attempt_descriptor(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    transaction_id: str,
    release_revision: str,
) -> _ActivationAttemptDescriptor | None:
    intents: list[tuple[int, str, Mapping[str, Any]]] = []
    missing_seen = False
    stable_reference: Mapping[str, Any] | None = None
    reauthentication_receipts: set[str] = set()
    attempt_ids: set[str] = set()
    contract_fields = ("resource_name", "role", "member", "condition")
    for index in range(MAX_ACTIVATION_ATTEMPTS):
        name = _intent_artifact_name(ACTION_ACTIVATE, index)
        operation_name = _operation_artifact_name(ACTION_ACTIVATE, index)
        value = artifacts.get(name)
        operation = artifacts.get(operation_name)
        if value is None:
            missing_seen = True
            if operation is not None:
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
            continue
        if missing_seen or not isinstance(value, Mapping):
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        unsigned = {
            key: item for key, item in value.items() if key != "intent_sha256"
        }
        evidence_set_sha256 = str(
            value.get("inert_evidence_set_sha256", "")
        )
        reauthentication_receipt_sha256 = str(
            value.get(
                "activation_owner_reauthentication_receipt_sha256",
                "",
            )
        )
        attempt_id = str(value.get("attempt_id", ""))
        if (
            value.get("schema") != INTENT_SCHEMA
            or value.get("action") != ACTION_ACTIVATE
            or value.get("activation_attempt_index") != index
            or value.get("intent_artifact_name") != name
            or value.get("transaction_id") != transaction_id
            or value.get("final_release_revision") != release_revision
            or _SHA256.fullmatch(evidence_set_sha256) is None
            or _SHA256.fullmatch(reauthentication_receipt_sha256) is None
            or _SHA256.fullmatch(attempt_id) is None
            or value.get("intent_sha256") != _sha256_json(unsigned)
            or (index > 0 and value.get("source_state") != "absent")
            or reauthentication_receipt_sha256
            in reauthentication_receipts
            or attempt_id in attempt_ids
        ):
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        reauthentication_receipts.add(reauthentication_receipt_sha256)
        attempt_ids.add(attempt_id)
        stable = {
            name: value.get(name)
            for name in (*_STABLE_LIFECYCLE_FIELDS, *contract_fields)
        }
        if stable_reference is None:
            stable_reference = stable
        elif stable != stable_reference:
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        intents.append((index, name, value))
    if not intents:
        return None
    for prior_index, _prior_name, _prior_intent in intents[:-1]:
        if artifacts.get(
            _operation_artifact_name(ACTION_ACTIVATE, prior_index)
        ) is not None:
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
    success = artifacts.get("activate-success")
    failure = artifacts.get("activate-failure")
    if success is not None and failure is not None:
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    terminal = success if success is not None else failure
    if terminal is None:
        index, name, intent = intents[-1]
        return _ActivationAttemptDescriptor(
            index,
            name,
            intent,
            None,
            None,
            tuple(intents),
        )
    if not isinstance(terminal, Mapping):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    unsigned_terminal = {
        key: item
        for key, item in terminal.items()
        if key != "receipt_sha256"
    }
    terminal_index = terminal.get("activation_attempt_index")
    if (
        type(terminal_index) is not int
        or not 0 <= terminal_index < len(intents)
    ):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    index, name, intent = intents[terminal_index]
    if (
        terminal_index != index
        or len(intents) != terminal_index + 1
        or terminal.get("action") != ACTION_ACTIVATE
        or terminal.get("transaction_id") != transaction_id
        or terminal.get("final_release_revision") != release_revision
        or terminal.get("intent_artifact_name") != name
        or terminal.get("intent_sha256") != intent.get("intent_sha256")
        or terminal.get("receipt_sha256")
        != _sha256_json(unsigned_terminal)
    ):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    return _ActivationAttemptDescriptor(
        index,
        name,
        intent,
        success,
        failure,
        tuple(intents),
    )


@dataclass(frozen=True)
class DeferredMutationIamContract:
    resource_name: str
    role: str
    member: str
    condition: Mapping[str, str]

    def binding(self) -> Mapping[str, Any]:
        return {
            "role": self.role,
            "members": [self.member],
            "condition": dict(self.condition),
        }


def _fixed_contract_values() -> DeferredMutationIamContract:
    return DeferredMutationIamContract(
        resource_name=f"projects/{foundation.PROJECT}",
        role=(
            f"projects/{foundation.PROJECT}/roles/"
            f"{foundation.MUTATION_ROLE_ID}"
        ),
        member=(
            "serviceAccount:"
            f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}."
            "iam.gserviceaccount.com"
        ),
        condition={
            "title": CONDITION_TITLE,
            "description": CONDITION_DESCRIPTION,
            "expression": foundation._condition_expression(),
        },
    )


def _fixed_contract(
    plan: foundation.OwnerGateFoundationPlan,
) -> DeferredMutationIamContract:
    if (
        type(plan) is not foundation.OwnerGateFoundationPlan
        or not plan.spec.final_release_bound
        or plan.spec.project != foundation.PROJECT
        or plan.spec.service_account_name != foundation.SERVICE_ACCOUNT_NAME
        or plan.spec.custom_role
        != f"projects/{foundation.PROJECT}/roles/{foundation.MUTATION_ROLE_ID}"
    ):
        _error("owner_gate_deferred_mutation_iam_plan_invalid")
    contract = _fixed_contract_values()
    condition = contract.condition
    if (
        contract.role != plan.spec.custom_role
        or contract.member
        != f"serviceAccount:{plan.spec.service_account_email}"
    ):
        _error("owner_gate_deferred_mutation_iam_plan_invalid")
    expected_activate = foundation.PlanStep(
        "activate_resource_conditioned_mutation_role_after_smoke",
        (
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            foundation.PROJECT,
            f"--member={contract.member}",
            f"--role={contract.role}",
            "--condition="
            f"title={CONDITION_TITLE},"
            f"description={CONDITION_DESCRIPTION},"
            f"expression={condition['expression']}",
            "--quiet",
        ),
    )
    expected_remove = foundation.PlanStep(
        "remove_exact_mutation_binding_if_present",
        (
            "gcloud",
            "projects",
            "remove-iam-policy-binding",
            foundation.PROJECT,
            f"--member={contract.member}",
            f"--role={contract.role}",
            "--condition="
            f"title={CONDITION_TITLE},"
            f"description={CONDITION_DESCRIPTION},"
            f"expression={condition['expression']}",
            "--quiet",
        ),
    )
    rollback_matches = tuple(
        step
        for step in plan.rollback_steps
        if step.name == expected_remove.name
    )
    if (
        plan.deferred_mutation_iam_steps != (expected_activate,)
        or rollback_matches != (expected_remove,)
        or expected_activate in plan.foundation_steps
    ):
        _error("owner_gate_deferred_mutation_iam_plan_invalid")
    return contract


_AUTHORITY_MARKER = object()


@dataclass(frozen=True, init=False)
class _DeferredMutationIamAuthority:
    plan: foundation.OwnerGateFoundationPlan
    foundation_apply_chain: foundation_apply.ValidatedFoundationApplyChain
    final_release_public_key: Ed25519PublicKey
    contract: DeferredMutationIamContract
    transaction_id: str
    lineage: Mapping[str, Any]
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_DeferredMutationIamAuthority":
        _error("owner_gate_deferred_mutation_iam_authority_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        plan: foundation.OwnerGateFoundationPlan,
        foundation_apply_chain: foundation_apply.ValidatedFoundationApplyChain,
        final_release_public_key: Ed25519PublicKey,
        contract: DeferredMutationIamContract,
        lineage: Mapping[str, Any],
    ) -> "_DeferredMutationIamAuthority":
        transaction_id = _stable_transaction_id(
            contract=contract,
            lineage=lineage,
        )
        value = object.__new__(cls)
        object.__setattr__(value, "plan", plan)
        object.__setattr__(value, "foundation_apply_chain", foundation_apply_chain)
        object.__setattr__(
            value,
            "final_release_public_key",
            final_release_public_key,
        )
        object.__setattr__(value, "contract", contract)
        object.__setattr__(value, "transaction_id", transaction_id)
        object.__setattr__(value, "lineage", dict(lineage))
        object.__setattr__(value, "_marker", _AUTHORITY_MARKER)
        return value


_STABLE_LIFECYCLE_FIELDS = (
    "final_release_revision",
    "final_source_tree_oid",
    "final_package_sha256",
    "foundation_source_revision",
    "foundation_source_tree_oid",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "final_release_public_key_id",
)


def _stable_transaction_id(
    *,
    contract: DeferredMutationIamContract,
    lineage: Mapping[str, Any],
) -> str:
    try:
        stable = {name: lineage[name] for name in _STABLE_LIFECYCLE_FIELDS}
    except (KeyError, TypeError) as exc:
        _error("owner_gate_deferred_mutation_iam_lineage_invalid", exc)
    if (
        not isinstance(contract, DeferredMutationIamContract)
        or _REVISION.fullmatch(str(stable["final_release_revision"]))
        is None
        or _REVISION.fullmatch(str(stable["final_source_tree_oid"])) is None
        or _REVISION.fullmatch(str(stable["foundation_source_revision"]))
        is None
        or _REVISION.fullmatch(str(stable["foundation_source_tree_oid"]))
        is None
        or any(
            _SHA256.fullmatch(str(stable[name])) is None
            for name in (
                "final_package_sha256",
                "pre_foundation_authority_sha256",
                "foundation_apply_receipt_sha256",
                "final_release_public_key_id",
            )
        )
    ):
        _error("owner_gate_deferred_mutation_iam_lineage_invalid")
    return _sha256_json({
        "schema": TRANSACTION_SCHEMA,
        **stable,
        "contract": {
            "resource_name": contract.resource_name,
            "role": contract.role,
            "member": contract.member,
            "condition": dict(contract.condition),
        },
    })


def _stable_spec_projection(spec: foundation.OwnerGateSpec) -> Mapping[str, Any]:
    excluded = {
        "release_revision",
        "source_tree_oid",
        "boot_image_numeric_id",
        "package_inventory_sha256",
        "network_collector_public_key_id",
        "cloud_collector_public_key_id",
        "host_collector_public_key_id",
    }
    return {
        name: item
        for name, item in asdict(spec).items()
        if name not in excluded
    }


def _validated_authority(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    foundation_apply_chain: foundation_apply.ValidatedFoundationApplyChain,
    final_network_evidence: foundation.ProductionNetworkEvidence,
    final_network_collector_public_key: Ed25519PublicKey,
    final_release_public_key: Ed25519PublicKey,
    final_source_tree_oid: str,
    final_package_sha256: str,
    inert_evidence_set_sha256: str,
    now_unix: int,
) -> _DeferredMutationIamAuthority:
    if (
        type(plan) is not foundation.OwnerGateFoundationPlan
        or type(foundation_apply_chain)
        is not foundation_apply.ValidatedFoundationApplyChain
        or getattr(foundation_apply_chain, "_marker", None)
        is not foundation_apply._CHAIN_MARKER
        or type(final_network_evidence)
        is not foundation.ProductionNetworkEvidence
        or not isinstance(final_network_collector_public_key, Ed25519PublicKey)
        or not isinstance(final_release_public_key, Ed25519PublicKey)
        or _REVISION.fullmatch(final_source_tree_oid or "") is None
        or _SHA256.fullmatch(final_package_sha256 or "") is None
        or _SHA256.fullmatch(inert_evidence_set_sha256 or "") is None
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error("owner_gate_deferred_mutation_iam_authority_invalid")
    try:
        expected_plan = foundation.build_plan(
            spec=plan.spec,
            network_evidence=final_network_evidence,
            network_collector_public_key=final_network_collector_public_key,
            now_unix=now_unix,
        )
        foundation_plan = foundation_apply_chain.foundation_a.plan
        foundation_receipt_sha256 = (
            foundation_apply_chain.foundation_apply_receipt_sha256
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        foundation.OwnerGateFoundationError,
        foundation_apply.OwnerGateFoundationApplyError,
    ) as exc:
        _error("owner_gate_deferred_mutation_iam_authority_invalid", exc)
    if (
        _canonical(expected_plan.report()) != _canonical(plan.report())
        or not plan.spec.final_release_bound
        or not foundation_plan.spec.pre_foundation_bound
        or plan.spec.release_revision
        == foundation_apply_chain.foundation_source_revision
        or _REVISION.fullmatch(plan.spec.release_revision or "") is None
        or _REVISION.fullmatch(
            foundation_apply_chain.foundation_source_revision or ""
        )
        is None
        or _REVISION.fullmatch(
            foundation_apply_chain.foundation_source_tree_oid or ""
        )
        is None
        or _stable_spec_projection(plan.spec)
        != _stable_spec_projection(foundation_plan.spec)
        or plan.spec.ancestry_evidence_sha256
        != foundation_plan.spec.ancestry_evidence_sha256
        or _SHA256.fullmatch(foundation_receipt_sha256 or "") is None
        or _SHA256.fullmatch(
            foundation_apply_chain.pre_foundation_authority_sha256 or ""
        )
        is None
    ):
        _error("owner_gate_deferred_mutation_iam_lineage_invalid")
    contract = _fixed_contract(plan)
    lineage = {
        "final_release_revision": plan.spec.release_revision,
        "final_source_tree_oid": final_source_tree_oid,
        "final_package_sha256": final_package_sha256,
        "final_plan_sha256": plan.sha256,
        "final_network_evidence_sha256": final_network_evidence.evidence_sha256,
        "final_network_collector_public_key_id": _sha256(
            final_network_collector_public_key.public_bytes_raw()
        ),
        "foundation_source_revision": (
            foundation_apply_chain.foundation_source_revision
        ),
        "foundation_source_tree_oid": (
            foundation_apply_chain.foundation_source_tree_oid
        ),
        "pre_foundation_authority_sha256": (
            foundation_apply_chain.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": foundation_receipt_sha256,
        "final_release_public_key_id": _sha256(
            final_release_public_key.public_bytes_raw()
        ),
        "inert_evidence_set_sha256": inert_evidence_set_sha256,
    }
    return _DeferredMutationIamAuthority._create(
        plan=plan,
        foundation_apply_chain=foundation_apply_chain,
        final_release_public_key=final_release_public_key,
        contract=contract,
        lineage=lineage,
    )


_ACTIVATION_AUTHORIZATION_MARKER = object()


@dataclass(frozen=True, init=False)
class _ActivationAuthorization:
    receipt: Mapping[str, Any]
    receipt_sha256: str
    runtime_sha256: str
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_ActivationAuthorization":
        _error(
            "owner_gate_deferred_mutation_iam_activation_authorization_"
            "factory_required"
        )

    @classmethod
    def _create(
        cls,
        *,
        receipt: Mapping[str, Any],
    ) -> "_ActivationAuthorization":
        value = object.__new__(cls)
        object.__setattr__(value, "receipt", dict(receipt))
        object.__setattr__(
            value,
            "receipt_sha256",
            str(receipt["owner_reauthentication_receipt_sha256"]),
        )
        object.__setattr__(
            value,
            "runtime_sha256",
            str(
                receipt["trusted_runtime_identity"][
                    "sealed_runtime_identity_sha256"
                ]
            ),
        )
        object.__setattr__(
            value,
            "_marker",
            _ACTIVATION_AUTHORIZATION_MARKER,
        )
        return value


def _validated_activation_authorization(
    *,
    authority: _DeferredMutationIamAuthority,
    receipt: Mapping[str, Any],
    expected_runtime_sha256: str,
    now_unix: int,
) -> _ActivationAuthorization:
    if (
        type(authority) is not _DeferredMutationIamAuthority
        or authority._marker is not _AUTHORITY_MARKER
        or not isinstance(receipt, Mapping)
        or _SHA256.fullmatch(expected_runtime_sha256 or "") is None
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error(
            "owner_gate_deferred_mutation_iam_activation_authorization_invalid"
        )
    try:
        checked = owner_reauth.validate_owner_reauth_receipt(
            receipt,
            public_key=authority.final_release_public_key,
            now_unix=now_unix,
        )
        runtime = checked["trusted_runtime_identity"]
    except (
        AttributeError,
        KeyError,
        TypeError,
        owner_reauth.OwnerGateOwnerReauthError,
    ) as exc:
        _error(
            "owner_gate_deferred_mutation_iam_activation_authorization_invalid",
            exc,
        )
    if (
        runtime["release_revision"] != authority.plan.spec.release_revision
        or runtime["sealed_runtime_identity_sha256"]
        != expected_runtime_sha256
    ):
        _error(
            "owner_gate_deferred_mutation_iam_activation_authorization_invalid"
        )
    return _ActivationAuthorization._create(receipt=checked)


def _authority_fields(
    authority: _DeferredMutationIamAuthority,
) -> Mapping[str, Any]:
    if (
        type(authority) is not _DeferredMutationIamAuthority
        or authority._marker is not _AUTHORITY_MARKER
        or _SHA256.fullmatch(authority.transaction_id or "") is None
    ):
        _error("owner_gate_deferred_mutation_iam_authority_invalid")
    return {
        "transaction_id": authority.transaction_id,
        **dict(authority.lineage),
        "resource_name": authority.contract.resource_name,
        "role": authority.contract.role,
        "member": authority.contract.member,
        "condition": dict(authority.contract.condition),
    }


def _normalized_policy(value: Any, *, resource_name: str) -> Mapping[str, Any]:
    normalized_fields = {
        "resource_name",
        "policy_etag",
        "policy_version",
        "policy_bindings",
        "policy_audit_configs",
    }
    candidate = value
    if isinstance(value, Mapping) and set(value) == normalized_fields:
        if value.get("resource_name") != resource_name:
            _error("owner_gate_deferred_mutation_iam_policy_invalid")
        candidate = {
            "etag": value.get("policy_etag"),
            "version": value.get("policy_version"),
            "bindings": value.get("policy_bindings"),
            "auditConfigs": value.get("policy_audit_configs"),
        }
    try:
        checked = foundation_apply._normalize_iam_policy(
            candidate,
            resource_name=resource_name,
        )
    except foundation_apply.OwnerGateFoundationApplyError as exc:
        _error("owner_gate_deferred_mutation_iam_policy_invalid", exc)
    if isinstance(value, Mapping) and set(value) == normalized_fields:
        if dict(checked) != dict(value):
            _error("owner_gate_deferred_mutation_iam_policy_invalid")
    return checked


def _api_policy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "etag": value["policy_etag"],
        "version": value["policy_version"],
        "bindings": value["policy_bindings"],
        "auditConfigs": value["policy_audit_configs"],
    }


def _binding_is_exact(
    value: Any,
    *,
    contract: DeferredMutationIamContract,
) -> bool:
    return isinstance(value, Mapping) and dict(value) == dict(contract.binding())


def _classify_policy(
    value: Mapping[str, Any],
    *,
    contract: DeferredMutationIamContract,
) -> str:
    policy = _normalized_policy(value, resource_name=contract.resource_name)
    bindings = policy["policy_bindings"]
    exact = [
        item for item in bindings if _binding_is_exact(item, contract=contract)
    ]
    ambiguous = []
    for item in bindings:
        if not isinstance(item, Mapping) or item in exact:
            continue
        members = item.get("members", [])
        if (
            item.get("role") == contract.role
            and (
                item.get("condition") == contract.condition
                or (
                    isinstance(members, list)
                    and contract.member in members
                )
            )
        ):
            ambiguous.append(item)
    if len(exact) == 1 and not ambiguous:
        return "exact"
    if exact or ambiguous:
        return "drift"
    return "absent"


def _edited_policy(
    precondition: Mapping[str, Any],
    *,
    contract: DeferredMutationIamContract,
    action: str,
) -> Mapping[str, Any]:
    if action not in _ACTIONS:
        _error("owner_gate_deferred_mutation_iam_action_invalid")
    normalized = _normalized_policy(
        _api_policy(precondition),
        resource_name=contract.resource_name,
    )
    if dict(normalized) != dict(precondition):
        _error("owner_gate_deferred_mutation_iam_precondition_invalid")
    state = _classify_policy(normalized, contract=contract)
    expected_source = "absent" if action == ACTION_ACTIVATE else "exact"
    if state != expected_source:
        _error("owner_gate_deferred_mutation_iam_precondition_invalid")
    bindings = [dict(item) for item in normalized["policy_bindings"]]
    if action == ACTION_ACTIVATE:
        bindings.append(dict(contract.binding()))
    else:
        bindings = [
            item
            for item in bindings
            if not _binding_is_exact(item, contract=contract)
        ]
    checked_bindings = foundation_apply._canonical_inventory(bindings)
    if not isinstance(checked_bindings, list):
        _error("owner_gate_deferred_mutation_iam_precondition_invalid")
    return {
        "etag": normalized["policy_etag"],
        "version": 3,
        "bindings": checked_bindings,
        "auditConfigs": normalized["policy_audit_configs"],
    }


@dataclass(frozen=True)
class DeferredMutationIamObservation:
    state: str
    policy: Mapping[str, Any] | None
    receipt_sha256: str

    def validate(self, *, contract: DeferredMutationIamContract) -> None:
        if (
            self.state not in _OBSERVATION_STATES
            or _SHA256.fullmatch(self.receipt_sha256 or "") is None
            or (self.state == "unknown") is not (self.policy is None)
        ):
            _error("owner_gate_deferred_mutation_iam_observation_invalid")
        if self.policy is not None:
            normalized = _normalized_policy(
                _api_policy(self.policy),
                resource_name=contract.resource_name,
            )
            if (
                dict(normalized) != dict(self.policy)
                or _classify_policy(normalized, contract=contract) != self.state
            ):
                _error("owner_gate_deferred_mutation_iam_observation_invalid")


class DeferredMutationIamProvider(Protocol):
    def assert_lineage(
        self,
        authority: _DeferredMutationIamAuthority,
        *,
        action: str,
    ) -> None: ...

    def observe_policy(
        self,
        authority: _DeferredMutationIamAuthority,
    ) -> DeferredMutationIamObservation: ...

    def mutate_policy(
        self,
        authority: _DeferredMutationIamAuthority,
        *,
        action: str,
        attempt_id: str,
        precondition: Mapping[str, Any],
        request_policy: Mapping[str, Any],
    ) -> foundation_apply.OperationObservation: ...


def _intent_attempt_id(
    *,
    transaction_id: Any,
    action: Any,
    activation_attempt_index: Any,
    intent_artifact_name: Any,
    precondition: Any,
    request_policy: Any,
    reauthentication_receipt_sha256: Any,
    paired_activation_success_receipt_sha256: Any,
) -> str:
    return _sha256_json({
        "transaction_id": transaction_id,
        "action": action,
        "activation_attempt_index": activation_attempt_index,
        "intent_artifact_name": intent_artifact_name,
        "precondition_sha256": _sha256_json(precondition),
        "request_policy_sha256": _sha256_json(request_policy),
        "owner_reauthentication_receipt_sha256": (
            reauthentication_receipt_sha256
        ),
        "paired_activation_success_receipt_sha256": (
            paired_activation_success_receipt_sha256
        ),
    })


def _build_intent(
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    observation: DeferredMutationIamObservation,
    activation_authorization: _ActivationAuthorization | None,
    activation_success: Mapping[str, Any] | None,
    activation_attempt_index: int | None,
) -> Mapping[str, Any]:
    allowed_sources = (
        {"absent"}
        if action == ACTION_ACTIVATE
        else {"exact", "absent"}
    )
    if observation.state not in allowed_sources or observation.policy is None:
        _error("owner_gate_deferred_mutation_iam_precondition_invalid")
    if action == ACTION_ACTIVATE:
        if (
            type(activation_authorization) is not _ActivationAuthorization
            or activation_authorization._marker
            is not _ACTIVATION_AUTHORIZATION_MARKER
            or activation_success is not None
            or type(activation_attempt_index) is not int
            or not 0 <= activation_attempt_index < MAX_ACTIVATION_ATTEMPTS
        ):
            _error(
                "owner_gate_deferred_mutation_iam_activation_authorization_invalid"
            )
        reauthentication_receipt: Mapping[str, Any] | None = dict(
            activation_authorization.receipt
        )
        reauthentication_receipt_sha256: str | None = (
            activation_authorization.receipt_sha256
        )
        reauthentication_runtime_sha256: str | None = (
            activation_authorization.runtime_sha256
        )
        paired_activation_success_receipt_sha256: str | None = None
    else:
        if (
            activation_authorization is not None
            or not isinstance(activation_success, Mapping)
            or _SHA256.fullmatch(
                str(activation_success.get("receipt_sha256", ""))
            )
            is None
            or activation_attempt_index is not None
        ):
            _error("owner_gate_deferred_mutation_iam_activation_missing")
        reauthentication_receipt = None
        reauthentication_receipt_sha256 = None
        reauthentication_runtime_sha256 = None
        paired_activation_success_receipt_sha256 = str(
            activation_success["receipt_sha256"]
        )
    request_policy = (
        _api_policy(observation.policy)
        if action == ACTION_REMOVE and observation.state == "absent"
        else _edited_policy(
            observation.policy,
            contract=authority.contract,
            action=action,
        )
    )
    attempt_id = _intent_attempt_id(
        transaction_id=authority.transaction_id,
        action=action,
        activation_attempt_index=activation_attempt_index,
        intent_artifact_name=_intent_artifact_name(
            action,
            activation_attempt_index,
        ),
        precondition=observation.policy,
        request_policy=request_policy,
        reauthentication_receipt_sha256=(
            reauthentication_receipt_sha256
        ),
        paired_activation_success_receipt_sha256=(
            paired_activation_success_receipt_sha256
        ),
    )
    unsigned = {
        "schema": INTENT_SCHEMA,
        "action": action,
        "activation_attempt_index": activation_attempt_index,
        "intent_artifact_name": _intent_artifact_name(
            action,
            activation_attempt_index,
        ),
        **_authority_fields(authority),
        "source_state": observation.state,
        "precondition": dict(observation.policy),
        "request_policy": dict(request_policy),
        "precondition_observation_receipt_sha256": observation.receipt_sha256,
        "activation_owner_reauthentication_receipt": (
            reauthentication_receipt
        ),
        "activation_owner_reauthentication_receipt_sha256": (
            reauthentication_receipt_sha256
        ),
        "activation_owner_reauthentication_runtime_sha256": (
            reauthentication_runtime_sha256
        ),
        "paired_activation_success_receipt_sha256": (
            paired_activation_success_receipt_sha256
        ),
        "attempt_id": attempt_id,
    }
    return {**unsigned, "intent_sha256": _sha256_json(unsigned)}


def _validate_intent(
    value: Any,
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_deferred_mutation_iam_intent_invalid")
    unsigned = {name: item for name, item in value.items() if name != "intent_sha256"}
    expected_fields = {
        "schema",
        "action",
        "activation_attempt_index",
        "intent_artifact_name",
        *_authority_fields(authority),
        "source_state",
        "precondition",
        "request_policy",
        "precondition_observation_receipt_sha256",
        "activation_owner_reauthentication_receipt",
        "activation_owner_reauthentication_receipt_sha256",
        "activation_owner_reauthentication_runtime_sha256",
        "paired_activation_success_receipt_sha256",
        "attempt_id",
        "intent_sha256",
    }
    # The starred mapping expression above contributes its keys to the set.
    if (
        set(value) != expected_fields
        or value.get("schema") != INTENT_SCHEMA
        or value.get("action") != action
        or value.get("intent_artifact_name")
        != _intent_artifact_name(
            action,
            (
                value.get("activation_attempt_index")
                if action == ACTION_ACTIVATE
                else None
            ),
        )
        or any(value.get(name) != item for name, item in _authority_fields(authority).items())
        or value.get("source_state")
        not in (
            {"absent"}
            if action == ACTION_ACTIVATE
            else {"exact", "absent"}
        )
        or _SHA256.fullmatch(str(value.get("attempt_id", ""))) is None
        or _SHA256.fullmatch(
            str(value.get("precondition_observation_receipt_sha256", ""))
        )
        is None
        or value.get("intent_sha256") != _sha256_json(unsigned)
        or not isinstance(value.get("precondition"), Mapping)
        or not isinstance(value.get("request_policy"), Mapping)
    ):
        _error("owner_gate_deferred_mutation_iam_intent_invalid")
    if action == ACTION_ACTIVATE:
        receipt = value.get("activation_owner_reauthentication_receipt")
        if (
            not isinstance(receipt, Mapping)
            or _SHA256.fullmatch(
                str(
                    value.get(
                        "activation_owner_reauthentication_receipt_sha256",
                        "",
                    )
                )
            )
            is None
            or _SHA256.fullmatch(
                str(
                    value.get(
                        "activation_owner_reauthentication_runtime_sha256",
                        "",
                    )
                )
            )
            is None
            or value.get("paired_activation_success_receipt_sha256")
            is not None
        ):
            _error("owner_gate_deferred_mutation_iam_intent_invalid")
        try:
            checked_reauthentication = (
                owner_reauth.validate_owner_reauth_receipt(
                    receipt,
                    public_key=authority.final_release_public_key,
                    now_unix=None,
                )
            )
        except owner_reauth.OwnerGateOwnerReauthError as exc:
            _error("owner_gate_deferred_mutation_iam_intent_invalid", exc)
        runtime = checked_reauthentication["trusted_runtime_identity"]
        if (
            checked_reauthentication[
                "owner_reauthentication_receipt_sha256"
            ]
            != value[
                "activation_owner_reauthentication_receipt_sha256"
            ]
            or runtime["sealed_runtime_identity_sha256"]
            != value[
                "activation_owner_reauthentication_runtime_sha256"
            ]
            or runtime["release_revision"]
            != authority.plan.spec.release_revision
        ):
            _error("owner_gate_deferred_mutation_iam_intent_invalid")
    elif (
        value.get("activation_owner_reauthentication_receipt") is not None
        or value.get(
            "activation_owner_reauthentication_receipt_sha256"
        )
        is not None
        or value.get(
            "activation_owner_reauthentication_runtime_sha256"
        )
        is not None
        or _SHA256.fullmatch(
            str(value.get("paired_activation_success_receipt_sha256", ""))
        )
        is None
    ):
        _error("owner_gate_deferred_mutation_iam_intent_invalid")
    precondition = _normalized_policy(
        _api_policy(value["precondition"]),
        resource_name=authority.contract.resource_name,
    )
    if (
        dict(precondition) != dict(value["precondition"])
        or value["request_policy"]
        != (
            _api_policy(precondition)
            if action == ACTION_REMOVE
            and value.get("source_state") == "absent"
            else _edited_policy(
                precondition,
                contract=authority.contract,
                action=action,
            )
        )
        or value["attempt_id"]
        != _intent_attempt_id(
            transaction_id=value["transaction_id"],
            action=value["action"],
            activation_attempt_index=value["activation_attempt_index"],
            intent_artifact_name=value["intent_artifact_name"],
            precondition=value["precondition"],
            request_policy=value["request_policy"],
            reauthentication_receipt_sha256=value[
                "activation_owner_reauthentication_receipt_sha256"
            ],
            paired_activation_success_receipt_sha256=value[
                "paired_activation_success_receipt_sha256"
            ],
        )
    ):
        _error("owner_gate_deferred_mutation_iam_intent_invalid")
    return dict(value)


def _operation_artifact(
    *,
    intent: Mapping[str, Any],
    operation: foundation_apply.OperationObservation,
) -> Mapping[str, Any]:
    operation.validate()
    unsigned = {
        "schema": OPERATION_SCHEMA,
        "action": intent["action"],
        "transaction_id": intent["transaction_id"],
        "intent_sha256": intent["intent_sha256"],
        "attempt_id": operation.attempt_id,
        "state": operation.state,
        "operation_receipt_sha256": operation.receipt_sha256,
        "provider_result_binding_sha256": operation.provider_result_binding_sha256,
        "cas_precondition_etag": operation.cas_precondition_etag,
        "cas_postcondition_etag": operation.cas_postcondition_etag,
    }
    return {**unsigned, "operation_sha256": _sha256_json(unsigned)}


def _validate_operation(
    value: Any,
    *,
    intent: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_deferred_mutation_iam_operation_invalid")
    unsigned = {
        name: item for name, item in value.items() if name != "operation_sha256"
    }
    if (
        set(value)
        != {
            "schema",
            "action",
            "transaction_id",
            "intent_sha256",
            "attempt_id",
            "state",
            "operation_receipt_sha256",
            "provider_result_binding_sha256",
            "cas_precondition_etag",
            "cas_postcondition_etag",
            "operation_sha256",
        }
        or value.get("schema") != OPERATION_SCHEMA
        or value.get("action") != intent.get("action")
        or value.get("transaction_id") != intent.get("transaction_id")
        or value.get("intent_sha256") != intent.get("intent_sha256")
        or value.get("attempt_id") != intent.get("attempt_id")
        or value.get("state") not in _OPERATION_STATES
        or _SHA256.fullmatch(str(value.get("operation_receipt_sha256", ""))) is None
        or value.get("operation_sha256") != _sha256_json(unsigned)
    ):
        _error("owner_gate_deferred_mutation_iam_operation_invalid")
    try:
        checked = foundation_apply.OperationObservation(
            str(value["state"]),
            str(value["operation_receipt_sha256"]),
            str(value["attempt_id"]),
            value.get("provider_result_binding_sha256"),
            value.get("cas_precondition_etag"),
            value.get("cas_postcondition_etag"),
        )
        checked.validate()
    except foundation_apply.OwnerGateFoundationApplyError as exc:
        _error("owner_gate_deferred_mutation_iam_operation_invalid", exc)
    expected_precondition_etag = intent.get("request_policy", {}).get(
        "etag"
    )
    if (
        not isinstance(expected_precondition_etag, str)
        or not expected_precondition_etag
        or checked.cas_precondition_etag != expected_precondition_etag
        or (
            checked.state == "completed"
            and (
                checked.cas_postcondition_etag is None
                or checked.cas_postcondition_etag
                == expected_precondition_etag
            )
        )
        or (
            checked.state != "completed"
            and checked.provider_result_binding_sha256 is not None
        )
        or (
            checked.state == "failed"
            and checked.cas_postcondition_etag is not None
        )
    ):
        _error("owner_gate_deferred_mutation_iam_operation_invalid")
    return dict(value)


def _success_operation_matrix(
    *,
    action: str,
    disposition: Any,
    operation: Mapping[str, Any] | None,
) -> bool:
    if disposition == "applied":
        return operation is not None and operation.get("state") == "completed"
    if disposition == "already_absent":
        return action == ACTION_REMOVE and operation is None
    if disposition == "reconciled_after_interruption":
        return operation is None or operation.get("state") in {
            "completed",
            "unknown",
        }
    return False


def _failure_operation_matrix(
    *,
    failure_code: Any,
    operation: Mapping[str, Any] | None,
) -> bool:
    expected_state = {
        "owner_gate_deferred_mutation_iam_provider_failed": "failed",
        "owner_gate_deferred_mutation_iam_postcondition_invalid": "completed",
    }.get(failure_code)
    return (
        expected_state is not None
        and operation is not None
        and operation.get("state") == expected_state
    )


def _success_receipt(
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    intent: Mapping[str, Any],
    operation: Mapping[str, Any] | None,
    observation: DeferredMutationIamObservation,
    disposition: str,
) -> Mapping[str, Any]:
    binding_present = action == ACTION_ACTIVATE
    expected_state = "exact" if binding_present else "absent"
    if (
        observation.state != expected_state
        or not _success_operation_matrix(
            action=action,
            disposition=disposition,
            operation=operation,
        )
    ):
        _error("owner_gate_deferred_mutation_iam_postcondition_invalid")
    unsigned = {
        "schema": SUCCESS_SCHEMA,
        "ok": True,
        "action": action,
        "activation_attempt_index": intent["activation_attempt_index"],
        "intent_artifact_name": intent["intent_artifact_name"],
        **_authority_fields(authority),
        "intent_sha256": intent["intent_sha256"],
        "activation_owner_reauthentication_receipt_sha256": intent[
            "activation_owner_reauthentication_receipt_sha256"
        ],
        "paired_activation_success_receipt_sha256": intent[
            "paired_activation_success_receipt_sha256"
        ],
        "operation_sha256": (
            operation["operation_sha256"] if operation is not None else None
        ),
        "postcondition_observation_receipt_sha256": observation.receipt_sha256,
        "disposition": disposition,
        "mutation_binding_present": binding_present,
        "cloud_mutation_performed": disposition != "already_absent",
        "foundation_steps_extended": False,
        "caller_selected_resource_accepted": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_success(
    value: Any,
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    operation: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_deferred_mutation_iam_success_invalid")
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    expected_fields = {
        "schema",
        "ok",
        "action",
        "activation_attempt_index",
        "intent_artifact_name",
        *_authority_fields(authority),
        "intent_sha256",
        "activation_owner_reauthentication_receipt_sha256",
        "paired_activation_success_receipt_sha256",
        "operation_sha256",
        "postcondition_observation_receipt_sha256",
        "disposition",
        "mutation_binding_present",
        "cloud_mutation_performed",
        "foundation_steps_extended",
        "caller_selected_resource_accepted",
        "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != SUCCESS_SCHEMA
        or value.get("ok") is not True
        or value.get("action") != action
        or value.get("intent_artifact_name")
        != _intent_artifact_name(
            action,
            (
                value.get("activation_attempt_index")
                if action == ACTION_ACTIVATE
                else None
            ),
        )
        or any(value.get(name) != item for name, item in _authority_fields(authority).items())
        or _SHA256.fullmatch(str(value.get("intent_sha256", ""))) is None
        or (
            action == ACTION_ACTIVATE
            and (
                _SHA256.fullmatch(
                    str(
                        value.get(
                            "activation_owner_reauthentication_receipt_sha256",
                            "",
                        )
                    )
                )
                is None
                or value.get("paired_activation_success_receipt_sha256")
                is not None
            )
        )
        or (
            action == ACTION_REMOVE
            and (
                value.get(
                    "activation_owner_reauthentication_receipt_sha256"
                )
                is not None
                or _SHA256.fullmatch(
                    str(
                        value.get(
                            "paired_activation_success_receipt_sha256",
                            "",
                        )
                    )
                )
                is None
            )
        )
        or value.get("operation_sha256")
        != (
            operation.get("operation_sha256")
            if operation is not None
            else None
        )
        or _SHA256.fullmatch(
            str(value.get("postcondition_observation_receipt_sha256", ""))
        )
        is None
        or not _success_operation_matrix(
            action=action,
            disposition=value.get("disposition"),
            operation=operation,
        )
        or value.get("mutation_binding_present")
        is not (action == ACTION_ACTIVATE)
        or value.get("cloud_mutation_performed")
        is not (value.get("disposition") != "already_absent")
        or value.get("foundation_steps_extended") is not False
        or value.get("caller_selected_resource_accepted") is not False
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        _error("owner_gate_deferred_mutation_iam_success_invalid")
    return dict(value)


def _failure_receipt(
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    intent: Mapping[str, Any],
    operation: Mapping[str, Any] | None,
    failure_code: str,
) -> Mapping[str, Any]:
    if not _failure_operation_matrix(
        failure_code=failure_code,
        operation=operation,
    ):
        _error("owner_gate_deferred_mutation_iam_failure_invalid")
    unsigned = {
        "schema": FAILURE_SCHEMA,
        "ok": False,
        "action": action,
        "activation_attempt_index": intent["activation_attempt_index"],
        "intent_artifact_name": intent["intent_artifact_name"],
        **_authority_fields(authority),
        "intent_sha256": intent["intent_sha256"],
        "activation_owner_reauthentication_receipt_sha256": intent[
            "activation_owner_reauthentication_receipt_sha256"
        ],
        "paired_activation_success_receipt_sha256": intent[
            "paired_activation_success_receipt_sha256"
        ],
        "operation_sha256": (
            operation["operation_sha256"] if operation is not None else None
        ),
        "failure_code": failure_code,
        "terminal_state": "manual_reconciliation_required",
        "foundation_steps_extended": False,
        "caller_selected_resource_accepted": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_failure(
    value: Any,
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    operation: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_deferred_mutation_iam_failure_invalid")
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    expected_fields = {
        "schema",
        "ok",
        "action",
        "activation_attempt_index",
        "intent_artifact_name",
        *_authority_fields(authority),
        "intent_sha256",
        "activation_owner_reauthentication_receipt_sha256",
        "paired_activation_success_receipt_sha256",
        "operation_sha256",
        "failure_code",
        "terminal_state",
        "foundation_steps_extended",
        "caller_selected_resource_accepted",
        "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != FAILURE_SCHEMA
        or value.get("ok") is not False
        or value.get("action") != action
        or value.get("intent_artifact_name")
        != _intent_artifact_name(
            action,
            (
                value.get("activation_attempt_index")
                if action == ACTION_ACTIVATE
                else None
            ),
        )
        or any(value.get(name) != item for name, item in _authority_fields(authority).items())
        or _SHA256.fullmatch(str(value.get("intent_sha256", ""))) is None
        or (
            action == ACTION_ACTIVATE
            and (
                _SHA256.fullmatch(
                    str(
                        value.get(
                            "activation_owner_reauthentication_receipt_sha256",
                            "",
                        )
                    )
                )
                is None
                or value.get("paired_activation_success_receipt_sha256")
                is not None
            )
        )
        or (
            action == ACTION_REMOVE
            and (
                value.get(
                    "activation_owner_reauthentication_receipt_sha256"
                )
                is not None
                or _SHA256.fullmatch(
                    str(
                        value.get(
                            "paired_activation_success_receipt_sha256",
                            "",
                        )
                    )
                )
                is None
            )
        )
        or value.get("operation_sha256")
        != (
            operation.get("operation_sha256")
            if operation is not None
            else None
        )
        or not _failure_operation_matrix(
            failure_code=value.get("failure_code"),
            operation=operation,
        )
        or value.get("terminal_state") != "manual_reconciliation_required"
        or value.get("foundation_steps_extended") is not False
        or value.get("caller_selected_resource_accepted") is not False
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        _error("owner_gate_deferred_mutation_iam_failure_invalid")
    return dict(value)


def _terminal_matches_intent(
    terminal: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> bool:
    return all(
        terminal.get(name) == intent.get(name)
        for name in (
            "intent_sha256",
            "activation_attempt_index",
            "intent_artifact_name",
            "activation_owner_reauthentication_receipt_sha256",
            "paired_activation_success_receipt_sha256",
        )
    )


def _require_remove_activation_pair(
    *,
    intent: Mapping[str, Any],
    activation_success: Mapping[str, Any] | None,
) -> None:
    if (
        activation_success is None
        or intent.get("paired_activation_success_receipt_sha256")
        != activation_success.get("receipt_sha256")
    ):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")


def _observe(
    provider: DeferredMutationIamProvider,
    authority: _DeferredMutationIamAuthority,
    *,
    action: str,
) -> DeferredMutationIamObservation:
    try:
        provider.assert_lineage(authority, action=action)
        observation = provider.observe_policy(authority)
    except OwnerGateDeferredMutationIamError:
        raise
    except Exception as exc:
        _error("owner_gate_deferred_mutation_iam_provider_unknown", exc)
    if type(observation) is not DeferredMutationIamObservation:
        _error("owner_gate_deferred_mutation_iam_observation_invalid")
    observation.validate(contract=authority.contract)
    return observation


def _publish_failure(
    *,
    journal: DeferredMutationIamJournal,
    authority: _DeferredMutationIamAuthority,
    action: str,
    intent: Mapping[str, Any],
    operation: Mapping[str, Any] | None,
    failure_code: str,
) -> Never:
    receipt = _failure_receipt(
        authority=authority,
        action=action,
        intent=intent,
        operation=operation,
        failure_code=failure_code,
    )
    try:
        stored = journal.publish(
            authority.transaction_id,
            f"{action}-failure",
            receipt,
        )
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_write_failed", exc)
    raise OwnerGateDeferredMutationIamFailed(
        _validate_failure(
            stored,
            authority=authority,
            action=action,
            operation=operation,
        )
    )


def _execute_with_provider(
    *,
    authority: _DeferredMutationIamAuthority,
    action: str,
    provider: DeferredMutationIamProvider,
    journal: DeferredMutationIamJournal,
    activation_authorization_factory: (
        Callable[[], _ActivationAuthorization] | None
    ) = None,
    activation_authorization_validator: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    activation_attempt_index: int | None = None,
    require_new_activation_intent: bool = False,
    require_contract_lease: bool = False,
) -> Mapping[str, Any]:
    if action == ACTION_ACTIVATE and activation_attempt_index is None:
        activation_attempt_index = 0
    if (
        type(authority) is not _DeferredMutationIamAuthority
        or authority._marker is not _AUTHORITY_MARKER
        or action not in _ACTIONS
        or not isinstance(journal, DeferredMutationIamJournal)
        or type(require_new_activation_intent) is not bool
        or type(require_contract_lease) is not bool
        or (
            action == ACTION_ACTIVATE
            and (
                not callable(activation_authorization_factory)
                or not callable(activation_authorization_validator)
            )
        )
        or (
            action == ACTION_REMOVE
            and (
                activation_attempt_index is not None
                or
                activation_authorization_factory is not None
                or activation_authorization_validator is not None
                or require_new_activation_intent
            )
        )
    ):
        _error("owner_gate_deferred_mutation_iam_boundary_invalid")
    try:
        with journal.transaction_lease(authority.transaction_id):
            if require_contract_lease:
                journal.require_contract_lease()
            artifacts = journal.list(authority.transaction_id)
            if action == ACTION_ACTIVATE and any(
                name.startswith(f"{ACTION_REMOVE}-") for name in artifacts
            ):
                _error("owner_gate_deferred_mutation_iam_lifecycle_complete")
            descriptor = _basic_activation_attempt_descriptor(
                artifacts=artifacts,
                transaction_id=authority.transaction_id,
                release_revision=authority.plan.spec.release_revision,
            )
            if action == ACTION_ACTIVATE:
                assert activation_attempt_index is not None
                requested_intent_name = _intent_artifact_name(
                    ACTION_ACTIVATE,
                    activation_attempt_index,
                )
                if require_new_activation_intent and (
                    requested_intent_name in artifacts
                    or (
                        descriptor is not None
                        and (
                            descriptor.index > activation_attempt_index
                            or descriptor.success is not None
                            or descriptor.failure is not None
                            or (
                                descriptor.index
                                == activation_attempt_index - 1
                                and _operation_artifact_name(
                                    ACTION_ACTIVATE,
                                    descriptor.index,
                                )
                                in artifacts
                            )
                        )
                    )
                ):
                    _error(
                        "owner_gate_deferred_mutation_iam_"
                        "fresh_attempt_conflict"
                    )
                if (
                    descriptor is not None
                    and descriptor.index
                    not in {
                        activation_attempt_index,
                        activation_attempt_index - 1,
                    }
                ):
                    _error("owner_gate_deferred_mutation_iam_attempt_invalid")
                if (
                    descriptor is None and activation_attempt_index != 0
                ):
                    _error("owner_gate_deferred_mutation_iam_attempt_invalid")
            activation_success: Mapping[str, Any] | None = None
            if action == ACTION_REMOVE:
                if descriptor is None:
                    _error("owner_gate_deferred_mutation_iam_activation_missing")
                activation_intent_raw = descriptor.intent
                activation_success_raw = descriptor.success
                activation_failure_raw = descriptor.failure
                activation_intent = _validate_intent(
                    activation_intent_raw,
                    authority=authority,
                    action=ACTION_ACTIVATE,
                )
                activation_operation_raw = artifacts.get(
                    _operation_artifact_name(
                        ACTION_ACTIVATE,
                        descriptor.index,
                    )
                )
                activation_operation = (
                    _validate_operation(
                        activation_operation_raw,
                        intent=activation_intent,
                    )
                    if activation_operation_raw is not None
                    else None
                )
                if activation_failure_raw is not None:
                    _error(
                        "owner_gate_deferred_mutation_iam_"
                        "manual_reconciliation_required"
                    )
                if activation_success_raw is None:
                    activation_observation = _observe(
                        provider,
                        authority,
                        action=action,
                    )
                    if activation_observation.state != "exact":
                        if activation_observation.state == "absent":
                            _error(
                                "owner_gate_deferred_mutation_iam_"
                                "activation_missing"
                            )
                        _error(
                            "owner_gate_deferred_mutation_iam_"
                            "manual_reconciliation_required"
                        )
                    activation_success_raw = journal.publish(
                        authority.transaction_id,
                        "activate-success",
                        _success_receipt(
                            authority=authority,
                            action=ACTION_ACTIVATE,
                            intent=activation_intent,
                            operation=activation_operation,
                            observation=activation_observation,
                            disposition="reconciled_after_interruption",
                        ),
                    )
                    artifacts["activate-success"] = activation_success_raw
                activation_success = _validate_success(
                    activation_success_raw,
                    authority=authority,
                    action=ACTION_ACTIVATE,
                    operation=activation_operation,
                )
                if (
                    activation_success["intent_sha256"]
                    != activation_intent["intent_sha256"]
                    or not _terminal_matches_intent(
                        activation_success,
                        activation_intent,
                    )
                ):
                    _error("owner_gate_deferred_mutation_iam_journal_invalid")
            success_name = f"{action}-success"
            failure_name = f"{action}-failure"
            intent_name = _intent_artifact_name(
                action,
                activation_attempt_index,
            )
            operation_name = _operation_artifact_name(
                action,
                activation_attempt_index,
            )
            success = artifacts.get(success_name)
            failure = artifacts.get(failure_name)
            intent_raw = artifacts.get(intent_name)
            operation_raw = artifacts.get(operation_name)
            if success is not None and failure is not None:
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
            if failure is not None:
                if intent_raw is None:
                    _error("owner_gate_deferred_mutation_iam_journal_invalid")
                failure_intent = _validate_intent(
                    intent_raw,
                    authority=authority,
                    action=action,
                )
                if action == ACTION_REMOVE:
                    _require_remove_activation_pair(
                        intent=failure_intent,
                        activation_success=activation_success,
                    )
                failure_operation = (
                    _validate_operation(
                        operation_raw,
                        intent=failure_intent,
                    )
                    if operation_raw is not None
                    else None
                )
                checked_failure = _validate_failure(
                    failure,
                    authority=authority,
                    action=action,
                    operation=failure_operation,
                )
                if not _terminal_matches_intent(
                    checked_failure,
                    failure_intent,
                ):
                    _error("owner_gate_deferred_mutation_iam_journal_invalid")
                raise OwnerGateDeferredMutationIamFailed(
                    checked_failure
                )
            desired_state = "exact" if action == ACTION_ACTIVATE else "absent"
            if success is not None:
                if intent_raw is None:
                    _error("owner_gate_deferred_mutation_iam_journal_invalid")
                success_intent = _validate_intent(
                    intent_raw,
                    authority=authority,
                    action=action,
                )
                if action == ACTION_REMOVE:
                    _require_remove_activation_pair(
                        intent=success_intent,
                        activation_success=activation_success,
                    )
                success_operation = (
                    _validate_operation(
                        operation_raw,
                        intent=success_intent,
                    )
                    if operation_raw is not None
                    else None
                )
                checked_success = _validate_success(
                    success,
                    authority=authority,
                    action=action,
                    operation=success_operation,
                )
                if not _terminal_matches_intent(
                    checked_success,
                    success_intent,
                ):
                    _error("owner_gate_deferred_mutation_iam_journal_invalid")
                if _observe(
                    provider,
                    authority,
                    action=action,
                ).state != desired_state:
                    _error(
                        "owner_gate_deferred_mutation_iam_manual_reconciliation_required"
                    )
                return checked_success
            if intent_raw is None:
                initial = _observe(provider, authority, action=action)
                if action == ACTION_ACTIVATE and initial.state == "exact":
                    _error(
                        "owner_gate_deferred_mutation_iam_unjournaled_binding_present"
                    )
                if action == ACTION_REMOVE and initial.state == "absent":
                    # A safe externally-completed rollback is still journaled as
                    # a paired no-op; no arbitrary binding can be removed.
                    intent = _build_intent(
                        authority=authority,
                        action=action,
                        observation=initial,
                        activation_authorization=None,
                        activation_success=activation_success,
                        activation_attempt_index=None,
                    )
                    stored_intent = journal.publish(
                        authority.transaction_id,
                        intent_name,
                        intent,
                    )
                    observation = initial
                    receipt = _success_receipt(
                        authority=authority,
                        action=action,
                        intent=stored_intent,
                        operation=None,
                        observation=observation,
                        disposition="already_absent",
                    )
                    return _validate_success(
                        journal.publish(
                            authority.transaction_id,
                            success_name,
                            receipt,
                        ),
                        authority=authority,
                        action=action,
                        operation=None,
                    )
                if initial.state not in {
                    "absent" if action == ACTION_ACTIVATE else "exact"
                }:
                    _error(
                        "owner_gate_deferred_mutation_iam_manual_reconciliation_required"
                    )
                if action == ACTION_ACTIVATE:
                    if activation_authorization_factory is None:
                        _error(
                            "owner_gate_deferred_mutation_iam_boundary_invalid"
                        )
                    activation_authorization = (
                        activation_authorization_factory()
                    )
                else:
                    activation_authorization = None
                intent = _build_intent(
                    authority=authority,
                    action=action,
                    observation=initial,
                    activation_authorization=activation_authorization,
                    activation_success=activation_success,
                    activation_attempt_index=activation_attempt_index,
                )
                intent = journal.publish(
                    authority.transaction_id,
                    intent_name,
                    intent,
                )
            else:
                intent = _validate_intent(
                    intent_raw,
                    authority=authority,
                    action=action,
                )
            intent = _validate_intent(
                intent,
                authority=authority,
                action=action,
            )
            if action == ACTION_REMOVE:
                _require_remove_activation_pair(
                    intent=intent,
                    activation_success=activation_success,
                )
            current = _observe(provider, authority, action=action)
            operation: Mapping[str, Any] | None = None
            if current.state == desired_state:
                disposition = "reconciled_after_interruption"
                if operation_raw is not None:
                    operation = _validate_operation(operation_raw, intent=intent)
                if operation is not None and operation["state"] == "failed":
                    _publish_failure(
                        journal=journal,
                        authority=authority,
                        action=action,
                        intent=intent,
                        operation=operation,
                        failure_code=(
                            "owner_gate_deferred_mutation_iam_provider_failed"
                        ),
                    )
                receipt = _success_receipt(
                    authority=authority,
                    action=action,
                    intent=intent,
                    operation=operation,
                    observation=current,
                    disposition=disposition,
                )
                return _validate_success(
                    journal.publish(
                        authority.transaction_id,
                        success_name,
                        receipt,
                    ),
                    authority=authority,
                    action=action,
                    operation=operation,
                )
            expected_source = "absent" if action == ACTION_ACTIVATE else "exact"
            if (
                current.state != expected_source
                or current.policy != intent["precondition"]
            ):
                _error(
                    "owner_gate_deferred_mutation_iam_manual_reconciliation_required"
                )
            if operation_raw is None:
                if action == ACTION_ACTIVATE:
                    assert activation_authorization_validator is not None
                    activation_authorization_validator(intent)
                if require_contract_lease:
                    journal.require_contract_lease()
                try:
                    observed_operation = provider.mutate_policy(
                        authority,
                        action=action,
                        attempt_id=str(intent["attempt_id"]),
                        precondition=intent["precondition"],
                        request_policy=intent["request_policy"],
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    _error(
                        "owner_gate_deferred_mutation_iam_manual_reconciliation_required",
                        exc,
                    )
                if type(observed_operation) is not foundation_apply.OperationObservation:
                    _error("owner_gate_deferred_mutation_iam_operation_invalid")
                observed_operation.validate()
                if observed_operation.state == "unknown":
                    post_unknown = _observe(
                        provider,
                        authority,
                        action=action,
                    )
                    if post_unknown.state != desired_state:
                        _error(
                            "owner_gate_deferred_mutation_iam_"
                            "manual_reconciliation_required"
                        )
                operation = journal.publish(
                    authority.transaction_id,
                    operation_name,
                    _operation_artifact(
                        intent=intent,
                        operation=observed_operation,
                    ),
                )
            else:
                operation = _validate_operation(operation_raw, intent=intent)
            operation = _validate_operation(operation, intent=intent)
            if operation["state"] == "failed":
                _publish_failure(
                    journal=journal,
                    authority=authority,
                    action=action,
                    intent=intent,
                    operation=operation,
                    failure_code="owner_gate_deferred_mutation_iam_provider_failed",
                )
            if operation["state"] == "unknown":
                post_unknown = _observe(
                    provider,
                    authority,
                    action=action,
                )
                if post_unknown.state != desired_state:
                    _error(
                        "owner_gate_deferred_mutation_iam_"
                        "manual_reconciliation_required"
                    )
                post = post_unknown
            else:
                post = _observe(provider, authority, action=action)
            if post.state != desired_state:
                _publish_failure(
                    journal=journal,
                    authority=authority,
                    action=action,
                    intent=intent,
                    operation=operation,
                    failure_code="owner_gate_deferred_mutation_iam_postcondition_invalid",
                )
            receipt = _success_receipt(
                authority=authority,
                action=action,
                intent=intent,
                operation=operation,
                observation=post,
                disposition=(
                    "applied"
                    if operation["state"] == "completed"
                    else "reconciled_after_interruption"
                ),
            )
            return _validate_success(
                journal.publish(
                    authority.transaction_id,
                    success_name,
                    receipt,
                ),
                authority=authority,
                action=action,
                operation=operation,
            )
    except OwnerGateDeferredMutationIamFailed:
        raise
    except OwnerGateDeferredMutationIamError:
        raise
    except (OSError, RuntimeError, PermissionError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_invalid", exc)


class _TrustedGcloudDeferredMutationIamProvider(
    foundation_apply._TrustedGcloudFoundationProvider
):
    """Exact owner gcloud provider; only the fixed final-plan binding exists."""

    def __init__(
        self,
        *,
        action: str,
        authority: _DeferredMutationIamAuthority,
        gcloud_executable: launcher.TrustedGcloudExecutable,
        gcloud_configuration: launcher.PinnedGcloudConfiguration,
        owner_identity: launcher.GcloudOwnerAccessToken,
    ) -> None:
        super().__init__(
            plan=authority.plan,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            expected_release_revision=authority.plan.spec.release_revision,
            runner=foundation_apply._SubprocessFoundationRunner(),
        )
        if action not in _ACTIONS:
            _error("owner_gate_deferred_mutation_iam_action_invalid")
        self._action = action
        self._authority = authority
        self._token_provider = owner_identity

    def assert_lineage(
        self,
        authority: _DeferredMutationIamAuthority,
        *,
        action: str,
    ) -> None:
        if authority is not self._authority or action != self._action:
            _error("owner_gate_deferred_mutation_iam_authority_changed")
        self.assert_stable()
        if action == ACTION_REMOVE:
            return
        foundation_apply._validate_live_ancestry(
            self,
            authority.foundation_apply_chain.foundation_a,
        )
        for name in (
            "create_dedicated_service_account",
            "create_narrow_storage_executor_role",
        ):
            matches = tuple(
                step for step in authority.plan.foundation_steps if step.name == name
            )
            if len(matches) != 1:
                _error("owner_gate_deferred_mutation_iam_lineage_invalid")
            observed = self.inspect_resource(matches[0], plan=authority.plan)
            observed.validate()
            if (
                observed.state != "exact"
                or observed.resource_identity
                != authority.foundation_apply_chain.resource_identity(name)
            ):
                _error("owner_gate_deferred_mutation_iam_lineage_invalid")
        self.assert_stable()

    def _read_policy_once(
        self,
        authority: _DeferredMutationIamAuthority,
    ) -> tuple[Mapping[str, Any], str]:
        value, receipt = self._read_json((
            "gcloud",
            "projects",
            "get-iam-policy",
            foundation.PROJECT,
            "--format=json",
        ))
        return (
            _normalized_policy(
                value,
                resource_name=authority.contract.resource_name,
            ),
            receipt,
        )

    def observe_policy(
        self,
        authority: _DeferredMutationIamAuthority,
    ) -> DeferredMutationIamObservation:
        if authority is not self._authority:
            _error("owner_gate_deferred_mutation_iam_authority_changed")
        try:
            first, first_receipt = self._read_policy_once(authority)
            second, second_receipt = self._read_policy_once(authority)
        except (
            OwnerGateDeferredMutationIamError,
            foundation_apply.OwnerGateFoundationApplyError,
        ):
            return DeferredMutationIamObservation(
                "unknown",
                None,
                _sha256_json({"state": "unknown", "read_count": 2}),
            )
        receipt = _sha256_json({
            "first_policy": first,
            "second_policy": second,
            "first_read_receipt_sha256": first_receipt,
            "second_read_receipt_sha256": second_receipt,
        })
        if first != second:
            return DeferredMutationIamObservation("unknown", None, receipt)
        return DeferredMutationIamObservation(
            _classify_policy(first, contract=authority.contract),
            first,
            receipt,
        )

    def mutate_policy(
        self,
        authority: _DeferredMutationIamAuthority,
        *,
        action: str,
        attempt_id: str,
        precondition: Mapping[str, Any],
        request_policy: Mapping[str, Any],
    ) -> foundation_apply.OperationObservation:
        if (
            authority is not self._authority
            or action != self._action
            or _SHA256.fullmatch(attempt_id or "") is None
            or request_policy
            != _edited_policy(
                precondition,
                contract=authority.contract,
                action=action,
            )
        ):
            _error("owner_gate_deferred_mutation_iam_operation_invalid")
        response = self._request_iam_policy_cas(
            resource_name=authority.contract.resource_name,
            policy=request_policy,
        )
        receipt = _sha256_json({
            "action": action,
            "attempt_id": attempt_id,
            "resource_name": authority.contract.resource_name,
            "request_policy_sha256": _sha256_json(request_policy),
            "update_mask": "bindings,etag",
            "http_status": response.status,
            "response_sha256": _sha256(response.body),
            "transport_unknown": response.transport_unknown,
        })
        pre_etag = str(request_policy["etag"])
        if (
            response.transport_unknown
            or response.status is None
            or response.status in {408, 409, 412, 425, 429}
            or 300 <= response.status < 400
            or response.status >= 500
        ):
            return foundation_apply.OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        if response.status < 200 or response.status >= 300:
            return foundation_apply.OperationObservation(
                "failed",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        try:
            output = foundation_apply._strict_provider_json(response.body)
            normalized = _normalized_policy(
                output,
                resource_name=authority.contract.resource_name,
            )
        except (
            OwnerGateDeferredMutationIamError,
            UnicodeError,
            ValueError,
        ):
            return foundation_apply.OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
            )
        expected = _normalized_policy(
            {**dict(request_policy), "etag": normalized["policy_etag"]},
            resource_name=authority.contract.resource_name,
        )
        post_etag = normalized["policy_etag"]
        if (
            not isinstance(post_etag, str)
            or not post_etag
            or post_etag == pre_etag
            or normalized != expected
        ):
            return foundation_apply.OperationObservation(
                "unknown",
                receipt,
                attempt_id,
                cas_precondition_etag=pre_etag,
                cas_postcondition_etag=(
                    str(post_etag) if isinstance(post_etag, str) else None
                ),
            )
        result_binding = _sha256_json({
            "action": action,
            "attempt_id": attempt_id,
            "resource_name": authority.contract.resource_name,
            "request_policy_sha256": _sha256_json(request_policy),
            "response_policy_sha256": _sha256_json(normalized),
            "operation_receipt_sha256": receipt,
        })
        return foundation_apply.OperationObservation(
            "completed",
            receipt,
            attempt_id,
            result_binding,
            pre_etag,
            str(post_etag),
        )


def _validate_owner_capabilities(
    *,
    authority: _DeferredMutationIamAuthority,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Any:
    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None)
        is not gcloud_executable
    ):
        _error("owner_gate_deferred_mutation_iam_capability_invalid")
    try:
        runtime = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=authority.plan.spec.release_revision
        )
        gcloud_configuration.assert_stable()
        if gcloud_configuration.account != owner_reauth.OWNER_ACCOUNT:
            _error("owner_gate_deferred_mutation_iam_owner_invalid")
        owner_identity.bind_approved_subject(
            _sha256(owner_reauth.OWNER_ACCOUNT.encode("ascii"))
        )
        owner_identity.require_stable()
        if (
            not isinstance(runtime, Mapping)
            or _SHA256.fullmatch(str(runtime.get("identity_sha256", "")))
            is None
        ):
            _error("owner_gate_deferred_mutation_iam_owner_invalid")
    except OwnerGateDeferredMutationIamError:
        raise
    except Exception as exc:
        _error("owner_gate_deferred_mutation_iam_capability_invalid", exc)
    return runtime


def _clock(now_unix: Callable[[], int]) -> int:
    value = now_unix()
    if type(value) is not int or value <= 0:
        _error("owner_gate_deferred_mutation_iam_time_invalid")
    return value


def _stable_release_lineage(
    *,
    binding: inert_observation._ReleaseBinding,
    loaded: inert_observation._LoadedFoundation,
) -> Mapping[str, Any]:
    try:
        lineage = {
            "final_release_revision": binding.release_revision,
            "final_source_tree_oid": binding.source_tree_oid,
            "final_package_sha256": binding.package["package_sha256"],
            "foundation_source_revision": loaded.chain.foundation_source_revision,
            "foundation_source_tree_oid": (
                loaded.chain.foundation_source_tree_oid
            ),
            "pre_foundation_authority_sha256": (
                loaded.chain.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                loaded.chain.foundation_apply_receipt_sha256
            ),
            "final_release_public_key_id": _sha256(
                binding.release_public_key.public_bytes_raw()
            ),
        }
    except (AttributeError, KeyError, TypeError) as exc:
        _error("owner_gate_deferred_mutation_iam_lineage_invalid", exc)
    _stable_transaction_id(
        contract=_fixed_contract_values(),
        lineage=lineage,
    )
    return lineage


@contextmanager
def _historical_attempt_authority(
    *,
    release_revision: str,
    transaction_id: str,
    intent: Mapping[str, Any],
    now_unix: int,
) -> Iterator[
    tuple[
        inert_observation._FrozenInertEvidence,
        _DeferredMutationIamAuthority,
        Mapping[str, Any],
    ]
]:
    evidence_set_sha256 = intent.get("inert_evidence_set_sha256")
    if _SHA256.fullmatch(str(evidence_set_sha256 or "")) is None:
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    with inert_observation._historical_inert_evidence_snapshot(
        release_revision=release_revision,
        evidence_set_sha256=str(evidence_set_sha256),
        now_unix=now_unix,
    ) as frozen:
        authority = _validated_authority(
            plan=frozen.plan,
            foundation_apply_chain=frozen.loaded.chain,
            final_network_evidence=frozen.network_evidence,
            final_network_collector_public_key=frozen.network_key,
            final_release_public_key=frozen.binding.release_public_key,
            final_source_tree_oid=frozen.binding.source_tree_oid,
            final_package_sha256=str(
                frozen.binding.package["package_sha256"]
            ),
            inert_evidence_set_sha256=str(evidence_set_sha256),
            now_unix=frozen.network_evidence.collected_at_unix,
        )
        if authority.transaction_id != transaction_id:
            _error("owner_gate_deferred_mutation_iam_lineage_invalid")
        checked_intent = _validate_intent(
            intent,
            authority=authority,
            action=ACTION_ACTIVATE,
        )
        yield frozen, authority, checked_intent


@contextmanager
def _historical_remove_authority(
    *,
    release_revision: str,
    journal: DeferredMutationIamJournal,
    now_unix: int,
) -> Iterator[
    tuple[
        inert_observation._FrozenInertEvidence,
        _DeferredMutationIamAuthority,
        _ActivationAttemptDescriptor,
    ]
]:
    """Recover the original activation authority without freshness adoption."""

    inputs = inert_observation._PinnedObservationInputs.load(release_revision)
    binding = inert_observation._load_release_binding(
        release_revision,
        inputs.bundle_stream,
    )
    loaded = inert_observation._load_successful_foundation(
        binding.foundation_source_revision
    )
    inert_observation._bind_release_to_foundation(binding, loaded)
    stable_lineage = _stable_release_lineage(binding=binding, loaded=loaded)
    transaction_id = _stable_transaction_id(
        contract=_fixed_contract_values(),
        lineage=stable_lineage,
    )
    try:
        artifacts = journal.list(transaction_id)
    except (OSError, PermissionError, RuntimeError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_invalid", exc)
    descriptor = _basic_activation_attempt_descriptor(
        artifacts=artifacts,
        transaction_id=transaction_id,
        release_revision=release_revision,
    )
    if descriptor is None:
        _error("owner_gate_deferred_mutation_iam_activation_missing")
    inputs.assert_stable()
    for index, _name, prior_intent in descriptor.attempts[:-1]:
        with _historical_attempt_authority(
            release_revision=release_revision,
            transaction_id=transaction_id,
            intent=prior_intent,
            now_unix=now_unix,
        ) as (_prior_frozen, _prior_authority, checked_prior):
            prior_operation = artifacts.get(
                _operation_artifact_name(ACTION_ACTIVATE, index)
            )
            if prior_operation is not None:
                _validate_operation(
                    prior_operation,
                    intent=checked_prior,
                )
    with _historical_attempt_authority(
        release_revision=release_revision,
        transaction_id=transaction_id,
        intent=descriptor.intent,
        now_unix=now_unix,
    ) as (frozen, authority, activation_intent):
        selected_operation_raw = artifacts.get(
            _operation_artifact_name(
                ACTION_ACTIVATE,
                descriptor.index,
            )
        )
        selected_operation = (
            _validate_operation(
                selected_operation_raw,
                intent=activation_intent,
            )
            if selected_operation_raw is not None
            else None
        )
        if descriptor.success is not None:
            checked_success = _validate_success(
                descriptor.success,
                authority=authority,
                action=ACTION_ACTIVATE,
                operation=selected_operation,
            )
            if not _terminal_matches_intent(
                checked_success,
                activation_intent,
            ):
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
        if descriptor.failure is not None:
            checked_failure = _validate_failure(
                descriptor.failure,
                authority=authority,
                action=ACTION_ACTIVATE,
                operation=selected_operation,
            )
            if not _terminal_matches_intent(
                checked_failure,
                activation_intent,
            ):
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
        yield frozen, authority, descriptor


def _release_transaction_id(release_revision: str) -> str:
    """Resolve R's fixed-contract transaction without adopting live IAM state."""

    inputs = inert_observation._PinnedObservationInputs.load(release_revision)
    binding = inert_observation._load_release_binding(
        release_revision,
        inputs.bundle_stream,
    )
    loaded = inert_observation._load_successful_foundation(
        binding.foundation_source_revision
    )
    inert_observation._bind_release_to_foundation(binding, loaded)
    transaction_id = _stable_transaction_id(
        contract=_fixed_contract_values(),
        lineage=_stable_release_lineage(binding=binding, loaded=loaded),
    )
    inputs.assert_stable()
    return transaction_id


def _journal_release_revision(
    *,
    transaction_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Validate enough immutable identity to route historical validation."""

    if not artifacts:
        return None
    base = artifacts.get("activate-intent")
    if not isinstance(base, Mapping):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    contract = _fixed_contract_values()
    if any(
        base.get(name) != expected
        for name, expected in (
            ("resource_name", contract.resource_name),
            ("role", contract.role),
            ("member", contract.member),
            ("condition", contract.condition),
        )
    ):
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    lineage = {name: base.get(name) for name in _STABLE_LIFECYCLE_FIELDS}
    if _stable_transaction_id(contract=contract, lineage=lineage) != transaction_id:
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    release_revision = str(base.get("final_release_revision", ""))
    if _REVISION.fullmatch(release_revision) is None:
        _error("owner_gate_deferred_mutation_iam_journal_invalid")
    return release_revision


def _other_lifecycle_released(
    *,
    transaction_id: str,
    release_revision: str,
    journal: DeferredMutationIamJournal,
    now_unix: int,
) -> bool:
    """Return true only for a fully validated, activation-paired removal."""

    with _historical_remove_authority(
        release_revision=release_revision,
        journal=journal,
        now_unix=now_unix,
    ) as (_frozen, authority, descriptor):
        if authority.transaction_id != transaction_id:
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        artifacts = journal.list(transaction_id)
        current_descriptor = _basic_activation_attempt_descriptor(
            artifacts=artifacts,
            transaction_id=transaction_id,
            release_revision=release_revision,
        )
        if current_descriptor is None or current_descriptor != descriptor:
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        activation_intent = _validate_intent(
            descriptor.intent,
            authority=authority,
            action=ACTION_ACTIVATE,
        )
        activation_operation_raw = artifacts.get(
            _operation_artifact_name(ACTION_ACTIVATE, descriptor.index)
        )
        activation_operation = (
            _validate_operation(
                activation_operation_raw,
                intent=activation_intent,
            )
            if activation_operation_raw is not None
            else None
        )
        activation_success = (
            _validate_success(
                descriptor.success,
                authority=authority,
                action=ACTION_ACTIVATE,
                operation=activation_operation,
            )
            if descriptor.success is not None
            else None
        )
        if activation_success is not None and not _terminal_matches_intent(
            activation_success,
            activation_intent,
        ):
            _error("owner_gate_deferred_mutation_iam_journal_invalid")

        remove_intent_raw = artifacts.get("remove-intent")
        remove_operation_raw = artifacts.get("remove-operation")
        remove_success_raw = artifacts.get("remove-success")
        remove_failure_raw = artifacts.get("remove-failure")
        if remove_intent_raw is None:
            if any(
                item is not None
                for item in (
                    remove_operation_raw,
                    remove_success_raw,
                    remove_failure_raw,
                )
            ):
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
            return False
        if remove_success_raw is not None and remove_failure_raw is not None:
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        remove_intent = _validate_intent(
            remove_intent_raw,
            authority=authority,
            action=ACTION_REMOVE,
        )
        _require_remove_activation_pair(
            intent=remove_intent,
            activation_success=activation_success,
        )
        remove_operation = (
            _validate_operation(
                remove_operation_raw,
                intent=remove_intent,
            )
            if remove_operation_raw is not None
            else None
        )
        if remove_failure_raw is not None:
            checked_failure = _validate_failure(
                remove_failure_raw,
                authority=authority,
                action=ACTION_REMOVE,
                operation=remove_operation,
            )
            if not _terminal_matches_intent(checked_failure, remove_intent):
                _error("owner_gate_deferred_mutation_iam_journal_invalid")
        if remove_success_raw is None:
            return False
        checked_success = _validate_success(
            remove_success_raw,
            authority=authority,
            action=ACTION_REMOVE,
            operation=remove_operation,
        )
        if not _terminal_matches_intent(checked_success, remove_intent):
            _error("owner_gate_deferred_mutation_iam_journal_invalid")
        return True


def _assert_contract_owner_available(
    *,
    current_transaction_id: str,
    journal: DeferredMutationIamJournal,
    now_unix: int,
) -> None:
    """Reject dual ownership of the one global binding across releases."""

    if _SHA256.fullmatch(current_transaction_id) is None:
        _error("owner_gate_deferred_mutation_iam_lineage_invalid")
    try:
        transaction_ids = journal.transaction_ids()
        for transaction_id in transaction_ids:
            if transaction_id == current_transaction_id:
                continue
            with journal.transaction_lease(transaction_id):
                artifacts = journal.list(transaction_id)
                release_revision = _journal_release_revision(
                    transaction_id=transaction_id,
                    artifacts=artifacts,
                )
                if release_revision is None:
                    continue
                if not _other_lifecycle_released(
                    transaction_id=transaction_id,
                    release_revision=release_revision,
                    journal=journal,
                    now_unix=now_unix,
                ):
                    _error(
                        "owner_gate_deferred_mutation_iam_contract_owned"
                    )
        journal.require_contract_lease()
    except OwnerGateDeferredMutationIamError:
        raise
    except (OSError, PermissionError, RuntimeError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_invalid", exc)


@contextmanager
def _fresh_activation_authority_context(
    *,
    release_revision: str,
    now_unix: int,
) -> Iterator[
    tuple[
        inert_observation._FrozenInertEvidence,
        _DeferredMutationIamAuthority,
        _ActivationAttemptDescriptor | None,
    ]
]:
    with inert_observation._fresh_inert_evidence_snapshot(
        release_revision=release_revision,
        now_unix=now_unix,
    ) as frozen:
        authority = _validated_authority(
            plan=frozen.plan,
            foundation_apply_chain=frozen.loaded.chain,
            final_network_evidence=frozen.network_evidence,
            final_network_collector_public_key=frozen.network_key,
            final_release_public_key=frozen.binding.release_public_key,
            final_source_tree_oid=frozen.binding.source_tree_oid,
            final_package_sha256=str(
                frozen.binding.package["package_sha256"]
            ),
            inert_evidence_set_sha256=str(
                frozen.receipt["evidence_set_sha256"]
            ),
            now_unix=now_unix,
        )
        yield frozen, authority, None


@contextmanager
def _pathless_authority_context(
    *,
    action: str,
    release_revision: str,
    journal: DeferredMutationIamJournal,
    now_unix: int,
) -> Iterator[
    tuple[
        inert_observation._FrozenInertEvidence,
        _DeferredMutationIamAuthority,
        _ActivationAttemptDescriptor | None,
    ]
]:
    try:
        with _historical_remove_authority(
            release_revision=release_revision,
            journal=journal,
            now_unix=now_unix,
        ) as recovered:
            yield recovered
        return
    except OwnerGateDeferredMutationIamError as exc:
        if (
            action != ACTION_ACTIVATE
            or str(exc)
            != "owner_gate_deferred_mutation_iam_activation_missing"
        ):
            raise
    with _fresh_activation_authority_context(
        release_revision=release_revision,
        now_unix=now_unix,
    ) as fresh:
        yield fresh


def _execute_pathless(
    *,
    action: str,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    reauth_runner: owner_reauth.OwnerReauthRunner | None,
    now_unix: Callable[[], int] = lambda: int(time.time()),
    journal: DeferredMutationIamJournal | None = None,
) -> Mapping[str, Any]:
    if (
        action not in _ACTIONS
        or _REVISION.fullmatch(release_revision or "") is None
        or type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration)
        is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None)
        is not gcloud_executable
        or (action == ACTION_ACTIVATE and reauth_runner is None)
        or (action == ACTION_REMOVE and reauth_runner is not None)
        or (journal is not None and not isinstance(journal, DeferredMutationIamJournal))
    ):
        _error("owner_gate_deferred_mutation_iam_capability_invalid")
    selected_journal = journal or DeferredMutationIamJournal()
    try:
        with selected_journal.contract_lease():
            locked_at = _clock(now_unix)
            expected_transaction_id = _release_transaction_id(
                release_revision
            )
            _assert_contract_owner_available(
                current_transaction_id=expected_transaction_id,
                journal=selected_journal,
                now_unix=locked_at,
            )
            return _execute_pathless_under_contract_lease(
                action=action,
                release_revision=release_revision,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
                owner_identity=owner_identity,
                reauth_runner=reauth_runner,
                now_unix=now_unix,
                selected_journal=selected_journal,
                expected_transaction_id=expected_transaction_id,
            )
    except OwnerGateDeferredMutationIamError:
        raise
    except (OSError, PermissionError, RuntimeError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_invalid", exc)


def _execute_pathless_under_contract_lease(
    *,
    action: str,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    reauth_runner: owner_reauth.OwnerReauthRunner | None,
    now_unix: Callable[[], int],
    selected_journal: DeferredMutationIamJournal,
    expected_transaction_id: str,
) -> Mapping[str, Any]:
    if (
        action not in _ACTIONS
        or _REVISION.fullmatch(release_revision or "") is None
        or type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration)
        is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None)
        is not gcloud_executable
        or (action == ACTION_ACTIVATE and reauth_runner is None)
        or (action == ACTION_REMOVE and reauth_runner is not None)
        or not isinstance(selected_journal, DeferredMutationIamJournal)
        or _SHA256.fullmatch(expected_transaction_id or "") is None
    ):
        _error("owner_gate_deferred_mutation_iam_capability_invalid")
    try:
        selected_journal.require_contract_lease()
    except (OSError, PermissionError, RuntimeError) as exc:
        _error("owner_gate_deferred_mutation_iam_journal_invalid", exc)
    launcher.require_trusted_owner_support_activation(
        gcloud_executable,
        release_sha=release_revision,
    )
    launcher.require_local_launcher_provenance(release_revision)
    started_at = _clock(now_unix)
    runtime_before: Any = None
    authority: _DeferredMutationIamAuthority | None = None
    retry_activation = object()
    retry_descriptor: _ActivationAttemptDescriptor | None = None
    try:
        def execute_bound(
            *,
            frozen: inert_observation._FrozenInertEvidence,
            bound_authority: _DeferredMutationIamAuthority,
            descriptor: _ActivationAttemptDescriptor | None,
            activation_attempt_index: int | None,
            historical_activation: bool,
        ) -> Mapping[str, Any] | object:
            nonlocal authority, retry_descriptor, runtime_before
            if bound_authority.transaction_id != expected_transaction_id:
                _error(
                    "owner_gate_deferred_mutation_iam_contract_owner_changed"
                )
            authority = bound_authority
            runtime = _validate_owner_capabilities(
                authority=authority,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
                owner_identity=owner_identity,
            )
            if runtime_before is None:
                runtime_before = runtime
            elif runtime_before != runtime:
                _error(
                    "owner_gate_deferred_mutation_iam_capability_changed"
                )
            provider = _TrustedGcloudDeferredMutationIamProvider(
                action=action,
                authority=authority,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
                owner_identity=owner_identity,
            )
            if historical_activation:
                if descriptor is None:
                    _error(
                        "owner_gate_deferred_mutation_iam_journal_invalid"
                    )
                if descriptor.failure is not None:
                    historical_intent = _validate_intent(
                        descriptor.intent,
                        authority=authority,
                        action=ACTION_ACTIVATE,
                    )
                    historical_artifacts = selected_journal.list(
                        authority.transaction_id
                    )
                    historical_operation_raw = historical_artifacts.get(
                        _operation_artifact_name(
                            ACTION_ACTIVATE,
                            descriptor.index,
                        )
                    )
                    historical_operation = (
                        _validate_operation(
                            historical_operation_raw,
                            intent=historical_intent,
                        )
                        if historical_operation_raw is not None
                        else None
                    )
                    checked_failure = _validate_failure(
                        descriptor.failure,
                        authority=authority,
                        action=ACTION_ACTIVATE,
                        operation=historical_operation,
                    )
                    raise OwnerGateDeferredMutationIamFailed(
                        checked_failure
                    )
                current = _observe(
                    provider,
                    authority,
                    action=ACTION_ACTIVATE,
                )
                if descriptor.success is not None and current.state != "exact":
                    _error(
                        "owner_gate_deferred_mutation_iam_"
                        "manual_reconciliation_required"
                    )
                historical_artifacts = selected_journal.list(
                    authority.transaction_id
                )
                historical_operation_present = (
                    _operation_artifact_name(
                        ACTION_ACTIVATE,
                        descriptor.index,
                    )
                    in historical_artifacts
                )
                if (
                    current.state == "absent"
                    and not historical_operation_present
                ):
                    retry_descriptor = descriptor
                    return retry_activation
                if current.state not in {"absent", "exact"}:
                    _error(
                        "owner_gate_deferred_mutation_iam_"
                        "manual_reconciliation_required"
                    )

                def reject_historical_authorization() -> _ActivationAuthorization:
                    _error(
                        "owner_gate_deferred_mutation_iam_fresh_attempt_required"
                    )

                def reject_historical_write(
                    _intent: Mapping[str, Any],
                ) -> None:
                    _error(
                        "owner_gate_deferred_mutation_iam_fresh_attempt_required"
                    )

                result = _execute_with_provider(
                    authority=authority,
                    action=ACTION_ACTIVATE,
                    provider=provider,
                    journal=selected_journal,
                    activation_authorization_factory=(
                        reject_historical_authorization
                    ),
                    activation_authorization_validator=(
                        reject_historical_write
                    ),
                    activation_attempt_index=descriptor.index,
                    require_contract_lease=True,
                )
                frozen.inputs.assert_stable()
                return result

            def author_activation() -> _ActivationAuthorization:
                if reauth_runner is None:
                    _error(
                        "owner_gate_deferred_mutation_iam_activation_"
                        "authorization_invalid"
                    )
                try:
                    receipt = owner_reauth.produce_owner_reauth_receipt(
                        runner=reauth_runner,
                        private_key=inert_observation._release_private_key(
                            frozen.binding
                        ),
                        now_unix=lambda: _clock(now_unix),
                        gcloud_executable=gcloud_executable,
                        gcloud_configuration=gcloud_configuration,
                        expected_release_revision=release_revision,
                    )
                except (
                    launcher.OwnerLauncherError,
                    owner_reauth.OwnerGateOwnerReauthError,
                ) as exc:
                    _error(
                        "owner_gate_deferred_mutation_iam_activation_"
                        "authorization_invalid",
                        exc,
                    )
                checked_at = _clock(now_unix)
                checked_runtime = _validate_owner_capabilities(
                    authority=bound_authority,
                    gcloud_executable=gcloud_executable,
                    gcloud_configuration=gcloud_configuration,
                    owner_identity=owner_identity,
                )
                frozen.assert_stable(now_unix=checked_at)
                return _validated_activation_authorization(
                    authority=bound_authority,
                    receipt=receipt,
                    expected_runtime_sha256=str(
                        checked_runtime["identity_sha256"]
                    ),
                    now_unix=checked_at,
                )

            def validate_activation_intent(
                intent: Mapping[str, Any],
            ) -> None:
                checked_at = _clock(now_unix)
                checked_runtime = _validate_owner_capabilities(
                    authority=bound_authority,
                    gcloud_executable=gcloud_executable,
                    gcloud_configuration=gcloud_configuration,
                    owner_identity=owner_identity,
                )
                authorization = _validated_activation_authorization(
                    authority=bound_authority,
                    receipt=intent.get(
                        "activation_owner_reauthentication_receipt",
                        {},
                    ),
                    expected_runtime_sha256=str(
                        checked_runtime["identity_sha256"]
                    ),
                    now_unix=checked_at,
                )
                if (
                    authorization.receipt_sha256
                    != intent.get(
                        "activation_owner_reauthentication_receipt_sha256"
                    )
                    or authorization.runtime_sha256
                    != intent.get(
                        "activation_owner_reauthentication_runtime_sha256"
                    )
                ):
                    _error(
                        "owner_gate_deferred_mutation_iam_activation_"
                        "authorization_invalid"
                    )
                frozen.assert_stable(now_unix=checked_at)

            result = _execute_with_provider(
                authority=authority,
                action=action,
                provider=provider,
                journal=selected_journal,
                activation_authorization_factory=(
                    author_activation if action == ACTION_ACTIVATE else None
                ),
                activation_authorization_validator=(
                    validate_activation_intent
                    if action == ACTION_ACTIVATE
                    else None
                ),
                activation_attempt_index=activation_attempt_index,
                require_new_activation_intent=(
                    action == ACTION_ACTIVATE
                ),
                require_contract_lease=True,
            )
            frozen.inputs.assert_stable()
            return result

        def reconcile_latest() -> Mapping[str, Any] | object:
            reconcile_now = _clock(now_unix)
            with _historical_remove_authority(
                release_revision=release_revision,
                journal=selected_journal,
                now_unix=reconcile_now,
            ) as (historical, historical_authority, latest):
                return execute_bound(
                    frozen=historical,
                    bound_authority=historical_authority,
                    descriptor=latest,
                    activation_attempt_index=latest.index,
                    historical_activation=True,
                )

        restart_codes = {
            "owner_gate_deferred_mutation_iam_fresh_attempt_conflict",
            "owner_gate_deferred_mutation_iam_unjournaled_binding_present",
        }
        try:
            with _pathless_authority_context(
                action=action,
                release_revision=release_revision,
                journal=selected_journal,
                now_unix=started_at,
            ) as (frozen, bound_authority, descriptor):
                outcome = execute_bound(
                    frozen=frozen,
                    bound_authority=bound_authority,
                    descriptor=descriptor,
                    activation_attempt_index=(
                        descriptor.index if descriptor is not None else 0
                    )
                    if action == ACTION_ACTIVATE
                    else None,
                    historical_activation=(
                        action == ACTION_ACTIVATE
                        and descriptor is not None
                    ),
                )
        except OwnerGateDeferredMutationIamError as exc:
            if action != ACTION_ACTIVATE or str(exc) not in restart_codes:
                raise
            outcome = reconcile_latest()
        while outcome is retry_activation:
            descriptor_for_retry = retry_descriptor
            if descriptor_for_retry is None:
                _error("owner_gate_deferred_mutation_iam_boundary_invalid")
            next_attempt = descriptor_for_retry.index + 1
            if next_attempt >= MAX_ACTIVATION_ATTEMPTS:
                _error("owner_gate_deferred_mutation_iam_attempts_exhausted")
            retry_now = _clock(now_unix)
            try:
                with _fresh_activation_authority_context(
                    release_revision=release_revision,
                    now_unix=retry_now,
                ) as (fresh, fresh_authority, _fresh_descriptor):
                    outcome = execute_bound(
                        frozen=fresh,
                        bound_authority=fresh_authority,
                        descriptor=None,
                        activation_attempt_index=next_attempt,
                        historical_activation=False,
                    )
            except OwnerGateDeferredMutationIamError as exc:
                if str(exc) not in restart_codes:
                    raise
                outcome = reconcile_latest()
        if not isinstance(outcome, Mapping):
            _error("owner_gate_deferred_mutation_iam_boundary_invalid")
        return cast(Mapping[str, Any], outcome)
    finally:
        failures: list[BaseException] = []
        runtime_after: Any = None
        for check in (
            gcloud_configuration.assert_stable,
            owner_identity.require_stable,
        ):
            try:
                check()
            except BaseException as exc:
                failures.append(exc)
        if authority is not None:
            try:
                runtime_after = gcloud_executable.sealed_runtime_identity(
                    expected_release_sha=release_revision
                )
            except BaseException as exc:
                failures.append(exc)
        try:
            launcher.require_trusted_owner_support_activation(
                gcloud_executable,
                release_sha=release_revision,
            )
            launcher.require_local_launcher_provenance(release_revision)
        except BaseException as exc:
            failures.append(exc)
        if failures or (
            authority is not None and runtime_after != runtime_before
        ):
            _error("owner_gate_deferred_mutation_iam_capability_changed")


def activate_deferred_mutation_iam(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Mapping[str, Any]:
    """Install only the final plan's fixed conditioned mutation binding."""

    return _execute_pathless(
        action=ACTION_ACTIVATE,
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
        reauth_runner=owner_reauth.SubprocessOwnerReauthRunner(),
    )


def remove_deferred_mutation_iam(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Mapping[str, Any]:
    """Remove only the binding proven owned by the paired activation receipt."""

    return _execute_pathless(
        action=ACTION_REMOVE,
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
        reauth_runner=None,
    )


__all__ = [
    "DeferredMutationIamJournal",
    "OwnerGateDeferredMutationIamError",
    "OwnerGateDeferredMutationIamFailed",
    "activate_deferred_mutation_iam",
    "remove_deferred_mutation_iam",
]
