"""The canonical transaction owner for installed Ares runtime activation."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterator, Protocol

from .contracts import (
    ActivationGrant,
    InstalledRuntimePointer,
    ReleaseReference,
    RuntimeIdentity,
    canonical_json,
    sha256_bytes,
)
from .errors import AresRuntimeError
from .layout import AresRuntimeLayout
from .materializer import MaterializedRelease, materialize_from_candidate_store


class ActivationState(StrEnum):
    CREATED = "CREATED"
    PRECHECKING = "PRECHECKING"
    PRECHECK_PASSED = "PRECHECK_PASSED"
    MATERIALIZING = "MATERIALIZING"
    MATERIALIZED = "MATERIALIZED"
    QUIESCING = "QUIESCING"
    QUIESCED = "QUIESCED"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    SWITCH_COMMITTED = "SWITCH_COMMITTED"
    STARTING = "STARTING"
    HEALTH_CHECKING = "HEALTH_CHECKING"
    LIVE_CERTIFYING = "LIVE_CERTIFYING"
    ACTIVATED = "ACTIVATED"
    ABORTED_PRE_COMMIT = "ABORTED_PRE_COMMIT"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    INCIDENT_HELD = "INCIDENT_HELD"


class RuntimeSupervisor(Protocol):
    """Platform adapter; it owns service and child-process mechanics."""

    def quiesce(self, transaction_id: str) -> None: ...

    def start(self, release: ReleaseReference) -> None: ...

    def health(self, release: ReleaseReference, generation: int) -> RuntimeIdentity: ...


Materializer = Callable[
    [object, AresRuntimeLayout, str, ActivationGrant], MaterializedRelease
]


@dataclass(frozen=True)
class ActivationResult:
    transaction_id: str
    state: ActivationState
    pointer: InstalledRuntimePointer


class AresReleaseActivator:
    """Coordinates exact authorization, materialization, switching, and rollback."""

    def __init__(
        self,
        *,
        store: object,
        layout: AresRuntimeLayout,
        supervisor: RuntimeSupervisor,
        materializer: Materializer = materialize_from_candidate_store,
    ) -> None:
        self.store = store
        self.layout = layout
        self.supervisor = supervisor
        self.materializer = materializer

    @contextmanager
    def _activation_lock(self) -> Iterator[None]:
        self.layout.initialize()
        descriptor = os.open(
            self.layout.activation_lock_path,
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
            0o600,
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AresRuntimeError("ACTIVATION_LOCK_BUSY") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _transaction_id(self, grant: ActivationGrant) -> str:
        return sha256_bytes(
            canonical_json({
                "schema": "AresActivationTransactionV1",
                "grant_id": grant.grant_id,
            })
        )

    def _write_journal(
        self,
        transaction_id: str,
        state: ActivationState,
        grant: ActivationGrant,
        detail: str = "",
    ) -> None:
        directory = self.layout.transactions_dir / transaction_id
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = directory / "journal.json"
        temporary = directory / ".journal.tmp"
        raw = canonical_json({
            "schema": "AresActivationJournalV1",
            "transaction_id": transaction_id,
            "grant_id": grant.grant_id,
            "sealed_candidate_id": grant.sealed_candidate_id,
            "state": state.value,
            "detail": detail,
        })
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _precheck(
        self, sealed_candidate_id: str
    ) -> tuple[ActivationGrant, InstalledRuntimePointer]:
        snapshot = self.store.verify(sealed_candidate_id)
        if snapshot.get("lifecycle_state") != "AWAITING_ACTIVATION":
            raise AresRuntimeError("CANDIDATE_LIFECYCLE_NOT_ACTIVATABLE")
        if snapshot.get("audit_state") != "AUDIT_PASSED":
            raise AresRuntimeError("AUDIT_NOT_PASSED")
        if snapshot.get("activation_authorization_state") != "AUTHORIZED":
            raise AresRuntimeError("ACTIVATION_NOT_AUTHORIZED")
        grant = self.store.read_activation_grant(sealed_candidate_id)
        if grant.sealed_candidate_id != sealed_candidate_id:
            raise AresRuntimeError("ACTIVATION_GRANT_MISMATCH")
        if Path(grant.target_release_root) != self.layout.releases_dir:
            raise AresRuntimeError("ACTIVATION_GRANT_TARGET_MISMATCH")
        try:
            pointer = self.layout.read_pointer()
        except AresRuntimeError as exc:
            if exc.code == "CURRENT_RELEASE_MISSING":
                raise AresRuntimeError("ROLLBACK_TARGET_MISSING") from exc
            raise
        return grant, pointer

    @staticmethod
    def _release_reference(grant: ActivationGrant) -> ReleaseReference:
        return ReleaseReference(
            kind="sealed_candidate",
            release_id=grant.sealed_candidate_id,
            release_manifest_sha256=grant.release_manifest_sha256,
            runtime_tree_sha256=grant.runtime_tree_sha256,
        )

    @staticmethod
    def _verify_identity(
        identity: RuntimeIdentity,
        reference: ReleaseReference,
        grant: ActivationGrant,
        generation: int,
    ) -> None:
        if (
            identity.sealed_candidate_id != reference.release_id
            or identity.release_manifest_sha256 != grant.release_manifest_sha256
            or identity.runtime_tree_sha256 != reference.runtime_tree_sha256
            or identity.generation != generation
        ):
            raise AresRuntimeError("RUNTIME_IDENTITY_MISMATCH")

    def activate(self, sealed_candidate_id: str) -> ActivationResult:
        """Activate exactly one previously authorized candidate or roll it back."""

        with self._activation_lock():
            grant: ActivationGrant | None = None
            transaction_id = ""
            previous: InstalledRuntimePointer | None = None
            committed: InstalledRuntimePointer | None = None
            try:
                provisional_grant = self.store.read_activation_grant(
                    sealed_candidate_id
                )
                transaction_id = self._transaction_id(provisional_grant)
                self._write_journal(
                    transaction_id, ActivationState.PRECHECKING, provisional_grant
                )
                grant, previous = self._precheck(sealed_candidate_id)
                self._write_journal(
                    transaction_id, ActivationState.PRECHECK_PASSED, grant
                )
                self._write_journal(
                    transaction_id, ActivationState.MATERIALIZING, grant
                )
                materialized = self.materializer(
                    self.store, self.layout, sealed_candidate_id, grant
                )
                if materialized.sealed_candidate_id != grant.sealed_candidate_id:
                    raise AresRuntimeError("MATERIALIZATION_IDENTITY_MISMATCH")
                self._write_journal(transaction_id, ActivationState.MATERIALIZED, grant)
                self._write_journal(transaction_id, ActivationState.QUIESCING, grant)
                self.supervisor.quiesce(transaction_id)
                self._write_journal(transaction_id, ActivationState.QUIESCED, grant)
                reference = self._release_reference(grant)
                committed = InstalledRuntimePointer(
                    generation=previous.generation + 1,
                    current=reference,
                    previous=previous.current,
                    committed_transaction_id=transaction_id,
                    state_root=previous.state_root,
                )
                self._write_journal(
                    transaction_id, ActivationState.READY_TO_COMMIT, grant
                )
                self.layout.write_pointer_atomic(committed)
                self._write_journal(
                    transaction_id, ActivationState.SWITCH_COMMITTED, grant
                )
                self._write_journal(transaction_id, ActivationState.STARTING, grant)
                self.supervisor.start(reference)
                self._write_journal(
                    transaction_id, ActivationState.HEALTH_CHECKING, grant
                )
                self._verify_identity(
                    self.supervisor.health(reference, committed.generation),
                    reference,
                    grant,
                    committed.generation,
                )
                self._write_journal(
                    transaction_id, ActivationState.LIVE_CERTIFYING, grant
                )
                self.store.record_activation_success(
                    sealed_candidate_id,
                    grant_id=grant.grant_id,
                    reason="ares-release-live-certification-passed",
                )
                self._write_journal(transaction_id, ActivationState.ACTIVATED, grant)
                return ActivationResult(
                    transaction_id, ActivationState.ACTIVATED, committed
                )
            except Exception as exc:
                if grant is None or committed is None or previous is None:
                    if grant is not None:
                        self._write_journal(
                            transaction_id,
                            ActivationState.ABORTED_PRE_COMMIT,
                            grant,
                            str(exc),
                        )
                    if isinstance(exc, AresRuntimeError):
                        raise
                    raise AresRuntimeError(
                        "ACTIVATION_PRECOMMIT_FAILED", str(exc)
                    ) from exc
                return self._rollback(
                    sealed_candidate_id, grant, transaction_id, previous, committed, exc
                )

    def _rollback(
        self,
        sealed_candidate_id: str,
        grant: ActivationGrant,
        transaction_id: str,
        previous: InstalledRuntimePointer,
        committed: InstalledRuntimePointer,
        cause: Exception,
    ) -> ActivationResult:
        self._write_journal(
            transaction_id, ActivationState.ROLLBACK_REQUIRED, grant, str(cause)
        )
        try:
            self.store.record_rollback_required(
                sealed_candidate_id,
                grant_id=grant.grant_id,
                reason="ares-release-postcommit-failure",
            )
            self._write_journal(transaction_id, ActivationState.ROLLING_BACK, grant)
            restored = InstalledRuntimePointer(
                generation=committed.generation + 1,
                current=previous.current,
                previous=committed.current,
                committed_transaction_id=transaction_id,
                state_root=previous.state_root,
            )
            self.layout.write_pointer_atomic(restored)
            self.supervisor.start(restored.current)
            previous_identity = self.supervisor.health(
                restored.current, restored.generation
            )
            if (
                previous_identity.sealed_candidate_id != restored.current.release_id
                or previous_identity.release_manifest_sha256
                != restored.current.release_manifest_sha256
                or previous_identity.runtime_tree_sha256
                != restored.current.runtime_tree_sha256
                or previous_identity.generation != restored.generation
            ):
                raise AresRuntimeError("ROLLBACK_VERIFICATION_FAILED")
            self.store.record_rollback_success(
                sealed_candidate_id,
                grant_id=grant.grant_id,
                reason="ares-release-previous-runtime-verified",
            )
            self._write_journal(transaction_id, ActivationState.ROLLED_BACK, grant)
            return ActivationResult(
                transaction_id, ActivationState.ROLLED_BACK, restored
            )
        except Exception as rollback_exc:
            self._write_journal(
                transaction_id, ActivationState.INCIDENT_HELD, grant, str(rollback_exc)
            )
            raise AresRuntimeError(
                "ROLLBACK_FAILED", str(rollback_exc)
            ) from rollback_exc
