from __future__ import annotations

import os
import time
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import hermes_state

from ._context import EXPECTED_GENERATION, ORIGINAL_SESSION_DB, PENDING_FAILURE
from ._lock import CrossProcessAdmissionLock
from ._model import (
    IntegrityVerdict,
    StateDBAdmissionBusyError,
    StateDBAdmissionError,
    StateDBAdmissionProof,
    StateDBFileIdentity,
    StateDBGenerationConflictError,
    StateDBIntegrityError,
    StateDBIntegrityReport,
    anchor_identity,
    canonical_state_db_path,
    format_refusal,
    problem_verdict,
    same_identity,
    stat_identity,
)
from ._verify import repair_and_reverify, verify_state_db_integrity


@dataclass
class _LiveGeneration:
    proof: StateDBAdmissionProof
    anchor_fd: int
    holders: int = 0


class GatewayStateDBAuthority:
    """Own proof-bearing first admission for every gateway writer handle."""

    def __init__(self) -> None:
        self._map_lock = threading.Lock()
        self._path_locks: dict[Path, threading.RLock] = {}
        self._live: dict[Path, _LiveGeneration] = {}

    def _path_lock(self, path: Path) -> threading.RLock:
        with self._map_lock:
            return self._path_locks.setdefault(path, threading.RLock())

    def _bootstrap(
        self,
        instance: Any,
        path: Path,
        original_init: Callable[..., None],
    ) -> None:
        """Initialize first-run bytes and verify before returning the object."""
        original_init(instance, db_path=path, read_only=False)
        try:
            rows = instance._conn.execute("PRAGMA integrity_check").fetchall()
            problems = [
                str(row[0])
                for row in rows
                if row and str(row[0]).strip().lower() != "ok"
            ]
            if problems:
                report = StateDBIntegrityReport(
                    path=path,
                    verdict=IntegrityVerdict.CORRUPT,
                    checked="bootstrap_full",
                    problems=tuple(problems[:3]),
                    identity=stat_identity(path),
                )
                raise StateDBIntegrityError(
                    format_refusal(report),
                    path=path,
                    report=report,
                )
            instance._conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
            instance._conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
        except StateDBAdmissionError:
            self._close_quietly(instance)
            raise
        except Exception as exc:
            self._close_quietly(instance)
            report = StateDBIntegrityReport(
                path=path,
                verdict=problem_verdict(str(exc)),
                checked="bootstrap_full",
                problems=(str(exc),),
                identity=(stat_identity(path) if path.exists() else None),
            )
            raise StateDBIntegrityError(
                format_refusal(report),
                path=path,
                report=report,
            ) from exc

    @staticmethod
    def _close_quietly(instance: Any) -> None:
        try:
            ORIGINAL_SESSION_DB.close(instance)
        except Exception:
            pass

    @staticmethod
    def _open_anchor(
        path: Path,
        expected: StateDBFileIdentity,
    ) -> tuple[int, StateDBFileIdentity]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        actual = anchor_identity(fd)
        if same_identity(expected, actual):
            return fd, actual
        os.close(fd)
        raise StateDBGenerationConflictError(
            f"state.db at {path} changed before its admission anchor opened",
            path=path,
        )

    @staticmethod
    def _assert_anchor_is_current(live: _LiveGeneration) -> None:
        try:
            current = stat_identity(live.proof.path)
            anchored = anchor_identity(live.anchor_fd)
        except OSError as exc:
            raise StateDBGenerationConflictError(
                f"cannot prove the admitted state.db is current: {exc}",
                path=live.proof.path,
                report=live.proof.report,
            ) from exc
        if not same_identity(current, anchored):
            raise StateDBGenerationConflictError(
                f"state.db at {live.proof.path} was replaced while generation "
                f"{live.proof.proof_id} still has live writers; refusing split-brain",
                path=live.proof.path,
                report=live.proof.report,
            )

    @staticmethod
    def _open_exact_generation(
        instance: Any,
        path: Path,
        identity: StateDBFileIdentity,
        original_init: Callable[..., None],
    ) -> None:
        token = EXPECTED_GENERATION.set((path, identity))
        try:
            original_init(instance, db_path=path, read_only=False)
        finally:
            EXPECTED_GENERATION.reset(token)

    def initialize_writable(
        self,
        instance: Any,
        *,
        db_path: Path | str | None,
        original_init: Callable[..., None],
    ) -> None:
        path = canonical_state_db_path(db_path)
        try:
            with self._path_lock(path), CrossProcessAdmissionLock(path):
                live = self._live.get(path)
                if live is not None:
                    self._assert_anchor_is_current(live)
                    self._open_exact_generation(
                        instance,
                        path,
                        live.proof.identity,
                        original_init,
                    )
                    live.holders += 1
                    instance._gateway_state_db_admission = live.proof
                    return

                report = verify_state_db_integrity(path)
                first_run = report.verdict in {
                    IntegrityVerdict.ABSENT,
                    IntegrityVerdict.EMPTY,
                }
                is_zeroed = getattr(
                    hermes_state,
                    "is_zeroed_state_db",
                    lambda _path: False,
                )
                if (
                    report.verdict is IntegrityVerdict.CORRUPT
                    and path.exists()
                    and is_zeroed(path)
                ):
                    first_run = True

                if first_run:
                    self._bootstrap(instance, path, original_init)
                    report = StateDBIntegrityReport(
                        path=path,
                        verdict=IntegrityVerdict.VERIFIED,
                        checked="bootstrap_full",
                        identity=stat_identity(path),
                        may_open_writer=True,
                    )
                else:
                    report = repair_and_reverify(path, report)
                    if not report.may_open_writer or report.identity is None:
                        error_type = (
                            StateDBAdmissionBusyError
                            if report.verdict is IntegrityVerdict.BUSY
                            else StateDBIntegrityError
                        )
                        raise error_type(
                            format_refusal(report),
                            path=path,
                            report=report,
                        )

                assert report.identity is not None
                anchor_fd, identity = self._open_anchor(path, report.identity)
                try:
                    if not first_run:
                        self._open_exact_generation(
                            instance,
                            path,
                            identity,
                            original_init,
                        )
                    if not same_identity(identity, stat_identity(path)):
                        raise StateDBGenerationConflictError(
                            f"state.db at {path} changed while its writer opened",
                            path=path,
                            report=report,
                        )
                    proof = StateDBAdmissionProof(
                        proof_id=uuid.uuid4().hex,
                        path=path,
                        identity=identity,
                        report=report,
                        verified_at=time.time(),
                    )
                    self._live[path] = _LiveGeneration(
                        proof=proof,
                        anchor_fd=anchor_fd,
                        holders=1,
                    )
                    instance._gateway_state_db_admission = proof
                except BaseException:
                    self._close_quietly(instance)
                    os.close(anchor_fd)
                    raise
        except StateDBAdmissionError as exc:
            PENDING_FAILURE.set(exc)
            raise

    def release(self, instance: Any) -> None:
        proof = getattr(instance, "_gateway_state_db_admission", None)
        if not isinstance(proof, StateDBAdmissionProof):
            return
        instance._gateway_state_db_admission = None
        with self._path_lock(proof.path):
            live = self._live.get(proof.path)
            if live is None or live.proof.proof_id != proof.proof_id:
                return
            live.holders -= 1
            if live.holders > 0:
                return
            self._live.pop(proof.path, None)
            try:
                os.close(live.anchor_fd)
            except OSError:
                pass

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._map_lock:
            paths = list(self._live)
        result: dict[str, dict[str, Any]] = {}
        for path in paths:
            with self._path_lock(path):
                live = self._live.get(path)
                if live is None:
                    continue
                result[str(path)] = {
                    "proof_id": live.proof.proof_id,
                    "holders": live.holders,
                    "identity": {
                        "device": live.proof.identity.device,
                        "inode": live.proof.identity.inode,
                    },
                    "report": live.proof.report.as_dict(),
                }
        return result


AUTHORITY = GatewayStateDBAuthority()
