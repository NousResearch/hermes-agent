from __future__ import annotations

import os
import threading
import time
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
        """Materialize required schema bytes, then close the bootstrap writer.

        Bootstrap is not an integrity proof. The file can be replaced after
        schema creation, so the temporary writer is always closed before the
        ordinary verify -> identity anchor -> exact-generation open sequence.
        """
        try:
            original_init(instance, db_path=path, read_only=False)
            conn = getattr(instance, "_conn", None)
            if conn is None:
                raise RuntimeError("SessionDB bootstrap returned without a connection")
            # Prove only that schema materialization reached the required tables.
            # The canonical full integrity probe runs after this writer closes.
            conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
            conn.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
        except StateDBAdmissionError:
            self._close_quietly(instance)
            raise
        except Exception as exc:
            self._close_quietly(instance)
            report = StateDBIntegrityReport(
                path=path,
                verdict=problem_verdict(str(exc)),
                checked="bootstrap_schema",
                problems=(str(exc),),
                identity=(stat_identity(path) if path.exists() else None),
            )
            raise StateDBIntegrityError(
                format_refusal(report),
                path=path,
                report=report,
            ) from exc

        try:
            ORIGINAL_SESSION_DB.close(instance)
            # Do not leave a closed bootstrap handle looking like the admitted
            # connection while the exact generation is being verified/opened.
            instance._conn = None
        except Exception as exc:
            report = StateDBIntegrityReport(
                path=path,
                verdict=problem_verdict(str(exc)),
                checked="bootstrap_close",
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
        try:
            instance._conn = None
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

    @staticmethod
    def _refuse_unhealthy(report: StateDBIntegrityReport, path: Path) -> None:
        if report.may_open_writer and report.identity is not None:
            return
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
                needs_schema_materialization = report.verdict in {
                    IntegrityVerdict.ABSENT,
                    IntegrityVerdict.EMPTY,
                    IntegrityVerdict.SCHEMA_INCOMPLETE,
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
                    needs_schema_materialization = True

                if needs_schema_materialization:
                    # Materialize schema only. Never inherit health from the
                    # temporary bootstrap handle: close it, then verify the
                    # exact path generation that will actually be admitted.
                    self._bootstrap(instance, path, original_init)
                    report = repair_and_reverify(
                        path,
                        verify_state_db_integrity(path),
                    )
                else:
                    report = repair_and_reverify(path, report)

                self._refuse_unhealthy(report, path)
                assert report.identity is not None
                anchor_fd, identity = self._open_anchor(path, report.identity)
                try:
                    # All writers, including schema-materializing writers, are
                    # admitted only after the same generation has a canonical
                    # health proof and an anchor. There is no bootstrap
                    # exception to verify -> anchor -> exact-open.
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
