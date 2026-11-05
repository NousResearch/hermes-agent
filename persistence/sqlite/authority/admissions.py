"""``AdmissionRepository`` on SQLite: the bound profile's admission ledger in ``session_admissions`` and its
``state_meta`` keys. Each verb converts its typed parameters to the transaction function's arguments, runs it under
:func:`refused` and converts the row to the contract record.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

from persistence.authority.admissions import (
    ANY_HEAD,
    AdmissionRecord,
    AdmissionResult,
    AnyHead,
    ApiTargetClaim,
    CapturedResult,
    ClaimedAdmission,
    GateRefusal,
    QueuedFold,
    QueuedReplacement,
    TranscriptRewrite,
    TurnBoundary,
    UnheldMedia,
)
from persistence.core.errors import PermanentStoreError

from .._backend import SQLiteStore, register_repository
from . import _admission_media as media
from . import _admission_owner as owner
from . import _admission_store as store
from ._runtime import refused, retry_terminal_admission

if TYPE_CHECKING:
    from persistence.gateway.hosted_rooms import HostedMemberGuard

__all__ = ["SQLiteAdmissionRepository"]

_RECORD_FIELDS = tuple(f.name for f in fields(AdmissionRecord))


def _record(row: Mapping[str, Any] | None) -> AdmissionRecord | None:
    if row is None:
        return None
    return AdmissionRecord(**{name: row[name] for name in _RECORD_FIELDS})


def _claimed(row: Mapping[str, Any]) -> ClaimedAdmission:
    return ClaimedAdmission(**{name: row[name] for name in _RECORD_FIELDS}, transcript_mark=row["transcript_mark"],
                            queued_owner_epoch=row["queued_owner_epoch"])


class SQLiteAdmissionRepository:
    """The admission ledger of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # Writes

    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any],
              intent: Literal["queue", "steer", "redirect"] = "queue", gate: GateRefusal | None = None,
              replaces: QueuedReplacement | None = None, supersedes: str | None = None,
              api_target: ApiTargetClaim | None = None, rewrite: TranscriptRewrite | None = None,
              hosted: HostedMemberGuard | None = None, fold: QueuedFold | None = None) -> tuple[AdmissionRecord, bool]:
        with refused():
            row, created = store.admit_session_input(
                self._store.holder(), epoch=epoch, principal_id=principal_id, session_id=session_id,
                request_id=request_id, payload=payload, intent=intent, gate=gate, replaces=replaces,
                supersedes=supersedes, api_target=api_target, rewrite=rewrite, hosted=hosted, fold=fold)
        return _record(row), created

    def claim(self, *, epoch: int, session_id: str, gate: GateRefusal | None = None,
              head: str | None | AnyHead = ANY_HEAD) -> ClaimedAdmission | Literal[False] | None:
        with refused():
            row = store.claim_session_input(self._store.holder(), epoch=epoch, session_id=session_id, gate=gate,
                                            head=store._ANY_HEAD if head is ANY_HEAD else head)
        return row if row is None or row is False else _claimed(row)

    def return_claim(self, *, epoch: int, admission_id: str, generation: int,
                     queued_owner_epoch: int | None) -> AdmissionRecord:
        with refused():
            return _record(store.return_session_claim(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                                      generation=generation, queued_owner_epoch=queued_owner_epoch))

    def settle(self, *, epoch: int, admission_id: str, generation: int,
               outcome: Literal["completed", "interrupted", "rejected", "failed"],
               result: Mapping[str, Any] | None = None, transfers: Mapping[str, bool] | None = None,
               boundary: TurnBoundary | None = None) -> AdmissionRecord:
        with refused():
            return _record(store.settle_session_input(
                self._store.holder(), epoch=epoch, admission_id=admission_id, generation=generation, outcome=outcome,
                result=result, _transfers=transfers, boundary=boundary))

    def cancel(self, *, epoch: int, admission_id: str) -> AdmissionRecord:
        with refused():
            return _record(store.cancel_session_input(self._store.holder(), epoch=epoch, admission_id=admission_id))

    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[AdmissionRecord, bool]:
        with refused():
            row, created = store.steer_session_input(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                                     generation=generation)
        return _record(row), created

    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        with refused():
            return store.revert_steer_input(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                            generation=generation)

    def mark_unknown(self, *, epoch: int, admission_id: str, generation: int,
                     transfers: Mapping[str, bool] | None = None) -> Literal["unknown", "terminal"]:
        with refused():
            return owner._mark_unknown(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                       generation=generation, transfers=transfers)

    def rehome_queued(self, *, epoch: int, old_session_id: str, new_session_id: str, principal_prefix: str) -> list[str]:
        with refused():
            return store.rehome_queued_inputs(self._store.holder(), epoch=epoch, old_session_id=old_session_id,
                                              new_session_id=new_session_id, principal_prefix=principal_prefix)

    def retarget_local_automation(self, *, epoch: int, session_id: str, principal_id: str, old_targets: Iterable[str],
                                  new_target: str) -> list[str]:
        with refused():
            return store.retarget_local_automation(self._store.holder(), epoch=epoch, session_id=session_id,
                                                   principal_id=principal_id, old_targets=list(old_targets),
                                                   new_target=new_target)

    def recover(self, *, epoch: int) -> int:
        with refused():
            return store.recover_session_inputs(self._store.holder(), epoch=epoch)

    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int, notice: str | None = None,
                        captured: CapturedResult | None = None) -> AdmissionRecord:
        with refused():
            return _record(store.resolve_unknown_session_input(
                self._store.holder(), epoch=epoch, admission_id=admission_id, generation=generation, notice=notice,
                _captured=None if captured is None else (captured.outcome, dict(captured.result))))

    def compact_api_payloads(self, *, admission_ids: Sequence[str]) -> int:
        with refused():
            return owner.compact_api_payloads(self._store.holder(), list(admission_ids))

    def unheld_media(self, references: Sequence[Mapping[str, Any]], *, after_id: int, retain_history: bool,
                     media_root: str) -> UnheldMedia:
        try:
            with refused():
                candidates, held = media.unheld_media(self._store.holder(), list(references), after_id=after_id,
                                                      retain_history=retain_history, media_root=media_root)
        except (KeyError, TypeError) as exc:  # a stored media reference without a path
            raise PermanentStoreError(f"admission media reference is corrupt: {exc!r}") from exc
        return UnheldMedia(candidates=tuple(candidates), held=frozenset(held))

    def record_retired_media(self, references: Sequence[Mapping[str, Any]]) -> None:
        with refused():
            media.record_retired_media(self._store.holder(), references)

    def drop_retired_candidates(self, keys: Sequence[str]) -> None:
        with refused():
            media.drop_retired_candidates(self._store.holder(), keys)

    # Reads

    def find(self, *, principal_id: str, session_id: str, request_id: str) -> AdmissionRecord | None:
        with refused():
            return _record(store.find_session_admission(self._store.holder(), principal_id=principal_id,
                                                        session_id=session_id, request_id=request_id))

    def match(self, *, request_id: str | None = None, admission_id: str | None = None, principal_id: str | None = None,
              session_id: str | None = None, include_retired: bool = False) -> list[AdmissionRecord]:
        with refused():
            rows = owner.match_admissions(self._store.holder(), include_retired=include_retired, request_id=request_id,
                                          admission_id=admission_id, principal_id=principal_id, session_id=session_id)
        return [_record(row) for row in rows]

    def list(self, *, session_id: str, pending_only: bool = True) -> list[AdmissionRecord]:
        with refused():
            rows = store.list_session_admissions(self._store.holder(), session_id=session_id, pending_only=pending_only)
        return [_record(row) for row in rows]

    def get(self, *, admission_id: str) -> AdmissionRecord | None:
        with refused():
            return _record(store.get_session_admission(self._store.holder(), admission_id=admission_id))

    def settled_result(self, *, admission_id: str, epoch: int | None = None,
                       generation: int | None = None) -> AdmissionResult | None:
        if (epoch is None) != (generation is None):
            raise ValueError("settled_result takes epoch and generation together or not at all")
        with refused():
            if epoch is None:
                found = owner.admission_result(self._store.holder(), admission_id)
            else:
                found = owner.finish_result_precheck(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                                     generation=generation)
        return None if found is None else AdmissionResult(record=_record(found[0]), result=found[1])

    def retired_replay(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
                       payload: Mapping[str, Any],
                       intent: Literal["queue", "steer", "redirect"] = "queue") -> AdmissionRecord | None:
        with refused():
            return _record(retry_terminal_admission(self._store.holder(), epoch=epoch, principal_id=principal_id,
                                                    session_id=session_id, request_id=request_id, payload=payload,
                                                    intent=intent))

    def pending_targets(self, *, statuses: frozenset[str], principal_id: str | None = None) -> frozenset[str]:
        with refused():
            return owner.pending_targets(self._store.holder(), statuses=statuses, principal_id=principal_id)

    def list_unknown(self) -> list[AdmissionRecord]:
        with refused():
            return [_record(row) for row in owner.list_unknown(self._store.holder())]

    def has_pending(self, *, session_ids: Sequence[str] = (), statuses: frozenset[str] | None = None) -> bool:
        with refused():
            return owner.has_pending(self._store.holder(), session_ids=tuple(session_ids), statuses=statuses)

    def ledger_busy(self, session_id: str, target: str) -> bool:
        with refused():
            return owner._busy(self._store.holder(), session_id, target)

    def webhook_ingress_payloads(self, *, chat_id: str) -> list[Mapping[str, Any]]:
        with refused():
            return owner.webhook_ingress_payloads(self._store.holder(), chat_id)

    def unfinalized_webhook_admissions(self) -> list[str]:
        with refused():
            return owner.unfinalized_webhook_admissions(self._store.holder())

    def terminal_ids(self, admission_ids: Sequence[str]) -> frozenset[str]:
        with refused():
            return owner.terminal_ids(self._store.holder(), admission_ids)

    def api_compaction_candidates(self) -> list[str]:
        with refused():
            return owner.api_compaction_candidates(self._store.holder())

    def history_mentions(self, references: Sequence[Mapping[str, Any]]) -> tuple[int, frozenset[str]]:
        with refused():
            watermark, mentioned = media.history_mentions(self._store.holder(), list(references))
        return watermark, frozenset(mentioned)

    def settled_media(self) -> list[Sequence[Mapping[str, Any]]]:
        with refused():
            return media.settled_media(self._store.holder())

    def retired_candidates(self, kind: Literal["media", "spillover"]) -> list[tuple[str, Any]]:
        with refused():
            return media.retired_candidates(self._store.holder(), kind)


register_repository("admissions", SQLiteAdmissionRepository, {"durable_admissions": True})
