"""``SessionRuntimeRepository`` on SQLite: the store-wide epoch, the receipted mutations, the deferred and chat picks,
the frozen-policy refresh, the viewer closes, the retirement reads and the ledger-gated ends.

Each verb converts its typed parameters to its body's arguments, runs one body (one transaction) inside
``refused`` and converts the result to the contract record. The bodies live in ``_mutation``, ``_runtime_ends``,
``_runtime_guards``, ``_runtime_picks`` and ``_runtime_retirement``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import api_generation_id, generation_request_id, local_session_id, room_session_id
from persistence.authority.runtime import (
    ApiGenerations,
    ChatPickParams,
    DeferredPick,
    DeleteFootprint,
    GenerationChain,
    GenerationPick,
    GenerationRow,
    HistoryWrite,
    LocalSnapshot,
    MutationCommand,
    MutationOutcome,
    PinnedIdentity,
    PreparedMutation,
    RoomGenerations,
    SessionRuntimeState,
    StoredDeferredPick,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.core.identity import SessionId

from .._backend import SQLiteStore, register_repository
from . import _mutation, _runtime_ends, _runtime_guards, _runtime_picks, _runtime_retirement
from ._runtime import refused

__all__ = ["SQLiteSessionRuntimeRepository"]


def _derive(chain: GenerationChain):
    """The id of each generation of ``chain`` (the contract package's pure derivations)."""
    if isinstance(chain, ApiGenerations):
        return lambda n: str(api_generation_id(chain.session_id, n))
    if isinstance(chain, RoomGenerations):
        return lambda n: str(room_session_id(chain.identity, n))
    return lambda n: str(local_session_id(chain.profile_id, chain.principal_id, generation_request_id(chain.base_request_id, n)))


def _mutation_call(command: MutationCommand) -> dict[str, Any]:
    """The ``operation`` and ``payload`` of ``command`` as the mutation body takes them; anything else is left for the
    payload check to refuse."""
    payload = getattr(command, "payload", None)
    return {"operation": getattr(command, "operation", None), "payload": payload() if callable(payload) else None}


def _legacy_prepared(prepared: PreparedMutation | None) -> dict[str, Any] | None:
    """The prepared commit as the mutation body takes it: the snapshot under ``snapshot`` beside the prepared values."""
    if prepared is None:
        return None
    snapshot = prepared.snapshot
    return {"snapshot": {"receipt": snapshot.receipt, "target": snapshot.target, "target_revision": snapshot.target_revision},
            **prepared.values}


def _chat_params(params: ChatPickParams) -> dict[str, Any]:
    return {"request_id": params.request_id, "operation": params.operation, "payload": dict(params.payload),
            "expected_revision": params.expected_revision, "expected_generation": params.expected_generation}


class SQLiteSessionRuntimeRepository:
    """The runtime authority of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # Writes

    def begin_epoch(self, *, instance_id: str) -> int:
        with refused():
            return _runtime_ends.begin_runtime_epoch(self._store.holder(), instance_id=instance_id)

    def prepare_mutation(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int,
                         command: MutationCommand, expected_generation: int | None = None,
                         live_guard: GateRefusal | None = None) -> LocalSnapshot | MutationOutcome:
        holder = self._store.holder()
        with refused(holder, session_id):
            result, _gone, replayed = _mutation.mutate_runtime_session(
                holder, epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id,
                expected_revision=expected_revision, expected_generation=expected_generation, live_guard=live_guard,
                prepare_only=True, **_mutation_call(command))
        if replayed:
            return MutationOutcome(result=result, replayed=True, gone=None)
        snapshot = result["snapshot"]
        return LocalSnapshot(receipt=snapshot["receipt"], target=snapshot["target"], target_revision=snapshot["target_revision"])

    def mutate(self, epoch: int, *, principal_id: str, session_id: str, request_id: str, expected_revision: int,
               command: MutationCommand, expected_generation: int | None = None, live_guard: GateRefusal | None = None,
               history: HistoryWrite | None = None, prepared: PreparedMutation | None = None,
               media_root: str | None = None, owner: OwnerRule | None) -> MutationOutcome:
        holder = self._store.holder()
        with refused(holder, session_id):
            result, gone, replayed = _mutation.mutate_runtime_session(
                holder, epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id,
                expected_revision=expected_revision, expected_generation=expected_generation, live_guard=live_guard,
                history=history, prepared=_legacy_prepared(prepared), media_root=media_root, owner=owner,
                **_mutation_call(command))
        return MutationOutcome(result=result, replayed=replayed, gone=gone)

    def defer_model_pick(self, epoch: int, *, key: str, digest: str, session_id: str, pick: DeferredPick) -> dict[str, Any]:
        with refused():
            return _runtime_picks.defer_pick(
                self._store.holder(), epoch=epoch, key=key, digest=digest, session_id=session_id,
                operation=pick.operation, principal_id=pick.principal_id, request_id=pick.request_id,
                payload=dict(pick.payload), shown=pick.shown)

    def forget_deferred_pick(self, session_id: str, *, stored: str) -> None:
        with refused():
            _runtime_picks.forget_deferred_pick(self._store.holder(), session_id, stored)

    def mark_deferred_pick_asked(self, session_id: str, *, stored: str, confirm: str) -> None:
        with refused():
            _runtime_picks.mark_deferred_pick_asked(self._store.holder(), session_id, stored, confirm)

    def replace_deferred_pick(self, session_id: str, *, stored: str, pick: Mapping[str, Any]) -> str:
        with refused():
            return _runtime_picks.replace_deferred_pick(self._store.holder(), session_id, stored, dict(pick))

    def commit_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams, result: Mapping[str, Any]) -> dict[str, Any]:
        with refused():
            return _runtime_picks.commit_chat_pick(self._store.holder(), epoch=epoch, key=key, digest=digest,
                                                   session_id=session_id, params=_chat_params(params), result=dict(result))

    def commit_refreshed_policy(self, epoch: int, *, session_id: str, old: Mapping[str, Any], new: Mapping[str, Any], generation: int) -> int:
        with refused():
            return _runtime_picks.commit_refreshed_policy(self._store.holder(), epoch=epoch, session_id=session_id,
                                                          old=dict(old), new=dict(new), generation=generation)

    def store_viewer_close(self, epoch: int, *, session_id: str, record: str) -> None:
        with refused():
            _runtime_ends._persist_close(self._store.holder(), epoch=epoch, session_id=session_id, record=record)

    def forget_viewer_close(self, epoch: int, *, session_id: str, record: str) -> bool:
        with refused():
            return _runtime_ends.forget_persisted_close(self._store.holder(), epoch=epoch, session_id=session_id, record=record)

    def end_unless_ledger_busy(self, session_id: str, *, session_key: str | None, end_reason: str) -> bool:
        with refused():
            return _runtime_ends._end_unless_ledger_busy(self._store.holder(), session_id, session_key, end_reason)

    def end_after_settlement(self, epoch: int, *, session_id: str, admission_id: str, generation: int, admission_epoch: int, reason: str) -> bool:
        with refused():
            return _runtime_ends.finalize_webhook(self._store.holder(), epoch=epoch, session_id=session_id,
                                                  admission_id=admission_id, generation=generation,
                                                  admission_epoch=admission_epoch, reason=reason)

    # Reads

    def current_epoch(self) -> int | None:
        with refused():
            return _runtime_ends.current_epoch(self._store.holder())

    def has_mutation_receipt(self, principal_id: str, session_id: str, request_id: str) -> bool:
        with refused():
            return _runtime_retirement.has_mutation_receipt(self._store.holder(), principal_id, session_id, request_id)

    def deferred_pick(self, session_id: str) -> StoredDeferredPick | None:
        with refused():
            stored = _runtime_picks.deferred_pick(self._store.holder(), session_id)
        return None if stored is None else StoredDeferredPick(pick=json.loads(stored), stored=stored)

    def prepare_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str, params: ChatPickParams) -> dict[str, Any] | SessionRuntimeState:
        with refused():
            prepared = _runtime_picks.prepare_chat_pick(self._store.holder(), epoch=epoch, key=key, digest=digest,
                                                        session_id=session_id, params=_chat_params(params))
        if "receipt" in prepared:
            return prepared["receipt"]
        row = prepared["row"]
        return SessionRuntimeState(revision=row["runtime_revision"], generation=row["runtime_generation"])

    def policy_picks(self, session_id: str) -> set[str]:
        with refused():
            return _runtime_picks.policy_picks(self._store.holder(), session_id)

    def retired_session(self, session_id: str) -> bool:
        with refused():
            return _runtime_retirement.retired_session(self._store.holder(), session_id)

    def first_generation(self, chain: GenerationChain, *, pinned: PinnedIdentity | None = None, start: int = 0) -> GenerationPick:
        with refused():
            generation, sid, row = _runtime_retirement.first_generation(
                self._store.holder(), _derive(chain),
                pinned=None if pinned is None else (pinned.principal_id, pinned.request_id), start=start)
        return GenerationPick(generation=generation, session_id=sid,
                              row=None if row is None else GenerationRow(title=row[0], source=row[1], archived=bool(row[2])))

    def expiring_retirements(self, session_ids: Sequence[str]) -> set[str]:
        with refused():
            return _runtime_retirement.expiring_retirements(self._store.holder(), session_ids)

    def viewer_close(self, session_id: str) -> str | None:
        with refused():
            return _runtime_ends.stored_close(self._store.holder(), session_id)

    def viewer_closes(self) -> list[tuple[str, str]]:
        with refused():
            return sorted(_runtime_ends.stored_closes(self._store.holder()))

    def pending_ledger_status(self, session_key: str) -> str | None:
        with refused():
            return _runtime_ends.pending_ledger_status(self._store.holder(), session_key)

    def delete_footprint(self, session_id: SessionId, owner: OwnerRule | None) -> DeleteFootprint:
        with refused():
            targets, children = _runtime_guards.delete_footprint(self._store.holder(), session_id, owner)
        return DeleteFootprint(targets=tuple(SessionId(sid) for sid in targets),
                               children=tuple(SessionId(sid) for sid in children))


register_repository("runtime", SQLiteSessionRuntimeRepository, {"authority_fences": True})
