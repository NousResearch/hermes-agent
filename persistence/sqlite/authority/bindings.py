"""``SessionBindingRepository`` on SQLite: local creation receipts, API bindings and the local lineage reads.

Each verb converts its typed parameters to its body's arguments, runs one body (one transaction) inside
``refused`` and converts the result to the contract record. The bodies live in ``_local``, ``_local_migration``,
``_binding_api`` and ``_binding_history``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from persistence.authority.runtime import HistoryActor
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import (
    ApiBinding,
    BindingMarks,
    LegacyAdoptionScan,
    LocalLineage,
    NewApiBinding,
    NewSessionBinding,
    ReservedBinding,
    SessionBinding,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.core.identity import JsonValue, SessionId
from session_rules.legacy_local import login_label

from .._backend import SQLiteStore, register_repository
from . import _binding_api, _binding_history, _local
from ._runtime import refused

__all__ = ["SQLiteSessionBindingRepository"]


def _new_receipt(binding: NewSessionBinding) -> dict:
    """The receipt dict the reservation stores (``lineage`` and ``legacy_session_id`` only when set)."""
    receipt = {"profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
               "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry),
               "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage is not None:
        receipt["lineage"] = list(binding.lineage)
    return receipt


class SQLiteSessionBindingRepository:
    """Local creation receipts, API bindings and the lineage reads of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # Writes

    def reserve(self, *, epoch: int, binding: NewSessionBinding, client_request: Mapping[str, JsonValue] | None = None,
                owner: OwnerRule | None) -> ReservedBinding:
        with refused():
            receipt, created = _local.commit_local_session(
                self._store.holder(), epoch=epoch, receipt=_new_receipt(binding),
                client_request=None if client_request is None else dict(client_request),
                created_source=binding.created_source, owner=owner)
            return ReservedBinding(binding=_local.binding_record(receipt), created=created)

    def adopt_untracked_compression(self, *, epoch: int, session_id: SessionId, expected: SessionBinding) -> None:
        with refused():
            _local.adopt_untracked_compression(self._store.holder(), epoch=epoch, session_id=session_id, expected=expected)

    def reset_target(self, *, epoch: int, parent_session_id: SessionId, entry: Mapping[str, JsonValue]) -> None:
        with refused():
            _local.reset_local_target(self._store.holder(), epoch=epoch, parent_session_id=parent_session_id, entry=dict(entry))

    def end_idle(self, *, epoch: int, session_id: SessionId, target_id: SessionId, reason: str) -> int:
        with refused():
            return _local.end_idle_local_session(self._store.holder(), epoch=epoch, session_id=session_id,
                                                 target_id=target_id, reason=reason)

    def bind_api(self, *, epoch: int, binding: NewApiBinding, gate: GateRefusal | None = None, walked_from: SessionId | None = None) -> None:
        with refused():
            _binding_api.bind_api(
                self._store.holder(), epoch=epoch, session_id=binding.session_id, route=binding.route,
                storage_source=binding.storage_source, title=binding.title, declared_key=binding.declared_key,
                caller_key=binding.caller_key, source=dict(binding.origin), entry=dict(binding.entry),
                receipt=dict(binding.receipt), now=binding.started_at, gate=gate, walked_from=walked_from)

    def release_handed_off_receipts(self, *, epoch: int, lanes: frozenset[str]) -> dict[str, tuple[str, str]]:
        with refused():
            return _local.release_handed_off_receipts(self._store.holder(), epoch=epoch, lanes=lanes)

    # Reads

    def target_has_history(self, session_id: str) -> bool:
        with refused():
            return _local.target_has_history(self._store.holder(), session_id)

    def receipt(self, session_id: SessionId) -> SessionBinding:
        with refused():
            return _local.binding_record(_local.local_receipt(self._store.holder(), session_id))

    def local_session_ids(self) -> list[SessionId]:
        with refused():
            return [SessionId(sid) for sid in _local.local_session_ids(self._store.holder())]

    def list_lineages(self) -> list[LocalLineage]:
        with refused():
            found = _local.local_lineages(self._store.holder())
        return [LocalLineage(owner=SessionId(owner), lineage=tuple(SessionId(sid) for sid in lineage)) for owner, lineage in found]

    def binding_marks(self, session_id: SessionId, *, lineage_of: SessionId | None = None) -> BindingMarks:
        with refused():
            policy, legacy, owner = _local.binding_marks(self._store.holder(), session_id, lineage_of)
        return BindingMarks(policy=policy, legacy=legacy, lineage_owner=None if owner is None else SessionId(owner))

    def lineage_owner(self, session_id: SessionId) -> SessionId:
        with refused():
            return SessionId(_local.lineage_owner(self._store.holder(), session_id))

    def conversation_ids(self, session_id: SessionId) -> list[SessionId]:
        with refused():
            return [SessionId(sid) for sid in _local.conversation_ids(self._store.holder(), session_id)]

    def validate_lineage(self, receipt: SessionBinding) -> SessionId:
        with refused():
            return SessionId(_local.validate_lineage(self._store.holder(), _local.receipt_of(receipt)))

    def compression_adoptable(self, receipt: SessionBinding) -> bool:
        with refused():
            return _local.compression_adoptable(self._store.holder(), _local.receipt_of(receipt))

    def restorable(self, receipt: SessionBinding) -> SessionBinding:
        with refused():
            return _local.binding_record(_local.restorable(self._store.holder(), _local.receipt_of(receipt)))

    def authorize_history(self, *, session_id: SessionId, profile_id: str, subject: str, may_create: bool, native_transport: bool | None = None,
                          owner: OwnerRule | None) -> bool:
        actor = HistoryActor(profile_id, subject, login_label(subject), may_create)
        with refused():
            return _binding_history.authorize_history_read(self._store.holder(), actor, session_id, native_transport, owner)

    def api_binding(self, session_id: SessionId) -> Optional[ApiBinding]:
        with refused():
            receipt = _binding_api.api_binding_receipt(self._store.holder(), session_id)
        if receipt is None:
            return None
        identity = receipt.get("room_identity")
        return ApiBinding(
            profile_id=receipt["profile_id"], session_id=SessionId(receipt["session_id"]), route=receipt["route"],
            entry=receipt["entry"], declared_key=receipt.get("declared_key"),
            storage_source=receipt.get("storage_source", "api_server"),
            room_identity=tuple(identity) if identity is not None else None,
            room_generation=int(receipt.get("room_generation", 0)))

    def declared_session(self, declared_key: str) -> Optional[SessionId]:
        with refused():
            holder = _binding_api.declared_session(self._store.holder(), declared_key)
        return None if holder is None else SessionId(holder)

    def legacy_adoption_scan(self, owner: OwnerRule) -> LegacyAdoptionScan:
        with refused():
            ids, marked = _local.legacy_adoption_scan(self._store.holder(), owner)
        return LegacyAdoptionScan(session_ids=tuple(SessionId(sid) for sid in ids), marked=frozenset(marked))

    def history_binding(self, session_id: SessionId) -> Optional[str]:
        holder = self._store.holder()
        with refused(), holder._read_ctx() as conn:
            return _binding_history.history_binding(conn, session_id)

    def handoff_head(self, session_id: SessionId) -> tuple[Optional[SessionId], SessionId]:
        with refused():
            owner, head = _local.handoff_head(self._store.holder(), session_id)
        return (None if owner is None else SessionId(owner)), SessionId(head)

    def boot_session_ids(self) -> list[SessionId]:
        with refused():
            return [SessionId(sid) for sid in _local.local_receipt_ids(self._store.holder())]


register_repository("bindings", SQLiteSessionBindingRepository, {"receipted_writes": True})
