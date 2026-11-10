"""The API binding verbs of the reference bindings repository.

API binding receipts and declared conversation keys live in the scope's ``api_bindings`` table (``receipts``: session id -> receipt dict; ``declared``: key -> session id), so ``bind_api`` stages them with the rows. Retirement markers are the authority state's ``retired`` table; a sweep marker carries ``expires``.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any, Optional

from persistence import AuthorityRefusal, ProofRejected, SessionId
from persistence.authority.admissions import GateRefusal
from persistence.authority.bindings import ApiBinding, NewApiBinding, api_generation_id
from persistence.conversation.sessions import SessionLineageKind
from persistence.core.canonical import canonical_json

from ..conversation.sessions import blank_session_row
from .bindings_lineage import decode, store_route

__all__ = ["ApiBindingVerbs"]


def _api_tables(namespace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = namespace.setdefault("api_bindings", {})
    tables.setdefault("receipts", {})
    tables.setdefault("declared", {})
    return tables


def _keep_passed_fences(retired: dict[str, Any], walked_from: str, session_id: str) -> bool:
    """Whether every ``api_generation_id(walked_from, n)`` before ``session_id`` is retired; each passed marker loses its expiry."""
    generation = 0
    while (sid := api_generation_id(walked_from, generation)) != session_id:
        marker = retired.get(sid)
        if marker is None:
            return False
        if isinstance(marker, dict):
            marker.pop("expires", None)
        generation += 1
    return True


def _reserve_key(declared: dict[str, Any], key: str, session_id: str) -> None:
    holder = declared.get(key)
    if holder is not None and holder != session_id:
        raise AuthorityRefusal("admission_conflict")
    declared.setdefault(key, session_id)


class ApiBindingVerbs:
    """``bind_api``, ``api_binding`` and ``declared_session``; the host class provides ``_backend`` and ``_state``."""

    _backend: Any

    def _state(self) -> Any:
        raise NotImplementedError

    def bind_api(self, *, epoch: int, binding: NewApiBinding, gate: GateRefusal | None = None, walked_from: SessionId | None = None) -> None:
        sid, storage_source = str(binding.session_id), binding.storage_source
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            if gate is not None:
                raise AuthorityRefusal(gate.reason, field=gate.field, detail=gate.detail)
            if sid in staged.authority["retired"]:
                raise AuthorityRefusal("not_found")
            if walked_from and not _keep_passed_fences(staged.authority["retired"], str(walked_from), sid):
                raise AuthorityRefusal("chain_moved")
            api = _api_tables(staged.scope)
            saved = api["receipts"].get(sid)
            if saved is not None:
                stored = decode(saved)
                if stored.get("storage_source", "api_server") != storage_source:
                    raise ProofRejected("session", sid)
                if binding.declared_key and stored.get("declared_key") != binding.declared_key:
                    raise AuthorityRefusal("admission_conflict")
                return
            if binding.declared_key:
                _reserve_key(api["declared"], binding.declared_key, sid)
            self._bind_row(staged, api, binding)

    def _bind_row(self, staged: Any, api: dict[str, dict[str, Any]], binding: NewApiBinding) -> None:
        """The row, route and receipt half of ``bind_api``."""
        sid, route, storage_source = str(binding.session_id), binding.route, binding.storage_source
        receipt: dict[str, Any] = copy.deepcopy(dict(binding.receipt))
        sessions = staged.scope.setdefault("sessions", {})
        row = sessions.get(sid)
        if row is not None and row["source"] != storage_source:
            raise ProofRejected("session", sid)
        if storage_source == "bot_room":
            if row is not None and row.get("title") != binding.title:
                raise AuthorityRefusal("admission_conflict")
            if any(other_id != sid and other.get("title") == binding.title for other_id, other in sessions.items()):
                raise AuthorityRefusal("admission_conflict")
        legacy_key = binding.declared_key or binding.caller_key
        if row is not None and row.get("session_key") not in (None, "", route):
            if not legacy_key or row["session_key"] != legacy_key:
                raise AuthorityRefusal("admission_conflict")
            _reserve_key(api["declared"], legacy_key, sid)
            receipt["declared_key"] = legacy_key
        existing = staged.scope.setdefault("routing", {}).setdefault("", {}).get(route)
        held = json.loads(existing.text(route))["session_id"] if existing is not None else sid
        if held != sid and held in api["receipts"]:
            raise AuthorityRefusal("admission_conflict")
        if row is None:
            row = sessions[sid] = blank_session_row(sid, lineage_kind=SessionLineageKind.NONE, model_config={})
            row.update(source=storage_source, title=binding.title, hidden=storage_source == "bot_room",
                       started_at=datetime.fromtimestamp(binding.started_at, UTC))
        row.update(session_key=route, chat_id=sid, user_id=binding.origin["user_id"], chat_type="dm", origin=copy.deepcopy(dict(binding.origin)))
        store_route(staged.scope, route, dict(binding.entry))
        api["receipts"][sid] = json.loads(canonical_json(receipt))

    def api_binding(self, session_id: SessionId) -> Optional[ApiBinding]:
        saved = _api_tables(self._backend.scope_store(self._backend.scope))["receipts"].get(str(session_id))
        if saved is None:
            return None
        receipt = copy.deepcopy(decode(saved))
        identity = receipt.get("room_identity")
        return ApiBinding(
            profile_id=receipt["profile_id"], session_id=SessionId(receipt["session_id"]), route=receipt["route"], entry=receipt["entry"],
            declared_key=receipt.get("declared_key"), storage_source=receipt.get("storage_source", "api_server"),
            room_identity=tuple(identity) if identity is not None else None, room_generation=int(receipt.get("room_generation", 0)))

    def declared_session(self, declared_key: str) -> Optional[SessionId]:
        holder = _api_tables(self._backend.scope_store(self._backend.scope))["declared"].get(declared_key)
        return None if holder is None else SessionId(holder)
