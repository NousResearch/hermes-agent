"""Caller-history delivery state of the in-memory message reference: claim, reserve, commit and release.

The state lives in a row's ``display_metadata`` as the store keeps it: ``caller_history_consumed`` is final and
``caller_history_reservation`` = ``{token, owner, until}`` is a lease that expires on its own.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import replace
from typing import Any, Callable, Optional, Sequence

from persistence import RowVisibility
from persistence.core.identity import JsonValue, SessionId

from .messages_base import MessageBase
from .transcript_store import _StoredRow, resume_lineage, rows_of

_DELIVERY_KINDS = ("async_delegation_complete", "hidden")
_LEASE = "caller_history_reservation"
_CONSUMED = "caller_history_consumed"


def _meta(row: _StoredRow) -> dict[str, Any]:
    return dict(row.message.display_metadata or {})


def _pending(row: _StoredRow, now: float) -> bool:
    meta = _meta(row)
    lease = meta.get(_LEASE) if isinstance(meta.get(_LEASE), dict) else {}
    return (row.message.role == "user" and row.message.display_kind in _DELIVERY_KINDS
            and row.visibility is RowVisibility.ACTIVE and meta.get("delegation_id") is not None
            and meta.get(_CONSUMED) is None and float(lease.get("until") or 0) <= now)


def _delivery(session_id: str, row: _StoredRow, token: Optional[str] = None) -> dict[str, JsonValue]:
    content = copy.deepcopy(row.message.content)
    out: dict[str, JsonValue] = {"id": row.id, "delivery_id": row.id, "session_id": session_id, "content": content,
                                 "result": content, "display_metadata": _meta(row)}
    if token is not None:
        out["reservation_token"] = token
    return out


def _set_meta(row: _StoredRow, meta: dict[str, Any]) -> None:
    row.message = replace(row.message, display_metadata=meta or None)


def _consume(meta: dict[str, Any]) -> dict[str, Any]:
    meta.pop(_LEASE, None)
    meta[_CONSUMED] = 1
    return meta


def _drop_lease(meta: dict[str, Any]) -> dict[str, Any]:
    meta.pop(_LEASE, None)
    return meta


class MessageCallerHistoryMixin(MessageBase):
    """The caller-history delivery verbs of ``MessageRepository``."""

    def _lineage_rows(self, session_id: str) -> list[tuple[str, _StoredRow]]:
        scope = self._backend.scope
        return rows_of(self._backend, scope, resume_lineage(self._backend, scope, str(session_id)))

    def _all_rows(self) -> list[_StoredRow]:
        transcripts = self._ns(self._backend.scope)["transcripts"]
        return [row for transcript in transcripts.values() for row in transcript.rows]

    def claim_caller_history_deliveries(self, session_id: SessionId) -> list[dict[str, JsonValue]]:
        now, out = time.time(), []
        for sid, row in self._lineage_rows(session_id):
            if _pending(row, now):
                out.append(_delivery(sid, row))
                _set_meta(row, _consume(_meta(row)))
        return out

    def reserve_caller_history_deliveries(self, session_id: SessionId, owner: str, ttl_seconds: float,
                                          limit: Optional[int] = None) -> list[dict[str, JsonValue]]:
        now, token, out = time.time(), uuid.uuid4().hex, []
        lease = {"token": token, "owner": str(owner), "until": now + float(ttl_seconds)}
        for sid, row in self._lineage_rows(session_id):
            if limit is not None and len(out) >= max(0, int(limit)):
                break
            if _pending(row, now):
                out.append(_delivery(sid, row, token))
                _set_meta(row, {**_meta(row), _LEASE: dict(lease)})
        return out

    def _settle(self, rewrite: Callable[[dict[str, Any]], dict[str, Any]],
                token_or_ids: str | Sequence[int] | None, owner: str) -> int:
        now = time.time()
        ids = None if isinstance(token_or_ids, str) else {int(i) for i in token_or_ids or ()}
        if ids is not None and not ids:
            return 0
        changed = 0
        for row in self._all_rows():
            meta = _meta(row)
            lease = meta.get(_LEASE) if isinstance(meta.get(_LEASE), dict) else None
            if meta.get(_CONSUMED) is not None or lease is None or lease.get("owner") != str(owner):
                continue
            if ids is None and lease.get("token") != token_or_ids:
                continue
            if ids is not None and (row.id not in ids or float(lease.get("until") or 0) <= now):
                continue
            _set_meta(row, rewrite(meta))
            changed += 1
        return changed

    def commit_caller_history_deliveries(self, reservation_token_or_ids: str | Sequence[int], owner: str) -> int:
        return self._settle(_consume, reservation_token_or_ids, owner)

    def release_caller_history_deliveries(self, session_id: Optional[SessionId] = None,
                                          row_ids: Optional[Sequence[int]] = None, *,
                                          reservation_token: Optional[str] = None, owner: Optional[str] = None) -> int:
        if owner is not None:
            return self._settle(_drop_lease, reservation_token if reservation_token is not None else row_ids, owner)
        ids = {int(i) for i in row_ids or ()}
        if not ids or not session_id:
            return 0
        changed = 0
        for _sid, row in self._lineage_rows(session_id):
            meta = _meta(row)
            if row.id in ids and meta.get(_CONSUMED) is not None:
                meta.pop(_CONSUMED)
                _set_meta(row, meta)
                changed += 1
        return changed
