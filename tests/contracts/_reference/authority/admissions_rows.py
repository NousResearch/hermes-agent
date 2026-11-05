"""Shared row helpers of the admission reference: keys, records, refusals and the steps several verbs run.

An admission row lives in the scope's authority state (``admissions``: id -> dict) with the keys the worker, session-delete and listing references read: ``admission_id``, ``request_id``, ``principal_id``, ``target_session_id``, ``lineage``, ``payload``, ``digest``, ``intent``, ``status``, ``outcome``, ``owner_epoch``, ``generation``, ``seq``. Durable values live in the authority ``meta`` table: the terminal tombstone and identity pointer under :func:`_terminal_key` and :func:`_identity_key` (session retirement writes them), results under :func:`_result_key`, and string keys for the replacement bindings, the API settings and the media and spillover retirement candidates.
"""

from __future__ import annotations

import copy
import json
import time
import sqlite3
from dataclasses import fields
from typing import Any, Mapping

from persistence.authority.admissions import AdmissionRecord, ClaimedAdmission
from persistence.authority.vocabulary import STEERING, admission_fingerprint
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, FenceStale, StaleFenceKind, UnknownExecution

from .._authority import session_runtime
from .._real_text import real_as_text

#: Meta key prefix of a queued replacement binding.
REPLACEMENT_PREFIX = "gateway.queued_replacement.v1."
#: Meta key prefix of an API target's settings.
API_SETTINGS_PREFIX = "gateway.api.settings.v1."
#: Meta key prefixes of the media and spillover retirement candidates.
MEDIA_PREFIX = "gateway.media.retired.v1."
SPILLOVER_PREFIX = "gateway.spillover.retired.v1."


def _identity_key(principal_id: str, session_id: str, request_id: str) -> tuple[str, str, str, str]:
    """Meta key of the terminal identity pointer (the retired-branch lookup); session retirement writes it."""
    return ("terminal_identity", principal_id, session_id, request_id)


def _terminal_key(admission_id: str) -> tuple[str, str]:
    """Meta key of a terminal admission tombstone; session retirement writes it."""
    return ("terminal_admission", "" + admission_id)


def _result_key(admission_id: str) -> tuple[str, str]:
    """Meta key of a settlement's stored result."""
    return ("admission_result", admission_id)


def _bound_id(value: Any, parameter: int = 1) -> Any:
    """The value SQLite binds for ``value`` (text stays text, an int binds as its text in a TEXT column, a float as SQLite's text cast, a buffer as bytes); an unbindable type raises sqlite3's error."""
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, int):
        value = int.__int__(value)
        if not -(2**63) <= value < 2**63:
            raise OverflowError("Python int too large to convert to SQLite INTEGER")
        return str(value)
    if isinstance(value, float):
        return real_as_text(value)
    try:
        return bytes(memoryview(value))
    except TypeError:
        pass
    raise sqlite3.ProgrammingError(f"Error binding parameter {parameter}: type '{type(value).__name__}' is not supported")


def text(value: Any) -> str:
    """A non-empty string of at most 1024 characters, else ``invalid_params``."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal("invalid_params")
    return value


def encode(value: Any) -> Any:
    """The canonical round trip of a dict, else ``invalid_params``."""
    return json.loads(canonical_json(value))


def digest(session_id: str, payload: Any, intent: str) -> str:
    """The admission digest stored on the row: target, payload and intent."""
    return admission_fingerprint(canonical_target=session_id, payload={"input": payload, "intent": intent})


def stale(entity: str, entity_id: Any) -> FenceStale:
    return FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(entity_id))


def revision_conflict(entity_id: Any) -> FenceStale:
    return FenceStale(StaleFenceKind.REVISION_CONFLICT, "session", str(entity_id))


def unknown(entity: str, entity_id: Any) -> UnknownExecution:
    return UnknownExecution(entity, str(entity_id))


def record(row: Mapping[str, Any]) -> AdmissionRecord:
    """The row without its digest and lineage, the payload a detached copy."""
    return AdmissionRecord(
        seq=row["seq"], admission_id=row["admission_id"], request_id=row["request_id"], principal_id=row["principal_id"],
        target_session_id=row["target_session_id"], payload=copy.deepcopy(row["payload"]), intent=row["intent"],
        status=row["status"], outcome=row.get("outcome"), owner_epoch=row.get("owner_epoch"), generation=row.get("generation"),
    )


def claimed(row: Mapping[str, Any], mark: int, queued_owner_epoch: Any) -> ClaimedAdmission:
    base = record(row)
    return ClaimedAdmission(**{f.name: getattr(base, f.name) for f in fields(AdmissionRecord)}, transcript_mark=mark, queued_owner_epoch=queued_owner_epoch)


def admission(authority: Mapping[str, Any], admission_id: Any) -> dict[str, Any]:
    """The live row, else its tombstone, else ``not_found``."""
    row = authority["admissions"].get(_bound_id(admission_id))
    if row is None:
        row = authority["meta"].get(_terminal_key(admission_id))
    if row is None:
        raise AuthorityRefusal("not_found")
    return row


def lookup(authority: Mapping[str, Any], admission_id: Any) -> dict[str, Any] | None:
    try:
        return admission(authority, admission_id)
    except AuthorityRefusal:
        return None


def require_session(tables: Mapping[str, Any], session_id: Any) -> dict[str, Any]:
    """The session row, else ``not_found``."""
    row = tables.get("sessions", {}).get(_bound_id(session_id))
    if row is None:
        raise AuthorityRefusal("not_found")
    return row


def rows_by_seq(authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(authority["admissions"].values(), key=lambda row: row["seq"])


def session_rows(authority: Mapping[str, Any], session_id: Any) -> list[dict[str, Any]]:
    sid = _bound_id(session_id)
    return [row for row in rows_by_seq(authority) if row["target_session_id"] == sid]


def bump_revision(authority: dict[str, Any], session_id: str) -> None:
    session_runtime(authority, session_id)["revision"] += 1


def max_message_id(tables: Mapping[str, Any]) -> int:
    """The highest message id over every session, 0 when there is none."""
    transcripts = tables.get("messages", {}).get("transcripts", {})
    return max((row.id for transcript in transcripts.values() for row in transcript.rows), default=0)


def retire_workers(authority: dict[str, Any], row: Mapping[str, Any], owner_epoch: Any) -> None:
    """The workers at the row's generation and ``owner_epoch`` on its target or its lineage tail go terminal."""
    lineage = row.get("lineage") or [row["target_session_id"]]
    targets = {row["target_session_id"], lineage[-1]}
    for worker in authority["workers"].values():
        if worker["session_id"] in targets and worker["generation"] == row["generation"] and worker["owner_epoch"] == owner_epoch:
            worker["status"] = "terminal"


def requeue_transfer(authority: dict[str, Any], admission_id: Any, epoch: int, generation: Any) -> bool:
    """Requeue a steering transfer still terminal at this epoch and generation; whether it was requeued."""
    row = authority["admissions"].get(_bound_id(admission_id))
    if row is None or not (row["status"] == "terminal" and row["outcome"] == STEERING and row["owner_epoch"] == epoch and row["generation"] == generation):
        return False
    row.update(status="queued", outcome=None, generation=None)
    return True


def settle_transfers(authority: dict[str, Any], *, epoch: int, generation: Any, transfers: Mapping[str, bool] | None, outcome: str) -> None:
    """Settle steering transfers: an undelivered one is requeued, a delivered one's outcome becomes ``steered`` when the turn completed, else ``interrupted``."""
    for admission_id, delivered in (transfers or {}).items():
        if not delivered:
            requeue_transfer(authority, admission_id, epoch, generation)
            continue
        row = authority["admissions"].get(_bound_id(admission_id))
        if row is not None and row["status"] == "terminal" and row["outcome"] == STEERING and row["owner_epoch"] == epoch and row["generation"] == generation:
            row["outcome"] = "steered" if outcome == "completed" else "interrupted"


def now() -> float:
    return time.time()


def physical_target(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> str:
    """The binding receipt's current segment, else the id itself (the binding reference's lineage walk over this view)."""
    from .bindings import resolve_physical_target

    return resolve_physical_target(store, store.scope, authority, tables, session_id)


def chain_tip(store: Any, tables: Mapping[str, Any], session_id: str) -> str:
    """The tip of the compression continuation walk, ``admission_conflict`` on a cycle or past 100 steps."""
    sessions = tables.get("sessions", {})
    if session_id not in sessions:
        raise AuthorityRefusal("not_found")
    chain = [session_id]
    for _ in range(100):
        child = store.lineage._chain_step(store.scope, sessions, chain[-1])
        if child is None:
            return chain[-1]
        if child in chain:
            raise AuthorityRefusal("admission_conflict")
        chain.append(child)
    raise AuthorityRefusal("admission_conflict")


def runtime_lineage(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> set[str]:
    """The session, its physical target, its local owner when the owner's receipt lineage holds it, and every compression relative of those."""
    targets = {session_id, physical_target(store, authority, tables, session_id)}
    sessions = tables.get("sessions", {})
    owner = (sessions.get(session_id) or {}).get("chat_id")
    if owner:
        receipt = local_receipt(authority, owner)
        if receipt is not None and session_id in receipt.get("lineage", [owner]):
            targets.update({owner, physical_target(store, authority, tables, owner)})
    expanded: set[str] = set()
    for sid in targets:
        expanded.update(store.sessions._lineage_ids(sessions, sid) if sid in sessions else {sid})
    return targets | expanded


def local_receipt(authority: Mapping[str, Any], session_id: str) -> dict[str, Any] | None:
    """The local policy receipt the binding reference stores for ``session_id``."""
    from .bindings import _RECEIPT_PREFIX

    return authority["meta"].get(_RECEIPT_PREFIX + session_id)


def local_lineage_owner(authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> str:
    """The row's local owner (its ``chat_id``) when the owner's receipt lineage holds the row, else the id itself."""
    row = tables.get("sessions", {}).get(session_id)
    owner = row.get("chat_id") if row else None
    if not owner or owner == session_id:
        return session_id
    receipt = local_receipt(authority, owner)
    lineage = receipt.get("lineage") if isinstance(receipt, dict) else None
    return owner if isinstance(lineage, list) and session_id in lineage else session_id
