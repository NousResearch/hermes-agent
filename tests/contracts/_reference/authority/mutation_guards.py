"""Mutation guards and the local snapshot of the runtime reference.

The local policy receipt is the binding reference's stored receipt; its validation is the binding reference's lineage walk. The transcript guard is the message reference's live-lease refusal, skipped on a row compression closed. :func:`delete_targets` is the delete target walk ``mutate(Delete)`` and ``delete_footprint`` run, over the binding reference's receipts and history bindings, the lineage reference's continuation rule and the session reference's delegate walk.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from persistence.authority.runtime import LocalSnapshot, PreparedMutation
from session_rules.legacy_local import bindings_agree, owners_agree
from persistence.core.errors import AuthorityRefusal
from persistence.core.identity import SessionId

from .._lease_fence import refuse_live_leases
from ..conversation.lineage import _CHAIN_CAP, _non_continuation
from . import admissions_rows as rows
from .bindings import _owner_user, _routed, binding_text, history_row
from .bindings_lineage import HISTORY_PREFIX, RECEIPT_PREFIX, decode, local_lineage_owner, validate_local_lineage


def _states(authority: Mapping[str, Any], session_id: str, admission_statuses: tuple[str, ...]) -> set[str]:
    states = {r["status"] for r in authority["admissions"].values() if r["target_session_id"] == session_id and r["status"] in admission_statuses}
    states |= {w["status"] for w in authority["workers"].values() if w["session_id"] == session_id and w["status"] in ("registered", "running", "unknown")}
    return states


def _refuse(states: set[str]) -> None:
    if "unknown" in states:
        raise AuthorityRefusal("unknown_execution")
    if states:
        raise AuthorityRefusal("session_busy")


def _require_transcript_unleased(store: Any, tables: Mapping[str, Any], session_id: str) -> None:
    row = tables.get("sessions", {}).get(session_id)
    if row is not None and row.get("end_reason") == "compression" and row.get("ended_at") is not None:
        return
    refuse_live_leases(store, store.scope, SessionId(session_id))


def require_not_executing(authority: Mapping[str, Any], session_ids: Iterable[str]) -> None:
    """A started or unknown admission, or a live worker, refuses."""
    for sid in session_ids:
        _refuse(_states(authority, sid, ("started", "unknown")))


def require_target_advanceable(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_ids: Iterable[str]) -> None:
    """Executing work refuses, then a live transcript lease."""
    ids = list(session_ids)
    require_not_executing(authority, ids)
    for sid in ids:
        _require_transcript_unleased(store, tables, sid)


def require_idle(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_ids: Iterable[str]) -> None:
    """Any queued, started or unknown admission or live worker refuses, then a live transcript lease, per id."""
    for sid in session_ids:
        _refuse(_states(authority, sid, ("queued", "started", "unknown")))
        _require_transcript_unleased(store, tables, sid)


def _continuation_edges(sessions: Mapping[str, Any], session_id: str) -> list[str]:
    """The continuation children of ``session_id`` when compression ended it, and its parent when compression ended that."""
    row = sessions.get(session_id)
    edges = []
    if row is not None and row["end_reason"] == "compression":
        edges = [sid for sid, child in sessions.items() if child["parent_session_id"] == session_id and not _non_continuation(child, session_id)]
    parent_id = row["parent_session_id"] if row is not None else None
    parent = sessions.get(parent_id) if parent_id else None
    if parent is not None and parent["end_reason"] == "compression" and not _non_continuation(row, parent_id):
        edges.append(parent_id)
    return edges


def _same_history_owner(meta: Mapping[Any, Any], sessions: Mapping[str, Any], anchor_id: str, session_id: str, rule: Any) -> bool:
    """Whether a member shares the anchor's history owner: a routed member carries the anchor's user, route and chat; a routeless one a binding agreeing with the anchor's, and unbound it is legacy history or shares an unrouted anchor's source, with no owner stamp disagreeing with the anchor's under ``rule``."""
    anchor, row = sessions.get(anchor_id), sessions.get(session_id)
    if anchor is None or row is None:
        return False
    if _routed(row):
        return all(row[k] == anchor[k] for k in ("user_id", "session_key", "chat_id"))
    binding = binding_text(meta.get(HISTORY_PREFIX + session_id))
    if not bindings_agree(binding, binding_text(meta.get(HISTORY_PREFIX + anchor_id)), rule):
        return False
    routeless_peer = row["source"] == anchor["source"] and not _routed(anchor)
    return binding is not None or ((history_row(row, rule) or routeless_peer)
                                   and (_owner_user(row) in (None, "") or owners_agree(_owner_user(row), anchor["user_id"], rule)))


def delete_targets(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str, rule: Any = None) -> list[str]:
    """The id, its local owner and the owner's validated receipt lineage (``storage_unavailable`` when it does not validate), then compression relatives of the same history owner and delegate subtrees to a fixed point; the id first, the rest sorted."""
    meta, sessions = authority["meta"], tables.get("sessions", {})
    targets = {session_id}
    owner = local_lineage_owner(meta, sessions, session_id)
    targets.add(owner)
    saved = meta.get(RECEIPT_PREFIX + owner)
    if saved is not None:
        receipt = decode(saved)
        validate_local_lineage(store, store.scope, meta, sessions, receipt)
        targets.update(receipt.get("lineage", [session_id]))
    anchor = dict.fromkeys(targets, session_id)
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        found: dict[str, str] = {}
        for sid in frontier:
            for nxt in _continuation_edges(sessions, sid):
                if nxt not in targets and nxt not in found and _same_history_owner(meta, sessions, anchor[sid], nxt, rule):
                    found[nxt] = anchor[sid]
        for sid in store.sessions._delegate_children(sessions, frontier):
            if sid not in targets:
                found.setdefault(sid, sid)
        if not found:
            break
        anchor.update(found)
        targets.update(found)
        frontier = list(found)
    return [session_id, *sorted(targets - {session_id})]


def local_snapshot(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> LocalSnapshot:
    """The receipt (``runtime_coordination_required`` without one), its validated target, the advanceable fence, the target's revision."""
    receipt = rows.local_receipt(authority, session_id)
    if receipt is None:
        raise AuthorityRefusal("runtime_coordination_required")
    target = rows.physical_target(store, authority, tables, session_id)
    require_target_advanceable(store, authority, tables, list(dict.fromkeys([session_id, target])))
    revision = authority["sessions"].get(target, {}).get("revision", 0)
    return LocalSnapshot(receipt=copy.deepcopy(receipt), target=target, target_revision=revision)


def validate_prepared(store: Any, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str, prepared: PreparedMutation | None) -> LocalSnapshot:
    """No prepared commit is ``runtime_coordination_required``; a snapshot that moved is ``revision_conflict``."""
    if not isinstance(prepared, PreparedMutation):
        raise AuthorityRefusal("runtime_coordination_required")
    current = local_snapshot(store, authority, tables, session_id)
    if current != prepared.snapshot:
        raise rows.revision_conflict(session_id)
    return current
