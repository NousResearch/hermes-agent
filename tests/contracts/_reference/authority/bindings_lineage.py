"""Local lineage steps of the reference bindings repository over one view.

Every function takes the view it reads and writes: ``meta`` is the authority state's ``meta`` space (creation receipts under ``binding:<id>``, legacy bindings under ``gateway.local_legacy.v1:<id>``, history-control records under ``history_control:<id>``), ``tables`` the scope's domain tables (``sessions``, ``routing``), and ``handle`` the store handle whose lineage reference answers the compression chain step. A receipt is stored decoded (a dict); a text value is read as JSON, so a test can store an undecodable receipt.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any, Optional

from persistence import AuthorityRefusal, ProfileScope
from persistence.authority.bindings import local_session_id

from ..gateway.routing import StoredRoute

__all__ = [
    "HISTORY_PREFIX",
    "LEGACY_PREFIX",
    "RECEIPT_PREFIX",
    "adoptable_compression",
    "advance_local_target",
    "commit_local_target",
    "decode",
    "local_conversation_ids",
    "local_lineage_owner",
    "local_physical_target",
    "routed_entry",
    "store_route",
    "validate_local_lineage",
]

RECEIPT_PREFIX = "binding:"
LEGACY_PREFIX = "gateway.local_legacy.v1:"
HISTORY_PREFIX = "history_control:"


def _unavailable() -> AuthorityRefusal:
    return AuthorityRefusal("storage_unavailable")


def decode(value: Any) -> Any:
    """A stored receipt decoded: text is read as JSON (``ValueError`` when malformed), a stored value is returned as is."""
    return json.loads(value) if isinstance(value, str) else value


def _chain_step(handle: Any, scope: ProfileScope, sessions: dict[str, Any], session_id: str) -> Optional[str]:
    step: Optional[str] = handle.lineage._chain_step(scope, sessions, session_id)
    return step


def legacy_lineage_root(meta: dict[str, Any], receipt: dict[str, Any]) -> str:
    """The receipt's lineage root: its legacy session id when it binds one (``storage_unavailable`` unless that id is the receipt's own and the legacy record matches), else its session id."""
    root = receipt.get("legacy_session_id", receipt["session_id"])
    if "legacy_session_id" in receipt:
        if root != receipt["session_id"]:
            raise _unavailable()
        expected = {k: receipt[k] for k in ("session_id", "profile_id", "principal_id", "legacy_session_id")}
        if meta.get(LEGACY_PREFIX + root) != expected:
            raise _unavailable()
    return str(root)


def validate_local_lineage(handle: Any, scope: ProfileScope, meta: dict[str, Any], sessions: dict[str, Any], receipt: dict[str, Any]) -> str:
    """The validated receipt lineage's tip, else ``storage_unavailable``."""
    if "legacy_session_id" not in receipt and receipt["session_id"] != local_session_id(receipt["profile_id"], receipt["principal_id"], receipt["request_id"]):
        raise _unavailable()
    root = legacy_lineage_root(meta, receipt)
    lineage = receipt.get("lineage", [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt["entry"]["session_id"]):
        raise _unavailable()
    for parent_id, child_id in pairwise(lineage):
        parent, child = sessions.get(parent_id), sessions.get(child_id)
        if parent is None or child is None or child["parent_session_id"] != parent_id:
            raise _unavailable()
        if parent["end_reason"] == "session_reset":
            valid = (child.get("model_config") or {}).get("_reset_from") == parent_id
        else:
            valid = parent["end_reason"] == "compression" and _chain_step(handle, scope, sessions, parent_id) == child_id
        if not valid:
            raise _unavailable()
    if _chain_step(handle, scope, sessions, lineage[-1]) is not None:
        raise _unavailable()
    return str(lineage[-1])


def local_physical_target(handle: Any, scope: ProfileScope, meta: dict[str, Any], sessions: dict[str, Any], session_id: str) -> str:
    """The receipt's validated tip; the id itself without a receipt."""
    saved = meta.get(RECEIPT_PREFIX + session_id)
    if saved is None:
        return session_id
    receipt = decode(saved)
    if receipt["session_id"] != session_id:
        raise _unavailable()
    return validate_local_lineage(handle, scope, meta, sessions, receipt)


def local_lineage_owner(meta: dict[str, Any], sessions: dict[str, Any], session_id: str) -> str:
    """The row's local owner (its ``chat_id``) when the owner's receipt lineage holds the row, else the id itself."""
    row = sessions.get(session_id)
    owner = row.get("chat_id") if row else None
    if not owner or owner == session_id:
        return session_id
    saved = meta.get(RECEIPT_PREFIX + owner)
    try:
        lineage = decode(saved).get("lineage") if saved is not None else None
    except (TypeError, ValueError, AttributeError):
        return session_id
    return str(owner) if isinstance(lineage, list) and session_id in lineage else session_id


def local_conversation_ids(meta: dict[str, Any], sessions: dict[str, Any], session_id: str) -> list[str]:
    """The local owner and its receipt lineage, in order without repeats; ``[session_id]`` when no lineage list decodes."""
    owner = local_lineage_owner(meta, sessions, session_id)
    saved = meta.get(RECEIPT_PREFIX + owner)
    try:
        lineage = decode(saved).get("lineage") if saved is not None else None
    except (TypeError, ValueError, AttributeError):
        lineage = None
    if not isinstance(lineage, list):
        return [session_id]
    return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))


def _routes(tables: dict[str, Any]) -> dict[str, StoredRoute]:
    routes: dict[str, StoredRoute] = tables.setdefault("routing", {}).setdefault("", {})
    return routes


def routed_entry(tables: dict[str, Any], route: str, tip: str) -> Optional[dict[str, Any]]:
    """``route``'s routing entry while it still targets ``tip``, else None."""
    stored = _routes(tables).get(route)
    routed = json.loads(stored.text(route)) if stored is not None else None
    return routed if routed is not None and routed["session_id"] == tip else None


def store_route(tables: dict[str, Any], route: str, entry: dict[str, Any]) -> None:
    """Write ``entry`` as the routing index row of ``route`` in the ``""`` namespace, as raw entry JSON."""
    _routes(tables)[route] = StoredRoute.of(route, json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def commit_local_target(authority: dict[str, Any], tables: dict[str, Any], receipt: dict[str, Any], routed: dict[str, Any]) -> None:
    """Commit the route's entry, the receipt and the logical revision, together."""
    from .._authority import session_runtime

    if receipt["route"] in _routes(tables):
        store_route(tables, receipt["route"], routed)
    authority["meta"][RECEIPT_PREFIX + receipt["session_id"]] = receipt
    if receipt["session_id"] in tables.get("sessions", {}):
        session_runtime(authority, receipt["session_id"])["revision"] += 1


def advance_local_target(handle: Any, scope: ProfileScope, authority: dict[str, Any], tables: dict[str, Any], parent_session_id: str, child_session_id: str, *, entry: Optional[dict[str, Any]] = None) -> None:
    """Advance the parent's local receipt to ``child_session_id`` and commit it with the route entry: ``storage_unavailable`` for a receipt that does not match the parent or fails validation, ``admission_conflict`` when the route no longer targets the parent."""
    meta, sessions = authority["meta"], tables.setdefault("sessions", {})
    parent = sessions.get(parent_session_id)
    logical_id = parent.get("chat_id") if parent else None
    if not logical_id:
        return
    raw = meta.get(RECEIPT_PREFIX + logical_id)
    if raw is None:
        if logical_id.startswith("local-") or LEGACY_PREFIX + logical_id in meta:
            raise _unavailable()
        return
    try:
        receipt = json.loads(json.dumps(decode(raw)))
        if (receipt["session_id"] != logical_id or receipt["entry"]["session_id"] != parent_session_id
                or receipt["principal_id"] != parent["user_id"] or receipt["route"] != parent["session_key"]):
            raise _unavailable()
        routed = routed_entry(tables, receipt["route"], parent_session_id)
        if routed is None:
            raise AuthorityRefusal("admission_conflict")
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt["entry"] = candidate
        receipt["lineage"] = [*receipt.get("lineage", [logical_id]), child_session_id]
        validate_local_lineage(handle, scope, meta, sessions, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AuthorityRefusal):
            raise
        raise _unavailable() from exc
    commit_local_target(authority, tables, receipt, candidate)


def _untracked_steps(handle: Any, scope: ProfileScope, sessions: dict[str, Any], tip: str, lineage: list[str]) -> Optional[list[str]]:
    """Every compression step past ``tip`` in order; None when the chain leads back into itself."""
    steps: list[str] = []
    seen = {*lineage, tip}
    current = tip
    while (step := _chain_step(handle, scope, sessions, current)) is not None:
        if step in seen:
            return None
        seen.add(step)
        steps.append(step)
        current = step
    return steps


def adoptable_compression(handle: Any, scope: ProfileScope, tables: dict[str, Any], receipt: dict[str, Any]) -> Optional[tuple[list[str], dict[str, Any]]]:
    """``(steps, routed)`` when every compression step past the tip carries the receipt's route, owner and principal and the route still targets the tip, else None."""
    sessions = tables.setdefault("sessions", {})
    tip = receipt["entry"]["session_id"]
    children = _untracked_steps(handle, scope, sessions, tip, receipt.get("lineage", [receipt["session_id"]]))
    if not children:
        return None
    identity = (receipt["route"], receipt["session_id"], receipt["principal_id"])
    for child_id in children:
        child = sessions[child_id]
        if (child.get("session_key"), child.get("chat_id"), child.get("user_id")) != identity:
            return None
    routed = routed_entry(tables, receipt["route"], tip)
    return (children, routed) if routed is not None else None


def with_adopted_compression(receipt: dict[str, Any], children: list[str]) -> dict[str, Any]:
    """``receipt`` as ``adopt_untracked_compression`` commits it: ``children`` appended to its lineage and its entry moved to the last one; the caller validates the copy."""
    return dict(receipt, entry=dict(receipt["entry"], session_id=children[-1]), lineage=[*receipt.get("lineage", [receipt["session_id"]]), *children])
