"""The lineage writes of the reference bindings repository: compression adoption, the local ``/new`` reset and the boot hand-off of moved receipts."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any

from persistence import AuthorityRefusal, PermanentStoreError, SessionId
from persistence.authority.bindings import SessionBinding
from persistence.conversation.sessions import SessionLineageKind
from persistence.core.identity import JsonValue
from session_rules.surfaces import created_surface

from ..conversation.sessions import blank_session_row
from ..conversation.sessions_titles_config import sync_lineage_markers
from .bindings_lineage import (
    LEGACY_PREFIX,
    RECEIPT_PREFIX,
    adoptable_compression,
    advance_local_target,
    commit_local_target,
    decode,
    validate_local_lineage,
)

__all__ = ["LineageWriteVerbs", "hand_off_conversations", "moved_receipts"]

_LANE_IDENTITY = ("session_key", "source", "user_id", "chat_id", "chat_type", "thread_id", "display_name", "origin", "transport_profile")
_TITLE_RANK = {"derived": 0, "llm": 1, "user": 2}


def _title_rank(source: Any) -> int:
    """Title provenance rank: a pre-provenance (NULL) title ranks as the user's."""
    return 2 if source is None else _TITLE_RANK.get(str(getattr(source, "value", source)), 0)


def _require_not_executing(authority: dict[str, Any], session_ids: list[str]) -> None:
    """Per id, a started or unknown admission or a live worker refuses: ``unknown_execution`` when any is unknown, else ``session_busy``."""
    for sid in session_ids:
        states = {row["status"] for row in authority["admissions"].values()
                  if row.get("target_session_id") == sid and row["status"] in ("started", "unknown")}
        states |= {row["status"] for row in authority["workers"].values()
                   if row.get("session_id") == sid and row["status"] in ("registered", "running", "unknown")}
        if "unknown" in states:
            raise AuthorityRefusal("unknown_execution")
        if states:
            raise AuthorityRefusal("session_busy")


def _reset_child(handle: Any, parent: dict[str, Any], child_id: str, policy: dict[str, Any]) -> dict[str, Any]:
    """The reset child row: routing, owner and display columns copied from the parent, no pin or archive state."""
    row = blank_session_row(child_id, lineage_kind=SessionLineageKind.NONE, model_config={"_reset_from": parent["session_id"]})
    row.update(source=policy["source"], created_source=created_surface(parent.get("created_source")) or created_surface(policy["source"]),
               model=policy["model"], parent_session_id=parent["session_id"], cwd=policy["cwd"] or parent.get("cwd"),
               profile_name=parent.get("profile_name") or handle.owner_profile, started_at=datetime.now(UTC))
    for name in ("git_branch", "git_repo_root", "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin", "tool_pin"):
        row[name] = copy.deepcopy(parent.get(name))
    sync_lineage_markers(row)
    return row


def _segment_root(sessions: dict[str, Any], segments: list[str], target: str) -> str:
    """The first row of ``target``'s ``/new`` segment."""
    if target not in segments:
        return target
    index = segments.index(target)
    while index > 0 and (sessions.get(segments[index - 1]) or {}).get("end_reason") == "compression":
        index -= 1
    return segments[index]


def _title_follows_head(sessions: dict[str, Any], owner: str, head: str) -> None:
    """The conversation's name moves to its head over a title of lower provenance there."""
    row = sessions.get(owner)
    if head == owner or row is None or not row.get("title"):
        return
    kept = sessions.get(head)
    if kept is None or (kept.get("title") and _title_rank(kept.get("title_source")) >= _title_rank(row.get("title_source"))):
        return
    title, source, hidden = row["title"], row.get("title_source"), row.get("hidden")
    row["title"] = None
    kept.update(title=title, title_source=source, hidden=hidden)


def moved_receipts(meta: dict[str, Any], sessions: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Owner -> (current target, its source) for receipts whose target row now carries another route."""
    moved: dict[str, tuple[str, str]] = {}
    for key, raw in meta.items():
        if not (isinstance(key, str) and key.startswith(RECEIPT_PREFIX)):
            continue
        try:
            receipt = decode(raw)
        except ValueError:
            continue
        entry = receipt.get("entry") if isinstance(receipt, dict) else None
        row = sessions.get(entry.get("session_id")) if isinstance(entry, dict) else None
        if row is not None and row.get("session_key") is not None and row["session_key"] != receipt.get("route"):
            moved[key[len(RECEIPT_PREFIX):]] = (row["session_id"], row["source"])
    return moved


def hand_off_conversations(meta: dict[str, Any], sessions: dict[str, Any], moved: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Retire each moved receipt and give its conversation's other segments the lane identity of its current target, the title following the head; a receipt with a malformed lineage stays."""
    handed: dict[str, tuple[str, str]] = {}
    for owner, (target, source) in moved.items():
        saved = meta.get(RECEIPT_PREFIX + owner)
        receipt = decode(saved) if saved is not None else None
        lineage = (receipt if isinstance(receipt, dict) else {}).get("lineage", [owner])
        if not isinstance(lineage, list) or not all(isinstance(sid, str) for sid in lineage):
            continue
        segments = list(dict.fromkeys([owner, *lineage]))
        lane = {name: copy.deepcopy(sessions[target].get(name)) for name in _LANE_IDENTITY}
        for sid in segments:
            if sid != target and sid in sessions:
                sessions[sid].update(copy.deepcopy(lane))
        _title_follows_head(sessions, owner, _segment_root(sessions, segments, target))
        handed[owner] = (target, source)
    for owner in handed:
        meta.pop(RECEIPT_PREFIX + owner, None)
        meta.pop(LEGACY_PREFIX + owner, None)
    return handed


class LineageWriteVerbs:
    """``adopt_untracked_compression``, ``reset_target`` and ``release_handed_off_receipts``; the host class provides ``_backend`` and ``_state``."""

    _backend: Any

    def _state(self) -> Any:
        raise NotImplementedError

    def adopt_untracked_compression(self, *, epoch: int, session_id: SessionId, expected: SessionBinding) -> None:
        from .bindings import _receipt_of, _to_binding

        handle, scope, state = self._backend, self._backend.scope, self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            saved = staged.authority["meta"].get(RECEIPT_PREFIX + str(session_id))
            if saved is None:
                raise TypeError("the JSON object must be str, bytes or bytearray, not NoneType")
            if _to_binding(saved) != expected:
                return
            receipt = _receipt_of(expected)
            adoptable = adoptable_compression(handle, scope, staged.scope, receipt)
            if adoptable is None:
                return
            children, routed = adoptable
            receipt["entry"] = dict(receipt["entry"], session_id=children[-1])
            receipt["lineage"] = [*receipt.get("lineage", [str(session_id)]), *children]
            validate_local_lineage(handle, scope, staged.authority["meta"], staged.scope.setdefault("sessions", {}), receipt)
            commit_local_target(staged.authority, staged.scope, receipt, dict(routed, session_id=children[-1]))

    def reset_target(self, *, epoch: int, parent_session_id: SessionId, entry: dict[str, JsonValue]) -> None:
        from .._authority import session_runtime
        from ..conversation.transcript_store import advance_conversation_generation
        from ..workers.sessions import _StagedView

        handle, scope, state = self._backend, self._backend.scope, self._state()
        with state.transaction() as staged:
            authority, sessions = staged.authority, staged.scope.setdefault("sessions", {})
            state.require_epoch(authority, epoch)
            parent = sessions.get(str(parent_session_id))
            if parent is None or parent["end_reason"] == "compression":
                raise AuthorityRefusal("admission_conflict")
            saved = authority["meta"].get(RECEIPT_PREFIX + parent["chat_id"])
            if saved is None:
                raise AuthorityRefusal("storage_unavailable")
            receipt = json.loads(json.dumps(decode(saved)))
            logical_id = receipt["session_id"]
            if validate_local_lineage(handle, scope, authority["meta"], sessions, receipt) != parent_session_id:
                raise AuthorityRefusal("admission_conflict")
            _require_not_executing(authority, list(dict.fromkeys([logical_id, str(parent_session_id)])))
            child_id = str(entry["session_id"])
            if child_id in sessions:
                raise PermanentStoreError("UNIQUE constraint failed: sessions.id")
            sessions[child_id] = _reset_child(handle, parent, child_id, receipt["policy"])
            parent.update(ended_at=datetime.now(UTC), end_reason="session_reset")
            advance_conversation_generation(_StagedView(handle, scope, staged.scope), scope, str(parent_session_id), "session_reset")
            advance_local_target(handle, scope, authority, staged.scope, str(parent_session_id), child_id, entry=json.loads(json.dumps(dict(entry))))
            session_runtime(authority, SessionId(logical_id))["generation"] += 1

    def release_handed_off_receipts(self, *, epoch: int, lanes: frozenset[str]) -> dict[str, tuple[str, str]]:
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            meta, sessions = staged.authority["meta"], staged.scope.setdefault("sessions", {})
            moved = {owner: pair for owner, pair in moved_receipts(meta, sessions).items() if pair[1] in lanes}
            return hand_off_conversations(meta, sessions, moved)
