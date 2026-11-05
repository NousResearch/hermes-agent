"""Deletion, retirement and backfills of the in-memory session reference, mixed into ``InMemorySessionRepository`` (``sessions.py``), which supplies the row storage."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Sequence

from persistence import (
    AuthorityRefusal,
    JsonValue,
    PermanentStoreError,
    ProfileScope,
    SessionId,
    SessionWriteGuarded,
    UnknownExecution,
)
from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.conversation.sessions import (
    END_REASON_COMPRESSION,
    RETAG_GATE_PREFIX,
    RETAG_SOURCE_FROM,
    RETAG_SOURCE_TO,
    DeleteManyOutcome,
)

from session_rules.media import admission_media_references, hosted_document_references, spillover_names_in_text

from .._authority import RESULT_SCRUBBED, authority_state, record_retirement
from .._lease_fence import lease_guarded
from ..housekeeping.state_meta import state_meta_table
from .sessions_titles_config import ascii_lower


def _retire_media(meta: dict[str, Any], row: Mapping[str, Any], media_root: Optional[str]) -> None:
    """The admission's native media, the hosted documents its prompt names under ``media_root`` and its API media become retired-media candidates, the first write kept. Without a media root no hosted document is named."""
    from ..authority.admissions_rows import MEDIA_PREFIX

    payload = row["payload"]
    hosted = hosted_document_references(row["request_id"], payload, media_root) if media_root is not None else []
    for reference in [*admission_media_references(payload), *hosted, *payload.get("api_turn_v1", {}).get("media", [])]:
        meta.setdefault(MEDIA_PREFIX + hashlib.sha256(reference["path"].encode()).hexdigest(), json.dumps(reference))


def _retire_spillover(meta: dict[str, Any], staged_scope: Mapping[str, Any], doomed: Sequence[str]) -> None:
    """Every spillover archive the doomed sessions' rows name in their stored content or API content becomes a retirement candidate, the first write kept."""
    from ..authority.admissions_rows import SPILLOVER_PREFIX
    from .messages_reads import _content_text

    transcripts = staged_scope.get("messages", {}).get("transcripts", {})
    for sid in doomed:
        for row in transcripts[sid].rows if sid in transcripts else ():
            for name in spillover_names_in_text(_content_text(row.message.content), row.message.api_content):
                meta.setdefault(SPILLOVER_PREFIX + name, json.dumps(name))


class SessionDeleteMixin:
    """Delete, retirement and backfill verbs; ``InMemorySessionRepository`` supplies ``_backend``, ``_table``, ``_message_count``, ``_transcripts``, ``_owned_lineage_ids``, ``_local_conversation_ids`` and ``_flags_lock``."""

    _backend: Any
    _table: Callable[[ProfileScope], dict[str, dict[str, Any]]]
    _message_count: Callable[[ProfileScope, Mapping[str, Any]], int]
    _transcripts: Callable[[ProfileScope], dict[str, Any]]
    _owned_lineage_ids: Callable[[ProfileScope, str], list[str]]
    _local_conversation_ids: Callable[[ProfileScope, str], list[str]]
    _flags_lock: Any

    # ── the ledger fence ──────────────────────────────────────────────────

    def seed_ledger_admission(self, session_id: SessionId, *, status: str = "queued") -> None:
        """Suite seeding seam standing in for the admissions capability: stages one admission row through the shared authority transaction, in the admissions column vocabulary (``target_session_id``, ``status``) so admissions rows satisfy the same reads."""
        scope = self._backend.scope
        state = authority_state(self._backend, scope)
        with state.transaction() as staged:
            admission_id = f"seed-adm-{len(staged.authority['admissions'])}"
            staged.authority["admissions"][admission_id] = {"target_session_id": str(session_id), "status": status}

    @staticmethod
    def _live_ledger_states(authority: Mapping[str, Any], session_id: str) -> set[Any]:
        """The non-terminal admission and worker statuses addressing one session."""
        states = {row.get("status") for row in authority["admissions"].values() if row.get("target_session_id") == session_id}
        states |= {row.get("status") for row in authority["workers"].values() if row.get("session_id") == session_id}
        return states - {"terminal", None}

    def _require_ledger_idle(self, authority: Mapping[str, Any], session_ids: Sequence[str]) -> None:
        """Refuse deleting a target with live ledger work; unknown wins over busy per target."""
        for sid in session_ids:
            live = self._live_ledger_states(authority, sid)
            if "unknown" in live:
                raise UnknownExecution("session", sid)
            if live:
                raise AuthorityRefusal("session_busy")

    def _ledger_idle(self, authority: Mapping[str, Any], session_id: str) -> bool:
        return not self._live_ledger_states(authority, session_id)

    @staticmethod
    def _retire_staged(staged_scope: dict[str, Any], staged_authority: dict[str, Any], doomed: Sequence[str], media_root: Optional[str]) -> None:
        """Record the spillover archives their rows name, remove the rows, drop their routing entries, retire their admissions' media under ``media_root`` and publish their permanent retirement markers on one staged view (published as one atomic swap)."""
        from ..authority.admissions import _identity_key, _terminal_key

        sessions: dict[str, dict[str, Any]] = staged_scope.get("sessions", {})
        doomed_set = set(doomed)
        admissions = staged_authority["admissions"]
        meta = staged_authority["meta"]
        _retire_spillover(meta, staged_scope, doomed)
        for admission_id, row in list(admissions.items()):
            if row["target_session_id"] not in doomed_set:
                continue
            _retire_media(meta, row, media_root)
            row["payload"] = {}
            row["lineage"] = []
            meta[_terminal_key(admission_id)] = row
            meta[_identity_key(row["principal_id"], row["target_session_id"], row["request_id"])] = admission_id
            del admissions[admission_id]
        workers = staged_authority["workers"]
        receipts = staged_authority["receipts"]
        for execution_id, row in list(workers.items()):
            if row["session_id"] not in doomed_set:
                continue
            keys = sorted(key for key in receipts if isinstance(key, tuple) and len(key) == 2 and key[0] == execution_id)
            saved = {key[1]: receipts.pop(key) for key in keys}
            for key in keys[:-1]:
                saved[key[1]]["result"] = None
                saved[key[1]][RESULT_SCRUBBED] = True
            terminal_key = ("terminal_worker", execution_id)
            if terminal_key in meta:
                raise PermanentStoreError(f"terminal worker receipt already exists: {execution_id}")
            meta[terminal_key] = dict(row, receipts=saved)
            del workers[execution_id]
        for sid in doomed:
            sessions.pop(sid, None)
        # Routes whose entry points at a doomed id vanish with the row, in the same transition.
        for entries in staged_scope.get("routing", {}).values():
            for key in [k for k, entry in entries.items() if str(entry.session_id) in doomed_set]:
                del entries[key]
        for sid in doomed:
            record_retirement(staged_authority, SessionId(sid))

    def _delete_cascade(self, scope: ProfileScope, doomed: Sequence[str], media_root: str) -> None:
        """Retire ``doomed`` in one transition (refusals raise before anything changes) and orphan surviving children."""
        state = authority_state(self._backend, scope)
        with state.transaction() as staged:
            self._require_ledger_idle(staged.authority, doomed)
            self._retire_staged(staged.scope, staged.authority, doomed, media_root)
            doomed_set = set(doomed)
            for row in staged.scope.get("sessions", {}).values():
                if row["parent_session_id"] in doomed_set:
                    row["parent_session_id"] = None

    def _guarded(self, scope: ProfileScope, session_ids: Sequence[str]) -> bool:
        """Whether any of the ids holds a live turn lease or compression lock."""
        return any(lease_guarded(self._backend, scope, SessionId(sid)) for sid in session_ids)

    # ── delete targets ────────────────────────────────────────────────────

    @staticmethod
    def _delegate_children(table: dict[str, dict[str, Any]], parent_ids: Sequence[str]) -> list[str]:
        """Delegate children of ``parent_ids`` walked recursively (a ``_delegate_from`` marker naming a frontier row, or any marker on a child of one); the parents themselves never return, even over a marker cycle."""
        seeds = {sid for sid in parent_ids if sid}
        found = set(seeds)
        frontier = set(seeds)
        while frontier:
            step = {sid for sid, row in table.items() if sid not in found and _delegate_of(row, frontier)}
            found |= step
            frontier = step
        return [sid for sid in found if sid not in seeds]

    def _with_delegates(self, table: dict[str, dict[str, Any]], session_ids: Sequence[str]) -> list[str]:
        return [*session_ids, *self._delegate_children(table, session_ids)]

    def _expand_compression_lineage(self, scope: ProfileScope, session_ids: Sequence[str]) -> list[str]:
        """Every member of the ids' compression chains, both directions (stale sibling continuations included), plus every segment of a local conversation they belong to."""
        table = self._table(scope)
        seen = dict.fromkeys(session_ids)
        frontier = list(session_ids)
        while frontier:
            step = [parent for sid in frontier if (parent := _continued_parent(table, table.get(sid)))]
            step += [child for child, row in table.items() if _continued_parent(table, row) in frontier]
            frontier = [sid for sid in dict.fromkeys(step) if sid not in seen]
            seen.update(dict.fromkeys(frontier))
        for sid in list(seen):
            seen.update(dict.fromkeys(self._local_conversation_ids(scope, sid)))
        return list(seen)

    def delete_targets(self, session_id: SessionId) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        table = self._table(scope)
        if session_id not in table:
            return ()
        lineage = self._owned_lineage_ids(scope, str(session_id))
        delegates = self._delegate_children(table, lineage)
        return tuple(SessionId(sid) for sid in (session_id, *sorted(set(lineage) - {session_id}), *sorted(delegates)))

    def _display_unchanged(self, scope: ProfileScope, session_id: SessionId, rows: Sequence[Mapping[str, JsonValue]]) -> bool:
        """``expected_display``'s per-session check: the session exists and its DISPLAY view reads exactly *rows* in wire shape."""
        if session_id not in self._table(scope):
            return False
        page = TranscriptRead(view=TranscriptView.DISPLAY)
        current = [row.as_wire_row() for row in self._backend.messages.read_transcript(session_id, page)]
        return current == [dict(row) for row in rows]

    # ── delete ────────────────────────────────────────────────────────────

    def delete(
        self,
        session_id: SessionId,
        *,
        expected_ids: Optional[Sequence[SessionId]] = None,
        expected_display: Optional[Mapping[SessionId, Sequence[Mapping[str, JsonValue]]]] = None,
        refuse_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        if include_compression_chain:
            # The chain variant takes the bulk path, where a guarded root is kept rather than refused.
            outcome = self.delete_many([session_id], skip_guarded=refuse_guarded, include_compression_chain=True, media_root=media_root)
            return outcome.removed_ids if outcome.deleted else ()
        targets = self.delete_targets(session_id)
        if not targets:
            return ()
        if refuse_guarded and self._guarded(scope, targets):
            raise SessionWriteGuarded(session_id)
        if expected_ids is not None and set(expected_ids) != set(targets):
            return ()
        if expected_display is not None and not all(self._display_unchanged(scope, sid, rows) for sid, rows in expected_display.items()):
            return ()
        self._delete_cascade(scope, targets, media_root)
        return targets

    def delete_many(
        self,
        session_ids: Sequence[SessionId],
        *,
        skip_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> DeleteManyOutcome:
        scope = self._backend.scope
        table = self._table(scope)
        existing = list(dict.fromkeys(str(sid) for sid in session_ids if sid and sid in table))

        def expand(ids: Sequence[str]) -> list[str]:
            if include_compression_chain:
                return self._expand_compression_lineage(scope, ids)
            return list(dict.fromkeys(member for sid in ids for member in self._owned_lineage_ids(scope, sid)))

        # A root is kept when it, its lineage or any delegate child the cascade would take is guarded.
        skipped = sorted(sid for sid in existing if skip_guarded and self._guarded(scope, self._with_delegates(table, expand([sid]))))
        existing = [sid for sid in existing if sid not in skipped]
        doomed = list(dict.fromkeys(self._with_delegates(table, expand(existing)))) if existing else []
        if doomed:
            self._delete_cascade(scope, doomed, media_root)
        return DeleteManyOutcome(
            deleted=len(existing),
            skipped_guarded=tuple(SessionId(sid) for sid in skipped),
            removed_ids=tuple(SessionId(sid) for sid in doomed),
        )

    # ── empty rows ────────────────────────────────────────────────────────

    def is_empty_candidate(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        table = self._table(scope)
        row = table.get(session_id)
        return (
            row is not None
            and row["title"] is None
            and self._message_count(scope, row) == 0
            and not any(other["parent_session_id"] == session_id for other in table.values())
        )

    def delete_if_empty(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        # A guarded row, or an empty reset tip that is still its conversation's current target, is no candidate.
        scope = self._backend.scope
        if self._guarded(scope, [session_id]) or self._owned_lineage_ids(scope, str(session_id)) != [session_id]:
            return ()
        if not self.is_empty_candidate(session_id):
            return ()
        state = authority_state(self._backend, scope)
        with state.transaction() as staged:
            self._require_ledger_idle(staged.authority, [str(session_id)])
            self._retire_staged(staged.scope, staged.authority, [str(session_id)], media_root)
        return (session_id,)

    def _is_empty_ended(self, scope: ProfileScope, row: Mapping[str, Any]) -> bool:
        """The empty-sweep predicate: ended, unarchived, no stored rows."""
        return row["ended_at"] is not None and not row["archived"] and self._message_count(scope, row) == 0

    def count_empty(self) -> int:
        scope = self._backend.scope
        return sum(1 for row in self._table(scope).values() if self._is_empty_ended(scope, row))

    def delete_empty(self, *, media_root: str) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        state = authority_state(self._backend, scope)
        with state.transaction() as staged:
            # Skip-busy polarity: one busy row must not abort the sweep; it survives untouched.
            doomed = sorted(
                sid
                for sid, row in staged.scope.get("sessions", {}).items()
                if self._is_empty_ended(scope, row) and self._ledger_idle(staged.authority, sid)
            )
            self._retire_staged(staged.scope, staged.authority, doomed, media_root)
            for row in staged.scope.get("sessions", {}).values():
                if row["parent_session_id"] in doomed:
                    row["parent_session_id"] = None
        return tuple(SessionId(sid) for sid in doomed)

    def discard_unadmitted(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        authority = authority_state(self._backend, scope).snapshot
        if any(row.get("target_session_id") == str(session_id) for row in authority["admissions"].values()) or any(
            row.get("session_id") == str(session_id) for row in authority["workers"].values()
        ):
            # A real session: the fenced delete path (refusals and retirement are delete's), reporting what it removed.
            return self.delete(session_id, media_root=media_root)
        with self._flags_lock:
            existed = self._table(scope).pop(str(session_id), None) is not None
            self._transcripts(scope).pop(str(session_id), None)
            # Imported transcripts live in the maintenance reference's namespace; every message row of the id goes.
            self._backend.scope_store(scope).get("transcripts", {}).pop(str(session_id), None)
        return (SessionId(str(session_id)),) if existed else ()

    # ── backfills ─────────────────────────────────────────────────────────

    def backfill_repo_roots(self, cwd_to_root: Mapping[str, str]) -> None:
        scope = self._backend.scope
        pairs = {cwd: root for cwd, root in cwd_to_root.items() if cwd and root}
        for row in self._table(scope).values():
            root = pairs.get(row["cwd"]) if row["cwd"] is not None else None
            if root is not None and not row["git_repo_root"]:
                row["git_repo_root"] = root

    def backfill_session_owners(self, *, stamp: Optional[str] = None) -> int:
        scope = self._backend.scope
        stamp = ((self._backend.owner_profile if stamp is None else stamp) or "").strip()
        unowned = [row for row in self._table(scope).values() if not (row.get("profile_name") or "").strip()]
        for row in unowned if stamp else ():
            row["profile_name"] = stamp
        return len(unowned) if stamp else 0

    def backfill_acp_session_cwds(self) -> int:
        scope = self._backend.scope
        changed = 0
        for row in self._table(scope).values():
            recorded = row["model_config"].get("cwd")
            if row["source"] == "acp" and not row["cwd"] and recorded not in (None, ""):
                row["cwd"] = recorded
                changed += 1
        return changed

    def retag_worker_sessions(self, workspaces_root: str) -> int:
        scope = self._backend.scope
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate_key = RETAG_GATE_PREFIX + prefix
        meta = state_meta_table(self._backend, scope)
        if meta.get(gate_key) == "1":
            return 0
        # The store matches the descendant arms with LIKE, which folds ASCII case; ``\`` separators count too.
        heads = (ascii_lower(prefix + "/"), ascii_lower(prefix + "\\"))
        retagged = 0
        for row in self._table(scope).values():
            cwd = row["cwd"]
            if row["source"] == RETAG_SOURCE_FROM and cwd is not None and (cwd == prefix or ascii_lower(cwd).startswith(heads)):
                row["source"] = RETAG_SOURCE_TO
                retagged += 1
        meta[gate_key] = "1"
        return retagged


def _delegate_of(row: Mapping[str, Any], frontier: set[str]) -> bool:
    """Whether ``row`` is a delegate child of a frontier row: its ``_delegate_from`` names one, or it carries any delegate marker under one."""
    marker = row["model_config"].get("_delegate_from")
    if marker is None:
        return False
    return (isinstance(marker, str) and marker in frontier) or row["parent_session_id"] in frontier


def _continued_parent(table: Mapping[str, Mapping[str, Any]], row: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The parent ``row`` continues over a compression edge (parent ended by compression; the child carries no branch, delegate or reset marker and is no tool row), else ``None``."""
    if row is None or not row["parent_session_id"]:
        return None
    parent = table.get(row["parent_session_id"])
    config = row["model_config"]
    if parent is None or parent["end_reason"] != END_REASON_COMPRESSION or row["source"] == "tool":
        return None
    if any(config.get(marker) is not None for marker in ("_branched_from", "_delegate_from", "_reset_from")):
        return None
    return str(parent["session_id"])
