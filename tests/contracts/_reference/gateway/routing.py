"""In-memory routing repository for the reference backend (test fixture only).

Implements ``persistence.gateway.routing.RoutingRepository`` in memory. Storage inside one scope store:

- ``routing``: routing scope -> {session_key: route}. The verbs store a :class:`StoredRoute`, the entry text with the ``session_id``, ``session_key`` and ``origin`` it names: the bindings, worker compaction, profile identity and sessions delete references read and rewrite those attributes, and a rewritten attribute is written back into the text when it is read. Every writer stores the gateway's raw entry JSON text; malformed text refuses with ``PermanentStoreError`` wherever a verb reads a field from it.
- Route fences are ``state_meta`` rows ``gateway.retired_route.v1.<namespace>:<id>`` holding JSON values; this store's own retirement markers are the authority state's ``retired`` table, which the sessions reference's deletes write.
- ``topic_tables`` is set once ``prepare_topic_storage`` ran; ``topic_mode`` maps (profile, chat_id) to the mode row and ``topic_bindings`` maps (profile, chat_id, thread_id) to a ``TopicBinding``. A binding whose session row is gone is dropped by the next topic verb.
- The peer, origin, orphan and never-active verbs read and write the sessions reference's rows and read the ``SessionRecord`` it builds for the counters.

Reference limits: no pre-v3 topic tables exist here, so the read heal never runs; ``record_session_peer(include_compression_ancestors=True)`` does not hand local conversations off to the messaging lane; ``origin_json`` is kept parsed on the session row.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Optional

from persistence import JsonValue, PermanentStoreError, SessionArtifacts, SessionId, SessionRowMissing
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.conversation.sessions import RECOVERABLE_END_REASONS, SessionLineageKind, SessionRecord
from persistence.gateway.routing import (
    END_REASON_SUPERSEDED_BY_REPAIR,
    ORPHAN_ADOPTION_MAX_GAP_S,
    PEER_RESET_END_REASONS,
    TOPIC_SCHEMA_META_KEY,
    NeverActivePrune,
    NeverActiveSession,
    OrphanReport,
    RecoveredPeerSession,
    TopicAlreadyLinked,
    TopicBinding,
    normalize_topic_profile_name,
    route_namespace,
)

from .._authority import authority_state
from ..backend import InMemoryBackend, register_repository
from ..conversation.sessions import blank_session_row
from ..housekeeping.state_meta import state_meta_table

ROUTE_FENCE_PREFIX = "gateway.retired_route.v1."
_PEER_COLUMNS = ("user_id", "chat_id", "chat_type", "thread_id")


def _now() -> datetime:
    return datetime.now(UTC)


def _blank(value: Any) -> str:
    return "" if value is None else str(value)


def _ascii_lower(value: str) -> str:
    # SQLite's LOWER() folds ASCII letters only.
    return "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in value)


def _parse(text: str) -> Any:
    try:
        return json.loads(text or "{}")
    except (ValueError, TypeError, RecursionError):
        return None


def _session_id_text(entry_json: str) -> Optional[str]:
    """The entry's ``session_id`` as text; malformed JSON refuses with ``PermanentStoreError('malformed JSON')``."""
    try:
        parsed = json.loads(entry_json)
    except ValueError as exc:
        raise PermanentStoreError("malformed JSON") from exc
    value = parsed.get("session_id") if isinstance(parsed, dict) else None
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value) if isinstance(value, (str, int, float)) else None


@dataclasses.dataclass(frozen=True, slots=True)
class StoredRoute:
    """One routing row: the entry text and the attributes sibling references read and rewrite."""

    entry_json: str
    session_key: Optional[str] = None
    session_id: Optional[JsonValue] = None
    origin: Optional[Mapping[str, JsonValue]] = None

    @classmethod
    def of(cls, key: str, entry_json: str) -> StoredRoute:
        parsed = _parse(entry_json)
        if not isinstance(parsed, dict):
            return cls(entry_json)
        origin = parsed.get("origin")
        return cls(entry_json, parsed.get("session_key") or key, parsed.get("session_id"), origin if isinstance(origin, dict) else None)

    def text(self, key: str) -> str:
        """The stored text, with any attribute a sibling rewrote written back into it."""
        fresh = StoredRoute.of(key, self.entry_json)
        changed = {name: getattr(self, name) for name in ("session_key", "session_id", "origin") if getattr(self, name) != getattr(fresh, name)}
        if not changed:
            return self.entry_json
        return json.dumps({**_parse(self.entry_json), **changed})


def _entry_text(key: str, route: StoredRoute) -> str:
    return route.text(key)


def _route_sid(route: StoredRoute) -> Optional[str]:
    """The session id a route names, when it is a string."""
    return route.session_id if isinstance(route.session_id, str) else None


def _epoch(value: Optional[datetime]) -> Optional[float]:
    return None if value is None else value.timestamp()


def _fence_expired(value: str, cutoff: float) -> bool:
    """A sweep fence (``expires``) retired before ``cutoff``; a value that is not JSON never expires."""
    parsed = _parse(value)
    if not isinstance(parsed, dict) or not parsed.get("expires"):
        return False
    retired_at = parsed.get("retired_at")
    return isinstance(retired_at, (int, float)) and retired_at < cutoff


def _stamp_peer(table: dict[str, dict[str, Any]], targets: Iterable[str], identity: Mapping[str, Any], display_name: Optional[str], origin_json: Optional[str], transport_profile: Optional[str]) -> None:
    """Write the routing identity on every target row; the display name, origin and transport profile only when given."""
    facets = {"display_name": display_name, "transport_profile": transport_profile}
    if origin_json is not None:
        facets["origin"] = json.loads(origin_json)
    for target in targets:
        table[target].update(identity)
        table[target].update({name: value for name, value in facets.items() if value is not None})


class InMemoryRoutingRepository:
    """Reference implementation; see ``persistence/gateway/routing.py`` for the contract text."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    # ── storage plumbing ──────────────────────────────────────────────────

    def _store(self) -> dict[str, Any]:
        return self._backend.scope_store(self._backend.scope)

    def _index(self) -> dict[str, dict[str, StoredRoute]]:
        index: dict[str, dict[str, StoredRoute]] = self._store().setdefault("routing", {})
        return index

    def _sessions(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self._store().setdefault("sessions", {})
        return table

    def _record(self, row: Mapping[str, Any]) -> SessionRecord:
        record: SessionRecord = self._backend.sessions._record(self._backend.scope, row)  # type: ignore[attr-defined]
        return record

    def _message_rows(self, row: Mapping[str, Any]) -> int:
        count: int = self._backend.sessions._message_count(self._backend.scope, row)  # type: ignore[attr-defined]
        return count

    def _last_active(self, row: Mapping[str, Any]) -> Optional[float]:
        stamp: Optional[float] = self._backend.sessions._last_active(self._backend.scope, row)  # type: ignore[attr-defined]
        return stamp

    # ── routing index ─────────────────────────────────────────────────────

    def _route_retired(self, session_key: str, entry_json: str, now: float) -> bool:
        """Whether the route is retired: a launch (``main``) route whose session this store deleted, or an unexpired route fence, which on a launch route counts only while the session row is gone."""
        session_id = _session_id_text(entry_json)
        if session_id is None:
            return False
        namespace = route_namespace(session_key)
        # This store's own markers are written by its deletes, so they fence only a launch (``main``) route.
        if namespace == "main" and session_id in authority_state(self._backend, self._backend.scope).snapshot["retired"]:
            return True
        fence = state_meta_table(self._backend, self._backend.scope).get(f"{ROUTE_FENCE_PREFIX}{namespace}:{session_id}")
        if fence is None or _fence_expired(fence, now - RETRY_HORIZON_S):
            return False
        return namespace != "main" or session_id not in self._sessions()

    def route_retired(self, session_key: str, entry_json: str) -> bool:
        return self._route_retired(session_key, entry_json, time.time())

    def upsert_routing_entry(self, session_key: str, entry_json: str, *, routing_scope: str = "") -> None:
        if not session_key or not entry_json:
            return
        if self._route_retired(session_key, entry_json, time.time()):
            return
        self._index().setdefault(routing_scope, {})[session_key] = StoredRoute.of(session_key, entry_json)

    def replace_routing_entries(self, entries: Mapping[str, str], *, routing_scope: str = "") -> set[str]:
        now = time.time()
        rows = [(key, text) for key, text in entries.items() if key and text]
        skipped = {key for key, text in rows if self._route_retired(key, text, now)}
        self._index()[routing_scope] = {key: StoredRoute.of(key, text) for key, text in rows if key not in skipped}
        return skipped

    def load_routing_entries(self, *, routing_scope: str = "") -> dict[str, str]:
        return {key: _entry_text(key, route) for key, route in self._index().get(routing_scope, {}).items()}

    def routing_entry_for_session(self, session_id: str) -> Mapping[str, JsonValue] | None:
        for entries in self._index().values():
            for key, route in entries.items():
                entry = _parse(_entry_text(key, route))
                if isinstance(entry, dict) and entry.get("session_id") == session_id:
                    return entry
        return None

    def _routes_naming(self, session_ids: set[str]) -> list[tuple[str, str, str]]:
        return [
            (routing_scope, key, str(route.session_id))
            for routing_scope, entries in self._index().items()
            for key, route in entries.items()
            if _route_sid(route) in session_ids
        ]

    def _mark_fence(self, meta: dict[str, str], key: str, *, expires: bool) -> None:
        """Write a route fence: a sweep's fence never shortens a permanent one."""
        value: dict[str, Any] = {"retired_at": time.time()}
        if expires:
            existing = meta.get(key)
            parsed = _parse(existing) if existing is not None else None
            if existing is None or (isinstance(parsed, dict) and parsed.get("expires")):
                meta[key] = json.dumps({**value, "expires": True})
        else:
            meta[key] = json.dumps(value)

    def fence_retired_routes(self, session_ids: Iterable[str], *, namespace: str, expiring: Iterable[str] = ()) -> None:
        ids = sorted({str(sid) for sid in session_ids if sid})
        if not ids:
            return
        meta = state_meta_table(self._backend, self._backend.scope)
        cutoff = time.time() - RETRY_HORIZON_S
        for key in [key for key, value in meta.items() if key.startswith(ROUTE_FENCE_PREFIX) and _fence_expired(value, cutoff)]:
            del meta[key]
        expiring = set(expiring)
        sessions = self._sessions()
        fenced: dict[str, set[str]] = {}
        for sid in ids:
            names = {namespace} if sid in sessions else {namespace, "main"}
            for name in names:
                self._mark_fence(meta, f"{ROUTE_FENCE_PREFIX}{name}:{sid}", expires=sid in expiring)
            fenced[sid] = names
        for entries in self._index().values():
            for key in [key for key, route in entries.items() if route_namespace(key) in fenced.get(_route_sid(route) or "", ())]:
                del entries[key]

    # ── routing peers ─────────────────────────────────────────────────────

    def _compression_lineage(self, table: dict[str, dict[str, Any]], session_id: str) -> list[str]:
        """The row and every parent that ended ``compression`` above a child with no branch, delegate or reset marker and not a ``tool`` row."""
        ids = [session_id]
        current = table[session_id]
        while True:
            parent_id = current["parent_session_id"]
            parent = table.get(parent_id) if parent_id else None
            config = current["model_config"]
            if (
                parent is None
                or parent["end_reason"] != "compression"
                or any(config.get(marker) is not None for marker in ("_branched_from", "_delegate_from", "_reset_from"))
                or (current["source"] or "") == "tool"
                or parent_id in ids
            ):
                return ids
            ids.append(parent_id)
            current = parent

    def record_session_peer(
        self,
        session_id: str,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
        display_name: str | None = None,
        origin_json: str | None = None,
        include_compression_ancestors: bool = False,
        messaging_lane: bool = False,
        transport_profile: str | None = None,
    ) -> None:
        if not session_id or not session_key:
            return
        identity = {"session_key": session_key, "source": source, "user_id": user_id, "chat_id": chat_id, "chat_type": chat_type, "thread_id": thread_id}
        if include_compression_ancestors:
            from ..authority.bindings_resets import hand_off_conversations, moved_receipts  # the bindings modules import this one

            with authority_state(self._backend, self._backend.scope).transaction() as staged:
                table = staged.scope.setdefault("sessions", {})
                targets = self._compression_lineage(table, session_id) if session_id in table else []
                _stamp_peer(table, targets, identity, display_name, origin_json, transport_profile)
                if messaging_lane:
                    meta = staged.authority["meta"]
                    hand_off_conversations(meta, table, {owner: found for owner, found in moved_receipts(meta, table).items() if found[0] in targets})
            return
        table = self._sessions()
        if session_id not in table:
            row = blank_session_row(session_id, lineage_kind=SessionLineageKind.NONE, model_config=None)
            row.update(source=source, created_source=source, profile_name=self._backend.owner_profile, started_at=_now())
            table[session_id] = row
        _stamp_peer(table, [session_id], identity, display_name, origin_json, transport_profile)

    def _recoverable(self, record: SessionRecord) -> bool:
        return record.ended_at is None or record.end_reason in RECOVERABLE_END_REASONS or record.handoff_state == "completed"

    @staticmethod
    def _recency(record: SessionRecord) -> float:
        return record.started_at if record.last_activity_at is None else record.last_activity_at

    def _has_messages(self, record: SessionRecord, row: Mapping[str, Any]) -> bool:
        return record.message_count > 0 or self._message_rows(row) > 0

    @staticmethod
    def _peer_tuple(record: SessionRecord) -> tuple[str, ...]:
        return tuple(_blank(getattr(record, column)) for column in _PEER_COLUMNS)

    def _owned_by_store(self, record: SessionRecord) -> bool:
        """The peer-tuple profile fence: the row's profile is the store owner or unset; no fence outside the profile tree."""
        owner = self._backend.owner_profile
        return owner is None or (owner if record.profile_name is None else record.profile_name) == owner

    @staticmethod
    def _fenced(candidate: SessionRecord, boundaries: Iterable[SessionRecord]) -> bool:
        recency = InMemoryRoutingRepository._recency(candidate)
        return any(
            boundary.ended_at is not None and boundary.end_reason in PEER_RESET_END_REASONS and boundary.ended_at > recency
            for boundary in boundaries
        )

    def latest_session_for_peer(
        self,
        *,
        source: str,
        user_id: str | None = None,
        session_key: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        thread_id: str | None = None,
    ) -> RecoveredPeerSession | None:
        if not session_key:
            return None
        rows = {sid: (self._record(row), row) for sid, row in self._sessions().items()}
        records = [record for record, _row in rows.values()]
        keyed = [record for record in records if record.session_key == session_key and record.source == source]
        exact = [
            (self._has_messages(record, rows[record.session_id][1]), self._recency(record), record)
            for record in keyed
            if self._recoverable(record) and not self._fenced(record, keyed)
        ]
        if exact:
            has_messages, _recency, winner = max(exact, key=lambda item: (item[0], item[1]))
            return RecoveredPeerSession(session=winner, has_messages=has_messages)
        if chat_id is None or chat_type is None:
            return None
        wanted = tuple(_blank(value) for value in (user_id, chat_id, chat_type, thread_id))
        peers = [record for record in records if record.source == source and self._owned_by_store(record)]
        tuple_rows = [
            record
            for record in peers
            if self._peer_tuple(record) == wanted
            and self._recoverable(record)
            and self._has_messages(record, rows[record.session_id][1])
            and not self._fenced(record, [peer for peer in peers if self._peer_tuple(peer) == self._peer_tuple(record)])
        ]
        if not tuple_rows:
            return None
        return RecoveredPeerSession(session=max(tuple_rows, key=self._recency), has_messages=True)

    def session_for_origin(
        self, *, platform: str, chat_id: str, thread_id: str | None = None, user_id: str | None = None
    ) -> str | None:
        if not platform or chat_id in (None, ""):
            return None
        rows = [
            row
            for row in self._sessions().values()
            if _ascii_lower(_blank(row["source"])) == _ascii_lower(platform)
            and row["session_key"] is not None
            and row["chat_id"] == str(chat_id)
            and row["ended_at"] is None
            and (thread_id is None or _blank(row["thread_id"]) == str(thread_id))
        ]
        if not rows:
            return None
        rows.sort(key=lambda row: row["started_at"], reverse=True)
        if user_id:
            exact = [row for row in rows if _blank(row["user_id"]) == str(user_id)]
            if exact:
                return str(exact[0]["session_id"])
            if len(rows) > 1:
                return None
        elif len({user for user in (_blank(row["user_id"]).strip() for row in rows) if user}) > 1:
            return None
        return str(rows[0]["session_id"])

    # ── orphan repair ─────────────────────────────────────────────────────

    def _profile_fence(self, donor: Mapping[str, Any], orphan_owner: Optional[str]) -> bool:
        """The donor profile fence: no store owner, or the donor's profile (the store owner when unset) is ``orphan_owner``."""
        owner = self._backend.owner_profile
        return owner is None or (owner if donor["profile_name"] is None else donor["profile_name"]) == orphan_owner

    def _is_orphan(self, row: Mapping[str, Any]) -> bool:
        config = row["model_config"]
        return (
            row["session_key"] is None
            and self._message_rows(row) > 0
            and (row["source"] or "") != "tool"
            and config.get("_branched_from") is None
            and config.get("_delegate_from") is None
        )

    def _contiguity_donors(self, table: dict[str, dict[str, Any]], orphan: Mapping[str, Any], last_active: Optional[float], gap: float, orphan_owner: Optional[str]) -> list[dict[str, Any]]:
        if last_active is None:
            return []
        started = _epoch(orphan["started_at"]) or 0
        donors = []
        for donor in table.values():
            if donor["session_key"] is None or donor["session_id"] == orphan["session_id"]:
                continue
            if (donor["source"] or "") != (orphan["source"] or "") or not self._profile_fence(donor, orphan_owner):
                continue
            if _blank(donor["user_id"]) and _blank(orphan["user_id"]) and donor["user_id"] != orphan["user_id"]:
                continue
            donor_active = self._last_active(donor)
            if donor_active is not None and started - gap <= donor_active <= started + gap and donor_active < last_active:
                donors.append((donor_active, donor))
        donors.sort(key=lambda item: item[0], reverse=True)
        return [donor for _active, donor in donors[:2]]

    def _orphan_report(self, table: dict[str, dict[str, Any]], orphan: Mapping[str, Any], gap: float) -> dict[str, Any]:
        orphan_owner = orphan["profile_name"] or self._backend.owner_profile
        last_active = self._last_active(orphan)
        donor: Optional[Mapping[str, Any]] = None
        reason = ""
        if orphan["parent_session_id"]:
            evidence = "lineage"
            parent = table.get(orphan["parent_session_id"])
            if parent is not None and parent["session_key"] is not None and (parent["source"] or "") == (orphan["source"] or "") and self._profile_fence(parent, orphan_owner):
                donor = parent
            else:
                reason = "parent session carries no gateway identity of this source"
        else:
            evidence = "contiguity"
            candidates = self._contiguity_donors(table, orphan, last_active, gap, orphan_owner)
            if not candidates:
                reason = f"no keyed predecessor fell quiet within {gap:.0f}s of this session's start"
            elif len(candidates) > 1:
                reason = "ambiguous: more than one keyed predecessor matches this window"
            else:
                donor = candidates[0]
        return {
            "orphan_id": SessionId(orphan["session_id"]), "source": orphan["source"],
            "message_count": self._message_rows(orphan), "started_at": _epoch(orphan["started_at"]),
            "last_active": last_active, "donor_id": SessionId(donor["session_id"]) if donor else None,
            "session_key": donor["session_key"] if donor else None,
            "evidence": evidence if donor else "", "adoptable": donor is not None, "reason": reason,
        }

    def orphaned_sessions(self, *, max_gap_s: float | None = None) -> tuple[OrphanReport, ...]:
        gap = ORPHAN_ADOPTION_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
        table = self._sessions()
        orphans = sorted((row for row in table.values() if self._is_orphan(row)), key=lambda row: row["started_at"])
        reports = [self._orphan_report(table, orphan, gap) for orphan in orphans]
        claims: dict[str, int] = {}
        for report in reports:
            if report["donor_id"] is not None:
                claims[report["donor_id"]] = claims.get(report["donor_id"], 0) + 1
        for report in reports:
            if report["donor_id"] is not None and claims[report["donor_id"]] > 1:
                report["adoptable"] = False
                report["reason"] = "ambiguous: more than one unkeyed session claims this predecessor"
        return tuple(OrphanReport(**report) for report in reports)

    def adopt_orphan(self, orphan_id: str, donor_id: str) -> bool:
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False
        table = self._sessions()
        owner = self._backend.owner_profile
        donor, orphan = table.get(donor_id), table.get(orphan_id)
        if (
            donor is None or orphan is None or not donor["session_key"] or orphan["session_key"]
            or (donor["source"] or "") != (orphan["source"] or "")
            or (owner is not None and (donor["profile_name"] or owner) != (orphan["profile_name"] or owner))
        ):
            return False
        if orphan["session_key"] is None:
            orphan["session_key"] = donor["session_key"]
            for column in ("chat_id", "chat_type", "thread_id", "user_id", "origin", "display_name"):
                if orphan[column] is None:
                    orphan[column] = copy.deepcopy(donor[column])
            if orphan["parent_session_id"] is None:
                orphan["parent_session_id"] = donor_id
        if donor["ended_at"] is None:
            donor["ended_at"] = _now()
        donor["end_reason"] = END_REASON_SUPERSEDED_BY_REPAIR
        return True

    # ── never-active keyed rows ───────────────────────────────────────────

    def _never_active(self, record: SessionRecord, row: Mapping[str, Any], cutoff: float) -> bool:
        counters = (record.message_count, record.tool_call_count, record.api_call_count, record.input_tokens, record.output_tokens)
        return (
            record.session_key is not None
            and record.ended_at is None
            and record.title is None
            and record.last_activity_at is None
            and not any(counters)
            and not record.pinned
            and not record.archived
            and record.started_at < cutoff
            and self._message_rows(row) == 0
        )

    def never_active_keyed_sessions(self, *, older_than_days: float) -> tuple[NeverActiveSession, ...]:
        if older_than_days < 0:
            raise ValueError(
                f"older_than_days must be >= 0, got {older_than_days!r}: a negative "
                "retention builds a future cutoff that matches every never-active keyed row."
            )
        cutoff = time.time() - float(older_than_days) * 86400.0
        records = [record for row in self._sessions().values() if self._never_active(record := self._record(row), row, cutoff)]
        records.sort(key=lambda record: record.started_at)
        return tuple(
            NeverActiveSession(
                session_id=record.session_id,
                session_key=str(record.session_key),
                source=record.source,
                chat_id=record.chat_id,
                chat_type=record.chat_type,
                user_id=record.user_id,
                started_at=record.started_at,
            )
            for record in records
        )

    def prune_never_active(self, *, older_than_days: float, artifacts: SessionArtifacts | None = None, media_root: str) -> NeverActivePrune:
        candidates = self.never_active_keyed_sessions(older_than_days=older_than_days)
        owned_lineage = self._backend.sessions._owned_lineage_ids  # type: ignore[attr-defined]
        ids = {str(row.session_id) for row in candidates if owned_lineage(self._backend.scope, str(row.session_id)) == [str(row.session_id)]}
        if not ids:
            return NeverActivePrune(sessions_deleted=0, routing_entries_deleted=0, sessions_skipped=0, deleted_ids=(), removed_ids=())
        routed_before = self._routes_naming(ids)
        outcome = self._backend.sessions.delete_many([SessionId(sid) for sid in sorted(ids)], skip_guarded=True, media_root=media_root)  # type: ignore[attr-defined]
        if artifacts is not None:
            artifacts.remove_for(outcome.removed_ids)
        kept = {str(sid) for sid in outcome.skipped_guarded}
        return NeverActivePrune(
            sessions_deleted=outcome.deleted,
            routing_entries_deleted=sum(1 for _scope, _key, sid in routed_before if sid not in kept),
            sessions_skipped=len(kept),
            deleted_ids=tuple(SessionId(sid) for sid in sorted(ids - kept)),
            removed_ids=tuple(SessionId(sid) for sid in sorted(map(str, outcome.removed_ids))),
        )

    def sweep_routing_entries_for_sessions(self, session_ids: Iterable[str]) -> int:
        doomed = self._routes_naming({str(sid) for sid in session_ids})
        for routing_scope, key, _sid in doomed:
            del self._index()[routing_scope][key]
        return len(doomed)

    # ── Telegram DM topics ────────────────────────────────────────────────

    def _topic_tables(self) -> Optional[tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], TopicBinding]]]:
        """The mode and binding tables, or ``None`` before ``prepare_topic_storage``; bindings of deleted sessions are dropped first."""
        store = self._store()
        if not store.get("topic_tables"):
            return None
        bindings: dict[tuple[str, str, str], TopicBinding] = store.setdefault("topic_bindings", {})
        sessions = self._sessions()
        for key in [key for key, binding in bindings.items() if binding.session_id not in sessions]:
            del bindings[key]
        modes: dict[tuple[str, str], dict[str, Any]] = store.setdefault("topic_mode", {})
        return modes, bindings

    def prepare_topic_storage(self) -> None:
        store = self._store()
        store["topic_tables"] = True
        store.setdefault("topic_mode", {})
        store.setdefault("topic_bindings", {})
        state_meta_table(self._backend, self._backend.scope)[TOPIC_SCHEMA_META_KEY] = "3"

    def enable_topic_mode(
        self,
        *,
        chat_id: str,
        user_id: str,
        profile_name: str = "default",
        has_topics_enabled: bool | None = None,
        allows_users_to_create_topics: bool | None = None,
    ) -> None:
        self.prepare_topic_storage()
        modes, _bindings = self._topic_tables()  # type: ignore[misc]
        now = time.time()
        row = modes.setdefault((normalize_topic_profile_name(profile_name), str(chat_id)), {"activated_at": now})
        row.update(
            user_id=str(user_id), enabled=True, updated_at=now, capability_checked_at=now,
            has_topics_enabled=has_topics_enabled, allows_users_to_create_topics=allows_users_to_create_topics,
        )

    def disable_topic_mode(self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True) -> None:
        tables = self._topic_tables()
        if tables is None:
            return
        modes, bindings = tables
        profile, chat = normalize_topic_profile_name(profile_name), str(chat_id)
        if (row := modes.get((profile, chat))) is not None:
            row.update(enabled=False, updated_at=time.time())
        if clear_bindings:
            for key in [key for key in bindings if key[:2] == (profile, chat)]:
                del bindings[key]

    def is_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        tables = self._topic_tables()
        row = tables[0].get((normalize_topic_profile_name(profile_name), str(chat_id))) if tables is not None else None
        return row is not None and row["user_id"] == str(user_id) and bool(row["enabled"])

    def get_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> TopicBinding | None:
        tables = self._topic_tables()
        return tables[1].get((normalize_topic_profile_name(profile_name), str(chat_id), str(thread_id))) if tables is not None else None

    def list_topic_bindings_for_chat(self, *, chat_id: str, profile_name: str = "default") -> tuple[TopicBinding, ...]:
        tables = self._topic_tables()
        if tables is None:
            return ()
        pair = (normalize_topic_profile_name(profile_name), str(chat_id))
        return tuple(sorted((binding for key, binding in tables[1].items() if key[:2] == pair), key=lambda binding: binding.updated_at, reverse=True))

    def get_topic_binding_for_session(self, *, session_id: str) -> TopicBinding | None:
        tables = self._topic_tables()
        found = [binding for binding in tables[1].values() if binding.session_id == str(session_id)] if tables is not None else []
        return found[0] if found else None

    def is_session_linked_to_topic(self, *, session_id: str) -> bool:
        return self.get_topic_binding_for_session(session_id=session_id) is not None

    def delete_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        tables = self._topic_tables()
        if tables is None:
            return 0
        modes, bindings = tables
        profile, chat = normalize_topic_profile_name(profile_name), str(chat_id)
        if bindings.pop((profile, chat, str(thread_id)), None) is None:
            return 0
        if not any(key[:2] == (profile, chat) for key in bindings) and (row := modes.get((profile, chat))) is not None:
            row.update(enabled=False, updated_at=time.time())
        return 1

    def bind_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        user_id: str,
        session_key: str,
        session_id: str,
        managed_mode: str = "auto",
        profile_name: str = "default",
        expected_session_id: str | None = None,
    ) -> bool:
        self.prepare_topic_storage()
        _modes, bindings = self._topic_tables()  # type: ignore[misc]
        now = time.time()
        key = (normalize_topic_profile_name(profile_name), str(chat_id), str(thread_id))
        if expected_session_id is not None:
            current = bindings.get(key)
            if current is None or str(current.session_id) != str(expected_session_id):
                return False
        sid = str(session_id)
        linked = [binding for binding in bindings.values() if binding.session_id == sid]
        if linked and (linked[0].profile_name, linked[0].chat_id, linked[0].thread_id) != key:
            raise TopicAlreadyLinked(sid)
        if sid not in self._sessions():
            raise SessionRowMissing(SessionId(sid))
        fields = {"user_id": str(user_id), "session_key": str(session_key), "session_id": SessionId(sid), "managed_mode": managed_mode, "updated_at": now}
        existing = bindings.get(key)
        bindings[key] = (
            dataclasses.replace(existing, **fields)
            if existing is not None
            else TopicBinding(profile_name=key[0], chat_id=key[1], thread_id=key[2], linked_at=now, **fields)
        )
        return True


register_repository("routing", InMemoryRoutingRepository)
