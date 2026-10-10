"""SessionStore durable-row recovery: session-key generation, legacy Slack key migration, rebuilding
a routing entry from state.db, and the SQLite side of routing transitions (promote/reopen/create/
peer). Mixin split out of ``gateway/session.py``; bound onto ``SessionStore`` via the MRO."""

from __future__ import annotations

import logging
import json
import math
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from gateway.config import Platform
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


@dataclass(frozen=True)
class DelegateRouteVerdict:
    """Read-only finding; only a recoverable verdict carries a proven owner."""

    kind: Literal["not_delegate", "recoverable", "invalid", "unverified"]
    owner_id: Optional[str] = None
    owner_end_state: Optional[tuple[Optional[float], Optional[str]]] = None


def _origin_json(source) -> Optional[str]:
    """``source.to_dict()`` as JSON, or None when absent/unserializable."""
    if source is None:
        return None
    try:
        return json.dumps(source.to_dict())
    except Exception:
        return None


def _is_delegate_execution_row(row: dict[str, Any]) -> bool:
    """Child execution provenance survives the legacy mutable-source gateway stamp."""
    if row.get("created_source") in ("subagent", "delegate") or row.get("source") in ("subagent", "delegate"):
        return True
    config = row.get("model_config")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (TypeError, ValueError):
            return False
    return isinstance(config, dict) and bool(config.get("_delegate_from"))


def _same_delegate_route_peer(store, row: dict[str, Any], entry, source, session_key: str) -> bool:
    """Match the route's peer, respecting whether its key isolates the participant."""
    if row.get("session_key") != session_key or row.get("source") != source.platform.value:
        return False
    if row.get("chat_id") != source.chat_id:
        return False
    if (row.get("chat_type"), row.get("thread_id")) != (source.chat_type, source.thread_id):
        # Discord keys normalize a channel message that will be threaded to its eventual
        # thread id. Historical rows retain the original group/prospective shape. Verify
        # that exact prospective id before accepting the continuation as the same peer.
        if not (source.platform == Platform.DISCORD and row.get("chat_type") == "group" and
                row.get("thread_id") is None and source.chat_type == "thread" and
                source.thread_id):
            return False
        try:
            origin = json.loads(row.get("origin_json") or "")
        except (TypeError, ValueError):
            return False
        if not isinstance(origin, dict) or origin.get("prospective_thread_id") != source.thread_id:
            return False
    # The key itself carries the participant only when build_session_key isolates it.
    # Discord threads are shared by default; their next human sender may differ from
    # the user_id last stamped onto the durable row.
    thread = source.thread_id or (
        source.prospective_thread_id if source.chat_type != "dm" else None
    )
    user_isolated = (
        not source.chat_id if source.chat_type == "dm" else
        bool(getattr(store.config, "group_sessions_per_user", True)) and
        (not thread or bool(getattr(store.config, "thread_sessions_per_user", False)))
    )
    if user_isolated and row.get("user_id") != source.user_id:
        return False
    if not store._recovered_row_matches_source_scope(row, source):
        return False
    transport = entry.transport_profile
    return not (transport and row.get("transport_profile") and
                row["transport_profile"] != transport)


def _delegate_owner_from_route_rows(
    rows_by_id: dict[str, dict[str, Any]], child_id: str, current: dict[str, Any], same_peer,
) -> DelegateRouteVerdict:
    """Walk internal ancestors, then fence later owners and explicit user boundaries."""
    try:
        child_started = float(current["started_at"])
    except (TypeError, ValueError, KeyError):
        return DelegateRouteVerdict("invalid")
    chain_ids = {child_id}
    parent_id = current.get("parent_session_id")
    owner = None
    for _ in range(255):
        if not parent_id or parent_id in chain_ids:
            return DelegateRouteVerdict("invalid")
        parent = rows_by_id.get(str(parent_id))
        if parent is None or parent.get("_ancestry_depth") is None:
            return DelegateRouteVerdict("invalid")
        chain_ids.add(str(parent_id))
        if not _is_delegate_execution_row(parent):
            owner = parent
            break
        parent_id = parent.get("parent_session_id")
    if owner is None or not same_peer(owner):
        return DelegateRouteVerdict("invalid")

    try:
        owner_started = float(owner["started_at"])
        if owner_started > child_started:
            return DelegateRouteVerdict("invalid")
        # The original owner's session_switch is the hijack's own stamp. A switch before
        # this child existed is a genuine earlier boundary, not evidence for restoration.
        if owner.get("end_reason") == "session_switch" and (
            owner.get("ended_at") is None or float(owner["ended_at"]) < child_started
        ):
            return DelegateRouteVerdict("invalid")
    except (TypeError, ValueError, KeyError):
        return DelegateRouteVerdict("invalid")

    from hermes_state_common import _BOUNDARY_END_REASONS

    for row in rows_by_id.values():
        row_id = str(row["id"])
        in_chain = row_id in chain_ids
        if not in_chain and not same_peer(row):
            continue
        if row_id != owner["id"] and not _is_delegate_execution_row(row):
            try:
                if row.get("ended_at") is None or float(row["started_at"]) > owner_started:
                    return DelegateRouteVerdict("invalid")
            except (TypeError, ValueError, KeyError):
                return DelegateRouteVerdict("invalid")
        if row_id == owner["id"] and row.get("end_reason") == "session_switch":
            continue
        if row.get("end_reason") in _BOUNDARY_END_REASONS:
            if in_chain:
                return DelegateRouteVerdict("invalid")
            try:
                if row.get("ended_at") is None or float(row["ended_at"]) >= child_started:
                    return DelegateRouteVerdict("invalid")
            except (TypeError, ValueError):
                return DelegateRouteVerdict("invalid")
    return DelegateRouteVerdict(
        "recoverable", str(owner["id"]), (owner.get("ended_at"), owner.get("end_reason")),
    )


class SessionRecoveryMixin:
    """SessionStore durable-row recovery and the SQLite side of routing transitions."""

    def _reconcile_poisoned_delegate_route(
        self, session_key: str, observed: SessionEntry, source: SessionSource,
        *, quarantine_invalid: bool = True,
    ) -> Optional[SessionEntry]:
        """Restore a proven chat owner before an inbound turn can wait on a delegate lease.

        Old gateway builds could persist a child as this key's route and stamp its peer row.
        Ordinary stale-session recovery misses it because the child remains live. The rare
        repair holds the route lock while rechecking the DB proof and publishing the owner;
        it never ends or acquires the still-running child's execution session.
        """
        from gateway.session import _now, is_internal_subagent_row, transport_profile_of

        db = self._db_for_key(session_key)
        if db is None:
            fallback_entry = (self._routing_fallback_baseline or {}).get(session_key)
            if self._routing_db_loaded or (
                fallback_entry is not None and fallback_entry["session_id"] == observed.session_id
            ):
                raise RuntimeError(
                    f"Cannot verify session provenance for route {session_key}; retry when state.db is available"
                )
            return observed
        try:
            routed_row = db.get_session(observed.session_id)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot verify session provenance for route {session_key}; retry the inbound turn"
            ) from exc
        if not is_internal_subagent_row(routed_row):
            return observed

        with self._lock:
            current = self._entry_locked(session_key)
            if current is not observed:
                return current
            verdict = self._poisoned_delegate_route_verdict(
                session_key=session_key, entry=current, source=source, db=db,
            )
            if verdict.kind == "unverified" or not self._routing_db_loaded:
                raise RuntimeError(
                    f"Cannot verify session provenance for route {session_key}; retry the inbound turn"
                )
            if verdict.kind == "not_delegate":
                return current
            if verdict.kind == "recoverable" and verdict.owner_id:
                try:
                    owner_row = db.get_session(verdict.owner_id)
                    if owner_row is None or is_internal_subagent_row(owner_row):
                        raise ValueError("verified owner row disappeared or became a delegate")
                    if verdict.owner_end_state is None or db.reopen_session(
                        verdict.owner_id, expected_end_state=verdict.owner_end_state,
                    ) is not True:
                        raise ValueError("verified owner lifecycle changed during recovery")
                except Exception as exc:
                    logger.warning(
                        "Could not reopen verified owner %s for poisoned route %s",
                        verdict.owner_id, session_key, exc_info=True,
                    )
                    raise RuntimeError(
                        f"Cannot reopen verified owner for route {session_key}; retry the inbound turn"
                    ) from exc
                else:
                    try:
                        created_at = datetime.fromtimestamp(float(owner_row["started_at"]))
                    except (KeyError, TypeError, ValueError, OSError):
                        created_at = current.created_at
                    repaired = replace(
                        current, session_id=verdict.owner_id, created_at=created_at,
                        updated_at=_now(), origin=source, platform=source.platform,
                        chat_type=source.chat_type, transport_profile=transport_profile_of(source),
                        active_turn_token=None, active_turn_started_at=None,
                        suspended=False, resume_pending=False, resume_reason=None,
                        last_resume_marked_at=None,
                    )
                    data = self._entries_as_dicts()
                    data[session_key] = repaired.to_dict()
                    self._persist_routing_data(
                        data, self._next_routing_generation_locked(), require_primary=True,
                    )
                    self._entries[session_key] = repaired
                    clear_child_peer = getattr(db, "clear_poisoned_delegate_gateway_peer", None)
                    if callable(clear_child_peer):
                        try:
                            clear_child_peer(current.session_id, session_key)
                        except Exception:
                            logger.warning(
                                "Could not clear legacy gateway peer on delegate %s",
                                current.session_id, exc_info=True,
                            )
                    logger.warning(
                        "Restored poisoned gateway route %s from delegate %s to owner %s",
                        session_key, current.session_id, verdict.owner_id,
                    )
                    return repaired

            # The row is definitely an internal execution session, but the previous owner
            # cannot be proven. Quarantine this key; the normal exact-peer recovery below may
            # find a newer human session, otherwise it creates a fresh one. Do not end the child.
            # Completion preflight may restore a proven owner, but cannot remove a route on
            # behalf of a stale event. Ordinary inbound owns quarantine and fresh creation.
            if not quarantine_invalid:
                return None
            data = self._entries_as_dicts()
            data.pop(session_key, None)
            self._persist_routing_data(
                data, self._next_routing_generation_locked(), require_primary=True,
            )
            self._entries.pop(session_key, None)
            logger.warning(
                "Quarantined poisoned gateway route %s from delegate %s (verdict=%s)",
                session_key, current.session_id, verdict.kind,
            )
            return None

    def _poisoned_delegate_route_verdict(
        self, *, session_key: str, entry: SessionEntry, source: SessionSource, db,
    ) -> DelegateRouteVerdict:
        """Find the original gateway owner of a route switched onto a delegate.

        A historical ``switch_session`` can copy the gateway peer onto an internal delegate row,
        changing its mutable ``source`` to the platform and ending the former owner with
        ``session_switch``. Read one DB snapshot containing the routed row, its parent chain and
        every row under this key. Only an immutable delegate birth/marker plus a same-key,
        same-peer non-delegate ancestor authorizes recovery. A later real owner or explicit
        boundary defeats that proof. The caller must CAS *entry* before changing the route.
        """
        if not session_key or not entry or entry.session_key != session_key or not entry.session_id:
            return DelegateRouteVerdict("invalid")
        try:
            if self._generate_session_key(source) != session_key:
                return DelegateRouteVerdict("invalid")
        except Exception:
            logger.debug("Delegate route key verification failed for %s", session_key, exc_info=True)
            return DelegateRouteVerdict("invalid")
        reader = getattr(db, "_read_all", None)
        if not callable(reader):
            return DelegateRouteVerdict("unverified")
        try:
            rows = reader(
                """WITH RECURSIVE ancestors(id, parent_session_id, depth) AS (
                       SELECT id, parent_session_id, 0 FROM sessions WHERE id = ?
                       UNION ALL
                       SELECT s.id, s.parent_session_id, a.depth + 1
                       FROM sessions s JOIN ancestors a ON s.id = a.parent_session_id
                       WHERE a.depth < 255
                   )
                   SELECT s.*,
                          (SELECT MIN(a.depth) FROM ancestors a WHERE a.id = s.id)
                              AS _ancestry_depth
                   FROM sessions s
                   WHERE s.id IN (SELECT id FROM ancestors) OR s.session_key = ?""",
                (entry.session_id, session_key),
            )
        except Exception as exc:
            logger.debug("Delegate route verdict could not read %s: %s", session_key, exc, exc_info=True)
            return DelegateRouteVerdict("unverified")

        rows_by_id = {str(row["id"]): dict(row) for row in rows}
        current = rows_by_id.get(entry.session_id)
        if current is None:
            return DelegateRouteVerdict("unverified")

        def same_peer(row: dict[str, Any]) -> bool:
            return _same_delegate_route_peer(self, row, entry, source, session_key)

        if not same_peer(current):
            return DelegateRouteVerdict("invalid")
        if not _is_delegate_execution_row(current):
            return DelegateRouteVerdict("not_delegate")
        return _delegate_owner_from_route_rows(rows_by_id, entry.session_id, current, same_peer)

    def _resolve_profile_for_key(self, source: Optional[SessionSource] = None) -> Optional[str]:
        """Profile namespace for session keys: None when multiplexing is off (legacy
        ``agent:main``), else the pinned identity's runtime profile, ``source.profile`` or the
        active profile."""
        if not getattr(self.config, "multiplex_profiles", False):
            return None
        from gateway.session_identity import identity_of
        identity = identity_of(source)
        if identity is not None:
            return identity.session_key_profile
        if source is not None and source.profile:
            return source.profile
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return None

    @staticmethod
    def _profile_from_session_key(session_key: Optional[str]) -> Optional[str]:
        """Extract the profile namespace encoded in a gateway session key."""
        if not session_key:
            return None
        parts = str(session_key).split(":")
        if len(parts) < 2 or parts[0] != "agent":
            return None
        from gateway.session import profile_from_session_key_namespace
        return profile_from_session_key_namespace(parts[1] or "main")

    @staticmethod
    def _active_profile_name() -> str:
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    def _recovered_row_allowed_for_active_profile(
        self, *, requested_session_key: str, recovered: dict[str, Any]
    ) -> bool:
        """Prevent a gateway from reviving another profile's row. Single-profile: the row's
        namespace must match the ACTIVE profile. Multiplexed: it must match the requested key's
        namespace (the active profile is meaningless there). Keyless rows stay adoptable.

        Multiplexed: several profiles serve traffic at once, so the active profile is meaningless — the
        requested key carries the profile the turn was routed to, and the recovered row must sit in the same
        ``agent:<ns>:`` namespace (#74285). Rows with no key namespace stay adoptable in both modes
        (legacy/keyless data owned by this store).
        """
        recovered_key = str(recovered.get("session_key") or "")
        if not recovered_key or recovered_key == requested_session_key:
            return True
        recovered_profile = self._profile_from_session_key(recovered_key)
        if recovered_profile is None:
            return True
        if getattr(self.config, "multiplex_profiles", False):
            requested_profile = self._profile_from_session_key(requested_session_key)
            return requested_profile is None or recovered_profile == requested_profile
        return recovered_profile == self._active_profile_name()

    def _generate_session_key(self, source: SessionSource, key_source: Optional[SessionSource] = None) -> str:
        """Session key for *source* (profile from *source*; key from *key_source* if given)."""
        from gateway.session import build_session_key
        return build_session_key(
            key_source if key_source is not None else source,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
            profile=self._resolve_profile_for_key(source))

    def _legacy_slack_session_key(self, source: SessionSource) -> Optional[str]:
        """Pre-workspace Slack key for an explicitly scoped source. Deliberately Slack-only: an
        unscoped Slack session may be claimed by only one workspace (old key cannot tell teams)."""
        if source.platform != Platform.SLACK or not source.scope_id:
            return None
        return self._generate_session_key(source, replace(source, scope_id=None, guild_id=None))

    def _claim_legacy_slack_key(self, legacy_key: Optional[str]) -> bool:
        """Atomically reserve one ambiguous legacy Slack key for migration."""
        if not legacy_key:
            return False
        with self._lazy("_legacy_slack_claim_lock", threading.Lock):
            claimed = self._lazy("_claimed_legacy_slack_keys", set)
            if legacy_key in claimed:
                return False
            claimed.add(legacy_key)
            return True

    @staticmethod
    def _recovered_row_matches_source_scope(
        recovered: dict[str, Any], source: SessionSource
    ) -> bool:
        """Reject recovered rows whose origin belongs to another workspace: a workspace-scoped Slack
        lookup adopts a row only if its origin_json names the same scope_id; rows without a
        parseable origin are rejected (an unattributable transcript is exactly the ambiguity)."""
        if source.platform != Platform.SLACK or source.chat_type == "dm" or not source.scope_id:
            return True
        try:
            origin = json.loads(recovered.get("origin_json") or "")
        except (TypeError, ValueError):
            return False
        if not isinstance(origin, dict):
            return False
        return origin.get("scope_id", origin.get("guild_id")) == source.scope_id

    def _create_entry_from_recovered_row(
        self, *, row: dict[str, Any], session_key: str, source: SessionSource, now: datetime,
    ) -> SessionEntry:
        from gateway.session import SessionEntry

        def _ts(value, default: datetime) -> datetime:
            try:
                return datetime.fromtimestamp(float(value))
            except (TypeError, ValueError, OSError):
                return default

        # An invalid durable timestamp must look old, never freshly active.
        created_at = _ts(row.get("started_at"), datetime.fromtimestamp(0))
        # The finder already returns durable recency; no extra round-trip.
        last_activity = row.get("last_activity_at")
        updated_at = _ts(last_activity, created_at) if last_activity is not None else created_at
        had_activity = row.get("_has_messages")
        if had_activity is None:
            had_activity = bool(row.get("message_count") or 0) or last_activity is not None
        from gateway.session_identity import transport_profile_of
        return SessionEntry(
            session_key=session_key, session_id=str(row["id"]), created_at=created_at,
            updated_at=updated_at, origin=source, display_name=source.chat_name,
            platform=source.platform, chat_type=source.chat_type,
            reset_had_activity=bool(had_activity), transport_profile=transport_profile_of(source))

    def _find_gateway_session_row(
        self, *, session_key: str, source: SessionSource, allow_peer_fallback: bool,
        raise_on_lookup_error: bool = False) -> Optional[dict[str, Any]]:
        """Query one durable gateway session row. Scoped Slack lookups disable SessionDB's
        platform/chat/user fallback: that tuple has no workspace id and could revive another team's
        session; the caller performs one explicit exact lookup of the old unscoped key instead."""
        return self._peer_row(
            self._db_for_key(session_key), source=source.platform.value, session_key=session_key,
            user_id=source.user_id,
            chat_id=source.chat_id if allow_peer_fallback else None,
            chat_type=source.chat_type if allow_peer_fallback else None,
            thread_id=source.thread_id, raise_on_lookup_error=raise_on_lookup_error)

    @staticmethod
    def _peer_row(db, *, source: str, session_key: str, raise_on_lookup_error: bool = False,
                  **peer: Any) -> Optional[dict[str, Any]]:
        """``db.find_latest_gateway_session_for_peer`` guarded for a missing store, a SessionDB
        without the finder, and a failing lookup (debug-logged -> None unless *raise_on_lookup_error*).
        Extra keyword arguments (user_id/chat_id/chat_type/thread_id) pass through to the finder."""
        finder = getattr(db, "find_latest_gateway_session_for_peer", None) if db else None
        if not callable(finder):
            return None
        try:
            return finder(source=source, session_key=session_key, **peer)
        except Exception as exc:
            logger.debug("Gateway session DB recovery failed for %s: %s", session_key, exc)
            if raise_on_lookup_error:
                raise
            return None

    def resolve_session_id_for_key(
        self, session_key: str, *, not_after: Optional[float] = None,
    ) -> Optional[tuple[str, Any]]:
        """Resolve a gateway session key to ``(session_id, db)`` for shutdown-flush recovery.

        The routing map (``peek_session_id``) is authoritative: it names the session the message
        was actually routed to at shutdown. When ``sessions.json`` was pruned, fall back to the
        durable row under the exact key (the peer finder keeps the reset fence: rows ended only by
        recoverable reasons match; explicit boundaries do not). A row started after the flush
        (``started_at > not_after``) cannot be the origin and is never adopted. Never mints a
        session; None means the caller must preserve the flush file. ``db`` is the store owning
        the key, so the append lands in the right profile partition. When that store cannot be
        resolved (``_db_for_key`` fails closed for a profile without a reachable home) the answer
        is None even if the routing map knows the id: appending to the ambient root store would
        split one session identity across two physical stores (#66887/#102157).
        """
        if not session_key:
            return None
        db = self._db_for_key(session_key)
        if db is None:
            return None
        session_id = self.peek_session_id(session_key)
        if session_id:
            return session_id, db
        parts = str(session_key).split(":")
        platform = parts[2] if len(parts) >= 3 and parts[0] == "agent" else None
        if not platform:
            return None
        # No scope/profile fences here, unlike _query_recoverable_row: with no chat tuple the finder
        # runs only the `s.session_key = ?` branch in the store _db_for_key picked for this key, so a
        # hit carries this very key — same profile namespace and (for scoped Slack) same scope_id slot.
        row = self._peer_row(db, source=platform, session_key=session_key)
        if not isinstance(row, dict) or not row.get("id"):
            return None
        started_at = row.get("started_at")
        # ``ts`` is ``int(time.time())`` while ``started_at`` is a REAL, so compare whole seconds:
        # a row minted in the same second as the flush is still a valid origin.
        if not_after is not None and started_at is not None and math.floor(float(started_at)) > int(not_after):
            return None
        return str(row["id"]), db

    def _recover_session_from_db(
        self, *, session_key: str, source: SessionSource, now: datetime,
        raise_on_lookup_error: bool = False) -> Optional[SessionEntry]:
        """Rebuild a missing session-key mapping from a recoverable durable row."""
        entry, migrated_legacy = self._query_recoverable_row(
            # The legacy (pre-workspace) Slack key fallback happens INSIDE _query_recoverable_session
            # (#20583/#66398 design): it performs the exact-key legacy lookup, claims the key once per
            # process, and rewrites the peer row to the scoped key on success.
            session_key=session_key, source=source, now=now,
            raise_on_lookup_error=raise_on_lookup_error)
        if entry is None:
            return None
        self._reopen_session_row(session_key, entry.session_id)
        if migrated_legacy:
            self._record_gateway_session_peer(
                entry.session_id, session_key, source, display_name=entry.display_name)
        return entry

    def _query_recoverable_session(self, *, session_key, source, now):
        """DB-only half of _recover_session_from_db (no lock needed): a SessionEntry or None; the
        caller assigns _entries[key] under lock. The row is NOT reopened here: the caller evaluates
        reset policy first (an agent_close/ws_orphan row may need promotion to a real reset)."""
        entry, migrated_legacy = self._query_recoverable_row(
            session_key=session_key, source=source, now=now)
        if entry is not None and migrated_legacy:
            self._record_gateway_session_peer(
                entry.session_id, session_key, source, display_name=entry.display_name)
        return entry

    def _query_recoverable_row(
        self, *, session_key, source, now, raise_on_lookup_error=False,
    ) -> tuple[Optional[SessionEntry], bool]:
        """Find and gate a recoverable row -> (entry or None, migrated_legacy). The legacy
        (pre-workspace) Slack key fallback lives here: exact-key lookup, claimed once per process;
        ``migrated_legacy`` tells the caller to rewrite the peer row to the scoped key."""
        legacy_key = self._legacy_slack_session_key(source)
        recovered = self._find_gateway_session_row(
            session_key=session_key, source=source, allow_peer_fallback=legacy_key is None,
            raise_on_lookup_error=raise_on_lookup_error)
        migrated_legacy = False
        if not recovered and legacy_key and self._claim_legacy_slack_key(legacy_key):
            recovered = self._find_gateway_session_row(
                session_key=legacy_key, source=source, allow_peer_fallback=False,
                raise_on_lookup_error=raise_on_lookup_error)
            migrated_legacy = bool(recovered)
        if not isinstance(recovered, dict):
            return None, False
        if not self._recovered_row_matches_source_scope(recovered, source):
            return None, False
        if not self._recovered_row_allowed_for_active_profile(
            requested_session_key=session_key, recovered=recovered):
            logger.warning(
                "Gateway session DB recovery ignored %s for %s because the row belongs to a "
                "different profile", recovered.get("session_key"), session_key)
            return None, False
        entry = self._create_entry_from_recovered_row(
            row=recovered, session_key=session_key, source=source, now=now)
        return entry, migrated_legacy

    def _promote_session_reset(self, session_key: str, session_id: str, reason: str, *, log) -> None:
        """End *session_id* with *reason* via ``promote_to_session_reset`` (``end_session`` on old
        SessionDBs). Promote, not plain end: a row already ended with a recoverable accidental
        reason (agent_close / ws_orphan_reap) must be upgraded to the explicit boundary, or
        stale-route recovery resurrects it over the reset. ``log(exc)`` reports failures."""
        try:
            db = self._db_for_key(session_key)
            promote = getattr(db, "promote_to_session_reset", None)
            if callable(promote):
                promote(session_id, reason)
            else:
                db.end_session(session_id, reason)
            # Stop the departed conversation's schedule in its owning profile, even when
            # the in-memory watch still holds a pre-reset session id.
            heartbeat_key = f"heartbeat:{session_id}"
            if db.get_meta(heartbeat_key):
                db.set_meta(heartbeat_key, "")
        except Exception as exc:
            log(exc)

    def _reopen_session_row(self, session_key: str, session_id: str, *, log_prefix: str = "") -> None:
        """Best-effort ``reopen_session``; failures are debug-logged only."""
        try:
            self._db_for_key(session_key).reopen_session(session_id)
        except Exception as exc:
            if log_prefix:
                logger.debug("%s: %s", log_prefix, exc)
            else:
                logger.debug("Gateway session DB reopen failed for %s: %s", session_key, exc)

    def _record_gateway_session_peer(
        self, session_id: str, session_key: str, source: Optional[SessionSource],
        display_name: Optional[str] = None, include_compression_ancestors: bool = False,
        transport_profile: Optional[str] = None) -> None:
        """Persist the routing peer for an existing gateway session row. ``transport_profile`` is the
        entry's persisted receiving-bot profile; when the caller has no entry it is read off the
        source's pinned identity (None = unknown, the column keeps whatever an earlier writer set)."""
        db = self._db_for_key(session_key)
        if not db or not source:
            return
        recorder = getattr(db, "record_gateway_session_peer", None)
        if not callable(recorder):
            return
        from gateway.session_identity import transport_profile_of
        peer = dict(
            source=source.platform.value, user_id=source.user_id, session_key=session_key,
            chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id)
        try:
            recorder(
                session_id, **peer, display_name=display_name or source.chat_name,
                origin_json=_origin_json(source),
                include_compression_ancestors=include_compression_ancestors,
                transport_profile=transport_profile or transport_profile_of(source))
        except TypeError:
            try:  # older SessionDB without display_name/origin_json kwargs
                recorder(session_id, **peer)
            except Exception as exc:
                logger.debug("Gateway session peer record failed for %s: %s", session_key, exc)
        except Exception as exc:
            logger.debug("Gateway session peer record failed for %s: %s", session_key, exc)

    def _adopt_legacy_slack_entry(self, source: SessionSource, session_key: str) -> None:
        """One-time migration of pre-workspace-scope Slack keys: MOVE (not copy) the legacy entry so
        a second workspace with identical Slack ids cannot attach to the same transcript. Adopt when
        the legacy origin names the same workspace; a scope-less DM is claimed once by the first
        workspace; a scope-less channel/group is refused (channel ids collide across workspaces)."""
        legacy_key = self._legacy_slack_session_key(source)
        if not legacy_key:
            return
        migrated: Optional[SessionEntry] = None
        with self._lock:
            self._ensure_loaded_locked()
            legacy_entry = self._entries.get(legacy_key)
            if session_key not in self._entries and legacy_entry is not None:
                origin_scope = getattr(legacy_entry.origin, "scope_id", None)
                if origin_scope is not None:
                    adopt = origin_scope == source.scope_id
                else:
                    adopt = source.chat_type == "dm"
                if adopt and self._claim_legacy_slack_key(legacy_key):
                    migrated = self._entries.pop(legacy_key)
                    migrated.session_key = session_key
                    migrated.origin = source
                    migrated.platform = source.platform
                    migrated.chat_type = source.chat_type
                    self._entries[session_key] = migrated
        if migrated is not None:
            self._save_entries()
            self._record_gateway_session_peer(
                migrated.session_id, session_key, source, display_name=migrated.display_name)

    def _finish_route_transition(
        self, session_key: str, *, end_session_id: Optional[str], end_reason: str,
        create_kwargs: Optional[dict[str, Any]], origin: Optional[SessionSource],
        display_name: Optional[str], during: str = "") -> None:
        """SQLite side of a routing transition, outside ``_lock``: promote the predecessor row to an
        explicit reset boundary (with the specific reason so state.db is auditable, e.g.
        ``suspended`` vs plain ``session_reset``), then INSERT the new row + routing
        peer. Both best-effort: failures are warned and self-healed by the next peer refresh."""
        if end_session_id:
            from hermes_cli.observability.relay_shared_metrics import close_session_run

            close_session_run(end_session_id)
        if self._db_for_key(session_key) and end_session_id:
            self._promote_session_reset(
                session_key, end_session_id, end_reason,
                log=lambda e: logger.warning(
                    "Failed to end predecessor session row %s for %s%s: %s — the old row remains "
                    "open and may win restart recovery until the next successful peer refresh",
                    end_session_id, session_key, during, e),
            )
        if self._db_for_key(session_key) and create_kwargs:
            self._create_session_row(
                session_key, create_kwargs, origin, display_name,
                log=lambda e: logger.warning(
                    "Failed to create session row %s for %s%s: %s — deferring to the "
                    "self-healing peer refresh on the next turn",
                    create_kwargs.get("session_id"), session_key, during, e),
            )

    @staticmethod
    def _session_create_kwargs(
        *, session_id, session_key, origin, source_value, display_name, parent_session_id,
    ) -> dict[str, Any]:
        """kwargs for ``SessionDB.create_session``. Identity (origin_json) and lineage
        (parent/_reset_from) land atomically in the INSERT so a crash right after cannot strand the
        row unroutable."""
        from gateway.session_identity import transport_profile_of
        return {
            "session_id": session_id,
            "source": source_value,
            "user_id": origin.user_id if origin else None,
            "session_key": session_key,
            "chat_id": origin.chat_id if origin else None,
            "chat_type": origin.chat_type if origin else None,
            "thread_id": origin.thread_id if origin else None,
            "profile_name": origin.profile if origin else None,
            "transport_profile": transport_profile_of(origin),
            "origin_json": _origin_json(origin),
            "display_name": display_name,
            "parent_session_id": parent_session_id,
            "model_config": {"_reset_from": parent_session_id} if parent_session_id else None,
        }

    def _create_session_row(self, session_key, db_create_kwargs, origin, display_name, *, log) -> None:
        """INSERT a session row and record its routing peer; ``log(exc)`` on failure. A failed
        create is a routing hazard (visible warning), but the row is self-healed with full identity
        by the next per-turn peer refresh."""
        try:
            self._db_for_key(session_key).create_session(**db_create_kwargs)
            self._record_gateway_session_peer(
                db_create_kwargs["session_id"], session_key, origin, display_name=display_name)
        except Exception as e:
            log(e)
