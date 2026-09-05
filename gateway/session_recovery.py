"""SessionStore durable-row recovery: session-key generation, legacy Slack key migration, rebuilding
a routing entry from state.db, and the SQLite side of routing transitions (promote/reopen/create/
peer). Mixin split out of ``gateway/session.py``; bound onto ``SessionStore`` via the MRO."""

from __future__ import annotations

import logging
import json
import threading
from dataclasses import replace
from datetime import datetime
from gateway.config import Platform
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


def _origin_json(source) -> Optional[str]:
    """``source.to_dict()`` as JSON, or None when absent/unserializable."""
    if source is None:
        return None
    try:
        return json.dumps(source.to_dict())
    except Exception:
        return None


class SessionRecoveryMixin:
    """SessionStore durable-row recovery and the SQLite side of routing transitions."""

    def _resolve_profile_for_key(self, source: Optional[SessionSource] = None) -> Optional[str]:
        """Profile namespace for session keys: None when multiplexing is off (legacy
        ``agent:main``), else ``source.profile`` or the active profile."""
        if not getattr(self.config, "multiplex_profiles", False):
            return None
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
        namespace = parts[1] or "main"
        return "default" if namespace == "main" else namespace

    @staticmethod
    def _active_profile_name() -> str:
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    def _recovered_row_allowed_for_active_profile(
        self, *, requested_session_key: str, recovered: Dict[str, Any]
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
        recovered: Dict[str, Any], source: SessionSource
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
        self, *, row: Dict[str, Any], session_key: str, source: SessionSource, now: datetime,
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
        return SessionEntry(
            session_key=session_key, session_id=str(row["id"]), created_at=created_at,
            updated_at=updated_at, origin=source, display_name=source.chat_name,
            platform=source.platform, chat_type=source.chat_type,
            reset_had_activity=bool(had_activity))

    def _find_gateway_session_row(
        self, *, session_key: str, source: SessionSource, allow_peer_fallback: bool,
        raise_on_lookup_error: bool = False) -> Optional[Dict[str, Any]]:
        """Query one durable gateway session row. Scoped Slack lookups disable SessionDB's
        platform/chat/user fallback: that tuple has no workspace id and could revive another team's
        session; the caller performs one explicit exact lookup of the old unscoped key instead."""
        db = self._db_for_key(session_key)
        finder = getattr(db, "find_latest_gateway_session_for_peer", None) if db else None
        if not callable(finder):
            return None
        try:
            return finder(
                source=source.platform.value, user_id=source.user_id, session_key=session_key,
                chat_id=source.chat_id if allow_peer_fallback else None,
                chat_type=source.chat_type if allow_peer_fallback else None,
                thread_id=source.thread_id)
        except Exception as exc:
            logger.debug("Gateway session DB recovery failed for %s: %s", session_key, exc)
            if raise_on_lookup_error:
                raise
            return None

    def _recover_session_from_db(
        self, *, session_key: str, source: SessionSource, now: datetime,
        raise_on_lookup_error: bool = False) -> Optional[SessionEntry]:
        """Rebuild a missing session-key mapping from durable state.db data. ``None`` when no row is
        recoverable, or when the recovered session is already overdue under the reset policy — the
        row is then durably promoted to a reset boundary instead of resurrected."""
        entry, migrated_legacy = self._query_recoverable_row(
            # The legacy (pre-workspace) Slack key fallback happens INSIDE _query_recoverable_session
            # (#20583/#66398 design): it performs the exact-key legacy lookup, claims the key once per
            # process, and rewrites the peer row to the scoped key on success.
            session_key=session_key, source=source, now=now,
            raise_on_lookup_error=raise_on_lookup_error)
        if entry is None:
            return None
        reset_reason = self._should_reset(entry, source)
        if reset_reason:
            self._promote_session_reset(
                session_key, entry.session_id, reset_reason,
                log=lambda exc: logger.debug(
                    "Gateway recovered-session reset promotion failed for %s: %s", session_key, exc,
                ),
            )
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
        display_name: Optional[str] = None, include_compression_ancestors: bool = False) -> None:
        """Persist the routing peer for an existing gateway session row."""
        db = self._db_for_key(session_key)
        if not db or not source:
            return
        recorder = getattr(db, "record_gateway_session_peer", None)
        if not callable(recorder):
            return
        peer = dict(
            source=source.platform.value, user_id=source.user_id, session_key=session_key,
            chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id)
        try:
            recorder(
                session_id, **peer, display_name=display_name or source.chat_name,
                origin_json=_origin_json(source),
                include_compression_ancestors=include_compression_ancestors)
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
        create_kwargs: Optional[Dict[str, Any]], origin: Optional[SessionSource],
        display_name: Optional[str], during: str = "") -> None:
        """SQLite side of a routing transition, outside ``_lock``: promote the predecessor row to an
        explicit reset boundary (with the specific reason so state.db is auditable, e.g.
        ``resume_pending_expired`` vs plain ``session_reset``), then INSERT the new row + routing
        peer. Both best-effort: failures are warned and self-healed by the next peer refresh."""
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
    ) -> Dict[str, Any]:
        """kwargs for ``SessionDB.create_session``. Identity (origin_json) and lineage
        (parent/_reset_from) land atomically in the INSERT so a crash right after cannot strand the
        row unroutable."""
        return {
            "session_id": session_id,
            "source": source_value,
            "user_id": origin.user_id if origin else None,
            "session_key": session_key,
            "chat_id": origin.chat_id if origin else None,
            "chat_type": origin.chat_type if origin else None,
            "thread_id": origin.thread_id if origin else None,
            "profile_name": origin.profile if origin else None,
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
