"""``SessionRepository`` on SQLite: the session verbs over the session store functions, with their rows and dicts
mapped to the contract's records."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Optional, Sequence

from persistence import AuthorityRefusal, RetiredSessionError, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant
from persistence.conversation.sessions import (
    END_REASON_SESSION_RESET, ArchiveChange, CanonicalTitleLocked, CronRunEntry, CronRunIdentity, DeclaredScopeIdentity, DeleteManyOutcome,
    ONBOARDING_RUN_KEY, LiveListingFacts, LiveListingRow, NewSession, OnboardingAdvance, PendingAdmissionRef, PickerFacts,
    PickerRoot, PickerSegment, RecentBrowseQuery, RecentBrowseTimeout,
    RecentSessionEntry, ReplaceSessionConfig, RosterSession, RosterSessions, RosterWorker, RuntimeLock,
    SessionCountQuery, SessionCwdUsage, SessionEdge, SessionExists, SessionFacetEntry, SessionLifecycleStatus,
    SessionLineageKind, SessionListEntry, SessionListRow, SessionOrder, SessionQuery, SessionRecencyEntry,
    SessionRecord, SessionWindow, SkillScaffoldedSession, TitleConflict, TitleSource, is_corrupt_model_config,
)
from session_rules import onboarding_flow
from session_rules.legacy_local import same_owner

from .._backend import SQLiteStore, register_repository
from .._holder import StoreHolder
from .._schema import TITLE_SOURCE_USER
from .._schema_ddl import _RESET_CHILD_SQL, _sql_json_extract
from hermes_state_common import _shape_preview
from ._store import stored_reads
from ._store.compression import _compression_tip_on, reopen_if_explicitly_closed
from ._store.core import is_kanban_owned_session, retag_kanban_worker_sessions
from ._store.gateway import list_gateway_sessions
from ._store.local import POLICY_PREFIX
from ._store.local_migration import LEGACY_PREFIX
from ._store.model_config import session_ids_with_model_config_key, update_conversation_model_config_value
from ._store.portability import (
    cron_run_identity, distinct_session_cwds, get_session_rich_row, list_cron_job_runs, list_skill_scaffolded_sessions,
)
from ._store.prompt_refresh import count_sessions_with_stored_prompt, prompt_refresh_epoch, request_prompt_refresh
from ._store.search import search_sessions_by_id
from ._store.sql_common import _sql_session_last_active
from ._store.sessions import (
    INTERNAL_LISTING_SOURCES, _own_profile_name, _parse_model_config,
    backfill_acp_session_cwd, backfill_null_session_profiles, backfill_repo_roots,
    clear_session_activity_labels, count_empty_sessions, create_session, declared_scope_identity, delete_empty_sessions,
    delete_session, delete_session_if_empty, delete_sessions, discard_unadmitted_session, end_session, get_session,
    get_session_delete_targets, get_session_list_row, get_session_model_config_value, list_recent_sessions_bounded,
    list_sessions_rich, patch_session_model_config, promote_to_session_reset, publish_session_git_metadata,
    reopen_session, resolve_session_id, retire_undrained_queue_rows, search_sessions, session_count,
    session_count_by_source, session_count_ge, session_lifecycle_statuses, session_unread, set_session_archived,
    set_session_hidden, set_session_pinned, set_session_read, set_session_yolo, touch_session_activity,
    unarchive_recoverable_session, update_session_cwd, update_session_meta, update_session_model,
    update_session_runtime_lock, update_session_tool_names, update_system_prompt,
)
from ._store.telegram import list_unlinked_telegram_sessions_for_user
from ._store.titles import (
    _resolve_title_conflict, get_next_title_in_lineage, get_session_by_title, get_session_title,
    get_session_title_source, resolve_session_by_title, sanitize_title, set_auto_title, set_session_title,
    set_session_title_source,
)
from ._errors import refusals
from ._store.retirement import create_conflict

__all__ = ["SQLiteSessionRepository", "session_record"]

# The fork markers a child carries in ``model_config``, in the order the first non-blank one names its kind.
_MARKERS = (
    ("_delegate_from", SessionLineageKind.DELEGATE),
    ("_branched_from", SessionLineageKind.BRANCH),
    ("_reset_from", SessionLineageKind.RESET),
)
_MARKER_OF_KIND = {kind: key for key, kind in _MARKERS}
_TITLE_SOURCES = {source.value: source for source in TitleSource}
_TITLE_HELD = re.compile(r"Title '(?P<title>.*)' is already in use by session (?P<holder>.+)", re.DOTALL)
_CANONICAL_TITLE_LOCKED = "This is the bot's canonical Bot Chat"
# The content hash a listing row's ``tool_names`` holds when the row was read without the prompt-store join.
_PROMPT_HASH = re.compile(r"[0-9a-f]{64}")

# A local chat: minted by this owner (``local-*``), or a pre-upgrade chat it adopted under a legacy binding.
_LOCAL_CHAT = "(chat_id LIKE 'local-%' OR EXISTS (SELECT 1 FROM state_meta WHERE key = ? || sessions.chat_id))"

_DESCENDANTS_SQL = f"""
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
                -- Continuation edges only (same predicate as the session list's chain CTE): a subagent run,
                -- a /branch fork, a /new reset child or a tool-owned row is its own conversation, and resuming
                -- INTO one parks the user's chat in a row the sidebar never lists (#115092).
                WHERE {_sql_json_extract('s.model_config', '$._delegate_from')} IS NULL
                  AND {_sql_json_extract('s.model_config', '$._branched_from')} IS NULL
                  AND NOT ({_RESET_CHILD_SQL.format(a='s')})
                  AND COALESCE(s.source, '') != 'tool'
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """


def _stripped(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_value(text: Any) -> Any:
    """A stored JSON column decoded, ``None`` when unset or unreadable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _tool_pin(text: Any) -> Any:
    """The decoded ``tools[]`` pin; ``None`` for an unresolved content hash."""
    return None if isinstance(text, str) and _PROMPT_HASH.fullmatch(text) else _json_value(text)


def _ids(values: Sequence[str]) -> tuple[SessionId, ...]:
    return tuple(SessionId(value) for value in values)


def _removed(values: Sequence[str]) -> tuple[SessionId, ...]:
    """Removed or archived ids, each once (a write retried on a busy store reports its ids again)."""
    return _ids(tuple(dict.fromkeys(values)))


def _listed(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    """A filter sequence as the store's list argument; empty or absent is no filter."""
    return list(values) if values else None


def session_record(row: Mapping[str, Any]) -> SessionRecord:
    """The record of one legacy session dict (``get_session`` and the listing rows; a compact row lacks the prompt
    columns)."""
    config = _parse_model_config(row.get("model_config"))
    kind, origin_id = SessionLineageKind.NONE, None
    for key, marker_kind in _MARKERS:
        if _stripped(config.get(key)):
            kind, origin_id = marker_kind, config[key]
            break
    origin = _json_value(row.get("origin_json"))
    # A stored instant that is not a trusted epoch reads degraded: unset, or the untrusted floor for ``started_at``.
    started = stored_instant(row.get("started_at"), session_id=row["id"], field="started_at")
    ended = stored_instant(row.get("ended_at"), session_id=row["id"], field="ended_at")
    return SessionRecord(
        session_id=SessionId(row["id"]),
        source=row.get("source") or "",
        created_source=row.get("created_source"),
        user_id=row.get("user_id"),
        session_key=row.get("session_key"),
        chat_id=row.get("chat_id"),
        chat_type=row.get("chat_type"),
        thread_id=row.get("thread_id"),
        display_name=row.get("display_name"),
        origin=dict(origin) if isinstance(origin, Mapping) else None,
        expiry_finalized=bool(row.get("expiry_finalized")),
        model=row.get("model"),
        model_config=config,
        system_prompt=row.get("system_prompt"),
        system_prompt_hash=row.get("system_prompt_hash"),
        parent_session_id=row.get("parent_session_id"),
        lineage_kind=kind,
        reset_from=_stripped(config.get("_reset_from")),
        branched_from=_stripped(config.get("_branched_from")),
        started_at=(started or UNTRUSTED_INSTANT).timestamp(),
        ended_at=None if ended is None else ended.timestamp(),
        end_reason=row.get("end_reason"),
        message_count=row.get("message_count", 0),
        tool_call_count=row.get("tool_call_count", 0),
        input_tokens=row.get("input_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
        cache_read_tokens=row.get("cache_read_tokens", 0),
        cache_write_tokens=row.get("cache_write_tokens", 0),
        reasoning_tokens=row.get("reasoning_tokens", 0),
        api_call_count=row.get("api_call_count", 0),
        estimated_cost_usd=row.get("estimated_cost_usd"),
        actual_cost_usd=row.get("actual_cost_usd"),
        billing_provider=row.get("billing_provider"),
        billing_base_url=row.get("billing_base_url"),
        billing_mode=row.get("billing_mode"),
        cost_status=row.get("cost_status"),
        cost_source=row.get("cost_source"),
        pricing_version=row.get("pricing_version"),
        cwd=row.get("cwd"),
        git_branch=row.get("git_branch"),
        git_repo_root=row.get("git_repo_root"),
        git_metadata_generation=int(row.get("git_metadata_generation") or 0),
        title=row.get("title"),
        title_source=_TITLE_SOURCES.get(row.get("title_source")),
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
        handoff_state=row.get("handoff_state"),
        handoff_platform=row.get("handoff_platform"),
        handoff_error=row.get("handoff_error"),
        compression_failure_cooldown_until=row.get("compression_failure_cooldown_until"),
        compression_failure_error=row.get("compression_failure_error"),
        compression_fallback_streak=int(row.get("compression_fallback_streak") or 0),
        compression_ineffective_count=int(row.get("compression_ineffective_count") or 0),
        compression_recovery_deadline=row.get("compression_recovery_deadline"),
        compression_overload_streak=int(row.get("compression_overload_streak") or 0),
        profile_name=row.get("profile_name"),
        transport_profile=row.get("transport_profile"),
        rewind_count=int(row.get("rewind_count") or 0),
        archived=bool(row.get("archived")),
        auto_archived=bool(row.get("auto_archived")),
        pinned=bool(row.get("pinned")),
        hidden=bool(row.get("hidden")),
        last_read_at=row.get("last_read_at"),
        tool_pin=_tool_pin(row.get("tool_names")),
        transcript_revision=int(row.get("transcript_revision") or 0),
        runtime_revision=int(row.get("runtime_revision") or 0),
        runtime_generation=int(row.get("runtime_generation") or 0),
        lineage_origin_id=origin_id,
        model_config_corrupt=is_corrupt_model_config(row.get("model_config")),
        model_config_json=row.get("model_config"),
        origin_json=row.get("origin_json"),
        tool_names=row.get("tool_names"),
    )


@contextmanager
def _title_refusals(session_id: SessionId) -> Iterator[None]:
    """The store's title ``ValueError`` refusals as :class:`TitleConflict` and :class:`CanonicalTitleLocked`, with the
    store's text; any other ``ValueError`` (a title too long) stays as raised."""
    try:
        yield
    except ValueError as exc:
        text = str(exc)
        if held := _TITLE_HELD.fullmatch(text):
            raise TitleConflict(held["title"], SessionId(held["holder"])) from exc
        if text.startswith(_CANONICAL_TITLE_LOCKED):
            raise CanonicalTitleLocked(session_id) from exc
        raise


def _optional_list(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    return list(values) if values is not None else None


def _sql_limit(window: SessionWindow) -> int:
    """The window's ``LIMIT`` operand; SQLite reads ``-1`` as no limit."""
    return -1 if window.limit is None else window.limit


def _last_active(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("last_active")
    return None if value is None else float(value)


def _list_entry(holder: StoreHolder, row: Mapping[str, Any], *, compact: bool = False) -> SessionListEntry:
    """One listing entry of a rich row; a row the store did not mark ``unread`` takes the store's unread rule."""
    return SessionListEntry(
        session=session_record(row),
        preview=row.get("preview") or "",
        unread=bool(row["unread"]) if "unread" in row else session_unread(dict(row)),
        last_active=_last_active(row),
        lineage_root_id=row.get("_lineage_root_id"),
        lineage_ids=_ids(row.get("_lineage_ids") or ()),
        continuation_kind=row.get("continuation_kind"),
        compact=compact,
    )


def _recency_entry(row: Mapping[str, Any]) -> SessionRecencyEntry:
    return SessionRecencyEntry(session=session_record(row), last_active=_last_active(row))


def _read_bot_sessions(holder, subject, operator, rule):
    """The Bots roster's session fields of the store, in one read connection."""
    hidden_sources = ','.join('?' * len(INTERNAL_LISTING_SOURCES))
    owner_filter, owner_params = ('', ()) if operator else ('user_id=? AND ', (subject,))
    # One read connection for the whole row. A store closed meanwhile (its profile unserved
    # mid-poll) refuses rather than reopening a writer into a home that is being deleted.
    with holder._read_ctx(reopen=False) as conn:
        def summary(root):
            tip = _compression_tip_on(holder, conn, root['id'])
            target = conn.execute('SELECT * FROM sessions WHERE id=?', (tip,)).fetchone() or root
            message = conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND active=1 "
                "AND role IN ('user','assistant') AND TRIM(COALESCE(content,''))!='' "
                "ORDER BY id DESC LIMIT 1", (tip,)).fetchone()
            text = ' '.join(str(message[0] or '').split()) if message else ''
            return {'id': root['id'], 'resolved_id': tip, 'title': target['title'] or '',
                    'root_title': root['title'] or '',
                    'preview': text[:80] + '...' if len(text) > 80 else text,
                    'started_at': target['started_at'] or 0,
                    'last_active': target['last_activity_at'] or target['started_at'] or 0,
                    'message_count': target['message_count'] or 0}

        fields: dict = {'last_session': None, 'worker_session': None, 'canonical_session': None}
        # The named registry is not a recency window and canonical chats are hidden.
        canonical = conn.execute(f'SELECT * FROM sessions WHERE title=? AND {_LOCAL_CHAT}',
                                 ('Bot Chat', LEGACY_PREFIX)).fetchone()
        if canonical and (same_owner(canonical['user_id'], subject, rule) or operator) and not canonical['archived']:
            fields['canonical_session'] = summary(canonical)
        # The newest conversation a picker shows: a hidden row (a Bot Chat, group member plumbing)
        # never wins.
        latest = conn.execute(
            f"SELECT * FROM sessions WHERE {owner_filter}{_LOCAL_CHAT} AND archived=0 AND hidden=0 "
            f"AND COALESCE(source,'') NOT IN ({hidden_sources}) "
            "ORDER BY COALESCE(last_activity_at,started_at) DESC LIMIT 1",
            (*owner_params, LEGACY_PREFIX, *INTERNAL_LISTING_SOURCES)).fetchone()
        if latest and fields['canonical_session'] and latest['id'] == canonical['id']:
            fields['last_session'] = dict(fields['canonical_session'])
        elif latest:
            fields['last_session'] = summary(latest)
        # The profile's freshest worker: newest internal source, never an archived or hidden row, no owner filter.
        worker = conn.execute(
            f"SELECT id, source, title, COALESCE(last_activity_at, started_at) AS last_active FROM sessions "
            f"WHERE COALESCE(source,'') IN ({hidden_sources}) AND archived=0 AND hidden=0 "
            "ORDER BY last_active DESC LIMIT 1", INTERNAL_LISTING_SOURCES).fetchone()
        if worker:
            fields['worker_session'] = {'id': worker['id'], 'source': (worker['source'] or '').strip().lower(),
                                        'title': worker['title'] or '', 'last_active': worker['last_active'] or 0}
    return fields


def _roster_session(fields: Optional[Mapping[str, Any]]) -> Optional[RosterSession]:
    if fields is None:
        return None
    return RosterSession(**{**fields, "id": SessionId(fields["id"]), "resolved_id": SessionId(fields["resolved_id"])})


class SQLiteSessionRepository:
    """Session rows of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # ── writes ──

    def create(self, spec: NewSession) -> SessionId:
        config = dict(spec.model_config or {})
        if marker := _MARKER_OF_KIND.get(spec.lineage_kind):
            config[marker] = spec.parent_session_id
        holder = self._store.holder()
        with refusals(holder, spec.session_id):
            try:
                create_session(holder,
                    spec.session_id, spec.source, model=spec.model, model_config=config or None,
                    system_prompt=spec.system_prompt, user_id=spec.user_id, session_key=spec.session_key,
                    chat_id=spec.chat_id, chat_type=spec.chat_type, thread_id=spec.thread_id,
                    parent_session_id=spec.parent_session_id, cwd=spec.cwd, profile_name=spec.profile_name,
                    git_repo_root=spec.git_repo_root,
                    origin_json=json.dumps(dict(spec.origin)) if spec.origin else None,
                    display_name=spec.display_name, transport_profile=spec.transport_profile,
                )
            except AuthorityRefusal as exc:
                if exc.reason != "not_found":
                    raise
                raise RetiredSessionError(spec.session_id, "not_found") from exc
        return SessionId(spec.session_id)

    def end(self, session_id: SessionId, reason: str) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            end_session(holder, session_id, reason)

    def set_title(self, session_id: SessionId, title: Optional[str]) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id), _title_refusals(session_id):
            return bool(set_session_title(holder, session_id, title))

    def set_auto_title(self, session_id: SessionId, title: str, *, source: TitleSource) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id), _title_refusals(session_id):
            return bool(set_auto_title(holder, session_id, title, source=source.value))

    def delete(
        self,
        session_id: SessionId,
        *,
        expected_ids: Optional[Sequence[SessionId]] = None,
        expected_display: Optional[Mapping[SessionId, Sequence[Mapping[str, Any]]]] = None,
        refuse_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            removed = delete_session(holder,
                session_id,
                expected_delete_ids=list(expected_ids) if expected_ids is not None else None,
                expected_display_messages=(
                    {sid: [dict(message) for message in messages] for sid, messages in expected_display.items()}
                    if expected_display is not None else None
                ),
                exclude_active_write_guards=refuse_guarded,
                include_compression_chain=include_compression_chain,
                media_root=media_root,
            )
        return _removed(removed)

    def delete_many(
        self,
        session_ids: Sequence[SessionId],
        *,
        skip_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> DeleteManyOutcome:
        holder = self._store.holder()
        skipped: list[str] = []
        with refusals(holder):
            deleted, removed = delete_sessions(holder,
                list(session_ids), exclude_active_write_guards=skip_guarded, skipped_ids=skipped,
                include_compression_chain=include_compression_chain, media_root=media_root,
            )
        return DeleteManyOutcome(deleted=deleted, skipped_guarded=_ids(skipped), removed_ids=_removed(removed))

    def delete_if_empty(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return _ids(delete_session_if_empty(holder, session_id, media_root=media_root))

    def delete_empty(self, *, media_root: str) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder):
            return _removed(delete_empty_sessions(holder, media_root=media_root))

    def create_exclusive(
        self,
        session_id: SessionId,
        *,
        source: str,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        holder = self._store.holder()
        model_config = dict(model_config) if model_config else None

        def _atomic(conn):
            if conflict := create_conflict(conn, session_id):  # one BEGIN IMMEDIATE write: a racing create sees the row
                return None, conflict
            conn.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, source, model, json.dumps(model_config) if model_config else None,
                 system_prompt, time.time()))
            if title is not None:
                clean_title = sanitize_title(str(title))
                if clean_title:
                    try:
                        _resolve_title_conflict(holder, conn, session_id, clean_title)
                    except ValueError as exc:  # the DB's uniqueness rule; undo the INSERT
                        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                        return None, ("title", str(exc))
                conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean_title, session_id))
            return dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()), None

        with refusals(holder, session_id):
            row, conflict = holder._execute_write(_atomic)
            if conflict is None:
                return session_record(row)
            code, message = conflict
            if code == "title":
                with _title_refusals(session_id):
                    raise ValueError(message)
            existing = get_session(holder, session_id) if code == "session_exists" else None
        if existing is not None:
            raise SessionExists(session_record(existing))
        raise RetiredSessionError(session_id, "session_deleted")

    def reopen(self, session_id: SessionId) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            reopen_session(holder, session_id)

    def retire_undrained_queue_rows(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return int(retire_undrained_queue_rows(holder, session_id) or 0)

    def promote_to_reset(self, session_id: SessionId, reason: str = END_REASON_SESSION_RESET) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return promote_to_session_reset(holder, session_id, reason)

    def reopen_if_explicitly_closed(
        self, session_id: SessionId, *, provenance: str, patience_s: Optional[float] = None
    ) -> Optional[str]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return reopen_if_explicitly_closed(holder, session_id, provenance=provenance, patience_s=patience_s)

    def discard_unadmitted(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return _removed(discard_unadmitted_session(holder, session_id, media_root=media_root))

    def set_title_source(self, session_id: SessionId, source: TitleSource) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return set_session_title_source(holder, session_id, source.value)

    def retitle_canonical_bot_chat(self, session_id: SessionId, title: str) -> bool:
        holder = self._store.holder()

        def write(conn):
            return conn.execute("UPDATE sessions SET title=?, title_source=?, hidden=0 WHERE id=? AND title='Bot Chat'",
                                (title, TITLE_SOURCE_USER, session_id)).rowcount
        with refusals(holder, session_id):
            return bool(holder._execute_write(write))

    def replace_session_config(self, session_id: SessionId, command: ReplaceSessionConfig) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_session_meta(holder, session_id, json.dumps(dict(command.model_config)), command.model)

    def patch_model_config(self, session_id: SessionId, patch: Mapping[str, Any]) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            patch_session_model_config(holder, session_id, dict(patch))

    def advance_onboarding_run(self, session_id: SessionId, event: Any, *, now: float) -> OnboardingAdvance:
        holder = self._store.holder()
        outcome: dict[str, Any] = {}

        def update(current: Any) -> Any:
            outcome.clear()
            if not isinstance(current, dict):
                return current
            outcome["was_active"] = onboarding_flow.is_active(current)
            outcome["step"] = onboarding_flow.apply(
                current, event, onboarding_flow.Facts.from_stored(current["facts"]), now=now)
            return outcome["step"].record
        with refusals(holder, session_id):
            tip = update_conversation_model_config_value(holder, session_id, ONBOARDING_RUN_KEY, update)
        return OnboardingAdvance(session_id=None if tip is None else SessionId(tip),
                                 was_active=outcome.get("was_active", False), step=outcome.get("step"))

    def update_model(
        self,
        session_id: SessionId,
        model: str,
        provider: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
    ) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_session_model(holder, session_id, model, provider, base_url=base_url, api_mode=api_mode)

    def set_runtime_lock(self, session_id: SessionId, lock: RuntimeLock) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_session_runtime_lock(holder,
                session_id, model=lock.model, provider=lock.provider,
                model_options=dict(lock.model_options) if lock.model_options is not None else None,
                route_source=lock.route_source, confirmed=lock.confirmed,
            )

    def set_yolo(self, session_id: SessionId, enabled: bool) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            set_session_yolo(holder, session_id, enabled)

    def update_system_prompt(self, session_id: SessionId, system_prompt: Optional[str]) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_system_prompt(holder, session_id, system_prompt)

    def request_prompt_refresh(self, session_id: Optional[SessionId] = None) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return request_prompt_refresh(holder, session_id)

    def update_tool_pin(self, session_id: SessionId, tool_pin: Any) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            update_session_tool_names(holder, session_id, tool_pin)

    def update_cwd(
        self,
        session_id: SessionId,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        *,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return update_session_cwd(holder, session_id, cwd, git_branch, git_repo_root, replace_git_meta=replace_git_meta)

    def publish_git_metadata(
        self,
        session_id: SessionId,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return bool(publish_session_git_metadata(holder, session_id, cwd, generation, git_branch, git_repo_root))

    def set_archived(self, session_id: SessionId, archived: bool, *, notify: bool = True) -> ArchiveChange:
        holder = self._store.holder()
        with refusals(holder, session_id):
            changed, archived_ids = set_session_archived(holder, session_id, archived, notify=notify)
        return ArchiveChange(changed=changed, archived_ids=_removed(archived_ids))

    def set_pinned(self, session_id: SessionId, pinned: bool) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return bool(set_session_pinned(holder, session_id, pinned))

    def set_hidden(self, session_id: SessionId, hidden: bool) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return bool(set_session_hidden(holder, session_id, hidden))

    def set_read(self, session_id: SessionId, read: bool = True) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return bool(set_session_read(holder, session_id, read))

    def unarchive_recoverable(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return unarchive_recoverable_session(holder, session_id)

    def backfill_repo_roots(self, cwd_to_root: Mapping[str, str]) -> None:
        holder = self._store.holder()
        with refusals(holder):
            backfill_repo_roots(holder, dict(cwd_to_root))

    def backfill_session_owners(self, *, stamp: Optional[str] = None) -> int:
        holder = self._store.holder()
        owner = _own_profile_name(holder) if stamp is None else stamp
        if not owner:
            return 0
        with refusals(holder):
            return int(backfill_null_session_profiles(holder, owner) or 0)

    def backfill_acp_session_cwds(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return int(backfill_acp_session_cwd(holder) or 0)

    def retag_worker_sessions(self, workspaces_root: str) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return int(retag_kanban_worker_sessions(holder, workspaces_root) or 0)

    def touch_activity(
        self,
        session_id: SessionId,
        at: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            touch_session_activity(holder, session_id, at, description=description, provenance=provenance)

    def clear_activity_labels(self, session_id: SessionId) -> None:
        holder = self._store.holder()
        with refusals(holder, session_id):
            clear_session_activity_labels(holder, session_id)

    # ── reads ──

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            row = get_session(holder, session_id)
        return session_record(row) if row else None

    def owns_session(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        owner = _own_profile_name(holder)
        with refusals(holder, session_id):
            row = holder._read_one("SELECT profile_name FROM sessions WHERE id = ?", (session_id,))
        return row is not None and row["profile_name"] in (None, owner)

    def store_owner(self) -> Optional[str]:
        # The copy reads only the holder's db_path, so the bound path answers without opening the store.
        return _own_profile_name(SimpleNamespace(db_path=self._store.store_path))

    def resolve_id(self, id_or_prefix: str) -> Optional[SessionId]:
        holder = self._store.holder()
        with refusals(holder):
            found = resolve_session_id(holder, id_or_prefix)
        return SessionId(found) if found else None

    def is_kanban_owned(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return is_kanban_owned_session(holder, session_id)

    def declared_scope_identity(self, session_id: SessionId) -> DeclaredScopeIdentity:
        holder = self._store.holder()
        with refusals(holder, session_id):
            is_fork_child, source = declared_scope_identity(holder, session_id)
        return DeclaredScopeIdentity(is_fork_child=is_fork_child, source=source)

    def get_title(self, session_id: SessionId) -> Optional[str]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_session_title(holder, session_id)

    def get_title_source(self, session_id: SessionId) -> Optional[TitleSource]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return _TITLE_SOURCES.get(get_session_title_source(holder, session_id))

    def get_by_title(self, title: str) -> Optional[SessionRecord]:
        holder = self._store.holder()
        with refusals(holder):
            row = get_session_by_title(holder, title)
        return session_record(row) if row else None

    def resolve_by_title(self, title: str) -> Optional[SessionId]:
        holder = self._store.holder()
        with refusals(holder):
            found = resolve_session_by_title(holder, title)
        return SessionId(found) if found else None

    def next_title_in_lineage(self, base_title: str) -> str:
        holder = self._store.holder()
        with refusals(holder):
            return get_next_title_in_lineage(holder, base_title)

    def title_holder(self, title: str, *, excluding: SessionId) -> Optional[SessionId]:
        holder = self._store.holder()
        if not title:
            return None
        with refusals(holder), holder._read_ctx() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE title = ? AND id != ?", (title, excluding)).fetchone()
        return SessionId(row[0]) if row else None

    def get_model_config_value(self, session_id: SessionId, key: str, default: Any = None) -> Any:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return get_session_model_config_value(holder, session_id, key, default)

    def session_ids_with_model_config_key(self, key: str) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder):
            return _ids(session_ids_with_model_config_key(holder, key))

    def prompt_refresh_epoch(self, session_id: SessionId) -> int:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return prompt_refresh_epoch(holder, session_id)

    def count_with_stored_prompt(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return count_sessions_with_stored_prompt(holder)

    def delete_targets(self, session_id: SessionId) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            return _ids(get_session_delete_targets(holder, session_id))

    def is_empty_candidate(self, session_id: SessionId) -> bool:
        holder = self._store.holder()
        with refusals(holder, session_id), holder._read_ctx() as conn:
            empty = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND title IS NULL"
                " AND NOT EXISTS (SELECT 1 FROM messages WHERE messages.session_id = sessions.id)"
                " AND NOT EXISTS (SELECT 1 FROM sessions child WHERE child.parent_session_id = sessions.id)",
                (session_id,)).fetchone()
        return empty is not None

    def count_empty(self) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return int(count_empty_sessions(holder))

    def lifecycle_statuses(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionLifecycleStatus]:
        holder = self._store.holder()
        with refusals(holder):
            statuses = session_lifecycle_statuses(holder, list(session_ids))
        return {SessionId(sid): SessionLifecycleStatus(status) for sid, status in statuses.items()}

    def descendants(self, session_id: SessionId) -> tuple[SessionEdge, ...]:
        holder = self._store.holder()
        with refusals(holder, session_id), holder._read_ctx() as conn:
            rows = conn.execute(_DESCENDANTS_SQL, (session_id,)).fetchall()
        return tuple(SessionEdge(session_id=SessionId(row[0]), parent_session_id=row[1], started_at=row[2])
                     for row in rows)

    def first_session_started_at(self) -> Optional[float]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            row = conn.execute("SELECT MIN(started_at) FROM sessions").fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def longest_session_id(self) -> Optional[SessionId]:
        holder = self._store.holder()
        with refusals(holder), holder._read_ctx() as conn:
            row = conn.execute(
                "SELECT id FROM sessions s ORDER BY "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) DESC LIMIT 1"
            ).fetchone()
        return SessionId(row[0]) if row else None

    def sessions_in_workspace(self, session_ids: Sequence[SessionId], workspace: str) -> frozenset[SessionId]:
        holder = self._store.holder()
        with refusals(holder):
            return frozenset(SessionId(sid) for sid in stored_reads.sessions_in_workspace(holder, session_ids, workspace))

    def picker_session_facts(self, root_ids: Sequence[SessionId], segment_ids: Sequence[SessionId],
                             reply_ids: Sequence[SessionId]) -> PickerFacts:
        holder = self._store.holder()
        with refusals(holder):
            roots, segments, replies = stored_reads.picker_session_facts(holder, root_ids, segment_ids, reply_ids)
        return PickerFacts(
            roots={SessionId(sid): PickerRoot(cwd=row["cwd"], git_repo_root=row["git_repo_root"]) for sid, row in roots.items()},
            segments={SessionId(sid): PickerSegment(preview=_shape_preview(row["_preview_raw"]), git_branch=row["git_branch"],
                                                    pinned=bool(row["pinned"]), last_active=row["last_active"])
                      for sid, row in segments.items()},
            last_replies={SessionId(sid): reply for sid, reply in replies.items()})

    def subagent_session_ids(self, subagent_id: str, marker_key: str) -> tuple[SessionId, ...]:
        holder = self._store.holder()
        with refusals(holder):
            return tuple(SessionId(sid) for sid in stored_reads.subagent_session_ids(holder, subagent_id, marker_key))

    def records_created_surface(self) -> bool:
        holder = self._store.holder()
        with refusals(holder):
            rows = holder._read_all("PRAGMA table_info(sessions)")
        return any(row[1] == "created_source" for row in rows)

    def last_active(self, session_id: SessionId) -> Optional[float]:
        holder = self._store.holder()
        with refusals(holder, session_id), holder._read_ctx() as conn:
            row = conn.execute(
                f"SELECT {_sql_session_last_active('s')} FROM sessions s WHERE s.id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def list_sessions(
        self,
        query: SessionQuery,
        window: SessionWindow = SessionWindow(limit=20),
        *,
        compact_rows: bool = False,
    ) -> tuple[SessionListEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = list_sessions_rich(holder,
                sources=_listed(query.sources), exclude_sources=_listed(query.exclude_sources),
                cwd_prefix=query.cwd_prefix, limit=_sql_limit(window), offset=window.offset,
                include_children=query.include_children, min_message_count=query.min_message_count,
                project_compression_tips=query.project_compression_tips,
                order_by_last_active=query.order is SessionOrder.LAST_ACTIVE, include_archived=query.include_archived,
                archived_only=query.archived_only, id_query=query.id_query, search_query=query.search_text,
                compact_rows=compact_rows, include_pinned=query.include_pinned, session_key=query.session_key,
                include_hidden=query.include_hidden, include_subagents=query.include_subagents,
                local_created_sources=_optional_list(query.local_created_sources), workspace_key=query.workspace_key,
            )
        return tuple(_list_entry(holder, row, compact=compact_rows) for row in rows)

    def search_sessions(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        workspace_key: Optional[str] = None,
        local_created_sources: Optional[Sequence[str]] = None,
        skip_launch_stubs: bool = False,
        window: SessionWindow = SessionWindow(limit=20),
    ) -> tuple[SessionRecencyEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = search_sessions(holder,
                source=_listed(sources), limit=_sql_limit(window), offset=window.offset, workspace_key=workspace_key,
                local_created_sources=_optional_list(local_created_sources), skip_launch_stubs=skip_launch_stubs,
            )
        return tuple(_recency_entry(row) for row in rows)

    def list_row(self, session_id: SessionId) -> Optional[SessionListRow]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            row = get_session_list_row(holder, session_id)
        if row is None:
            return None
        return SessionListRow(
            session_id=SessionId(row["id"]), source=row["source"] or "", title=row["title"],
            started_at=float(row["started_at"] or 0.0), ended_at=row["ended_at"],
            message_count=row["message_count"], model=row["model"], hidden=bool(row["hidden"]),
            archived=bool(row["archived"]),
        )

    def summary(self, session_id: SessionId, *, compact_rows: bool = False) -> Optional[SessionListEntry]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            row = get_session_rich_row(holder, session_id, compact_rows=compact_rows)
        return _list_entry(holder, row, compact=compact_rows) if row else None

    def search_by_id(
        self,
        text: str,
        *,
        limit: int = 20,
        include_archived: bool = True,
        sources: Optional[Sequence[str]] = None,
        exclude_sources: Sequence[str] = (),
    ) -> tuple[SessionListEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = search_sessions_by_id(holder,
                text, limit=limit, include_archived=include_archived, sources=_listed(sources),
                exclude_sources=_listed(exclude_sources),
            )
        return tuple(_list_entry(holder, row) for row in rows)

    def count_sessions(self, query: SessionCountQuery = SessionCountQuery()) -> int:
        holder = self._store.holder()
        with refusals(holder):
            return int(session_count(holder,
                sources=_listed(query.sources), cwd_prefix=query.cwd_prefix, min_message_count=query.min_message_count,
                include_archived=query.include_archived, archived_only=query.archived_only,
                exclude_children=query.exclude_children, exclude_sources=_listed(query.exclude_sources),
                include_subagents=query.include_subagents, local_created_sources=_optional_list(query.local_created_sources),
            ))

    def count_at_least(self, minimum: int = 1) -> bool:
        holder = self._store.holder()
        with refusals(holder):
            return session_count_ge(holder, minimum)

    def count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False
    ) -> Mapping[str, int]:
        holder = self._store.holder()
        with refusals(holder):
            return session_count_by_source(holder,
                include_archived=include_archived, archived_only=archived_only, exclude_children=exclude_children,
            )

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True
    ) -> tuple[SessionRecencyEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = list_gateway_sessions(holder, platform=platform, active_only=active_only)
        return tuple(_recency_entry(row) for row in rows)

    def list_recent_sessions(self, query: RecentBrowseQuery) -> tuple[RecentSessionEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            try:
                rows = list_recent_sessions_bounded(holder,
                    limit=query.limit, exclude_sources=_listed(query.exclude_sources),
                    timeout_seconds=query.timeout_seconds, candidate_limit=query.candidate_limit,
                    lineage_limit=query.lineage_limit,
                )
            except TimeoutError as exc:
                raise RecentBrowseTimeout(str(exc)) from exc
        return tuple(
            RecentSessionEntry(
                session_id=SessionId(row["id"]), source=row["source"] or "", model=row["model"], title=row["title"],
                started_at=float(row["started_at"] or 0.0), ended_at=row["ended_at"], end_reason=row["end_reason"],
                message_count=row["message_count"], tool_call_count=row["tool_call_count"],
                last_active=_last_active(row), preview=row.get("preview") or "",
                lineage_root_id=row.get("_lineage_root_id"), unread=bool(row.get("unread")),
            )
            for row in rows
        )

    def list_cron_job_runs(
        self, job_id: str, window: SessionWindow = SessionWindow(limit=20)
    ) -> tuple[CronRunEntry, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = list_cron_job_runs(holder, job_id, limit=_sql_limit(window), offset=window.offset)
        return tuple(
            CronRunEntry(entry=_list_entry(holder, row), cron_job_id=row["cron_job_id"],
                         cron_execution_id=row.get("cron_execution_id"))
            for row in rows
        )

    def cron_run_identity(self, session_id: SessionId) -> Optional[CronRunIdentity]:
        holder = self._store.holder()
        with refusals(holder, session_id):
            found = cron_run_identity(holder, session_id)
        if not found:
            return None
        return CronRunIdentity(cron_job_id=found["cron_job_id"], cron_execution_id=found["cron_execution_id"])

    def list_skill_scaffolded_sessions(self, *, limit: int = 200) -> tuple[SkillScaffoldedSession, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = list_skill_scaffolded_sessions(holder, limit)
        return tuple(SkillScaffoldedSession(session_id=SessionId(row["id"]), title=row["title"], content=row["content"])
                     for row in rows)

    def list_unlinked_telegram_sessions(
        self, *, user_id: str, profile_name: Optional[str] = None, limit: int = 10,
    ) -> tuple[SessionFacetEntry, ...]:
        holder = self._store.holder()
        profile = (_own_profile_name(holder) or "default") if profile_name is None else profile_name
        with refusals(holder):
            rows = list_unlinked_telegram_sessions_for_user(holder,
                chat_id="", user_id=user_id, profile_name=profile, limit=limit,
            )
        return tuple(
            SessionFacetEntry(session=session_record(row), last_active=_last_active(row), preview=row.get("preview") or "")
            for row in rows
        )

    def distinct_session_cwds(self, *, include_archived: bool = False) -> tuple[SessionCwdUsage, ...]:
        holder = self._store.holder()
        with refusals(holder):
            rows = distinct_session_cwds(holder, include_archived=include_archived)
        return tuple(SessionCwdUsage(cwd=row["cwd"], sessions=row["sessions"], last_active=row["last_active"])
                     for row in rows)

    def roster_sessions(self, *, owner_user_id: Optional[str], owner: OwnerRule | None) -> RosterSessions:
        holder = self._store.holder(reacquire=False)
        with refusals(holder):
            fields = _read_bot_sessions(holder, owner_user_id, owner_user_id is None, owner)
        worker = fields['worker_session']
        return RosterSessions(
            canonical_session=_roster_session(fields['canonical_session']),
            last_session=_roster_session(fields['last_session']),
            worker_session=RosterWorker(
                id=SessionId(worker['id']), source=worker['source'], title=worker['title'],
                last_active=worker['last_active'],
            ) if worker else None,
        )

    def live_listing_facts(self, session_ids: Sequence[SessionId]) -> LiveListingFacts:
        holder = self._store.holder()
        ids = json.dumps(list(session_ids))
        with refusals(holder), holder._read_ctx() as conn:
            rows = conn.execute(
                "SELECT s.id, s.source, s.title, s.started_at, s.ended_at, s.message_count, s.model, s.hidden, "
                "s.archived, s.runtime_revision, s.runtime_generation, "
                "(SELECT t.ended_at FROM state_meta m JOIN sessions t ON t.id = CASE WHEN json_valid(m.value) "
                "THEN json_extract(m.value, '$.entry.session_id') END WHERE m.key = ? || s.id) AS target_ended_at "
                "FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))", (POLICY_PREFIX, ids)).fetchall()
            pending = conn.execute(
                "SELECT target_session_id, admission_id, seq, status FROM session_admissions "
                "WHERE target_session_id IN (SELECT value FROM json_each(?)) "
                "AND status IN ('queued','started','unknown') ORDER BY seq", (ids,)).fetchall()
        by_session: dict[SessionId, list[PendingAdmissionRef]] = {}
        for row in pending:
            by_session.setdefault(SessionId(row['target_session_id']), []).append(
                PendingAdmissionRef(admission_id=row['admission_id'], seq=int(row['seq']), status=row['status']))
        return LiveListingFacts(
            rows={
                SessionId(row['id']): LiveListingRow(
                    id=SessionId(row['id']), source=row['source'] or '', title=row['title'],
                    started_at=float(row['started_at'] or 0.0), ended_at=row['ended_at'],
                    message_count=row['message_count'], model=row['model'], hidden=bool(row['hidden']),
                    archived=bool(row['archived']), runtime_revision=int(row['runtime_revision'] or 0),
                    runtime_generation=int(row['runtime_generation'] or 0), target_ended_at=row['target_ended_at'],
                )
                for row in rows
            },
            pending={sid: tuple(refs) for sid, refs in by_session.items()},
        )


register_repository("sessions", SQLiteSessionRepository)
