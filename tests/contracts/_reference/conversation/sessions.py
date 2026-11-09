"""In-memory reference implementation of ``SessionRepository`` (test fixture).

Registered onto the assembled backend as the ``sessions`` attribute per the append-only extension contract in ``backend.py``. Rows live under the ``"sessions"`` key of the owning scope's namespace, so cross-scope isolation is structural. A row keeps its stamps as UTC ``datetime`` (the other reference capabilities read them so); :class:`SessionRecord` is built in one place, ``_record``, which converts them to epoch floats and composes the columns other capabilities own (usage, runtime revision, handoff, compression cooldown, rewind count).

The verbs are split by topic across this module (rows, record, lifecycle, simple reads), ``sessions_titles_config`` (titles, model config, flags), ``sessions_delete`` (deletion, retirement, backfills) and ``sessions_listing`` (listings and counts); the one repository class composes them.

Retirement fencing: the delete verbs commit through the scope's shared :class:`AuthorityState` transaction so row removal, routing-entry drops and the permanent retirement markers publish in one atomic swap. Deletes refuse targets with live ledger work (``AuthorityRefusal('session_busy')``) or unknown ledger work (:class:`persistence.UnknownExecution`; unknown wins per target), read from the authority ``admissions``/``workers`` tables; ``create`` refuses a retired absent id.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from typing import Any, Mapping, Optional

from persistence import (
    JsonValue,
    PermanentStoreError,
    ProfileScope,
    RetiredSessionError,
    SessionId,
)
from persistence.conversation.sessions import (
    ACTIVITY_PROVENANCE_UNKNOWN,
    END_REASON_COMPRESSION,
    END_REASON_SESSION_RESET,
    RECOVERABLE_END_REASONS,
    DeclaredScopeIdentity,
    NewSession,
    SessionExists,
    SessionLineageKind,
    SessionRecord,
    bound_activity_description,
    is_corrupt_model_config,
    is_explicit_close_reason,
    normalize_activity_provenance,
    parse_model_config,
    sanitize_title,
)
from persistence.core.conventions import UNTRUSTED_INSTANT, RowVisibility, stored_instant

from .._authority import authority_state
from ..backend import InMemoryBackend, register_repository
from ..housekeeping.maintenance import newest_imported_timestamp
from .transcript_store import _messages_namespace, advance_conversation_generation, newest_transcript_timestamp
from .sessions_delete import SessionDeleteMixin
from .sessions_listing import SessionListingMixin
from .sessions_titles_config import MARKER_BY_KIND as _MARKER_BY_KIND
from .sessions_titles_config import SessionTitlesConfigMixin, sync_lineage_markers, tool_pin_text
from .sessions_titles_config import ascii_lower as _ascii_lower

__all__ = ["InMemorySessionRepository", "blank_session_row", "stored_tool_pin", "sync_lineage_markers", "tool_pin_text"]

# NewSession fields copied verbatim into a fresh row and null-filled on upsert (``user_id`` is never filled later).
_SPEC_FIELDS = (
    "model",
    "cwd",
    "parent_session_id",
    "session_key",
    "chat_id",
    "chat_type",
    "thread_id",
    "git_repo_root",
)
_ROUTING_FIELDS = ("user_id", "session_key", "chat_id", "chat_type", "thread_id")
# The constructor facets ``NewSession`` carries into the creation step (keep-existing on upsert like the spec fields).
_SEAM_FIELDS = ("system_prompt", "display_name", "origin", "transport_profile")
# The upsert keeps a real creating surface; these are no creation provenance.
_UNSET_CREATED = (None, "", "unknown", "local")
# The display-metadata marker of a busy-queue accept row that never drained.
_QUEUED_PROMPT_KEY = "_queued_prompt"


def _now() -> datetime:
    return datetime.now(UTC)


def epoch(value: Optional[datetime]) -> Optional[float]:
    """A stored stamp as the epoch float the records carry."""
    return None if value is None else value.timestamp()


def stored_tool_pin(text: Optional[str]) -> Optional[JsonValue]:
    """The stored pin, decoded whole into a fresh value per read."""
    if text is None:
        return None
    value: JsonValue = json.loads(text)
    return value


def blank_session_row(session_id: str, *, lineage_kind: Any, model_config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The ONE shape of a never-touched session row in this backend. Every operation that materializes a row (create, compression publish, ...) starts here, so a facet one operation adds exists for all of them; spec fields start None and the caller fills them."""
    row: dict[str, Any] = {
        "session_id": session_id,
        "source": None,
        "created_source": None,
        "user_id": None,
        "profile_name": None,
        "started_at": _now(),
        "ended_at": None,
        "end_reason": None,
        "lineage_kind": lineage_kind,
        "lineage_origin_id": None,
        "title": None,
        "title_source": None,
        "archived": False,
        "auto_archived": False,
        "pinned": False,
        "hidden": False,
        # Epoch watermark: None is never tracked (read), 0.0 is explicitly unread.
        "last_read_at": None,
        "expiry_finalized": False,
        "cwd_generation": 0,
        "transcript_revision": 0,
        "compression_fork": False,
        "model_config": copy.deepcopy(dict(model_config or {})),
        # Whether a config was ever written: an empty create stores none, which a later create fills; an explicit empty config is kept.
        "model_config_stored": bool(model_config),
        "system_prompt": None,
        "tool_pin": None,
        "display_name": None,
        "origin": None,
        "transport_profile": None,
        # No create path writes it; a child inherits its parent's.
        "git_branch": None,
        "last_activity_at": None,
        "last_activity_description": None,
        "last_activity_provenance": None,
    }
    for field in _SPEC_FIELDS:
        row[field] = None
    return row


def _key_namespace(key: str) -> str:
    """The ``agent:<ns>:`` head of a routing key (the first six characters when no namespace follows)."""
    return key[: key[6:].find(":") + 7]


def _same_key_namespace(parent_key: Optional[str], child_key: Optional[str]) -> bool:
    """Whether a child may inherit its parent's owner: either row is keyless, or both keys share the namespace."""
    return parent_key is None or child_key is None or _key_namespace(parent_key) == _key_namespace(child_key)


def _legacy_reset_child(parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    """A markerless child ``reopen`` freezes as a reset child: started after the parent, same routing key, not a tool row."""
    config = child["model_config"]
    return (
        child["parent_session_id"] == parent["session_id"]
        and child["session_key"] == parent["session_key"]
        and child["started_at"] >= parent["started_at"]
        and child["source"] != "tool"
        and all(config.get(marker) is None for marker in _MARKER_BY_KIND.values())
    )



def _marker(config: Mapping[str, Any], key: str) -> Optional[SessionId]:
    """A fork marker's stripped value, ``None`` when absent or blank."""
    value = config.get(key)
    return SessionId(value.strip()) if isinstance(value, str) and value.strip() else None


def _config_cell(row: Mapping[str, Any]) -> Any:
    """The stored ``model_config`` cell: the JSON text of a written config, ``None`` before any write, a planted raw cell as planted."""
    config = row["model_config"]
    if not isinstance(config, Mapping):
        return config
    return json.dumps(config) if row["model_config_stored"] else None


def _instant(row: Mapping[str, Any], field: str) -> Optional[float]:
    """A stored row stamp as an epoch float; a cell that is not a trusted instant reads ``None``."""
    return epoch(stored_instant(row[field], session_id=row["session_id"], field=field))


def _usage_columns(usage_row: Any) -> dict[str, Any]:
    """The accounting columns the usage capability keeps for a row; zero and unset before any usage."""
    if usage_row is None:
        return {
            "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "reasoning_tokens": 0, "api_call_count": 0, "estimated_cost_usd": None, "actual_cost_usd": None,
            "billing_provider": None, "billing_base_url": None, "billing_mode": None,
            "cost_status": None, "cost_source": None, "pricing_version": None,
        }
    usage, route = usage_row.usage, usage_row.route
    tokens = usage.tokens
    return {
        "input_tokens": tokens.input_tokens, "output_tokens": tokens.output_tokens,
        "cache_read_tokens": tokens.cache_read_tokens, "cache_write_tokens": tokens.cache_write_tokens,
        "reasoning_tokens": tokens.reasoning_tokens, "api_call_count": usage.api_call_count,
        "estimated_cost_usd": usage.estimated_cost_usd, "actual_cost_usd": usage.actual_cost_usd,
        "billing_provider": route.billing_provider, "billing_base_url": route.billing_base_url,
        "billing_mode": route.billing_mode, "cost_status": usage.cost_status,
        "cost_source": usage.cost_source, "pricing_version": usage.pricing_version,
    }


def _cooldown_columns(cooldown: Any) -> dict[str, Any]:
    """The compression failure cooldown the messages capability keeps for a row."""
    if cooldown is None:
        return {"compression_failure_cooldown_until": None, "compression_failure_error": None}
    return {"compression_failure_cooldown_until": cooldown.until, "compression_failure_error": cooldown.error}


class InMemorySessionRepository(SessionTitlesConfigMixin, SessionDeleteMixin, SessionListingMixin):
    """See module docstring; semantics follow persistence/conversation/sessions.py exactly."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend
        # touch_activity, clear_activity_labels, reopen_if_explicitly_closed and the unadmitted discard hold this lock across their read-then-write steps; the corresponding store writes are transactional.
        self._flags_lock = threading.Lock()

    # ── storage plumbing ──────────────────────────────────────────────────

    def _table(self, scope: ProfileScope) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self._backend.scope_store(scope).setdefault("sessions", {})
        return table

    def _owned_columns(self, scope: ProfileScope, session_id: str) -> dict[str, Any]:
        """The record columns other reference capabilities own: usage, runtime revision, handoff, cooldown, transcript counters."""
        store = self._backend.scope_store(scope)
        runtime = authority_state(self._backend, scope).snapshot["sessions"].get(session_id) or {}
        handoff = store.get("handoffs", {}).get(session_id) or {}
        messages = _messages_namespace(self._backend, scope)
        transcript = messages["transcripts"].get(session_id)
        message_count, tool_call_count = transcript.counters() if transcript is not None else (0, 0)
        status = handoff.get("status")
        return {
            **_usage_columns(store.get("usage", {}).get("sessions", {}).get(session_id)),
            **_cooldown_columns(messages["cooldowns"].get(session_id)),
            "runtime_revision": int(runtime.get("revision", 0)),
            "runtime_generation": int(runtime.get("generation", 0)),
            "handoff_state": getattr(status, "value", status),
            "handoff_platform": handoff.get("platform"),
            "handoff_error": handoff.get("error"),
            "message_count": message_count,
            "tool_call_count": tool_call_count,
            "rewind_count": transcript.rewind_count if transcript is not None else 0,
        }

    def _record(self, scope: ProfileScope, row: Mapping[str, Any], *, resolve_tool_pin: bool = True) -> SessionRecord:
        """The one place a :class:`SessionRecord` is built from a stored row. Pins are stored content-addressed; a read that does not join ``tool_names`` to the prompt store (``resolve_tool_pin=False``) carries the hash, so its ``tool_pin`` is ``None``."""
        # A planted raw cell (text) reads as the store reads it: tolerant decode, with the corrupt flag.
        config = copy.deepcopy(parse_model_config(row["model_config"]))
        prompt = row["system_prompt"]
        pin = row["tool_pin"]
        return SessionRecord(
            session_id=SessionId(row["session_id"]),
            source=row["source"],
            created_source=row.get("created_source"),
            user_id=row["user_id"],
            session_key=row["session_key"],
            chat_id=row["chat_id"],
            chat_type=row["chat_type"],
            thread_id=row["thread_id"],
            display_name=row["display_name"],
            origin=copy.deepcopy(dict(row["origin"])) if row["origin"] else None,
            expiry_finalized=bool(row.get("expiry_finalized")),
            model=row["model"],
            model_config=config,
            system_prompt=prompt,
            system_prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt is not None else None,
            parent_session_id=row["parent_session_id"],
            lineage_kind=row["lineage_kind"],
            reset_from=_marker(config, "_reset_from"),
            branched_from=_marker(config, "_branched_from"),
            started_at=epoch(stored_instant(row["started_at"], session_id=row["session_id"], field="started_at") or UNTRUSTED_INSTANT),
            ended_at=_instant(row, "ended_at"),
            end_reason=row["end_reason"],
            cwd=row["cwd"],
            git_branch=row["git_branch"],
            git_repo_root=row["git_repo_root"],
            # Counter cells as stored: rows reach the caller without coercion.
            git_metadata_generation=row.get("cwd_generation", 0),
            title=row["title"],
            title_source=row["title_source"],
            last_activity_at=row["last_activity_at"],
            last_activity_description=row["last_activity_description"],
            last_activity_provenance=row["last_activity_provenance"],
            compression_fallback_streak=row.get("compression_fallback_streak", 0),
            compression_ineffective_count=row.get("compression_ineffective_count", 0),
            compression_recovery_deadline=row.get("compression_recovery_deadline"),
            compression_overload_streak=row.get("compression_overload_streak", 0),
            profile_name=row.get("profile_name"),
            transport_profile=row.get("transport_profile"),
            archived=bool(row["archived"]),
            auto_archived=bool(row.get("auto_archived")),
            pinned=bool(row["pinned"]),
            hidden=bool(row["hidden"]),
            last_read_at=row.get("last_read_at"),
            tool_pin=stored_tool_pin(row["tool_pin"]) if resolve_tool_pin else None,
            transcript_revision=row.get("transcript_revision", 0),
            lineage_origin_id=row.get("lineage_origin_id"),
            model_config_corrupt=is_corrupt_model_config(row["model_config"]),
            model_config_json=_config_cell(row),
            origin_json=json.dumps(row["origin"]) if row["origin"] else None,
            # The store keeps the pin content-addressed; only a joined read resolves the hash to its text.
            tool_names=pin if resolve_tool_pin or pin is None else hashlib.sha256(pin.encode("utf-8")).hexdigest(),
            **self._owned_columns(scope, row["session_id"]),
        )

    def _transcripts(self, scope: ProfileScope) -> dict[str, Any]:
        """The messages reference's transcripts of the scope, keyed by session id, each with ``rows`` over every visibility."""
        transcripts: dict[str, Any] = _messages_namespace(self._backend, scope)["transcripts"]
        return transcripts

    def _message_count(self, scope: ProfileScope, row: Mapping[str, Any]) -> int:
        """Stored rows for the session, any visibility, across both transcript stores of this fixture (the messages reference's appended rows and the maintenance reference's imported rows), the way a ``NOT EXISTS (SELECT 1 FROM messages ...)`` guard sees every row."""
        appended = self._transcripts(scope).get(row["session_id"])
        imported = self._backend.scope_store(scope).get("transcripts", {}).get(row["session_id"], [])
        return (len(appended.rows) if appended is not None else 0) + len(imported)

    def _activity_stamps(self, scope: ProfileScope, row: Mapping[str, Any]) -> tuple[Optional[float], ...]:
        """The activity stamp and the newest stored transcript row of any visibility (appended or imported)."""
        return (
            row["last_activity_at"],
            epoch(newest_transcript_timestamp(self._backend, scope, row["session_id"])),
            epoch(newest_imported_timestamp(self._backend, scope, row["session_id"])),
        )

    def _last_active(self, scope: ProfileScope, row: Mapping[str, Any]) -> Optional[float]:
        """The newest trusted ``_activity_stamps``, else a trusted ``started_at``; None when no cell is trusted."""
        sid = row["session_id"]
        trusted = [instant for stamp in self._activity_stamps(scope, row) if (instant := stored_instant(stamp, session_id=sid, field="last_active")) is not None]
        return epoch(max(trusted) if trusted else stored_instant(row["started_at"], session_id=sid, field="started_at"))

    def _lineage_ids(self, table: dict[str, dict[str, Any]], session_id: str) -> list[str]:
        """The connected component over compression-fork edges containing ``session_id``."""
        seen = {session_id}
        frontier = [session_id]
        while frontier:
            current = frontier.pop()
            row = table.get(current)
            if row is None:
                continue
            parent = row["parent_session_id"]
            if row["compression_fork"] and parent and parent not in seen:
                seen.add(parent)
                frontier.append(parent)
            for other in table.values():
                if other["compression_fork"] and other["parent_session_id"] == current and other["session_id"] not in seen:
                    seen.add(other["session_id"])
                    frontier.append(other["session_id"])
        return [sid for sid in seen if sid in table]

    @staticmethod
    def _is_fork_child_row(row: Mapping[str, Any], *, include_reset: bool = False) -> bool:
        """A branch, delegate or tool child of its parent (``include_reset``: a reset fork too); a marker counts only when it names the parent, since a continuation inherits its parent's markers."""
        if row["source"] == "tool":
            return True
        config = row["model_config"]
        keys = ("_branched_from", "_delegate_from", "_reset_from") if include_reset else ("_branched_from", "_delegate_from")
        markers = [config.get(key) for key in keys]
        parent = row["parent_session_id"]
        return parent in markers if parent else any(marker is not None for marker in markers)

    def _local_receipt(self, scope: ProfileScope, owner: str) -> Optional[Mapping[str, Any]]:
        """The local creation receipt the bindings capability stored for ``owner``."""
        from ..authority.bindings import _RECEIPT_PREFIX

        stored = authority_state(self._backend, scope).snapshot["meta"].get(_RECEIPT_PREFIX + owner)
        return stored if isinstance(stored, dict) else None

    def _local_conversation_ids(self, scope: ProfileScope, session_id: str) -> list[str]:
        """Every row of the local conversation ``session_id`` belongs to: the creation id (a segment carries it as ``chat_id``), then each segment of its receipt lineage; ``[session_id]`` outside one."""
        row = self._table(scope).get(session_id)
        owner = row["chat_id"] if row is not None else None
        if not owner or owner == session_id:
            owner = session_id
        receipt = self._local_receipt(scope, owner)
        lineage = receipt.get("lineage") if receipt is not None else None
        if not isinstance(lineage, list) or (owner != session_id and session_id not in lineage):
            return [session_id]
        return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))

    def _owned_lineage_ids(self, scope: ProfileScope, session_id: str) -> list[str]:
        """``[session_id]`` outside a local reset lineage, else the whole conversation with ``session_id`` first."""
        return list(dict.fromkeys([session_id, *self._local_conversation_ids(scope, session_id)]))

    @staticmethod
    def _classify(role: str, has_tool_calls: bool, finish_reason: Optional[str]) -> Any:
        """A transcript's last row as a lifecycle status: an error finish wins, then a trailing user or tool row or pending tool calls read interrupted."""
        from persistence.conversation.sessions import ERROR_FINISH_REASONS, SessionLifecycleStatus

        if (finish_reason or "").strip().lower() in ERROR_FINISH_REASONS:
            return SessionLifecycleStatus.ERROR
        normalized = (role or "").strip().lower()
        if normalized in {"user", "tool"} or (normalized == "assistant" and has_tool_calls):
            return SessionLifecycleStatus.INTERRUPTED
        return SessionLifecycleStatus.COMPLETE

    # ── lifecycle ─────────────────────────────────────────────────────────

    def create(self, spec: NewSession) -> SessionId:
        scope = self._backend.scope
        config = copy.deepcopy(dict(spec.model_config or {}))
        if marker := _MARKER_BY_KIND.get(spec.lineage_kind):
            config[marker] = spec.parent_session_id
        table = self._table(scope)
        existing = table.get(spec.session_id)
        if existing is not None:
            self._upsert(existing, spec, config, self._backend.owner_profile)
            if spec.parent_session_id:
                self._inherit_from_parent(table, existing)
            return SessionId(spec.session_id)
        # A retired ABSENT id is never creatable again; the fence sits after the upsert arm.
        if spec.session_id in authority_state(self._backend, scope).snapshot["retired"]:
            raise RetiredSessionError(spec.session_id, "not_found")
        if spec.parent_session_id is not None and spec.parent_session_id not in table:
            from sqlite3 import IntegrityError

            raise PermanentStoreError("FOREIGN KEY constraint failed") from IntegrityError("FOREIGN KEY constraint failed")
        row = blank_session_row(spec.session_id, lineage_kind=spec.lineage_kind, model_config=config)
        for field in (*_SPEC_FIELDS, *_SEAM_FIELDS, "user_id"):
            row[field] = getattr(spec, field)
        row["origin"] = copy.deepcopy(dict(spec.origin)) if spec.origin else None
        row["source"] = spec.source
        row["created_source"] = spec.source
        owner = self._backend.owner_profile
        row["profile_name"] = spec.profile_name if (spec.profile_name or "").strip() else owner
        sync_lineage_markers(row)
        self._inherit_from_parent(table, row)
        table[spec.session_id] = row
        self._backend.lineage._stamp_ancestors(scope, table, spec.session_id)
        return SessionId(spec.session_id)

    @staticmethod
    def _upsert(row: dict[str, Any], spec: NewSession, config: dict[str, Any], owner: Optional[str]) -> None:
        """Fill what an earlier writer left unset: a placeholder ``unknown`` source takes the spec's, an unset creating surface takes a real one, and a config holding only a ``_reset_from`` marker takes the spec's config with the marker kept."""
        for field in (*_SPEC_FIELDS, *_SEAM_FIELDS):
            value = getattr(spec, field)
            if row[field] is None and value is not None:
                row[field] = copy.deepcopy(dict(value)) if field == "origin" else value
        if row["source"] == "unknown":
            row["source"] = spec.source
        if row.get("created_source") in _UNSET_CREATED and spec.source != "local":
            row["created_source"] = spec.source
        if row.get("profile_name") is None:
            row["profile_name"] = spec.profile_name if (spec.profile_name or "").strip() else owner
        current = row["model_config"]
        if config and set(current) == {"_reset_from"}:
            row["model_config"] = {**config, "_reset_from": current["_reset_from"]}
        elif not row["model_config_stored"] and config:
            row["model_config"] = config
            row["model_config_stored"] = True
        sync_lineage_markers(row)

    @staticmethod
    def _inherit_from_parent(table: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
        """Fill the unset columns a child inherits: cwd and git always, the owner within the same ``agent:<ns>:`` key namespace, and the routing and peer columns when the parent ended by compression and the child is no delegate or branch fork of it."""
        parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
        if parent is None:
            return
        for field in ("cwd", "git_repo_root", "git_branch"):
            if row[field] is None:
                row[field] = parent[field]
        if row.get("profile_name") is None and _same_key_namespace(parent["session_key"], row["session_key"]):
            row["profile_name"] = parent.get("profile_name")
        if parent["end_reason"] != END_REASON_COMPRESSION:
            return
        row["compression_fork"] = row["lineage_kind"] is SessionLineageKind.NONE and row["source"] != "tool"
        config = row["model_config"]
        if parent["session_id"] in (config.get("_delegate_from"), config.get("_branched_from")):
            return
        for field in (*_ROUTING_FIELDS, "display_name", "origin", "transport_profile"):
            if row[field] is None:
                row[field] = dict(parent[field]) if field == "origin" and parent[field] is not None else parent[field]

    def create_exclusive(
        self,
        session_id: SessionId,
        *,
        source: str,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, JsonValue]] = None,
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        scope = self._backend.scope
        table = self._table(scope)
        existing = table.get(session_id)
        if existing is not None:
            raise SessionExists(self._record(scope, existing))
        if session_id in authority_state(self._backend, scope).snapshot["retired"]:
            raise RetiredSessionError(session_id, "session_deleted")
        clean = sanitize_title(title)
        if clean:
            self._resolve_title_conflict(scope, table, session_id, clean)
        row = blank_session_row(session_id, lineage_kind=SessionLineageKind.NONE, model_config=model_config)
        # An API-server create sets only the source, model, model config, system prompt and title on a blank row.
        row.update(source=source, model=model, system_prompt=system_prompt, title=clean)
        sync_lineage_markers(row)
        table[session_id] = row
        return self._record(scope, row)

    def end(self, session_id: SessionId, reason: str) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None or row["ended_at"] is not None:
            return
        row["ended_at"] = _now()
        row["end_reason"] = reason
        advance_conversation_generation(self._backend, scope, str(session_id), reason)

    def reopen(self, session_id: SessionId) -> None:
        from .transcript_store import RESET_END_REASONS

        scope = self._backend.scope
        self.retire_undrained_queue_rows(session_id)
        table = self._table(scope)
        row = table.get(session_id)
        if row is None:
            return
        if row["end_reason"] in RESET_END_REASONS and row["session_key"]:
            for child in table.values():
                if _legacy_reset_child(row, child):
                    child["model_config"]["_reset_from"] = row["session_id"]
                    sync_lineage_markers(child)
        row["ended_at"] = None
        row["end_reason"] = None
        self._unarchive_auto_archived_lineage(table, str(session_id))

    def retire_undrained_queue_rows(self, session_id: SessionId) -> int:
        scope = self._backend.scope
        transcript = self._transcripts(scope).get(session_id)
        retired = 0
        for stored in transcript.rows if transcript is not None else ():
            metadata = stored.message.display_metadata or {}
            if stored.visibility is RowVisibility.ACTIVE and stored.message.role == "user" and metadata.get(_QUEUED_PROMPT_KEY) == 1:
                # Deactivated like ``deactivate_message``: the counters still include the row.
                stored.visibility = RowVisibility.REWOUND
                transcript.counter_skew_messages += 1
                transcript.counter_skew_tool_calls += len(stored.message.tool_calls)
                retired += 1
        return retired

    def promote_to_reset(self, session_id: SessionId, reason: str = END_REASON_SESSION_RESET) -> bool:
        scope = self._backend.scope
        if not session_id:
            return False
        try:
            row = self._table(scope).get(session_id)
            if row is None or (row["ended_at"] is not None and row["end_reason"] not in RECOVERABLE_END_REASONS):
                return False
            row["ended_at"] = _now()
            row["end_reason"] = reason
            advance_conversation_generation(self._backend, scope, str(session_id), reason)
        except Exception:
            return False
        return True

    def _has_continuation_child(self, table: dict[str, dict[str, Any]], session_id: str) -> bool:
        """Whether a child continues the row: no fork marker naming it, not a tool row."""
        return any(
            other["parent_session_id"] == session_id
            and other["source"] != "tool"
            and session_id not in [other["model_config"].get(marker) for marker in _MARKER_BY_KIND.values()]
            for other in table.values()
        )

    def reopen_if_explicitly_closed(
        self, session_id: SessionId, *, provenance: str, patience_s: Optional[float] = None
    ) -> Optional[str]:
        # Reference writes never wait for admission, so every patience admits at once; the first attempt runs even when the deadline has passed.
        scope = self._backend.scope
        if not session_id:
            return None
        table = self._table(scope)
        with self._flags_lock:
            row = table.get(session_id)
            if row is None or row["ended_at"] is None:
                return None
            reason = row["end_reason"]
            if not is_explicit_close_reason(reason) or self._has_continuation_child(table, str(session_id)):
                return None
            # One atomic transition under the lock: the stamp cleared is exactly the stamp read.
            row["ended_at"] = None
            row["end_reason"] = None
            return str(reason)

    def touch_activity(
        self,
        session_id: SessionId,
        at: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> None:
        scope = self._backend.scope
        if not session_id:
            return
        stamp = float(at if at is not None else time.time())
        with self._flags_lock:
            row = self._table(scope).get(session_id)
            if row is None:
                return
            # Monotonic: only a strictly newer stamp lands, and the labels ride only with it.
            current = row["last_activity_at"]
            if current is not None and stamp <= current:
                return
            row["last_activity_at"] = stamp
            row["last_activity_description"] = bound_activity_description(description)
            row["last_activity_provenance"] = normalize_activity_provenance(provenance)

    def clear_activity_labels(self, session_id: SessionId) -> None:
        scope = self._backend.scope
        if not session_id:
            return
        with self._flags_lock:
            row = self._table(scope).get(session_id)
            if row is None:
                return
            provenance = row["last_activity_provenance"]
            if not row["last_activity_description"] and (not provenance or provenance == ACTIVITY_PROVENANCE_UNKNOWN):
                return
            # The stamp stays: idle and watchdog clocks are continuous across the clear.
            row["last_activity_description"] = ""
            row["last_activity_provenance"] = ACTIVITY_PROVENANCE_UNKNOWN

    # ── reads ─────────────────────────────────────────────────────────────

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        return None if row is None else self._record(scope, row)

    def records_created_surface(self) -> bool:
        return True

    def last_active(self, session_id: SessionId) -> Optional[float]:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        return None if row is None else self._last_active(scope, row)

    def owns_session(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        return row is not None and row.get("profile_name") in (None, self._backend.owner_profile)

    def store_owner(self) -> Optional[str]:
        return self._backend.owner_profile

    def resolve_id(self, id_or_prefix: str) -> Optional[SessionId]:
        scope = self._backend.scope
        table = self._table(scope)
        if id_or_prefix in table:
            return SessionId(id_or_prefix)
        # The store's prefix LIKE folds ASCII case.
        needle = _ascii_lower(id_or_prefix)
        matches = [sid for sid in table if _ascii_lower(sid).startswith(needle)]
        return SessionId(matches[0]) if len(matches) == 1 else None

    def _resume_lineage_ids(self, table: dict[str, dict[str, Any]], session_id: str) -> list[str]:
        """The rows a display resume materializes: the session and its verified compression ancestors (each hop's parent ended by compression and the child is no fork of it)."""
        row = table.get(session_id)
        if row is None or self._is_fork_child_row(row):
            return [session_id]
        chain = [session_id]
        while len(chain) < 100 and not self._is_fork_child_row(row, include_reset=True):
            parent = table.get(row["parent_session_id"]) if row["parent_session_id"] else None
            if parent is None or parent["end_reason"] != END_REASON_COMPRESSION or parent["session_id"] in chain:
                break
            chain.append(parent["session_id"])
            row = parent
        return chain

    def is_kanban_owned(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        table = self._table(scope)
        rows = [table[sid] for sid in self._resume_lineage_ids(table, str(session_id)) if sid in table]
        return any("kanban" in (row["source"], row.get("created_source")) for row in rows)

    def declared_scope_identity(self, session_id: SessionId) -> DeclaredScopeIdentity:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return DeclaredScopeIdentity(is_fork_child=False, source="")
        return DeclaredScopeIdentity(is_fork_child=self._is_fork_child_row(row), source=str(row["source"] or "").strip())


register_repository("sessions", InMemorySessionRepository)
