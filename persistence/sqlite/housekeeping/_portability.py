"""Logical export and import of sessions: the SessionMaintenance verbs over whole session rows and their messages.

``SessionPortability`` is mixed into the SQLite ``SessionMaintenance``; its verbs run on ``self._store``, the bound
store handle. The import validation, the row writer and the parent re-attachment are module functions here; the
session, lineage and message bodies they share are the conversation store's (``conversation._store``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import time
from typing import Any, Mapping, Optional, Sequence

from persistence import Conflict, SessionId
from persistence.housekeeping.maintenance import (
    IMPORT_MAX_MESSAGES_PER_SESSION, IMPORT_MAX_SESSION_BYTES, IMPORT_MAX_SESSIONS, IMPORT_MAX_TOTAL_BYTES,
    IMPORT_MAX_TOTAL_MESSAGES, ForeignImportResult, ForeignOrigin, ForeignTurn, ImportResult, SessionExport,
    SessionExportRow, SessionLineageExport,
)
from session_rules.session_ids import new_session_id
from utils import safe_json_loads

from .._backend import translated
from ..conversation._store.compression import get_compression_lineage
from ..conversation._store.messages import (
    _active_clause, _insert_message_rows, _prune_shadowed_checkpoints, _row_to_message_dict, get_messages,
)
from ..conversation._store.messages_sql import _parse_tool_calls, _tool_calls_count
from ..conversation._store.portability import _SQLITE_INT_MAX, _SQLITE_INT_MIN
from ..conversation._store.retirement import retired_in
from ..conversation._store.rules import coerce_epoch
from ..conversation._store.sessions import get_session, search_sessions
from ..conversation.messages import _transcript_row

_IMPORT_SESSION_TEXT_FIELDS = (
    "source", "user_id", "model", "system_prompt", "end_reason", "cwd", "git_branch", "git_repo_root",
    "billing_provider", "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
# ``role`` is validated separately (non-empty string).
_IMPORT_MESSAGE_TEXT_FIELDS = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason",
    "reasoning", "reasoning_content", "platform_message_id", "message_id",
)
_IMPORT_MESSAGE_JSON_FIELDS = (
    "reasoning_details", "codex_reasoning_items", "codex_message_items", "absorbed_message_uids", "tool_call_uids",
)
_IMPORT_SESSION_INSERT_SQL = """INSERT INTO sessions (
                           id, source, user_id, model, model_config, system_prompt,
                           system_prompt_hash,
                           parent_session_id, started_at, ended_at, end_reason,
                           message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived, auto_archived, pinned, hidden
                       )
                       VALUES (
                           :id, :source, :user_id, :model, :model_config,
                           NULL, :system_prompt_hash, NULL, :started_at, :ended_at,
                           :end_reason, 0, 0, :input_tokens, :output_tokens,
                           :cache_read_tokens, :cache_write_tokens,
                           :reasoning_tokens, :cwd, :git_branch, :git_repo_root,
                           :billing_provider, :billing_base_url, :billing_mode,
                           :estimated_cost_usd, :actual_cost_usd, :cost_status,
                           :cost_source, :pricing_version, :title,
                           :api_call_count, :archived, :auto_archived, :pinned, :hidden
                       )"""
# Columns copied verbatim from the payload; typed columns are converted below.
_IMPORT_PASSTHROUGH_COLS = (
    "user_id", "model", "model_config", "end_reason", "cwd", "git_branch", "git_repo_root", "billing_provider",
    "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
_IMPORT_INT_COLS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_call_count",
)
_IMPORT_FLOAT_COLS = ("ended_at", "estimated_cost_usd", "actual_cost_usd")
# Flags a restore must keep: pinned/hidden/auto_archived drive the prune, listings and resume-unarchive.
_IMPORT_FLAG_COLS = ("archived", "auto_archived", "pinned", "hidden")

_EXPORT_COLUMNS = tuple(field.name for field in dataclasses.fields(SessionExportRow)[:-3])


def _export_row(session: dict[str, Any]) -> SessionExportRow:
    """The contract record of one session row dict (the ``get_session`` / ``search_sessions`` shape)."""
    return SessionExportRow(
        **{column: session[column] for column in _EXPORT_COLUMNS}, last_active=session.get("last_active"),
        reset_from=session.get("_reset_from"), branched_from=session.get("_branched_from"),
    )


def _find_foreign_import_on_conn(conn, origin):
    rows = conn.execute("SELECT id, origin_json FROM sessions WHERE source = ? AND origin_json IS NOT NULL",
                        (origin["tool"],)).fetchall()
    for row in rows:
        imported = (safe_json_loads(row["origin_json"], default={}) or {}).get("imported_from", {})
        if imported.get("tool") != origin["tool"]:
            continue
        foreign_id = origin.get("foreign_session_id")
        if ((foreign_id and foreign_id == imported.get("foreign_session_id"))
                or (not foreign_id and imported.get("path") == origin["path"])):
            return row["id"]
    return None


def _import_text_or_none(value: Any, field: str) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{field} must be a string")


def _import_int_or_none(value: Any, field: str) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _import_non_finite_path(value: Any, path: str) -> Optional[str]:
    """First ``path`` of a NaN/inf number anywhere in decoded JSON data, else None.

    Iterative walk (a recursive visitor would overflow on crafted nesting); cycles
    can't arrive through ``json.loads`` but guard anyway.
    """
    pending = [(value, path)]
    while pending:
        current, current_path = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return current_path
        if isinstance(current, dict):
            pending.extend((child, f"{current_path}.{key}") for key, child in current.items())
        elif isinstance(current, list):
            pending.extend((child, f"{current_path}[{key}]") for key, child in enumerate(current))
    return None


def _import_load_json(value: str, path: str) -> Any:
    """``json.loads`` that refuses NaN/Infinity anywhere in the decoded value.

    A JSON exponent like ``1e309`` decodes to ``float("inf")``, which strict
    serializers reject and cost analytics mis-handle. Callers that intentionally
    decode a JSON-backed column (model_config, reasoning items) re-check through
    here so non-finite values can't reappear after the outer payload was inspected.
    """
    parsed = json.loads(value)  # NaN/Infinity are accepted syntax; catch them after decode
    non_finite = _import_non_finite_path(parsed, path)
    if non_finite is not None:
        raise ValueError(f"non-finite number at {non_finite}")
    return parsed


def _import_json_object_or_none(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    path = f"session.{field}"
    if isinstance(value, str):
        try:
            parsed = _import_load_json(value, path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field} must be a JSON object")
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _import_non_finite_path(value, path) is not None:
        raise ValueError(f"non-finite number at {path}")
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc


def _coerce_or(value: Any, cast, default):
    """``cast(value)``; *default* for None or an unparsable value."""
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError):
        return default


def _import_int_or_bounded(value: Any, field: str, default: int) -> int:
    """Import an int column: SQL-int-range or *default* (never fails the import)."""
    number = _coerce_or(value, int, default)
    if number is None:
        return default
    if isinstance(number, bool) or not (_SQLITE_INT_MIN <= number <= _SQLITE_INT_MAX):
        return default
    return number


def _normalize_import_message(message: dict[str, Any], message_index: int) -> dict[str, Any]:
    """Type-check one payload message; raises ValueError naming the offending field."""
    clean_message = dict(message)
    role = clean_message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError(f"messages[{message_index}].role must be a non-empty string")
    for field in _IMPORT_MESSAGE_TEXT_FIELDS:
        clean_message[field] = _import_text_or_none(clean_message.get(field), field)
    token_count = _import_int_or_none(clean_message.get("token_count"), "token_count")
    if token_count is not None and not (_SQLITE_INT_MIN <= token_count <= _SQLITE_INT_MAX):
        raise ValueError("token_count is outside SQLite's integer range")
    clean_message["token_count"] = token_count
    message_path = f"session.messages[{message_index}]"
    timestamp = clean_message.get("timestamp")
    if timestamp is not None and coerce_epoch(timestamp, field="messages[].timestamp") is None:
        raise ValueError(f"{message_path}.timestamp must be a supported Unix timestamp")
    _validate_import_message_content(clean_message.get("content"), message_path)
    for field in _IMPORT_MESSAGE_JSON_FIELDS:
        json_value = clean_message.get(field)
        if isinstance(json_value, str):
            # Malformed JSON text persists verbatim (readers fall back via _json_or);
            # only a PARSED non-finite number is rejected: it would break strict
            # JSON serializers once decoded back out of the row.
            try:
                _import_load_json(json_value, f"{message_path}.{field}")
            except json.JSONDecodeError:
                pass
            except ValueError:
                raise
        elif json_value is not None:
            non_finite = _import_non_finite_path(json_value, f"{message_path}.{field}")
            if non_finite is not None:
                raise ValueError(f"non-finite number at {non_finite}")
    return clean_message


def _validate_import_message_content(content: Any, message_path: str) -> None:
    """Scalar message content must be bindable and later serializable."""
    if isinstance(content, int) and not isinstance(content, bool) and not (
            _SQLITE_INT_MIN <= content <= _SQLITE_INT_MAX):
        # SQLite can't bind a wider int: this reached INSERT and raised OverflowError.
        raise ValueError(f"{message_path}.content is outside SQLite's integer range")
    if isinstance(content, float) and not math.isfinite(content):
        raise ValueError(f"non-finite number at {message_path}.content")
    if isinstance(content, (bytes, bytearray)):
        raise ValueError(f"{message_path}.content must be a string, list, object, finite number, or null")


def _import_float_or_finite(value: Any, field: str) -> Optional[float]:
    """Import a float column: finite float or None (never fails the import).

    A JSON exponent like ``1e309`` decodes to ``float("inf")``; persisted, it later
    breaks strict JSON serialization in the session APIs. Missing stays missing.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _normalize_import_session(raw: dict[str, Any], session_id: str, messages: list) -> dict[str, Any]:
    """Type-check one payload session + its messages; raises ValueError."""
    clean_session = dict(raw)
    clean_session["id"] = session_id
    clean_session["model_config"] = _import_json_object_or_none(clean_session.get("model_config"), "model_config")
    for field in ("parent_session_id", *_IMPORT_SESSION_TEXT_FIELDS):
        clean_session[field] = _import_text_or_none(clean_session.get(field), field)
    started_at = clean_session.get("started_at")
    if started_at is not None and coerce_epoch(started_at, field="started_at") is None:
        raise ValueError("session.started_at must be a supported Unix timestamp")
    for field in _IMPORT_FLOAT_COLS:
        value = clean_session.get(field)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite number at session.{field}")
    for field in _IMPORT_INT_COLS:
        value = clean_session.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and not (
                _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX):
            raise ValueError(f"{field} is outside SQLite's integer range")
    clean_messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        clean_messages.append(_normalize_import_message(message, message_index))
    return {"session": clean_session, "messages": clean_messages}


def _validate_import_payload(sessions: list[dict[str, Any]]) -> tuple:
    """Size/shape/type validation of the whole payload; returns ``(normalized_items,
    errors)``. Every rejected entry is reported."""
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    totals = {"messages": 0, "bytes": 0}
    for index, raw in enumerate(sessions):
        session_id = str(raw.get("id") or "").strip() if isinstance(raw, dict) else ""
        try:
            item = _validate_import_session(raw, session_id, seen_ids, totals)
        except ValueError as exc:
            item = {"index": index, "error": str(exc)}
            if session_id:
                item["session_id"] = session_id
            errors.append(item)
            continue
        seen_ids.add(session_id)
        normalized.append({"index": index, **item})
    return normalized, errors


def _validate_import_session(raw: Any, session_id: str, seen_ids: set, totals: dict[str, int]) -> dict[str, Any]:
    """One payload session -> normalized item; ValueError(message) on rejection. *totals*
    accumulate before their limit check (a rejected oversize entry still counts)."""
    if not isinstance(raw, dict):
        raise ValueError("session must be an object")
    if not session_id:
        raise ValueError("session id is required")
    if session_id in seen_ids:
        raise ValueError("duplicate session id")
    messages = raw.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if len(messages) > IMPORT_MAX_MESSAGES_PER_SESSION:
        raise ValueError("messages exceeds the per-session import limit")
    if any(not isinstance(msg, dict) for msg in messages):
        raise ValueError("messages must contain only objects")
    try:
        # `timings` is derived from the messages at export time and rebuilt on the next export;
        # it must not eat into the size budget of the content it merely describes.
        measured = {k: v for k, v in raw.items() if k != "timings"}
        session_bytes = len(json.dumps(measured, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("session must be JSON serializable") from None
    if session_bytes > IMPORT_MAX_SESSION_BYTES:
        raise ValueError("session exceeds the import size limit")
    totals["bytes"] += session_bytes
    if totals["bytes"] > IMPORT_MAX_TOTAL_BYTES:
        raise ValueError("import exceeds the total size limit")
    item = _normalize_import_session(raw, session_id, messages)
    totals["messages"] += len(item["messages"])
    if totals["messages"] > IMPORT_MAX_TOTAL_MESSAGES:
        raise ValueError("messages exceeds the total import limit")
    return item


def _import_session_row(holder, conn, raw: dict[str, Any], messages: list[dict[str, Any]], session_id: str) -> None:
    """INSERT one normalized session + its messages; counts fixed up after."""
    started_at = coerce_epoch(raw.get("started_at"), session_id=session_id, field="started_at")
    params = {
        "id": session_id, "source": str(raw.get("source") or "import"),
        "system_prompt_hash": holder._store_system_prompt(conn, raw.get("system_prompt")),
        "started_at": time.time() if started_at is None else started_at,
        **{col: 1 if _coerce_or(raw.get(col), int, 0) else 0 for col in _IMPORT_FLAG_COLS},
        **{col: raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS},
        **{col: _import_float_or_finite(raw.get(col), col) for col in _IMPORT_FLOAT_COLS},
        **{col: _import_int_or_bounded(raw.get(col), col, 0) for col in _IMPORT_INT_COLS},
    }
    params["auto_archived"] &= params["archived"]  # only the sweep sets it, and only on rows it archives
    conn.execute(_IMPORT_SESSION_INSERT_SQL, params)
    def _json_value(value: Any) -> Any:
        return safe_json_loads(value, default=value) if isinstance(value, str) else value
    sanitized_messages = [
        {**msg, **{key: _json_value(msg.get(key)) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages
    ]
    # A row exported archived (``include_inactive``) must stay archived: inserted live, compacted or
    # rewound turns would re-enter model context. Flags are coerced like the int session columns, so a
    # hand-edited "0" archives and a missing/null/unparsable flag imports live (older exports have none).
    live: list[dict[str, Any]] = []
    archived: list[dict[str, Any]] = []
    for msg in sanitized_messages:
        (live if _coerce_or(msg.get("active"), int, 1) else archived).append(msg)
    _insert_message_rows(holder, conn, session_id, sanitized_messages, prune_checkpoints=False)
    if archived:
        conn.executemany("UPDATE messages SET active = 0, compacted = ? WHERE id = ?",
                         [(1 if _coerce_or(msg.get("compacted"), int, 0) else 0, msg["_row_id"])
                          for msg in archived])
    # Pruning keys on live rows, so it runs only now: while every row was still live, an archived row's
    # newer checkpoint would strip the newest live one, and archived rows keep theirs as in the donor.
    _prune_shadowed_checkpoints(holder, conn, session_id, live)
    # Session counters count live rows only.
    conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                 (len(live), sum(_tool_calls_count(_parse_tool_calls(msg.get("tool_calls"))) for msg in live),
                  session_id))


def _attach_import_parents(conn, parent_updates: list[tuple]) -> int:
    """Re-attach imported children whose parent exists (in the store or the same payload)
    without creating a cycle; returns the detached count. Only the closing edge of a
    cycle is dropped, so later entries can still attach to the now-root session."""
    parent_by_child = dict(parent_updates)

    def _would_create_cycle(session_id: str, parent_id: str) -> bool:
        seen = {session_id}
        current = parent_id
        while current:
            if current in seen:
                return True
            seen.add(current)
            if current in parent_by_child:
                current = parent_by_child[current]
                continue
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1", (current,),
            ).fetchone()
            if row is None:
                return False
            current = row["parent_session_id"]
        return False

    detached = 0
    for session_id, parent_id in parent_updates:
        parent_exists = conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (parent_id,)).fetchone()
        if parent_exists and not _would_create_cycle(session_id, parent_id):
            conn.execute("UPDATE sessions SET parent_session_id = ? WHERE id = ?", (parent_id, session_id))
        else:
            parent_by_child.pop(session_id, None)
            detached += 1
    return detached


class SessionPortability:
    """Export, import and foreign-import verbs; mixed into the SQLite SessionMaintenance, which sets ``_store``."""

    @staticmethod
    def _with_messages(holder, session: dict[str, Any], include_compacted: bool = False,
                       include_inactive: bool = False) -> SessionExport:
        messages = get_messages(holder, session["id"], include_inactive=include_inactive,
                                include_compacted=include_compacted)
        return SessionExport(session=_export_row(session), messages=tuple(map(_transcript_row, messages)))

    @classmethod
    def _export_session(cls, holder, session_id: SessionId, include_compacted: bool = False,
                        include_inactive: bool = False) -> Optional[SessionExport]:
        session = get_session(holder, session_id)
        return cls._with_messages(holder, session, include_compacted, include_inactive) if session else None

    def export_session(self, session_id: SessionId, include_compacted: bool = False,
                       include_inactive: bool = False) -> Optional[SessionExport]:
        with translated():
            return self._export_session(self._store.holder(), session_id, include_compacted, include_inactive)

    def export_session_lineage(self, session_id: SessionId, include_compacted: bool = False,
                               include_inactive: bool = False) -> Optional[SessionLineageExport]:
        holder = self._store.holder()
        with translated():
            lineage_ids = get_compression_lineage(holder, session_id)
            if not lineage_ids:
                return None
            segments = tuple(seg for seg in (self._export_session(holder, sid, include_compacted, include_inactive)
                                             for sid in lineage_ids) if seg)
        if not segments:
            return None
        messages = tuple(msg for seg in segments for msg in seg.messages)
        return SessionLineageExport(segments=segments, lineage_session_ids=tuple(seg.session.id for seg in segments),
                                    message_count=len(messages), messages=messages)

    @classmethod
    def _export_all(cls, holder, source: Optional[str] = None, include_compacted: bool = False,
                    include_inactive: bool = False) -> tuple[SessionExport, ...]:
        sessions = search_sessions(holder, source=source, limit=100000)
        if include_compacted:
            return tuple(cls._with_messages(holder, session, True, include_inactive) for session in sessions)
        messages_by_session = {session["id"]: [] for session in sessions}
        session_ids = list(messages_by_session)
        active_clause = _active_clause(include_inactive, False)
        # Stay below SQLite's legacy 999-variable limit while replacing the per-session N+1 reads.
        for start in range(0, len(session_ids), 900):
            chunk = session_ids[start:start + 900]
            rows = holder._read_all(
                f"SELECT * FROM messages WHERE session_id IN ({','.join('?' for _ in chunk)}) "
                f"{active_clause} ORDER BY session_id, id",
                chunk,
            )
            for row in rows:
                messages_by_session[row["session_id"]].append(
                    _row_to_message_dict(holder, row, warn_context="get_messages", summary_flag=True)
                )
        return tuple(SessionExport(session=_export_row(session),
                                   messages=tuple(map(_transcript_row, messages_by_session[session["id"]])))
                     for session in sessions)

    def export_all(self, source: Optional[str] = None, include_compacted: bool = False,
                   include_inactive: bool = False) -> tuple[SessionExport, ...]:
        with translated():
            return self._export_all(self._store.holder(), source, include_compacted, include_inactive)

    def import_sessions(self, sessions: Sequence[Mapping[str, Any]]) -> ImportResult:
        if not isinstance(sessions, (list, tuple)):
            raise ValueError("sessions must be a list")
        if len(sessions) > IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {IMPORT_MAX_SESSIONS} entries")
        normalized, errors = _validate_import_payload(list(sessions))
        if errors:
            return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(errors))
        holder = self._store.holder()

        def _do(conn):
            # A user-deleted id stays fenced for good: its row restored beside the fence would list
            # fine and refuse every turn, so the import is refused by name and writes nothing.
            ids = [str(item["session"].get("id") or "").strip() for item in normalized]
            if deleted := [sid for sid in ids if retired_in(conn, sid)]:
                return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(
                    f"session {sid} was deleted here; a deleted session is not restored" for sid in deleted))
            imported_ids: list[SessionId] = []
            skipped_ids: list[SessionId] = []
            parent_updates: list[tuple[str, str]] = []
            for item in normalized:
                raw = item["session"]
                session_id = str(raw.get("id") or "").strip()
                if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
                    skipped_ids.append(SessionId(session_id))
                    continue
                _import_session_row(holder, conn, raw, item["messages"], session_id)
                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(SessionId(session_id))
            return ImportResult(
                ok=True, imported=len(imported_ids), skipped=len(skipped_ids),
                detached=_attach_import_parents(conn, parent_updates),
                imported_ids=tuple(imported_ids), skipped_ids=tuple(skipped_ids),
            )

        with translated():
            return holder._execute_write(_do)

    def find_foreign_import(self, origin: ForeignOrigin) -> Optional[SessionId]:
        with translated():
            with self._store.holder()._read_ctx() as conn:
                found = _find_foreign_import_on_conn(conn, origin.as_dict())
        return None if found is None else SessionId(found)

    def import_foreign_history(self, origin: ForeignOrigin, messages: Sequence[ForeignTurn], *, title: str,
                               cwd: Optional[str], profile: str) -> ForeignImportResult:
        """Adopt or mint a foreign snapshot in one transaction, including its provenance.

        BEGIN IMMEDIATE serializes duplicate clicks across connections/processes.
        Reuse the portability validator and message writer so counters and FTS
        obey the same contract as ordinary transcript imports.
        """
        origin_dict = origin.as_dict()
        turns = [{"role": turn.role, "content": turn.content} for turn in messages]
        session_id = new_session_id(hex_len=12)
        normalized, errors = _validate_import_payload([
            {"id": session_id, "source": origin_dict["tool"], "title": title,
             "cwd": cwd, "messages": turns}])
        if errors:
            raise ValueError(errors[0]["error"])
        holder = self._store.holder()

        def _do(conn):
            existing = _find_foreign_import_on_conn(conn, origin_dict)
            if existing:
                return ForeignImportResult(session_id=SessionId(existing), already_imported=True)
            # Titles are globally unique within a profile. Preserve a readable
            # title while giving unrelated conversations with the same text room.
            item = normalized[0]
            if conn.execute("SELECT 1 FROM sessions WHERE title = ?", (title,)).fetchone():
                item["session"]["title"] = f"{title} ({session_id[-12:]})"
            _import_session_row(holder, conn, item["session"], item["messages"], session_id)
            conn.execute("UPDATE sessions SET origin_json = ?, profile_name = ? WHERE id = ?",
                         (json.dumps({"imported_from": origin_dict}), profile, session_id))
            return ForeignImportResult(session_id=SessionId(session_id), already_imported=False)

        with translated():
            try:
                return holder._execute_write(_do)
            except sqlite3.IntegrityError as exc:
                # The minted id is the primary key: a taken one fails the insert and the write rolls back.
                if "sessions.id" in str(exc):
                    raise Conflict("session", session_id) from exc
                raise
