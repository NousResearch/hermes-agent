"""CLI-only session-store acquisition; PostgreSQL never falls back to SQLite."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from state_store import MessageRecord, open_state_store, resolve_state_store_config


_LOCAL_FILTER_CANDIDATE_LIMIT = 1000


class PostgreSQLCLISessionCapabilityError(RuntimeError):
    """A CLI path requested a SessionDB feature not yet ported to PostgreSQL."""


class PostgreSQLCLISessionStore:
    """Strict CLI compatibility facade over the PostgreSQL StateStore contract.

    This is deliberately not a general SessionDB replacement.  Every method is
    either implemented with PostgreSQL semantics or rejected before a fallback
    can open SQLite state.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    # StateStoreInterface conformance members (see state_store_interface.py).
    TITLE_SOURCE_LLM = "llm"

    @staticmethod
    def sanitize_title(title: str | None) -> str | None:
        from state_store_postgresql import PostgreSQLStateStore
        return PostgreSQLStateStore.sanitize_title(title)

    @staticmethod
    def _record(message: Mapping[str, Any]) -> MessageRecord:
        fields = MessageRecord.__dataclass_fields__
        return MessageRecord(**{name: message[name] for name in fields if name in message})

    def create_session(self, session_id: str, source: str, **kwargs: Any) -> str:
        model_config = kwargs.get("model_config")
        if isinstance(model_config, str):
            try:
                model_config = json.loads(model_config)
            except json.JSONDecodeError as exc:
                raise ValueError("CLI session model_config must be JSON") from exc
        metadata = {
            key: value for key, value in {
                "model": kwargs.get("model"), "model_config": model_config,
                "user_id": kwargs.get("user_id"), "session_key": kwargs.get("session_key"),
                "chat_id": kwargs.get("chat_id"), "chat_type": kwargs.get("chat_type"),
                "thread_id": kwargs.get("thread_id"), "display_name": kwargs.get("display_name"),
                "origin_json": kwargs.get("origin_json"), "parent_session_id": kwargs.get("parent_session_id"),
                "cwd": kwargs.get("cwd"), "profile_name": kwargs.get("profile_name"),
            }.items() if value is not None
        }
        created = self._store.ensure_session(session_id, source, metadata=metadata)
        if "system_prompt" in kwargs:
            self._store.set_system_prompt(session_id, kwargs["system_prompt"])
        return created

    def import_foreign_history(self, origin: Mapping[str, Any], messages: list[Mapping[str, Any]], *, title: str,
                               cwd: str | None, profile: str | None):
        return self._store.import_foreign_history(origin, messages, title=title, cwd=cwd, profile=profile)

    ensure_session = create_session

    def append_message(
        self, session_id: str, role: str, content: Any = None, tool_name: str | None = None,
        tool_calls: Any = None, tool_call_id: str | None = None, token_count: int | None = None,
        finish_reason: str | None = None, reasoning: str | None = None,
        reasoning_content: str | None = None, reasoning_details: Any = None,
        codex_reasoning_items: Any = None, codex_message_items: Any = None,
        platform_message_id: str | None = None, observed: bool = False,
        effect_disposition: str | None = None, _compressed_summary: bool = False,
        timestamp: Any = None, api_content: str | None = None,
        display_kind: str | None = None, display_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """Persist the SessionDB-compatible single-message CLI surface.

        The facade accepts only fields representable by ``MessageRecord``.  In
        particular, SQLite-specific write-lock controls are rejected before a
        database write rather than being silently ignored or falling back.
        """
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError(
                "PostgreSQL CLI persistence does not support single-message controls: "
                + ", ".join(sorted(kwargs)))
        return self._store.append_message_record(session_id, MessageRecord(
            role=role, content=content, tool_name=tool_name, tool_calls=tool_calls,
            tool_call_id=tool_call_id, token_count=token_count,
            finish_reason=finish_reason, reasoning=reasoning,
            reasoning_content=reasoning_content, reasoning_details=reasoning_details,
            codex_reasoning_items=codex_reasoning_items,
            codex_message_items=codex_message_items,
            platform_message_id=platform_message_id, observed=observed,
            effect_disposition=effect_disposition,
            _compressed_summary=_compressed_summary, timestamp=timestamp,
            api_content=api_content, display_kind=display_kind,
            display_metadata=display_metadata,
        ))

    def append_messages_batch(self, session_id: str, messages: list[Mapping[str, Any]], **kwargs: Any) -> int:
        # The agent's append path carries the holder which owns the surrounding
        # turn/compression lease. PostgreSQL coordinates those leases separately
        # from append-only rows; accepting the metadata here preserves the public
        # SessionDB call shape without pretending that it is an SQLite mutex.
        for name in ("compression_lock_holder", "turn_lease_holder", "turn_lease_ttl_seconds", "chunk_rows"):
            kwargs.pop(name, None)
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError(
                "PostgreSQL CLI persistence does not support batch controls: " + ", ".join(sorted(kwargs)))
        return self._store.append_message_records(session_id, [self._record(message) for message in messages])

    def update_session_meta(self, session_id: str, model_config_json: str, model: str | None = None) -> None:
        """Replace the session's model config (and fill a missing model) after queued usage is durable.

        ACP's persistence path calls this on every turn save
        (acp_adapter/session.py:336) with ``json.dumps(meta)`` and the live model
        string, so the passthrough must preserve the underlying argument shape.
        """
        return self._store.update_session_meta(session_id, model_config_json, model)

    def replace_messages(self, session_id: str, messages: list[Mapping[str, Any]], *, active_only: bool = False,
                         archive_dropped: bool = False, reject_active_turn_lease: bool = False,
                         **kwargs: Any) -> None:
        """Atomically replace the transcript, mirroring the PG adapter's controls.

        ACP calls with ``active_only=True`` to preserve soft-archived rows
        (acp_adapter/session.py:360): the non-owning agent replaces only the
        active=1 set so pre-compaction archived rows survive the rewrite. Any
        control outside the PostgreSQL contract is rejected before a write
        rather than being silently ignored or falling back.
        """
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError(
                "PostgreSQL CLI persistence does not support replace controls: " + ", ".join(sorted(kwargs)))
        return self._store.replace_messages(session_id, messages, active_only=active_only,
                                            archive_dropped=archive_dropped,
                                            reject_active_turn_lease=reject_active_turn_lease)

    def get_messages_as_conversation(self, session_id: str, *, include_ancestors: bool = False,
                                     repair_alternation: bool = False, include_row_ids: bool = False,
                                     **kwargs: Any) -> list[dict[str, Any]]:
        """Resume projection in OpenAI format.

        ``repair_alternation`` is honored for live-replay callers (ACP resumes
        this list as the agent's live conversation, acp_adapter/session.py:383):
        a durable same-role pair would otherwise re-fire the per-request repair
        on every call. The shared agent-side repair runs here on the already
        projected list; the stored transcript is never mutated, and
        ``include_ancestors``/``include_row_ids`` handling is unchanged.
        """
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError("PostgreSQL CLI resume does not support: " + ", ".join(sorted(kwargs)))
        restored, _display = self._store.get_resume_conversations(session_id)
        if repair_alternation and restored:
            # Lazy import: keeps agent-side tooling out of store-only import paths.
            # The helper only mutates the list in place; its ``agent`` argument is unused.
            from agent.agent_runtime_helpers import repair_message_sequence
            repair_message_sequence(None, restored)
        if not include_row_ids:
            restored = [{key: value for key, value in message.items() if key != "_row_id"} for message in restored]
        if not include_ancestors:
            # The adapter's resume projection is already the only safe model-fed view.
            return restored
        return restored

    def get_resume_conversations(self, session_id: str):
        return self._store.get_resume_conversations(session_id)

    def get_active_message_ids(self, session_id: str):
        return self._store.get_active_message_ids(session_id)

    def rewind_to_message(self, session_id: str, target_message_id: int, **kwargs: Any):
        return self._store.rewind_to_message(session_id, target_message_id, **kwargs)

    def get_rewind_receipt(self, request_id: str):
        return self._store.get_rewind_receipt(request_id)

    def rewind_user_turn(self, session_id: str, user_ordinal: int, **kwargs: Any):
        from hermes_state_rewind import rewind_user_turn
        return rewind_user_turn(self, session_id, user_ordinal, **kwargs)

    # Observation and cooldown coordination are durable PostgreSQL operations.
    # Deliberate compression publication remains unavailable through this facade
    # until its parent/child transaction is ported as one contract.
    def touch_session_activity(self, session_id: str, ts: float | None = None, **kwargs: Any) -> None:
        return self._store.touch_session_activity(session_id, ts, **kwargs)
    def clear_session_activity_labels(self, session_id: str) -> None:
        return self._store.clear_session_activity_labels(session_id)
    def get_compression_failure_cooldown(self, session_id: str):
        return self._store.get_compression_failure_cooldown(session_id)
    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: str | None = None) -> None:
        return self._store.record_compression_failure_cooldown(session_id, cooldown_until, error)
    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        return self._store.clear_compression_failure_cooldown(session_id)
    def get_compression_failure_cooldown_row(self, session_id: str):
        return self._store.get_compression_failure_cooldown_row(session_id)
    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Mapping[str, Any]) -> None:
        return self._store.restore_compression_failure_cooldown_row(session_id, snapshot)

    # Compression coordination is an all-or-nothing PostgreSQL contract. These
    # wrappers deliberately preserve the production adapter's arguments and
    # return values rather than emulating SQLite or offering a partial fallback.
    @property
    def capabilities(self): return self._store.capabilities
    def get_compression_fallback_streak(self, session_id: str): return self._store.get_compression_fallback_streak(session_id)
    def set_compression_fallback_streak(self, session_id: str, streak: int): return self._store.set_compression_fallback_streak(session_id, streak)
    def get_compression_ineffective_count(self, session_id: str): return self._store.get_compression_ineffective_count(session_id)
    def set_compression_ineffective_count(self, session_id: str, count: int): return self._store.set_compression_ineffective_count(session_id, count)
    def get_compression_recovery_deadline(self, session_id: str): return self._store.get_compression_recovery_deadline(session_id)
    def set_compression_recovery_deadline(self, session_id: str, deadline: float): return self._store.set_compression_recovery_deadline(session_id, deadline)
    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None): return self._store.get_session_model_config_value(session_id, key, default)
    def patch_session_model_config(self, session_id: str, patch: Mapping[str, Any]): return self._store.patch_session_model_config(session_id, patch)
    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0): return self._store.try_acquire_compression_lock(session_id, holder, ttl_seconds)
    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0): return self._store.refresh_compression_lock(session_id, holder, ttl_seconds)
    def release_compression_lock(self, session_id: str, holder: str): return self._store.release_compression_lock(session_id, holder)
    def get_compression_lock_holder(self, session_id: str): return self._store.get_compression_lock_holder(session_id)
    def try_acquire_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0, **kwargs: Any): return self._store.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=ttl_seconds, **kwargs)
    def refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0): return self._store.refresh_session_turn_lease(session_id, holder, ttl_seconds=ttl_seconds)
    def release_session_turn_lease(self, session_id: str, holder: str): return self._store.release_session_turn_lease(session_id, holder)
    def get_active_message_watermark(self, session_id: str): return self._store.get_active_message_watermark(session_id)
    def get_compression_publication_receipt(self, request_id: str): return self._store.get_compression_publication_receipt(request_id)
    def publish_compression_child(self, **kwargs: Any): return self._store.publish_compression_child(**kwargs)
    def branch_session(self, **kwargs: Any): return self._store.branch_session(**kwargs)

    def get_session(self, session_id: str): return self._store.get_session(session_id)
    def get_recent_session_model_route(self, session_id: str): return self._store.get_recent_session_model_route(session_id)
    def read_insights_snapshot(self, *, cutoff: float, source: str | None): return self._store.read_insights_snapshot(cutoff=cutoff, source=source)
    def resolve_session_id(self, session_id_or_prefix: str) -> str | None:
        """Resolve an exact ID or one unambiguous PostgreSQL-local prefix."""
        exact = self._store.get_session(session_id_or_prefix)
        if exact:
            return str(exact["id"])
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {self._store._schema}.sessions WHERE id LIKE %s ORDER BY started_at DESC LIMIT 2",
                (f"{session_id_or_prefix}%",),
            )
            matches = cursor.fetchall()
        return str(matches[0][0]) if len(matches) == 1 else None

    def session_count(self, source: str | None = None) -> int:
        """Return the complete PostgreSQL session count for CLI statistics."""
        with self._store._connection() as connection, connection.cursor() as cursor:
            if source is None:
                cursor.execute(f"SELECT COUNT(*) FROM {self._store._schema}.sessions")
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {self._store._schema}.sessions WHERE source=%s", (source,))
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def message_count(self) -> int:
        """Return the complete physical PostgreSQL message count for CLI statistics."""
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._store._schema}.messages")
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def export_session(
        self, session_id: str, include_compacted: bool = False,
    ) -> dict[str, Any] | None:
        """Project one complete active PostgreSQL segment into SessionDB export shape."""
        session = self._store.get_session(session_id)
        if session is None:
            return None
        messages = self._store.get_message_records(
            session_id, include_compacted=include_compacted,
        )
        from hermes_state_portability import _export_timings
        return {**session, "message_count": len(messages), "messages": messages, "timings": _export_timings(messages, session_id)}

    def export_session_lineage(
        self, session_id: str, include_compacted: bool = False,
    ) -> dict[str, Any] | None:
        """Export the complete compression lineage when every segment is available."""
        lineage_ids = self._store.get_compression_lineage(session_id)
        if not lineage_ids:
            return None
        segments = [
            segment for segment in (
                self.export_session(item, include_compacted=include_compacted)
                for item in lineage_ids
            ) if segment
        ]
        if not segments:
            return None
        messages = [message for segment in segments for message in segment["messages"]]
        from hermes_state_portability import _export_timings
        return {
            **segments[-1], "segments": segments, "lineage_session_ids": [segment["id"] for segment in segments],
            "message_count": len(messages), "messages": messages,
            "timings": _export_timings(messages, session_id),
        }
    def get_compression_tip(self, session_id: str): return self._store.get_compression_tip(session_id)
    def get_conversation_root(self, session_id: str): return self._store.get_conversation_root(session_id)
    def get_compression_lineage(self, session_id: str): return self._store.get_compression_lineage(session_id)
    def assert_resume_safe(self, session_id: str, *args: Any, **kwargs: Any): return self._store.assert_resume_safe(session_id, *args, **kwargs)
    def get_session_title(self, session_id: str): return self._store.get_session_title(session_id)
    def get_session_title_source(self, session_id: str): return self._store.get_session_title_source(session_id)
    def set_session_title_source(self, session_id: str, source: str): return self._store.set_session_title_source(session_id, source)
    def set_session_title(self, session_id: str, title: str): return self._store.set_session_title(session_id, title)
    def set_auto_title(self, session_id: str, title: str, *, source: str): return self._store.set_auto_title(session_id, title, source=source)
    def set_session_pinned(self, session_id: str, pinned: bool): return self._store.set_session_pinned(session_id, pinned)
    def get_session_by_title(self, title: str): return self._store.get_session_by_title(title)
    def resolve_session_by_title(self, title: str): return self._store.resolve_session_by_title(title)
    def session_lifecycle_statuses(self, session_ids: list[str]): return self._store.session_lifecycle_statuses(session_ids)
    def list_skill_scaffolded_sessions(self, limit: int = 200): return self._store.list_skill_scaffolded_sessions(limit)
    def list_gateway_sessions(self, *, platform: str | None = None, active_only: bool = True):
        return self._store.list_gateway_sessions(platform=platform, active_only=active_only)
    # The inline public session_search consumer receives this CLI facade from
    # AIAgent.  These are the complete read-only contextual primitives already
    # implemented by its selected PostgreSQL store, not SQLite emulation.
    def get_messages(self, session_id: str): return self._store.get_messages(session_id)
    def get_messages_around(self, session_id: str, around_message_id: int, *, window: int = 5): return self._store.get_messages_around(session_id, around_message_id, window=window)
    def get_anchored_view(self, session_id: str, around_message_id: int, *, window: int, bookend: int): return self._store.get_anchored_view(session_id, around_message_id, window=window, bookend=bookend)
    def get_message_storage_state(self, message_id: int): return self._store.get_message_storage_state(message_id)
    def search_messages(self, *args: Any, **kwargs: Any): return self._store.search_messages(*args, **kwargs)
    def list_recent_sessions_bounded(self, *, limit: int, exclude_sources: list[str], timeout_seconds: float): return self._store.list_recent_sessions_bounded(limit=limit, exclude_sources=exclude_sources, timeout_seconds=timeout_seconds)
    def search_index_status(self): return self._store.search_index_status()
    def rebuild_search_index(self): return self._store.rebuild_search_index()
    def get_next_title_in_lineage(self, title: str): return self._store.get_next_title_in_lineage(title)

    def _prune_where(self, older_than_days: float | None, source: str | None, filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
        """Tenant-scoped PG equivalent of SessionMaintenanceMixin's bounded filter grammar."""
        allowed = {
            "last_active_before", "last_active_after", "started_before", "started_after", "source", "title_like",
            "end_reason", "cwd_prefix", "min_messages", "max_messages", "model_like", "provider", "user_id",
            "chat_id", "chat_type", "branch_like", "min_tokens", "max_tokens", "min_cost", "max_cost",
            "min_tool_calls", "max_tool_calls", "archived", "include_pinned", "lineage_tips_only",
        }
        unknown = set(filters) - allowed
        if unknown:
            raise TypeError(f"unexpected PostgreSQL maintenance filter: {sorted(unknown)[0]}")
        message_count = "(SELECT COUNT(*) FROM " + self._store._schema + ".messages message WHERE message.session_id=s.id)"
        tool_call_count = "(SELECT COUNT(*) FROM " + self._store._schema + ".messages tool_message WHERE tool_message.session_id=s.id AND tool_message.role='tool')"
        last_active = "COALESCE(s.last_activity_at, (SELECT MAX(m.created_at) FROM " + self._store._schema + ".messages m WHERE m.session_id=s.id), s.started_at)"
        clauses, params = ["s.ended_at IS NOT NULL"], []
        values = dict(filters)
        if values.get("lineage_tips_only", False):
            clauses.append("COALESCE(s.end_reason, '') <> 'compression'")
        if older_than_days is not None and values.get("last_active_before") is None and values.get("started_before") is None:
            values["last_active_before"] = time.time() - float(older_than_days) * 86400
        def add(clause: str, value: Any, enabled: bool = True) -> None:
            if enabled:
                clauses.append(clause); params.append(value)
        add(last_active + " < %s", values.get("last_active_before"), values.get("last_active_before") is not None)
        if values.get("last_active_before") is not None:
            clauses.append("(COALESCE(s.end_reason, '') != 'startup_orphan_reap' OR s.ended_at < %s)")
            params.append(values["last_active_before"])
        add(last_active + " >= %s", values.get("last_active_after"), values.get("last_active_after") is not None)
        add("s.started_at < %s", values.get("started_before"), values.get("started_before") is not None)
        add("s.started_at >= %s", values.get("started_after"), values.get("started_after") is not None)
        for key, column in (("source", "source"), ("end_reason", "end_reason"), ("provider", "billing_provider"), ("user_id", "user_id"), ("chat_id", "chat_id"), ("chat_type", "chat_type")):
            value = values.get(key)
            add((f"LOWER(COALESCE(s.{column}, '')) = LOWER(%s)" if key == "provider" else f"s.{column} = %s"), value, bool(value))
        for key, column in (("title_like", "title"), ("model_like", "model"), ("branch_like", "git_branch")):
            value = values.get(key)
            add(f"LOWER(COALESCE(s.{column}, '')) LIKE %s ESCAPE '\\\\'", "%" + str(value).lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", bool(value))
        value = values.get("cwd_prefix")
        add("(s.cwd = %s OR s.cwd LIKE %s ESCAPE '\\\\')", value, bool(value))
        if value:
            params.append(str(value).rstrip("/") + "/%")
        for key, expression in (("min_messages", message_count + " >= %s"), ("max_messages", message_count + " <= %s"), ("min_tokens", "(COALESCE(s.input_tokens,0)+COALESCE(s.output_tokens,0)) >= %s"), ("max_tokens", "(COALESCE(s.input_tokens,0)+COALESCE(s.output_tokens,0)) <= %s"), ("min_cost", "COALESCE(s.actual_cost_usd,s.estimated_cost_usd,0) >= %s"), ("max_cost", "COALESCE(s.actual_cost_usd,s.estimated_cost_usd,0) <= %s"), ("min_tool_calls", tool_call_count + " >= %s"), ("max_tool_calls", tool_call_count + " <= %s")):
            add(expression, values.get(key), values.get(key) is not None)
        if isinstance(values.get("archived"), bool):
            clauses.append("s.archived = %s"); params.append(values["archived"])
        if not values.get("include_pinned", False):
            clauses.append("NOT s.pinned")
        return " AND ".join(clauses), params

    def list_prune_candidates(self, older_than_days: float | None = None, source: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        where, params = self._prune_where(older_than_days, source, filters)
        query = f"SELECT s.id, s.source, s.title, s.model, s.started_at, COALESCE(s.last_activity_at, (SELECT MAX(m.created_at) FROM {self._store._schema}.messages m WHERE m.session_id=s.id), s.started_at) AS last_active, s.ended_at, (SELECT COUNT(*) FROM {self._store._schema}.messages message WHERE message.session_id=s.id) AS message_count, s.archived FROM {self._store._schema}.sessions s WHERE {where} ORDER BY last_active ASC, s.started_at ASC, s.id ASC LIMIT 10000"
        with self._store._connection() as connection, connection.cursor(row_factory=self._store._psycopg.rows.dict_row) as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())

    def count_prune_matches(self, older_than_days: float | None = None, source: str | None = None, **filters: Any) -> int:
        where, params = self._prune_where(older_than_days, source, filters)
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._store._schema}.sessions s WHERE {where}", params)
            return int(cursor.fetchone()[0])

    def count_open_prune_matches(self, older_than_days: float | None = None, source: str | None = None, **filters: Any) -> int:
        where, params = self._prune_where(older_than_days, source, filters)
        where = where.replace("s.ended_at IS NOT NULL", "s.ended_at IS NULL", 1)
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {self._store._schema}.sessions s WHERE {where}", params)
            return int(cursor.fetchone()[0])

    def prune_sessions(self, older_than_days: float | None = 90, source: str | None = None, *, sessions_dir: Any = None, **filters: Any) -> int:
        del sessions_dir
        where, params = self._prune_where(older_than_days, source, filters)
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT s.id FROM {self._store._schema}.sessions s WHERE {where} FOR UPDATE", params)
            doomed = [str(row[0]) for row in cursor.fetchall()]
            if not doomed:
                return 0
            # Bulk retention must not inherit explicit-delete's delegate cascade: delete exactly the
            # ended, filtered roots and detach every surviving child in this same transaction.
            cursor.execute(
                f"DELETE FROM {self._store._schema}.compression_rotation_receipts WHERE parent_session_id = ANY(%s) OR child_session_id = ANY(%s)",
                (doomed, doomed),
            )
            cursor.execute(f"DELETE FROM {self._store._schema}.messages WHERE session_id = ANY(%s)", (doomed,))
            cursor.execute(f"UPDATE {self._store._schema}.sessions SET parent_session_id=NULL WHERE parent_session_id = ANY(%s) AND NOT (id = ANY(%s))", (doomed, doomed))
            cursor.execute(f"DELETE FROM {self._store._schema}.sessions WHERE id = ANY(%s)", (doomed,))
            cursor.execute(f"DELETE FROM {self._store._schema}.system_prompts prompt WHERE NOT EXISTS (SELECT 1 FROM {self._store._schema}.sessions session WHERE session.system_prompt_hash=prompt.hash)")
            return len(doomed)

    def archive_sessions(self, older_than_days: float | None = None, source: str | None = None, **filters: Any) -> int:
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(older_than_days, source, **filters)
        # set_session_archived owns the compression-lineage transaction and lock; do not split
        # that canonical operation into a facade-side UPDATE.
        for row in rows:
            self._store.set_session_archived(str(row["id"]), True)
        return len(rows)

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
        if backup:
            # SQLite's file snapshot is intentionally non-equivalent; PG changes are transactionally atomic.
            backup_path = None
        else:
            backup_path = None
        from hermes_state import _STALE_TOOL_CALL_MARKER_RE
        with self._store._connection() as connection, connection.cursor(row_factory=self._store._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT id, content FROM {self._store._schema}.messages WHERE role='assistant' AND tool_calls IS NOT NULL")
            ids = [int(row["id"]) for row in cursor.fetchall() if isinstance(row["content"], str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(row["content"].strip())]
            if ids and not dry_run:
                cursor.execute(f"UPDATE {self._store._schema}.messages SET content='' WHERE id = ANY(%s)", (ids,))
        return {"dry_run": dry_run, "rows_affected": len(ids), "row_ids": ids, "backup_path": backup_path}

    def update_session_billing_route(self, session_id: str, **kwargs: Any): return self._store.update_session_billing_route(session_id, **kwargs)
    def queue_token_counts(self, session_id: str, **kwargs: Any): return self._store.queue_token_counts(session_id, **kwargs)
    def flush_token_counts(self, timeout: float = 5.0): return self._store.flush_token_counts(timeout)
    def record_auxiliary_usage(self, session_id: str, task: str, **kwargs: Any): return self._store.record_auxiliary_usage(session_id, task, **kwargs)
    def update_system_prompt(self, session_id: str, prompt: str | None): return self._store.set_system_prompt(session_id, prompt)
    def clear_stored_system_prompts(self): return self._store.clear_stored_system_prompts()
    def reopen_session(self, session_id: str) -> None:
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"UPDATE {self._store._schema}.sessions SET ended_at=NULL, end_reason=NULL WHERE id=%s", (session_id,))
    def end_session(self, session_id: str, reason: str): return self._store.end_session(session_id, reason)
    def list_sessions_rich(
        self, source: str | None = None, sources: list[str] | None = None,
        exclude_sources: list[str] | None = None, cwd_prefix: str | None = None,
        limit: int = 20, offset: int = 0, include_children: bool = False,
        min_message_count: int = 0, project_compression_tips: bool = True,
        order_by_last_active: bool = False, include_archived: bool = False,
        archived_only: bool = False, id_query: str | None = None,
        search_query: str | None = None, compact_rows: bool = False,
        include_pinned: bool = False, session_key: str | None = None,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        """Project PostgreSQL summary rows into the bounded CLI listing contract.

        The state-store owns the canonical source/archive/hidden filters and MRU
        order.  This facade adds only SessionDB's presentation and local CLI
        filters, then resolves compression heads to their resumable descendants.
        """
        if sources is not None or cwd_prefix is not None or min_message_count or id_query is not None:
            raise PostgreSQLCLISessionCapabilityError(
                "PostgreSQL CLI session listing does not support these filters")
        if limit < 0 or offset < 0:
            raise ValueError("session list limit and offset must be non-negative")
        search = (search_query or "").strip().lower()
        # Source/archive/hidden/pinned filters are backend predicates, so they
        # must be applied before pagination.  Bare /resume has no local filter:
        # fetch only its requested page and hydrate only those rows.  Title/ID
        # and workspace filters require post-fetch inspection, which stays
        # explicitly bounded rather than imposing that scan on every listing.
        local_filter = bool(search or session_key)
        candidate_limit = limit + offset
        if local_filter:
            candidate_limit = max(candidate_limit, _LOCAL_FILTER_CANDIDATE_LIMIT)
        rows = self._store.list_session_summaries(
            source=source, exclude_sources=tuple(exclude_sources or ()),
            limit=candidate_limit, offset=0,
            include_archived=include_archived, archived_only=archived_only,
            include_hidden=include_hidden, include_pinned=include_pinned,
        )
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for summary in rows:
            original_id = str(summary["id"])
            session_id = self._store.get_compression_tip(original_id) if project_compression_tips and not include_children else original_id
            if session_id in seen:
                continue
            session = self._store.get_session(session_id)
            if session is None:
                continue
            title = str(session.get("title") or "")
            if search and search not in session_id.lower() and search not in title.lower():
                continue
            if session_key and session.get("git_repo_root") != session_key and session.get("cwd") != session_key:
                continue
            messages = self._store.get_message_records(session_id)
            preview_raw = next((row.get("content") for row in messages if row.get("role") == "user" and row.get("content") is not None), "")
            from hermes_state_sessions import _shape_preview
            preview = _shape_preview(preview_raw)
            row = {**session, **summary, "id": session_id, "preview": preview,
                   "last_active": session.get("last_active", summary.get("last_active")),
                   "message_count": len(messages)}
            seen.add(session_id)
            result.append(row)
        if order_by_last_active or search:
            result.sort(key=lambda row: (row.get("last_active") or 0, row.get("started_at") or 0, row["id"]), reverse=True)
        else:
            result.sort(key=lambda row: (row.get("started_at") or 0, row["id"]), reverse=True)
        page = result[offset:offset + limit]
        if not include_pinned:
            return page
        # SessionDB's include_pinned contract is a page plus every pinned row
        # that the page missed. The backend summary query already back-fills
        # those rows; do not discard them with the final presentation slice.
        page_ids = {row["id"] for row in page}
        return [*page, *(row for row in result if row.get("pinned") and row["id"] not in page_ids)]

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Return the PostgreSQL compression tip that existing resume APIs read."""
        return self._store.get_compression_tip(session_id)

    def search_sessions(self, source: str | None = None, limit: int = 20, offset: int = 0,
                        workspace_key: str | None = None, **kwargs: Any):
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError("PostgreSQL CLI session search does not support: " + ", ".join(sorted(kwargs)))
        rows = self._store.list_session_summaries(
            source=source, limit=max(limit + offset, 1000), offset=0,
            include_archived=True, include_hidden=True, include_pinned=True)
        if workspace_key:
            # CLI's resume scoping is an exact workspace identity (git root or CWD),
            # not a path-prefix search.  Resolve it in PostgreSQL before slicing.
            with self._store._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM {self._store._schema}.sessions WHERE git_repo_root=%s OR cwd=%s",
                    (workspace_key, workspace_key))
                allowed = {row[0] for row in cursor.fetchall()}
            rows = [row for row in rows if row["id"] in allowed]
        return rows[offset:offset + limit]
    def _delegate_delete_ids(self, cursor: Any, session_id: str) -> list[str]:
        """Return recursively tagged delegate descendants, excluding *session_id*.

        The recursive CTE uses ``UNION`` deliberately: malformed delegation cycles
        terminate, and the root can never become its own descendant.  Untagged
        branch/compression children remain outside this list and are orphaned.
        """
        schema = self._store._schema
        cursor.execute(
            f"WITH RECURSIVE delegates(id) AS ("
            f"SELECT id FROM {schema}.sessions WHERE id=%s "
            "UNION "
            f"SELECT child.id FROM {schema}.sessions child "
            "JOIN delegates parent ON ("
            "child.model_config ->> '_delegate_from' = parent.id "
            "OR (child.parent_session_id = parent.id "
            "AND child.model_config ? '_delegate_from'))"
            ") SELECT id FROM delegates WHERE id <> %s ORDER BY id",
            (session_id, session_id),
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def get_session_delete_targets(self, session_id: str) -> list[str]:
        """Rows an explicit delete would remove, matching SessionDB semantics."""
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {self._store._schema}.sessions WHERE id=%s",
                (session_id,),
            )
            if cursor.fetchone() is None:
                return []
            return [session_id, *self._delegate_delete_ids(cursor, session_id)]

    def _delete_session_ids(self, cursor: Any, root_ids: list[str]) -> list[str]:
        """Delete existing roots and their delegates in the caller's transaction."""
        schema = self._store._schema
        cursor.execute(
            f"SELECT id FROM {schema}.sessions WHERE id = ANY(%s) FOR UPDATE",
            (root_ids,),
        )
        existing_roots = [str(row[0]) for row in cursor.fetchall()]
        if not existing_roots:
            return []
        doomed = set(existing_roots)
        for root_id in existing_roots:
            doomed.update(self._delegate_delete_ids(cursor, root_id))
        doomed_ids = sorted(doomed)
        # Lock the entire doomed set before changing dependencies. A concurrent
        # child insertion needs a conflicting FK lock on its parent.
        cursor.execute(f"SELECT id FROM {schema}.sessions WHERE id = ANY(%s) FOR UPDATE", (doomed_ids,))
        cursor.execute(
            f"DELETE FROM {schema}.compression_rotation_receipts "
            "WHERE parent_session_id = ANY(%s) OR child_session_id = ANY(%s)",
            (doomed_ids, doomed_ids),
        )
        cursor.execute(f"DELETE FROM {schema}.messages WHERE session_id = ANY(%s)", (doomed_ids,))
        cursor.execute(
            f"UPDATE {schema}.sessions SET parent_session_id=NULL "
            "WHERE parent_session_id = ANY(%s) AND NOT (id = ANY(%s))",
            (doomed_ids, doomed_ids),
        )
        cursor.execute(f"DELETE FROM {schema}.sessions WHERE id = ANY(%s)", (doomed_ids,))
        cursor.execute(
            f"DELETE FROM {schema}.system_prompts prompt WHERE NOT EXISTS ("
            f"SELECT 1 FROM {schema}.sessions session "
            "WHERE session.system_prompt_hash = prompt.hash)"
        )
        return existing_roots

    def delete_session_if_empty(self, session_id: str, *, sessions_dir: Any = None, **kwargs: Any) -> bool:
        if kwargs:
            raise PostgreSQLCLISessionCapabilityError("PostgreSQL CLI deletion does not support these controls")
        # PostgreSQL owns all durable session data; the SQLite caller supplies
        # sessions_dir only to remove legacy transcript files, so it is a strict
        # compatibility no-op in this backend.
        del sessions_dir
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {self._store._schema}.sessions session WHERE id=%s "
                "AND title IS NULL AND NOT EXISTS (SELECT 1 FROM "
                f"{self._store._schema}.messages message WHERE message.session_id=session.id) "
                "AND NOT EXISTS (SELECT 1 FROM "
                f"{self._store._schema}.sessions child WHERE child.parent_session_id=session.id) FOR UPDATE",
                (session_id,),
            )
            if cursor.fetchone() is None:
                return False
            return bool(self._delete_session_ids(cursor, [session_id]))

    def delete_session(self, session_id: str, **kwargs: Any) -> bool:
        unsupported = set(kwargs) - {"sessions_dir", "expected_delete_ids"}
        if unsupported:
            raise PostgreSQLCLISessionCapabilityError("PostgreSQL CLI deletion does not support these controls")
        expected = kwargs.get("expected_delete_ids")
        if expected is not None and not isinstance(expected, list):
            raise PostgreSQLCLISessionCapabilityError("expected_delete_ids must be a list")
        # PostgreSQL owns all durable session data; sessions_dir is a strict
        # compatibility no-op and no filesystem data is removed in this mode.
        with self._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {self._store._schema}.sessions WHERE id=%s FOR UPDATE",
                (session_id,),
            )
            if cursor.fetchone() is None:
                return False
            current = {session_id, *self._delegate_delete_ids(cursor, session_id)}
            if expected is not None and current != set(expected):
                return False
            return bool(self._delete_session_ids(cursor, [session_id]))
    def close(self): self._store.close()

    def __getattr__(self, name: str):
        raise PostgreSQLCLISessionCapabilityError(
            f"PostgreSQL CLI session store does not implement '{name}'; no SQLite fallback is permitted")


def open_cli_session_store(config: Mapping[str, Any], *, read_only: bool = False) -> Any:
    """Acquire the selected CLI store without changing SQLite's legacy path."""
    resolved = resolve_state_store_config(config)
    if resolved.backend == "sqlite":
        from hermes_state import SessionDB
        if read_only:
            return SessionDB(read_only=True)
        # CLI's normal writer shares the registry handle with its REPL follow-up
        # paths, preserving the SQLite runtime's single-open behavior.
        from hermes_state_registry import acquire
        return acquire()
    return PostgreSQLCLISessionStore(open_state_store(config))


def open_selected_read_store(config: Mapping[str, Any]) -> Any | None:
    """Open the configured read store, preserving SQLite's empty-install behavior.

    PostgreSQL selection never consults ``state.db``; failures propagate to the command surface.
    """
    resolved = resolve_state_store_config(config)
    if resolved.backend == "sqlite":
        from hermes_state import _default_db_path
        if not _default_db_path().exists():
            return None
    return open_cli_session_store(config, read_only=True)
