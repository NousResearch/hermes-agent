"""Backend-neutral state-store configuration and bootstrap boundary.

This module intentionally does not expose database connections.  It is the
first incremental seam for moving SQLite-owned state behind a contract while
keeping existing SQLite installations operational without configuration changes.
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Protocol, cast, runtime_checkable

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_BACKENDS = frozenset({"sqlite", "postgresql"})


class StateStoreConfigurationError(ValueError):
    """A selected state-store backend has unsafe or unusable configuration."""


@dataclass(frozen=True)
class PostgreSQLStateStoreConfig:
    """Non-secret PostgreSQL connection policy; the DSN is never retained here."""

    dsn_env: str
    connect_timeout_seconds: int
    pool_max_size: int


@dataclass(frozen=True)
class ResolvedStateStoreConfig:
    """Selected backend and non-secret connection policy.

    The DSN stays in the secret resolver and must not be logged, repr'd, or
    propagated as normal configuration data.
    """

    backend: str
    postgresql: PostgreSQLStateStoreConfig | None = None


@dataclass(frozen=True)
class MessageRecord:
    """Canonical appendable transcript row for the bounded StateStore record slice.

    This intentionally covers active-message persistence and ordered record reads;
    rewrite, compaction, lineage, and conversation projections remain SessionDB-only.
    ``timestamp`` is caller-supplied when present. Structured content, tool calls,
    and display metadata retain SessionDB's decoded Python representation.
    """

    role: str
    content: Any = None
    tool_call_id: str | None = None
    tool_calls: Any = None
    tool_name: str | None = None
    effect_disposition: str | None = None
    timestamp: Any = None
    token_count: int | None = None
    finish_reason: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: Any = None
    codex_reasoning_items: Any = None
    codex_message_items: Any = None
    platform_message_id: str | None = None
    observed: bool = False
    _compressed_summary: bool = False
    api_content: str | None = None
    display_kind: str | None = None
    display_metadata: dict[str, Any] | None = None


class ContextualSessionSearchStore(Protocol):
    """Read-only contextual recall contract used by ``session_search``.

    This is deliberately separate from the incremental write-oriented
    ``StateStore`` contract: recall needs lineage-aware anchors, bounded browse,
    and index-health reporting as one atomic capability.  Backends must not
    advertise it until every method is implemented.
    """

    def get_session(self, session_id: str) -> dict[str, Any] | None: ...
    def get_messages(self, session_id: str) -> list[dict[str, Any]]: ...
    def get_messages_around(self, session_id: str, around_message_id: int, *, window: int) -> dict[str, Any]: ...
    def get_anchored_view(self, session_id: str, around_message_id: int, *, window: int, bookend: int) -> dict[str, Any]: ...
    def get_message_storage_state(self, message_id: int) -> dict[str, Any] | None: ...
    def search_messages(
        self, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
        include_inactive: bool = False, fields: Collection[str] | None = None,
        after_ts: int | None = None, before_ts: int | None = None,
    ) -> list[dict[str, Any]]: ...
    def resolve_session_by_title(self, title: str) -> str | None: ...
    def list_recent_sessions_bounded(
        self, *, limit: int, exclude_sources: list[str], timeout_seconds: float,
    ) -> list[dict[str, Any]]: ...
    def search_index_status(self) -> dict[str, Any] | None: ...
    def rebuild_search_index(self) -> dict[str, Any] | None: ...
    def close(self) -> None: ...


class ContextualSessionSearchUnavailable(RuntimeError):
    """The selected backend has not implemented the complete recall contract."""


@runtime_checkable
class CompressionCoordinationStore(Protocol):
    """Durable observation, cooldown, counter, and lease primitives.

    This intentionally excludes parent/child compression publication.  A backend
    may expose this API without claiming that destructive rotation is available.
    """
    def touch_session_activity(self, session_id: str, ts: float | None = None, **kwargs: Any) -> None: ...
    def clear_session_activity_labels(self, session_id: str) -> None: ...
    def get_compression_failure_cooldown(self, session_id: str) -> dict[str, Any] | None: ...
    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: str | None = None) -> None: ...
    def clear_compression_failure_cooldown(self, session_id: str) -> None: ...
    def get_compression_failure_cooldown_row(self, session_id: str) -> dict[str, Any]: ...
    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Mapping[str, Any]) -> None: ...
    def get_compression_fallback_streak(self, session_id: str) -> int: ...
    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None: ...
    def get_compression_ineffective_count(self, session_id: str) -> int: ...
    def set_compression_ineffective_count(self, session_id: str, count: int) -> None: ...
    def get_compression_recovery_deadline(self, session_id: str) -> float: ...
    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None: ...
    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool: ...
    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool: ...
    def release_compression_lock(self, session_id: str, holder: str) -> None: ...
    def try_acquire_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0, **kwargs: Any) -> bool: ...
    def refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0) -> bool: ...
    def release_session_turn_lease(self, session_id: str, holder: str) -> None: ...


class StateStore(Protocol):
    """Incremental session/message/title/visibility contract; broader SessionDB APIs stay out of scope."""

    def ensure_session(
        self, session_id: str, source: str = "unknown", *, metadata: Mapping[str, Any] | None = None,
    ) -> str: ...

    def append_message(self, session_id: str, *, role: str, content: str | None = None) -> int: ...

    def append_message_record(self, session_id: str, record: MessageRecord) -> int: ...

    def append_message_records(self, session_id: str, records: list[MessageRecord]) -> int: ...

    def get_message_records(self, session_id: str) -> list[dict[str, Any]]: ...

    def get_messages(self, session_id: str) -> list[dict[str, Any]]: ...

    def search_messages(
        self, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
        include_inactive: bool = False, fields: Collection[str] | None = None,
        after_ts: int | None = None, before_ts: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_compression_tip(self, session_id: str) -> str | None: ...

    def get_compression_lineage(self, session_id: str) -> list[str]: ...

    def get_conversation_root(self, session_id: str) -> str: ...

    def get_resume_conversations(self, session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...

    def get_ancestor_display_prefix(self, session_id: str) -> list[dict[str, Any]]: ...

    def get_resume_message_count(self, session_id: str, *, tip_only: bool = False) -> int: ...

    def assert_resume_safe(self, session_id: str, max_messages: int | None = None, *, tip_only: bool = False) -> int: ...

    def end_session(self, session_id: str, end_reason: str) -> None: ...

    def promote_to_session_reset(self, session_id: str, reason: str = "session_reset") -> bool: ...

    def latest_conversation_boundary(self, session_key: str, source: str) -> int | None: ...

    def queue_token_counts(self, session_id: str, **kwargs: Any) -> None: ...

    def flush_token_counts(self, timeout: float = 5.0) -> bool: ...

    def update_token_counts(self, session_id: str, input_tokens: int = 0, output_tokens: int = 0, model: str | None = None, cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0, estimated_cost_usd: float | None = None, actual_cost_usd: float | None = None, cost_status: str | None = None, cost_source: str | None = None, pricing_version: str | None = None, billing_provider: str | None = None, billing_base_url: str | None = None, billing_mode: str | None = None, api_call_count: int = 0, absolute: bool = False, source: str | None = None) -> None: ...

    def record_auxiliary_usage(self, session_id: str, task: str, **kwargs: Any) -> None: ...

    def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    def get_recent_session_model_route(self, session_id: str) -> dict[str, Any] | None: ...

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: str | None = None,
        git_repo_root: str | None = None, replace_git_meta: bool = False,
    ) -> int | None: ...

    def publish_session_git_metadata(
        self, session_id: str, cwd: str, generation: int, git_branch: str | None = None,
        git_repo_root: str | None = None,
    ) -> bool: ...

    def update_session_meta(self, session_id: str, model_config_json: str, model: str | None = None) -> None: ...

    def update_session_model(
        self, session_id: str, model: str, provider: str | None = None, *,
        base_url: str | None = None, api_mode: str | None = None,
    ) -> None: ...

    def patch_session_model_config(self, session_id: str, patch: Mapping[str, Any]) -> None: ...

    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any: ...

    def update_session_billing_route(self, session_id: str, *, provider: str, base_url: str, billing_mode: str | None = None) -> None: ...

    def set_system_prompt(self, session_id: str, system_prompt: str | None) -> None: ...

    def get_system_prompt(self, session_id: str) -> str | None: ...

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool: ...

    def set_session_archived(self, session_id: str, archived: bool) -> bool: ...

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool: ...

    def list_session_summaries(
        self, *, source: str | None = None, exclude_sources: tuple[str, ...] = (),
        limit: int = 20, offset: int = 0, include_archived: bool = False,
        archived_only: bool = False, include_hidden: bool = False,
        include_pinned: bool = False,
    ) -> list[dict[str, Any]]: ...

    def set_session_title(self, session_id: str, title: str) -> bool: ...

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool: ...

    def get_session_title(self, session_id: str) -> str | None: ...

    def get_session_title_source(self, session_id: str) -> str | None: ...

    def set_session_title_source(self, session_id: str, source: str) -> bool: ...

    def get_session_by_title(self, title: str) -> dict[str, Any] | None: ...

    def resolve_session_by_title(self, title: str) -> str | None: ...

    def get_next_title_in_lineage(self, base_title: str) -> str: ...

    def close(self) -> None: ...


def _scoped_secret(name: str) -> str | None:
    """Use Hermes's profile-aware secret scope rather than borrowing another profile's env."""
    try:
        from hermes_cli.config import _env_ref_lookup
    except Exception:
        return os.environ.get(name)
    return _env_ref_lookup(name)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise StateStoreConfigurationError(f"{path} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StateStoreConfigurationError(f"{path} must be a positive integer") from exc
    if parsed <= 0:
        raise StateStoreConfigurationError(f"{path} must be a positive integer")
    return parsed


def resolve_state_store_config(
    config: Mapping[str, Any], *, secret_lookup: Callable[[str], str | None] | None = None,
) -> ResolvedStateStoreConfig:
    """Resolve and validate state-store policy without retaining the DSN secret."""
    raw = config.get("state_store", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise StateStoreConfigurationError("state_store must be a mapping")
    backend = str(raw.get("backend", "sqlite")).strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise StateStoreConfigurationError(f"state_store.backend must be one of: {allowed}")
    if backend == "sqlite":
        return ResolvedStateStoreConfig(backend="sqlite")

    pg = raw.get("postgresql", {})
    if not isinstance(pg, Mapping):
        raise StateStoreConfigurationError("state_store.postgresql must be a mapping")
    dsn_env = str(pg.get("dsn_env", "")).strip()
    if not _ENV_NAME.fullmatch(dsn_env):
        raise StateStoreConfigurationError("state_store.postgresql.dsn_env must name an environment variable")
    settings = PostgreSQLStateStoreConfig(
        dsn_env=dsn_env,
        connect_timeout_seconds=_positive_int(pg.get("connect_timeout_seconds", 10), "state_store.postgresql.connect_timeout_seconds"),
        pool_max_size=_positive_int(pg.get("pool_max_size", 8), "state_store.postgresql.pool_max_size"),
    )
    lookup = secret_lookup or _scoped_secret
    if not str(lookup(settings.dsn_env) or "").strip():
        raise StateStoreConfigurationError(
            f"PostgreSQL state store requires secret {settings.dsn_env}; configure it in the active profile secret scope")
    return ResolvedStateStoreConfig(backend="postgresql", postgresql=settings)


class SqliteStateStore:
    """Narrow adapter over the existing SessionDB compatibility facade."""

    def __init__(self, db_path: Path | None = None) -> None:
        from hermes_state import SessionDB

        self._session_db = SessionDB() if db_path is None else SessionDB(db_path=db_path)

    def ensure_session(
        self, session_id: str, source: str = "unknown", *, metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return self._session_db.ensure_session(session_id, source=source, **dict(metadata or {}))

    def append_message(self, session_id: str, *, role: str, content: str | None = None) -> int:
        if content is None:
            return self._session_db.append_message(session_id, role=role)
        return self._session_db.append_message(session_id, role=role, content=content)

    @staticmethod
    def _record_kwargs(record: MessageRecord) -> dict[str, Any]:
        return {
            key: getattr(record, key) for key in (
                "role", "content", "tool_call_id", "tool_calls", "tool_name", "effect_disposition",
                "timestamp", "token_count", "finish_reason", "reasoning", "reasoning_content",
                "reasoning_details", "codex_reasoning_items", "codex_message_items", "platform_message_id",
                "observed", "_compressed_summary", "api_content", "display_kind", "display_metadata",
            )
        }

    def append_message_record(self, session_id: str, record: MessageRecord) -> int:
        return self._session_db.append_message(session_id, **self._record_kwargs(record))

    def append_message_records(self, session_id: str, records: list[MessageRecord]) -> int:
        return self._session_db.append_messages_batch(
            session_id, [self._record_kwargs(record) for record in records])

    def get_message_records(self, session_id: str) -> list[dict[str, Any]]:
        records = []
        for row in self._session_db.get_messages(session_id):
            record = dict(row)
            record.pop("display_identity", None)
            record.pop("display_order", None)
            record["observed"] = bool(record["observed"])
            record["active"] = bool(record["active"])
            record["compacted"] = bool(record["compacted"])
            record["_compressed_summary"] = bool(record.pop("_compressed_summary", False))
            records.append(record)
        return records

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._session_db.get_messages(session_id)

    def search_messages(
        self, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
        include_inactive: bool = False, fields: Collection[str] | None = None,
        after_ts: int | None = None, before_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._session_db.search_messages(
            query, source_filter=source_filter, exclude_sources=exclude_sources, role_filter=role_filter,
            limit=limit, offset=offset, sort=sort, include_inactive=include_inactive, fields=fields,
            after_ts=after_ts, before_ts=before_ts,
        )

    def get_compression_tip(self, session_id: str) -> str:
        return self._session_db.get_compression_tip(session_id)

    def get_compression_lineage(self, session_id: str) -> list[str]:
        return self._session_db.get_compression_lineage(session_id)

    def get_conversation_root(self, session_id: str) -> str:
        return self._session_db.get_conversation_root(session_id)

    def get_resume_conversations(self, session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return self._session_db.get_resume_conversations(session_id)

    def get_ancestor_display_prefix(self, session_id: str) -> list[dict[str, Any]]:
        return self._session_db.get_ancestor_display_prefix(session_id)

    def get_resume_message_count(self, session_id: str, *, tip_only: bool = False) -> int:
        return self._session_db.get_resume_message_count(session_id, tip_only=tip_only)

    def assert_resume_safe(self, session_id: str, max_messages: int | None = None, *, tip_only: bool = False) -> int:
        return self._session_db.assert_resume_safe(session_id, max_messages, tip_only=tip_only)

    def end_session(self, session_id: str, end_reason: str) -> None:
        self._session_db.end_session(session_id, end_reason)

    def promote_to_session_reset(self, session_id: str, reason: str = "session_reset") -> bool:
        return self._session_db.promote_to_session_reset(session_id, reason)

    def latest_conversation_boundary(self, session_key: str, source: str) -> int | None:
        return self._session_db.latest_conversation_boundary(session_key, source)

    # The current SQLite path remains the reference implementation of this
    # backend-neutral non-destructive coordination contract.
    def touch_session_activity(self, session_id: str, ts: float | None = None, **kwargs: Any) -> None: return self._session_db.touch_session_activity(session_id, ts, **kwargs)
    def clear_session_activity_labels(self, session_id: str) -> None: return self._session_db.clear_session_activity_labels(session_id)
    def get_compression_failure_cooldown(self, session_id: str): return self._session_db.get_compression_failure_cooldown(session_id)
    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: str | None = None) -> None: return self._session_db.record_compression_failure_cooldown(session_id, cooldown_until, error)
    def clear_compression_failure_cooldown(self, session_id: str) -> None: return self._session_db.clear_compression_failure_cooldown(session_id)
    def get_compression_failure_cooldown_row(self, session_id: str): return self._session_db.get_compression_failure_cooldown_row(session_id)
    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Mapping[str, Any]) -> None: return self._session_db.restore_compression_failure_cooldown_row(session_id, dict(snapshot))
    def get_compression_fallback_streak(self, session_id: str) -> int: return self._session_db.get_compression_fallback_streak(session_id)
    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None: return self._session_db.set_compression_fallback_streak(session_id, streak)
    def get_compression_ineffective_count(self, session_id: str) -> int: return self._session_db.get_compression_ineffective_count(session_id)
    def set_compression_ineffective_count(self, session_id: str, count: int) -> None: return self._session_db.set_compression_ineffective_count(session_id, count)
    def get_compression_recovery_deadline(self, session_id: str) -> float: return self._session_db.get_compression_recovery_deadline(session_id)
    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None: return self._session_db.set_compression_recovery_deadline(session_id, deadline)
    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool: return self._session_db.try_acquire_compression_lock(session_id, holder, ttl_seconds)
    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool: return self._session_db.refresh_compression_lock(session_id, holder, ttl_seconds)
    def release_compression_lock(self, session_id: str, holder: str) -> None: return self._session_db.release_compression_lock(session_id, holder)
    def try_acquire_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0, **kwargs: Any) -> bool: return self._session_db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=ttl_seconds, **kwargs)
    def refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0) -> bool: return self._session_db.refresh_session_turn_lease(session_id, holder, ttl_seconds=ttl_seconds)
    def release_session_turn_lease(self, session_id: str, holder: str) -> None: return self._session_db.release_session_turn_lease(session_id, holder)

    def queue_token_counts(self, session_id: str, **kwargs: Any) -> None:
        self._session_db.queue_token_counts(session_id, **kwargs)

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        return self._session_db.flush_token_counts(timeout)

    def update_token_counts(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        self._session_db.update_token_counts(session_id, *args, **kwargs)

    def record_auxiliary_usage(self, session_id: str, task: str, **kwargs: Any) -> None:
        self._session_db.record_auxiliary_usage(session_id, task, **kwargs)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._session_db.get_session(session_id)
        if row is not None:
            for key in ("hidden", "archived", "pinned"):
                row[key] = bool(row.get(key))
        return row

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: str | None = None,
        git_repo_root: str | None = None, replace_git_meta: bool = False,
    ) -> int | None:
        return self._session_db.update_session_cwd(
            session_id, cwd, git_branch, git_repo_root, replace_git_meta)

    def publish_session_git_metadata(
        self, session_id: str, cwd: str, generation: int, git_branch: str | None = None,
        git_repo_root: str | None = None,
    ) -> bool:
        return self._session_db.publish_session_git_metadata(
            session_id, cwd, generation, git_branch, git_repo_root)

    def update_session_meta(self, session_id: str, model_config_json: str, model: str | None = None) -> None:
        self._session_db.update_session_meta(session_id, model_config_json, model)

    def update_session_model(
        self, session_id: str, model: str, provider: str | None = None, *,
        base_url: str | None = None, api_mode: str | None = None,
    ) -> None:
        self._session_db.update_session_model(
            session_id, model, provider, base_url=base_url, api_mode=api_mode)

    def patch_session_model_config(self, session_id: str, patch: Mapping[str, Any]) -> None:
        self._session_db.patch_session_model_config(session_id, dict(patch))

    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._session_db.get_session_model_config_value(session_id, key, default)

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: str | None = None,
    ) -> None:
        self._session_db.update_session_billing_route(
            session_id, provider=provider, base_url=base_url, billing_mode=billing_mode)

    def set_system_prompt(self, session_id: str, system_prompt: str | None) -> None:
        self._session_db.update_system_prompt(session_id, system_prompt)

    def get_system_prompt(self, session_id: str) -> str | None:
        row = self._session_db.get_session(session_id)
        return None if row is None else row.get("system_prompt")

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        return self._session_db.set_session_hidden(session_id, hidden)

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        return self._session_db.set_session_archived(session_id, archived)

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        return self._session_db.set_session_pinned(session_id, pinned)

    def list_session_summaries(
        self, *, source: str | None = None, exclude_sources: tuple[str, ...] = (),
        limit: int = 20, offset: int = 0, include_archived: bool = False,
        archived_only: bool = False, include_hidden: bool = False,
        include_pinned: bool = False,
    ) -> list[dict[str, Any]]:
        """Narrow list contract retaining SessionDB's filters, MRU order, and pin back-fill."""
        rows = self._session_db.list_sessions_rich(
            source=cast(Any, source), exclude_sources=cast(Any, list(exclude_sources) or None), limit=limit, offset=offset,
            include_archived=include_archived, archived_only=archived_only,
            include_hidden=include_hidden, include_pinned=include_pinned,
            include_children=True, project_compression_tips=False, order_by_last_active=True, compact_rows=True,
        )
        keys = (
            "id", "source", "started_at", "ended_at", "end_reason", "parent_session_id", "title",
            "title_source", "hidden", "archived", "pinned", "last_active",
        )
        return [
            {
                **{key: row.get(key) for key in keys},
                "hidden": bool(row.get("hidden")), "archived": bool(row.get("archived")),
                "pinned": bool(row.get("pinned")), "message_count": len(self._session_db.get_messages(row["id"])),
            }
            for row in rows
        ]

    def set_session_title(self, session_id: str, title: str) -> bool:
        return self._session_db.set_session_title(session_id, title)

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        return self._session_db.set_auto_title(session_id, title, source=source)

    def get_session_title(self, session_id: str) -> str | None:
        return self._session_db.get_session_title(session_id)

    def get_session_title_source(self, session_id: str) -> str | None:
        return self._session_db.get_session_title_source(session_id)

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        return self._session_db.set_session_title_source(session_id, source)

    def get_session_by_title(self, title: str) -> dict[str, Any] | None:
        return self._session_db.get_session_by_title(title)

    def resolve_session_by_title(self, title: str) -> str | None:
        return self._session_db.resolve_session_by_title(title)

    def get_next_title_in_lineage(self, base_title: str) -> str:
        return self._session_db.get_next_title_in_lineage(base_title)

    def close(self) -> None:
        self._session_db.close()


class SqliteContextualSessionSearchStore:
    """Complete SQLite implementation of :class:`ContextualSessionSearchStore`.

    ``SessionDB`` remains the owner of SQLite lifecycle/WAL locking.  This
    adapter is the only place the legacy private message visibility lookup is
    expressed, so the public tool no longer reaches into a backend connection.
    """

    def __init__(self, session_db=None, *, db_path: Path | None = None, read_only: bool = False) -> None:
        if session_db is None:
            from hermes_state import SessionDB
            session_db = SessionDB(db_path=db_path, read_only=read_only) if db_path is not None else SessionDB()
        self._session_db = session_db

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._session_db.get_session(session_id)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._session_db.get_messages(session_id)

    def get_messages_around(self, session_id: str, around_message_id: int, *, window: int) -> dict[str, Any]:
        return self._session_db.get_messages_around(session_id, around_message_id, window=window)

    def get_anchored_view(self, session_id: str, around_message_id: int, *, window: int, bookend: int) -> dict[str, Any]:
        return self._session_db.get_anchored_view(session_id, around_message_id, window=window, bookend=bookend)

    def get_message_storage_state(self, message_id: int) -> dict[str, Any] | None:
        with self._session_db._lock:
            row = self._session_db._conn.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return dict(row) if row else None

    def search_messages(self, query: str, source_filter: list[str] | None = None,
                        exclude_sources: list[str] | None = None, role_filter: list[str] | None = None,
                        limit: int = 20, offset: int = 0, sort: str | None = None,
                        include_inactive: bool = False, fields: Collection[str] | None = None,
                        after_ts: int | None = None, before_ts: int | None = None) -> list[dict[str, Any]]:
        return self._session_db.search_messages(
            query, source_filter=source_filter, exclude_sources=exclude_sources, role_filter=role_filter,
            limit=limit, offset=offset, sort=sort, include_inactive=include_inactive, fields=fields,
            after_ts=after_ts, before_ts=before_ts,
        )

    def resolve_session_by_title(self, title: str) -> str | None:
        return self._session_db.resolve_session_by_title(title)

    def list_recent_sessions_bounded(self, *, limit: int, exclude_sources: list[str],
                                     timeout_seconds: float) -> list[dict[str, Any]]:
        bounded = getattr(self._session_db, "list_recent_sessions_bounded", None)
        if bounded is None:
            raise RuntimeError("session database does not support bounded recent-session browse")
        return bounded(
            limit=limit, exclude_sources=exclude_sources, timeout_seconds=timeout_seconds)

    def search_index_status(self) -> dict[str, Any] | None:
        return self._session_db.fts_rebuild_status()

    def rebuild_search_index(self) -> dict[str, Any] | None:
        self._session_db.rebuild_fts()
        return self.search_index_status()

    def close(self) -> None:
        self._session_db.close()


def contextual_session_search_store(session_db=None, *, db_path: Path | None = None,
                                    read_only: bool = False, backend: str = "sqlite") -> ContextualSessionSearchStore:
    """Resolve the complete recall capability, refusing partial backends.

    PostgreSQL's current lexical StateStore slice intentionally has no recall
    implementation; returning it here would expose partial histories and break
    lineage/compaction semantics.
    """
    # AIAgent supplies its selected PostgreSQL CLI facade to the inline public
    # tool.  Identify it from the backend-owned health payload, then admit the
    # same complete read-only contract rather than wrapping it as SQLite.
    if backend == "sqlite" and session_db is not None:
        search_status = getattr(session_db, "search_index_status", None)
        if callable(search_status):
            status = search_status()
            if isinstance(status, Mapping) and status.get("backend") == "postgresql":
                backend = "postgresql"
    if backend == "postgresql":
        required = (
            "get_session", "get_messages", "get_messages_around", "get_anchored_view",
            "get_message_storage_state", "search_messages", "resolve_session_by_title",
            "list_recent_sessions_bounded", "search_index_status", "rebuild_search_index", "close",
        )
        candidate = session_db
        missing = [name for name in required if not callable(getattr(candidate, name, None))]
        if missing:
            raise ContextualSessionSearchUnavailable(
                "PostgreSQL state store does not implement contextual session search: missing " + ", ".join(missing))
        assert candidate is not None
        status = candidate.search_index_status()
        if not status.get("available"):
            raise ContextualSessionSearchUnavailable(
                "PostgreSQL contextual session search is unavailable: generated-search health is not valid")
        return cast(ContextualSessionSearchStore, session_db)
    if backend != "sqlite":
        raise ContextualSessionSearchUnavailable(
            f"state-store backend '{backend}' does not implement contextual session search")
    if isinstance(session_db, SqliteContextualSessionSearchStore):
        return session_db
    return SqliteContextualSessionSearchStore(session_db, db_path=db_path, read_only=read_only)


@contextmanager
def _contextual_profile_home(home: Path):
    """Temporarily bind backend acquisition to one canonical profile home.

    The ContextVar override is task-local; unlike changing ``HERMES_HOME`` it
    cannot redirect another multiplexed request while this resolver acquires a
    named profile's store.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _canonical_contextual_profile(profile: str | None) -> tuple[str | None, Path]:
    """Return the canonical target identity and home without accepting caller paths.

    ``None`` is the caller's already-bound local profile.  A supplied selector
    is explicit application routing: ``root``/``global`` and ``default`` all
    name the installation-root profile, while named profiles are registry
    entries.  The selector is never a PostgreSQL identifier or permission.
    """
    if profile is None:
        from hermes_constants import get_hermes_home

        return None, get_hermes_home().expanduser().resolve(strict=True)
    from hermes_cli import profiles as profiles_mod

    requested = str(profile).strip()
    canonical_profile = "default" if requested.casefold() in {"root", "global"} else profiles_mod.normalize_profile_name(requested)
    profiles_mod.validate_profile_name(canonical_profile)
    if not profiles_mod.profile_exists(canonical_profile):
        raise ValueError(f"profile '{canonical_profile}' does not exist")
    return canonical_profile, profiles_mod.get_profile_dir(canonical_profile).expanduser().resolve(strict=True)


def _profile_contextual_home(profile: str | None) -> Path:
    """Compatibility helper returning only the canonical contextual home."""
    return _canonical_contextual_profile(profile)[1]


def _profile_state_store_config(home: Path) -> Mapping[str, Any]:
    """Read exactly one profile's config, rejecting malformed config instead of defaulting.

    ``load_config()`` targets the ambient profile and its cache, so it is not a
    safe cross-profile resolver.  This deliberately reads the resolved home
    directly and leaves defaults (including SQLite) to ``resolve_state_store_config``.
    """
    config_path = home / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StateStoreConfigurationError(f"cannot read state-store config for {home}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise StateStoreConfigurationError("profile config.yaml must be a mapping")
    return raw


def resolve_contextual_session_search_store(
    *, profile: str | None = None, session_db=None, session_db_factory: Callable[[], Any] | None = None,
    read_only: bool = False,
) -> ContextualSessionSearchStore:
    """Resolve contextual recall through the requested profile's actual backend.

    A named profile is identified only through the profile registry; neither a
    caller-supplied database path nor a caller-supplied PostgreSQL schema is
    accepted.  PostgreSQL is acquired under that canonical home so its tenant
    selection remains the trusted ``postgresql_tenant_schema`` path, then is
    admitted only when its complete contextual and generated-search contracts
    are healthy.
    """
    target_profile, home = _canonical_contextual_profile(profile)
    if target_profile is not None and (session_db is not None or session_db_factory is not None):
        raise ValueError("an explicit contextual target profile cannot use an injected session database")
    # The current agent already acquired its selected store through the CLI
    # factory.  Preserve that tenant/secret decision for the inline tool rather
    # than resolving configuration a second time; explicit named targets remain
    # resolver-owned below and cannot inject a caller handle.
    if target_profile is None and session_db is not None:
        return contextual_session_search_store(session_db, read_only=read_only)
    config = _profile_state_store_config(home)
    resolved = resolve_state_store_config(
        config,
        secret_lookup=lambda name: _profile_secret_lookup(home, name),
    )
    if resolved.backend == "sqlite":
        if session_db is None and session_db_factory is not None:
            session_db = session_db_factory()
        if session_db is not None:
            return contextual_session_search_store(session_db, read_only=read_only)
        return contextual_session_search_store(db_path=home / "state.db", read_only=read_only)

    # Do not infer a fallback from the presence of state.db.  Acquiring this
    # backend is intentional: it proves the configured tenant is selected by
    # canonical profile identity before capability is considered.
    with _contextual_profile_home(home):
        store = open_state_store(config, secret_lookup=lambda name: _profile_secret_lookup(home, name))
    try:
        return contextual_session_search_store(store, backend=resolved.backend, read_only=read_only)
    except Exception:
        store.close()
        raise


def _profile_secret_lookup(home: Path, name: str) -> str | None:
    """Read only the selected profile's secret scope, never ambient process secrets."""
    from agent.secret_scope import build_profile_secret_scope

    return build_profile_secret_scope(home).get(name)


def postgresql_tenant_schema() -> str:
    """Return the identifier-safe schema for the already resolved Hermes profile.

    The profile identity comes only from the active ``HERMES_HOME`` resolution,
    never from StateStore caller metadata or a configuration value.  Hashing the
    canonical home and canonical profile name makes identifiers deterministic and
    keeps even unusual profile names out of SQL text.
    """
    from hermes_constants import get_hermes_home, profile_name_for_home

    home = get_hermes_home().resolve()
    profile_name = profile_name_for_home(home)
    # The formerly global schema is deliberately the root/default compatibility
    # tenant only. Named profiles never acquire it, so a shared DSN cannot expose
    # legacy root rows to a named profile.
    if profile_name == "default":
        return "hermes_state_store_slice"
    digest = hashlib.sha256(f"{home}\0{profile_name}".encode("utf-8")).hexdigest()[:32]
    return f"hermes_state_store_tenant_{digest}"


def open_state_store(
    config: Mapping[str, Any], *, db_path: Path | None = None,
    secret_lookup: Callable[[str], str | None] | None = None,
) -> StateStore:
    """Open the selected narrow State Store; PostgreSQL never falls back to SQLite."""
    resolved = resolve_state_store_config(config, secret_lookup=secret_lookup)
    if resolved.backend == "sqlite":
        return SqliteStateStore(db_path=db_path)
    assert resolved.postgresql is not None
    dsn = (secret_lookup or _scoped_secret)(resolved.postgresql.dsn_env)
    if not str(dsn or "").strip():
        raise StateStoreConfigurationError(
            f"PostgreSQL state store requires secret {resolved.postgresql.dsn_env}; configure it in the active profile secret scope")
    from state_store_postgresql import PostgreSQLStateStore

    return PostgreSQLStateStore(resolved.postgresql, str(dsn), schema=postgresql_tenant_schema())
