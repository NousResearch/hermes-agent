"""Supabase memory provider scaffold for Hermes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from agent.memory_provider import MemoryProvider
except Exception:  # pragma: no cover - fallback for repo-only test environments
    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs: Any) -> None: ...

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
            return None

        def sync_turn(
            self, user_content: str, assistant_content: str, *, session_id: str = ""
        ) -> None:
            return None

        @abstractmethod
        def get_tool_schemas(self) -> list[dict[str, Any]]: ...

        def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
            raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

        def shutdown(self) -> None:
            return None

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - fallback for repo-only test environments
    def get_hermes_home() -> Path:
        return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

from .client import SupabaseClient, SupabaseConstraintError
from .config import get_config_schema, is_valid_supabase_url, load_config, save_config

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
_PREFETCH_CHAR_BUDGET = 8000
_PREFETCH_SELECT_LIMIT = 200
_THREAD_JOIN_TIMEOUT_SECONDS = 35.0
_PROVIDER_NOT_READY = "Supabase memory provider is not ready"
_VALID_SOURCE_MODES = {"live", "delegation", "replay", "import", "operator_edit", "system"}
_VALID_WRITER_ROLES = {"agent", "operator", "system", "importer"}
_VALID_TOOL_SCOPES = {"agent_private", "user", "workspace", "global"}
_CREATE_CANDIDATE_OPTION_KEYS = {
    "rem_run_id",
    "session_id",
    "scope",
    "visibility",
    "candidate_scope",
    "subject_entity_id",
    "predicate",
    "object_text",
    "object_json",
    "status",
    "confidence",
    "support_count",
    "explicitness_score",
    "recency_score",
    "source_quality_score",
    "promotion_authority",
    "review_status",
    "reviewed_by",
    "source_agent_profile",
    "source_user_id",
    "source_mode",
    "writer_role",
}
_PREFETCH_FACT_COLUMNS = ",".join(
    [
        "id",
        "profile",
        "workspace",
        "project_id",
        "user_id",
        "scope",
        "visibility",
        "fact_class",
        "fact_type",
        "predicate",
        "object_text",
        "object_json",
        "status",
        "stale_after",
        "expires_at",
        "superseded_by_fact_id",
        "updated_at",
        "created_at",
    ]
)
_PREFETCH_SOURCE_COLUMNS = "fact_id"
_TOOL_SOURCE_COLUMNS = ",".join(
    [
        "id",
        "fact_id",
        "observation_id",
        "source_excerpt",
        "extractor",
        "confidence",
        "support_role",
        "created_at",
    ]
)
_TOOL_OBSERVATION_COLUMNS = "id,content,source_type,source_mode,writer_role,created_at"
_REM_BATCH_OBSERVATION_COLUMNS = ",".join(
    [
        "id",
        "content",
        "scope",
        "session_id",
        "source_type",
        "created_at",
        "metadata",
    ]
)
_ORDER_CREATED_ASC = "created_at.asc"
_ORDER_CREATED_DESC = "created_at.desc"
_ORDER_UPDATED_DESC_THEN_CREATED_DESC = ["updated_at.desc", _ORDER_CREATED_DESC]
_FILTER_STATUS_ACTIVE = "eq.active"
_UTC_OFFSET = "+00:00"
_FACT_CLAIM_PATTERN = re.compile(
    r"^(?P<subject>[A-Za-z][\w -]{0,80}?)\s+"
    r"(?P<verb>uses?|is|are|has|supports?|runs on)\s+"
    r"(?P<object>[^.?!]+)",
    re.IGNORECASE,
)
_TOOL_SCHEMAS = [
    {
        "name": "supabase_memory_search",
        "description": "Search active Supabase memory facts for case-insensitive matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against stored memory facts.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter: agent_private, user, workspace, or global.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "supabase_memory_store",
        "description": "Store a manual observation in Supabase memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Observation content to store.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope for the stored observation: agent_private, user, workspace, or global.",
                },
                "visibility": {
                    "type": "string",
                    "description": "Optional visibility for the stored observation (defaults to private).",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata object. Useful for file-backed memory references such as file_path, repo_path, content_hash, file_role, title, summary, source_uri, and last_indexed_at.",
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "supabase_memory_inspect",
        "description": "Inspect a fact and return its provenance sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact_id": {
                    "type": "string",
                    "description": "Fact identifier to inspect.",
                }
            },
            "required": ["fact_id"],
        },
    },
    {
        "name": "supabase_memory_reflect",
        "description": "Run rule-based reflection over unprocessed observations and stage candidates.",
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter: agent_private, user, workspace, or global.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier to restrict reflection to one session.",
                },
            },
            "required": [],
        },
    },
]


try:
    from tools.registry import tool_error as _tool_error
except Exception:  # pragma: no cover - fallback for repo-only imports
    _tool_error = None


class SupabaseMemoryProvider(MemoryProvider):
    """Scaffold implementation for the Supabase Hermes memory provider."""

    def __init__(self) -> None:
        self._client: SupabaseClient | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_lock = threading.Lock()
        self._rem_thread: threading.Thread | None = None
        self._rem_lock = threading.Lock()
        self._promotion_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_records: list[dict[str, Any]] = []
        self._prefetch_ready = False
        self._prefetch_query = ""
        self._prefetch_session_id = ""
        self._session_id = ""
        self._hermes_home = ""
        self._platform = ""
        self._agent_context = ""
        self._agent_identity = ""
        self._profile = ""
        self._agent_workspace = ""
        self._workspace = ""
        self._project_id = ""
        self._user_id = ""
        self._write_suppressed = False
        self._initialized = False
        self._is_shutdown = False
        self._observation_batch_key_supported = True

    @property
    def name(self) -> str:
        return "supabase"

    def is_available(self) -> bool:
        config = load_config(environ=None)
        supabase_url = config.get("supabase_url", "")
        secret_key = config.get("supabase_secret_key", "")
        return bool(secret_key and is_valid_supabase_url(supabase_url))

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._cleanup_runtime_resources()

        self._session_id = str(session_id or "")
        self._hermes_home = str(kwargs.get("hermes_home") or get_hermes_home())
        self._platform = str(kwargs.get("platform", "") or "")
        self._agent_context = str(kwargs.get("agent_context", "") or "")
        self._profile = str(kwargs.get("agent_identity", "") or "")
        self._agent_identity = self._profile
        self._workspace = str(
            kwargs.get("workspace", kwargs.get("agent_workspace", "")) or ""
        )
        self._agent_workspace = self._workspace
        self._project_id = str(kwargs.get("project_id", "") or "")
        self._user_id = str(kwargs.get("user_id", "") or "")
        self._write_suppressed = self._agent_context in {"cron", "flush"}
        self._initialized = True
        self._is_shutdown = False
        self._observation_batch_key_supported = True

        try:
            config = load_config(hermes_home=self._hermes_home)
            supabase_url = config.get("supabase_url", "")
            secret_key = config.get("supabase_secret_key", "")

            if secret_key and is_valid_supabase_url(supabase_url):
                self._client = SupabaseClient(supabase_url, secret_key)
            else:
                logger.warning(
                    "Failed to initialize Supabase memory client: missing or invalid configuration"
                )
        except Exception as exc:  # pragma: no cover - defensive safety
            logger.warning("Failed to initialize Supabase memory client: %s", exc)
            self._client = None

    def system_prompt_block(self) -> str:
        if not self._initialized or self._is_shutdown:
            return ""

        tool_schemas = self.get_tool_schemas()
        tool_names = [
            str(schema.get("name", "")).strip()
            for schema in tool_schemas
            if isinstance(schema, dict) and schema.get("name")
        ]
        tool_summary = ", ".join(tool_names) if tool_names else "none yet"
        return (
            "Supabase memory provider is active for durable cross-session storage. "
            f"Available tools: {tool_summary}."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._can_prefetch():
            return ""

        prefetch_thread = self._prefetch_thread
        self._join_thread(prefetch_thread)

        cached_result = ""
        cached_records: list[dict[str, Any]] = []
        normalized_query = self._normalize_text(query)
        effective_session_id = session_id or self._session_id
        with self._prefetch_lock:
            if prefetch_thread is not None and self._prefetch_thread is prefetch_thread:
                if prefetch_thread.is_alive():
                    if self._prefetch_ready and self._prefetch_query == normalized_query and self._prefetch_session_id == effective_session_id:
                        return self._prefetch_result
                    return ""
                self._prefetch_thread = None
            if self._prefetch_ready and self._prefetch_query == normalized_query and self._prefetch_session_id == effective_session_id:
                cached_result = self._prefetch_result
                cached_records = [dict(record) for record in self._prefetch_records]

        if cached_records or cached_result:
            self._log_retrieval_events(
                query,
                cached_records,
                retrieval_stage="prompt_selected",
                selected_for_prompt=True,
                reason={"surface": "prefetch"},
                session_id=session_id,
            )
            return cached_result

        result, selected_records = self._fetch_prefetch_result(query, session_id=session_id)
        with self._prefetch_lock:
            self._prefetch_result = result
            self._prefetch_records = [dict(record) for record in selected_records]
            self._prefetch_ready = True
            self._prefetch_query = normalized_query
            self._prefetch_session_id = effective_session_id
        self._log_retrieval_events(
            query,
            selected_records,
            retrieval_stage="prompt_selected",
            selected_for_prompt=True,
            reason={"surface": "prefetch"},
            session_id=session_id,
        )
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        content = self._format_turn_content(user_content, assistant_content)
        if not content:
            return None

        self._queue_observation_write(
            self._build_observation_payload(
                source_type="conversation_turn",
                content=content,
                metadata={},
                session_id=session_id,
            )
        )
        return None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return deepcopy(_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self._is_shutdown:
            return self._tool_error("Supabase memory provider is shut down")

        normalized_args = args if isinstance(args, dict) else {}
        if tool_name == "supabase_memory_search":
            return self._handle_search_tool(normalized_args)
        if tool_name == "supabase_memory_store":
            return self._handle_store_tool(normalized_args)
        if tool_name == "supabase_memory_inspect":
            return self._handle_inspect_tool(normalized_args)
        if tool_name == "supabase_memory_reflect":
            return self._handle_reflect_tool(normalized_args)
        return self._tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        self._cleanup_runtime_resources()
        self._initialized = False
        self._is_shutdown = True

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        return None

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        content = self._format_session_summary(messages)
        if not content:
            return None

        payload = self._build_observation_payload(
            source_type="session_summary",
            content=content,
            metadata={},
        )
        self._queue_observation_write(payload)
        if payload is None or self._write_suppressed:
            return None

        self._queue_session_end_rem(session_id=self._session_id)
        return None

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return ""

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        normalized_task = self._normalize_text(task)
        normalized_result = self._normalize_text(result)
        if not normalized_task and not normalized_result:
            return None

        content_parts: list[str] = []
        if normalized_task:
            content_parts.append(f"Delegated task:\n{normalized_task}")
        if normalized_result:
            content_parts.append(f"Delegation result:\n{normalized_result}")

        self._queue_observation_write(
            self._build_observation_payload(
                source_type="delegation",
                source_mode="delegation",
                content="\n\n".join(content_parts),
                metadata={"child_session_id": self._normalize_text(child_session_id) or None},
            )
        )
        return None

    def get_config_schema(self) -> list[dict[str, Any]]:
        return get_config_schema()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        save_config(values, hermes_home)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action not in {"add", "replace", "remove"}:
            logger.warning("Skipping unsupported memory write action: %s", action)
            return None

        normalized_content = self._normalize_text(content)
        if not normalized_content:
            return None

        self._queue_observation_write(
            self._build_observation_payload(
                source_type="memory_write",
                content=normalized_content,
                metadata={"action": action, "target": target},
            )
        )
        return None

    def create_rem_run(
        self,
        *,
        scope: str = "agent_private",
        mode: str = "manual",
        status: str = "pending",
        trigger: str = "",
        started_at: str | None = None,
        completed_at: str | None = None,
        model: str = "",
        extractor_version: str = "",
        observation_batch_key: str | None = None,
        input_scope: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> dict[str, Any]:
        if not self._can_manage_rem_records():
            return {}

        payload = {
            "profile": self._profile,
            "agent_id": self._agent_identity,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "scope": scope,
            "mode": mode,
            "status": status,
            "trigger": self._normalize_text(trigger) or None,
            "started_at": started_at,
            "completed_at": completed_at,
            "model": self._normalize_text(model) or None,
            "extractor_version": self._normalize_text(extractor_version) or None,
            "input_scope": self._normalize_json_object(input_scope),
            "parameters": self._normalize_json_object(parameters),
            "summary": self._normalize_json_object(summary),
            "error_text": self._normalize_text(error_text) or None,
        }
        normalized_batch_key = self._normalize_text(observation_batch_key) or None
        if normalized_batch_key and self._observation_batch_key_supported:
            payload["observation_batch_key"] = normalized_batch_key

        rem_run, insert_error = self._insert_single_row_with_error("memory_rem_runs", payload)
        if rem_run or not normalized_batch_key or not self._observation_batch_key_supported:
            return rem_run

        if isinstance(insert_error, SupabaseConstraintError) and self._is_observation_batch_key_duplicate_error(insert_error):
            return {}

        if not self._is_missing_observation_batch_key_error(insert_error):
            return {}

        self._observation_batch_key_supported = False
        payload.pop("observation_batch_key", None)
        rem_run, _ = self._insert_single_row_with_error("memory_rem_runs", payload)
        return rem_run

    def get_rem_run(self, rem_run_id: str) -> dict[str, Any]:
        return self._select_single_row("memory_rem_runs", rem_run_id)

    def get_rem_run_by_batch_key(self, observation_batch_key: str) -> dict[str, Any]:
        client = self._client
        normalized_batch_key = self._normalize_text(observation_batch_key)
        if (
            client is None
            or not normalized_batch_key
            or not self._profile
            or not self._observation_batch_key_supported
        ):
            return {}

        try:
            rows = client.select(
                "memory_rem_runs",
                filters={
                    "profile": f"eq.{self._profile}",
                    "observation_batch_key": f"eq.{normalized_batch_key}",
                },
                order=_ORDER_CREATED_DESC,
                limit=1,
            )
        except Exception as exc:
            logger.warning("Failed to load REM run by batch key %s: %s", normalized_batch_key, exc)
            if self._is_missing_observation_batch_key_error(exc):
                self._observation_batch_key_supported = False
            return {}

        if not rows:
            return {}
        return dict(rows[0])

    def store_reflection(
        self,
        *,
        rem_run_id: str,
        scope: str = "agent_private",
        observation_group_key: str = "",
        summary_what_happened: str = "",
        reflections: list[Any] | None = None,
        candidates: list[Any] | None = None,
        lasting_updates: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._can_manage_rem_records():
            return {}

        payload = {
            "rem_run_id": self._normalize_text(rem_run_id),
            "profile": self._profile,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "scope": scope,
            "observation_group_key": self._normalize_text(observation_group_key) or None,
            "summary_what_happened": self._normalize_text(summary_what_happened) or None,
            "reflections": self._normalize_json_array(reflections),
            "candidates": self._normalize_json_array(candidates),
            "lasting_updates": self._normalize_json_array(lasting_updates),
            "metadata": self._normalize_json_object(metadata),
        }
        return self._insert_single_row("memory_reflections", payload)

    def get_reflection(self, reflection_id: str) -> dict[str, Any]:
        return self._select_single_row("memory_reflections", reflection_id)

    def create_candidate(
        self,
        *,
        candidate_class: str,
        candidate_type: str,
        metadata: dict[str, Any] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        if not self._can_manage_rem_records():
            return {}

        self._validate_candidate_options(options)
        payload = self._build_candidate_payload(
            candidate_class=candidate_class,
            candidate_type=candidate_type,
            metadata=metadata,
            options=options,
        )
        return self._insert_single_row("memory_candidates", payload)

    def _validate_candidate_options(self, options: dict[str, Any]) -> None:
        unexpected_options = sorted(set(options) - _CREATE_CANDIDATE_OPTION_KEYS)
        if unexpected_options:
            option_list = ", ".join(unexpected_options)
            raise TypeError(f"Unexpected candidate options: {option_list}")

    def _build_candidate_payload(
        self,
        *,
        candidate_class: str,
        candidate_type: str,
        metadata: dict[str, Any] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._candidate_identity_payload(options)
        payload.update(
            {
                "scope": str(options.get("scope") or "agent_private"),
                "visibility": str(options.get("visibility") or "private"),
                "candidate_scope": self._normalize_text(options.get("candidate_scope")) or None,
                "candidate_class": self._normalize_text(candidate_class),
                "candidate_type": self._normalize_text(candidate_type),
                "subject_entity_id": self._normalize_text(options.get("subject_entity_id")) or None,
                "predicate": self._normalize_text(options.get("predicate")) or None,
                "object_text": self._normalize_text(options.get("object_text")) or None,
                "object_json": options.get("object_json"),
                "status": str(options.get("status") or "proposed"),
                "confidence": options.get("confidence"),
                "support_count": int(options.get("support_count") or 0),
                "explicitness_score": options.get("explicitness_score"),
                "recency_score": options.get("recency_score"),
                "source_quality_score": options.get("source_quality_score"),
                "metadata": self._normalize_json_object(metadata),
            }
        )
        payload.update(self._candidate_source_payload(options))
        return payload

    def _candidate_identity_payload(self, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "rem_run_id": self._normalize_text(options.get("rem_run_id")) or None,
            "profile": self._profile,
            "agent_id": self._agent_identity,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "user_id": self._user_id,
            "session_id": self._normalize_text(options.get("session_id")) or self._session_id,
        }

    def _candidate_source_payload(self, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "promotion_authority": self._normalize_text(options.get("promotion_authority")) or None,
            "review_status": str(options.get("review_status") or "unreviewed"),
            "reviewed_by": self._normalize_text(options.get("reviewed_by")) or None,
            "source_agent_profile": self._normalize_text(options.get("source_agent_profile"))
            or self._profile,
            "source_user_id": self._normalize_text(options.get("source_user_id")) or self._user_id or None,
            "source_mode": self._normalize_event_source_mode(options.get("source_mode") or "live"),
            "writer_role": self._normalize_writer_role(options.get("writer_role") or "agent"),
        }

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        return self._select_single_row("memory_candidates", candidate_id)

    def promote_candidate(
        self,
        candidate_id: str,
        *,
        status: str = "promoted",
        promotion_authority: str | None = None,
    ) -> dict[str, Any]:
        normalized_status = self._normalize_text(status) or "promoted"
        normalized_authority = self._normalize_text(promotion_authority) or None

        if normalized_status != "promoted":
            payload = {
                "status": normalized_status,
                "promotion_authority": normalized_authority,
            }
            return self._update_single_row("memory_candidates", candidate_id, payload)

        with self._promotion_lock:
            candidate = self.get_candidate(candidate_id)
            if not candidate:
                return {}

            original_candidate = dict(candidate)
            existing_fact = self._load_fact_for_candidate(candidate_id)
            if existing_fact:
                return self._coerce_candidate_to_promoted(
                    candidate_id,
                    candidate,
                    promotion_authority=normalized_authority,
                )
            if str(candidate.get("status") or "") == "promoted":
                return candidate

            claimed_candidate = self._claim_candidate_for_promotion(
                candidate_id,
                candidate,
                promotion_authority=normalized_authority,
            )
            if not self._should_create_fact_for_claimed_candidate(claimed_candidate):
                return claimed_candidate

            return self._promote_claimed_candidate(
                candidate_id,
                claimed_candidate,
                original_candidate=original_candidate,
                promotion_authority=normalized_authority,
            )

    def _should_create_fact_for_claimed_candidate(self, claimed_candidate: dict[str, Any]) -> bool:
        if not claimed_candidate:
            return False
        return str(claimed_candidate.get("status") or "") == "staged"

    def _promote_claimed_candidate(
        self,
        candidate_id: str,
        candidate: dict[str, Any],
        *,
        original_candidate: dict[str, Any],
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        fact = self._create_fact_from_candidate(candidate, promotion_authority=promotion_authority)
        if not fact:
            return {}

        fact_id = self._normalize_text(fact.get("id"))
        try:
            self._create_fact_sources_or_raise(fact=fact, candidate=candidate)
            promoted_candidate = self._mark_candidate_promoted(
                candidate_id,
                promotion_authority=promotion_authority,
            )
            self._record_candidate_promotion_events(
                promoted_candidate=promoted_candidate,
                fact=fact,
                fact_id=fact_id,
            )
            return promoted_candidate
        except Exception as exc:
            logger.warning(
                "Failed to promote candidate %s atomically; cleaning up created fact: %s",
                candidate_id,
                exc,
            )
            self._restore_candidate_promotion_state(candidate_id, original_candidate)
            if fact_id:
                self._delete_fact_with_sources(fact_id)
            return {}

    def _create_fact_sources_or_raise(self, *, fact: dict[str, Any], candidate: dict[str, Any]) -> None:
        source_rows = self._create_fact_sources_for_candidate(fact=fact, candidate=candidate)
        if not source_rows:
            raise RuntimeError("Failed to create fact sources for promoted candidate")

    def _mark_candidate_promoted(
        self,
        candidate_id: str,
        *,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        promoted_candidate = self._update_single_row(
            "memory_candidates",
            candidate_id,
            {
                "status": "promoted",
                "promotion_authority": promotion_authority,
            },
        )
        if not promoted_candidate:
            promoted_candidate = self.get_candidate(candidate_id)
        if not promoted_candidate:
            raise RuntimeError("Failed to update candidate status after fact creation")
        if str(promoted_candidate.get("status") or "") != "promoted":
            raise RuntimeError("Candidate status did not transition to promoted")
        return promoted_candidate

    def _record_candidate_promotion_events(
        self,
        *,
        promoted_candidate: dict[str, Any],
        fact: dict[str, Any],
        fact_id: str,
    ) -> None:
        if not fact_id:
            return
        provenance_chain = self._load_fact_provenance_chain([fact_id])
        self._insert_memory_events(
            [
                self._build_candidate_promoted_event_row(
                    candidate=promoted_candidate,
                    fact=fact,
                ),
                self._build_fact_lifecycle_event_row(
                    fact,
                    provenance_chain.get(fact_id, {"sources": [], "observations": {}}),
                ),
            ],
            fail_fast=True,
        )

    def _claim_candidate_for_promotion(
        self,
        candidate_id: str,
        candidate: dict[str, Any],
        *,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        existing_authority = self._normalize_text(candidate.get("promotion_authority")) or None
        claim_payload = {
            "status": "staged",
            "promotion_authority": promotion_authority or existing_authority,
        }
        claimed_rows = self._update_rows(
            "memory_candidates",
            claim_payload,
            filters={
                "id": f"eq.{candidate_id}",
                "profile": f"eq.{self._profile}",
                "status": "eq.proposed",
            },
        )
        if claimed_rows:
            return claimed_rows[0]

        current_candidate = self.get_candidate(candidate_id)
        if not current_candidate:
            return {}

        if str(current_candidate.get("status") or "") == "promoted":
            return current_candidate

        if str(current_candidate.get("status") or "") == "staged":
            return self._wait_for_candidate_promotion(candidate_id, promotion_authority=promotion_authority)

        existing_fact = self._load_fact_for_candidate(candidate_id)
        if existing_fact:
            return self._coerce_candidate_to_promoted(
                candidate_id,
                current_candidate,
                promotion_authority=promotion_authority,
            )

        return current_candidate

    def _wait_for_candidate_promotion(
        self,
        candidate_id: str,
        *,
        promotion_authority: str | None,
        attempts: int = 40,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        for _ in range(attempts):
            current_candidate = self.get_candidate(candidate_id)
            if not current_candidate:
                return {}

            current_status = str(current_candidate.get("status") or "")
            if current_status == "promoted":
                return current_candidate

            existing_fact = self._load_fact_for_candidate(candidate_id)
            if existing_fact:
                return self._coerce_candidate_to_promoted(
                    candidate_id,
                    current_candidate,
                    promotion_authority=promotion_authority,
                )

            if current_status != "staged":
                return current_candidate

            time.sleep(poll_interval)

        current_candidate = self.get_candidate(candidate_id)
        if not current_candidate:
            return {}
        existing_fact = self._load_fact_for_candidate(candidate_id)
        if existing_fact:
            return self._coerce_candidate_to_promoted(
                candidate_id,
                current_candidate,
                promotion_authority=promotion_authority,
            )
        return current_candidate

    def _load_fact_for_candidate(self, candidate_id: str) -> dict[str, Any]:
        client = self._client
        normalized_candidate_id = self._normalize_text(candidate_id)
        if client is None or not normalized_candidate_id:
            return {}

        try:
            source_rows = client.select(
                "memory_fact_sources",
                columns="fact_id,candidate_id",
                filters={"candidate_id": f"eq.{normalized_candidate_id}"},
                order=_ORDER_CREATED_ASC,
                limit=1,
            )
        except Exception as exc:
            logger.warning("Failed to read fact sources for candidate %s: %s", normalized_candidate_id, exc)
            return {}

        fact_id = self._normalize_text(source_rows[0].get("fact_id")) if source_rows else ""
        if not fact_id:
            return {}

        return self._select_single_row("memory_facts", fact_id)

    def _coerce_candidate_to_promoted(
        self,
        candidate_id: str,
        candidate: dict[str, Any],
        *,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        payload = {
            "status": "promoted",
            "promotion_authority": promotion_authority
            or self._normalize_text(candidate.get("promotion_authority"))
            or None,
        }
        if (
            str(candidate.get("status") or "") == "promoted"
            and payload["promotion_authority"] == candidate.get("promotion_authority")
        ):
            return candidate

        promoted_candidate = self._update_single_row("memory_candidates", candidate_id, payload)
        if promoted_candidate:
            return promoted_candidate

        current_candidate = self.get_candidate(candidate_id)
        if current_candidate:
            return current_candidate

        fallback_candidate = dict(candidate)
        fallback_candidate.update(payload)
        return fallback_candidate

    def _restore_candidate_promotion_state(
        self,
        candidate_id: str,
        candidate: dict[str, Any],
    ) -> None:
        rollback_payload = {
            "status": self._normalize_text(candidate.get("status")) or "proposed",
            "promotion_authority": self._normalize_text(candidate.get("promotion_authority")) or None,
        }
        restored_candidate = self._update_single_row(
            "memory_candidates",
            candidate_id,
            rollback_payload,
        )
        if restored_candidate:
            return

        logger.warning(
            "Failed to restore candidate %s after promotion rollback attempt",
            candidate_id,
        )

    def _delete_fact_with_sources(self, fact_id: str) -> None:
        client = self._client
        normalized_fact_id = self._normalize_text(fact_id)
        if client is None or not normalized_fact_id:
            return

        try:
            client.delete(
                "memory_fact_sources",
                {"fact_id": f"eq.{normalized_fact_id}"},
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete fact sources for fact %s during promotion rollback: %s",
                normalized_fact_id,
                exc,
            )

        try:
            client.delete(
                "memory_facts",
                {"id": f"eq.{normalized_fact_id}"},
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete fact %s during promotion rollback: %s",
                normalized_fact_id,
                exc,
            )

    def _create_fact_from_candidate(
        self,
        candidate: dict[str, Any],
        *,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        source_observation_ids = self._extract_candidate_source_observation_ids(candidate)
        object_text, object_json = self._resolve_candidate_fact_object(candidate)
        if object_text is None and object_json is None:
            return {}

        payload = self._build_fact_payload(
            candidate,
            source_observation_ids=source_observation_ids,
            object_text=object_text,
            object_json=object_json,
            promotion_authority=promotion_authority,
        )
        return self._insert_single_row("memory_facts", payload)

    def _resolve_candidate_fact_object(self, candidate: dict[str, Any]) -> tuple[str | None, Any]:
        object_text = self._normalize_text(candidate.get("object_text")) or None
        object_json = candidate.get("object_json")
        if object_text is not None or object_json is not None:
            return object_text, object_json
        return self._normalize_text(candidate.get("predicate")) or None, object_json

    def _build_fact_payload(
        self,
        candidate: dict[str, Any],
        *,
        source_observation_ids: list[str],
        object_text: str | None,
        object_json: Any,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        return {
            **self._build_fact_identity_payload(candidate),
            **self._build_fact_review_payload(candidate, promotion_authority=promotion_authority),
            "object_text": object_text,
            "object_json": object_json,
            "status": "active",
            "confidence": candidate.get("confidence"),
            "source_count": len(source_observation_ids),
            "last_supported_at": self._candidate_last_supported_at(candidate),
            "first_promoted_at": self._normalize_iso_timestamp(datetime.now(timezone.utc)),
            "metadata": self._build_fact_metadata(candidate, source_observation_ids),
        }

    def _build_fact_identity_payload(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "profile": str(candidate.get("profile") or self._profile),
            "agent_id": str(candidate.get("agent_id") or self._agent_identity),
            "workspace": str(candidate.get("workspace") or self._workspace),
            "project_id": str(candidate.get("project_id") or self._project_id),
            "user_id": str(candidate.get("user_id") or self._user_id),
            "scope": str(candidate.get("scope") or "agent_private"),
            "visibility": str(candidate.get("visibility") or "private"),
            "fact_class": str(candidate.get("candidate_class") or "fact"),
            "fact_type": str(candidate.get("candidate_type") or "statement"),
            "subject_entity_id": self._normalize_text(candidate.get("subject_entity_id")) or None,
            "predicate": self._normalize_text(candidate.get("predicate")) or "states",
        }

    def _build_fact_review_payload(
        self,
        candidate: dict[str, Any],
        *,
        promotion_authority: str | None,
    ) -> dict[str, Any]:
        return {
            "promotion_authority": promotion_authority
            or self._normalize_text(candidate.get("promotion_authority"))
            or None,
            "review_status": str(candidate.get("review_status") or "unreviewed"),
            "reviewed_by": self._normalize_text(candidate.get("reviewed_by")) or None,
        }

    def _build_fact_metadata(
        self,
        candidate: dict[str, Any],
        source_observation_ids: list[str],
    ) -> dict[str, Any]:
        metadata = self._normalize_json_object(candidate.get("metadata"))
        metadata.update(
            {
                "promoted_from_candidate_id": self._normalize_text(candidate.get("id")),
                "source_observation_ids": source_observation_ids,
            }
        )
        return metadata

    def _candidate_last_supported_at(self, candidate: dict[str, Any]) -> str | None:
        return self._normalize_iso_timestamp(candidate.get("updated_at") or candidate.get("created_at")) or None

    def _extract_candidate_source_observation_ids(
        self,
        candidate: dict[str, Any],
    ) -> list[str]:
        metadata = self._normalize_json_object(candidate.get("metadata"))
        raw_ids = metadata.get("source_observation_ids")
        if not isinstance(raw_ids, list):
            return []

        normalized_ids: list[str] = []
        for raw_id in raw_ids:
            observation_id = self._normalize_text(raw_id)
            if observation_id and observation_id not in normalized_ids:
                normalized_ids.append(observation_id)
        return normalized_ids

    def _extract_candidate_evidence(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        metadata = self._normalize_json_object(candidate.get("metadata"))
        raw_evidence = metadata.get("evidence")
        if not isinstance(raw_evidence, list):
            return {}

        evidence_by_observation_id: dict[str, dict[str, Any]] = {}
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            observation_id = self._normalize_text(item.get("observation_id"))
            if observation_id:
                evidence_by_observation_id[observation_id] = item
        return evidence_by_observation_id

    def _load_observations_by_id(
        self,
        observation_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        client = self._client
        normalized_ids = [
            observation_id
            for observation_id in observation_ids
            if self._normalize_text(observation_id)
        ]
        if client is None or not normalized_ids:
            return {}

        quoted_ids = ",".join(f'"{observation_id}"' for observation_id in normalized_ids)
        try:
            rows = client.select(
                "memory_observations",
                columns=_TOOL_OBSERVATION_COLUMNS,
                filters={
                    "id": f"in.({quoted_ids})",
                    "profile": f"eq.{self._profile}",
                },
            )
        except Exception as exc:
            logger.warning("Failed to load source observations for fact promotion: %s", exc)
            return {}

        return {
            self._normalize_text(row.get("id")): row
            for row in rows
            if self._normalize_text(row.get("id"))
        }

    def _create_fact_sources_for_candidate(
        self,
        *,
        fact: dict[str, Any],
        candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        client = self._client
        fact_id = self._normalize_text(fact.get("id"))
        candidate_id = self._normalize_text(candidate.get("id"))
        if client is None or not fact_id or not candidate_id:
            return []

        observation_ids = self._extract_candidate_source_observation_ids(candidate)
        observations_by_id = self._load_observations_by_id(observation_ids)
        evidence_by_observation_id = self._extract_candidate_evidence(candidate)
        normalized_excerpt = self._normalize_text(candidate.get("object_text")) or None

        rows: list[dict[str, Any]] = [
            {
                "fact_id": fact_id,
                "candidate_id": candidate_id,
                "observation_id": None,
                "rem_run_id": self._normalize_text(candidate.get("rem_run_id")) or None,
                "source_excerpt": normalized_excerpt,
                "extractor": "promote_candidate",
                "confidence": candidate.get("confidence"),
                "support_role": "direct",
            }
        ]

        for index, observation_id in enumerate(observation_ids):
            observation = observations_by_id.get(observation_id, {})
            evidence = evidence_by_observation_id.get(observation_id, {})
            rows.append(
                {
                    "fact_id": fact_id,
                    "candidate_id": candidate_id,
                    "observation_id": observation_id,
                    "rem_run_id": self._normalize_text(candidate.get("rem_run_id")) or None,
                    "source_excerpt": self._normalize_text(evidence.get("excerpt"))
                    or self._normalize_text(observation.get("content"))
                    or normalized_excerpt,
                    "extractor": "promote_candidate",
                    "confidence": candidate.get("confidence"),
                    "support_role": "direct" if index == 0 else "supporting",
                }
            )

        try:
            payload: dict[str, Any] | list[dict[str, Any]]
            payload = rows[0] if len(rows) == 1 else rows
            inserted_rows = client.insert("memory_fact_sources", payload)
        except Exception as exc:
            logger.warning("Failed to insert fact sources for candidate %s: %s", candidate_id, exc)
            return []

        return [dict(row) for row in inserted_rows]

    def _build_candidate_promoted_event_row(
        self,
        *,
        candidate: dict[str, Any],
        fact: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidate_id = self._normalize_text(candidate.get("id"))
        fact_id = self._normalize_text(fact.get("id"))
        if not candidate_id or not fact_id:
            return None

        return {
            "profile": str(candidate.get("profile") or self._profile),
            "agent_id": str(candidate.get("agent_id") or self._agent_identity),
            "workspace": str(candidate.get("workspace") or self._workspace),
            "project_id": str(candidate.get("project_id") or self._project_id),
            "user_id": str(candidate.get("user_id") or self._user_id),
            "session_id": str(candidate.get("session_id") or self._session_id),
            "scope": str(candidate.get("scope") or "agent_private"),
            "source_mode": str(candidate.get("source_mode") or "live"),
            "writer_role": str(candidate.get("writer_role") or "agent"),
            "actor_id": self._agent_identity,
            "actor_type": "agent",
            "event_type": "candidate_promoted",
            "target_table": "memory_candidates",
            "target_id": candidate_id,
            "payload": {
                "fact_id": fact_id,
                "rem_run_id": self._normalize_text(candidate.get("rem_run_id")),
                "candidate_class": str(candidate.get("candidate_class") or ""),
                "candidate_type": str(candidate.get("candidate_type") or ""),
                "promotion_authority": self._normalize_text(candidate.get("promotion_authority"))
                or None,
                "source_observation_ids": self._extract_candidate_source_observation_ids(candidate),
            },
        }

    def select_observation_batch(
        self,
        *,
        scope: str | None = None,
        session_id: str | None = None,
        since: datetime | str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        client = self._require_tool_client(require_profile=True)
        if client is None:
            return []

        normalized_limit = self._normalize_batch_limit(limit)
        if normalized_limit <= 0:
            return []

        observation_filters = self._build_observation_batch_filters(
            scope=scope,
            session_id=session_id,
            since=since,
        )

        try:
            observations = client.select(
                "memory_observations",
                columns=_REM_BATCH_OBSERVATION_COLUMNS,
                filters=observation_filters,
                order=_ORDER_CREATED_ASC,
            )
            completed_runs = client.select(
                "memory_rem_runs",
                columns="summary",
                filters={
                    "profile": f"eq.{self._profile}",
                    "status": "eq.completed",
                },
            )
        except Exception as exc:
            logger.warning("Failed to select REM observation batch: %s", exc)
            return []

        processed_observation_ids = self._extract_processed_observation_ids(completed_runs)
        batch: list[dict[str, Any]] = []
        for observation in observations:
            observation_id = self._normalize_text(observation.get("id"))
            if observation_id and observation_id in processed_observation_ids:
                continue

            batch_item = self._build_observation_batch_item(observation)
            if batch_item is None:
                continue

            batch.append(batch_item)
            if len(batch) >= normalized_limit:
                break

        return batch

    def _normalize_batch_limit(self, limit: int) -> int:
        try:
            return int(limit)
        except (TypeError, ValueError):
            return 50

    def _build_observation_batch_filters(
        self,
        *,
        scope: str | None,
        session_id: str | None,
        since: datetime | str | None,
    ) -> dict[str, str]:
        filters = {"profile": f"eq.{self._profile}"}

        normalized_scope = self._normalize_text(scope)
        if normalized_scope:
            filters["scope"] = f"eq.{normalized_scope}"

        normalized_session_id = self._normalize_text(session_id)
        if normalized_session_id:
            filters["session_id"] = f"eq.{normalized_session_id}"

        normalized_since = self._normalize_iso_timestamp(since)
        if normalized_since:
            filters["created_at"] = f"gte.{normalized_since}"

        return filters

    def _build_observation_batch_item(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any] | None:
        observation_id = self._normalize_text(observation.get("id"))
        if not observation_id:
            return None

        return {
            "id": observation_id,
            "content": str(observation.get("content") or ""),
            "scope": str(observation.get("scope") or ""),
            "session_id": str(observation.get("session_id") or ""),
            "source_type": str(observation.get("source_type") or ""),
            "created_at": self._normalize_iso_timestamp(observation.get("created_at")),
            "metadata": self._normalize_event_metadata(observation.get("metadata")),
        }

    def _extract_processed_observation_ids(
        self,
        completed_runs: list[dict[str, Any]],
    ) -> set[str]:
        processed_ids: set[str] = set()
        for run in completed_runs:
            processed_ids.update(self._collect_observation_ids_from_summary(run.get("summary")))
        return processed_ids

    def _collect_observation_ids_from_summary(self, value: Any) -> set[str]:
        parsed_value = self._parse_summary_value(value)
        if isinstance(parsed_value, list):
            processed_ids: set[str] = set()
            for item in parsed_value:
                processed_ids.update(self._collect_observation_ids_from_summary(item))
            return processed_ids

        if not isinstance(parsed_value, dict):
            return set()

        processed_ids = self._coerce_observation_id_set(
            parsed_value.get("observation_ids") or parsed_value.get("processed_observation_ids")
        )
        for key in ("batch", "observations", "processed_observations", "selected_observations"):
            nested_value = parsed_value.get(key)
            if not nested_value:
                continue
            processed_ids.update(self._collect_observation_ids_from_summary(nested_value))
        return processed_ids

    def _coerce_observation_id_set(self, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, list):
            return {
                normalized
                for normalized in (
                    self._normalize_text(item.get("id")) if isinstance(item, dict) else self._normalize_text(item)
                    for item in value
                )
                if normalized
            }
        if isinstance(value, dict):
            normalized = self._normalize_text(value.get("id"))
            return {normalized} if normalized else set()

        normalized = self._normalize_text(value)
        return {normalized} if normalized else set()

    def _parse_summary_value(self, value: Any) -> Any:
        if isinstance(value, str):
            normalized = self._normalize_text(value)
            if not normalized:
                return {}
            try:
                return json.loads(normalized)
            except ValueError:
                return {}
        return value

    def run_rem(
        self,
        *,
        scope: str | None = None,
        session_id: str | None = None,
        since: datetime | str | None = None,
        limit: int = 50,
        mode: str = "manual",
        trigger: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._empty_rem_result()
        client = self._require_tool_client(require_profile=True)
        if client is None:
            return result

        rem_context = self._prepare_rem_run_context(
            scope=scope,
            session_id=session_id,
            since=since,
            limit=limit,
            mode=mode,
            trigger=trigger,
            parameters=parameters,
        )
        rem_run = self.create_rem_run(
            scope=rem_context["normalized_scope"],
            mode=mode,
            status="running",
            trigger=trigger,
            started_at=rem_context["started_at"],
            extractor_version="rule_based_v1",
            observation_batch_key=rem_context["observation_batch_key"],
            input_scope=rem_context["input_scope"],
            parameters=rem_context["run_parameters"],
        )
        run_id = self._normalize_text(rem_run.get("id"))
        if not run_id:
            return self._resolve_existing_rem_run(rem_context["observation_batch_key"], result)

        result["run_id"] = run_id
        self._record_rem_started(
            run_id,
            normalized_scope=rem_context["normalized_scope"],
            mode=mode,
            trigger=trigger,
            input_scope=rem_context["input_scope"],
            run_parameters=rem_context["run_parameters"],
        )

        try:
            summary = self._process_rem_batch(run_id, rem_context["batch"], result)
            return self._finalize_rem_success(
                run_id,
                summary=summary,
                result=result,
                normalized_scope=rem_context["normalized_scope"],
            )
        except Exception as exc:
            return self._finalize_rem_failure(
                run_id,
                error=exc,
                result=result,
                normalized_scope=rem_context["normalized_scope"],
            )

    def _empty_rem_result(self) -> dict[str, Any]:
        return {
            "run_id": "",
            "status": "failed",
            "observations_processed": 0,
            "reflections_created": 0,
            "candidates_created": 0,
        }

    def _prepare_rem_run_context(
        self,
        *,
        scope: str | None,
        session_id: str | None,
        since: datetime | str | None,
        limit: int,
        mode: str,
        trigger: str,
        parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_parameters = self._normalize_json_object(parameters)
        input_scope = self._build_rem_input_scope(
            scope=scope,
            session_id=session_id,
            since=since,
            limit=limit,
        )
        batch = self.select_observation_batch(
            scope=scope,
            session_id=session_id,
            since=since,
            limit=limit,
        )
        processed_observation_ids = self._extract_batch_observation_ids(batch)
        observation_batch_key = self._build_observation_batch_key(processed_observation_ids)
        if observation_batch_key and self._observation_batch_key_supported:
            run_parameters["observation_batch_key"] = observation_batch_key
        return {
            "batch": batch,
            "input_scope": input_scope,
            "normalized_scope": self._normalize_text(scope) or "agent_private",
            "observation_batch_key": observation_batch_key,
            "processed_observation_ids": processed_observation_ids,
            "run_parameters": run_parameters,
            "started_at": self._normalize_iso_timestamp(datetime.now(timezone.utc)),
        }

    def _extract_batch_observation_ids(self, batch: list[dict[str, Any]]) -> list[str]:
        return [
            observation_id
            for observation_id in (
                self._normalize_text(observation.get("id")) for observation in batch
            )
            if observation_id
        ]

    def _resolve_existing_rem_run(
        self,
        observation_batch_key: str,
        default_result: dict[str, Any],
    ) -> dict[str, Any]:
        if observation_batch_key and self._observation_batch_key_supported:
            existing_run = self._wait_for_rem_run_by_batch_key(observation_batch_key)
            if existing_run:
                return self._result_from_rem_run(existing_run)
        return default_result

    def _record_rem_started(
        self,
        run_id: str,
        *,
        normalized_scope: str,
        mode: str,
        trigger: str,
        input_scope: dict[str, Any],
        run_parameters: dict[str, Any],
    ) -> None:
        self._insert_memory_events(
            [
                self._build_rem_run_event_row(
                    rem_run_id=run_id,
                    scope=normalized_scope,
                    event_type="rem_started",
                    payload={
                        "mode": mode,
                        "trigger": self._normalize_text(trigger) or None,
                        "input_scope": input_scope,
                        "parameters": run_parameters,
                    },
                )
            ]
        )

    def _process_rem_batch(
        self,
        run_id: str,
        batch: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        result["observations_processed"] = len(batch)
        reflection_ids: list[str] = []
        candidate_ids: list[str] = []
        group_keys: list[str] = []
        result["_reflection_ids"] = reflection_ids
        result["_candidate_ids"] = candidate_ids
        result["_group_keys"] = group_keys
        observation_entity_ids = self._load_observation_entity_ids(batch)
        grouped_observations = self._group_observations_for_reflection(batch)
        for group in grouped_observations:
            group_summary = self._process_rem_group(
                run_id,
                group,
                observation_entity_ids=observation_entity_ids,
            )
            reflection_id = group_summary.get("reflection_id")
            if reflection_id:
                reflection_ids.append(reflection_id)
                group_keys.append(group["group_key"])
            candidate_ids.extend(group_summary["candidate_ids"])

        result["reflections_created"] = len(reflection_ids)
        result["candidates_created"] = len(candidate_ids)
        return {
            "observation_ids": self._extract_batch_observation_ids(batch),
            "observations_processed": result["observations_processed"],
            "reflections_created": len(reflection_ids),
            "candidates_created": len(candidate_ids),
            "reflection_ids": reflection_ids,
            "candidate_ids": candidate_ids,
            "group_keys": group_keys,
        }

    def _process_rem_group(
        self,
        run_id: str,
        group: dict[str, Any],
        *,
        observation_entity_ids: dict[str, str],
    ) -> dict[str, Any]:
        group_reflection = self._build_group_reflection(
            group,
            observation_entity_ids=observation_entity_ids,
        )
        reflection_row = self._store_group_reflection(run_id, group, group_reflection)
        reflection_id = self._record_group_reflection_event(reflection_row, group, group_reflection)
        candidate_ids = self._create_group_candidates(run_id, group, group_reflection)
        return {
            "reflection_id": reflection_id,
            "candidate_ids": candidate_ids,
        }

    def _store_group_reflection(
        self,
        run_id: str,
        group: dict[str, Any],
        group_reflection: dict[str, Any],
    ) -> dict[str, Any]:
        reflection_row = self.store_reflection(
            rem_run_id=run_id,
            scope=group["scope"],
            observation_group_key=group["group_key"],
            summary_what_happened=group_reflection["summary_what_happened"],
            reflections=group_reflection["reflections"],
            candidates=group_reflection["candidates"],
            lasting_updates=group_reflection["lasting_updates"],
            metadata=group_reflection["metadata"],
        )
        if not reflection_row:
            raise RuntimeError("Failed to store reflection row")
        return reflection_row

    def _record_group_reflection_event(
        self,
        reflection_row: dict[str, Any],
        group: dict[str, Any],
        group_reflection: dict[str, Any],
    ) -> str:
        reflection_id = self._normalize_text(reflection_row.get("id"))
        if not reflection_id:
            return ""
        self._insert_memory_events(
            [
                self._build_reflection_created_event_row(
                    reflection=reflection_row,
                    group=group,
                    observation_ids=group_reflection["observation_ids"],
                )
            ]
        )
        return reflection_id

    def _create_group_candidates(
        self,
        run_id: str,
        group: dict[str, Any],
        group_reflection: dict[str, Any],
    ) -> list[str]:
        candidate_ids: list[str] = []
        for candidate_blueprint in group_reflection["candidate_blueprints"]:
            candidate_scope = self._normalize_text(candidate_blueprint.get("scope")) or str(group["scope"])
            candidate_row = self.create_candidate(
                rem_run_id=run_id,
                session_id=group["session_id"],
                scope=candidate_scope,
                candidate_scope=candidate_scope,
                candidate_class=str(candidate_blueprint["candidate_class"]),
                candidate_type=str(candidate_blueprint["candidate_type"]),
                subject_entity_id=candidate_blueprint.get("subject_entity_id"),
                predicate=candidate_blueprint.get("predicate"),
                object_text=candidate_blueprint.get("object_text"),
                confidence=candidate_blueprint.get("confidence"),
                support_count=candidate_blueprint.get("support_count"),
                explicitness_score=candidate_blueprint.get("explicitness_score"),
                recency_score=candidate_blueprint.get("recency_score"),
                source_quality_score=candidate_blueprint.get("source_quality_score"),
                metadata=candidate_blueprint.get("metadata"),
            )
            candidate_id = self._record_candidate_created_event(candidate_row, candidate_blueprint)
            if candidate_id:
                candidate_ids.append(candidate_id)
                self._maybe_auto_promote_candidate(candidate_id, candidate_row)
        return candidate_ids

    def _record_candidate_created_event(
        self,
        candidate_row: dict[str, Any],
        candidate_blueprint: dict[str, Any],
    ) -> str:
        if not candidate_row:
            raise RuntimeError("Failed to create candidate row")
        candidate_id = self._normalize_text(candidate_row.get("id"))
        if not candidate_id:
            return ""
        self._insert_memory_events(
            [
                self._build_candidate_created_event_row(
                    candidate=candidate_row,
                    candidate_blueprint=candidate_blueprint,
                )
            ]
        )
        return candidate_id

    def _finalize_rem_success(
        self,
        run_id: str,
        *,
        summary: dict[str, Any],
        result: dict[str, Any],
        normalized_scope: str,
    ) -> dict[str, Any]:
        completed_at = self._normalize_iso_timestamp(datetime.now(timezone.utc))
        self._update_single_row(
            "memory_rem_runs",
            run_id,
            {
                "status": "completed",
                "completed_at": completed_at,
                "summary": summary,
                "error_text": None,
            },
        )
        result["status"] = "completed"
        result["reflections_created"] = summary["reflections_created"]
        result["candidates_created"] = summary["candidates_created"]
        result.pop("_reflection_ids", None)
        result.pop("_candidate_ids", None)
        result.pop("_group_keys", None)
        self._insert_memory_events(
            [
                self._build_rem_run_event_row(
                    rem_run_id=run_id,
                    scope=normalized_scope,
                    event_type="rem_completed",
                    payload=summary,
                )
            ]
        )
        return result

    def _finalize_rem_failure(
        self,
        run_id: str,
        *,
        error: Exception,
        result: dict[str, Any],
        normalized_scope: str,
    ) -> dict[str, Any]:
        failure_summary = {
            "observation_ids": [],
            "observations_processed": result["observations_processed"],
            "reflections_created": result["reflections_created"],
            "candidates_created": result["candidates_created"],
            "reflection_ids": list(result.get("_reflection_ids", [])),
            "candidate_ids": list(result.get("_candidate_ids", [])),
            "group_keys": list(result.get("_group_keys", [])),
        }
        completed_at = self._normalize_iso_timestamp(datetime.now(timezone.utc))
        self._update_single_row(
            "memory_rem_runs",
            run_id,
            {
                "status": "failed",
                "completed_at": completed_at,
                "summary": failure_summary,
                "error_text": str(error),
            },
        )
        result["status"] = "failed"
        result.pop("_reflection_ids", None)
        result.pop("_candidate_ids", None)
        result.pop("_group_keys", None)
        self._insert_memory_events(
            [
                self._build_rem_run_event_row(
                    rem_run_id=run_id,
                    scope=normalized_scope,
                    event_type="rem_completed",
                    payload={**failure_summary, "status": "failed", "error_text": str(error)},
                )
            ]
        )
        return result

    def _build_observation_batch_key(self, observation_ids: list[str]) -> str:
        normalized_ids = sorted(
            {
                normalized
                for observation_id in observation_ids
                if (normalized := self._normalize_text(observation_id))
            }
        )
        if not normalized_ids:
            return ""
        return hashlib.sha256("|".join(normalized_ids).encode("utf-8")).hexdigest()

    def _wait_for_rem_run_by_batch_key(
        self,
        observation_batch_key: str,
        *,
        attempts: int = 40,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        for _ in range(attempts):
            existing_run = self.get_rem_run_by_batch_key(observation_batch_key)
            if not existing_run:
                time.sleep(poll_interval)
                continue

            if str(existing_run.get("status") or "") in {"completed", "failed", "cancelled"}:
                return existing_run

            time.sleep(poll_interval)

        return self.get_rem_run_by_batch_key(observation_batch_key)

    def _result_from_rem_run(self, rem_run: dict[str, Any]) -> dict[str, Any]:
        summary = self._parse_summary_value(rem_run.get("summary"))
        if not isinstance(summary, dict):
            summary = {}
        observation_ids = summary.get("observation_ids")
        observations_processed = summary.get("observations_processed")
        if observations_processed is None and isinstance(observation_ids, list):
            observations_processed = len(observation_ids)
        return {
            "run_id": self._normalize_text(rem_run.get("id")),
            "status": self._normalize_text(rem_run.get("status")) or "failed",
            "observations_processed": int(observations_processed or 0),
            "reflections_created": int(summary.get("reflections_created") or 0),
            "candidates_created": int(summary.get("candidates_created") or 0),
        }

    def _build_rem_input_scope(
        self,
        *,
        scope: str | None,
        session_id: str | None,
        since: datetime | str | None,
        limit: int,
    ) -> dict[str, Any]:
        input_scope: dict[str, Any] = {}
        normalized_scope = self._normalize_text(scope)
        normalized_session_id = self._normalize_text(session_id)
        normalized_since = self._normalize_iso_timestamp(since)
        if normalized_scope:
            input_scope["scope"] = normalized_scope
        if normalized_session_id:
            input_scope["session_id"] = normalized_session_id
        if normalized_since:
            input_scope["since"] = normalized_since
        input_scope["limit"] = self._normalize_batch_limit(limit)
        return input_scope

    def _group_observations_for_reflection(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        ordered_groups: list[dict[str, Any]] = []
        for observation in observations:
            session_id = self._normalize_text(observation.get("session_id")) or self._session_id
            scope = self._normalize_text(observation.get("scope")) or "agent_private"
            group_key = f"session:{session_id}|scope:{scope}"
            group = grouped.get(group_key)
            if group is None:
                group = {
                    "group_key": group_key,
                    "session_id": session_id,
                    "scope": scope,
                    "observations": [],
                }
                grouped[group_key] = group
                ordered_groups.append(group)
            group["observations"].append(observation)
        return ordered_groups

    def _build_group_reflection(
        self,
        group: dict[str, Any],
        *,
        observation_entity_ids: dict[str, str],
    ) -> dict[str, Any]:
        observations = list(group["observations"])
        extracted_claims: list[dict[str, Any]] = []
        for observation in observations:
            observation_id = self._normalize_text(observation.get("id"))
            extracted_claims.extend(
                self._extract_claims_from_observation(
                    observation,
                    subject_entity_id=observation_entity_ids.get(observation_id, ""),
                )
            )

        semantic_claims = self._extract_semantic_claims_for_group(
            group,
            observations=observations,
            subject_entity_id=next(iter({claim.get("subject_entity_id") for claim in extracted_claims if claim.get("subject_entity_id")}), "") or "",
        )
        aggregated_claims = self._aggregate_claims(
            extracted_claims + semantic_claims,
            observations=observations,
            group_key=str(group["group_key"]),
            session_id=str(group["session_id"]),
        )
        reflection_entries = [
            self._build_reflection_entry(claim) for claim in aggregated_claims
        ]
        candidate_blueprints = [
            self._build_candidate_blueprint(claim, group=group) for claim in aggregated_claims
        ]
        lasting_updates = [
            self._build_lasting_update_entry(claim) for claim in aggregated_claims
        ]
        observation_ids = [
            observation_id
            for observation_id in (
                self._normalize_text(observation.get("id")) for observation in observations
            )
            if observation_id
        ]

        return {
            "summary_what_happened": self._summarize_group_reflection(
                group=group,
                observations=observations,
                aggregated_claims=aggregated_claims,
            ),
            "reflections": reflection_entries,
            "candidates": [
                {
                    "candidate_class": blueprint["candidate_class"],
                    "candidate_type": blueprint["candidate_type"],
                    "predicate": blueprint["predicate"],
                    "object_text": blueprint["object_text"],
                    "confidence": blueprint["confidence"],
                    "support_count": blueprint["support_count"],
                    "subject_entity_id": blueprint.get("subject_entity_id"),
                    "observation_ids": blueprint["metadata"]["source_observation_ids"],
                }
                for blueprint in candidate_blueprints
            ],
            "lasting_updates": lasting_updates,
            "candidate_blueprints": candidate_blueprints,
            "metadata": {
                "session_id": group["session_id"],
                "group_key": group["group_key"],
                "observation_ids": observation_ids,
            },
            "observation_ids": observation_ids,
        }

    def _extract_claims_from_observation(
        self,
        observation: dict[str, Any],
        *,
        subject_entity_id: str = "",
    ) -> list[dict[str, Any]]:
        content = str(observation.get("content") or "")
        normalized_content = self._normalize_text(content)
        if not normalized_content:
            return []

        claims: list[dict[str, Any]] = []
        observation_id = self._normalize_text(observation.get("id"))
        for sentence in self._split_observation_sentences(normalized_content):
            if self._is_trivial_observation_sentence(sentence):
                continue

            claim = self._extract_preference_claim(
                sentence,
                observation_id=observation_id,
                observation=observation,
                subject_entity_id=subject_entity_id,
            )
            if claim is None:
                claim = self._extract_decision_claim(
                    sentence,
                    observation_id=observation_id,
                    observation=observation,
                    subject_entity_id=subject_entity_id,
                )
            if claim is None:
                claim = self._extract_fact_claim(
                    sentence,
                    observation_id=observation_id,
                    observation=observation,
                    subject_entity_id=subject_entity_id,
                )
            if claim is not None:
                claims.append(claim)
        return claims

    def _split_observation_sentences(self, content: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", content)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _is_trivial_observation_sentence(self, sentence: str) -> bool:
        normalized = self._normalize_text(sentence).lower()
        if not normalized:
            return True
        if normalized in {"ok", "okay", "hi", "hello", "thanks", "thank you", "noted"}:
            return True
        return len(normalized.split()) <= 1

    def _extract_preference_claim(
        self,
        sentence: str,
        *,
        observation_id: str,
        observation: dict[str, Any],
        subject_entity_id: str,
    ) -> dict[str, Any] | None:
        match = re.search(r"\b(?:the user|user|i)\s+prefers?\s+(?P<object>[^.?!]+)", sentence, re.IGNORECASE)
        if match is None:
            return None
        object_text = self._clean_claim_fragment(match.group("object"))
        if not object_text:
            return None
        return self._build_raw_claim(
            claim_type="preference",
            candidate_class="preference",
            candidate_type="stated_preference",
            predicate="prefers",
            object_text=object_text,
            observation_id=observation_id,
            observation=observation,
            sentence=sentence,
            subject_entity_id=subject_entity_id,
            explicitness_score=0.95,
            scope_hint="user",
        )

    def _extract_decision_claim(
        self,
        sentence: str,
        *,
        observation_id: str,
        observation: dict[str, Any],
        subject_entity_id: str,
    ) -> dict[str, Any] | None:
        match = re.search(
            r"\b(?:we|i)\s+(?:decided to|decided on|chose to|selected|picked|opted to|will use)\s+(?P<object>[^.?!]+)",
            sentence,
            re.IGNORECASE,
        )
        if match is None:
            return None
        object_text = self._clean_claim_fragment(match.group("object"))
        if not object_text:
            return None
        return self._build_raw_claim(
            claim_type="decision",
            candidate_class="decision",
            candidate_type="chosen_action",
            predicate="decided",
            object_text=object_text,
            observation_id=observation_id,
            observation=observation,
            sentence=sentence,
            subject_entity_id=subject_entity_id,
            explicitness_score=0.98,
            scope_hint=self._infer_scope_hint(sentence, observation=observation),
        )

    def _extract_fact_claim(
        self,
        sentence: str,
        *,
        observation_id: str,
        observation: dict[str, Any],
        subject_entity_id: str,
    ) -> dict[str, Any] | None:
        match = _FACT_CLAIM_PATTERN.match(sentence)
        if match is None:
            return None
        object_text = self._clean_claim_fragment(match.group("object"))
        predicate = self._clean_claim_fragment(match.group("verb")).lower()
        if not object_text or not predicate:
            return None
        return self._build_raw_claim(
            claim_type="fact",
            candidate_class="fact",
            candidate_type="statement",
            predicate=predicate,
            object_text=object_text,
            observation_id=observation_id,
            observation=observation,
            sentence=sentence,
            subject_entity_id=subject_entity_id,
            explicitness_score=0.9,
            scope_hint=self._infer_scope_hint(sentence, observation=observation),
        )

    def _clean_claim_fragment(self, value: Any) -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return ""
        normalized = normalized.strip(" .,:;!?")
        normalized = re.sub(r"^(?:that|to)\s+", "", normalized, flags=re.IGNORECASE)
        return normalized.strip()

    def _build_raw_claim(
        self,
        *,
        claim_type: str,
        candidate_class: str,
        candidate_type: str,
        predicate: str,
        object_text: str,
        observation_id: str,
        observation: dict[str, Any],
        sentence: str,
        subject_entity_id: str,
        explicitness_score: float,
        scope_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_scope_hint = self._normalize_text(scope_hint)
        if normalized_scope_hint and normalized_scope_hint not in _VALID_TOOL_SCOPES:
            normalized_scope_hint = ""
        return {
            "claim_type": claim_type,
            "candidate_class": candidate_class,
            "candidate_type": candidate_type,
            "predicate": predicate,
            "object_text": object_text,
            "observation_id": observation_id,
            "created_at": self._normalize_iso_timestamp(observation.get("created_at")),
            "sentence": self._normalize_text(sentence),
            "subject_entity_id": self._normalize_text(subject_entity_id) or None,
            "explicitness_score": explicitness_score,
            "source_quality_score": 0.8,
            "scope_hint": normalized_scope_hint,
            "metadata": self._normalize_json_object(metadata),
        }

    def _aggregate_claims(
        self,
        claims: list[dict[str, Any]],
        *,
        observations: list[dict[str, Any]],
        group_key: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        if not claims:
            return []

        recency_scores = self._build_observation_recency_scores(observations)
        aggregated_by_key: dict[str, dict[str, Any]] = {}
        ordered_aggregates: list[dict[str, Any]] = []
        for claim in claims:
            claim_key = self._build_claim_key(claim)
            aggregate = aggregated_by_key.get(claim_key)
            if aggregate is None:
                aggregate = {
                    "claim_type": claim["claim_type"],
                    "candidate_class": claim["candidate_class"],
                    "candidate_type": claim["candidate_type"],
                    "predicate": claim["predicate"],
                    "object_text": claim["object_text"],
                    "subject_entity_id": claim.get("subject_entity_id"),
                    "scope": self._normalize_text(claim.get("scope_hint")) or "",
                    "observation_ids": [],
                    "evidence": [],
                    "explicitness_score": float(claim["explicitness_score"]),
                    "source_quality_score": float(claim["source_quality_score"]),
                    "group_key": group_key,
                    "session_id": session_id,
                    "metadata": self._normalize_json_object(claim.get("metadata")),
                }
                aggregated_by_key[claim_key] = aggregate
                ordered_aggregates.append(aggregate)

            observation_id = self._normalize_text(claim.get("observation_id"))
            if observation_id and observation_id not in aggregate["observation_ids"]:
                aggregate["observation_ids"].append(observation_id)

            aggregate["evidence"].append(
                {
                    "observation_id": observation_id,
                    "excerpt": claim["sentence"],
                    "created_at": claim["created_at"],
                }
            )
            if not aggregate.get("subject_entity_id") and claim.get("subject_entity_id"):
                aggregate["subject_entity_id"] = claim["subject_entity_id"]
            if not aggregate.get("scope") and claim.get("scope_hint"):
                aggregate["scope"] = self._normalize_text(claim.get("scope_hint"))
            if claim.get("metadata"):
                aggregate["metadata"] = self._merge_json_objects(aggregate.get("metadata"), claim.get("metadata"))

        for aggregate in ordered_aggregates:
            support_count = len(aggregate["observation_ids"])
            recency_score = max(
                (recency_scores.get(observation_id, 0.0) for observation_id in aggregate["observation_ids"]),
                default=0.0,
            )
            aggregate["support_count"] = support_count
            aggregate["recency_score"] = round(recency_score, 4)
            aggregate["confidence"] = self._calculate_claim_confidence(
                support_count=support_count,
                recency_score=aggregate["recency_score"],
                explicitness_score=float(aggregate["explicitness_score"]),
            )
        return ordered_aggregates

    def _build_claim_key(self, claim: dict[str, Any]) -> str:
        parts = [
            self._normalize_text(claim.get("claim_type")).lower(),
            self._normalize_text(claim.get("predicate")).lower(),
            self._normalize_text(claim.get("object_text")).lower(),
            self._normalize_text(claim.get("subject_entity_id")).lower(),
            self._normalize_text(claim.get("scope_hint") or claim.get("scope")).lower(),
        ]
        return "|".join(parts)

    def _build_observation_recency_scores(
        self,
        observations: list[dict[str, Any]],
    ) -> dict[str, float]:
        timestamps: list[tuple[str, float]] = []
        for observation in observations:
            observation_id = self._normalize_text(observation.get("id"))
            if not observation_id:
                continue
            timestamp = self._coerce_timestamp(observation.get("created_at"))
            if timestamp is None:
                continue
            timestamps.append((observation_id, timestamp))

        if not timestamps:
            return {}

        minimum = min(value for _, value in timestamps)
        maximum = max(value for _, value in timestamps)
        if maximum <= minimum:
            return dict.fromkeys((observation_id for observation_id, _ in timestamps), 1.0)

        return {
            observation_id: round((value - minimum) / (maximum - minimum), 4)
            for observation_id, value in timestamps
        }

    def _coerce_timestamp(self, value: Any) -> float | None:
        normalized = self._normalize_iso_timestamp(value)
        if not normalized:
            return None
        try:
            return datetime.fromisoformat(normalized.replace("Z", _UTC_OFFSET)).timestamp()
        except ValueError:
            return None

    def _calculate_claim_confidence(
        self,
        *,
        support_count: int,
        recency_score: float,
        explicitness_score: float,
    ) -> float:
        frequency_score = min(1.0, support_count / 3.0)
        confidence = 0.25 + (0.35 * frequency_score) + (0.3 * recency_score) + (0.1 * explicitness_score)
        return round(min(0.99, confidence), 4)

    def _build_reflection_entry(self, aggregate: dict[str, Any]) -> dict[str, Any]:
        return {
            "claim_type": aggregate["claim_type"],
            "summary": aggregate["evidence"][0]["excerpt"] if aggregate["evidence"] else "",
            "predicate": aggregate["predicate"],
            "object_text": aggregate["object_text"],
            "confidence": aggregate["confidence"],
            "support_count": aggregate["support_count"],
            "explicitness_score": aggregate["explicitness_score"],
            "recency_score": aggregate["recency_score"],
            "source_quality_score": aggregate["source_quality_score"],
            "observation_ids": aggregate["observation_ids"],
            "subject_entity_id": aggregate.get("subject_entity_id"),
            "evidence": aggregate["evidence"][:3],
        }

    def _build_candidate_blueprint(
        self,
        aggregate: dict[str, Any],
        *,
        group: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_scope = self._normalize_text(aggregate.get("scope")) or self._normalize_text(group.get("scope")) or "agent_private"
        return {
            "candidate_class": aggregate["candidate_class"],
            "candidate_type": aggregate["candidate_type"],
            "predicate": aggregate["predicate"],
            "object_text": aggregate["object_text"],
            "confidence": aggregate["confidence"],
            "support_count": aggregate["support_count"],
            "explicitness_score": aggregate["explicitness_score"],
            "recency_score": aggregate["recency_score"],
            "source_quality_score": aggregate["source_quality_score"],
            "subject_entity_id": aggregate.get("subject_entity_id"),
            "scope": candidate_scope,
            "metadata": self._merge_json_objects(
                {
                    "claim_type": aggregate["claim_type"],
                    "source_observation_ids": aggregate["observation_ids"],
                    "observation_group_key": group["group_key"],
                    "session_id": group["session_id"],
                    "evidence": aggregate["evidence"][:3],
                },
                aggregate.get("metadata"),
            ),
        }

    def _build_lasting_update_entry(self, aggregate: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "proposed",
            "candidate_class": aggregate["candidate_class"],
            "candidate_type": aggregate["candidate_type"],
            "predicate": aggregate["predicate"],
            "object_text": aggregate["object_text"],
            "confidence": aggregate["confidence"],
            "support_count": aggregate["support_count"],
        }

    def _summarize_group_reflection(
        self,
        *,
        group: dict[str, Any],
        observations: list[dict[str, Any]],
        aggregated_claims: list[dict[str, Any]],
    ) -> str:
        session_id = self._normalize_text(group.get("session_id")) or self._session_id
        observation_count = len(observations)
        if not aggregated_claims:
            return f"Processed {observation_count} observations for session {session_id} with no durable updates."

        return (
            f"Processed {observation_count} observations for session {session_id} and extracted "
            f"{len(aggregated_claims)} structured updates."
        )

    def _load_observation_entity_ids(
        self,
        observations: list[dict[str, Any]],
    ) -> dict[str, str]:
        observation_entity_ids: dict[str, str] = {}
        pending_names: dict[str, list[str]] = {}
        for observation in observations:
            observation_id = self._normalize_text(observation.get("id"))
            if not observation_id:
                continue
            entity_id, normalized_names = self._extract_observation_entity_link(observation)
            if entity_id:
                observation_entity_ids[observation_id] = entity_id
                continue
            if normalized_names:
                pending_names[observation_id] = normalized_names

        for observation_id, normalized_names in pending_names.items():
            if observation_id in observation_entity_ids:
                continue
            entity_id = self._find_entity_id_by_normalized_names(normalized_names)
            if entity_id:
                observation_entity_ids[observation_id] = entity_id
        return observation_entity_ids

    def _extract_observation_entity_link(self, observation: dict[str, Any]) -> tuple[str, list[str]]:
        normalized_names: list[str] = []
        entity_refs = self._extract_entity_refs({"metadata": observation.get("metadata")})
        for entity_ref in entity_refs:
            if isinstance(entity_ref, dict):
                entity_id = self._normalize_text(entity_ref.get("id"))
                if entity_id:
                    return entity_id, []
                normalized_name = self._normalize_entity_name(
                    entity_ref.get("normalized_name")
                    or entity_ref.get("canonical_name")
                    or entity_ref.get("name")
                )
            else:
                normalized_name = self._normalize_entity_name(entity_ref)
            if normalized_name:
                normalized_names.append(normalized_name)
        return "", normalized_names

    def _find_entity_id_by_normalized_names(self, normalized_names: list[str]) -> str:
        client = self._client
        if client is None:
            return ""

        for normalized_name in normalized_names:
            if not normalized_name:
                continue
            try:
                entity_rows = client.select(
                    "memory_entities",
                    columns="id",
                    filters={
                        "profile": f"eq.{self._profile}",
                        "normalized_name": f"eq.{normalized_name}",
                        "status": _FILTER_STATUS_ACTIVE,
                    },
                    order=_ORDER_CREATED_ASC,
                    limit=1,
                )
            except Exception as exc:
                logger.warning("Failed to resolve REM entity link for %s: %s", normalized_name, exc)
                return ""

            if entity_rows:
                return self._normalize_text(entity_rows[0].get("id"))
        return ""

    def _insert_memory_events(
        self,
        event_rows: list[dict[str, Any] | None],
        *,
        fail_fast: bool = False,
    ) -> None:
        client = self._client
        if client is None:
            return

        normalized_rows = [row for row in event_rows if row is not None]
        if not normalized_rows:
            return

        try:
            payload: dict[str, Any] | list[dict[str, Any]]
            payload = normalized_rows[0] if len(normalized_rows) == 1 else normalized_rows
            client.insert("memory_events", payload)
        except Exception as exc:
            logger.warning("Failed to write REM lifecycle events: %s", exc)
            if fail_fast:
                raise

    def _build_rem_run_event_row(
        self,
        *,
        rem_run_id: str,
        scope: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile": self._profile,
            "agent_id": self._agent_identity,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "user_id": self._user_id,
            "session_id": self._session_id,
            "scope": scope,
            "source_mode": "live",
            "writer_role": "agent",
            "actor_id": self._agent_identity,
            "actor_type": "agent",
            "event_type": event_type,
            "target_table": "memory_rem_runs",
            "target_id": rem_run_id,
            "payload": payload,
        }

    def _build_reflection_created_event_row(
        self,
        *,
        reflection: dict[str, Any],
        group: dict[str, Any],
        observation_ids: list[str],
    ) -> dict[str, Any] | None:
        reflection_id = self._normalize_text(reflection.get("id"))
        if not reflection_id:
            return None
        return {
            "profile": self._profile,
            "agent_id": self._agent_identity,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "user_id": self._user_id,
            "session_id": str(group["session_id"]),
            "scope": str(group["scope"]),
            "source_mode": "live",
            "writer_role": "agent",
            "actor_id": self._agent_identity,
            "actor_type": "agent",
            "event_type": "reflection_created",
            "target_table": "memory_reflections",
            "target_id": reflection_id,
            "payload": {
                "rem_run_id": self._normalize_text(reflection.get("rem_run_id")),
                "observation_group_key": str(group["group_key"]),
                "observation_ids": observation_ids,
                "candidate_count": len(reflection.get("candidates") or []),
            },
        }

    def _build_candidate_created_event_row(
        self,
        *,
        candidate: dict[str, Any],
        candidate_blueprint: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidate_id = self._normalize_text(candidate.get("id"))
        if not candidate_id:
            return None
        return {
            "profile": str(candidate.get("profile") or self._profile),
            "agent_id": str(candidate.get("agent_id") or self._agent_identity),
            "workspace": str(candidate.get("workspace") or self._workspace),
            "project_id": str(candidate.get("project_id") or self._project_id),
            "user_id": str(candidate.get("user_id") or self._user_id),
            "session_id": str(candidate.get("session_id") or self._session_id),
            "scope": str(candidate.get("scope") or "agent_private"),
            "source_mode": str(candidate.get("source_mode") or "live"),
            "writer_role": str(candidate.get("writer_role") or "agent"),
            "actor_id": self._agent_identity,
            "actor_type": "agent",
            "event_type": "candidate_created",
            "target_table": "memory_candidates",
            "target_id": candidate_id,
            "payload": {
                "rem_run_id": self._normalize_text(candidate.get("rem_run_id")),
                "candidate_class": str(candidate.get("candidate_class") or ""),
                "candidate_type": str(candidate.get("candidate_type") or ""),
                "confidence": candidate_blueprint.get("confidence"),
                "support_count": candidate_blueprint.get("support_count"),
                "source_observation_ids": candidate_blueprint["metadata"]["source_observation_ids"],
            },
        }

    def _normalize_iso_timestamp(self, value: Any) -> str:
        if isinstance(value, datetime):
            timestamp = value
        else:
            normalized = self._normalize_text(value)
            if not normalized:
                return ""
            try:
                timestamp = datetime.fromisoformat(normalized.replace("Z", _UTC_OFFSET))
            except ValueError:
                return normalized

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()

    def _queue_observation_write(self, payload: dict[str, Any] | None) -> None:
        if payload is None or not self._can_write_observations():
            return

        client = self._client
        if client is None:
            return

        def _write() -> None:
            try:
                inserted_rows = client.insert("memory_observations", payload)
                self._log_observation_events(inserted_rows)
                for observation_row in inserted_rows:
                    self._create_entities_for_observation(observation_row)
            except Exception as exc:
                logger.warning("Failed to write observation: %s", exc)

        with self._sync_lock:
            previous_thread = self._sync_thread
            if (
                previous_thread is not None
                and previous_thread.is_alive()
                and previous_thread is not threading.current_thread()
            ):
                self._join_thread(previous_thread)

            self._sync_thread = threading.Thread(
                target=_write,
                daemon=True,
                name="supabase-observation-sync",
            )
            self._sync_thread.start()

    def _queue_session_end_rem(self, *, session_id: str = "") -> None:
        if self._is_shutdown or not self._initialized or self._client is None:
            return

        normalized_session_id = self._normalize_text(session_id) or self._session_id
        summary_thread = self._sync_thread

        with self._rem_lock:
            previous_thread = self._rem_thread
            if self._should_join_thread(previous_thread):
                self._join_thread(previous_thread)

            self._rem_thread = threading.Thread(
                target=lambda: self._run_session_end_rem_after_summary(
                    normalized_session_id=normalized_session_id,
                    summary_thread=summary_thread,
                ),
                daemon=True,
                name="supabase-session-end-rem",
            )
            self._rem_thread.start()

    def _should_join_thread(self, thread: threading.Thread | None) -> bool:
        return (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        )

    def _run_session_end_rem_after_summary(
        self,
        *,
        normalized_session_id: str,
        summary_thread: threading.Thread | None,
    ) -> None:
        if self._should_join_thread(summary_thread):
            self._join_thread(summary_thread)

        try:
            batch = self.select_observation_batch(
                session_id=normalized_session_id or None,
                limit=1,
            )
        except Exception as exc:
            logger.warning("Failed to prepare session-end REM trigger: %s", exc)
            return

        if not batch:
            return

        try:
            self.run_rem(
                session_id=normalized_session_id or None,
                mode="live",
                trigger="session_end",
            )
        except Exception as exc:
            logger.warning("Failed to run session-end REM: %s", exc)

    def _build_observation_payload(
        self,
        *,
        source_type: str,
        source_mode: str = "live",
        content: str,
        metadata: dict[str, Any] | None = None,
        session_id: str = "",
        scope: str = "agent_private",
        visibility: str = "private",
        writer_role: str = "agent",
    ) -> dict[str, Any] | None:
        normalized_content = self._normalize_text(content)
        if not normalized_content or not self._can_write_observations():
            return None

        normalized_scope = self._normalize_text(scope) or "agent_private"
        if normalized_scope not in _VALID_TOOL_SCOPES:
            normalized_scope = "agent_private"
        normalized_visibility = self._normalize_text(visibility) or "private"
        normalized_writer_role = self._normalize_writer_role(writer_role or "agent")

        return {
            "profile": self._profile,
            "agent_id": self._agent_identity,
            "workspace": self._workspace,
            "project_id": self._project_id,
            "user_id": self._user_id,
            "session_id": session_id or self._session_id,
            "scope": normalized_scope,
            "visibility": normalized_visibility,
            "source_mode": source_mode,
            "writer_role": normalized_writer_role,
            "source_type": source_type,
            "content": normalized_content,
            "content_hash": hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest(),
            "metadata": dict(metadata or {}),
        }

    def _format_turn_content(self, user_content: str, assistant_content: str) -> str:
        parts: list[str] = []
        normalized_user = self._normalize_text(user_content)
        normalized_assistant = self._normalize_text(assistant_content)

        if normalized_user:
            parts.append(f"User:\n{normalized_user}")
        if normalized_assistant:
            parts.append(f"Assistant:\n{normalized_assistant}")

        return "\n\n".join(parts)

    def _format_session_summary(self, messages: list[dict[str, Any]]) -> str:
        summary_lines = ["Session summary:"]
        has_content = False

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = self._normalize_text(message.get("role", "unknown")) or "unknown"
            content = self._normalize_text(message.get("content", ""))
            if not content:
                continue

            summary_lines.append(f"- {role}: {content}")
            has_content = True

        if not has_content:
            return ""

        return "\n".join(summary_lines)

    def _create_entities_for_observation(
        self, observation_row: dict[str, Any]
    ) -> list[dict[str, Any]]:
        client = self._client
        if client is None or not isinstance(observation_row, dict):
            return []

        created_entities: list[dict[str, Any]] = []
        observation_id = self._normalize_text(observation_row.get("id"))
        for entity_ref in self._extract_entity_refs(observation_row):
            payload = self._build_entity_payload(observation_row, entity_ref)
            if payload is None:
                continue

            try:
                inserted_rows = client.insert("memory_entities", payload)
            except SupabaseConstraintError as exc:
                logger.info(
                    "Skipping duplicate entity for observation %s: %s",
                    observation_id or "<unknown>",
                    exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to create entity for observation %s: %s",
                    observation_id or "<unknown>",
                    exc,
                )
                continue

            created_entities.extend(inserted_rows)

        return created_entities

    def _extract_entity_refs(self, observation_row: dict[str, Any]) -> list[Any]:
        metadata = observation_row.get("metadata")
        if not isinstance(metadata, dict):
            return []

        entity_refs = metadata.get("entity_refs")
        if entity_refs is None:
            entity_refs = metadata.get("entities")
        if not isinstance(entity_refs, list):
            return []

        return [entity_ref for entity_ref in entity_refs if isinstance(entity_ref, (dict, str))]

    def _build_entity_payload(
        self,
        observation_row: dict[str, Any],
        entity_ref: Any,
    ) -> dict[str, Any] | None:
        canonical_name = ""
        entity_type = ""
        normalized_name = ""
        aliases: list[str] = []
        confidence: float | None = None
        metadata: dict[str, Any] = {}

        if isinstance(entity_ref, str):
            canonical_name = self._normalize_text(entity_ref)
            entity_type = "concept"
        elif isinstance(entity_ref, dict):
            canonical_name = self._normalize_text(
                entity_ref.get("canonical_name") or entity_ref.get("name")
            )
            entity_type = self._normalize_text(
                entity_ref.get("entity_type") or entity_ref.get("type")
            )
            normalized_name = self._normalize_entity_name(
                entity_ref.get("normalized_name") or canonical_name
            )
            aliases = self._normalize_entity_aliases(entity_ref.get("aliases"))
            confidence = self._normalize_entity_confidence(entity_ref.get("confidence"))
            raw_metadata = entity_ref.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata = dict(raw_metadata)

        if not canonical_name or not entity_type:
            return None
        if not normalized_name:
            normalized_name = self._normalize_entity_name(canonical_name)
        if not normalized_name:
            return None

        metadata.update(
            {
                "observation_id": self._normalize_text(observation_row.get("id")),
                "source_type": self._normalize_text(observation_row.get("source_type")),
            }
        )

        return {
            "profile": str(observation_row.get("profile") or self._profile),
            "workspace": str(observation_row.get("workspace") or self._workspace),
            "scope": str(observation_row.get("scope") or "agent_private"),
            "visibility": str(observation_row.get("visibility") or "private"),
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "normalized_name": normalized_name,
            "status": "active",
            "confidence": confidence,
            "aliases": aliases,
            "metadata": metadata,
        }

    def _normalize_entity_name(self, value: Any) -> str:
        normalized_value = self._normalize_text(value)
        if not normalized_value:
            return ""
        return " ".join(normalized_value.lower().split())

    def _normalize_entity_aliases(self, value: Any) -> list[str]:
        raw_aliases = self._coerce_list(value)
        aliases: list[str] = []
        seen_aliases: set[str] = set()
        for alias in raw_aliases:
            normalized_alias = self._normalize_text(alias)
            if not normalized_alias or normalized_alias in seen_aliases:
                continue
            aliases.append(normalized_alias)
            seen_aliases.add(normalized_alias)
        return aliases

    def _coerce_list(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def _normalize_entity_confidence(self, value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if 0 <= numeric_value <= 1:
            return numeric_value
        return None

    def _normalize_json_object(self, value: dict[str, Any] | None) -> dict[str, Any]:
        return deepcopy(value) if isinstance(value, dict) else {}

    def _normalize_json_array(self, value: list[Any] | None) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return deepcopy(value)
        return [deepcopy(value)]

    def _can_manage_rem_records(self) -> bool:
        return self._initialized and not self._is_shutdown and self._client is not None and bool(self._profile)

    def _select_single_row(self, table: str, row_id: str) -> dict[str, Any]:
        client = self._client
        normalized_row_id = self._normalize_text(row_id)
        if client is None or not normalized_row_id:
            return {}

        filters: dict[str, Any] = {"id": f"eq.{normalized_row_id}"}
        if self._profile:
            filters["profile"] = f"eq.{self._profile}"

        try:
            rows = client.select(table, filters=filters, limit=1)
        except Exception as exc:
            logger.warning("Failed to read %s row %s: %s", table, normalized_row_id, exc)
            return {}

        if not rows:
            return {}
        return dict(rows[0])

    def _insert_single_row(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        row, _ = self._insert_single_row_with_error(table, payload)
        return row

    def _insert_single_row_with_error(
        self,
        table: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], Exception | None]:
        client = self._client
        if client is None:
            return {}, None

        try:
            rows = client.insert(table, payload)
        except Exception as exc:
            logger.warning("Failed to insert %s row: %s", table, exc)
            return {}, exc

        if not rows:
            return {}, None
        return dict(rows[0]), None

    def _is_missing_observation_batch_key_error(self, exc: Exception | None) -> bool:
        if exc is None:
            return False
        message = str(exc)
        return (
            "observation_batch_key" in message
            and (
                "schema cache" in message
                or "does not exist" in message
                or "Could not find" in message
            )
        )

    def _is_observation_batch_key_duplicate_error(self, exc: Exception | None) -> bool:
        if exc is None:
            return False
        message = str(exc)
        return (
            "uq_memory_rem_runs_profile_observation_batch_key_active" in message
            or (
                "duplicate key value violates unique constraint" in message
                and "observation_batch_key" in message
            )
        )

    def _update_rows(
        self,
        table: str,
        payload: dict[str, Any],
        *,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        client = self._client
        if client is None:
            return []

        try:
            rows = client.update(table, payload, filters)
        except Exception as exc:
            logger.warning("Failed to update %s rows with filters %s: %s", table, filters, exc)
            return []
        return [dict(row) for row in rows]

    def _update_single_row(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._client
        normalized_row_id = self._normalize_text(row_id)
        if client is None or not normalized_row_id:
            return {}

        filters: dict[str, Any] = {"id": f"eq.{normalized_row_id}"}
        if self._profile:
            filters["profile"] = f"eq.{self._profile}"

        rows = self._update_rows(table, payload, filters=filters)

        if not rows:
            return {}
        return dict(rows[0])

    def _can_write_observations(self) -> bool:
        return (
            self._initialized
            and not self._is_shutdown
            and not self._write_suppressed
            and self._client is not None
            and bool(self._profile)
        )

    def _can_prefetch(self) -> bool:
        return self._initialized and not self._is_shutdown and self._client is not None

    def _run_prefetch(self, query: str, session_id: str) -> None:
        result, selected_records = self._fetch_prefetch_result(query, session_id=session_id)
        with self._prefetch_lock:
            self._prefetch_result = result
            self._prefetch_records = [dict(record) for record in selected_records]
            self._prefetch_ready = True
            self._prefetch_query = normalized_query
            self._prefetch_session_id = effective_session_id

    def _fetch_prefetch_result(
        self, query: str, *, session_id: str = ""
    ) -> tuple[str, list[dict[str, Any]]]:
        client = self._client
        if client is None or not self._profile:
            return "", []

        filters: dict[str, Any] = {
            "profile": f"eq.{self._profile}",
            "status": _FILTER_STATUS_ACTIVE,
        }
        normalized_query = self._normalize_text(query)
        if normalized_query:
            search_pattern = self._tool_search_pattern(normalized_query)
            filters["or"] = f"(predicate.ilike.{search_pattern},object_text.ilike.{search_pattern})"

        try:
            facts = client.select(
                "memory_facts",
                columns=_PREFETCH_FACT_COLUMNS,
                filters=filters,
                order=_ORDER_UPDATED_DESC_THEN_CREATED_DESC,
                limit=_PREFETCH_SELECT_LIMIT,
            )
        except Exception as exc:
            logger.warning("Failed to prefetch facts: %s", exc)
            return "", []

        eligible_facts = [fact for fact in facts if self._is_prefetch_fact_eligible(fact)]
        sourced_fact_ids = self._load_prefetch_sourced_fact_ids(eligible_facts)
        return self._format_prefetch_result(
            eligible_facts,
            sourced_fact_ids=sourced_fact_ids,
        )

    def _format_prefetch_result(
        self,
        facts: list[dict[str, Any]],
        *,
        sourced_fact_ids: set[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        eligible_facts = [fact for fact in facts if self._is_prefetch_fact_eligible(fact)]
        normalized_sourced_fact_ids = {
            fact_id for fact_id in (sourced_fact_ids or set()) if fact_id
        }
        eligible_facts.sort(
            key=lambda fact: (
                self._normalize_text(fact.get("id")) in normalized_sourced_fact_ids,
                self._parse_timestamp(fact.get("updated_at"))
                or self._parse_timestamp(fact.get("created_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )

        lines: list[str] = []
        selected_facts: list[dict[str, Any]] = []
        current_length = 0
        for fact in eligible_facts:
            has_sources = self._normalize_text(fact.get("id")) in normalized_sourced_fact_ids
            line = self._format_fact_line(fact, sourced=has_sources)
            if not line:
                continue

            added_length = len(line) if not lines else len(line) + 1
            if current_length + added_length > _PREFETCH_CHAR_BUDGET:
                break

            lines.append(line)
            selected_fact = dict(fact)
            selected_fact["has_sources"] = has_sources
            selected_facts.append(selected_fact)
            current_length += added_length

        return "\n".join(lines), selected_facts

    def _load_prefetch_sourced_fact_ids(self, facts: list[dict[str, Any]]) -> set[str]:
        client = self._client
        if client is None:
            return set()

        fact_ids = sorted(
            {
                self._normalize_text(fact.get("id"))
                for fact in facts
                if self._normalize_text(fact.get("id"))
            }
        )
        if not fact_ids:
            return set()

        quoted_fact_ids = ",".join(f'"{fact_id}"' for fact_id in fact_ids)
        try:
            source_rows = client.select(
                "memory_fact_sources",
                columns=_PREFETCH_SOURCE_COLUMNS,
                filters={"fact_id": f"in.({quoted_fact_ids})"},
            )
        except Exception as exc:
            logger.warning("Failed to load prefetch fact sources: %s", exc)
            return set()

        return {
            self._normalize_text(source_row.get("fact_id"))
            for source_row in source_rows
            if self._normalize_text(source_row.get("fact_id"))
        }

    def _is_prefetch_fact_eligible(self, fact: dict[str, Any]) -> bool:
        if str(fact.get("profile") or "") != self._profile:
            return False
        if str(fact.get("status") or "") != "active":
            return False
        if fact.get("superseded_by_fact_id"):
            return False
        if self._timestamp_is_past(fact.get("stale_after")):
            return False
        if self._timestamp_is_past(fact.get("expires_at")):
            return False
        if not self._scope_matches(fact):
            return False
        if not self._visibility_matches(fact):
            return False
        return True

    def _scope_matches(self, fact: dict[str, Any]) -> bool:
        scope = str(fact.get("scope") or "")
        if scope == "agent_private":
            return True
        if scope == "user":
            return bool(self._user_id) and str(fact.get("user_id") or "") == self._user_id
        if scope == "workspace":
            return bool(self._workspace) and str(fact.get("workspace") or "") == self._workspace
        if scope == "global":
            return True
        if scope == "operator_governed":
            return True
        return False

    def _visibility_matches(self, fact: dict[str, Any]) -> bool:
        visibility = str(fact.get("visibility") or "private")
        if visibility == "operator_only":
            return False
        if visibility == "workspace":
            return bool(self._workspace) and str(fact.get("workspace") or "") == self._workspace
        if visibility in {"private", "authorized_agents"}:
            return True
        return False

    def _format_fact_line(self, fact: dict[str, Any], *, sourced: bool = False) -> str:
        prefix = "- [sourced] " if sourced else "- "
        predicate = self._normalize_text(fact.get("predicate"))
        value = self._format_fact_value(fact.get("object_text"), fact.get("object_json"))
        if predicate and value:
            return f"{prefix}{predicate}: {value}"
        if value:
            return f"{prefix}{value}"
        if predicate:
            return f"{prefix}{predicate}"
        return ""

    def _format_fact_value(self, object_text: Any, object_json: Any) -> str:
        normalized_text = self._normalize_text(object_text)
        if normalized_text:
            return normalized_text

        if isinstance(object_json, dict):
            parts = [
                f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                for key, value in sorted(object_json.items())
            ]
            return ", ".join(parts)
        if isinstance(object_json, list):
            return ", ".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in object_json)
        if object_json is None:
            return ""
        return self._normalize_text(object_json)

    def _timestamp_is_past(self, value: Any) -> bool:
        parsed = self._parse_timestamp(value)
        if parsed is None:
            return False
        return parsed <= datetime.now(timezone.utc)

    def _parse_timestamp(self, value: Any) -> datetime | None:
        normalized_value = self._normalize_text(value)
        if not normalized_value:
            return None
        try:
            parsed = datetime.fromisoformat(normalized_value.replace("Z", _UTC_OFFSET))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _normalize_text(self, value: Any) -> str:
        return str(value or "").strip()

    def _handle_search_tool(self, args: dict[str, Any]) -> str:
        client = self._require_tool_client(require_profile=True)
        if client is None:
            return self._tool_error(_PROVIDER_NOT_READY)

        query = self._normalize_text(args.get("query"))
        if not query:
            return self._tool_error("query is required")

        scope = self._normalize_text(args.get("scope"))
        if scope and scope not in _VALID_TOOL_SCOPES:
            return self._tool_error(f"Unsupported scope: {scope}")

        search_pattern = self._tool_search_pattern(query)
        filters: dict[str, Any] = {
            "profile": f"eq.{self._profile}",
            "status": _FILTER_STATUS_ACTIVE,
            "or": f"(predicate.ilike.{search_pattern},object_text.ilike.{search_pattern})",
        }
        if scope:
            filters["scope"] = f"eq.{scope}"

        try:
            facts = client.select(
                "memory_facts",
                columns=_PREFETCH_FACT_COLUMNS,
                filters=filters,
                order=_ORDER_UPDATED_DESC_THEN_CREATED_DESC,
                limit=20,
            )
        except Exception as exc:
            return self._tool_error(f"Search failed: {exc}")

        matched_facts = self._filter_matching_tool_facts(facts, scope)
        results = [
            {
                "result_type": "fact",
                "fact_id": str(fact.get("id") or ""),
                "content": self._format_tool_fact_content(fact),
                "fact_class": str(fact.get("fact_class") or ""),
                "scope": str(fact.get("scope") or ""),
                "updated_at": str(fact.get("updated_at") or fact.get("created_at") or ""),
            }
            for fact in matched_facts
        ]
        self._log_retrieval_events(
            query,
            matched_facts,
            retrieval_stage="facts_first",
            selected_for_prompt=False,
            reason={"surface": "tool_search", "scope": scope or "all"},
        )
        if results:
            return self._tool_result({"results": results})

        try:
            candidate_filters: dict[str, Any] = {
                "profile": f"eq.{self._profile}",
                "or": f"(predicate.ilike.{search_pattern},object_text.ilike.{search_pattern})",
            }
            if scope:
                candidate_filters["scope"] = f"eq.{scope}"
            candidates = client.select(
                "memory_candidates",
                columns="id,predicate,object_text,object_json,candidate_class,scope,workspace,user_id,visibility,updated_at,created_at,confidence,status,metadata",
                filters=candidate_filters,
                order=_ORDER_UPDATED_DESC_THEN_CREATED_DESC,
                limit=20,
            )
        except Exception as exc:
            return self._tool_error(f"Search failed: {exc}")

        matched_candidates = self._filter_matching_tool_candidates(candidates, scope)
        fallback_results = [
            {
                "result_type": "candidate",
                "candidate_id": str(candidate.get("id") or ""),
                "content": self._format_tool_fact_content(candidate),
                "fact_class": str(candidate.get("candidate_class") or ""),
                "scope": str(candidate.get("scope") or ""),
                "updated_at": str(candidate.get("updated_at") or candidate.get("created_at") or ""),
                "status": str(candidate.get("status") or ""),
                "confidence": candidate.get("confidence"),
            }
            for candidate in matched_candidates
        ]
        return self._tool_result({"results": fallback_results})

    def _handle_store_tool(self, args: dict[str, Any]) -> str:
        if not self._can_write_observations():
            return self._tool_error(_PROVIDER_NOT_READY)

        content = self._normalize_text(args.get("content"))
        if not content:
            return self._tool_error("content is required")

        scope = self._normalize_text(args.get("scope")) or self._infer_scope_hint(content)
        if scope and scope not in _VALID_TOOL_SCOPES:
            return self._tool_error(f"Unsupported scope: {scope}")
        metadata = self._normalize_json_object(args.get("metadata"))
        metadata["tool_name"] = "supabase_memory_store"
        visibility = self._normalize_text(args.get("visibility")) or self._default_visibility_for_scope(scope or "agent_private")

        payload = self._build_observation_payload(
            source_type="tool",
            source_mode="live",
            content=content,
            metadata=metadata,
            scope=scope or "agent_private",
            visibility=visibility,
        )
        if payload is None or self._client is None:
            return self._tool_error(_PROVIDER_NOT_READY)

        try:
            inserted = self._client.insert("memory_observations", payload)
        except Exception as exc:
            return self._tool_error(f"Store failed: {exc}")

        self._log_observation_events(inserted)
        observation_id = ""
        if inserted:
            observation_id = str(inserted[0].get("id") or "")
        return self._tool_result({"success": True, "observation_id": observation_id})

    def _handle_inspect_tool(self, args: dict[str, Any]) -> str:
        client = self._require_tool_client(require_profile=True)
        if client is None:
            return self._tool_error(_PROVIDER_NOT_READY)

        fact_id = self._normalize_text(args.get("fact_id"))
        if not fact_id:
            return self._tool_error("fact_id is required")

        try:
            facts = self._select_tool_fact(client, fact_id)
        except Exception as exc:
            return self._tool_error(f"Inspect failed: {exc}")

        if not facts:
            return self._tool_error(f"Fact not found: {fact_id}")

        fact = facts[0]
        if not self._is_prefetch_fact_eligible(fact):
            return self._tool_error(f"Fact not accessible: {fact_id}")
        try:
            provenance_chain = self._load_fact_provenance_chain([fact_id])
        except Exception as exc:
            return self._tool_error(f"Inspect failed: {exc}")

        fact_provenance = provenance_chain.get(fact_id, {"sources": [], "observations": {}})
        source_entries = self._build_inspect_source_entries(fact_provenance)

        self._ensure_fact_lifecycle_events([fact], provenance_chain)
        return self._tool_result(
            {
                "fact_id": str(fact.get("id") or ""),
                "content": self._format_tool_fact_content(fact),
                "fact_class": str(fact.get("fact_class") or ""),
                "scope": str(fact.get("scope") or ""),
                "updated_at": str(fact.get("updated_at") or fact.get("created_at") or ""),
                "sources": source_entries,
            }
        )

    def _handle_reflect_tool(self, args: dict[str, Any]) -> str:
        client = self._require_tool_client(require_profile=True)
        if client is None:
            return self._tool_error(_PROVIDER_NOT_READY)

        scope = self._normalize_text(args.get("scope"))
        if scope and scope not in _VALID_TOOL_SCOPES:
            return self._tool_error(f"Unsupported scope: {scope}")

        session_id = self._normalize_text(args.get("session_id"))
        try:
            result = self.run_rem(
                scope=scope or None,
                session_id=session_id or None,
                trigger="tool_invocation",
            )
        except Exception as exc:
            return self._tool_error(f"Reflect failed: {exc}")

        if str(result.get("status") or "") != "completed":
            status = self._normalize_text(result.get("status")) or "unknown"
            return self._tool_error(f"Reflect failed: REM run status {status}")

        if int(result.get("observations_processed") or 0) <= 0:
            return self._tool_error("No unprocessed observations available for reflection")

        return self._tool_result(result)

    def _load_fact_provenance_chain(
        self, fact_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        client = self._client
        if client is None:
            return {}

        normalized_fact_ids = sorted(
            {
                self._normalize_text(fact_id)
                for fact_id in fact_ids
                if self._normalize_text(fact_id)
            }
        )
        if not normalized_fact_ids:
            return {}

        quoted_fact_ids = ",".join(f'"{fact_id}"' for fact_id in normalized_fact_ids)
        source_rows = client.select(
            "memory_fact_sources",
            columns=_TOOL_SOURCE_COLUMNS,
            filters={"fact_id": f"in.({quoted_fact_ids})"},
            order=_ORDER_CREATED_ASC,
        )

        observation_ids = sorted(
            {
                self._normalize_text(source.get("observation_id"))
                for source in source_rows
                if self._normalize_text(source.get("observation_id"))
            }
        )
        observation_rows: list[dict[str, Any]] = []
        if observation_ids:
            quoted_observation_ids = ",".join(
                f'"{observation_id}"' for observation_id in observation_ids
            )
            observation_rows = client.select(
                "memory_observations",
                columns=_TOOL_OBSERVATION_COLUMNS,
                filters={
                    "id": f"in.({quoted_observation_ids})",
                    "profile": f"eq.{self._profile}",
                },
            )

        observations_by_id = {
            str(row.get("id") or ""): row for row in observation_rows if row.get("id")
        }
        provenance_chain = {
            fact_id: {"sources": [], "observations": {}}
            for fact_id in normalized_fact_ids
        }
        for source in source_rows:
            fact_id = self._normalize_text(source.get("fact_id"))
            if not fact_id:
                continue
            entry = provenance_chain.setdefault(fact_id, {"sources": [], "observations": {}})
            entry["sources"].append(source)

        for entry in provenance_chain.values():
            for source in entry["sources"]:
                observation_id = self._normalize_text(source.get("observation_id"))
                if observation_id and observation_id in observations_by_id:
                    entry["observations"][observation_id] = observations_by_id[observation_id]

        return provenance_chain

    def _ensure_fact_lifecycle_events(
        self,
        facts: list[dict[str, Any]],
        provenance_chain: dict[str, dict[str, Any]],
    ) -> None:
        client = self._client
        if client is None or not facts:
            return

        fact_ids = [
            self._normalize_text(fact.get("id"))
            for fact in facts
            if self._normalize_text(fact.get("id"))
            and provenance_chain.get(self._normalize_text(fact.get("id")), {}).get("sources")
        ]
        if not fact_ids:
            return

        quoted_fact_ids = ",".join(f'"{fact_id}"' for fact_id in sorted(set(fact_ids)))
        try:
            existing_events = client.select(
                "memory_events",
                columns="target_id,event_type",
                filters={
                    "target_table": "eq.memory_facts",
                    "event_type": "eq.fact_created",
                    "target_id": f"in.({quoted_fact_ids})",
                },
            )
        except Exception as exc:
            logger.warning("Failed to load existing fact lifecycle events: %s", exc)
            return

        existing_fact_ids = {
            self._normalize_text(event.get("target_id"))
            for event in existing_events
            if self._normalize_text(event.get("target_id"))
        }
        event_rows = []
        for fact in facts:
            fact_id = self._normalize_text(fact.get("id"))
            if not fact_id or fact_id in existing_fact_ids:
                continue

            event_row = self._build_fact_lifecycle_event_row(
                fact,
                provenance_chain.get(fact_id, {"sources": [], "observations": {}}),
            )
            if event_row is not None:
                event_rows.append(event_row)

        if not event_rows:
            return

        try:
            payload: dict[str, Any] | list[dict[str, Any]]
            payload = event_rows[0] if len(event_rows) == 1 else event_rows
            client.insert("memory_events", payload)
        except Exception as exc:
            logger.warning("Failed to write fact lifecycle event rows: %s", exc)

    def _build_fact_lifecycle_event_row(
        self,
        fact: dict[str, Any],
        fact_provenance: dict[str, Any],
    ) -> dict[str, Any] | None:
        fact_id = self._normalize_text(fact.get("id"))
        source_rows = list(fact_provenance.get("sources", []))
        observations = fact_provenance.get("observations", {})
        if not fact_id or not source_rows:
            return None

        source_observation_ids = [
            self._normalize_text(source.get("observation_id"))
            for source in source_rows
            if self._normalize_text(source.get("observation_id"))
        ]
        first_observation = (
            observations.get(source_observation_ids[0], {})
            if source_observation_ids
            else {}
        )

        return {
            "profile": str(fact.get("profile") or self._profile),
            "agent_id": str(fact.get("agent_id") or self._agent_identity),
            "workspace": str(fact.get("workspace") or self._workspace),
            "project_id": str(fact.get("project_id") or self._project_id),
            "user_id": str(fact.get("user_id") or self._user_id),
            "session_id": self._session_id,
            "scope": str(fact.get("scope") or "agent_private"),
            "source_mode": self._normalize_event_source_mode(
                first_observation.get("source_mode")
            ),
            "writer_role": self._normalize_writer_role(first_observation.get("writer_role")),
            "actor_id": self._agent_identity,
            "actor_type": "agent",
            "event_type": "fact_created",
            "target_table": "memory_facts",
            "target_id": fact_id,
            "payload": {
                "fact_class": str(fact.get("fact_class") or ""),
                "fact_type": str(fact.get("fact_type") or ""),
                "source_count": len(source_rows),
                "source_observation_ids": source_observation_ids,
                "source_links": [
                    {
                        "fact_source_id": self._normalize_text(source.get("id")),
                        "observation_id": self._normalize_text(source.get("observation_id")),
                        "support_role": str(source.get("support_role") or ""),
                        "source_excerpt": str(source.get("source_excerpt") or ""),
                    }
                    for source in source_rows
                ],
            },
        }

    def _filter_matching_tool_candidates(
        self, candidates: list[dict[str, Any]], scope: str
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_scope = str(candidate.get("scope") or "")
            if scope and candidate_scope != scope:
                continue
            if not self._candidate_scope_matches(candidate):
                continue
            if not self._candidate_visibility_matches(candidate):
                continue
            filtered.append(candidate)
        return filtered

    def _candidate_scope_matches(self, candidate: dict[str, Any]) -> bool:
        scope = str(candidate.get("scope") or "")
        if scope == "agent_private":
            return True
        if scope == "user":
            return bool(self._user_id) and str(candidate.get("user_id") or "") == self._user_id
        if scope == "workspace":
            return bool(self._workspace) and str(candidate.get("workspace") or "") == self._workspace
        if scope == "global":
            return True
        if scope == "operator_governed":
            return False
        return False

    def _candidate_visibility_matches(self, candidate: dict[str, Any]) -> bool:
        visibility = str(candidate.get("visibility") or "private")
        if visibility == "operator_only":
            return False
        if visibility == "workspace":
            return bool(self._workspace) and str(candidate.get("workspace") or "") == self._workspace
        if visibility in {"private", "authorized_agents"}:
            return True
        return False

    def _default_visibility_for_scope(self, scope: str) -> str:
        normalized_scope = self._normalize_text(scope) or "agent_private"
        if normalized_scope == "workspace":
            return "workspace"
        return "private"

    def _infer_scope_hint(self, text: str, *, observation: dict[str, Any] | None = None) -> str:
        normalized = self._normalize_text(text).lower()
        observation_scope = self._normalize_text((observation or {}).get("scope"))
        if observation_scope in _VALID_TOOL_SCOPES and observation_scope != "agent_private":
            return observation_scope
        if any(phrase in normalized for phrase in [
            "my name is",
            "my favorite",
            "i prefer",
            "i am ",
            "i'm ",
            "i am the ",
            "i’m ",
        ]):
            return "user"
        if any(phrase in normalized for phrase in [
            "repo location",
            "repo lives at",
            "source of truth doc",
            "file is located at",
            "config file",
            "plugin repo",
        ]):
            return "workspace"
        return observation_scope or ""

    def _merge_json_objects(self, left: Any, right: Any) -> dict[str, Any]:
        merged = self._normalize_json_object(left)
        merged.update(self._normalize_json_object(right))
        return merged

    def _safe_json_loads(self, text: str) -> Any:
        normalized = self._normalize_text(text)
        if not normalized:
            return None
        try:
            return json.loads(normalized)
        except Exception:
            pass
        fence = re.search(r"```(?:json)?\s*(.*?)```", normalized, re.DOTALL | re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except Exception:
                pass
        for start, end in (("{", "}"), ("[", "]")):
            start_idx = normalized.find(start)
            end_idx = normalized.rfind(end)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate = normalized[start_idx:end_idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        return None

    def _extract_semantic_claims_for_group(
        self,
        group: dict[str, Any],
        *,
        observations: list[dict[str, Any]],
        subject_entity_id: str = "",
    ) -> list[dict[str, Any]]:
        prompt_payload = {
            "instructions": [
                "Extract only durable memories worth remembering across sessions.",
                "Ignore transient chatter, acknowledgements, and tool noise.",
                "Return strict JSON with a top-level key 'claims'.",
                "Each claim must include: claim_type, candidate_class, candidate_type, predicate, object_text, confidence, durable, scope_hint, sentence.",
                "scope_hint must be one of: agent_private, user, workspace, global.",
                "Prefer user for personal identity/preferences, workspace for repo/file/project facts.",
            ],
            "session_id": group.get("session_id"),
            "observations": [
                {
                    "id": self._normalize_text(observation.get("id")),
                    "source_type": self._normalize_text(observation.get("source_type")),
                    "scope": self._normalize_text(observation.get("scope")),
                    "content": self._normalize_text(observation.get("content")),
                }
                for observation in observations
                if self._normalize_text(observation.get("content"))
            ],
        }
        try:
            from agent.auxiliary_client import call_llm, extract_content_or_reasoning
            response = call_llm(
                provider="auto",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract durable memory candidates for an agent memory system. Return strict JSON only.",
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=900,
                timeout=30,
            )
            response_text = extract_content_or_reasoning(response)
            parsed = self._safe_json_loads(response_text)
        except Exception as exc:
            logger.debug("Semantic memory extraction unavailable for group %s: %s", group.get("group_key"), exc)
            return []

        if isinstance(parsed, dict):
            raw_claims = parsed.get("claims")
        elif isinstance(parsed, list):
            raw_claims = parsed
        else:
            raw_claims = None
        if not isinstance(raw_claims, list):
            return []

        observation_lookup = {
            self._normalize_text(observation.get("id")): observation for observation in observations if self._normalize_text(observation.get("id"))
        }
        semantic_claims: list[dict[str, Any]] = []
        for raw_claim in raw_claims:
            if not isinstance(raw_claim, dict):
                continue
            if not bool(raw_claim.get("durable", True)):
                continue
            confidence = raw_claim.get("confidence")
            try:
                confidence_value = float(confidence)
            except Exception:
                confidence_value = 0.0
            if confidence_value < 0.45:
                continue
            observation_id = self._normalize_text(raw_claim.get("observation_id"))
            observation = observation_lookup.get(observation_id) or (observations[0] if observations else {})
            predicate = self._normalize_text(raw_claim.get("predicate")).lower()
            object_text = self._clean_claim_fragment(raw_claim.get("object_text"))
            sentence = self._normalize_text(raw_claim.get("sentence") or raw_claim.get("summary") or object_text)
            if not predicate or not object_text:
                continue
            scope_hint = self._normalize_text(raw_claim.get("scope_hint")) or self._infer_scope_hint(sentence, observation=observation)
            if scope_hint not in _VALID_TOOL_SCOPES:
                scope_hint = self._normalize_text(group.get("scope")) or "agent_private"
            semantic_claims.append(
                self._build_raw_claim(
                    claim_type=self._normalize_text(raw_claim.get("claim_type")) or "fact",
                    candidate_class=self._normalize_text(raw_claim.get("candidate_class")) or "fact",
                    candidate_type=self._normalize_text(raw_claim.get("candidate_type")) or "statement",
                    predicate=predicate,
                    object_text=object_text,
                    observation_id=observation_id or self._normalize_text(observation.get("id")),
                    observation=observation,
                    sentence=sentence,
                    subject_entity_id=subject_entity_id,
                    explicitness_score=max(0.7, min(0.99, confidence_value)),
                    scope_hint=scope_hint,
                    metadata={
                        "extraction_method": "semantic_llm",
                        "reason": self._normalize_text(raw_claim.get("reason")),
                        **self._normalize_json_object(raw_claim.get("metadata")),
                    },
                )
            )
        return semantic_claims

    def _should_auto_promote_candidate_row(self, candidate: dict[str, Any]) -> bool:
        if not candidate:
            return False
        if str(candidate.get("status") or "") not in {"proposed", "staged"}:
            return False
        candidate_class = self._normalize_text(candidate.get("candidate_class"))
        if candidate_class not in {"preference", "fact", "decision", "project", "environment", "file"}:
            return False
        object_text = self._normalize_text(candidate.get("object_text")).lower()
        if not object_text:
            return False
        if any(token in object_text for token in ["right now", "today", "currently", "this week", "for lunch", "temporary"]):
            return False
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        scope = self._normalize_text(candidate.get("scope")) or "agent_private"
        threshold = 0.72
        if scope == "workspace":
            threshold = 0.74
        if scope == "global":
            threshold = 0.85
        return confidence >= threshold

    def _maybe_auto_promote_candidate(self, candidate_id: str, candidate_row: dict[str, Any]) -> None:
        if not self._should_auto_promote_candidate_row(candidate_row):
            return
        try:
            self.promote_candidate(candidate_id, promotion_authority="auto")
        except Exception as exc:
            logger.warning("Failed auto-promotion for candidate %s: %s", candidate_id, exc)

    def _format_tool_fact_content(self, fact: dict[str, Any]) -> str:
        line = self._format_fact_line(fact)
        if line.startswith("- "):
            return line[2:]
        return line

    def _tool_search_pattern(self, query: str) -> str:
        normalized_query = query.replace("*", "").replace("%", "").strip()
        return f"*{normalized_query}*"

    def _tool_error(self, message: str, **extra: Any) -> str:
        if _tool_error is not None:
            return _tool_error(message, **extra)
        return json.dumps({"error": str(message), **extra}, ensure_ascii=False)

    def _tool_result(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _log_observation_events(self, observation_rows: list[dict[str, Any]]) -> None:
        client = self._client
        if client is None or not observation_rows:
            return

        event_rows = [self._build_observation_event_row(row) for row in observation_rows]
        event_rows = [row for row in event_rows if row is not None]
        if not event_rows:
            return

        try:
            payload: dict[str, Any] | list[dict[str, Any]]
            payload = event_rows[0] if len(event_rows) == 1 else event_rows
            client.insert("memory_events", payload)
        except Exception as exc:
            logger.warning("Failed to write memory event rows: %s", exc)

    def _build_observation_event_row(
        self, observation_row: dict[str, Any]
    ) -> dict[str, Any] | None:
        target_id = self._normalize_text(observation_row.get("id"))
        if not target_id:
            return None

        metadata = self._normalize_event_metadata(observation_row.get("metadata"))

        return {
            "profile": str(observation_row.get("profile") or self._profile),
            "agent_id": str(observation_row.get("agent_id") or self._agent_identity),
            "workspace": str(observation_row.get("workspace") or self._workspace),
            "project_id": str(observation_row.get("project_id") or self._project_id),
            "user_id": str(observation_row.get("user_id") or self._user_id),
            "session_id": str(observation_row.get("session_id") or self._session_id),
            "scope": str(observation_row.get("scope") or "agent_private"),
            "source_mode": self._normalize_event_source_mode(
                observation_row.get("source_mode")
            ),
            "writer_role": self._normalize_writer_role(observation_row.get("writer_role")),
            "actor_id": str(observation_row.get("agent_id") or self._agent_identity),
            "actor_type": "agent",
            "event_type": "observation_ingested",
            "target_table": "memory_observations",
            "target_id": target_id,
            "payload": {
                "source_type": str(observation_row.get("source_type") or ""),
                "source_mode": str(observation_row.get("source_mode") or ""),
                "session_id": str(observation_row.get("session_id") or self._session_id),
                "content_hash": str(observation_row.get("content_hash") or ""),
                "metadata": metadata,
            },
        }

    def _require_tool_client(self, *, require_profile: bool = False) -> SupabaseClient | None:
        client = self._client
        if not self._initialized or self._is_shutdown or client is None:
            return None
        if require_profile and not self._profile:
            return None
        return client

    def _filter_matching_tool_facts(
        self, facts: list[dict[str, Any]], scope: str
    ) -> list[dict[str, Any]]:
        return [
            fact
            for fact in facts
            if self._is_prefetch_fact_eligible(fact)
            and (not scope or str(fact.get("scope") or "") == scope)
        ]

    def _select_tool_fact(
        self, client: SupabaseClient, fact_id: str
    ) -> list[dict[str, Any]]:
        fact_filters: dict[str, Any] = {"id": f"eq.{fact_id}"}
        if self._profile:
            fact_filters["profile"] = f"eq.{self._profile}"
        return client.select(
            "memory_facts",
            columns=_PREFETCH_FACT_COLUMNS,
            filters=fact_filters,
            limit=1,
        )

    def _build_inspect_source_entries(
        self, fact_provenance: dict[str, Any]
    ) -> list[dict[str, str]]:
        source_entries: list[dict[str, str]] = []
        observations = fact_provenance.get("observations", {})
        for source in fact_provenance.get("sources", []):
            observation_id = str(source.get("observation_id") or "")
            observation = observations.get(observation_id, {})
            source_entries.append(
                {
                    "fact_source_id": str(source.get("id") or ""),
                    "observation_id": observation_id,
                    "source_excerpt": str(source.get("source_excerpt") or ""),
                    "support_role": str(source.get("support_role") or ""),
                    "content": str(observation.get("content") or ""),
                    "source_type": str(observation.get("source_type") or ""),
                    "created_at": str(source.get("created_at") or ""),
                }
            )
        return source_entries

    def _normalize_event_metadata(self, metadata: Any) -> dict[str, Any]:
        if isinstance(metadata, dict):
            return metadata
        if metadata is None:
            return {}
        return {"value": metadata}

    def _log_retrieval_events(
        self,
        query: str,
        facts: list[dict[str, Any]],
        *,
        retrieval_stage: str,
        selected_for_prompt: bool,
        reason: dict[str, Any] | None = None,
        session_id: str = "",
    ) -> None:
        client = self._client
        if client is None or not facts:
            return

        query_text = str(query or "")
        query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        reason_payload = dict(reason or {})
        rows: list[dict[str, Any]] = []
        for index, fact in enumerate(facts, start=1):
            record_id = self._normalize_text(fact.get("id"))
            if not record_id:
                continue

            rows.append(
                {
                    "profile": self._profile,
                    "agent_id": self._agent_identity,
                    "workspace": self._workspace,
                    "project_id": self._project_id,
                    "user_id": self._user_id,
                    "session_id": session_id or self._session_id,
                    "query_text": query_text,
                    "query_hash": query_hash,
                    "retrieval_stage": retrieval_stage,
                    "record_type": "fact",
                    "record_id": record_id,
                    "rank": index,
                    "reason": {
                        **reason_payload,
                        "fact_class": str(fact.get("fact_class") or ""),
                        "scope": str(fact.get("scope") or ""),
                    },
                    "selected_for_prompt": selected_for_prompt,
                }
            )

        if not rows:
            return

        try:
            client.insert("memory_retrieval_events", rows)
        except Exception as exc:
            logger.warning("Failed to write retrieval diagnostic rows: %s", exc)

    def _normalize_event_source_mode(self, value: Any) -> str:
        normalized = self._normalize_text(value)
        if normalized in _VALID_SOURCE_MODES:
            return normalized
        return "live"

    def _normalize_writer_role(self, value: Any) -> str:
        normalized = self._normalize_text(value)
        if normalized in _VALID_WRITER_ROLES:
            return normalized
        return "agent"

    def _cleanup_runtime_resources(self) -> None:
        self._join_thread(self._sync_thread)
        self._join_thread(self._rem_thread)
        self._join_thread(self._prefetch_thread)
        self._sync_thread = None
        self._rem_thread = None
        if self._prefetch_thread is not None and not self._prefetch_thread.is_alive():
            self._prefetch_thread = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:  # pragma: no cover - defensive safety
                logger.warning("Failed to close Supabase memory client: %s", exc)
            finally:
                self._client = None

        with self._prefetch_lock:
            self._prefetch_result = ""
            self._prefetch_records = []
            self._prefetch_ready = False

    def _join_thread(self, thread: threading.Thread | None) -> None:
        if thread is None or not thread.is_alive() or thread is threading.current_thread():
            return

        try:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        except Exception as exc:  # pragma: no cover - defensive safety
            logger.warning("Failed to join background thread %s: %s", thread.name, exc)


def register(ctx: Any) -> None:
    """Register the Supabase memory provider with Hermes."""

    ctx.register_memory_provider(SupabaseMemoryProvider())


def create_rem_run(provider: SupabaseMemoryProvider, **kwargs: Any) -> dict[str, Any]:
    return provider.create_rem_run(**kwargs)


def get_rem_run(provider: SupabaseMemoryProvider, rem_run_id: str) -> dict[str, Any]:
    return provider.get_rem_run(rem_run_id)


def store_reflection(provider: SupabaseMemoryProvider, **kwargs: Any) -> dict[str, Any]:
    return provider.store_reflection(**kwargs)


def get_reflection(provider: SupabaseMemoryProvider, reflection_id: str) -> dict[str, Any]:
    return provider.get_reflection(reflection_id)


def create_candidate(provider: SupabaseMemoryProvider, **kwargs: Any) -> dict[str, Any]:
    return provider.create_candidate(**kwargs)


def get_candidate(provider: SupabaseMemoryProvider, candidate_id: str) -> dict[str, Any]:
    return provider.get_candidate(candidate_id)


def run_rem(provider: SupabaseMemoryProvider, **kwargs: Any) -> dict[str, Any]:
    return provider.run_rem(**kwargs)


def promote_candidate(
    provider: SupabaseMemoryProvider,
    candidate_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return provider.promote_candidate(candidate_id, **kwargs)


__all__ = [
    "SupabaseClient",
    "SupabaseMemoryProvider",
    "__version__",
    "create_candidate",
    "create_rem_run",
    "get_candidate",
    "get_reflection",
    "get_rem_run",
    "promote_candidate",
    "register",
    "run_rem",
    "store_reflection",
]
