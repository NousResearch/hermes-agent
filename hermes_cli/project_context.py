"""Project-boundary primitives for long-running session work.

H2/H3 minimal implementation:
- ActiveProjectContext: per-session metadata stored outside the prompt prefix.
- DurableWriteIntent: explicit metadata required for writes to global stores.
- validation helpers for memory/skill durable writes.

The active project context is persisted in SessionDB.state_meta under
``project:<session_id>``.  It is intentionally loaded when building dynamic
session context, not injected by mutating the static system prompt mid-session.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_SCOPES = frozenset({"global", "user", "project", "local", "none"})
PROJECT_DERIVED_SCOPES = frozenset({"project", "local"})
_MUTATING_SKILL_ACTIONS = frozenset({"create", "edit", "patch", "delete", "write_file", "remove_file"})
_MUTATING_MEMORY_ACTIONS = frozenset({"add", "replace", "remove"})


@dataclass
class ActiveProjectContext:
    """Project metadata attached to a session.

    This is deliberately small and declarative so it can be shown in the
    dynamic session context and used by write gates without importing project
    facts into global memory.
    """

    project_id: str
    project_name: str
    capsule_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def normalized(self) -> "ActiveProjectContext":
        self.project_id = (self.project_id or "").strip()
        self.project_name = (self.project_name or "").strip()
        self.capsule_path = (self.capsule_path or "").strip()
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        return self

    def validate(self) -> Optional[str]:
        self.normalized()
        if not self.project_id:
            return "project_id is required."
        if not self.project_name:
            return "project_name is required."
        return None

    def to_json(self) -> str:
        self.updated_at = time.time()
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "ActiveProjectContext":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("stored project context is not an object")
        return cls(
            project_id=str(data.get("project_id") or ""),
            project_name=str(data.get("project_name") or ""),
            capsule_path=str(data.get("capsule_path") or ""),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        ).normalized()

    def prompt_lines(self) -> list[str]:
        lines = ["", "## Current Project Context", ""]
        lines.append(f"**Project:** {self.project_id} — {self.project_name}")
        if self.capsule_path:
            lines.append(f"**Capsule path:** `{self.capsule_path}`")
        if self.metadata:
            safe_items = []
            for key in sorted(self.metadata):
                value = self.metadata[key]
                if isinstance(value, (str, int, float, bool)) and str(value).strip():
                    safe_items.append(f"{key}={value}")
            if safe_items:
                lines.append(f"**Metadata:** {', '.join(safe_items[:8])}")
        lines.append(
            "**Durable write boundary:** Project facts, design choices, source evidence, "
            "and task progress are project-local by default. Global memory/skill writes "
            "must declare `scope`, `source_reference`, and set `approved_global=true` "
            "when derived from this project."
        )
        return lines


@dataclass
class DurableWriteIntent:
    """Explicit durable-write metadata supplied by the tool caller."""

    tool_name: str
    action: str
    destination: str
    scope: str = ""
    source_reference: str = ""
    project_id: str = ""
    approved_global: bool = False

    def normalized_scope(self) -> str:
        return (self.scope or "").strip().lower()

    def is_mutating(self) -> bool:
        if self.tool_name == "memory":
            return self.action in _MUTATING_MEMORY_ACTIONS
        if self.tool_name == "skill_manage":
            return self.action in _MUTATING_SKILL_ACTIONS
        return True

    def project_derived(self, active_project: Optional[ActiveProjectContext]) -> bool:
        scope = self.normalized_scope()
        if scope in PROJECT_DERIVED_SCOPES:
            return True
        pid = (self.project_id or "").strip()
        return bool(active_project and pid and pid == active_project.project_id)


_DB_CACHE: Dict[str, Any] = {}


def _meta_key(session_id: str) -> str:
    return f"project:{session_id}"


def _get_session_db() -> Optional[Any]:
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("project context: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("project context: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_project_context(session_id: str) -> Optional[ActiveProjectContext]:
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("project context: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        ctx = ActiveProjectContext.from_json(raw)
    except Exception as exc:
        logger.warning("project context: could not parse stored project for %s: %s", session_id, exc)
        return None
    return ctx if ctx.validate() is None else None


def save_project_context(session_id: str, context: ActiveProjectContext) -> Optional[str]:
    if not session_id:
        return "session_id is required."
    err = context.validate()
    if err:
        return err
    db = _get_session_db()
    if db is None:
        return "Session database is unavailable."
    try:
        db.set_meta(_meta_key(session_id), context.to_json())
    except Exception as exc:
        logger.debug("project context: set_meta failed: %s", exc)
        return f"Failed to persist project context: {exc}"
    return None


def clear_project_context(session_id: str) -> bool:
    """Clear by storing an empty JSON marker.

    SessionDB currently has get/set only for state_meta.  Storing an empty value
    keeps the operation non-invasive and makes load_project_context return None.
    """
    if not session_id:
        return False
    db = _get_session_db()
    if db is None:
        return False
    try:
        db.set_meta(_meta_key(session_id), "")
        return True
    except Exception as exc:
        logger.debug("project context: clear failed: %s", exc)
        return False


def validate_durable_write_intent(
    *,
    session_id: str = "",
    intent: DurableWriteIntent,
    active_project: Optional[ActiveProjectContext] = None,
) -> Optional[str]:
    """Return a rejection message, or None if the durable write may proceed."""
    if not intent.is_mutating():
        return None
    active_project = active_project or load_project_context(session_id)
    if active_project is None:
        return None

    scope = intent.normalized_scope()
    if not scope:
        return (
            "Durable write rejected: this session has an active project context "
            f"({active_project.project_id}). Provide an explicit `scope` "
            "('global', 'user', or 'project') and `source_reference`."
        )
    if scope not in PROJECT_SCOPES:
        return (
            f"Durable write rejected: unknown scope '{intent.scope}'. Use one of: "
            f"{', '.join(sorted(PROJECT_SCOPES))}."
        )
    if not (intent.source_reference or "").strip():
        return (
            "Durable write rejected: active project sessions require a "
            "`source_reference` (capsule path, note path, issue, or explicit user instruction)."
        )
    if intent.project_derived(active_project) and not intent.approved_global:
        return (
            "Durable write rejected: project-derived content cannot be written to "
            f"global {intent.tool_name} storage without `approved_global=true`. "
            f"Keep it in the project capsule instead: {active_project.capsule_path or active_project.project_id}."
        )
    return None
