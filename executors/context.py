#!/usr/bin/env python3
"""
WorkspaceContextManager — CRUD for project context stored at .hermes/context.json.

Follows the v0.6 workspace-context-injection.md spec:
  - Persists to JSON (not YAML, to avoid yaml dependency)
  - Supports all 9 context fields + injection control
  - recent_tasks auto-capped at 10 entries
  - Hash-based integrity check for prompt snapshots
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from executors.types import (
    AdrSummary,
    CommandEntry,
    ProjectContext,
    RecentTask,
)

logger = logging.getLogger(__name__)

CONTEXT_FILENAME = "context.json"
MAX_RECENT_TASKS = 10


class WorkspaceContextManager:
    """Manages project context at ``<project_root>/.hermes/context.json``."""

    def __init__(self, project_root: Path):
        self._project_root = Path(project_root).resolve()
        self._context_dir = self._project_root / ".hermes"
        self._context_path = self._context_dir / CONTEXT_FILENAME
        self._context: Optional[ProjectContext] = None

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> ProjectContext:
        """Load context from disk, or return default if file doesn't exist."""
        if self._context is not None:
            return self._context

        if self._context_path.exists():
            try:
                raw = json.loads(self._context_path.read_text())
                self._context = self._from_dict(raw)
                logger.debug("Loaded context from %s", self._context_path)
                return self._context
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Corrupt context.json, using defaults: %s", e)

        self._context = ProjectContext()
        return self._context

    def save(self) -> None:
        """Persist current context to disk."""
        if self._context is None:
            self._context = ProjectContext()

        self._context_dir.mkdir(parents=True, exist_ok=True)

        data = self._to_dict(self._context)
        self._context_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str)
        )
        logger.debug("Saved context to %s", self._context_path)

    def is_loaded(self) -> bool:
        """Check if context has been loaded."""
        return self._context is not None

    # ------------------------------------------------------------------
    # Full get/set
    # ------------------------------------------------------------------

    def get_context(self) -> ProjectContext:
        """Get the full project context (loads if needed)."""
        return self.load()

    def set_context(self, ctx: ProjectContext) -> None:
        """Replace the entire context and save."""
        self._context = ctx
        self.save()

    # ------------------------------------------------------------------
    # Field-level getters / setters
    # ------------------------------------------------------------------

    def get_overview(self) -> str:
        return self.load().project_overview

    def set_overview(self, text: str) -> None:
        ctx = self.load()
        ctx.project_overview = text
        self.save()

    def get_architecture(self) -> str:
        return self.load().architecture_notes

    def set_architecture(self, text: str) -> None:
        ctx = self.load()
        ctx.architecture_notes = text
        self.save()

    def get_adrs(self) -> List[AdrSummary]:
        return self.load().adr_summaries

    def add_adr(self, adr_id: str, title: str, decision: str) -> None:
        ctx = self.load()
        ctx.adr_summaries.append(AdrSummary(id=adr_id, title=title, decision=decision))
        self.save()

    def remove_adr(self, adr_id: str) -> None:
        ctx = self.load()
        ctx.adr_summaries = [a for a in ctx.adr_summaries if a.id != adr_id]
        self.save()

    def get_sprint(self) -> str:
        return self.load().current_sprint

    def set_sprint(self, text: str) -> None:
        ctx = self.load()
        ctx.current_sprint = text
        self.save()

    def get_common_commands(self) -> List[CommandEntry]:
        return self.load().common_commands

    def add_common_command(self, label: str, command: str) -> None:
        ctx = self.load()
        ctx.common_commands.append(CommandEntry(label=label, command=command))
        self.save()

    def remove_common_command(self, label: str) -> None:
        ctx = self.load()
        ctx.common_commands = [c for c in ctx.common_commands if c.label != label]
        self.save()

    def get_test_commands(self) -> List[CommandEntry]:
        return self.load().test_commands

    def add_test_command(self, label: str, command: str) -> None:
        ctx = self.load()
        ctx.test_commands.append(CommandEntry(label=label, command=command))
        self.save()

    def get_forbidden_areas(self) -> List[str]:
        return self.load().forbidden_areas

    def add_forbidden_area(self, path: str) -> None:
        ctx = self.load()
        if path not in ctx.forbidden_areas:
            ctx.forbidden_areas.append(path)
            self.save()

    def remove_forbidden_area(self, path: str) -> None:
        ctx = self.load()
        ctx.forbidden_areas = [p for p in ctx.forbidden_areas if p != path]
        self.save()

    def get_conventions(self) -> str:
        return self.load().coding_conventions

    def set_conventions(self, text: str) -> None:
        ctx = self.load()
        ctx.coding_conventions = text
        self.save()

    def get_recent_tasks(self) -> List[RecentTask]:
        return self.load().recent_tasks

    def add_recent_task(self, task: RecentTask) -> None:
        """Append a recent task, capping at MAX_RECENT_TASKS."""
        ctx = self.load()
        ctx.recent_tasks.append(task)
        if len(ctx.recent_tasks) > MAX_RECENT_TASKS:
            ctx.recent_tasks = ctx.recent_tasks[-MAX_RECENT_TASKS:]
        self.save()

    def get_injection_enabled(self) -> bool:
        return self.load().context_injection_enabled

    def set_injection_enabled(self, enabled: bool) -> None:
        ctx = self.load()
        ctx.context_injection_enabled = enabled
        self.save()

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def context_hash(self) -> str:
        """SHA-256 hash of the current context, for snapshot integrity."""
        ctx = self.load()
        raw = json.dumps(self._to_dict(ctx), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(ctx: ProjectContext) -> Dict[str, Any]:
        return {
            "project_overview": ctx.project_overview,
            "architecture_notes": ctx.architecture_notes,
            "adr_summaries": [asdict(a) for a in ctx.adr_summaries],
            "current_sprint": ctx.current_sprint,
            "common_commands": [asdict(c) for c in ctx.common_commands],
            "test_commands": [asdict(c) for c in ctx.test_commands],
            "forbidden_areas": ctx.forbidden_areas,
            "coding_conventions": ctx.coding_conventions,
            "recent_tasks": [asdict(t) for t in ctx.recent_tasks],
            "context_injection_enabled": ctx.context_injection_enabled,
        }

    @staticmethod
    def _from_dict(raw: Dict[str, Any]) -> ProjectContext:
        return ProjectContext(
            project_overview=raw.get("project_overview", ""),
            architecture_notes=raw.get("architecture_notes", ""),
            adr_summaries=[
                AdrSummary(**a) for a in raw.get("adr_summaries", [])
            ],
            current_sprint=raw.get("current_sprint", ""),
            common_commands=[
                CommandEntry(**c) for c in raw.get("common_commands", [])
            ],
            test_commands=[
                CommandEntry(**c) for c in raw.get("test_commands", [])
            ],
            forbidden_areas=raw.get("forbidden_areas", []),
            coding_conventions=raw.get("coding_conventions", ""),
            recent_tasks=[
                RecentTask(**t) for t in raw.get("recent_tasks", [])
            ],
            context_injection_enabled=raw.get("context_injection_enabled", True),
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_context_manager(project_root: Path) -> WorkspaceContextManager:
    return WorkspaceContextManager(project_root)
