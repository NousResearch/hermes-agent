#!/usr/bin/env python3
"""
PromptBuilder — composes executor prompts from user goal + workspace context.

Per-executor cropping rules (from workspace-context-injection.md §3.2):
  claude-code   : full context, all fields, 2000 token cap
  codex-cli     : architecture truncated to 300 chars, max 3 ADRs, 1500 token cap
  opencode      : full context, all fields, 2000 token cap
  deepseek-tui  : minimal (overview + sprint + commands + forbidden), 500 token cap
  hermes-local  : full context, all fields, no cap (in-process has unlimited)

Token estimation is a rough heuristic: ~4 chars per token for English, ~2 for CJK.
This is NOT an exact count — it's used only for the cap warning in UI.
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Set

from executors.types import (
    ExecutorId,
    ProjectContext,
    PromptSnapshot,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-executor field inclusion tables
# ---------------------------------------------------------------------------

# Fields that each executor includes (True = include, False = skip)
_FIELD_TABLE: Dict[ExecutorId, Dict[str, bool]] = {
    "claude-code": {
        "project_overview": True,
        "architecture_notes": True,
        "adr_summaries": True,
        "current_sprint": True,
        "common_commands": True,
        "test_commands": True,
        "forbidden_areas": True,
        "coding_conventions": True,
        "recent_tasks": True,
    },
    "codex-cli": {
        "project_overview": True,
        "architecture_notes": True,   # truncated to 300 chars
        "adr_summaries": True,         # max 3
        "current_sprint": True,
        "common_commands": True,
        "test_commands": True,
        "forbidden_areas": True,
        "coding_conventions": True,
        "recent_tasks": True,          # max 3
    },
    "opencode": {
        "project_overview": True,
        "architecture_notes": True,
        "adr_summaries": True,
        "current_sprint": True,
        "common_commands": True,
        "test_commands": True,
        "forbidden_areas": True,
        "coding_conventions": True,
        "recent_tasks": True,          # max 5
    },
    "deepseek-tui": {
        "project_overview": True,
        "architecture_notes": False,
        "adr_summaries": False,
        "current_sprint": True,
        "common_commands": True,
        "test_commands": False,
        "forbidden_areas": True,
        "coding_conventions": False,
        "recent_tasks": False,
    },
    "hermes-local": {
        "project_overview": True,
        "architecture_notes": True,
        "adr_summaries": True,
        "current_sprint": True,
        "common_commands": True,
        "test_commands": True,
        "forbidden_areas": True,
        "coding_conventions": True,
        "recent_tasks": True,
    },
}

# Token caps per executor
_TOKEN_CAP: Dict[ExecutorId, int] = {
    "claude-code": 2000,
    "codex-cli": 1500,
    "opencode": 2000,
    "deepseek-tui": 500,
    "hermes-local": 99999,  # no practical cap
}

# Truncation limits per executor
_ADR_LIMITS: Dict[ExecutorId, int] = {
    "claude-code": 999,
    "codex-cli": 3,
    "opencode": 999,
    "deepseek-tui": 0,
    "hermes-local": 999,
}

_ARCH_TRUNCATION: Dict[ExecutorId, int] = {
    "claude-code": 99999,
    "codex-cli": 300,
    "opencode": 99999,
    "deepseek-tui": 0,
    "hermes-local": 99999,
}

_RECENT_TASK_LIMITS: Dict[ExecutorId, int] = {
    "claude-code": 5,
    "codex-cli": 3,
    "opencode": 5,
    "deepseek-tui": 0,
    "hermes-local": 5,
}

# Priority order for truncation (highest priority kept)
_TRUNCATION_PRIORITY = [
    "forbidden_areas",
    "project_overview",
    "current_sprint",
    "common_commands",
    "coding_conventions",
    "adr_summaries",
    "architecture_notes",
    "recent_tasks",
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Build executor prompts by injecting workspace context.

    Usage::

        builder = PromptBuilder()
        ctx = context_mgr.get_context()
        snapshot = builder.build(
            user_prompt="Refactor auth module",
            context=ctx,
            executor_id="codex-cli",
            include_flags={"architecture_notes": False},
        )
        print(snapshot.injected_prompt)
    """

    def build(
        self,
        user_prompt: str,
        context: ProjectContext,
        executor_id: ExecutorId,
        include_flags: Optional[Dict[str, bool]] = None,
    ) -> PromptSnapshot:
        """Compose the full executor prompt.

        Args:
            user_prompt: The user's original task description.
            context: The loaded project context.
            executor_id: Target executor (claude-code, codex-cli, opencode, etc.).
            include_flags: Optional per-field overrides. True = include, False = exclude.
                           Merged with the executor's default field table.

        Returns:
            PromptSnapshot with user_prompt, injected_prompt, context hash, etc.
        """
        if not context.context_injection_enabled:
            return PromptSnapshot(
                user_prompt=user_prompt,
                injected_prompt=user_prompt,
                context_include_flags={},
                estimated_tokens=_estimate_tokens(user_prompt),
                generated_at=datetime.datetime.now(datetime.timezone.utc),
            )

        # Merge include flags
        field_table = dict(_FIELD_TABLE.get(executor_id, _FIELD_TABLE["hermes-local"]))
        if include_flags:
            field_table.update(include_flags)

        # Build context sections
        sections: List[str] = []
        sections.append("--- Workspace Context ---")

        # Project overview
        if field_table.get("project_overview") and context.project_overview:
            sections.append(f"Project: {context.project_overview}")

        # Current sprint
        if field_table.get("current_sprint") and context.current_sprint:
            sections.append(f"Sprint: {context.current_sprint}")

        # Architecture notes
        if field_table.get("architecture_notes") and context.architecture_notes:
            arch = context.architecture_notes
            limit = _ARCH_TRUNCATION.get(executor_id, 99999)
            if limit and len(arch) > limit:
                arch = arch[:limit] + "…"
            sections.append(f"Architecture: {arch}")

        # ADR summaries
        if field_table.get("adr_summaries") and context.adr_summaries:
            adr_limit = _ADR_LIMITS.get(executor_id, 999)
            adrs = context.adr_summaries[:adr_limit] if adr_limit else []
            if adrs:
                sections.append("ADRs:")
                for a in adrs:
                    sections.append(f"  - {a.id}: {a.decision}")

        # Forbidden areas
        if field_table.get("forbidden_areas") and context.forbidden_areas:
            sections.append(f"Forbidden: {', '.join(context.forbidden_areas)}")

        # Coding conventions
        if field_table.get("coding_conventions") and context.coding_conventions:
            sections.append(f"Conventions: {context.coding_conventions}")

        # Common commands
        if field_table.get("common_commands") and context.common_commands:
            sections.append("Commands:")
            for c in context.common_commands:
                sections.append(f"  {c.label}: {c.command}")

        # Test commands
        if field_table.get("test_commands") and context.test_commands:
            sections.append("Test commands:")
            for c in context.test_commands:
                sections.append(f"  {c.label}: {c.command}")

        # Recent tasks
        if field_table.get("recent_tasks") and context.recent_tasks:
            recent_limit = _RECENT_TASK_LIMITS.get(executor_id, 3)
            recent = context.recent_tasks[-recent_limit:]
            if recent:
                sections.append("Recent tasks:")
                for t in recent:
                    status_icon = "✓" if t.status == "done" else "✗"
                    sections.append(f"  [{status_icon}] {t.title} ({t.executor})")

        sections.append("--- End Context ---")
        sections.append("")
        sections.append(user_prompt)

        injected = "\n".join(sections)

        # Token estimation and truncation warning
        estimated = _estimate_tokens(injected)
        cap = _TOKEN_CAP.get(executor_id, 2000)
        if estimated > cap:
            logger.warning(
                "Prompt may exceed %d token cap (estimated %d) for %s",
                cap, estimated, executor_id,
            )

        return PromptSnapshot(
            user_prompt=user_prompt,
            injected_prompt=injected,
            context_sha=None,  # filled by caller from context_mgr.context_hash()
            context_include_flags=dict(field_table),
            estimated_tokens=estimated,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def build_injection_preview(
        self,
        context: ProjectContext,
        executor_id: ExecutorId,
        include_flags: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Build just the injected context portion (no user prompt), for preview."""
        snapshot = self.build(
            user_prompt="<!-- USER PROMPT WOULD BE HERE -->",
            context=context,
            executor_id=executor_id,
            include_flags=include_flags,
        )
        parts = snapshot.injected_prompt.split("--- End Context ---")
        if len(parts) > 1:
            return parts[0] + "--- End Context ---"
        return snapshot.injected_prompt


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, ~2 for CJK.

    Not exact — used only for cap warnings.
    """
    if not text:
        return 0
    # Count CJK characters separately
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_count = len(text) - cjk_count
    return max(1, int(ascii_count / 4 + cjk_count / 2))


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_default_builder() -> PromptBuilder:
    return PromptBuilder()
