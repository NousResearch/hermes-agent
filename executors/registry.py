#!/usr/bin/env python3
"""
Executor registry — single source of truth for available executors.

Usage:
    registry = create_default_registry()
    adapter = registry.get("claude-code")
    result = await adapter.start(run, config)

The registry owns:
  - The mapping from ExecutorId → adapter instance
  - The mapping from ExecutorId → ExecutorManifest
  - Health status caching (populated via check_all_executors_health)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from executors.types import (
    ExecutorId,
    ExecutorHealthResult,
    ExecutorHealthStatus,
    ExecutorManifest,
    ExecutorCapabilities,
    AgentExecutorAdapter,
    AgentRun,
    ExecutorConfig,
    AdapterStartResult,
    RunEvent,
    RunStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutorRegistry:
    """Registry of all known executor adapters and their manifests."""

    _adapters: Dict[ExecutorId, AgentExecutorAdapter] = field(default_factory=dict)
    _manifests: Dict[ExecutorId, ExecutorManifest] = field(default_factory=dict)
    _health: Dict[ExecutorId, ExecutorHealthResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        manifest: ExecutorManifest,
        adapter: AgentExecutorAdapter,
    ) -> None:
        """Register an executor with its manifest and adapter instance."""
        eid = manifest.id
        self._manifests[eid] = manifest
        self._adapters[eid] = adapter
        # Default health to unknown until first check
        if eid not in self._health:
            self._health[eid] = ExecutorHealthResult(
                executor_id=eid,
                status=ExecutorHealthStatus.UNKNOWN,
            )
        logger.debug("Registered executor: %s", eid)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, executor_id: ExecutorId) -> AgentExecutorAdapter:
        """Get the adapter for an executor. Raises KeyError if not found."""
        if executor_id not in self._adapters:
            raise KeyError(
                f"Executor '{executor_id}' not registered. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[executor_id]

    def get_manifest(self, executor_id: ExecutorId) -> ExecutorManifest:
        """Get the manifest for an executor. Raises KeyError if not found."""
        if executor_id not in self._manifests:
            raise KeyError(
                f"Manifest for '{executor_id}' not found. "
                f"Available: {list(self._manifests.keys())}"
            )
        return self._manifests[executor_id]

    def list_executors(self) -> List[ExecutorManifest]:
        """Return all registered executor manifests."""
        return sorted(
            self._manifests.values(),
            key=lambda m: (0 if m.id == "hermes-local" else 1, m.label),
        )

    def list_available(self) -> List[ExecutorManifest]:
        """Return manifests for executers whose health status is AVAILABLE."""
        return [
            m for m in self.list_executors()
            if self._health.get(m.id, ExecutorHealthResult(
                executor_id=m.id, status=ExecutorHealthStatus.UNKNOWN
            )).status == ExecutorHealthStatus.AVAILABLE
        ]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self, executor_id: ExecutorId) -> ExecutorHealthResult:
        """Get cached health for an executor (or UNKNOWN if never checked)."""
        return self._health.get(
            executor_id,
            ExecutorHealthResult(
                executor_id=executor_id,
                status=ExecutorHealthStatus.UNKNOWN,
            ),
        )

    def set_health(self, result: ExecutorHealthResult) -> None:
        """Cache a health check result."""
        self._health[result.executor_id] = result

    def get_all_health(self) -> Dict[ExecutorId, ExecutorHealthResult]:
        """Return all cached health results."""
        return dict(self._health)


# ---------------------------------------------------------------------------
# Default manifest definitions
# ---------------------------------------------------------------------------

def _default_manifests() -> Dict[ExecutorId, ExecutorManifest]:
    """Return the default manifests for all known executors."""
    return {
        "hermes-local": ExecutorManifest(
            id="hermes-local",
            label="Hermes Local",
            description="Built-in Hermes agent session (same process)",
            capabilities=ExecutorCapabilities(
                structured_tool_calls=True,
                native_diff_events=False,
                reasoning_blocks=True,
                review_gate=True,
                streaming="realtime",
            ),
            default_model="deepseek-v4-pro",
            ui_fidelity="full",
            supports_worktree=False,
        ),
        "claude-code": ExecutorManifest(
            id="claude-code",
            label="Claude Code",
            description="Anthropic Claude Code CLI (claude-code)",
            capabilities=ExecutorCapabilities(
                structured_tool_calls=True,
                native_diff_events=False,
                reasoning_blocks=True,
                review_gate=False,
                streaming="line-buffered",
            ),
            default_model="claude-sonnet-4-6",
            ui_fidelity="full",
            supports_worktree=False,
        ),
        "codex-cli": ExecutorManifest(
            id="codex-cli",
            label="Codex CLI",
            description="OpenAI Codex CLI (codex)",
            capabilities=ExecutorCapabilities(
                structured_tool_calls=True,
                native_diff_events=False,
                reasoning_blocks=True,
                review_gate=False,
                streaming="line-buffered",
            ),
            default_model="gpt-5",
            ui_fidelity="full",
            supports_worktree=False,
        ),
        "deepseek-tui": ExecutorManifest(
            id="deepseek-tui",
            label="DeepSeek TUI",
            description="DeepSeek terminal agent (deepseek-tui) — low-fidelity log-only executor",
            capabilities=ExecutorCapabilities(
                structured_tool_calls=False,
                native_diff_events=False,
                reasoning_blocks=False,
                review_gate=False,
                streaming="batch",
            ),
            default_model="deepseek-v4-flash",
            ui_fidelity="low",
            supports_worktree=False,
        ),
        "opencode": ExecutorManifest(
            id="opencode",
            label="OpenCode",
            description="OpenCode CLI — local open-source coding agent (opencode)",
            capabilities=ExecutorCapabilities(
                structured_tool_calls=True,
                native_diff_events=False,
                reasoning_blocks=True,
                review_gate=False,
                streaming="line-buffered",
            ),
            default_model="deepseek-v4-flash",
            ui_fidelity="full",
            supports_worktree=False,
        ),
    }
