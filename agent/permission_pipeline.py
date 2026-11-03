"""Permission Pipeline — Rule-priority based approval for dangerous commands.

This module provides a staged permission pipeline that checks commands through
multiple stages in priority order. Each stage returns a PermissionResult that
determines whether to continue to the next stage or terminate immediately.

Priority ordering: lower number = runs first.
- Stage 10: DANGEROUS_PATTERNS check (BLOCK if matched)
- Stage 50: SmartApproval via LLM (REVIEW/escalate)
- Stage 90: UserConfirmation (APPROVE with confirmation)

Terminal states (return immediately without checking subsequent stages):
- BLOCK: command is blocked, do not execute
- SKIP: bypass all remaining checks (trusted tool)
- REVIEW: escalated to LLM judgment

Non-terminal state (continues to next stage):
- APPROVE: auto-approved by this stage, continue checking

Final fallback: if all stages return APPROVE (or pipeline is empty),
returns explicit APPROVE with reason "Default approval: pipeline complete".
"""

import asyncio
import logging
from enum import IntEnum
from typing import Callable, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """Permission levels in ascending order of permissiveness.

    Values:
        BLOCK (0): Terminal. Command is blocked immediately.
        REVIEW (1): Terminal. Escalated to LLM/smart approval.
        APPROVE (2): Non-terminal. Continue checking next stage.
        SKIP (3): Terminal. Bypass all remaining checks.
    """
    BLOCK = 0
    REVIEW = 1
    APPROVE = 2
    SKIP = 3


class PermissionResult:
    """Result of a permission check stage.

    Attributes:
        level: The PermissionLevel returned by the checker.
        approved: True if the command is approved at this stage.
        reason: Human-readable explanation of the decision.
        requires_confirmation: True if user confirmation is needed before execution.
    """

    __slots__ = ("level", "approved", "reason", "requires_confirmation")

    def __init__(
        self,
        level: PermissionLevel,
        approved: bool,
        reason: str,
        requires_confirmation: bool = False,
    ):
        self.level = level
        self.approved = approved
        self.reason = reason
        self.requires_confirmation = requires_confirmation

    def __repr__(self) -> str:
        return (
            f"PermissionResult(level={self.level.name}, "
            f"approved={self.approved}, reason={self.reason!r}, "
            f"requires_confirmation={self.requires_confirmation})"
        )


# Type alias for a checker function
PermissionChecker = Callable[[str, dict], PermissionResult]
"""Signature: (command: str, context: dict) -> PermissionResult"""


class PermissionPipeline:
    """A staged permission checker that runs stages in priority order.

    Stages are sorted by priority ascending (lower number = runs first).
    Each stage receives the command and context and returns a PermissionResult.

    Terminal stages (return immediately without checking subsequent stages):
    - BLOCK: command blocked
    - SKIP: bypass all checks
    - REVIEW: escalated to LLM

    Non-terminal stage (continues to next stage):
    - APPROVE: checked by this stage, but continues to subsequent stages
    """

    def __init__(self):
        self._stages: List[Tuple[int, str, PermissionChecker]] = []

    def add_stage(
        self,
        name: str,
        checker: PermissionChecker,
        priority: int = 100,
    ) -> None:
        """Add a stage to the permission pipeline.

        Args:
            name: Human-readable name for this stage (e.g., "dangerous_pattern").
            checker: Callable taking (command: str, context: dict) returning PermissionResult.
            priority: Lower values run first. Stages are checked in ascending priority order.
        """
        self._stages.append((priority, name, checker))
        self._stages.sort(key=lambda x: x[0])

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name.

        Returns True if the stage was found and removed, False otherwise.
        """
        for i, (prio, stage_name, checker) in enumerate(self._stages):
            if stage_name == name:
                self._stages.pop(i)
                return True
        return False

    async def check(
        self,
        command: str,
        context: Optional[dict] = None,
    ) -> PermissionResult:
        """Run all stages in priority order.

        Args:
            command: The shell command to check.
            context: Optional dict with additional context (session_key, env_type, etc.).

        Returns:
            PermissionResult. Terminal states (BLOCK/SKIP/REVIEW) return immediately.
            If all stages return APPROVE (or pipeline is empty), returns explicit
            APPROVE with reason "Default approval: pipeline complete".
        """
        if context is None:
            context = {}

        last_approved: Optional[PermissionResult] = None

        for priority, name, checker in self._stages:
            try:
                if asyncio.iscoroutinefunction(checker):
                    result = await checker(command, context)
                else:
                    result = checker(command, context)
            except Exception as e:
                logger.warning("Permission stage '%s' raised: %s", name, e)
                # Treat checker exceptions as APPROVE to avoid blocking commands
                # due to checker bugs
                result = PermissionResult(
                    PermissionLevel.APPROVE,
                    True,
                    f"{name} check error: {e}",
                )

            # Terminal: return immediately
            if result.level == PermissionLevel.BLOCK:
                return result
            if result.level == PermissionLevel.SKIP:
                return result
            if result.level == PermissionLevel.REVIEW:
                return result

            # APPROVE: track and continue to next stage
            if result.level == PermissionLevel.APPROVE:
                last_approved = result

        # All stages reached APPROVE (or pipeline was empty)
        if last_approved is not None:
            return last_approved

        # Pipeline empty — return explicit APPROVE
        return PermissionResult(
            PermissionLevel.APPROVE,
            True,
            "Default approval: pipeline complete, all stages approved",
        )

    @property
    def stages(self) -> List[Tuple[int, str]]:
        """Return list of (priority, name) pairs for registered stages."""
        return [(prio, name) for prio, name, _ in self._stages]

    def clear(self) -> None:
        """Remove all stages from the pipeline."""
        self._stages.clear()


# Convenience function to check if a result is terminal
def is_terminal(result: PermissionResult) -> bool:
    """Return True if the result is a terminal state (BLOCK/SKIP/REVIEW)."""
    return result.level in (
        PermissionLevel.BLOCK,
        PermissionLevel.SKIP,
        PermissionLevel.REVIEW,
    )
