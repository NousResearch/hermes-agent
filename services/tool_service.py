"""
Tool Service — Standardised tool execution interface.

Migrated from agent/tool_executor.py business logic.
Responsibility: Tool discovery, validation, formatting, middleware, budget enforcement.
Does NOT handle dispatch mechanics — those stay in tool_executor.py as the pure dispatcher.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


# ─── Data Transfer Objects ────────────────────────────────────────────────────

@dataclass
class ToolExecutionRequest:
    """Standard tool execution request DTO"""
    function_name: str
    function_args: dict
    tool_call_id: str
    effective_task_id: str
    session_id: str = ""
    turn_id: str = ""
    api_request_id: str = ""


@dataclass
class ToolExecutionResult:
    """Standard tool execution result DTO"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolValidationResult:
    """Tool argument validation result"""
    valid: bool
    errors: list[str] = field(default_factory=list)
    sanitized_args: Optional[dict] = None


# ─── Abstract Interface ───────────────────────────────────────────────────────

class BaseToolService(ABC):
    """
    Abstract tool service — defines the standard interface for tool operations.

    All tool-related business logic (discovery, validation, formatting,
    middleware, budget enforcement) lives here. The tool_executor.py
    is a pure dispatcher that only calls these methods.
    """

    @abstractmethod
    def discover_tools(self, agent) -> frozenset:
        """
        Return the set of tool names the session may invoke.
        Corresponds to _tool_search_scoped_names in tool_executor.py.
        """
        ...

    @abstractmethod
    def validate_args(self, function_name: str, function_args: dict) -> ToolValidationResult:
        """
        Validate and sanitise tool arguments.
        Returns validation result with errors or sanitized args.
        """
        ...

    @abstractmethod
    def format_output(self, result: Any, function_name: str) -> dict[str, Any]:
        """
        Format tool execution result for transcript injection.
        Corresponds to output formatting logic in tool_executor.py.
        """
        ...

    @abstractmethod
    async def apply_middleware(
        self,
        agent,
        request: ToolExecutionRequest,
    ) -> tuple[dict, list[dict]]:
        """
        Apply tool request middleware.
        Returns (modified_args, trace_events).
        Corresponds to _apply_tool_request_middleware_for_agent.
        """
        ...

    @abstractmethod
    def enforce_budget(self, messages: list, task_id: str) -> None:
        """
        Enforce per-turn aggregate budget.
        Raises BudgetExceededError if over limit.
        Uses conflict/resolver for priority-based budget allocation.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        agent,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        """
        Execute a tool with full business logic pipeline.
        This is the main entry point for tool execution.
        """
        ...


# ─── Default Implementation ────────────────────────────────────────────────────

class ToolService(BaseToolService):
    """
    Standard implementation of BaseToolService.

    Encapsulates:
    - Tool discovery & scoping
    - Argument validation & sanitisation
    - Middleware application
    - Budget enforcement (via conflict/resolver)
    - Output formatting
    """

    def discover_tools(self, agent) -> frozenset:
        """Tool discovery — cached on agent, refreshed on registry generation change."""
        try:
            import model_tools
            from tools import tool_search as _ts
            from tools.registry import registry as _registry

            cache_key = "_deferrable_tool_names"
            generation = getattr(_registry, "_generation", 0)
            cached_gen = getattr(agent, "_deferrable_tool_names_gen", None)

            if cached_gen == generation:
                cached = getattr(agent, cache_key, None)
                if cached is not None:
                    return cached

            enabled = getattr(agent, "tools", None) or []
            if not enabled:
                return frozenset()

            deferred = frozenset(
                name for name in enabled
                if getattr(_registry.for_tool(name), "deferrable", False)
            )

            setattr(agent, cache_key, deferred)
            setattr(agent, "_deferrable_tool_names_gen", generation)
            return deferred

        except Exception:
            return frozenset()

    def validate_args(self, function_name: str, function_args: dict) -> ToolValidationResult:
        """Validate tool arguments. Currently a pass-through; extend for schema validation."""
        errors = []
        sanitized = dict(function_args)

        # Placeholder for schema-based validation
        # TODO: integrate JSON schema validation from tools/*.py
        if not function_name:
            errors.append("function_name cannot be empty")

        return ToolValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_args=sanitized if errors else None
        )

    def format_output(self, result: Any, function_name: str) -> dict[str, Any]:
        """Format tool result for transcript injection."""
        if isinstance(result, dict):
            return result
        if hasattr(result, "__dict__"):
            return {"content": str(result)}
        return {"content": result}

    async def apply_middleware(
        self,
        agent,
        request: ToolExecutionRequest,
    ) -> tuple[dict, list[dict]]:
        """Apply middleware chain to tool request."""
        try:
            from hermes_cli.middleware import apply_tool_request_middleware

            result = apply_tool_request_middleware(
                request.function_name,
                request.function_args,
                task_id=request.effective_task_id or "",
                session_id=request.session_id or getattr(agent, "session_id", "") or "",
                tool_call_id=request.tool_call_id or "",
                turn_id=request.turn_id or getattr(agent, "_current_turn_id", "") or "",
                api_request_id=request.api_request_id or getattr(agent, "_current_api_request_id", "") or "",
            )
            payload = result.payload if isinstance(result.payload, dict) else request.function_args
            return payload, list(result.trace)
        except Exception as exc:
            logger.debug("tool_request middleware error: %s", exc)
            return request.function_args, []

    def enforce_budget(self, messages: list, task_id: str) -> None:
        """
        Enforce per-turn aggregate budget.

        Delegates to conflict/resolver for priority-based budget allocation
        when multiple budget constraints conflict.
        """
        try:
            from tools.tool_result_storage import enforce_turn_budget

            # Lazy import to avoid circular dependency
            import sys
            if 'tools.runtime' in sys.modules:
                from tools.runtime import get_active_env
                env = get_active_env(task_id)
            else:
                env = None

            enforce_turn_budget(messages, env=env)
        except ImportError:
            logger.debug("budget enforcement unavailable (tools not loaded)")

    async def execute(
        self,
        agent,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        """
        Full tool execution pipeline:
        1. Validate args
        2. Apply middleware
        3. (placeholder for actual tool call — done by tool_executor.py dispatcher)
        4. Format output
        """
        # Step 1: Validation
        validation = self.validate_args(request.function_name, request.function_args)
        if not validation.valid:
            return ToolExecutionResult(
                success=False,
                result=None,
                error=f"Validation failed: {', '.join(validation.errors)}"
            )

        # Step 2: Middleware
        modified_args, trace = await self.apply_middleware(agent, request)

        # Step 3: Execute (delegated to caller — we return the prepared request)
        # The actual tool call is done by tool_executor.py after this returns

        # Step 4: Return execution context (not the result itself)
        return ToolExecutionResult(
            success=True,
            result={
                "function_name": request.function_name,
                "function_args": modified_args,
                "tool_call_id": request.tool_call_id,
                "trace": trace,
            },
            metadata={"validated": True, "middleware_applied": len(trace) > 0}
        )


# ─── Conflict Integration ─────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    """Raised when a tool execution exceeds its budget allocation."""
    pass


class ToolConflictResolver:
    """
    Tool-specific conflict resolution.
    Uses conflict/resolver for multi-tool budget disputes.
    """

    def resolve_budget_conflict(
        self,
        tool_name: str,
        options: dict[str, Any],
    ) -> Any:
        """
        Resolve budget conflict between multiple tool priority options.
        Delegates to the central ConflictResolver.
        """
        from conflict.resolver import ConflictResolver, ConflictEvent

        resolver = ConflictResolver()
        event = ConflictEvent(
            source_module="AGENTS",
            conflict_type="tool_budget_override",
            options={
                "pinned": options.get("pinned"),
                "config_override": options.get("config_override"),
                "default": options.get("default"),
            }
        )
        resolution = resolver.resolve(event)
        return resolution.winner_value