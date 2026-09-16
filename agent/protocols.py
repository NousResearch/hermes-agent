"""
Formal Protocol Interfaces for Hermes-Agent.

These ABCs define the public API contracts for the core agent components.
They serve as DOCUMENTATION and IDE aids (type hints, autocompletion) ONLY.

IMPORTANT: Python ABCs do NOT enforce these interfaces at runtime without
explicit isinstance() checks. This module makes no runtime enforcement —
it exists purely to:
  1. Document the required methods for each interface
  2. Enable static type checkers (mypy, pyright) to catch interface gaps
  3. Provide IDE autocompletion and navigation

No isinstance() checks are performed anywhere in the codebase based on
these protocols.

Interface hierarchy:
  IAgent          — main conversation loop entry point
  IToolRegistry    — tool registration, schema retrieval, dispatch
  IMemoryProvider  — persistent cross-session memory (from agent/memory_provider.py)
  IEventBus        — internal analytics event emission

References:
  - IAgent methods modeled after AIAgent in run_agent.py
  - IToolRegistry modeled after ToolRegistry in tools/registry.py
  - IMemoryProvider mirrors MemoryProvider in agent/memory_provider.py
  - IEventBus mirrors EventBus in agent/hermes/analytics.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set


# ---------------------------------------------------------------------------
# IAgent
# ---------------------------------------------------------------------------


class IAgent(ABC):
    """
    Protocol for the main agent conversation loop.

    The concrete implementation is AIAgent (run_agent.py). This interface
    documents the minimal surface area needed by callers and subsystems
    that interact with the agent without coupling to its internals.
    """

    @abstractmethod
    def run_conversation(self, message: str, **kwargs) -> str:
        """
        Run a single conversation turn.

        Args:
            message: The user's input message.
            **kwargs: Turn options (platform, session_id, etc.).

        Returns:
            The agent's response as a string.
        """

    @abstractmethod
    def get_session_id(self) -> str:
        """
        Return the current session identifier.

        Returns:
            The active session ID string.
        """


# ---------------------------------------------------------------------------
# IToolRegistry
# ---------------------------------------------------------------------------


class IToolRegistry(ABC):
    """
    Protocol for the tool registry.

    The concrete implementation is ToolRegistry (tools/registry.py). It
    collects tool schemas and handlers declared at module-import time and
    dispatches tool calls at runtime.
    """

    @abstractmethod
    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: List[str] | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
    ) -> None:
        """
        Register a tool with its schema and handler.

        Args:
            name: Unique tool name.
            toolset: Logical group (e.g. 'browser', 'file', 'memory').
            schema: OpenAI function-calling format schema dict.
            handler: Callable that executes the tool.
            check_fn: Optional availability check; None means always available.
            requires_env: List of required env var names.
            is_async: True if the handler is an async callable.
            description: Human-readable description.
            emoji: Optional emoji icon.
        """

    @abstractmethod
    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """
        Return OpenAI-format tool schemas for the requested tool names.

        Args:
            tool_names: Set of tool names to include.
            quiet: Suppress check_fn failure log messages.

        Returns:
            List of tool definition dicts in OpenAI function format.
        """

    @abstractmethod
    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """
        Execute a tool handler by name.

        Args:
            name: The tool name to execute.
            args: Positional arguments dict passed to the handler.
            **kwargs: Additional context (platform, session_id, etc.).

        Returns:
            JSON string result or error.
        """


# ---------------------------------------------------------------------------
# IMemoryProvider
# ---------------------------------------------------------------------------


class IMemoryProvider(ABC):
    """
    Protocol for pluggable memory providers.

    The concrete base class is MemoryProvider (agent/memory_provider.py).
    External providers (Honcho, Hindsight, Mem0, etc.) subclass it.
    Built-in memory is always active as the first provider.

    Required properties and methods that every provider must implement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'builtin', 'honcho', 'hindsight')."""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this provider is configured and ready.

        Called during agent init. Should not make network calls —
        only check config and installed deps.
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """
        Initialize for a session.

        kwargs include: hermes_home, platform, agent_context,
        agent_identity, agent_workspace, parent_session_id, user_id.
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return tool schemas this provider exposes.

        Each schema follows OpenAI function-calling format:
        {"name": "...", "description": "...", "parameters": {...}}

        Return empty list if context-only (no tools).
        """

    # -- Optional hooks (implement to opt in) --------------------------------

    def system_prompt_block(self) -> str:
        """Return static text for the system prompt. Default: empty string."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn. Default: empty string."""
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the next turn. Default: no-op."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to the backend. Default: no-op."""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools. Default: raise."""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown. Default: no-op."""

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn. Default: no-op."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends. Default: no-op."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text before context compression. Default: empty string."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """Called on parent agent when a subagent completes. Default: no-op."""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for 'hermes memory setup'. Default: empty list."""
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to the provider's native location. Default: no-op."""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Called when built-in memory tool writes an entry. Default: no-op."""


# ---------------------------------------------------------------------------
# IEventBus
# ---------------------------------------------------------------------------


class IEventBus(ABC):
    """
    Protocol for the internal analytics event bus.

    The concrete implementation is EventBus (agent/hermes/analytics.py).
    It provides thread-safe synchronous event delivery for internal
    analytics and observability — separate from invoke_hook (plugin
    extensibility).
    """

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable[..., None]) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The event type to subscribe to (e.g. 'tool.call').
            handler: Callback function that receives the Event.
        """

    @abstractmethod
    def emit(self, event: Any) -> None:
        """
        Emit an event to all subscribed handlers.

        Delivery is SYNCHRONOUS — all handlers are called in the
        emitting thread. Handlers must NOT block.

        Args:
            event: An Event (or dict-like) with a ``type`` attribute.
        """
