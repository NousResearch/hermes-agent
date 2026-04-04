"""Typed Event System — structured events for agent lifecycle.

Replaces scattered callback functions with a unified event bus that supports
typed event classes, multiple subscribers, and bridging to the existing
HookRegistry (gateway) and ACP event system.

Inspired by agno's RunStarted/ToolCallStarted/ReasoningStep event stream.

Usage:
    bus = EventBus()
    bus.subscribe(ToolCallStarted, my_handler)
    bus.emit(ToolCallStarted(tool_name="web_search", args={"query": "AAPL"}))

    # Or subscribe to all events:
    bus.subscribe(AgentEvent, catch_all_handler)

Bridge pattern (backward-compatible):
    # Wrap existing callbacks into event subscribers
    bus.bridge_callback("tool_progress", existing_callback_fn)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ============================================================================
# Event Base + Concrete Types
# ============================================================================

@dataclass
class AgentEvent:
    """Base class for all agent events. Subscribe to this to catch everything."""
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None


# --- Run lifecycle ---

@dataclass
class RunStarted(AgentEvent):
    """Agent begins processing a user message."""
    model: Optional[str] = None
    turn_number: int = 0


@dataclass
class RunCompleted(AgentEvent):
    """Agent finished processing (all tool calls done, response delivered)."""
    model: Optional[str] = None
    turn_number: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls_made: int = 0
    duration_ms: float = 0.0


@dataclass
class RunError(AgentEvent):
    """Agent encountered an error during processing."""
    error: str = ""
    error_type: str = ""
    recoverable: bool = True


# --- Tool lifecycle ---

@dataclass
class ToolCallStarted(AgentEvent):
    """A tool call is about to execute."""
    tool_call_id: str = ""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    args_preview: str = ""


@dataclass
class ToolCallCompleted(AgentEvent):
    """A tool call finished execution."""
    tool_call_id: str = ""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# --- Streaming ---

@dataclass
class StreamDelta(AgentEvent):
    """A chunk of streaming text from the model."""
    content: str = ""
    role: str = "assistant"


# --- Reasoning ---

@dataclass
class ReasoningStarted(AgentEvent):
    """Model began extended thinking/reasoning."""
    model: Optional[str] = None
    budget_tokens: int = 0


@dataclass
class ReasoningStep(AgentEvent):
    """A structured step in the reasoning chain."""
    step_number: int = 0
    title: str = ""
    action: str = ""
    result: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    next_action: str = "continue"  # continue, validate, final_answer


@dataclass
class ReasoningCompleted(AgentEvent):
    """Extended thinking/reasoning finished."""
    total_steps: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0


# --- Memory & State ---

@dataclass
class MemoryUpdated(AgentEvent):
    """A memory entry was added, replaced, or removed."""
    target: str = ""  # "memory" or "user"
    action: str = ""  # "add", "replace", "remove"
    content_preview: str = ""


@dataclass
class SessionStateChanged(AgentEvent):
    """Session state key-value pair was modified."""
    key: str = ""
    action: str = ""  # "set", "delete", "clear"
    value: Any = None


# --- Session lifecycle ---

@dataclass
class SessionStarted(AgentEvent):
    """A new session began."""
    source: str = ""  # "cli", "telegram", etc.
    model: Optional[str] = None


@dataclass
class SessionEnded(AgentEvent):
    """A session ended."""
    reason: str = ""  # "cli_close", "compression", "new_session"
    total_turns: int = 0


# --- Approval / HITL ---

@dataclass
class ApprovalRequired(AgentEvent):
    """A tool requires user confirmation before executing."""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ApprovalResolved(AgentEvent):
    """User approved or denied a tool execution."""
    tool_name: str = ""
    approved: bool = False
    reason: str = ""


# --- Budget warnings ---

@dataclass
class BudgetWarning(AgentEvent):
    """Context window or cost budget approaching limit."""
    warning_type: str = ""  # "context_window", "cost", "tool_calls"
    current: float = 0.0
    limit: float = 0.0
    message: str = ""


# ============================================================================
# EventBus
# ============================================================================

class EventBus:
    """Publish-subscribe event bus with type-based routing.

    Subscribers registered for a parent class receive all child events.
    E.g., subscribing to AgentEvent receives ToolCallStarted, RunCompleted, etc.
    """

    def __init__(self):
        self._subscribers: Dict[Type[AgentEvent], List[Callable]] = {}

    def subscribe(self, event_type: Type[AgentEvent], handler: Callable) -> None:
        """Register a handler for an event type (and all subtypes)."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: Type[AgentEvent], handler: Callable) -> None:
        """Remove a handler."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: AgentEvent) -> None:
        """Dispatch an event to all matching subscribers.

        Matches subscribers registered for the event's exact type AND any
        parent types in the MRO (so AgentEvent catches everything).
        """
        event_type = type(event)
        for cls in event_type.__mro__:
            if cls is object:
                break
            for handler in self._subscribers.get(cls, []):
                try:
                    handler(event)
                except Exception as e:
                    logger.warning(
                        "Event handler %s failed for %s: %s",
                        handler.__name__ if hasattr(handler, '__name__') else handler,
                        event_type.__name__, e,
                    )

    def subscriber_count(self, event_type: Type[AgentEvent] = None) -> int:
        """Count subscribers. If event_type given, count only that type."""
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(v) for v in self._subscribers.values())

    # ------------------------------------------------------------------
    # Callback Bridge — backward-compatible with AIAgent callbacks
    # ------------------------------------------------------------------

    def bridge_callback(self, callback_name: str, callback_fn: Callable) -> None:
        """Register an existing AIAgent callback as an event subscriber.

        Maps callback names to event types:
            tool_progress    -> ToolCallStarted
            tool_start       -> ToolCallStarted
            tool_complete    -> ToolCallCompleted
            thinking         -> ReasoningStarted
            reasoning        -> ReasoningStep
            stream_delta     -> StreamDelta
            step             -> RunCompleted (per-turn)
            status           -> RunStarted
        """
        if callback_fn is None:
            return

        mapping = {
            "tool_progress": ToolCallStarted,
            "tool_start": ToolCallStarted,
            "tool_complete": ToolCallCompleted,
            "thinking": ReasoningStarted,
            "reasoning": ReasoningStep,
            "stream_delta": StreamDelta,
            "step": RunCompleted,
            "status": RunStarted,
        }

        event_type = mapping.get(callback_name)
        if event_type:
            self.subscribe(event_type, lambda event: _safe_bridge_call(callback_name, callback_fn, event))


def _safe_bridge_call(callback_name: str, callback_fn: Callable, event: AgentEvent) -> None:
    """Adapt typed events back to the callback signatures AIAgent expects."""
    try:
        if callback_name == "tool_progress" and isinstance(event, ToolCallStarted):
            callback_fn(event.tool_name, event.args_preview, event.args)
        elif callback_name == "tool_start" and isinstance(event, ToolCallStarted):
            callback_fn(event.tool_call_id, event.tool_name, event.args)
        elif callback_name == "tool_complete" and isinstance(event, ToolCallCompleted):
            callback_fn(event.tool_call_id, event.tool_name, event.result_preview, event.duration_ms)
        elif callback_name == "stream_delta" and isinstance(event, StreamDelta):
            callback_fn(event.content)
        elif callback_name == "thinking" and isinstance(event, ReasoningStarted):
            callback_fn(event.budget_tokens)
        elif callback_name == "reasoning" and isinstance(event, ReasoningStep):
            callback_fn(event.reasoning)
        elif callback_name == "step" and isinstance(event, RunCompleted):
            callback_fn(event.turn_number, event.tool_calls_made)
        elif callback_name == "status" and isinstance(event, RunStarted):
            callback_fn(event.model, event.turn_number)
    except Exception as e:
        logger.debug("Bridge callback %s failed: %s", callback_name, e)


# ============================================================================
# Gateway HookRegistry Bridge
# ============================================================================

def bridge_to_hook_registry(bus: EventBus, hook_registry: Any) -> None:
    """Subscribe to agent events and forward them as gateway hook emissions.

    Maps typed events to the string-based hook event types that
    gateway/hooks.py HookRegistry expects.
    """
    import asyncio

    event_to_hook = {
        RunStarted: "agent:start",
        RunCompleted: "agent:end",
        ToolCallStarted: "agent:step",
        SessionStarted: "session:start",
        SessionEnded: "session:end",
    }

    def _make_forwarder(hook_event_type: str):
        def _forward(event: AgentEvent):
            context = {
                "event_class": type(event).__name__,
                "session_id": event.session_id,
                "timestamp": event.timestamp,
            }
            # Add type-specific fields
            if isinstance(event, ToolCallStarted):
                context["tool_name"] = event.tool_name
            elif isinstance(event, RunCompleted):
                context["tool_calls_made"] = event.tool_calls_made
                context["duration_ms"] = event.duration_ms

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(hook_registry.emit(hook_event_type, context))
                else:
                    loop.run_until_complete(hook_registry.emit(hook_event_type, context))
            except RuntimeError:
                # No event loop — skip hook emission (CLI mode without asyncio)
                pass
        return _forward

    for event_cls, hook_type in event_to_hook.items():
        bus.subscribe(event_cls, _make_forwarder(hook_type))
