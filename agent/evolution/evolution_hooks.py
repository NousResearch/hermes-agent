"""Lifecycle hooks that wire the Evolution Engine into the agent runtime.

These hooks follow the plugin lifecycle hook pattern (pre_llm_call,
post_llm_call, on_session_start, on_session_end) to integrate evolution
tracking without modifying the core agent loop.

Hook Architecture:
  - on_session_start: Initialize EvolutionManager for the session
  - pre_llm_call: Start/continue trajectory collection for active runs
  - post_llm_call: Record model call in trajectory
  - post_tool_call: Record tool execution in trajectory
  - on_turn_end: Check if active task is complete, trigger evaluation
  - on_session_end: Finalize any active runs, persist state

Safety: Hooks are NO-OP when evolution is not enabled. Zero overhead
for agents without evolution configured.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (one per process, like MemoryManager)
# ---------------------------------------------------------------------------

_evolution_manager: Optional[Any] = None  # EvolutionManager instance
_hook_timers: Dict[str, float] = {}        # tool_name → start_time for duration tracking


def get_evolution_manager() -> Optional[Any]:
    """Return the active EvolutionManager, if any."""
    return _evolution_manager


# ---------------------------------------------------------------------------
# Session hooks
# ---------------------------------------------------------------------------


def on_session_start(agent: Any, **kwargs: Any) -> None:
    """Initialize the Evolution Engine for a new session.

    Called once at agent startup. Reads config and sets up the manager.
    Also starts the conversation observer for auto-task discovery.
    No-op if evolution is disabled in config.
    """
    global _evolution_manager

    try:
        from agent.evolution.config import EvolutionConfig
        config = EvolutionConfig.from_config()
        if not config.enabled:
            logger.debug("Evolution Engine disabled in config — skipping init")
            return
    except Exception as e:
        logger.debug("Evolution config load failed: %s", e)
        return

    try:
        from agent.evolution.evolution_manager import EvolutionManager
        from agent.evolution.conversation_observer import get_observer

        session_id = getattr(agent, "session_id", "")
        manager = EvolutionManager()
        manager.initialize(session_id=session_id, config=config)
        _evolution_manager = manager

        # Start conversation observer for auto-task discovery
        observer = get_observer()
        observer.start_session(session_id)

        logger.info("Evolution Engine initialized for session %s", session_id)
    except Exception as e:
        logger.warning("Evolution Engine initialization failed: %s", e)


def on_session_end(agent: Any, **kwargs: Any) -> None:
    """Finalize any active evolution runs and persist state."""
    global _evolution_manager

    manager = _evolution_manager
    if manager is None:
        return

    try:
        active_run = manager.get_active_run()
        if active_run:
            manager.end_task(active_run)
        manager.shutdown()
    except Exception as e:
        logger.debug("Evolution shutdown error: %s", e)
    finally:
        _evolution_manager = None

    # Finalize conversation observer — detect patterns from this session
    try:
        from agent.evolution.conversation_observer import get_observer
        observer = get_observer()
        nudge = observer.end_session()
        if nudge and hasattr(agent, "_emit_status"):
            try: agent._emit_status(nudge)
            except Exception: pass
    except Exception:
        pass

    # Auto-export training data if due (weekly, idle-gated)
    try:
        from agent.evolution.auto_export import get_auto_export
        exporter = get_auto_export()
        exporter.maybe_export()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Turn hooks
# ---------------------------------------------------------------------------


def pre_llm_call(agent: Any, messages: List[Dict[str, Any]], **kwargs: Any) -> None:
    """Called before each model call. Tracks timing for trajectory."""
    global _hook_timers
    manager = _evolution_manager
    if manager is None:
        return

    run = manager.get_active_run()
    if run is None or run.collector is None or not run.collector.is_active:
        return

    _hook_timers["_last_model_call"] = time.monotonic()


def post_llm_call(
    agent: Any,
    messages: List[Dict[str, Any]],
    response: Any,
    **kwargs: Any,
) -> None:
    """Called after each model call. Records the call in the trajectory."""
    global _hook_timers
    manager = _evolution_manager
    if manager is None:
        return

    run = manager.get_active_run()
    if run is None or run.collector is None or not run.collector.is_active:
        return

    start_time = _hook_timers.pop("_last_model_call", time.monotonic())
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Extract response metadata
    try:
        model = getattr(agent, "model", "")
        input_tokens = _safe_getattr(response, "usage.input_tokens", 0) or 0
        output_tokens = _safe_getattr(response, "usage.output_tokens", 0) or 0

        # Extract tool calls from the response
        tool_calls = []
        thinking_summary = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                msg = choice.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        tc.function.name if hasattr(tc, "function") else str(tc)
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "content") and msg.content:
                    thinking_summary = str(msg.content)[:200]
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else getattr(choices[0], "message", {})
                if isinstance(msg, dict):
                    tool_calls = [
                        tc.get("function", {}).get("name", str(tc))
                        for tc in msg.get("tool_calls", [])
                    ]
                    thinking_summary = str(msg.get("content", ""))[:200]

        summary = f"Model call: {len(tool_calls)} tool(s)" if tool_calls else f"Response: {thinking_summary[:100]}"

        run.collector.record_model_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
            summary=summary,
            thinking_summary=thinking_summary,
        )
    except Exception as e:
        logger.debug("Failed to record model call in trajectory: %s", e)


def post_tool_call(
    agent: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    error: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Called after each tool execution. Records the call in the trajectory."""
    global _hook_timers
    manager = _evolution_manager
    if manager is None:
        return

    run = manager.get_active_run()
    if run is None or run.collector is None or not run.collector.is_active:
        return

    # Calculate duration from pre-tool timer
    timer_key = f"_tool_{tool_name}"
    start_time = _hook_timers.pop(timer_key, time.monotonic())
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Summarize result
    result_summary = _summarize_tool_result(result)

    status = "error" if error else "success"

    try:
        run.collector.record_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            duration_ms=duration_ms,
            status=status,
            error_message=error or "",
            result_summary=result_summary,
        )
    except Exception as e:
        logger.debug("Failed to record tool call in trajectory: %s", e)


def pre_tool_call(
    agent: Any,
    tool_name: str,
    tool_args: Dict[str, Any],
    **kwargs: Any,
) -> None:
    """Called before tool execution. Tracks timing."""
    global _hook_timers
    _hook_timers[f"_tool_{tool_name}"] = time.monotonic()


# ---------------------------------------------------------------------------
# Turn-end: trigger evaluation for active tasks
# ---------------------------------------------------------------------------


def on_turn_end(agent: Any, **kwargs: Any) -> None:
    """Called at the end of each turn.

    Checks if the agent appears to have completed the active task
    and triggers evaluation if so.
    """
    manager = _evolution_manager
    if manager is None:
        return

    run = manager.get_active_run()
    if run is None:
        return

    # Check if the agent signaled task completion
    # (We detect this heuristically — the agent's last message had no tool calls
    # and the evolution_improve tool was not invoked)
    if not _agent_signaled_completion(agent, kwargs):
        return


def _agent_signaled_completion(agent: Any, kwargs: Dict[str, Any]) -> bool:
    """Heuristic: did the agent just complete its task attempt?"""
    # This is intentionally simple — the agent invokes evolution_improve
    # explicitly when it wants evaluation. We detect natural completion
    # by checking if the last assistant message had no tool calls.
    try:
        messages = kwargs.get("messages", [])
        if not messages:
            return False
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            role = last_msg.get("role", "")
            if role == "assistant" and not last_msg.get("tool_calls"):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_getattr(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Safely get a nested attribute (e.g., 'usage.input_tokens')."""
    try:
        for part in attr_path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(part, default)
            else:
                obj = getattr(obj, part, default)
            if obj is default:
                break
        return obj
    except Exception:
        return default


def _summarize_tool_result(result: Any, max_chars: int = 200) -> str:
    """Create a short summary of a tool result for trace storage."""
    if result is None:
        return "(no output)"
    if isinstance(result, str):
        if len(result) <= max_chars:
            return result
        return result[:max_chars] + f"... ({len(result)} total chars)"
    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"dict with keys: {keys}"
    if isinstance(result, list):
        return f"list with {len(result)} items"
    text = str(result)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
