"""
Replay engine for recorded action sequences.

Executes each step in a recording sequentially via model_tools.handle_function_call().
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def replay_recording(
    recording: dict,
    on_step: Optional[Callable[[int, dict, str], None]] = None,
    on_deviation: Optional[Callable[[int, dict, str], bool]] = None,
    task_id: str = "replay",
) -> dict:
    """Execute each step in a recording sequentially.

    Args:
        recording: Full recording dict from store.get_recording().
        on_step: Optional callback(step_index, step, result) for progress.
        on_deviation: Optional callback(step_index, step, result) when result
            status differs from expected. Return True to continue, False to abort.
            If None, deviations are logged but execution continues.
        task_id: Task ID for tool dispatch.

    Returns:
        Dict with keys: success, steps_completed, steps_total, results, error.
    """
    from model_tools import handle_function_call

    steps = recording.get("steps", [])
    results = []
    error = None

    for i, step in enumerate(steps):
        tool = step.get("tool", "")
        arguments = step.get("arguments", {})
        expected_status = step.get("expected_status", "success")

        try:
            result = handle_function_call(tool, arguments, task_id)
        except Exception as e:
            result = f"Error executing tool '{tool}': {e}"
            logger.error("Replay step %d (%s) raised: %s", i, tool, e)

        # Determine if this step's result matches expectations
        is_error = _is_error_result(result)
        actual_status = "error" if is_error else "success"
        deviated = actual_status != expected_status

        step_result = {
            "step": i,
            "tool": tool,
            "status": actual_status,
            "deviated": deviated,
            "result_preview": result[:500] if len(result) > 500 else result,
        }
        results.append(step_result)

        if on_step:
            try:
                on_step(i, step, result)
            except Exception as cb_err:
                logger.debug("on_step callback error: %s", cb_err)

        if deviated:
            logger.warning(
                "Replay step %d (%s): expected %s, got %s",
                i, tool, expected_status, actual_status,
            )
            if on_deviation:
                try:
                    should_continue = on_deviation(i, step, result)
                except Exception as cb_err:
                    logger.debug("on_deviation callback error: %s", cb_err)
                    should_continue = True
                if not should_continue:
                    error = f"Aborted at step {i} ({tool}): {actual_status} (expected {expected_status})"
                    return {
                        "success": False,
                        "steps_completed": i + 1,
                        "steps_total": len(steps),
                        "results": results,
                        "error": error,
                    }

    return {
        "success": True,
        "steps_completed": len(steps),
        "steps_total": len(steps),
        "results": results,
        "error": None,
    }


def _is_error_result(result: str) -> bool:
    """Heuristic check if a tool result indicates an error."""
    if not result:
        return False
    # Check for common error patterns in tool results
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            if parsed.get("success") is False:
                return True
            if "error" in parsed and parsed["error"]:
                return True
    except (json.JSONDecodeError, TypeError):
        pass
    # Check for error prefix patterns
    lower = result.lower()
    if lower.startswith("error executing tool") or lower.startswith("error:"):
        return True
    return False
