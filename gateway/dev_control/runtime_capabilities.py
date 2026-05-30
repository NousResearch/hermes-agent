"""Runtime capability contract for Dev worker adapters."""

from __future__ import annotations

from typing import Any, Dict, Iterable


RUNTIME_CAPABILITY_KEYS = (
    "can_spawn",
    "can_send",
    "can_stop",
    "can_capture_output",
    "can_open",
    "supports_worktree",
    "supports_terminal",
    "supports_follow_up",
    "supports_cost_reporting",
    "test_only",
)


def runtime_capabilities(
    *,
    runtime_id: str,
    supported_actions: Iterable[str],
    test_only: bool = False,
    launch_supported: bool = False,
) -> Dict[str, bool]:
    actions = {str(action) for action in supported_actions}
    runtime = str(runtime_id or "").lower()
    return {
        "can_spawn": bool(launch_supported and "spawn" in actions and not test_only),
        "can_send": "send" in actions and not test_only,
        "can_stop": "kill" in actions and not test_only,
        "can_capture_output": "capture_output" in actions,
        "can_open": runtime in {"ao", "openhands"} and not test_only,
        "supports_worktree": runtime == "ao",
        "supports_terminal": runtime == "ao",
        "supports_follow_up": "send" in actions and not test_only,
        "supports_cost_reporting": runtime in {"ao", "openhands"},
        "test_only": bool(test_only),
    }


def attach_capabilities(payload: Dict[str, Any], capabilities: Dict[str, bool]) -> Dict[str, Any]:
    result = dict(payload)
    normalized = {key: bool(capabilities.get(key)) for key in RUNTIME_CAPABILITY_KEYS}
    result["capabilities"] = normalized
    result.update(normalized)
    return result

