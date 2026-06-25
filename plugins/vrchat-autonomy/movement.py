"""VRChat movement via official OSC /input/* addresses."""

from __future__ import annotations

import time
from typing import Any

INPUT_MAP: dict[str, str] = {
    "forward": "MoveForward",
    "move_forward": "MoveForward",
    "backward": "MoveBackward",
    "back": "MoveBackward",
    "move_back": "MoveBackward",
    "left": "MoveLeft",
    "move_left": "MoveLeft",
    "right": "MoveRight",
    "move_right": "MoveRight",
    "run": "Run",
    "jump": "Jump",
    "look_left": "LookLeft",
    "look_right": "LookRight",
    "look_up": "LookUp",
    "look_down": "LookDown",
}

RESET_INPUTS: tuple[str, ...] = (
    "MoveForward",
    "MoveBackward",
    "MoveLeft",
    "MoveRight",
    "Run",
    "Jump",
    "LookLeft",
    "LookRight",
    "LookUp",
    "LookDown",
)


def send_move(
    direction: str,
    *,
    value: float = 1.0,
    duration_ms: int = 400,
) -> dict[str, Any]:
    """Pulse one VRChat input axis, then reset to zero."""
    from tools.vrchat_osc_tool import vrchat_send_osc

    direction_key = (direction or "").strip().lower()
    if direction_key in {"", "stop", "halt"}:
        return _stop_all(vrchat_send_osc)

    input_name = INPUT_MAP.get(direction_key)
    if not input_name:
        return {
            "success": False,
            "error": f"unknown_direction:{direction}",
            "allowed": sorted({*INPUT_MAP.keys(), "stop"}),
        }

    address = f"/input/{input_name}"
    start = vrchat_send_osc(address, [float(value)])
    if not start.get("success"):
        return {"success": False, "direction": direction_key, "input": input_name, "start": start}

    reset_result = None
    clamped_ms = max(0, min(int(duration_ms), 5000))
    if clamped_ms:
        time.sleep(clamped_ms / 1000.0)
        reset_result = vrchat_send_osc(address, [0.0])

    return {
        "success": bool(reset_result.get("success") if reset_result else True),
        "direction": direction_key,
        "input": input_name,
        "value": float(value),
        "duration_ms": clamped_ms,
        "start": start,
        "reset": reset_result,
    }


def _stop_all(send_fn) -> dict[str, Any]:
    results = []
    for name in RESET_INPUTS:
        results.append({"input": name, "result": send_fn(f"/input/{name}", [0.0])})
    return {
        "success": all(bool(item["result"].get("success")) for item in results),
        "direction": "stop",
        "results": results,
    }
