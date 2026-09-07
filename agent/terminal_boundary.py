"""Live adapter from Hermes to canonical Core Control terminal policy."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_INTERNAL_PLATFORMS = frozenset({"subagent", "internal", "tool"})
_CORE_CONTROL_FILE = Path("/Users/peterbeddow/Core/core-control/core_control/public_terminal.py")


def _load_live_policy():
    """Load current Core Control source; never retain a stale policy module."""
    spec = importlib.util.spec_from_file_location(
        f"core_control_public_terminal_live_{_CORE_CONTROL_FILE.stat().st_mtime_ns}",
        _CORE_CONTROL_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Core Control public-terminal policy is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enforce_public_terminal_response(result: dict[str, Any], *, platform: str = "") -> dict[str, Any]:
    if not isinstance(result, dict) or platform in _INTERNAL_PLATFORMS:
        return result
    try:
        return _load_live_policy().validate_public_terminal_response(result)
    except Exception as exc:
        # The public contract must fail closed if Core Control is unavailable.
        result = result if isinstance(result, dict) else {}
        result["final_response"] = (
            "ARGH\n"
            "Action required: continue the governed technical repair and rerun verification. "
            "Core Control was unavailable to validate this owner-facing response."
        )
        result["failed"] = True
        result["completed"] = False
        result["terminal_boundary_enforced"] = True
        result["terminal_boundary_error"] = f"Core Control unavailable: {type(exc).__name__}"
        result["public_terminal_outcome"] = "ARGH"
        return result
