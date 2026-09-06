"""Fail-closed public terminal response boundary for Hermes turns."""
from __future__ import annotations

import re
from typing import Any

_PUBLIC_OUTCOMES = ("DONE", "ARGH", "MEETING")
_MARKER_RE = re.compile(r"(?<![A-Z])(DONE|ARGH|MEETING)(?![A-Z])")
_INTERNAL_PLATFORMS = frozenset({"subagent", "internal", "tool"})


def enforce_public_terminal_response(result: dict[str, Any], *, platform: str = "") -> dict[str, Any]:
    """Prevent an owner-facing turn from ending without one canonical outcome.

    This is deliberately a boundary adapter, not a completion judge. It does not promote
    incomplete work to DONE. If the model/runtime did not provide a valid public terminal
    marker, it fails closed as ARGH and gives the owner one concrete recovery action.
    Internal child/tool turns remain structured data and are not owner-facing responses.
    """
    if not isinstance(result, dict) or platform in _INTERNAL_PLATFORMS:
        return result

    response = str(result.get("final_response") or "").strip()
    markers = _MARKER_RE.findall(response)
    if len(markers) == 1 and markers[0] in _PUBLIC_OUTCOMES:
        result["public_terminal_outcome"] = markers[0]
        result["terminal_boundary_enforced"] = True
        return result

    result["final_response"] = (
        "ARGH\n"
        "Action required: restart this Hermes process so the enforced public terminal "
        "boundary can take control of the turn. No completion is claimed, and no external "
        "or project change is authorized by this failure."
    )
    result["failed"] = True
    result["completed"] = False
    result["terminal_boundary_enforced"] = True
    result["terminal_boundary_error"] = "owner-facing response lacked exactly one canonical outcome"
    result["public_terminal_outcome"] = "ARGH"
    return result
