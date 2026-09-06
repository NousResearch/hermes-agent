"""Fail-closed public terminal response boundary for Hermes turns."""
from __future__ import annotations

import re
from typing import Any

_PUBLIC_OUTCOMES = ("DONE", "ARGH", "MEETING")
_MARKER_RE = re.compile(r"(?<![A-Z])(DONE|ARGH|MEETING)(?![A-Z])")
_INTERNAL_PLATFORMS = frozenset({"subagent", "internal", "tool"})


def _done_contract_errors(response: str) -> list[str]:
    """Return missing parts of the owner-directed DONE receipt."""
    lowered = response.casefold()
    errors: list[str] = []
    if not re.search(r"what was done|work completed|completed work", lowered):
        errors.append("what was done")
    if not re.search(r"\bevidence\b", lowered):
        errors.append("evidence")
    if not re.search(r"verif(y|ied|ication)|independent readback", lowered):
        errors.append("how the evidence was verified")
    if not re.search(r"advanced the project|project advance|completion advances", lowered):
        errors.append("how completion advanced the project")

    next_objective_count = len(re.findall(r"recommended next objective", lowered))
    if next_objective_count != 1:
        errors.append("exactly one recommended next objective")
    else:
        next_section = lowered[lowered.index("recommended next objective"):]
        if not re.search(r"\bevidence\b", next_section):
            errors.append("next objective evidence")
        if not re.search(r"acceptance[- ]state criteria|acceptance criteria", next_section):
            errors.append("next objective acceptance-state criteria")
        if not re.search(r"verification method|how .*verif", next_section):
            errors.append("next objective verification method")
    return errors


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
        if markers[0] == "DONE":
            errors = _done_contract_errors(response)
            if errors:
                result["final_response"] = (
                    "ARGH\n"
                    "Action required: Hermes cannot claim DONE because the completion receipt "
                    "is missing: " + ", ".join(errors) + ". Continue the technical work and rerun verification."
                )
                result["failed"] = True
                result["completed"] = False
                result["terminal_boundary_enforced"] = True
                result["terminal_boundary_error"] = "DONE response failed the directed completion contract"
                result["done_contract_missing"] = errors
                result["public_terminal_outcome"] = "ARGH"
                return result
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
