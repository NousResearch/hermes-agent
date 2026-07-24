"""Outbound API request contract checks.

Centralizes the tools ↔ tool_choice invariant so the main conversation loop
and the max-iterations summary path share one preflight gate.  Violations
raise :class:`agent.errors.RuntimeContractViolation` and must never reach a
provider.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agent.errors import RuntimeContractViolation

# Fields that only make sense when tools are present on the wire.
_TOOL_DEPENDENT_FIELDS = (
    "tool_choice",
    "parallel_tool_calls",
)


def _tools_are_present(api_kwargs: Mapping[str, Any]) -> bool:
    """Return True when the payload exposes a non-empty tools collection."""
    if "tools" not in api_kwargs:
        return False
    tools = api_kwargs.get("tools")
    if tools is None:
        return False
    if isinstance(tools, (list, tuple, set)):
        return len(tools) > 0
    # Non-sequence tool payloads (rare provider shapes) count as present.
    return True


def validate_api_kwargs(
    api_kwargs: Optional[Mapping[str, Any]],
    *,
    api_mode: Optional[str] = None,
    where: str = "api_request",
) -> None:
    """Assert outbound kwargs satisfy tools/tool_choice invariants.

    Rules:
    - ``tools=None`` must not appear as an explicit key (omit the key instead).
    - ``tool_choice`` / ``parallel_tool_calls`` require a non-empty ``tools`` list.
    - Empty ``tools=[]`` is treated as absent (must not carry tool-dependent fields).

    Raises:
        RuntimeContractViolation: on any invariant breach.
    """
    if not isinstance(api_kwargs, Mapping):
        raise RuntimeContractViolation(
            f"{where}: api_kwargs must be a mapping, got {type(api_kwargs).__name__}",
            field="api_kwargs",
            context={"api_mode": api_mode},
        )

    if "tools" in api_kwargs and api_kwargs.get("tools") is None:
        raise RuntimeContractViolation(
            f"{where}: tools=None is illegal — omit the key entirely "
            f"(api_mode={api_mode!r})",
            field="tools",
            context={"api_mode": api_mode, "keys": sorted(api_kwargs.keys())},
        )

    tools_present = _tools_are_present(api_kwargs)

    for field in _TOOL_DEPENDENT_FIELDS:
        if field not in api_kwargs:
            continue
        value = api_kwargs.get(field)
        if value is None:
            # Explicit null is also illegal — omit the key.
            raise RuntimeContractViolation(
                f"{where}: {field}=None is illegal without tools — omit the key "
                f"(api_mode={api_mode!r})",
                field=field,
                context={"api_mode": api_mode, "tools_present": tools_present},
            )
        if not tools_present:
            raise RuntimeContractViolation(
                f"{where}: {field}={value!r} requires a non-empty tools list "
                f"(api_mode={api_mode!r})",
                field=field,
                context={
                    "api_mode": api_mode,
                    "tools_present": tools_present,
                    "keys": sorted(api_kwargs.keys()),
                },
            )
