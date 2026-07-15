"""Preserve model-authored control arguments across plugin middleware.

The ``todo`` tool carries model-owned semantic state (plans, goal outcomes,
delivery choices, adaptive effort, and approval requests). Middleware may
observe the call or block it through the normal permission/safety boundary,
but it must never originate or rewrite those arguments. This module performs
only a mechanical provenance copy; it does not inspect or interpret content.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


MODEL_AUTHORED_ARGUMENT_TOOLS = frozenset({"todo"})


def capture_model_authored_tool_args(
    function_name: str,
    function_args: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Capture the exact model-authored payload for a sealed tool."""

    if function_name not in MODEL_AUTHORED_ARGUMENT_TOOLS:
        return None
    return copy.deepcopy(dict(function_args))


def restore_model_authored_tool_args(
    function_name: str,
    candidate_args: Any,
    authoritative_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return middleware output, except for exact model-authored tools."""

    if function_name in MODEL_AUTHORED_ARGUMENT_TOOLS:
        return copy.deepcopy(dict(authoritative_args or {}))
    if isinstance(candidate_args, dict):
        return candidate_args
    return {}


__all__ = [
    "MODEL_AUTHORED_ARGUMENT_TOOLS",
    "capture_model_authored_tool_args",
    "restore_model_authored_tool_args",
]
