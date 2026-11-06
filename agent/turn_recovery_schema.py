"""Tool-schema recovery rungs: rewrite the agent's tool schemas a backend rejected, then retry."""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger(__name__)


def _rewrite_agent_tools(
    agent: Any, _retry: TurnRetryState, rewrite: Callable[[list[dict]], tuple[list[dict], int]],
) -> int:
    """Apply a ``tools.schema_sanitizer`` reactive strip to a private deep copy of ``agent.tools``
    and swap it in; returns how many keywords changed.

    Never in place on the live dicts: ``get_tool_definitions`` hands out shallow copies of a
    process-wide cache, so they are every session's schemas. ``tools_rewritten`` makes the retry
    rebuild its request copy, since the failed attempt's (a deep copy) still holds the old shape."""
    rewritten, changed = rewrite(copy.deepcopy(agent.tools or []))
    if changed:
        agent.tools[:] = rewritten
        _retry.tools_rewritten = True
    return changed


def _recover_tool_schema_unions(agent: Any, classified: Any, _retry: TurnRetryState) -> bool:
    """Apple's ``fm serve`` 400s every request whose tool schemas hold an ``anyOf``/``oneOf`` (the
    built-in ``terminal``, ``skill_manage`` and ``manage_connections`` do), so no full-toolset turn
    reaches the model: collapse each union to its first branch and retry once."""
    if classified.reason != FailoverReason.tool_schema_union or _retry.tool_schema_union_retry_attempted:
        return False
    _retry.tool_schema_union_retry_attempted = True
    from tools.schema_sanitizer import flatten_unions
    flattened = _rewrite_agent_tools(agent, _retry, flatten_unions)
    if not flattened:
        logger.warning("%sTool-schema decode failure but no union to flatten — falling through",
                       agent.log_prefix)
        return False
    from agent.turn_recovery import _vlines  # facade owns the verbose printer; late import breaks the cycle
    _vlines(agent, f"⚠️  Provider could not decode tool-schema unions — flattened {flattened}, retrying...")
    logger.warning("%sTool-schema union recovery: flattened %d union(s) in tool schemas",
                   agent.log_prefix, flattened)
    return True
