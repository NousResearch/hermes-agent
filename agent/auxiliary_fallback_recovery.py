"""Parameter-rejection recovery for auxiliary fallback candidates.

The primary auxiliary request runs the full recovery ladder in ``auxiliary_client``; a fallback
candidate (per-task chain, main fallback chain, discovery) used to get a single shot, so a fallback
model that rejects ``temperature``, ``max_tokens`` or a reasoning field failed the whole task even
though the same parameter rungs would have recovered it on the primary path (#78273, #72351).
This module runs those rungs — and only those — around a candidate's request.
"""
from typing import Any, Awaitable, Callable, Dict, Optional


def _parameter_ladder(first_err: Exception, client: Any, kwargs: Dict[str, Any], *,
                      task: Optional[str], tag: str):
    from agent.auxiliary_client import _LadderRoute, _ladder_parameter_rungs
    # Keyword construction: the route tuple grows with every new ladder rung (a positional 13-tuple
    # broke the moment a sibling PR added ``timeout``); fields this ladder never reads stay None.
    route = _LadderRoute(**{**dict.fromkeys(_LadderRoute._fields), "client": client, "task": task,
                            "tag": tag, "async_mode": bool(tag), "resolved_provider": "",
                            "base_info": str(getattr(client, "base_url", "") or "")})
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    resp, err, _ = yield from _ladder_parameter_rungs(first_err, route, kwargs, max_tokens)
    if err is None:
        return resp
    raise err


def send_with_parameter_rungs(
    send: Callable[[Any, Dict[str, Any]], Any], client: Any, kwargs: Dict[str, Any], *, task: Optional[str],
) -> Any:
    """``send(client, kwargs)``; on a parameter 400, retry through the parameter rungs. Any other
    error (auth, payment, connection) propagates unchanged for the caller's own handling."""
    from agent.auxiliary_client import _drive_ladder
    try:
        return send(client, kwargs)
    except Exception as first_err:
        ladder = _parameter_ladder(first_err, client, kwargs, task=task, tag="")
        return _drive_ladder(ladder, lambda step: send(*step.args))


async def send_with_parameter_rungs_async(
    send: Callable[[Any, Dict[str, Any]], Awaitable[Any]], client: Any, kwargs: Dict[str, Any], *,
    task: Optional[str],
) -> Any:
    """Async twin of :func:`send_with_parameter_rungs`."""
    from agent.auxiliary_client import _drive_ladder_async
    try:
        return await send(client, kwargs)
    except Exception as first_err:
        ladder = _parameter_ladder(first_err, client, kwargs, task=task, tag=" (async)")
        return await _drive_ladder_async(ladder, lambda step: send(*step.args))
