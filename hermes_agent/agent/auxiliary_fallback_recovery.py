"""Parameter-rejection recovery for auxiliary fallback candidates.

The primary auxiliary request runs the full recovery ladder in ``auxiliary_client``; a fallback
candidate (per-task chain, main fallback chain, discovery) used to get a single shot, so a fallback
model that rejects ``temperature``, ``max_tokens`` or a reasoning field failed the whole task even
though the same parameter rungs would have recovered it on the primary path (#78273, #72351).
This module runs those rungs — and only those — around a candidate's request.
"""
from typing import Any, Awaitable, Callable, Dict, Optional


def reapply_repairs(kwargs: dict[str, Any], repairs: tuple) -> dict[str, Any]:
    """A rebuilt request with the parameter rungs' strips the failed one already carried (a strip returns None
    when its field is already gone)."""
    for strip in repairs:
        kwargs = strip(kwargs) or kwargs
    return kwargs


def _parameter_ladder(first_err: Exception, client: Any, kwargs: dict[str, Any], *,
                      task: Optional[str], tag: str, provider: str, base_url: str):
    """Returns ``(response, None, repairs)`` or ``(None, failure, repairs)``: ``repairs``, the strips that ran."""
    from hermes_agent.agent.auxiliary_client import _failure, _LadderRoute, _ladder_parameter_rungs
    # Keyword construction: the route tuple grows with every new ladder rung (a positional 13-tuple
    # broke the moment a sibling PR added ``timeout``); fields this ladder never reads stay None.
    route = _LadderRoute(**{**dict.fromkeys(_LadderRoute._fields), "client": client, "task": task,
                            "tag": tag, "async_mode": bool(tag), "resolved_provider": provider,
                            "final_model": kwargs.get("model"),
                            "base_info": base_url or str(getattr(client, "base_url", "") or "")})
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    repair = yield from _ladder_parameter_rungs(_failure(route, first_err), route, kwargs, max_tokens)
    return repair.response, repair.failure, repair.repairs


def send_with_parameter_rungs(
    send: Callable[[Any, dict[str, Any]], Any], client: Any, kwargs: dict[str, Any], *, task: Optional[str],
    provider: str = "", base_url: str = "",
) -> Any:
    """``send(client, kwargs)``; on a parameter 400, retry through the parameter rungs. Returns
    ``(response, None, repairs)`` or ``(None, failure, repairs)`` for the caller's own handling; ``repairs`` are the
    strips that ran, so a rebuilt request can carry them and a failure after them is judged as after a spent rung."""
    from hermes_agent.agent.auxiliary_client import _drive_ladder, _outcome
    resp, first_err = _outcome(send, client, kwargs)
    if first_err is None:
        return resp, None, ()
    ladder = _parameter_ladder(first_err, client, kwargs, task=task, tag="", provider=provider, base_url=base_url)
    return _drive_ladder(ladder, lambda step: send(*step.args))


async def send_with_parameter_rungs_async(
    send: Callable[[Any, dict[str, Any]], Awaitable[Any]], client: Any, kwargs: dict[str, Any], *,
    task: Optional[str], provider: str = "", base_url: str = "",
) -> Any:
    """Async twin of :func:`send_with_parameter_rungs`."""
    from hermes_agent.agent.auxiliary_client import _aoutcome, _drive_ladder_async
    resp, first_err = await _aoutcome(send(client, kwargs))
    if first_err is None:
        return resp, None, ()
    ladder = _parameter_ladder(first_err, client, kwargs, task=task, tag=" (async)", provider=provider,
                               base_url=base_url)
    return await _drive_ladder_async(ladder, lambda step: send(*step.args))
