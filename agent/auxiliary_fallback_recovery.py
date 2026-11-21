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
    from agent.auxiliary_client import _failure, _LadderRoute, _ladder_parameter_rungs
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
    from agent.auxiliary_client import _drive_ladder, _outcome
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
    from agent.auxiliary_client import _aoutcome, _drive_ladder_async
    resp, first_err = await _aoutcome(send(client, kwargs))
    if first_err is None:
        return resp, None, ()
    ladder = _parameter_ladder(first_err, client, kwargs, task=task, tag=" (async)", provider=provider,
                               base_url=base_url)
    return await _drive_ladder_async(ladder, lambda step: send(*step.args))


import re
import logging
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

logger = logging.getLogger(__name__)


def _task_minimum_context_length(task: Optional[str]) -> Optional[int]:
    """Minimum context length for an auxiliary task; None = no floor (only ``compression`` has one)."""
    return MINIMUM_CONTEXT_LENGTH if task == "compression" else None


def _fallback_entry_api_key(entry: dict[str, Any]) -> Optional[str]:
    """Resolve inline or env-backed API key via the secret-scope-aware resolver (no raw os.getenv under multiplexing)."""
    from hermes_cli.fallback_config import resolve_entry_api_key
    return resolve_entry_api_key(entry)


def _candidate_context_window(provider: str, model: str, base_url: str = "", api_key: str = "") -> Optional[int]:
    """Best-effort context window for a fallback candidate; ``None`` = unknown (never raises; callers pass it through)."""
    if not model:
        return None
    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(model, base_url=base_url, api_key=api_key, provider=provider)
    except Exception as exc:
        logger.debug("Auxiliary fallback: could not resolve context window for %s/%s: %s", provider, model, exc)
        return None
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def _context_too_small(
    entry: dict[str, Any], provider: str, model: str, min_ctx: Optional[int], *,
    task: Optional[str], label: str, name_model: bool = False,
) -> Optional[str]:
    """Screen one fallback candidate by context window; returns the ``tried`` note when it is too small."""
    if min_ctx is None:
        return None
    fb_ctx = _candidate_context_window(
        provider, model, base_url=str(entry.get("base_url") or ""), api_key=_fallback_entry_api_key(entry) or "")
    if fb_ctx is None or fb_ctx >= min_ctx:
        return None
    if name_model:
        logger.info("Auxiliary %s: skipping %s (%s context=%d < min=%d), continuing chain",
                    task, label, model, fb_ctx, min_ctx)
    else:
        logger.info("Auxiliary %s: skipping %s (context=%d < min=%d), continuing chain",
                    task or "call", label, fb_ctx, min_ctx)
    return f"{label} (context too small: {fb_ctx}<{min_ctx})"


def _fallback_chain_entry(task: Optional[str], fb_label: str) -> Optional[dict[str, Any]]:
    """Resolve the fallback entry a label points at, or None when unresolvable."""
    if not fb_label:
        return None
    m_task = re.match(r"fallback_chain\[(\d+)\]", fb_label)
    if m_task and task:
        try:
            from agent.auxiliary_task_config import _get_auxiliary_task_config
            chain = _get_auxiliary_task_config(task).get("fallback_chain")
            entry = chain[int(m_task.group(1))] if isinstance(chain, list) else None
            return entry if isinstance(entry, dict) else None
        except (IndexError, ValueError, TypeError):
            return None
    m_top = re.match(r"fallback_providers\[(\d+)\]", fb_label)
    if m_top:
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.fallback_config import get_fallback_chain
            chain = get_fallback_chain(load_config_readonly())
            idx = int(m_top.group(1))
            if isinstance(chain, list) and 0 <= idx < len(chain):
                entry = chain[idx]
                return entry if isinstance(entry, dict) else None
        except (IndexError, ValueError, TypeError):
            return None
    return None


def _coerce_positive_timeout(raw: Any) -> Optional[float]:
    """Coerce a config ``timeout`` to a positive float, or None (rejects bools, which are ints)."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return None


def _fallback_entry_timeout(task: Optional[str], fb_label: str) -> Optional[float]:
    """Per-entry ``timeout`` for a configured fallback candidate, or None (keep the task-level timeout)."""
    entry = _fallback_chain_entry(task, fb_label)
    return _coerce_positive_timeout(entry.get("timeout") if entry else None)


def _fallback_provider_from_label(label: str) -> str:
    """Recover the provider identifier from a fallback display label."""
    match = re.match(r"(?:fallback_chain\[\d+\]|fallback_providers\[\d+\]|main-agent)\(([^)]+)\)$", label or "")
    return match.group(1).strip() if match else str(label or "").strip()
