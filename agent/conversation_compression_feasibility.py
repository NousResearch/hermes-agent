"""When the compression feasibility probe runs: after a main-runtime change, once a request first reaches
the smallest summariser window, and again when a warm agent meets a compression route other than the one
it probed. The probe itself is ``check_compression_model_feasibility`` in ``agent.conversation_compression``."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_compression")


def _probe(agent: Any) -> None:
    from agent.conversation_compression import check_compression_model_feasibility
    check_compression_model_feasibility(agent)


def _mark_checked(agent: Any, route: tuple) -> None:
    """Latch the probe as done for *route*, the compression route read before the probe ran: an edit that
    lands during the probe is then a moved route on the next request, not a measured one."""
    agent._compression_feasibility_checked = True
    agent._compression_feasibility_route = route


def compression_route_identity() -> tuple:
    """The configured compression route a probe measures: the task's provider, model, URL and context
    length, plus the URL of a ``providers:`` / ``custom_providers:`` entry the task names. A config this
    cannot read yields ``("unreadable",)``: the main request carries on and the probe reports the problem."""
    try:
        return _route_identity()
    except Exception:
        logger.debug("Compression route unreadable for the feasibility latch", exc_info=True)
        return ("unreadable",)


def _route_identity() -> tuple:
    from agent.auxiliary_task_config import _get_auxiliary_task_config
    cfg = _get_auxiliary_task_config("compression")
    route = tuple(str(cfg.get(k) or "") for k in ("provider", "model", "base_url", "context_length"))
    provider = route[0].strip()
    if not provider or provider in ("auto", "main", "custom") or route[2]:
        return route
    from hermes_cli.config import get_custom_provider_context_length
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider
    entry = _get_named_custom_provider(provider, resolve_credentials=False) or {}
    url, model = str(entry.get("base_url") or ""), route[1] or str(entry.get("model") or "")
    # The context pin the probe will read for this model at this URL (per-model, else entry-level).
    pin = get_custom_provider_context_length(model=model, base_url=url) if url and model else None
    return route + (url, str(entry.get("model") or ""), str(pin or ""))


def _drop_stale_ceiling(agent: Any) -> None:
    """A warm agent met a compression route other than the one it probed: the ceiling it installed and
    any failure cooldown belong to the old summariser, so the trigger returns to the main window and the
    cooldown is cleared (in memory and in the session row) before the new probe."""
    agent._compression_feasibility_checked = False
    compressor = agent.context_compressor
    clear_cooldown = getattr(compressor, "_clear_compression_failure_cooldown", None)
    if callable(clear_cooldown):
        clear_cooldown()
    if getattr(compressor, "_aux_context_ceiling", None) is not None:
        compressor._aux_context_ceiling = None
        compressor.update_model(compressor.model, compressor.context_length, base_url=compressor.base_url,
                                api_key=compressor.api_key, provider=compressor.provider,
                                api_mode=compressor.api_mode)


def revalidate_compression_feasibility(agent: Any) -> None:
    """Re-run the aux feasibility probe after the main runtime changed (model switch, fallback activation,
    primary restore). ``update_model()`` already voided the previous ceiling; probing now clamps the trigger
    before the first compaction on the new window rather than after it (#114707). A probe failure leaves the
    latch unset so the lazy probe at the next compaction re-raises hard rejections."""
    agent._compression_feasibility_checked = False
    if not getattr(agent, "context_compressor", None):
        return
    route = compression_route_identity()
    try:
        _probe(agent)
    except Exception as exc:
        logger.debug("Compression feasibility re-check deferred to the next compaction: %s", exc, exc_info=True)
        return
    _mark_checked(agent, route)


def ensure_compression_feasibility_checked(agent: Any, estimated_tokens: int) -> None:
    """Run the deferred aux feasibility probe once a request first reaches ``MINIMUM_CONTEXT_LENGTH`` — the
    smallest window any summariser may have — so an aux clamp lands before the first compaction fires on the
    main-window threshold instead of after it (#114707). Below that size no summariser can be too small, so
    short sessions keep the probe-free cold start (#28957). A probe failure leaves the latch unset for the
    lazy probe in ``compress_context`` to re-raise hard rejections. A warm agent whose compression route moved
    since its last probe (side tasks follow the profile's config.yaml) probes again."""
    if not getattr(agent, "context_compressor", None):
        return
    route = compression_route_identity()
    if getattr(agent, "_compression_feasibility_checked", False):
        probed = getattr(agent, "_compression_feasibility_route", None)
        if probed is None or route == probed:
            return
        _drop_stale_ceiling(agent)
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
    if int(estimated_tokens or 0) < MINIMUM_CONTEXT_LENGTH:
        return
    try:
        _probe(agent)
    except Exception as exc:
        logger.debug("Compression feasibility probe deferred to the first compaction: %s", exc, exc_info=True)
        return
    _mark_checked(agent, route)


def lazy_compression_feasibility_check(agent: Any) -> None:
    """The probe ``compress_context`` runs when nothing probed yet, or the route moved since the probe
    that recorded one; a hard rejection propagates to the caller. A latch with no recorded route (set
    before route tracking, or by a caller marking the agent checked) is trusted as it always was."""
    route = compression_route_identity()
    if getattr(agent, "_compression_feasibility_checked", False):
        probed = getattr(agent, "_compression_feasibility_route", None)
        if probed is None or route == probed:
            return
        _drop_stale_ceiling(agent)
    _probe(agent)
    _mark_checked(agent, route)
