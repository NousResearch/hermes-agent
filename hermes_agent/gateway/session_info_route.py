"""The ``channel_overrides`` route a chat's turn runs on, for displays (/new banner, /footer preview)."""

import logging
from typing import Any, Optional

logger = logging.getLogger("hermes_agent.gateway.run")


def channel_override_route(config: Any, source: Any) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """``(model, route, force_provider)`` for ``source``'s channel override, as the turn resolves it
    (``_resolve_session_agent_runtime``): the override model, and the override provider's own
    endpoint/credentials. ``force_provider`` names the provider when its runtime cannot be
    resolved, so the display still reports it. ``(None, None, None)`` without an override."""
    from hermes_agent.gateway.run import _get_channel_override, _resolve_runtime_agent_kwargs_for_provider
    if source is None or config is None:
        return None, None, None
    ch = _get_channel_override(
        config, source.platform, str(source.chat_id) if source.chat_id else "",
        thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
        parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
    )
    if ch is None:
        return None, None, None
    model = ch.model or None
    if not ch.provider:
        return model, None, None
    try:
        runtime = _resolve_runtime_agent_kwargs_for_provider(ch.provider, target_model=model)
    except Exception:  # health: allow BLE001 -- a broken override must not break /new; logged
        logger.debug("Channel override provider %s unresolvable for display", ch.provider, exc_info=True)
        return model, None, ch.provider
    # The configured identity (``custom:<name>`` / a ``providers:`` entry), not the bare ``custom``
    # transport: /model disambiguates a model several named endpoints declare by this provider.
    from hermes_agent.hermes_cli.runtime_provider_custom import session_provider_identity
    provider = session_provider_identity(runtime.get("provider"), runtime.get("requested_provider"),
                                         runtime.get("base_url"), model) or ch.provider
    route = {"provider": provider, "base_url": runtime.get("base_url") or "",
             "api_key": runtime.get("api_key"), "_runtime_resolved": True}
    return model or runtime.get("model") or None, route, None
