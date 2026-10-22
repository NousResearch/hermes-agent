"""Init-time fallback chain: entry normalization, chain setup, and the fallback's api_mode."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("run_agent")


def _fallback_entries(fallback_model) -> list[dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _init_fallback_chain(agent, fallback_model):
    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    agent._fallback_chain = _fallback_entries(fallback_model)
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    chain = agent._fallback_chain
    if chain and not agent.quiet_mode:
        labels = [f"{f['model']} ({f['provider']})" for f in chain]
        if len(chain) == 1:
            print(f"🔄 Fallback model: {labels[0]}")
        else:
            print(f"🔄 Fallback chain ({len(chain)} providers): " + " → ".join(labels))


def recompute_init_fallback_api_mode(agent, fb_client, fallback_entry: dict[str, Any]) -> None:
    """Give an init-time fallback its own api_mode instead of the unreachable primary's.

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a Copilot
    gpt-5-mini fallback on the Responses API, which silently drops reasoning content (#46527).
    The per-turn path (``try_activate_fallback``) already recomputes via
    ``_fallback_api_mode_resolved``; this keeps init in agreement, and runs before the
    ``_primary_runtime`` snapshot so ``restore_primary_runtime`` replays the recomputed mode.
    A detection failure keeps the primary's mode and never blocks the fallback itself.
    """
    from agent.chat_completion_helpers import _fallback_api_mode_hint, _fallback_api_mode_resolved
    from hermes_cli.providers import is_actual_route
    try:
        provider = str(agent.provider or "").strip().lower()
        base_url_hint = str(fallback_entry.get("base_url") or "").strip() or None
        explicit, api_mode = _fallback_api_mode_hint(fallback_entry, provider, base_url_hint)
        base_url = str(getattr(fb_client, "base_url", "") or base_url_hint or "")
        if explicit:
            agent.api_mode = api_mode
        elif is_actual_route(provider, base_url):
            agent.api_mode = "chat_completions"
        elif api_mode == "chat_completions":
            agent.api_mode = _fallback_api_mode_resolved(agent, provider, agent.model, base_url)
        else:
            agent.api_mode = api_mode
    except Exception:
        logger.debug("Init-time fallback api_mode detection failed for %s", agent.provider, exc_info=True)
        return
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
