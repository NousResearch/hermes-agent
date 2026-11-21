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


def bind_init_fallback(agent, fallback_model, refused: list[tuple[str, str]]) -> tuple[bool, Any]:
    """Bind the first usable fallback entry while the agent is being built.

    ``(True, routed_client)`` once an entry resolves; ``(True, None)`` when that entry is a MoA preset
    (its facade is already bound, there is no OpenAI client to build); ``(False, None)`` when no entry
    is usable, with each refused entry's ``(provider, reason)`` appended to ``refused``.
    """
    from agent.auxiliary_client import resolve_provider_client
    for _fb in _fallback_entries(fallback_model):
        _fb_provider = str(_fb["provider"])
        try:
            from hermes_cli.fallback_config import resolve_entry_api_key
            _fb_explicit_key = resolve_entry_api_key(_fb)
            _fb_client, _fb_model = resolve_provider_client(
                _fb["provider"], model=_fb["model"], raw_codex=True,
                explicit_base_url=_fb.get("base_url"), explicit_api_key=_fb_explicit_key,
            )
        except Exception as _fb_exc:
            logger.debug("Init-time fallback entry %s failed: %s", _fb_provider, _fb_exc, exc_info=True)
            # A bare exception (``KeyError()``) stringifies empty; name its type instead.
            refused.append((_fb_provider, str(_fb_exc) or type(_fb_exc).__name__))
            continue
        if _fb_client is None:
            # The router returns None when no credentials are usable for the entry — a skip
            # that leaves no trace otherwise, hiding key-less fallback entries from the log.
            logger.debug(
                "Init-time fallback entry %s resolved no usable credentials", _fb_provider
            )
            refused.append((_fb_provider, "no usable credentials"))
            continue
        agent._fallback_activated = True
        if _fb_provider.strip().lower() == "moa":
            # The chokepoint handed back the preset's aggregator client, which only proves the
            # preset resolves and its aggregator has credentials. A MoA entry means the preset
            # itself (same as ``provider: moa`` in config), so bind the facade, not the aggregator.
            from agent.moa_loop import bind_moa_runtime
            bind_moa_runtime(agent, _fb["model"])
            return True, None
        agent.provider = agent.requested_provider = _fb["provider"]
        agent.model = _fb_model or _fb["model"]
        recompute_init_fallback_api_mode(agent, _fb_client)
        return True, _fb_client
    return False, None


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


def recompute_init_fallback_api_mode(agent, fb_client) -> None:
    """Give an init-time fallback its own api_mode instead of the unreachable primary's.

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a fallback whose
    profile serves that model on chat completions (``api_mode_for_model``) on the Responses API,
    which silently drops reasoning content (#46527).
    The per-turn path (``try_activate_fallback``) already recomputes via
    ``_fallback_api_mode_resolved``; this keeps init in agreement, and runs before the
    ``_primary_runtime`` snapshot so ``restore_primary_runtime`` replays the recomputed mode.
    A detection failure keeps the primary's mode and never blocks the fallback itself.
    """
    from agent.chat_completion_helpers import _fallback_api_mode_resolved
    try:
        agent.api_mode = _fallback_api_mode_resolved(
            agent, agent.provider, agent.model, str(getattr(fb_client, "base_url", "") or ""))
    except Exception:
        logger.debug("Init-time fallback api_mode detection failed for %s", agent.provider, exc_info=True)
        return
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
