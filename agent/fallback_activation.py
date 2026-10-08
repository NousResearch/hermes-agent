"""Provider-chain activation using the shared runtime binder."""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger("agent.chat_completion_helpers")


def activate_next_fallback(agent, reason=None, reset_at=None) -> bool:
    """Walk only the supplied chain; manual activation supplies one authorized entry."""
    from agent.chat_completion_helpers import (
        _buffer_fallback_notice, _fallback_chain_exhausted, _fallback_entry_key,
        _fallback_reason_text, _log_fallback_activated, _reset_stale_streak,
        _should_skip_fallback_candidate, rewrite_prompt_model_identity,
    )
    from agent.fallback_cooldown import _arm_rate_limit_cooldown
    cooldown_seconds = _arm_rate_limit_cooldown(agent, reason, reset_at=reset_at)
    while True:
        if agent._fallback_index >= len(agent._fallback_chain):
            return _fallback_chain_exhausted(agent, reason)
        fb = agent._fallback_chain[agent._fallback_index]
        agent._fallback_index += 1
        fb_key = _fallback_entry_key(fb)
        if getattr(agent, "_unavailable_fallback_keys", None) is None:
            agent._unavailable_fallback_keys = set()
        unavailable = agent._unavailable_fallback_keys
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if _should_skip_fallback_candidate(agent, fb, fb_key, fb_provider, fb_model, unavailable):
            continue

        try:
            from agent.route_binding import bind_route_entry
            bound = bind_route_entry(agent, fb, fb_provider, fb_model)
            if bound is None:
                logger.warning("Fallback to %s failed: provider not configured", fb_provider)
                unavailable.add(fb_key)
                continue
            old_model, old_provider = bound
            fb_model = agent.model  # normalized by the binder
            rewrite_prompt_model_identity(agent, fb_model, fb_provider)

            notice = (
                f"⚠️ Model fallback: {old_model} via {old_provider} unavailable "
                f"({_fallback_reason_text(reason)}); using {fb_model} via {fb_provider}.")
            if cooldown_seconds is not None:
                remaining = max(0, math.ceil(agent._rate_limited_until - time.monotonic()))
                notice += f" Primary retry eligible in ~{remaining} s; recovery is not guaranteed."
            _buffer_fallback_notice(agent, notice)
            # ``_fallback_activated`` is also reused by `/model --once` restoration; separate
            # provenance so the restore path only emits a recovery notice after a real fallback.
            agent._provider_fallback_active = True
            agent._provider_fallback_route = (str(fb_model), str(fb_provider))
            _log_fallback_activated(agent, reason, old_model, old_provider, fb_model, fb_provider)
            from hermes_cli.observability.shared_metrics_events import record_fallback
            record_fallback(from_provider=old_provider, to_provider=fb_provider, reason=reason)
            # The stale-call streak measured the OLD provider; carrying it over would
            # short-circuit the fresh fallback before its first stream attempt.
            _reset_stale_streak(agent)
            from agent.native_compaction import resolve_native_compaction_capabilities
            agent.runtime_capabilities = resolve_native_compaction_capabilities(
                model=agent.model, base_url=agent.base_url, provider=fb_provider, is_codex_backend=fb_provider == "openai-codex")
            return True
        except Exception as e:
            if fb_provider == "nous":
                unavailable.add(fb_key)
            logger.error("Failed to activate fallback %s: %s", fb_model, e, exc_info=True)
            continue  # try next in chain
