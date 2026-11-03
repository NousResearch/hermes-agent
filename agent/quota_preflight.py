"""Quota gates on native fallback binding, never on observer/context hooks."""
from __future__ import annotations

import logging
import time
from typing import Any

from agent.quota_routing import account_quota, invalidate_quota

logger = logging.getLogger(__name__)


def current_quota(agent: Any):
    return account_quota(str(getattr(agent, "provider", "")),
                         str(getattr(agent, "base_url", "") or ""), getattr(agent, "api_key", None))


def candidate_quota(agent: Any, entry: dict):
    """Check explicit entry credentials, or the same bound account's model aliases.

    Other unbound destinations remain unknown: do not select a pool credential or
    construct an inference client merely to inspect quota.
    """
    from hermes_cli.fallback_config import resolve_entry_api_key

    provider = str(entry.get("provider") or "").strip().lower()
    base_url = str(entry.get("base_url") or "")
    key = resolve_entry_api_key(entry)
    if not key and provider == getattr(agent, "provider", None):
        if not base_url or base_url.rstrip("/") == str(getattr(agent, "base_url", "")).rstrip("/"):
            # Multiple pool credentials may be different accounts. Leave their selection
            # to native rotation instead of treating the whole pool as exhausted.
            pool = getattr(agent, "_credential_pool", None)
            if pool is None or _pool_shares_current_group(agent, pool):
                key = getattr(agent, "api_key", None)
                base_url = base_url or str(getattr(agent, "base_url", "") or "")
    return account_quota(provider, base_url, key)


def _pool_shares_current_group(agent: Any, pool: Any) -> bool:
    from agent.quota_routing import quota_group

    current = quota_group(agent.provider, agent.base_url, agent.api_key)
    entries = pool.entries()
    if current is None or not isinstance(entries, list) or not entries:
        return False
    return all(quota_group(agent.provider, str(entry.runtime_base_url or agent.base_url),
                           entry.runtime_api_key) == current for entry in entries)


def _primary_quota_reopened_early(agent, primary_provider, primary_model, matches_primary, load_primary_pool) -> bool:
    """True when a Codex quota window that the primary pool still has benched has reopened.

    A Codex 429 benches the entry, and arms ``_rate_limited_until``, until a ``resets_at`` that can be
    days out (weekly window). The window can reopen sooner (redeemed reset, top-up, plan change); the
    pool's throttled usage probe notices, but only on ``select()``, which a session pinned to its
    fallback never reaches. Fails closed: any doubt leaves both cooldowns in force.

    At most one check per probe interval per agent: the probe caches its verdict and nothing clears
    it on the next 429, so a "restored" answer the endpoint got wrong would otherwise restore,
    fail, and fall back again on every turn of the cached window. Checks in between could only
    replay that cached verdict, so they return before loading the pool.
    """
    if primary_provider != "openai-codex":
        return False
    from hermes_cli.auth_codex import CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS
    now = time.monotonic()
    if now - getattr(agent, "_codex_reopen_checked_at", float("-inf")) < CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS:
        return False
    agent._codex_reopen_checked_at = now
    try:
        pool = getattr(agent, "_credential_pool", None)
        if pool is None or not matches_primary(pool):
            pool = load_primary_pool()
        if pool is None:
            return False
        model = primary_model or None
        benched_until = pool.next_available_at(model=model)
        if benched_until is None or benched_until <= time.time():
            return False
        return pool.lift_reopened_cooldowns(model=model)
    except Exception:
        logger.debug("Early quota-reopen check failed; keeping the cooldown", exc_info=True)
        return False


def retain_quota_fallback(agent: Any, primary: dict) -> bool:
    """A quota-selected fallback stays stable until primary recovery is observed.

    Unknown does not restore a known-exhausted primary or announce recovery. Other
    fallback origins retain the existing cooldown/early-reopen behavior unchanged.
    """
    if getattr(agent, "_quota_fallback_active", False) is not True:
        return False
    verdict = account_quota(str(primary.get("provider") or ""),
                            str(primary.get("base_url") or ""), primary.get("api_key"))
    return verdict.state != "available"


def preflight_quota_route(agent: Any) -> bool:
    """Use only the caller's existing authorized chain; unknown preserves the route.

    The chain already encodes cost/privacy/capability/reviewer restrictions. No
    global config lookup or provider expansion is allowed here; explicit [] wins.
    """
    chain = getattr(agent, "_fallback_chain", None)
    if not isinstance(chain, list) or not chain:
        return False
    verdict = current_quota(agent)
    if verdict.state != "exhausted":
        return False
    from agent.chat_completion_helpers import try_activate_fallback

    requested_provider, requested_model = agent.provider, agent.model
    changed = try_activate_fallback(agent)
    if changed:
        agent._quota_fallback_active = True
        # Deliberately omit endpoint/account/credential and exception contents.
        logger.info("Quota preflight requested=%s/%s actual=%s/%s reason=account_window_exhausted",
                    requested_provider, requested_model, agent.provider, agent.model)
    return changed


def prepare_quota_turn(agent: Any, state: Any) -> None:
    """Cover whole-turn transports as well as the ordinary per-attempt loop."""
    if preflight_quota_route(agent):
        from agent.conversation_loop import _sync_failover_system_message
        state.active_system_prompt = _sync_failover_system_message(agent, None, state.active_system_prompt)


def invalidate_on_quota_error(agent: Any, reason: Any) -> None:
    if getattr(reason, "value", reason) in {"rate_limit", "billing"}:
        invalidate_quota(str(getattr(agent, "provider", "")),
                         str(getattr(agent, "base_url", "") or ""), getattr(agent, "api_key", None))
