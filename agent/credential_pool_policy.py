"""Operator policy generations, distinct from token refresh and rotation bookkeeping."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


def policy_generation(store, provider):
    generations = store.get("credential_pool_generations", {})
    if not isinstance(generations, dict):
        raise ValueError("Invalid pool generations")
    generation = generations.get(provider, 0)
    if type(generation) is not int or generation < 0:
        raise ValueError("Invalid pool generation")
    return generation


def read_policy_snapshot(provider):
    """Read one atomic auth file; reject incomplete updates without recovering or writing it."""
    from hermes_cli.auth import _auth_file_path
    store = json.loads(_auth_file_path().read_text(encoding="utf-8-sig"))
    if not isinstance(store, dict) or not isinstance(store.get("credential_pool"), dict):
        raise ValueError("Invalid pool store")
    rows = store["credential_pool"].get(provider)
    if not isinstance(rows, list):
        raise ValueError("Missing pool")
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or row["id"] in ids
                or type(row.get("priority", 0)) is not int
                or not isinstance(row.get("access_token", ""), str)):
            raise ValueError("Invalid pool entry")
        ids.add(row["id"])
    return rows, policy_generation(store, provider)


def read_policy_strategy(provider):
    from agent.credential_pool import STRATEGY_FILL_FIRST, SUPPORTED_POOL_STRATEGIES
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.config import get_config_path
    from utils import fast_safe_load

    # The effective loader intentionally treats non-mapping YAML as empty. A live
    # policy refresh must reject that transient edit rather than reset preference.
    path = Path(get_config_path())
    text = path.read_text(encoding="utf-8")
    raw = fast_safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Invalid config")
    config = load_user_config_effective(fail_closed=True)
    if path.read_text(encoding="utf-8") != text:
        raise ValueError("Config changed during refresh")
    strategies = config.get("credential_pool_strategies", {})
    if not isinstance(strategies, dict):
        raise ValueError("Invalid pool strategies")
    strategy = strategies.get(provider, STRATEGY_FILL_FIRST)
    if strategy not in SUPPORTED_POOL_STRATEGIES:
        raise ValueError("Invalid pool strategy")
    return strategy


def preserve_newer_policy(entries, disk_entries):
    """A stale runtime may update health/tokens, but cannot undo admission or order."""
    by_id = {row.get("id"): row for row in entries if isinstance(row, dict)}
    return [{**by_id.get(row.get("id"), row), "priority": row.get("priority", 0)}
            for row in disk_entries if isinstance(row, dict)]


def bind_pool_policy(agent):
    """Remember the policy applied to this agent, independently of a shared pool cursor."""
    from agent.credential_pool import CredentialPool
    pool = getattr(agent, "_credential_pool", None)
    if isinstance(pool, CredentialPool):
        agent._credential_pool_policy = (pool, pool._policy_generation, pool._strategy)
        agent._credential_pool_policy_entry = (agent._credential_pool_entry_id, agent.api_key)


def refresh_pool_policy(agent):
    """Called only by admitted turn setup, after primary runtime restoration.

    Admin move/add/remove increments auth.json's provider generation. External
    schedulers use those APIs (or write_credential_pool(policy_update=True));
    token/status writes do not request an idle session to abandon its selection.
    """
    from agent.credential_pool import (
        CredentialPool, credential_pool_matches_provider,
        credential_pool_entry_serves_endpoint, load_pool,
    )
    pool = getattr(agent, "_credential_pool", None)
    if not isinstance(pool, CredentialPool):
        logger.debug("Credential selection source=pinned_or_overridden")
        return
    entry_id = getattr(agent, "_credential_pool_entry_id", None)
    current = next((e for e in pool.entries() if e.id == entry_id), None)
    applied = getattr(agent, "_credential_pool_policy", None)
    # An operator may remove the active row from this same shared pool. The
    # agent's last applied identity still proves this was a pool-owned selection.
    known_selection = (current is not None and current.runtime_api_key == agent.api_key) or (
        applied is not None and applied[0] is pool
        and getattr(agent, "_credential_pool_policy_entry", None) == (entry_id, agent.api_key)
    )
    if (getattr(pool, "_policy_home", None) != hermes_home_key() or not known_selection
            or not credential_pool_matches_provider(pool, agent.provider, base_url=agent.base_url)
            or not credential_pool_entry_serves_endpoint(current, agent.base_url)
            or pool._active_leases):
        logger.info("Credential selection source=pinned_or_overridden")
        return
    try:
        snapshot = read_policy_snapshot(pool.provider)
        strategy = read_policy_strategy(pool.provider)
        generation = snapshot[1]
        if applied is None or applied[0] is not pool:
            applied = (pool, pool._policy_generation, pool._strategy)
        if (generation, strategy) == applied[1:]:
            logger.debug("Credential selection source=sticky generation=%d", generation)
            return
        candidate = load_pool(pool.provider, policy_snapshot=snapshot)
        candidate._strategy = strategy
        entry = candidate.select(model=getattr(agent, "model", None))
        if (entry is None or not credential_pool_entry_serves_endpoint(entry, agent.base_url)
                or not adopt_policy_credential(agent, entry)):
            logger.info("Credential selection source=last_good generation=%d", generation)
            return
        agent._credential_pool = candidate
        bind_pool_policy(agent)
        agent._credential_pool_revert_id = None
        logger.info("Credential selection source=pool generation=%d", generation)
    except Exception:
        # Never log exception text: parse/SDK exceptions may quote credentials.
        logger.warning("Credential selection source=last_good policy_update=unavailable")


def adopt_policy_credential(agent, entry):
    """Build the replacement before publishing any live state; the route is unchanged."""
    from contextlib import suppress

    key = entry.runtime_api_key
    if not key:
        return False
    if key == agent.api_key:
        agent._credential_pool_entry_id = entry.id
        return True
    if agent.api_mode == "anthropic_messages":
        client = agent._build_direct_anthropic_client(key, agent.base_url)
        oauth = agent._anthropic_oauth_flag(key)
        old = agent._anthropic_client
        agent._anthropic_client = client
        agent._anthropic_api_key = key
        agent._anthropic_base_url = agent.base_url
        agent._is_anthropic_oauth = oauth
        agent.api_key = key
        agent._credential_pool_entry_id = entry.id
        with suppress(Exception):
            old.close()
    else:
        kwargs = dict(agent._client_kwargs, api_key=key)
        with agent._openai_client_lock():
            client = agent._create_openai_client(kwargs, reason="pool_policy", shared=True)
            old = agent.client
            agent.client = client
            agent._client_kwargs = kwargs
            agent.api_key = key
            agent._credential_pool_entry_id = entry.id
        with suppress(Exception):
            agent._retire_shared_openai_client(old, reason="pool_policy")
    return True
