"""Adopt subscription pool state changed by another process between turns."""
import logging

from agent.credential_pool import (
    credential_pool_entry_serves_endpoint, credential_pool_matches_provider, load_pool,
    resolve_runtime_pool_key,
)

logger = logging.getLogger(__name__)


def rotate_externally_exhausted_codex_key(agent) -> None:
    """Keep a cached session off a credential another session has exhausted."""
    if getattr(agent, "provider", None) != "openai-codex":
        return
    try:
        pool = load_pool(resolve_runtime_pool_key(agent.provider, agent.base_url))
        if not credential_pool_matches_provider(pool, agent.provider, base_url=agent.base_url):
            return
        entry = next((e for e in pool.entries() if e.runtime_api_key == agent.api_key), None)
        if entry is None or pool.reclaim(entry.id, model=agent.model) is not None:
            return
        replacement = pool.select(model=agent.model)
        if replacement is None or not credential_pool_entry_serves_endpoint(replacement, agent.base_url):
            return
        if agent._swap_credential(replacement) is False:
            return
        agent._credential_pool = pool
        runtime = getattr(agent, "_primary_runtime", None)
        if isinstance(runtime, dict):
            runtime["api_key"] = replacement.runtime_api_key
            if isinstance(runtime.get("client_kwargs"), dict):
                runtime["client_kwargs"]["api_key"] = replacement.runtime_api_key
        logger.info("Adopted subscription pool rotation for entry %s", replacement.id)
    except Exception:
        logger.debug("Subscription pool turn refresh failed", exc_info=True)


def attach_matching_codex_pool(agent):
    """Direct integrations may pass a pooled token without passing its pool."""
    if getattr(agent, "provider", None) != "openai-codex":
        return None
    try:
        pool = load_pool(resolve_runtime_pool_key(agent.provider, agent.base_url))
        entry = next((e for e in pool.entries() if e.runtime_api_key == agent.api_key), None)
        if (entry is None
                or not credential_pool_matches_provider(pool, agent.provider, base_url=agent.base_url)
                or not credential_pool_entry_serves_endpoint(entry, agent.base_url)):
            return None
        agent._credential_pool = pool
        agent._credential_pool_entry_id = pool.entry_id_for_api_key(agent.api_key)
        return pool
    except Exception:
        logger.debug("Matching subscription pool attachment failed", exc_info=True)
        return None
