"""Bind an agent's credential to the env var a ``model_aliases`` entry names (``key_env``).

A second key for the same provider (another account) only stays in use if every later credential
path re-reads that var. While bound: the per-request refresh re-reads only the var, the provider's
credential pool is never attached (its entries are the provider-default account's keys, so a
429/billing rotation would move the session there), and the default token chain is never consulted.
"""

from __future__ import annotations

from typing import Any


def bind_api_key_env(agent: Any, api_key_env: Any, credential_pool: Any) -> Any:
    """Record the var NAME on ``agent._api_key_env`` and return the credential pool to attach:
    ``None`` while bound, else ``credential_pool`` unchanged."""
    agent._api_key_env = api_key_env.strip() if isinstance(api_key_env, str) else ""
    return None if agent._api_key_env else credential_pool


def default_token_chain_allowed(agent: Any, provider: str) -> bool:
    """Whether an empty key may fall back to the default Anthropic token chain: native Anthropic only
    (other anthropic_messages providers must never receive Anthropic credentials), and never for a
    key_env-bound agent (an empty var must fail auth, not bill another account)."""
    return provider == "anthropic" and not getattr(agent, "_api_key_env", "")


def drop_pool_if_key_env_bound(agent: Any) -> bool:
    """Detach any provider pool from a key_env-bound agent. Returns whether the agent is bound."""
    if not getattr(agent, "_api_key_env", ""):
        return False
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    return True
