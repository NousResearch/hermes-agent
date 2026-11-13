"""The first request after a cross-provider fallback carries the key a direct selection would.

Registry API-key providers build their fallback client through a resolver that reads the
``.env`` key ahead of the credential pool; selecting the same provider directly reads the pool
first and skips a key a 429 has benched. These tests drive the real pool and the real resolver
(only the agent is a stand-in) through ``try_activate_fallback``.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool

_TOKEN_PLAN = "alibaba-token-plan"
_TOKEN_PLAN_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
_ENV_KEY = "sk-env-key-spent-for-the-month"
_POOL_KEY = "sk-pool-key-with-quota-left"


def _agent(fallback_entry):
    from agent.client_lifecycle import ClientLifecycleMixin

    agent = MagicMock()
    agent.provider, agent.model, agent.api_key = "custom", "primary-model", "primary-key"
    agent.base_url, agent.api_mode = "http://127.0.0.1:9/v1", "chat_completions"
    agent._fallback_activated, agent._fallback_index, agent._fallback_chain = False, 0, [fallback_entry]
    agent._credential_pool = agent._credential_pool_entry_id = None
    agent._client_kwargs = {"api_key": "primary-key", "base_url": agent.base_url}
    agent._transport_cache, agent._config_context_length, agent.context_compressor = {}, None, None
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    agent._swap_credential = lambda entry: ClientLifecycleMixin._swap_credential(agent, entry)
    return agent


def _write_env(**keys):
    Path(os.environ["HERMES_HOME"], ".env").write_text("".join(f"{k}={v}\n" for k, v in keys.items()))


def _add_pool_key(provider, entry_id, key, base_url, priority=0):
    load_pool(provider).add_entry(PooledCredential(
        provider=provider, id=entry_id, label=entry_id, auth_type=AUTH_TYPE_API_KEY, priority=priority,
        source=SOURCE_MANUAL, access_token=key, base_url=base_url))


def _bench(provider, key):
    load_pool(provider).mark_exhausted_and_rotate(
        status_code=429, api_key_hint=key, failure_reason="insufficient_quota")


def _activate(entry):
    from agent.chat_completion_helpers import try_activate_fallback

    agent = _agent(entry)
    assert try_activate_fallback(agent) is True
    return agent


class TestFallbackFirstRequestKey:
    def test_a_key_the_pool_benched_is_not_sent_again(self):
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL)
        _bench(_TOKEN_PLAN, _ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max"})

        assert agent.provider == _TOKEN_PLAN
        assert agent.api_key == _POOL_KEY and agent._client_kwargs["api_key"] == _POOL_KEY
        assert agent.client.api_key == _POOL_KEY
        assert agent._credential_pool.current().id == "added"
        assert agent._credential_pool_entry_id == "added"

    def test_an_env_key_whose_pool_row_was_removed_is_not_sent(self):
        # `hermes auth remove` drops the .env row and suppresses its source; a key written back
        # into .env afterwards matches no pool row, so the pool never benches it.
        from types import SimpleNamespace
        from hermes_cli.auth_commands import auth_remove_command

        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL)
        env_row = next(e.id for e in load_pool(_TOKEN_PLAN).entries() if e.source.startswith("env:"))
        auth_remove_command(SimpleNamespace(provider=_TOKEN_PLAN, target=env_row))
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max"})

        assert agent.api_key == _POOL_KEY
        assert agent._credential_pool_entry_id == "added"

    def test_nothing_changes_while_no_key_has_failed(self):
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL, priority=5)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max"})

        assert agent.api_key == _ENV_KEY
        assert agent._credential_pool.current().source == "env:ALIBABA_TOKEN_PLAN_API_KEY"

    def test_numbered_env_keys_rotate_like_a_direct_pick(self):
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY, ALIBABA_TOKEN_PLAN_API_KEY_2=_POOL_KEY)
        _bench(_TOKEN_PLAN, _ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max"})

        assert agent.api_key == _POOL_KEY
        assert agent._credential_pool.current().source == "env:ALIBABA_TOKEN_PLAN_API_KEY_2"

    @pytest.mark.parametrize("own", [{"api_key": "sk-entry-own-key"}, {"key_env": "ENTRY_OWN_KEY"}])
    def test_an_entry_naming_its_own_key_keeps_it(self, own, monkeypatch):
        monkeypatch.setenv("ENTRY_OWN_KEY", "sk-entry-own-key")
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL)
        _bench(_TOKEN_PLAN, _ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max", **own})

        assert agent.api_key == "sk-entry-own-key"

    def test_an_entry_saved_with_the_providers_url_uses_the_pool(self):
        # `hermes fallback add` copies the picker's base_url into the entry.
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL)
        _bench(_TOKEN_PLAN, _ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max", "base_url": _TOKEN_PLAN_URL})

        assert agent.api_key == _POOL_KEY

    def test_a_pool_key_never_moves_to_another_endpoint(self):
        _write_env(ALIBABA_TOKEN_PLAN_API_KEY=_ENV_KEY)
        _add_pool_key(_TOKEN_PLAN, "added", _POOL_KEY, _TOKEN_PLAN_URL)
        _bench(_TOKEN_PLAN, _ENV_KEY)

        agent = _activate({"provider": _TOKEN_PLAN, "model": "qwen3.8-max", "base_url": "https://relay.example/v1"})

        assert agent.api_key == _ENV_KEY
        assert agent.base_url.rstrip("/") == "https://relay.example/v1"

    def test_minimax_keeps_the_resolvers_route(self):
        # MiniMax pool rows store the /anthropic URL; the resolver builds /v1 for chat completions.
        _write_env(MINIMAX_API_KEY=_ENV_KEY, MINIMAX_API_KEY_2=_POOL_KEY)
        _bench("minimax", _ENV_KEY)

        agent = _activate({"provider": "minimax", "model": "MiniMax-M2.7"})

        assert agent.api_key == _POOL_KEY
        assert agent.base_url.rstrip("/") == "https://api.minimax.io/v1"
        assert agent.api_mode == "chat_completions"

    def test_a_round_robin_pool_rotates_once_per_fallback(self):
        # openrouter selects from its own pool inside the resolver; the fallback must not select again.
        from hermes_constants import OPENROUTER_BASE_URL

        Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
            "credential_pool_strategies:\n  openrouter: round_robin\n")
        _write_env()
        _add_pool_key("openrouter", "key-a", "sk-or-key-a", OPENROUTER_BASE_URL, priority=0)
        _add_pool_key("openrouter", "key-b", "sk-or-key-b", OPENROUTER_BASE_URL, priority=1)

        sent = [_activate({"provider": "openrouter", "model": "qwen/qwen3-coder"}).api_key for _ in range(2)]

        assert sent == ["sk-or-key-a", "sk-or-key-b"]
