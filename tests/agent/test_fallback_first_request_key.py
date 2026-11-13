"""The first request after a fallback carries the key a direct pick would, and a pick nobody sends
leaves the pool as it was. Real pool and resolver; only the agent around them is a stand-in."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool

_TOKEN_PLAN = "alibaba-token-plan"
_TOKEN_PLAN_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
_ENV_KEY = "sk-env-key-spent-for-the-month"
_POOL_KEY = "sk-pool-key-with-quota-left"
_OTHER_KEY = "sk-key-for-another-endpoint"


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


def _add_pool_key(provider, entry_id, key, base_url, priority):
    load_pool(provider).add_entry(PooledCredential(
        provider=provider, id=entry_id, label=entry_id, auth_type=AUTH_TYPE_API_KEY, priority=priority,
        source=SOURCE_MANUAL, access_token=key, base_url=base_url))


def _setup(provider, env_var, url, strategy=None):
    """``.env`` key spent and benched, a higher-priority key for another endpoint, a healthy key for ``url``."""
    home = Path(os.environ["HERMES_HOME"])
    (home / ".env").write_text(f"{env_var}={_ENV_KEY}\n")
    if strategy:
        (home / "config.yaml").write_text(f"credential_pool_strategies:\n  {provider}: {strategy}\n")
    _add_pool_key(provider, "other-endpoint", _OTHER_KEY, "https://relay.example/v1", priority=1)
    _add_pool_key(provider, "healthy", _POOL_KEY, url, priority=10)
    load_pool(provider).mark_exhausted_and_rotate(
        status_code=429, api_key_hint=_ENV_KEY, failure_reason="insufficient_quota")


@pytest.mark.parametrize("provider,env_var,pool_url,entry_url,sent_url", [
    (_TOKEN_PLAN, "ALIBABA_TOKEN_PLAN_API_KEY", _TOKEN_PLAN_URL, None, _TOKEN_PLAN_URL),
    # MiniMax rows keep the /anthropic URL; a saved entry names the /v1 sibling the resolver calls.
    ("minimax", "MINIMAX_API_KEY", "https://api.minimax.io/anthropic", "https://api.minimax.io/v1",
     "https://api.minimax.io/v1"),
])
def test_the_first_fallback_request_uses_the_healthy_key_for_its_endpoint(provider, env_var, pool_url, entry_url,
                                                                          sent_url):
    from agent.chat_completion_helpers import try_activate_fallback

    _setup(provider, env_var, pool_url)
    agent = _agent({"provider": provider, "model": "some-model", **({"base_url": entry_url} if entry_url else {})})

    assert try_activate_fallback(agent) is True

    assert agent.api_key == agent.client.api_key == _POOL_KEY
    assert agent.base_url.rstrip("/") == sent_url
    assert agent._credential_pool_entry_id == "healthy"


def test_a_pick_that_is_not_sent_leaves_the_pool_unchanged(monkeypatch):
    """A voice turn binds its route, then runs on the main model when the history would not fit:
    the key it picked was never sent, so the round-robin order and request counts stay as they were."""
    import agent.voice_turn_route as voice
    from agent.route_binding import resolve_route_entry_client

    _setup(_TOKEN_PLAN, "ALIBABA_TOKEN_PLAN_API_KEY", _TOKEN_PLAN_URL, strategy="round_robin")
    _add_pool_key(_TOKEN_PLAN, "healthy-2", "sk-second-healthy-key", _TOKEN_PLAN_URL, priority=11)
    auth = Path(os.environ["HERMES_HOME"], "auth.json")
    before = auth.read_bytes()

    readiness = resolve_route_entry_client({"provider": _TOKEN_PLAN, "model": "some-model"}, _TOKEN_PLAN, "some-model")
    assert readiness.client.api_key == _POOL_KEY  # `hermes fallback check --live` builds the same client

    agent = _agent({"provider": "custom", "model": "primary-model"})
    agent._voice_turn_pending, agent.reasoning_config = True, None
    monkeypatch.setattr("agent.auxiliary_task_config._get_auxiliary_task_config",
                        lambda task: {"provider": _TOKEN_PLAN, "model": "voice-model"})
    monkeypatch.setattr(voice, "_capture", lambda agent: {name: None for name in voice._EXTRA_FIELDS})
    monkeypatch.setattr(voice, "_reinstall", lambda agent, state: None)
    monkeypatch.setattr(voice, "_fits", lambda agent, messages, system_prompt: False)

    voice.begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], "system")

    assert auth.read_bytes() == before


def test_a_provider_whose_endpoint_follows_from_the_key_keeps_its_env_key():
    """kimi-coding picks its endpoint from the ``.env`` key (Moonshot for a legacy key); a Kimi Code key from
    the pool only works on its own endpoint, so even with the ``.env`` key benched the fallback keeps the
    key the endpoint was derived from, as on main."""
    from agent.route_binding import resolve_route_entry_client

    Path(os.environ["HERMES_HOME"], ".env").write_text(f"KIMI_API_KEY={_ENV_KEY}\n")
    _add_pool_key("kimi-coding", "kimi-code", "sk-kimi-pooled-key", "https://api.moonshot.ai/v1", priority=0)
    load_pool("kimi-coding").mark_exhausted_and_rotate(
        status_code=429, api_key_hint=_ENV_KEY, failure_reason="insufficient_quota")

    route = resolve_route_entry_client({"provider": "kimi-coding", "model": "kimi-k2"}, "kimi-coding", "kimi-k2")

    assert route.client.api_key == _ENV_KEY
