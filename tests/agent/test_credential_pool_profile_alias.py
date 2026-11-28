"""A credential pool stays bound when the session names its provider by a declared alias.

``/model`` stores a provider's models.dev id (``opencode`` for OpenCode Zen, ``github-copilot`` for
Copilot), which the provider profile declares as an alias, while the pool is keyed by the profile
name. ``credential_pool_matches_provider`` compared the two literally, so agent init dropped the
pool and a 401/429 had no row to recover (#87526). Custom endpoints, a user's provider named after
an alias, and fallbacks on another provider keep their boundaries.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import credential_pool_matches_provider, load_pool, resolve_runtime_pool_key
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState
from providers import get_provider_profile
from run_agent import AIAgent

ZEN_URL = "https://opencode.ai/zen/v1"
USER_URL = "https://proxy.example/v1"


def _home(tmp_path, monkeypatch, *, config=None, pool=None):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump(config or {}), encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": pool or {}}), encoding="utf-8")
    return home


@pytest.mark.parametrize(("pool_provider", "provider"), [
    ("opencode-zen", "opencode"), ("opencode", "opencode-zen"), ("opencode-zen", "zen"),
])
def test_declared_alias_matches_its_profile_pool(tmp_path, monkeypatch, pool_provider, provider):
    _home(tmp_path, monkeypatch)
    assert credential_pool_matches_provider(SimpleNamespace(provider=pool_provider), provider, base_url=ZEN_URL)


@pytest.mark.parametrize(("pool_provider", "provider"), [
    ("anthropic", "claude"),
    ("openai-codex", "codex"),
    ("nous", "nous-portal"),
    ("nous-portal", "nous"),
    ("nous-portal", "nousresearch"),
])
def test_single_use_refresh_pools_require_literal_provider(tmp_path, monkeypatch, pool_provider, provider):
    _home(tmp_path, monkeypatch)
    assert credential_pool_matches_provider(pool_provider, pool_provider)
    assert not credential_pool_matches_provider(pool_provider, provider)


def test_single_use_refresh_pool_key_cannot_be_aliased_by_another_profile(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    # xai-oauth has a built-in pool but no bundled profile; a plugin can claim the spelling.
    plugin = home / "plugins" / "model-providers" / "test-xai"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='test-xai', aliases=('xai-oauth',)))\n",
        encoding="utf-8",
    )
    assert not credential_pool_matches_provider("xai-oauth", "test-xai")
    assert not credential_pool_matches_provider("test-xai", "xai-oauth")


@pytest.mark.parametrize(("pool_provider", "provider"), [
    ("opencode-zen", "go"),  # opencode-go's alias
    ("opencode-zen", "openrouter"),
    ("opencode-zen", "not-a-provider"),
    ("custom", "ollama"),  # the generic custom profile's alias: endpoint identity, not a name
    ("ollama", "custom"),
])
def test_alias_of_another_provider_or_custom_endpoint_does_not_match(tmp_path, monkeypatch, pool_provider, provider):
    _home(tmp_path, monkeypatch)
    assert not credential_pool_matches_provider(pool_provider, provider, base_url=ZEN_URL)


@pytest.mark.parametrize("config", [
    {"providers": {"opencode": {"api": USER_URL, "transport": "chat_completions"}}},
    {"custom_providers": [{"name": "opencode", "base_url": USER_URL}]},
])
def test_users_provider_named_like_an_alias_is_not_the_profile(tmp_path, monkeypatch, config):
    """``opencode`` is the user's endpoint here: the OpenCode Zen pool must not bind to it, its pool
    must not serve OpenCode Zen, and the Zen runtime keeps its own pool key."""
    _home(tmp_path, monkeypatch, config=config)
    assert not credential_pool_matches_provider("opencode-zen", "opencode", base_url=USER_URL)
    assert not credential_pool_matches_provider("opencode", "opencode-zen", base_url=ZEN_URL)
    assert resolve_runtime_pool_key("opencode-zen", ZEN_URL) == "opencode-zen"


def test_custom_pool_keyed_by_an_alias_still_needs_its_endpoint(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, config={"providers": {"zen": {"api": USER_URL, "transport": "chat_completions"}}})
    assert credential_pool_matches_provider("zen", "custom", base_url=USER_URL)
    assert not credential_pool_matches_provider("zen", "custom", base_url="https://other.example/v1")
    assert not credential_pool_matches_provider("zen", "opencode-zen", base_url=ZEN_URL)


@pytest.mark.parametrize("fallback_provider", ["openrouter", "go"])
def test_fallback_on_another_provider_leaves_the_primary_pool_alone(fallback_provider):
    """#33088/#33163: errors on a fallback never mutate the primary's pool, alias spelling or not."""
    pool = MagicMock()
    pool.provider = "opencode-zen"
    agent = SimpleNamespace(provider=fallback_provider, base_url="https://fallback.example/v1", _credential_pool=pool)
    assert recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True, classified_reason=FailoverReason.rate_limit,
    ) == (False, True)
    assert not pool.method_calls


class _Unauthorized(Exception):
    status_code = 401

    def __init__(self):
        super().__init__("Error code: 401 - {'error': {'message': 'Invalid API key'}}")
        self.body = {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}


def _row(idx, key):
    return {"id": f"cred-{idx}", "label": f"key-{idx}", "auth_type": "api_key", "priority": idx,
            "source": "manual", "access_token": key, "base_url": ZEN_URL}


def test_agent_on_an_alias_recovers_a_401_on_the_dispatched_pool_row(tmp_path, monkeypatch):
    """``AIAgent(provider="opencode")`` with the OpenCode Zen pool keeps it; the 401 on the key it
    sent benches that row and rotates, instead of reaching the provider's row-less refresher."""
    _home(tmp_path, monkeypatch, pool={"opencode-zen": [_row(0, "key-a"), _row(1, "key-b")]})
    pool = load_pool("opencode-zen")
    assert [entry.id for entry in pool.entries()] == ["cred-0", "cred-1"]
    with patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.process_bootstrap.OpenAI"), \
            patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        agent = AIAgent(provider="opencode", base_url=ZEN_URL, api_key="key-b", model="glm-5", credential_pool=pool,
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    assert agent._credential_pool is pool
    assert agent._credential_pool_entry_id == "cred-1"

    # A profile that owns a runtime refresher (as Copilot's does) must still leave the 401 to the pool.
    rowless_refresh = agent._try_refresh_provider_runtime_credentials = MagicMock(return_value=True)
    err = _Unauthorized()
    with patch("agent.conversation_loop._credential_refresh_profile", return_value=get_provider_profile("opencode-zen")), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        result = recover_after_classification(
            agent, err, classify_api_error(err, provider=agent.provider, model=agent.model), TurnRetryState(),
            status_code=401, error_context={}, messages=[], api_messages=[],
        )

    assert result == (True, True)
    rowless_refresh.assert_not_called()
    assert (agent.api_key, agent._credential_pool_entry_id) == ("key-a", "cred-0")
    stored = {entry.id: entry for entry in load_pool("opencode-zen").entries()}
    assert (stored["cred-1"].last_status, stored["cred-1"].last_error_code) == ("exhausted", 401)
    assert stored["cred-0"].last_status != "exhausted"
