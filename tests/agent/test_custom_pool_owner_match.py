"""Owner-aware bare-custom pool matching (#124593).

A bare ``custom`` model and a named provider sharing one base_url must not
run on each other's credentials. With the runtime key known, the shared
matchers resolve/accept only the entry whose credential can be that key;
without it the lookup stays URL-only, exactly as before.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import credential_pool as cp

URL = "https://llm.example/v1"
MAIN_KEY = "main-key-12345678"
SECOND_KEY = "second-key-87654321"

ENTRIES = [
    ("second", {"name": "Second", "base_url": URL, "api_key": SECOND_KEY}),
    ("own", {"name": "Own", "base_url": URL, "api_key": MAIN_KEY}),
]


@pytest.fixture
def sibling_config():
    with patch.object(cp, "_iter_custom_providers", return_value=list(ENTRIES)):
        yield


class TestOwnerAwareBareCustomMatch:
    def test_resolves_own_pool_not_first_on_url(self, sibling_config):
        assert cp.resolve_runtime_pool_key("custom", URL, owner_api_key=MAIN_KEY) == "custom:own"

    def test_sibling_key_resolves_sibling_pool(self, sibling_config):
        assert cp.resolve_runtime_pool_key("custom", URL, owner_api_key=SECOND_KEY) == "custom:second"

    def test_no_owner_stays_url_only(self, sibling_config):
        assert cp.resolve_runtime_pool_key("custom", URL) == "custom:second"

    def test_own_pool_accepted_sibling_rejected(self, sibling_config):
        assert cp.credential_pool_matches_provider("custom:own", "custom", base_url=URL, owner_api_key=MAIN_KEY) is True
        assert cp.credential_pool_matches_provider("custom:second", "custom", base_url=URL, owner_api_key=MAIN_KEY) is False

    def test_legacy_url_only_without_owner(self, sibling_config):
        assert cp.credential_pool_matches_provider("custom:own", "custom", base_url=URL) is False
        assert cp.credential_pool_matches_provider("custom:second", "custom", base_url=URL) is True

    def test_candidates_skip_sibling_with_owner(self, sibling_config):
        assert cp.custom_provider_pool_key_candidates(URL, owner_api_key=MAIN_KEY) == ["custom:own"]

    def test_placeholder_owner_keeps_legacy_lookup(self, sibling_config):
        assert cp.resolve_runtime_pool_key("custom", URL, owner_api_key="no-key-required") == "custom:second"

    def test_unknown_key_matches_nothing(self, sibling_config):
        assert cp.resolve_runtime_pool_key("custom", URL, owner_api_key="stranger-key") == "custom"
        assert cp.credential_pool_matches_provider("custom:own", "custom", base_url=URL, owner_api_key="stranger-key") is False


class TestOwnerEntryShapes:
    def test_credentialless_entry_serves_any_owner(self):
        # #100413: entries without credentials still belong to the model.
        entry = {"name": "Local", "base_url": URL}
        assert cp._custom_entry_serves_owner_key(entry, MAIN_KEY) is True

    def test_key_cmd_entry_belongs_to_another_provider(self):
        entry = {"name": "Cmd", "base_url": URL, "key_cmd": "mint-token"}
        assert cp._custom_entry_serves_owner_key(entry, MAIN_KEY) is False

    def test_key_env_resolving_to_owner_serves(self):
        entry = {"name": "Env", "base_url": URL, "key_env": "OWN_KEY_VAR"}
        with patch.object(cp, "get_secret_str", return_value=MAIN_KEY):
            assert cp._custom_entry_serves_owner_key(entry, MAIN_KEY) is True

    def test_key_env_resolving_elsewhere_rejects(self):
        entry = {"name": "Env", "base_url": URL, "key_env": "OWN_KEY_VAR"}
        with patch.object(cp, "get_secret_str", return_value=SECOND_KEY):
            assert cp._custom_entry_serves_owner_key(entry, MAIN_KEY) is False

    def test_key_env_unset_rejects(self):
        entry = {"name": "Env", "base_url": URL, "key_env": "MISSING_VAR"}
        with patch.object(cp, "get_secret_str", return_value=""):
            assert cp._custom_entry_serves_owner_key(entry, MAIN_KEY) is False

    def test_unresolved_placeholder_without_key_env_rejects(self):
        entry = {"name": "Model", "base_url": URL, "api_key": "${MODEL_KEY_VAR}"}
        assert cp._custom_entry_serves_owner_key(entry, "${MODEL_KEY_VAR}") is False


class TestInitDropsSiblingPool:
    def _stub_agent(self, pool_provider):
        agent = MagicMock()
        agent.provider = "custom"
        agent.base_url = URL
        agent.api_key = MAIN_KEY
        agent.model = "test-model"
        agent.api_mode = "chat_completions"
        agent._credential_pool = SimpleNamespace(provider=pool_provider)
        agent._is_openrouter_url.return_value = False
        return agent

    def test_finalize_routing_drops_sibling_pool(self, sibling_config):
        from agent.agent_init import _finalize_routing
        agent = self._stub_agent("custom:second")
        _finalize_routing(agent, "chat_completions", agent._credential_pool)
        assert agent._credential_pool is None

    def test_finalize_routing_keeps_own_pool(self, sibling_config):
        from agent.agent_init import _finalize_routing
        pool = SimpleNamespace(provider="custom:own")
        agent = self._stub_agent("custom:own")
        agent._credential_pool = pool
        _finalize_routing(agent, "chat_completions", pool)
        assert agent._credential_pool is pool


class TestDelegationLeasesOwnPool:
    def test_bare_custom_child_with_own_key_leases_own_pool(self, sibling_config):
        from tools.delegate_tool_config import _resolve_child_credential_pool
        parent = SimpleNamespace(provider="custom", base_url=URL, requested_provider=None, _credential_pool=None)
        loaded = {}

        def _fake_loaded_pool(key):
            loaded["key"] = key
            pool = MagicMock()
            pool.has_credentials.return_value = True
            return pool

        with patch("tools.delegate_tool_config._loaded_pool", side_effect=_fake_loaded_pool):
            pool = _resolve_child_credential_pool("custom", parent, URL, owner_api_key=MAIN_KEY)
        assert loaded["key"] == "custom:own"
        assert pool is not None

    def test_bare_custom_child_without_matching_entry_keeps_fixed_credential(self, sibling_config):
        from tools.delegate_tool_config import _resolve_child_credential_pool
        parent = SimpleNamespace(provider="custom", base_url=URL, requested_provider=None, _credential_pool=None)
        with patch("tools.delegate_tool_config._loaded_pool") as mock_loaded:
            pool = _resolve_child_credential_pool("custom", parent, URL, owner_api_key="stranger-key")
        assert pool is None
        mock_loaded.assert_not_called()
