"""A ``model_aliases`` credential bound via ``key_env`` stays on that variable through the per-request refresh.

``_try_refresh_anthropic_client_credentials`` runs before every native Anthropic request. Unbound, it
re-resolves the provider-default chain (``ANTHROPIC_API_KEY`` / OAuth), which for an alias such as
``key_env: ANTHROPIC_API_KEY_ALT`` belongs to a different account: the session silently bills that
account. Bound, it re-reads only that variable, so a rotated value is picked up and the default never is.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

ALIAS_ENV = "ANTHROPIC_API_KEY_ALT"
ALIAS_KEY = "sk-ant-api03-alias-account-AAAA"
ALIAS_KEY_ROTATED = "sk-ant-api03-alias-account-BBBB"
DEFAULT_KEY = "sk-ant-api03-default-account-ZZZZ"


def test_refresh_reads_only_the_bound_key_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DEFAULT_KEY)
    monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY)
    default_pool = MagicMock(name="anthropic-default-pool", provider="anthropic")
    with patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.anthropic_credentials.read_claude_code_credentials", return_value=None):
        agent = AIAgent(model="claude-opus-5-5", provider="anthropic", api_mode="anthropic_messages",
                        base_url="https://api.anthropic.com/v1", api_key=ALIAS_KEY, api_key_env=ALIAS_ENV,
                        credential_pool=default_pool, quiet_mode=True, skip_context_files=True, skip_memory=True)
        with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as rebuild:
            agent._try_refresh_anthropic_client_credentials()
            assert agent._anthropic_api_key == ALIAS_KEY, (
                f"refresh swapped in the provider-default key ...{agent._anthropic_api_key[-4:]}")
            assert agent._credential_pool is None  # its entries are the default account's keys
            rebuild.assert_not_called()

            monkeypatch.setenv(ALIAS_ENV, ALIAS_KEY_ROTATED)  # the alias key is rotated in .env
            assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == agent.api_key == ALIAS_KEY_ROTATED
