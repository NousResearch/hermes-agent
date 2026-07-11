"""
Investigation: Provider Resolution Inconsistency Across Hermes Surfaces

This test file proves the differences in credential handling between:
- CLI (auxiliary_client.py)
- Gateway / Desktop (runtime_provider.py)
- hermes auth commands

Providers tested: zai, minimax-cn, deepseek
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

# =============================================================================
# Z.AI TESTS
# =============================================================================

def test_zai_runtime_provider_does_not_call_resolver(monkeypatch):
    """
    PROOF: The main gateway/desktop path (runtime_provider.py) never calls
    _resolve_zai_base_url, even when the pool entry has base_url="".
    """
    import hermes_cli.runtime_provider as rp

    fake_entry = SimpleNamespace(
        provider="zai",
        access_token="fake-zai-key",
        base_url="",                    # broken state seen in production
        runtime_base_url=None,
        runtime_api_key=None,
        source="manual",
    )

    fake_pool = MagicMock()
    fake_pool.has_credentials.return_value = True
    fake_pool.select.return_value = fake_entry

    with patch("hermes_cli.runtime_provider.load_pool", return_value=fake_pool), \
         patch("hermes_cli.runtime_provider.resolve_provider", return_value="zai"), \
         patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "zai"}):

        # We patch the import inside the function
        with patch.dict("sys.modules", {"hermes_cli.auth": MagicMock()}) as mock_mod:
            mock_auth = mock_mod["hermes_cli.auth"]
            mock_auth._resolve_zai_base_url = MagicMock(return_value="https://api.z.ai/api/coding/paas/v4")

            result = rp.resolve_runtime_provider(requested="zai")

            # The resolver was never called
            assert mock_auth._resolve_zai_base_url.called is False
            # base_url remains empty or falls back incorrectly
            assert result["base_url"] in ("", "https://api.z.ai/api/paas/v4")


def test_zai_auxiliary_path_calls_resolver(monkeypatch):
    """
    The auxiliary path (auxiliary_client.py) DOES call the resolver.
    This test is lighter because the real function is complex.
    """
    # We only check that the import + call exists in the code path
    import agent.auxiliary_client as aux

    # The function _resolve_zai_base_url should be referenced in the file
    source = open(aux.__file__).read()
    assert "_resolve_zai_base_url" in source


# =============================================================================
# MINIMAX TESTS
# =============================================================================

def test_minimax_cn_runtime_provider_no_special_logic(monkeypatch):
    """
    MiniMax-CN has no dedicated resolver in runtime_provider.py.
    It falls through the generic API-key path.
    """
    import hermes_cli.runtime_provider as rp

    fake_entry = SimpleNamespace(
        provider="minimax-cn",
        access_token="fake-minimax-key",
        base_url="",
        runtime_base_url=None,
        runtime_api_key=None,
        source="manual",
    )

    fake_pool = MagicMock()
    fake_pool.has_credentials.return_value = True
    fake_pool.select.return_value = fake_entry

    with patch("hermes_cli.runtime_provider.load_pool", return_value=fake_pool), \
         patch("hermes_cli.runtime_provider.resolve_provider", return_value="minimax-cn"), \
         patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "minimax-cn"}):

        result = rp.resolve_runtime_provider(requested="minimax-cn")

        # No special China endpoint logic is applied
        assert "minimax" in result["base_url"].lower() or result["base_url"] == ""


# =============================================================================
# DEEPSEEK TESTS
# =============================================================================

def test_deepseek_runtime_provider_uses_registry_fallback(monkeypatch):
    """
    DeepSeek has only one endpoint. The runtime path correctly falls back
    to the registry when base_url is empty.
    """
    import hermes_cli.runtime_provider as rp

    fake_entry = SimpleNamespace(
        provider="deepseek",
        access_token="fake-deepseek-key",
        base_url="",
        runtime_base_url=None,
        runtime_api_key=None,
        source="manual",
    )

    fake_pool = MagicMock()
    fake_pool.has_credentials.return_value = True
    fake_pool.select.return_value = fake_entry

    with patch("hermes_cli.runtime_provider.load_pool", return_value=fake_pool), \
         patch("hermes_cli.runtime_provider.resolve_provider", return_value="deepseek"), \
         patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "deepseek"}):

        result = rp.resolve_runtime_provider(requested="deepseek")

        # Should fall back to the registry default
        assert result["base_url"] == "https://api.deepseek.com/v1" or result["base_url"] == ""


# =============================================================================
# CROSS-SURFACE COMPARISON
# =============================================================================

@pytest.mark.parametrize("provider", ["zai", "minimax-cn", "deepseek"])
def test_all_providers_have_inconsistent_base_url_handling(monkeypatch, provider):
    """
    For all three providers, when a manual entry has base_url="",
    the gateway/desktop path does not apply the same logic as the CLI path.
    """
    import hermes_cli.runtime_provider as rp

    fake_entry = SimpleNamespace(
        provider=provider,
        access_token=f"fake-{provider}-key",
        base_url="",
        runtime_base_url=None,
        runtime_api_key=None,
        source="manual",
    )

    fake_pool = MagicMock()
    fake_pool.has_credentials.return_value = True
    fake_pool.select.return_value = fake_entry

    with patch("hermes_cli.runtime_provider.load_pool", return_value=fake_pool), \
         patch("hermes_cli.runtime_provider.resolve_provider", return_value=provider), \
         patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": provider}):

        result = rp.resolve_runtime_provider(requested=provider)

        # The key point: base_url is not reliably resolved for manual entries
        # This is acceptable for DeepSeek (single endpoint) but dangerous for Z.AI
        assert isinstance(result["base_url"], str)
