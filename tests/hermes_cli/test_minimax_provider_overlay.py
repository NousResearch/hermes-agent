from hermes_cli.providers import get_provider


def test_minimax_offline_overlay_preserves_api_key_env(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_provider_info", lambda *args, **kwargs: None)

    provider = get_provider("minimax", allow_network=False)

    assert provider is not None
    assert provider.source == "hermes"
    assert provider.transport == "anthropic_messages"
    assert provider.api_key_env_vars == ("MINIMAX_API_KEY",)
    assert provider.base_url_env_var == "MINIMAX_BASE_URL"


def test_minimax_cn_offline_overlay_preserves_api_key_env(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_provider_info", lambda *args, **kwargs: None)

    provider = get_provider("minimax-cn", allow_network=False)

    assert provider is not None
    assert provider.source == "hermes"
    assert provider.transport == "anthropic_messages"
    assert provider.api_key_env_vars == ("MINIMAX_CN_API_KEY",)
    assert provider.base_url_env_var == "MINIMAX_CN_BASE_URL"
