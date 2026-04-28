import hermes_cli.nous_subscription as nous_subscription


def test_web_feature_marks_searxng_as_active_provider(monkeypatch):
    config = {
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "web": {
            "backend": "searxng",
            "search_backend": "searxng",
            "searxng": {"base_url": "http://localhost:8888"},
        },
    }

    monkeypatch.setattr(nous_subscription, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(nous_subscription, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(nous_subscription, "is_managed_tool_gateway_ready", lambda vendor: False)
    monkeypatch.setattr(nous_subscription, "_toolset_enabled", lambda cfg, key: key == "web")
    monkeypatch.setattr(nous_subscription, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(nous_subscription, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(nous_subscription, "_has_agent_browser", lambda: False)

    features = nous_subscription.get_nous_subscription_features(config)

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.current_provider == "searxng"
    assert features.web.managed_by_nous is False


def test_web_feature_uses_search_backend_when_legacy_backend_is_absent(monkeypatch):
    config = {
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "web": {
            "search_backend": "searxng",
            "searxng": {"base_url": "http://localhost:8888"},
        },
    }

    monkeypatch.setattr(nous_subscription, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(nous_subscription, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(nous_subscription, "is_managed_tool_gateway_ready", lambda vendor: False)
    monkeypatch.setattr(nous_subscription, "_toolset_enabled", lambda cfg, key: key == "web")
    monkeypatch.setattr(nous_subscription, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(nous_subscription, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(nous_subscription, "_has_agent_browser", lambda: False)

    features = nous_subscription.get_nous_subscription_features(config)

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.current_provider == "searxng"
