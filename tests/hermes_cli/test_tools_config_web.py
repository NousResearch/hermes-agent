from hermes_cli.nous_subscription import NousFeatureState, NousSubscriptionFeatures
import hermes_cli.tools_config as tools_config


def _features_stub() -> NousSubscriptionFeatures:
    return NousSubscriptionFeatures(
        subscribed=False,
        nous_auth_present=False,
        provider_is_nous=False,
        features={
            "web": NousFeatureState("web", "Web tools", True, False, False, False, False, True, ""),
            "image_gen": NousFeatureState("image_gen", "Image generation", True, False, False, False, False, True, ""),
            "tts": NousFeatureState("tts", "TTS", True, False, False, False, False, True, ""),
            "browser": NousFeatureState("browser", "Browser", True, False, False, False, False, True, ""),
            "modal": NousFeatureState("modal", "Modal", False, False, False, False, False, True, ""),
        },
    )


def _searxng_provider() -> dict:
    return {
        "name": "SearXNG (self-hosted)",
        "web_backend": "searxng",
        "env_vars": [],
        "config_prompts": [
            {
                "config_path": "web.searxng.base_url",
                "prompt": "SearXNG base URL",
                "default": "http://localhost:8888",
            }
        ],
    }


def test_configure_provider_sets_searxng_backend_and_base_url(monkeypatch):
    config = {}
    provider = _searxng_provider()

    monkeypatch.setattr(tools_config, "_prompt", lambda *args, **kwargs: "http://localhost:8888")
    monkeypatch.setattr(tools_config, "_print_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_config, "_print_info", lambda *args, **kwargs: None)

    tools_config._configure_provider(provider, config)

    assert config["web"]["backend"] == "searxng"
    assert config["web"]["search_backend"] == "searxng"
    assert config["web"]["use_gateway"] is False
    assert config["web"]["searxng"]["base_url"] == "http://localhost:8888"


def test_is_provider_active_considers_search_backend_for_searxng():
    provider = _searxng_provider()
    config = {"web": {"search_backend": "searxng"}}

    assert tools_config._is_provider_active(provider, config) is True


def test_toolset_has_keys_does_not_treat_unconfigured_searxng_provider_as_ready(monkeypatch):
    config = {}

    monkeypatch.setattr(tools_config, "get_nous_subscription_features", lambda cfg: _features_stub())

    assert tools_config._toolset_has_keys("web", config) is False
