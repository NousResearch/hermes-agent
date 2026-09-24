"""Named custom routes use the same reasoning contract as bare custom.

Regression for https://github.com/NousResearch/hermes-agent/issues/119681
"""
from providers import get_provider_profile, register_provider
from providers.base import ProviderProfile
from agent.transports.chat_completions import ChatCompletionsTransport


def test_bare_named_custom_provider_gets_custom_profile(monkeypatch):
    """Bare name (no ``custom:`` prefix) configured in providers: resolves to custom profile.

    Regression for https://github.com/NousResearch/hermes-agent/issues/119681
    """
    import providers
    get_provider_profile("custom")
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    custom_profile = get_provider_profile("custom")
    assert custom_profile is not None

    def fake_has_named(name):
        return name == "my-endpoint"

    with monkeypatch.context() as m:
        m.setattr("hermes_cli.runtime_provider_custom.has_named_custom_provider", fake_has_named)
        result = get_provider_profile("my-endpoint")
    assert result is custom_profile


def test_named_custom_route_keeps_final_reasoning_effort():
    transport = ChatCompletionsTransport()
    for effort in ("low", "medium", "high"):
        outputs = [transport.build_kwargs(
            "fixture-model", [{"role": "user", "content": "fixture"}],
            provider_profile=get_provider_profile(provider),
            base_url="http://127.0.0.1:1/v1",
            reasoning_config={"enabled": True, "effort": effort},
        ) for provider in ("custom", "custom:fixture")]
        assert outputs[0]["reasoning_effort"] == effort
        assert outputs[1] == outputs[0]


def test_named_custom_fallback_does_not_override_registered_routes(monkeypatch):
    import providers
    get_provider_profile("custom")
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    dedicated = ProviderProfile(name="custom:fixture")
    register_provider(dedicated)
    assert get_provider_profile("custom:fixture") is dedicated
    assert get_provider_profile("CUSTOM:unregistered") is get_provider_profile("custom")
    assert get_provider_profile("NONEXISTENT") is None
