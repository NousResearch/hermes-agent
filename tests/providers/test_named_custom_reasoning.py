"""Named custom routes use the same reasoning contract as bare custom.

Regression for https://github.com/NousResearch/hermes-agent/issues/119681
"""
from providers import get_provider_profile, register_provider
from providers.base import ProviderProfile
from agent.transports.chat_completions import ChatCompletionsTransport
import pytest


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


@pytest.mark.parametrize("provider", ["ollama", "custom:projection-fixture", "projection-named-custom"])
def test_glm53_projection_uses_generic_custom_for_aliases_and_named_custom(provider, monkeypatch):
    import providers
    from gateway.session_local import effort_fields

    assert get_provider_profile("custom") is not None
    monkeypatch.setattr(
        providers, "_has_named_custom_provider",
        lambda name, home, hkey: name == "projection-named-custom",
    )
    fields = effort_fields(
        {"enabled": True, "effort": "xhigh"}, "", "glm-5.3-flash", provider,
        "https://relay.example/v1", "chat_completions",
    )
    assert fields["reasoning_effort_wire"] == "max"


def test_dedicated_custom_route_does_not_inherit_generic_glm53_mapping(monkeypatch):
    import providers
    from agent.reasoning_effort import GLM53_EFFORTS
    from agent.transports.codex import _resolve_reasoning
    from gateway.session_local import effort_fields

    get_provider_profile("custom")
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    dedicated = ProviderProfile(name="custom:fixture")
    monkeypatch.setattr(dedicated, "supported_reasoning_efforts", lambda model: GLM53_EFFORTS)
    register_provider(dedicated)

    fields = effort_fields(
        {"enabled": True, "effort": "xhigh"}, "", "glm-5.3-flash", "custom:fixture",
        "https://relay.example/v1", "chat_completions",
    )
    assert fields["reasoning_effort_wire"] == "xhigh"
    disabled = effort_fields(
        {"enabled": False}, "", "glm-5.3-flash", "custom:fixture",
        "https://relay.example/v1", "chat_completions",
    )
    assert disabled["reasoning_effort_wire"] == ""
    params = {"provider": "custom:fixture", "base_url": "https://relay.example/v1"}
    assert _resolve_reasoning(
        "glm-5.3-flash", {**params, "reasoning_config": {"enabled": True, "effort": "xhigh"}},
    ) == ("high", True)
    assert _resolve_reasoning(
        "glm-5.3-flash", {**params, "reasoning_config": {"enabled": False}},
    ) == (None, False)


def test_bare_named_custom_memo_caches_the_helper_result(monkeypatch):
    """The miss-path helper runs once per (home, name); repeats are memo hits.

    Uncached it walks the provider registry per call (~2.5ms; #120901 review),
    which lands inside per-model prefix loops.
    """
    import providers
    from hermes_constants import get_hermes_home, hermes_home_key

    home, hkey = get_hermes_home(), hermes_home_key()
    calls = []

    def fake(name):
        calls.append(name)
        return False

    with monkeypatch.context() as m:
        m.setattr("hermes_cli.runtime_provider_custom.has_named_custom_provider", fake)
        assert providers._has_named_custom_provider("zzz-unknown", home, hkey) is False
        assert providers._has_named_custom_provider("zzz-unknown", home, hkey) is False
    assert calls == ["zzz-unknown"], f"second lookup must be a memo hit, got calls={calls}"


def test_bare_named_custom_memo_invalidates_on_config_change(monkeypatch):
    """A config edit re-arms the memo: the signature is the same signal load_config uses."""
    import providers
    from hermes_constants import get_hermes_home, hermes_home_key

    home, hkey = get_hermes_home(), hermes_home_key()
    cfg = home / "config.yaml"

    # No config file: nothing configured.
    assert providers._has_named_custom_provider("my-endpoint", home, hkey) is False

    cfg.write_text("providers:\n  my-endpoint:\n    base_url: http://127.0.0.1:1/v1\n")
    assert providers._has_named_custom_provider("my-endpoint", home, hkey) is True

    cfg.write_text("{}\n")
    assert providers._has_named_custom_provider("my-endpoint", home, hkey) is False


def test_bare_named_custom_memo_is_keyed_by_home(monkeypatch):
    """Two home keys never borrow each other's answers (multiplex isolation)."""
    import providers
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    answers = {"key-a": True, "key-b": False}
    current = {"k": "key-a"}

    def fake(name):
        return answers[current["k"]]

    with monkeypatch.context() as m:
        m.setattr("hermes_cli.runtime_provider_custom.has_named_custom_provider", fake)
        assert providers._has_named_custom_provider("n", home, "key-a") is True
        current["k"] = "key-b"
        assert providers._has_named_custom_provider("n", home, "key-b") is False
        # key-a's slot still holds True — key-b's False never leaked into it.
        assert providers._has_named_custom_provider("n", home, "key-a") is True
