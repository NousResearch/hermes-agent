"""External-process profiles receive their resolved ACP launch details at client construction."""

from types import SimpleNamespace


def test_explicit_client_kwargs_injects_command_for_any_external_process_profile(monkeypatch):
    from agent.agent_init import _explicit_client_kwargs
    from providers.base import ProviderProfile

    profile = ProviderProfile(name="test-process-provider", auth_type="external_process")
    monkeypatch.setattr("providers.get_provider_profile", lambda _name: profile)
    agent = SimpleNamespace(
        provider="test-process-provider", acp_command="/tmp/test-process", acp_args=["--acp", "--stdio"])

    kwargs = _explicit_client_kwargs(agent, "process-placeholder", "acp://test-process", None)

    assert kwargs["command"] == "/tmp/test-process"
    assert kwargs["args"] == ["--acp", "--stdio"]


def _route(provider: str, base_url: str, model: str = "gpt-5.4"):
    from providers.routing import InvocationRequest, resolve_invocation_route

    return resolve_invocation_route(InvocationRequest(
        provider=provider,
        base_url=base_url,
        model=model,
    ))


def test_responses_upgrade_is_skipped_by_acp_scheme_not_vendor_slug(monkeypatch):
    """ACP schemes are external-process routes; the same slug on an HTTP OpenAI URL is not."""
    monkeypatch.setattr("providers.routing._get_profile", lambda _name: None)

    for provider, base_url in (
        ("copilot-acp", "acp://copilot"),
        ("acme-acp", "acp://acme"),
        ("acme-acp", "acp+tcp://127.0.0.1:9000"),
    ):
        route = _route(provider, base_url)
        assert route.api_mode == "chat_completions", (provider, base_url)
        assert route.runtime_kind == "external_process", (provider, base_url)

    upgraded = _route("acme-acp", "https://api.openai.com/v1")
    assert upgraded.api_mode == "codex_responses"
    assert upgraded.runtime_kind == "http"


def test_responses_upgrade_is_skipped_for_external_process_profile_on_any_base_url(monkeypatch):
    """An external-process profile keeps chat semantics even when its marker is an HTTPS URL."""
    from providers.base import ProviderProfile

    process_profile = ProviderProfile(name="acme-acp", auth_type="external_process")

    def profile_for(name):
        return process_profile if name == "acme-acp" else None

    monkeypatch.setattr("providers.routing._get_profile", profile_for)

    process_route = _route("acme-acp", "https://proxy.example.invalid/v1", "gpt-5.6")
    assert process_route.api_mode == "chat_completions"
    assert process_route.runtime_kind == "external_process"

    plain = _route("acme-http", "https://proxy.example.invalid/v1", "gpt-5.6")
    assert plain.api_mode == "codex_responses"
    assert plain.runtime_kind == "http"


def test_fallback_activation_keeps_external_process_provider_on_chat_completions(monkeypatch):
    """Fallback route projection uses the same canonical external-process/model policy."""
    from agent.chat_completion_helpers import _fallback_api_mode_resolved
    from providers.base import ProviderProfile

    profiles = {
        name: ProviderProfile(name=name, auth_type="external_process")
        for name in ("copilot-acp", "acme-acp")
    }
    monkeypatch.setattr("providers.routing._get_profile", profiles.get)
    agent = SimpleNamespace()

    for provider in profiles:
        assert _fallback_api_mode_resolved(
            agent, provider, "gpt-5.6", "https://proxy.example.invalid/v1"
        ) == "chat_completions", provider

    assert _fallback_api_mode_resolved(
        agent, "acme-http", "gpt-5.6", "https://proxy.example.invalid/v1"
    ) == "codex_responses"


def test_should_stream_is_off_for_any_external_process_profile(monkeypatch):
    """Streaming is disabled for every ACP provider, keyed on the profile — not one vendor slug."""
    from agent.turn_api_call import _should_stream
    from providers.base import ProviderProfile

    profile = ProviderProfile(name="acme-acp", auth_type="external_process")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name == "acme-acp" else None)
    make = lambda provider: SimpleNamespace(  # noqa: E731
        provider=provider, base_url="https://proxy.example.invalid/v1", _has_stream_consumers=lambda: True)

    assert _should_stream(make("acme-acp")) is False
    assert _should_stream(make("acme-http")) is True
