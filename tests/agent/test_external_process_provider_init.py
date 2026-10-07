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


def _routing_agent(provider: str, base_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider, base_url=base_url, model="gpt-5.4", api_mode="chat_completions",
        _get_transport=lambda: None, _is_azure_openai_url=lambda: False, _is_openrouter_url=lambda: False,
        _is_direct_openai_url=lambda: base_url.startswith("https://api.openai.com"),
        _provider_model_requires_responses_api=lambda model, provider=None: False, _transport_cache={})


def test_responses_upgrade_is_skipped_by_acp_scheme_not_vendor_slug(monkeypatch):
    """The Responses auto-upgrade guard keys on the ``acp://`` scheme alone: every external-process
    provider on an ACP marker keeps chat_completions (bundled and out-of-tree alike), while a direct
    OpenAI URL is upgraded whatever the slug — the vendor literal carried no behaviour of its own."""
    from agent.agent_init import _finalize_routing

    monkeypatch.setattr("hermes_cli.anon_auth.pin_model_for_route", lambda provider, base_url, model: model)
    for provider, base_url in (("copilot-acp", "acp://copilot"), ("acme-acp", "acp://acme"),
                               ("acme-acp", "acp+tcp://127.0.0.1:9000")):
        agent = _routing_agent(provider, base_url)
        _finalize_routing(agent, None, None)
        assert agent.api_mode == "chat_completions", (provider, base_url)

    upgraded = _routing_agent("acme-acp", "https://api.openai.com/v1")
    _finalize_routing(upgraded, None, None)
    assert upgraded.api_mode == "codex_responses"


def test_responses_upgrade_is_skipped_for_external_process_profile_on_any_base_url(monkeypatch):
    """An ``<X>_ACP_BASE_URL`` override may carry an https marker, so the ACP guard must also key on
    the profile's ``external_process`` auth_type — an ACP client never speaks the Responses API."""
    from agent.agent_init import _finalize_routing
    from providers.base import ProviderProfile

    profile = ProviderProfile(name="copilot-acp", auth_type="external_process")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name == "copilot-acp" else None)
    monkeypatch.setattr("hermes_cli.anon_auth.pin_model_for_route", lambda provider, base_url, model: model)

    agent = _routing_agent("copilot-acp", "https://proxy.example.invalid/v1")
    agent._provider_model_requires_responses_api = lambda model, provider=None: True
    _finalize_routing(agent, None, None)
    assert agent.api_mode == "chat_completions"

    plain = _routing_agent("acme-http", "https://proxy.example.invalid/v1")
    plain._provider_model_requires_responses_api = lambda model, provider=None: True
    _finalize_routing(plain, None, None)
    assert plain.api_mode == "codex_responses"


def test_fallback_activation_keeps_external_process_provider_on_chat_completions(monkeypatch):
    """GPT-5 fallback activation re-derives api_mode through ``_provider_model_requires_responses_api``;
    an external-process (ACP) facade has no ``responses`` surface, so the predicate must decline for
    any such profile — the bundled copilot-acp and an out-of-tree one alike (#65842, #107754)."""
    from agent.chat_completion_helpers import _fallback_api_mode_resolved
    from providers.base import ProviderProfile
    from run_agent import AIAgent

    profiles = {name: ProviderProfile(name=name, auth_type="external_process") for name in ("copilot-acp", "acme-acp")}
    monkeypatch.setattr("providers.get_provider_profile", profiles.get)
    agent = SimpleNamespace(
        _is_azure_openai_url=lambda url: False, _is_direct_openai_url=lambda url: False,
        _provider_model_requires_responses_api=AIAgent._provider_model_requires_responses_api)

    for provider in profiles:
        assert _fallback_api_mode_resolved(agent, provider, "gpt-5.6", "https://proxy.example.invalid/v1") == "chat_completions", provider
    assert _fallback_api_mode_resolved(agent, "acme-http", "gpt-5.6", "https://proxy.example.invalid/v1") == "codex_responses"


def test_should_stream_is_off_for_any_external_process_profile(monkeypatch):
    """Streaming is disabled for every ACP provider, keyed on the profile — not on the ``acp://``
    marker alone and not on one vendor's slug."""
    from agent.turn_api_call import _should_stream
    from providers.base import ProviderProfile

    profile = ProviderProfile(name="acme-acp", auth_type="external_process")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name == "acme-acp" else None)
    make = lambda provider: SimpleNamespace(  # noqa: E731
        provider=provider, base_url="https://proxy.example.invalid/v1", _has_stream_consumers=lambda: True)

    assert _should_stream(make("acme-acp")) is False
    assert _should_stream(make("acme-http")) is True
