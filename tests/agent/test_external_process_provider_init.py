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
