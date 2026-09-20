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
