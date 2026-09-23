"""Gateway provider-override credential resolution keys off the override's MODEL (#112600).

Channel overrides, persisted ``/model`` switches and API-server provider refreshes all resolve
credentials through ``gateway.run._resolve_runtime_agent_kwargs_for_provider``; without the model
the ladder keys off config's ``default`` and a ``*-free`` default decides the api_mode/base_url
for a Go-only model ("Model ... is not supported")."""

import pytest


@pytest.fixture()
def _zen_free_default_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: mimo-v2.5-free\n  provider: opencode\n  base_url: https://opencode.ai/zen/v1\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test-go")


def test_provider_override_runtime_uses_the_override_model(_zen_free_default_home):
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

    runtime = _resolve_runtime_agent_kwargs_for_provider("opencode-go", target_model="mimo-v2.5")
    assert runtime["provider"] == "opencode-go"
    assert runtime["base_url"] == "https://opencode.ai/zen/go/v1"


def test_provider_override_auth_error_uses_configured_fallback(monkeypatch):
    """Channel overrides use the shared fallback chain and retain their target model."""
    import gateway.run as gateway_run
    import hermes_cli.runtime_provider as rp
    from hermes_cli.auth import AuthError

    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if kwargs["requested"] == "openai-codex":
            raise AuthError("No Codex credentials stored")
        return {
            "provider": kwargs["requested"],
            "api_key": "fallback-key",
            "base_url": "https://anthropic.example.test",
        }

    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"fallback_providers": [{"provider": "anthropic", "model": "claude-fallback"}]},
    )

    runtime = gateway_run._resolve_runtime_agent_kwargs_for_provider(
        "openai-codex", target_model="gpt-override"
    )

    assert runtime["provider"] == "anthropic"
    assert runtime["api_key"] == "fallback-key"
    assert [(call["requested"], call["target_model"]) for call in calls] == [
        ("openai-codex", "gpt-override"),
        ("anthropic", "claude-fallback"),
    ]


def test_provider_override_success_does_not_walk_fallback(monkeypatch):
    """A successful explicit provider resolution remains a single targeted lookup."""
    import gateway.run as gateway_run
    import hermes_cli.runtime_provider as rp

    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {"provider": "openai-codex", "api_key": "primary-key", "base_url": "https://api.example.test"}

    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"fallback_providers": []})

    runtime = gateway_run._resolve_runtime_agent_kwargs_for_provider(
        "openai-codex", target_model="gpt-override"
    )

    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == "primary-key"
    assert [(call["requested"], call["target_model"]) for call in calls] == [
        ("openai-codex", "gpt-override")
    ]


def test_fallback_chain_runtime_uses_the_entry_model(_zen_free_default_home, monkeypatch):
    """The gateway's AuthError fallback goes through the shared ``resolve_runtime_with_fallback``
    walker (no gateway-private loop) and keeps the entry's own model."""
    import gateway.run as gateway_run
    import hermes_cli.runtime_provider as rp
    from hermes_cli.auth import AuthError

    real_resolve = rp.resolve_runtime_provider

    def primary_auth_fails(**kw):
        if kw.get("requested") is None:  # the primary, resolved from config.yaml
            raise AuthError("primary quota exhausted (429)")
        return real_resolve(**kw)

    monkeypatch.setattr(rp, "resolve_runtime_provider", primary_auth_fails)
    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"fallback_model": [{"provider": "opencode-go", "model": "mimo-v2.5"}]})
    fb = gateway_run._resolve_runtime_agent_kwargs()
    assert fb["model"] == "mimo-v2.5"
    assert fb["provider"] == "opencode-go"
    assert fb["base_url"] == "https://opencode.ai/zen/go/v1"
