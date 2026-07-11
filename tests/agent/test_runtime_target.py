from agent.runtime_target import (
    CLAUDE_AGENT_SDK_RUNTIME,
    CODEX_APP_SERVER_RUNTIME,
    HERMES_RUNTIME,
    attach_runtime_identity,
    resolve_runtime_identity,
)


def test_explicit_runtime_selects_claude_agent_sdk():
    identity = resolve_runtime_identity(
        provider="anthropic",
        api_mode="anthropic_messages",
        route_config={"runtime": "claude_agent_sdk"},
    )

    assert identity == CLAUDE_AGENT_SDK_RUNTIME
    assert identity != HERMES_RUNTIME


def test_openai_runtime_compatibility_alias_selects_codex_app_server():
    identity = resolve_runtime_identity(
        provider="openai-codex",
        api_mode="codex_responses",
        route_config={"openai_runtime": "codex_app_server"},
    )

    assert identity == CODEX_APP_SERVER_RUNTIME


def test_unknown_explicit_runtime_fails_closed():
    with pytest.raises(ValueError, match="Unknown agent runtime"):
        resolve_runtime_identity(
            provider="anthropic",
            api_mode="anthropic_messages",
            route_config={"runtime": "mystery_runtime"},
        )


def test_resolved_target_carries_provider_neutral_runtime_identity():
    resolved = {
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
        "base_url": "https://api.anthropic.com",
    }

    target = attach_runtime_identity(
        resolved,
        route_config={"runtime": "claude_agent_sdk"},
    )

    assert target["runtime"] == CLAUDE_AGENT_SDK_RUNTIME
    assert resolved.get("runtime") is None


def test_provider_resolution_attaches_configured_runtime(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        """
model:
  provider: anthropic
  default: claude-sonnet-4-6
  runtime: claude_agent_sdk
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.runtime_provider import resolve_runtime_provider

    target = resolve_runtime_provider(
        requested="anthropic",
        explicit_api_key="test-token",
        explicit_base_url="https://api.anthropic.com",
        target_model="claude-sonnet-4-6",
    )

    assert target["runtime"] == CLAUDE_AGENT_SDK_RUNTIME


def test_claude_subscription_route_resolves_without_paid_api_credentials(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        """
model:
  provider: anthropic
  default: claude-sonnet-4-6
  runtime: claude_agent_sdk
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    from hermes_cli.runtime_provider import resolve_runtime_provider

    target = resolve_runtime_provider(
        requested="anthropic",
        target_model="claude-sonnet-4-6",
    )

    assert target == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_mode": "anthropic_messages",
        "runtime": "claude_agent_sdk",
        "api_key": "",
        "base_url": "",
        "source": "claude_max_subscription",
        "credential_pool": None,
    }


def test_explicit_other_provider_does_not_inherit_global_claude_runtime(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        """
model:
  provider: anthropic
  default: claude-sonnet-4-6
  runtime: claude_agent_sdk
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.runtime_provider import resolve_runtime_provider

    target = resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="test-token",
        target_model="gpt-5",
    )

    assert target["provider"] == "openrouter"
    assert target["runtime"] == HERMES_RUNTIME
import pytest
