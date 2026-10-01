"""Regression coverage for authoritative auxiliary main-runtime routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_text_named_custom_keeps_live_endpoint_over_configured_defaults():
    client = MagicMock()
    with patch(
        "agent.auxiliary_client.get_configured_provider_entry",
        return_value={"base_url": "https://configured.example/v1"},
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "live-model"),
    ) as resolve:
        from agent.auxiliary_client import _resolve_auto_route

        routed, model, provider = _resolve_auto_route(main_runtime={
            "provider": "custom",
            "requested_provider": "custom:edge",
            "model": "live-model",
            "base_url": "https://live.example/anthropic",
            "api_key": "live-key",
            "api_mode": "anthropic_messages",
        })

    assert routed is client
    assert model == "live-model"
    assert provider == "custom:edge"
    assert resolve.call_args.args[:2] == ("custom:edge", "live-model")
    assert resolve.call_args.kwargs["explicit_base_url"] == "https://live.example/anthropic"
    assert resolve.call_args.kwargs["explicit_api_key"] == "live-key"
    assert resolve.call_args.kwargs["api_mode"] == "anthropic_messages"


def test_main_route_preserves_requested_alias_while_canonicalizing_provider():
    from agent.auxiliary_client import _main_route_target

    route = _main_route_target({
        "provider": "github-copilot",
        "requested_provider": "github-copilot",
        "model": "gpt-5-mini",
        "base_url": "https://api.githubcopilot.com",
        "api_key": "live-token",
        "api_mode": "chat_completions",
    }, None)

    assert route.provider == "copilot"
    assert route.requested_provider == "github-copilot"
    assert route.base_url == "https://api.githubcopilot.com"
    assert route.api_key == "live-token"
    assert route.api_mode == "chat_completions"


def test_vision_named_custom_reuses_same_live_route():
    client = MagicMock()
    with patch(
        "agent.auxiliary_client.get_configured_provider_entry",
        return_value={"base_url": "https://configured.example/v1"},
    ), patch(
        "agent.auxiliary_client.get_provider_profile",
        return_value=SimpleNamespace(rejects_vision_input=False, default_vision_model=lambda: "vision-model"),
    ), patch(
        "agent.auxiliary_client.resolve_supports_vision",
        return_value=True,
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "vision-model"),
    ) as resolve:
        from agent.auxiliary_client import _vision_auto_route

        provider, routed, model = _vision_auto_route({
            "provider": "custom:edge",
            "requested_provider": "custom:edge",
            "model": "main-model",
            "base_url": "https://live.example/anthropic",
            "api_key": "live-key",
            "api_mode": "anthropic_messages",
        }, None, None, False)

    assert (provider, routed, model) == ("custom:edge", client, "vision-model")
    assert resolve.call_args.args[:2] == ("custom:edge", "vision-model")
    assert resolve.call_args.kwargs["explicit_base_url"] == "https://live.example/anthropic"
    assert resolve.call_args.kwargs["explicit_api_key"] == "live-key"
    assert resolve.call_args.kwargs["api_mode"] == "anthropic_messages"


def test_vision_without_live_endpoint_leaves_config_resolution_to_provider_branch():
    client = MagicMock()
    with patch(
        "agent.auxiliary_client.get_provider_profile",
        return_value=SimpleNamespace(rejects_vision_input=False, default_vision_model=lambda: "vision-model"),
    ), patch(
        "agent.auxiliary_client.resolve_supports_vision",
        return_value=True,
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ), patch(
        "agent.auxiliary_client._resolve_custom_runtime",
    ) as legacy_resolve, patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "vision-model"),
    ) as resolve:
        from agent.auxiliary_client import _vision_auto_route

        provider, routed, model = _vision_auto_route({
            "provider": "custom",
            "model": "main-model",
        }, None, None, False)

    assert (provider, routed, model) == ("custom", client, "vision-model")
    legacy_resolve.assert_not_called()
    assert resolve.call_args.kwargs["explicit_base_url"] is None
    assert resolve.call_args.kwargs["explicit_api_key"] is None


def test_agent_runtime_snapshot_exports_requested_provider():
    from run_agent import AIAgent

    agent = SimpleNamespace(
        model="gpt-5-mini",
        provider="copilot",
        requested_provider="github-copilot",
        base_url="https://api.githubcopilot.com",
        api_key="live-token",
        api_mode="chat_completions",
        runtime_kind="http",
        auth_mode="copilot",
        session_id="session-1",
    )

    runtime = AIAgent._current_main_runtime(agent)

    assert runtime["provider"] == "copilot"
    assert runtime["requested_provider"] == "github-copilot"
