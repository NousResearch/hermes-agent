"""Regression tests for MiniMax OAuth auxiliary-client routing."""
from unittest.mock import MagicMock, patch


def test_minimax_oauth_builds_anthropic_auxiliary_client():
    import agent.auxiliary_client as aux

    token_provider = lambda: "fresh-token"
    real_client = MagicMock(name="real_anthropic_client")

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value={
            "provider": "minimax-oauth",
            "api_key": token_provider,
            "base_url": "https://api.minimax.io/anthropic",
            "source": "oauth",
        },
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=real_client,
    ) as build_client:
        client, model = aux.resolve_provider_client(
            "minimax-oauth", model="MiniMax-M3"
        )

    assert isinstance(client, aux.AnthropicAuxiliaryClient)
    assert model == "MiniMax-M3"
    assert client.base_url == "https://api.minimax.io/anthropic"
    assert client.api_key is token_provider
    build_client.assert_called_once()
    assert build_client.call_args.args[0] is token_provider
