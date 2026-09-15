"""Regression coverage for provider-aware @-context sizing in the CLI."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_at_context_resolution_passes_active_provider():
    """The CLI @-reference path must preserve the active Codex provider."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "gpt-5.6-terra"
    cli.base_url = "https://chatgpt.com/backend-api/codex"
    cli.api_key = "token"
    cli.provider = "openai-codex"
    cli.agent = SimpleNamespace(_config_context_length=None)
    cli._active_agent_route_signature = "route"
    cli._secret_capture_callback = lambda *_args, **_kwargs: None
    cli._last_turn_interrupted = False
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda _message: {
        "signature": "route",
        "model": cli.model,
        "runtime": None,
        "request_overrides": None,
    }
    cli._init_agent = lambda **_kwargs: True

    blocked_result = SimpleNamespace(
        expanded=False,
        blocked=True,
        references=[],
        injected_tokens=0,
        warnings=["blocked for test"],
    )
    with patch("agent.context_references.preprocess_context_references", return_value=blocked_result), \
         patch("agent.model_metadata.get_model_context_length", return_value=372_000) as mock_context, \
         patch("cli._cprint"):
        result = cli.chat("inspect @file:example.py")

    assert result == "blocked for test"
    mock_context.assert_called_once_with(
        "gpt-5.6-terra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="token",
        provider="openai-codex",
        config_context_length=None,
    )
