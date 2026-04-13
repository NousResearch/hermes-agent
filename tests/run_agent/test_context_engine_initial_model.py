"""Tests for initial context-engine model metadata sync."""

from unittest.mock import MagicMock, patch

from run_agent import _update_context_engine_model


def test_update_context_engine_model_resolves_context_length():
    engine = MagicMock()

    with patch("agent.model_metadata.get_model_context_length", return_value=204_800) as ctx_len:
        _update_context_engine_model(
            engine,
            model="minimax-m2.7",
            base_url="https://api.example.test/v1",
            api_key="sk-test",
            provider="minimax",
            api_mode="anthropic_messages",
            config_context_length=101_000,
        )

    ctx_len.assert_called_once_with(
        "minimax-m2.7",
        base_url="https://api.example.test/v1",
        api_key="sk-test",
        config_context_length=101_000,
        provider="minimax",
    )
    engine.update_model.assert_called_once_with(
        model="minimax-m2.7",
        context_length=204_800,
        base_url="https://api.example.test/v1",
        api_key="sk-test",
        provider="minimax",
        api_mode="anthropic_messages",
    )
