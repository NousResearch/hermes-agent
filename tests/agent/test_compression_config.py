"""Tests for model-specific compression config overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.compression_config import resolve_compression_settings


def test_global_compression_settings_remain_the_default() -> None:
    settings = resolve_compression_settings(
        model="claude-sonnet-4.6",
        compression_cfg={"threshold": 0.42, "target_ratio": 0.30},
        model_cfg={},
    )

    assert settings.threshold == 0.42
    assert settings.target_ratio == 0.30


def test_model_compression_overrides_global_threshold_and_target_ratio() -> None:
    settings = resolve_compression_settings(
        model="deepseek-v4-pro",
        compression_cfg={"threshold": 0.50, "target_ratio": 0.20},
        model_cfg={"compression": {"threshold": 0.15, "target_ratio": 0.25}},
    )

    assert settings.threshold == 0.15
    assert settings.target_ratio == 0.25


def test_hardcoded_model_threshold_still_applies_without_user_override() -> None:
    settings = resolve_compression_settings(
        model="arcee-ai/trinity-large-thinking",
        provider="openrouter",
        compression_cfg={"threshold": 0.50, "target_ratio": 0.20},
        model_cfg={},
    )

    assert settings.threshold == 0.75
    assert settings.threshold_autoraised is None
    assert settings.target_ratio == 0.20


def test_explicit_model_threshold_wins_over_hardcoded_model_default() -> None:
    settings = resolve_compression_settings(
        model="arcee-ai/trinity-large-thinking",
        provider="openrouter",
        compression_cfg={"threshold": 0.50, "target_ratio": 0.20},
        model_cfg={"compression": {"threshold": 0.60}},
    )

    assert settings.threshold == 0.60
    assert settings.threshold_autoraised is None
    assert settings.target_ratio == 0.20


def test_codex_gpt55_autoraise_is_preserved_for_codex_provider() -> None:
    settings = resolve_compression_settings(
        model="gpt-5.5",
        provider="openai-codex",
        compression_cfg={"threshold": 0.50, "target_ratio": 0.20},
        model_cfg={},
    )

    assert settings.threshold == 0.85
    assert settings.threshold_autoraised == {"from": 0.50, "to": 0.85}
    assert settings.target_ratio == 0.20


def test_model_threshold_override_suppresses_codex_gpt55_autoraise() -> None:
    settings = resolve_compression_settings(
        model="gpt-5.5",
        provider="openai-codex",
        compression_cfg={"threshold": 0.50, "target_ratio": 0.20},
        model_cfg={"compression": {"threshold": 0.60}},
    )

    assert settings.threshold == 0.60
    assert settings.threshold_autoraised is None
    assert settings.target_ratio == 0.20


def test_agent_init_passes_model_compression_settings_to_context_compressor() -> None:
    from run_agent import AIAgent

    compressor = MagicMock()
    compressor.context_length = 1_000_000
    compressor.threshold_tokens = 150_000
    cfg = {
        "model": {
            "default": "deepseek-v4-pro",
            "compression": {"threshold": 0.15, "target_ratio": 0.25},
        },
        "compression": {"threshold": 0.50, "target_ratio": 0.20},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.agent_init.ContextCompressor", return_value=compressor) as mock_compressor,
    ):
        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="deepseek-v4-pro",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    kwargs = mock_compressor.call_args.kwargs
    assert kwargs["threshold_percent"] == 0.15
    assert kwargs["summary_target_ratio"] == 0.25
