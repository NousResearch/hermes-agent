"""A failed config load hands ``_parse_compression_config`` ``{}``; every key must fall back to
DEFAULT_CONFIG, and an explicit ``threshold_tokens: null`` must stay the ratio-only opt-out."""

from types import SimpleNamespace

import pytest

from agent.agent_init import _parse_compression_config
from hermes_cli.config import DEFAULT_CONFIG


def _agent():
    return SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)


@pytest.mark.parametrize(
    ("agent_cfg", "expected"),
    [
        ({}, DEFAULT_CONFIG["compression"]["threshold_tokens"]),  # config-load failure → shipped default
        ({"compression": {"threshold_tokens": None}}, None),  # explicit null → ratio-only opt-out
    ],
)
def test_threshold_tokens_default_and_null_opt_out(agent_cfg, expected):
    cs = _parse_compression_config(_agent(), agent_cfg)
    assert cs.threshold_tokens == expected
    assert cs.threshold == DEFAULT_CONFIG["compression"]["threshold"]


def test_platform_override_wins_per_key_without_changing_other_platforms():
    cfg = {
        "compression": {
            "threshold": 0.50,
            "threshold_tokens": 200_000,
            "model_thresholds": {"m": 0.25},
            "platform_overrides": {
                "telegram": {"threshold": 0.80, "threshold_tokens": 60_000},
            },
        },
    }

    telegram = _parse_compression_config(_agent(), cfg, platform="telegram")
    cli = _parse_compression_config(_agent(), cfg, platform="cli")

    assert (telegram.threshold, telegram.threshold_tokens) == (0.80, 60_000)
    assert telegram.model_thresholds == {}
    assert (cli.threshold, cli.threshold_tokens) == (0.50, 200_000)
    assert cli.model_thresholds == {"m": 0.25}
