"""``_parse_compression_config`` promises "Defaults here MUST match DEFAULT_CONFIG" — the config-load
failure path hands it ``{}`` and every key falls back inline. Regression for the one key that did not:
``threshold_tokens`` fell to None (no cap) while ``threshold`` kept 0.50, so a broken config.yaml
silently restored the pre-#115986 500K trigger on 1M-window models."""

from types import SimpleNamespace

import pytest

from agent.agent_init import _parse_compression_config
from hermes_cli.config import DEFAULT_CONFIG


def _agent():
    return SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({}, DEFAULT_CONFIG["compression"]["threshold_tokens"]),  # absent → shipped default
        ({"threshold_tokens": None}, None),  # explicit null → ratio-only opt-out
    ],
)
def test_absent_threshold_tokens_falls_back_like_every_other_key(section, expected):
    cs = _parse_compression_config(_agent(), {"compression": section} if section else {})
    assert cs.threshold_tokens == expected
    assert cs.threshold == DEFAULT_CONFIG["compression"]["threshold"]
