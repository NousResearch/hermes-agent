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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        (0, 0),
        (None, 2),
        ("abc", 2),
        (True, 2),
        (-3, 0),
    ],
)
def test_warn_after_compressions_coercion(raw, expected):
    cs = _parse_compression_config(
        _agent(), {"compression": {"warn_after_compressions": raw}}
    )
    assert cs.warn_after_compressions == expected


def test_warn_after_compressions_defaults_when_unset():
    cs = _parse_compression_config(_agent(), {})
    assert cs.warn_after_compressions == DEFAULT_CONFIG["compression"]["warn_after_compressions"] == 2
