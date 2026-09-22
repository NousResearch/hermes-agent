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
    ("user_compression", "expected_cap"),
    [
        ({"threshold": 0.85}, None),  # user-authored ratio, no cap written → shipped cap yields (#117915)
        ({"threshold": 0.85, "threshold_tokens": 300_000}, 300_000),  # explicit cap stays absolute
        ({"model_thresholds": {"m": 0.85}}, DEFAULT_CONFIG["compression"]["threshold_tokens"]),  # per-model: resolved in the compressor
    ],
)
def test_user_global_ratio_outranks_the_shipped_cap(user_compression, expected_cap):
    user_cfg = {"compression": user_compression}
    merged = {"compression": {**DEFAULT_CONFIG["compression"], **user_compression}}
    cs = _parse_compression_config(_agent(), merged, user_cfg)
    assert cs.threshold_tokens == expected_cap
    assert cs.threshold_tokens_is_default is ("threshold_tokens" not in user_compression)
