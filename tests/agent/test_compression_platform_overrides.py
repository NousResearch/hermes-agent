"""``compression.platform_overrides.<platform>`` merges over the global section, per key.

The trigger is read once at agent construction, so a surface with no block must resolve to
exactly the global behaviour, a partial block must inherit the keys it omits, and the loaded
config must come back untouched.
"""

import copy
from types import SimpleNamespace

import pytest

from agent.agent_init import _parse_compression_config


def _agent():
    return SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)


def _cfg(overrides=None, **compression):
    section = dict(compression)
    if overrides is not None:
        section["platform_overrides"] = overrides
    return {"compression": section}


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("telegram", 0.60),  # block for this surface wins
        ("local", 0.50),  # a surface with no block falls back to the global value
        (None, 0.50),  # no surface at all (config-load-failure callers)
    ],
)
def test_block_wins_and_absent_falls_back(platform, expected):
    cfg = _cfg({"telegram": {"threshold": 0.60}}, threshold=0.50)
    assert _parse_compression_config(_agent(), cfg, platform).threshold == expected


def test_partial_block_inherits_omitted_keys():
    cfg = _cfg({"telegram": {"threshold_tokens": 400000}}, threshold=0.50, threshold_tokens=800000)
    cs = _parse_compression_config(_agent(), cfg, "telegram")
    assert cs.threshold == 0.50  # inherited from the global section
    assert cs.threshold_tokens == 400000  # overridden for this surface


def test_explicit_null_in_block_stays_the_ratio_only_opt_out():
    cfg = _cfg({"telegram": {"threshold_tokens": None}}, threshold=0.50, threshold_tokens=800000)
    assert _parse_compression_config(_agent(), cfg, "telegram").threshold_tokens is None


def test_surface_key_matched_case_insensitively():
    cfg = _cfg({"Telegram": {"threshold": 0.60}}, threshold=0.50)
    assert _parse_compression_config(_agent(), cfg, "telegram").threshold == 0.60


def test_platform_enum_member_is_accepted():
    from gateway.config import Platform

    cfg = _cfg({"telegram": {"threshold": 0.60}}, threshold=0.50)
    assert _parse_compression_config(_agent(), cfg, Platform.TELEGRAM).threshold == 0.60


@pytest.mark.parametrize("block", [0.60, "0.6", [], None])
def test_non_mapping_block_is_ignored(block):
    cfg = _cfg({"telegram": block}, threshold=0.50)
    assert _parse_compression_config(_agent(), cfg, "telegram").threshold == 0.50


def test_loaded_config_is_not_mutated():
    cfg = _cfg({"telegram": {"threshold": 0.60}}, threshold=0.50)
    snapshot = copy.deepcopy(cfg)
    _parse_compression_config(_agent(), cfg, "telegram")
    assert cfg == snapshot
