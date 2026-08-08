from __future__ import annotations

from agent.dcp_config import parse_dcp_config, resolve_limit, resolve_model_limit


def test_dcp_config_defaults_match_supported_surface():
    cfg = parse_dcp_config({})

    assert cfg.enabled is True
    assert cfg.prune_notification == "detailed"
    assert cfg.compress.mode == "range"
    assert cfg.compress.permission == "allow"
    assert cfg.compress.max_context_limit == 100000
    assert cfg.compress.min_context_limit == 50000
    assert cfg.deduplication.enabled is True
    assert cfg.purge_errors.enabled is True
    assert cfg.purge_errors.turns == 4


def test_dcp_config_parses_percent_limits_and_model_overrides():
    cfg = parse_dcp_config(
        {
            "compress": {
                "maxContextLimit": "80%",
                "minContextLimit": "40%",
                "modelMaxLimits": {"openai/test-model": "90%"},
                "modelMinLimits": {"test-model": 12345},
            }
        }
    )

    assert resolve_limit(cfg.compress.max_context_limit, 200000) == 160000
    assert resolve_limit(cfg.compress.min_context_limit, 200000) == 80000
    assert resolve_model_limit(
        cfg.compress.model_max_limits,
        provider="openai",
        model="test-model",
        context_length=200000,
        fallback=cfg.compress.max_context_limit,
    ) == 180000
    assert resolve_model_limit(
        cfg.compress.model_min_limits,
        provider="openai",
        model="test-model",
        context_length=200000,
        fallback=cfg.compress.min_context_limit,
    ) == 12345


def test_dcp_config_rejects_invalid_choices_to_defaults():
    cfg = parse_dcp_config(
        {
            "compress": {
                "mode": "bad",
                "permission": "root",
                "nudgeForce": "loud",
            }
        }
    )

    assert cfg.compress.mode == "range"
    assert cfg.compress.permission == "allow"
    assert cfg.compress.nudge_force == "soft"
