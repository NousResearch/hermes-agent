"""Tests for shared config-driven context-length resolution helpers."""

from agent.context_length_config import (
    coerce_positive_context_length,
    resolve_config_context_length,
    resolve_display_context_length,
)


def test_coerce_positive_context_length_accepts_int_and_numeric_string():
    assert coerce_positive_context_length(123) == 123
    assert coerce_positive_context_length("456") == 456


def test_coerce_positive_context_length_rejects_invalid_values():
    assert coerce_positive_context_length(None) is None
    assert coerce_positive_context_length(False) is None
    assert coerce_positive_context_length("256K") is None
    assert coerce_positive_context_length(0) is None


def test_resolve_config_context_length_prefers_custom_provider_match_by_base_url():
    cfg = {
        "model": {"context_length": 200000},
        "custom_providers": [
            {
                "name": "infini-ai",
                "base_url": "https://cloud.infini-ai.com/maas/coding/v1",
                "models": {"glm-5": {"context_length": 262144}},
            }
        ],
    }

    value = resolve_config_context_length(
        model="glm-5",
        provider="custom",
        base_url="https://cloud.infini-ai.com/maas/coding/v1",
        agent_cfg=cfg,
    )

    assert value == 262144


def test_resolve_config_context_length_falls_back_to_global_model_context():
    cfg = {"model": {"context_length": 200000}}

    value = resolve_config_context_length(
        model="glm-5",
        provider="custom",
        base_url="https://cloud.infini-ai.com/maas/coding/v1",
        agent_cfg=cfg,
    )

    assert value == 200000


def test_resolve_display_context_length_prefers_config_override_over_model_info():
    class _Info:
        context_window = 128000

    value = resolve_display_context_length(
        model="glm-5",
        provider="custom:infini-ai",
        base_url="https://cloud.infini-ai.com/maas/coding/v1",
        api_key="",
        model_info=_Info(),
        agent_cfg={
            "custom_providers": [
                {
                    "name": "infini-ai",
                    "base_url": "https://cloud.infini-ai.com/maas/coding/v1",
                    "models": {"glm-5": {"context_length": 200000}},
                }
            ]
        },
    )

    assert value == 200000
