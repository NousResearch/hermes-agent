"""Unit tests for Telegram-only OpenRouter per-turn cost footer."""

from __future__ import annotations

from gateway.openrouter_cost_footer import (
    build_openrouter_background_review_cost_line,
    build_openrouter_cost_line,
)


def test_openrouter_cost_line_for_telegram_estimated():
    line = build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 2,
            "turn_estimated_cost_usd": 0.012345,
            "turn_cost_status": "estimated",
        },
    )
    assert line == "💸 OpenRouter: ~$0.0123 per request"


def test_openrouter_cost_line_formats_tiny_amounts():
    line = build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": 0.0000123,
            "turn_cost_status": "estimated",
        },
    )
    assert line == "💸 OpenRouter: ~$0.000012 per request"


def test_openrouter_cost_line_skips_non_telegram():
    assert build_openrouter_cost_line(
        platform_key="discord",
        agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": 0.1,
            "turn_cost_status": "estimated",
        },
    ) == ""


def test_openrouter_cost_line_skips_non_openrouter():
    assert build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "openai-codex",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": 0.1,
            "turn_cost_status": "included",
        },
    ) == ""


def test_openrouter_cost_line_detects_custom_provider_by_base_url():
    line = build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "custom",
            "billing_base_url": "https://openrouter.ai/api/v1",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": 0.006789,
            "turn_cost_status": "estimated",
        },
    )
    assert line == "💸 OpenRouter: ~$0.0068 per request"


def test_openrouter_cost_line_detects_custom_openrouter_provider_name():
    line = build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "custom:openrouter",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": 0.006789,
            "turn_cost_status": "estimated",
        },
    )
    assert line == "💸 OpenRouter: ~$0.0068 per request"


def test_openrouter_cost_line_unknown_pricing():
    line = build_openrouter_cost_line(
        platform_key="telegram",
        agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 1,
            "turn_estimated_cost_usd": None,
            "turn_cost_status": "unknown",
        },
    )
    assert line == "💸 OpenRouter: cost n/a per request"


def test_openrouter_background_review_cost_line_adds_main_and_background_costs():
    line = build_openrouter_background_review_cost_line(
        platform_key="telegram",
        main_agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 2,
            "turn_estimated_cost_usd": 0.0316,
            "turn_cost_status": "estimated",
        },
        background_review_result={
            "api_calls": 1,
            "estimated_cost_usd": 0.0184,
            "cost_status": "estimated",
        },
    )

    assert line == "💸 OpenRouter total: ~$0.0500 per request (main ~$0.0316 + bg ~$0.0184)"


def test_openrouter_background_review_cost_line_skips_when_no_background_call():
    assert build_openrouter_background_review_cost_line(
        platform_key="telegram",
        main_agent_result={
            "billing_provider": "openrouter",
            "turn_api_calls": 2,
            "turn_estimated_cost_usd": 0.0316,
            "turn_cost_status": "estimated",
        },
        background_review_result={
            "api_calls": 0,
            "estimated_cost_usd": 0.0,
            "cost_status": "estimated",
        },
    ) == ""
