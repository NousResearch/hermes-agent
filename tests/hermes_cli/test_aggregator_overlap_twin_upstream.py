"""Aggregator overlap strip vs a user row that IS the same upstream as a built-in aggregator (#115623)."""

import pytest

from hermes_cli import inventory


def _builtin_openrouter_row(models: list) -> dict:
    return {
        "slug": "openrouter",
        "name": "OpenRouter",
        "is_current": False,
        "is_user_defined": False,
        "models": list(models),
        "total_models": len(models),
        "source": "builtin",
    }


def _user_row(slug: str, api_url: str, models: list) -> dict:
    return {
        "slug": slug,
        "name": slug.split(":", 1)[-1],
        "is_current": False,
        "is_user_defined": True,
        "api_url": api_url,
        "models": list(models),
        "total_models": len(models),
        "source": "user-config",
    }


@pytest.mark.parametrize("twin_slug", ["custom:openrouter", "custom:my-or-relay"])
def test_built_in_row_keeps_models_when_user_row_is_same_upstream(twin_slug):
    # The twin's catalog is a superset of the built-in row's; counting it as "another provider
    # serves these ids" emptied the built-in row beside a live twin row. The twin is detected by
    # its custom:<aggregator> slug or, under any other name, by its upstream URL.
    rows = [
        _builtin_openrouter_row(["anthropic/claude-fable-5", "openai/gpt-6"]),
        _user_row(twin_slug, "https://openrouter.ai/api/v1", [
            "anthropic/claude-fable-5",
            "openai/gpt-6",
            "qwen/qwen-4-max",
        ]),
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["anthropic/claude-fable-5", "openai/gpt-6"]
    assert rows[0]["total_models"] == 2


def test_genuinely_different_upstream_still_strips_overlaps():
    # A user proxy that is NOT the same upstream must still re-route picks away from aggregators
    # (the dedup's original intent).
    rows = [
        _builtin_openrouter_row(["openai/gpt-6", "deepseek/deepseek-v4.1-flash"]),
        _user_row("custom:proxy", "https://my-proxy.example.com/v1", ["openai/gpt-6"]),
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["deepseek/deepseek-v4.1-flash"]
    assert rows[0]["total_models"] == 1
