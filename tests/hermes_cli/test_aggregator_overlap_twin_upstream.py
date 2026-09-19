"""Aggregator overlap strip vs a user row that IS the same upstream as a built-in aggregator."""

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


def _custom_openrouter_twin(models: list) -> dict:
    return {
        "slug": "custom:openrouter",
        "name": "openrouter",
        "is_current": False,
        "is_user_defined": True,
        "api_url": "https://openrouter.ai/api/v1",
        "models": list(models),
        "total_models": len(models),
        "source": "user-config",
    }


def test_built_in_row_keeps_models_when_twin_is_same_upstream():
    # The twin's catalog is a superset of the built-in row's; counting it as "another provider
    # serves these ids" emptied the built-in row beside a live custom:openrouter row.
    rows = [
        _builtin_openrouter_row(["anthropic/claude-fable-5", "openai/gpt-6"]),
        _custom_openrouter_twin([
            "anthropic/claude-fable-5",
            "openai/gpt-6",
            "qwen/qwen-4-max",
        ]),
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["anthropic/claude-fable-5", "openai/gpt-6"]
    assert rows[0]["total_models"] == 2


def test_same_upstream_twin_detected_by_url_when_slug_differs():
    # The twin registered under a different name still resolves to the built-in upstream by URL.
    rows = [
        _builtin_openrouter_row(["openai/gpt-6"]),
        {
            "slug": "custom:my-or-relay",
            "name": "my-or-relay",
            "is_current": False,
            "is_user_defined": True,
            "api_url": "https://openrouter.ai/api/v1",
            "models": ["openai/gpt-6"],
            "total_models": 1,
            "source": "user-config",
        },
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["openai/gpt-6"]


def test_genuinely_different_upstream_still_strips_overlaps():
    # A user proxy that is NOT the same upstream must still re-route picks away from aggregators
    # (the dedup's original intent).
    rows = [
        _builtin_openrouter_row(["openai/gpt-6", "deepseek/deepseek-v4.1-flash"]),
        {
            "slug": "custom:proxy",
            "name": "proxy",
            "is_current": False,
            "is_user_defined": True,
            "api_url": "https://my-proxy.example.com/v1",
            "models": ["openai/gpt-6"],
            "total_models": 1,
            "source": "user-config",
        },
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["deepseek/deepseek-v4.1-flash"]
    assert rows[0]["total_models"] == 1


def test_no_builtin_aggregator_rows_leaves_dedup_untouched():
    # Without a built-in aggregator in the payload there is no twin to exempt.
    rows = [
        {
            "slug": "custom:proxy-a",
            "name": "proxy-a",
            "is_current": False,
            "is_user_defined": True,
            "api_url": "https://a.example.com/v1",
            "models": ["m1", "m2"],
            "total_models": 2,
        },
        {
            "slug": "custom:proxy-b",
            "name": "proxy-b",
            "is_current": False,
            "is_user_defined": True,
            "api_url": "https://b.example.com/v1",
            "models": ["m1"],
            "total_models": 1,
        },
    ]
    inventory._strip_aggregator_overlaps(rows)
    assert rows[0]["models"] == ["m1", "m2"]
    assert rows[1]["models"] == ["m1"]
