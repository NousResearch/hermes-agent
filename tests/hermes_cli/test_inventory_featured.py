"""Regression tests for the featured_models shortlist in hermes_cli.inventory.

The shortlist exists to keep multi-lab routing aggregators (OpenRouter) from
flooding the desktop picker's default-visible set. A user-defined provider's
``models:`` list is an explicit allow-list: every configured model must stay
default-visible, so such rows (and every non-aggregator row) carry an empty
``featured_models`` and the frontend falls back to top-N behaviour instead.
"""

from hermes_cli.inventory import _apply_featured

# Mixed org prefixes in the exact shape a user-defined provider lists them.
_USER_DEFINED_MODELS = [
    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Pro",
    "ZhipuAI/GLM-5.2",
]


def test_user_defined_row_gets_no_featured_shortlist():
    """A mixed-prefix user-defined provider must not be misread as a multi-lab
    aggregator: ranking its explicit allow-list against models.dev release
    dates hides models the user configured by hand (newest flagships lose the
    "top 5 per lab" race when undated)."""
    row = {
        "slug": "modelscope",
        "name": "modelscope",
        "is_current": True,
        "is_user_defined": True,
        "models": list(_USER_DEFINED_MODELS),
        "total_models": len(_USER_DEFINED_MODELS),
    }

    _apply_featured([row])

    assert row["featured_models"] == []


def test_builtin_non_aggregator_row_gets_no_featured_shortlist():
    """A first-party provider (no aggregator flag) keeps top-N behaviour —
    the shortlist is an aggregator-only affordance."""
    row = {
        "slug": "anthropic",
        "name": "Anthropic",
        "is_current": False,
        "is_user_defined": False,
        "models": ["claude-sonnet-5", "claude-opus-5"],
        "total_models": 2,
    }

    _apply_featured([row])

    assert row["featured_models"] == []


def test_builtin_routing_aggregator_keeps_featured_shortlist():
    """The aggregator affordance survives: OpenRouter's row still carries a
    non-empty shortlist so the default-visible set stays curated."""
    row = {
        "slug": "openrouter",
        "name": "OpenRouter",
        "is_current": False,
        "is_user_defined": False,
        "models": [f"lab{i % 3}/model-{i}" for i in range(12)],
        "total_models": 12,
    }

    _apply_featured([row])

    assert row["featured_models"]
    # Newest-first per lab, at most _FEATURED_PER_LAB per lab, order preserved.
    assert set(row["featured_models"]) <= set(row["models"])
