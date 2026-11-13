"""Regression tests for the ``featured_models`` shortlist in hermes_cli.inventory.

The shortlist exists to keep multi-lab *routing aggregators* (OpenRouter) from
flooding the desktop picker's default-visible set. A user-defined provider's
``models:`` list is an explicit allow-list: every configured model must stay
default-visible, so such rows (and every other non-aggregator row) carry an
empty ``featured_models`` and the frontend falls back to top-N behaviour
(#120217).
"""

from hermes_cli.inventory import _apply_featured

# The exact mixed-prefix allow-list from #120217's config.yaml reproduction.
_MODELScope_MODELS = [
    "Qwen/Qwen3.8-Flash-Next",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Pro",
    "ZhipuAI/GLM-4.7-Flash",
    "ZhipuAI/GLM-5.2",
    "stepfun-ai/Step-3.7-Flash",
]


def test_user_defined_row_gets_no_featured_shortlist():
    """A mixed-prefix user-defined provider must not be misread as a multi-lab
    aggregator: ranking its explicit allow-list against models.dev release dates
    hides models the user configured by hand — brand-new flagships lose the
    "top 5 per lab" race whenever the catalog has no/older ``release_date``."""
    row = {
        "slug": "modelscope",
        "name": "modelscope",
        "is_current": True,
        "is_user_defined": True,
        "models": list(_MODELScope_MODELS),
        "total_models": len(_MODELScope_MODELS),
    }

    _apply_featured([row])

    assert row["featured_models"] == []
    assert row["models"] == _MODELScope_MODELS  # never trimmed, only shortlisted


def test_custom_proxy_row_gets_no_featured_shortlist():
    """``custom:*`` slugs are aggregators by :func:`is_routing_aggregator`, but a
    user-defined row wearing one (a hand-written proxy) still owns an explicit
    allow-list — ``is_user_defined`` wins over the slug shape."""
    row = {
        "slug": "custom:lab",
        "name": "Lab",
        "is_current": False,
        "is_user_defined": True,
        "models": ["openai/model-a", "anthropic/model-b", "openai/model-c"],
        "total_models": 3,
    }

    _apply_featured([row])

    assert row["featured_models"] == []


def test_builtin_non_aggregator_row_gets_no_featured_shortlist():
    """A first-party provider keeps top-N behaviour — the shortlist is an
    aggregator-only affordance, per the docstring."""
    row = {
        "slug": "anthropic",
        "name": "Anthropic",
        "is_current": False,
        "is_user_defined": False,
        "models": ["claude-opus-5", "claude-sonnet-5"],
        "total_models": 2,
    }

    _apply_featured([row])

    assert row["featured_models"] == []


def test_builtin_flat_namespace_reseller_gets_no_featured_shortlist():
    """Flat-namespace resellers are first-party catalogs, not routing
    aggregators, so they are not featured either."""
    row = {
        "slug": "opencode-go",
        "name": "OpenCode Go",
        "is_current": False,
        "is_user_defined": False,
        "models": [f"lab{i % 3}/model-{i}" for i in range(12)],
        "total_models": 12,
    }

    _apply_featured([row])

    assert row["featured_models"] == []


def test_builtin_routing_aggregator_keeps_featured_shortlist():
    """The affordance survives: OpenRouter's row still carries a curated
    shortlist so its huge catalog does not flood the default-visible set."""
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
    assert set(row["featured_models"]) <= set(row["models"])
    # Top-_FEATURED_PER_LAB per lab, row order preserved.
    assert len(row["featured_models"]) <= 3 * 5
    order = {m: i for i, m in enumerate(row["models"])}
    assert [order[m] for m in row["featured_models"]] == sorted(
        order[m] for m in row["featured_models"]
    )
