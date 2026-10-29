"""Free models float to the front of each picker provider's models list (stable)."""

from hermes_cli.model_switch_providers import _float_free_models_first


def test_openrouter_row_floats_free_suffix_models_to_the_front_stably():
    row = {
        "slug": "openrouter",
        "models": ["paid/a", "recursive/flint-v1.0:free", "paid/b", "nvidia/nemotron-3.5-super-vl:free"],
        "total_models": 4,
    }
    _float_free_models_first([row])
    assert row["models"] == [
        "recursive/flint-v1.0:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "paid/a",
        "paid/b",
    ]
    assert row["total_models"] == 4


def test_empty_models_list_is_untouched():
    row = {"slug": "openrouter", "models": [], "total_models": 0}
    _float_free_models_first([row])
    assert row["models"] == []
    assert row["total_models"] == 0


def test_pricing_free_flag_without_free_suffix_also_floats():
    row = {
        "slug": "nous",
        "models": ["paid/x", "nous/freebie", "paid/y"],
        "pricing": {
            "nous/freebie": {"input": "free", "output": "free", "free": True},
            "paid/x": {"input": 1.0, "output": 2.0, "free": False},
        },
    }
    _float_free_models_first([row])
    assert row["models"] == ["nous/freebie", "paid/x", "paid/y"]


def test_multiple_provider_rows_are_reordered_independently():
    rows = [
        {
            "slug": "openrouter",
            "models": ["paid/a", "x:free", "paid/b"],
            "total_models": 3,
        },
        {
            "slug": "nous",
            "models": ["paid/p", "nous/freebie", "paid/q"],
            "pricing": {"nous/freebie": {"free": True}},
            "total_models": 3,
        },
        {"slug": "anthropic", "models": ["claude-1", "claude-2"], "total_models": 2},
    ]
    _float_free_models_first(rows)
    assert rows[0]["models"] == ["x:free", "paid/a", "paid/b"]
    assert rows[0]["total_models"] == 3
    assert rows[1]["models"] == ["nous/freebie", "paid/p", "paid/q"]
    assert rows[1]["total_models"] == 3
    # No free models → order preserved exactly.
    assert rows[2]["models"] == ["claude-1", "claude-2"]
