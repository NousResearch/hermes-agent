"""``_float_free_models_first`` unit tests: free models float to the front of each
provider row's ``models`` list (stable partition, ``featured_models`` untouched)."""

from hermes_cli.inventory import _float_free_models_first


def _pricing(*, free_ids=(), paid_ids=()):
    pricing = {}
    for mid in free_ids:
        pricing[mid] = {"input": "free", "output": "free", "free": True}
    for mid in paid_ids:
        pricing[mid] = {"input": "$1", "output": "$2", "free": False}
    return pricing


def test_free_models_float_first_stable():
    models = ["paid-a", "free-a", "paid-b", "free-b"]
    row = {
        "slug": "openrouter",
        "models": list(models),
        "featured_models": ["paid-b", "free-a"],
        "pricing": _pricing(free_ids=("free-a", "free-b"), paid_ids=("paid-a", "paid-b")),
    }

    _float_free_models_first([row])

    # free-a/free-b keep their relative order at the front, paid-a/paid-b after.
    assert row["models"] == ["free-a", "free-b", "paid-a", "paid-b"]
    # featured_models is an independent curated list: membership and order unchanged.
    assert row["featured_models"] == ["paid-b", "free-a"]


def test_untouched_when_models_empty_or_pricing_missing():
    empty = {"models": [], "featured_models": ["x"], "pricing": {"x": {"free": True}}}
    no_pricing = {"models": ["a:free", "b"], "featured_models": ["a:free"]}
    none_pricing = {"models": ["a:free", "b"], "pricing": None}

    _float_free_models_first([empty, no_pricing, none_pricing])

    assert empty["models"] == []
    assert no_pricing["models"] == ["a:free", "b"]  # unchanged: no pricing dict at all
    assert none_pricing["models"] == ["a:free", "b"]


def test_free_suffix_without_pricing_entry_still_floats():
    # pricing dict exists but has no entry for the :free-suffixed model -> still free.
    row = {
        "models": ["paid-only", "openrouter/qux:free", "other"],
        "featured_models": ["paid-only"],
        "pricing": {"paid-only": {"free": False}, "other": {"free": False}},
    }

    _float_free_models_first([row])

    assert row["models"] == ["openrouter/qux:free", "paid-only", "other"]
    assert row["featured_models"] == ["paid-only"]


def test_multiple_rows_reorder_independently():
    row_a = {
        "models": ["p1", "f1", "p2"],
        "featured_models": ["p1"],
        "pricing": _pricing(free_ids=("f1",), paid_ids=("p1", "p2")),
    }
    row_b = {
        "models": ["q1", "q2:free", "q3"],
        "featured_models": ["q2:free", "q3"],
        "pricing": _pricing(paid_ids=("q1", "q3")),
    }
    row_c = {
        "models": ["z1", "z2"],
        "featured_models": [],
        "pricing": _pricing(paid_ids=("z1", "z2")),
    }

    _float_free_models_first([row_a, row_b, row_c])

    assert row_a["models"] == ["f1", "p1", "p2"]
    assert row_b["models"] == ["q2:free", "q1", "q3"]
    assert row_c["models"] == ["z1", "z2"]  # no free models: order unchanged
    assert row_a["featured_models"] == ["p1"]
    assert row_b["featured_models"] == ["q2:free", "q3"]
    assert row_c["featured_models"] == []
