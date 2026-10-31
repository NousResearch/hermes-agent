"""Free models float to the front of the /model picker list (``_picker_order`` free-first)."""

from types import SimpleNamespace

from agent.plan_limits import is_free
from hermes_cli.auth_model_picker import _picker_order
from hermes_cli.model_picker_usage import SECTION_OTHER, SECTION_RECOMMENDED


_PRICING = {
    "a-free": {"prompt": "0", "completion": "0"},
    "b-paid": {"prompt": "0.001", "completion": "0.002"},
    "c-free": {"prompt": "0", "completion": "0"},
    "d-paid": {"prompt": "0.003", "completion": "0.004"},
    "e-paid": {"prompt": "0.005", "completion": "0.006"},
}


def _free_ids(model_ids, pricing):
    """The exact free-set the caller builds in ``_prompt_model_selection``."""
    return {mid for mid in model_ids if is_free(mid, pricing)}


def test_non_window_free_first_stable_partition_cursor_on_current():
    model_ids = ["b-paid", "a-free", "d-paid", "c-free", "e-paid"]
    ordered, sections, start = _picker_order(
        model_ids, "d-paid", None, free_ids=_free_ids(model_ids, _PRICING))
    # Free floats to the front (relative order kept), then the rest the same way.
    # current_model keeps its historical first slot inside the non-free group.
    assert ordered == ["a-free", "c-free", "d-paid", "b-paid", "e-paid"]
    assert sections == {}
    assert start == ordered.index("d-paid") == 2


def test_free_ids_none_or_empty_keeps_historical_ordering():
    model_ids = ["b-paid", "a-free", "d-paid", "c-free"]
    expected = ["b-paid", "a-free", "d-paid", "c-free"]  # current first, then catalog order
    for kwargs in ({}, {"free_ids": None}, {"free_ids": set()}):
        ordered, sections, start = _picker_order(model_ids, "b-paid", None, **kwargs)
        assert ordered == expected
        assert sections == {}
        assert start == 0


def test_window_mode_free_first_only_inside_other_models():
    window_usage = SimpleNamespace(featured_model_ids=("b-paid", "a-free"))
    model_ids = ["a-free", "b-paid", "e-paid", "c-free", "d-paid"]
    ordered, sections, start = _picker_order(
        model_ids, "d-paid", window_usage, free_ids=_free_ids(model_ids, _PRICING))
    head_len = 2  # featured picks in plan order: b-paid, a-free
    # Recommended head untouched (plan order — even though a-free is free).
    assert ordered[:head_len] == ["b-paid", "a-free"]
    # Other models: free (c-free) first, the rest in catalog order.
    assert ordered[head_len:] == ["c-free", "e-paid", "d-paid"]
    # Heading indexes unchanged: {0, len(head)} — no remapping needed.
    assert sections == {0: SECTION_RECOMMENDED, head_len: SECTION_OTHER}
    assert start == ordered.index("d-paid") == 4


def test_current_model_that_is_free_lands_first_with_cursor_zero():
    model_ids = ["b-paid", "a-free", "c-free"]
    ordered, sections, start = _picker_order(
        model_ids, "a-free", None, free_ids=_free_ids(model_ids, _PRICING))
    assert ordered == ["a-free", "c-free", "b-paid"]
    assert start == 0


def test_missing_pricing_rows_and_no_pricing_never_crash_or_reorder():
    # A model key missing from pricing is not free (unknown price is not a promise of free).
    pricing = {"b-paid": {"prompt": "1", "completion": "1"},
               "a-free": {"prompt": "0", "completion": "0"}}
    model_ids = ["u-unknown", "b-paid", "a-free"]
    free = _free_ids(model_ids, pricing)
    assert free == {"a-free"}  # u-unknown: no pricing row → not free
    ordered, _sections, start = _picker_order(model_ids, "b-paid", None, free_ids=free)
    assert ordered == ["a-free", "b-paid", "u-unknown"]
    assert start == 1
    # pricing=None → no free info for ordinary ids → ordering unchanged.
    model_ids2 = ["b-paid", "x-m", "y-m"]
    assert _free_ids(model_ids2, None) == set()
    ordered2, _s2, start2 = _picker_order(model_ids2, "b-paid", None, free_ids=_free_ids(model_ids2, None))
    assert ordered2 == model_ids2 and start2 == 0
