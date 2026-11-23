"""A passive (observed) group row never welds with an addressed user turn in the restore-time repair.

The merged row keeps ``observed=True``, so gateway replay would fold the addressed text into the passive
observation string. Rows of the same class still merge, and a row written before the flag existed counts as
addressed.
"""
from session_rules.turns import _merge_consecutive_users

_TAIL = {"role": "assistant", "content": "ok"}


def test_passive_row_and_addressed_turn_stay_apart():
    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": "Passive observation", "observed": True},
        {"role": "user", "content": "Addressed tool question"}, dict(_TAIL)])
    assert repairs == 0
    assert [row["content"] for row in merged[:2]] == ["Passive observation", "Addressed tool question"]
    assert merged[0]["observed"] is True and "observed" not in merged[1]


def test_rows_of_the_same_class_still_merge():
    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": "one", "observed": True},
        {"role": "user", "content": "two", "observed": True}, dict(_TAIL)])
    assert repairs == 1 and merged[0]["content"] == "one\n\ntwo" and merged[0]["observed"] is True
    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": "three"}, {"role": "user", "content": "four", "observed": False}, dict(_TAIL)])
    assert repairs == 1 and merged[0]["content"] == "three\n\nfour"
