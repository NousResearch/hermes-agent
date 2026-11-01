"""Invariant tests for ``cron/scripts/classify_items.py`` score parsing.

Regression for the silent-empty-monitor bug: the prompt asked only for "the JSON
array of scores … (one object per item, same order)", so the aux model answers with
bare ``{"score": N}`` objects (or an id-bearing variant) and the parser found no
integer ``"index"``. Every batch then scored empty and the monitor went permanently
quiet while exiting 0 — the worst failure mode for a cron watchdog, since silence is
indistinguishable from "nothing matched".
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "cron" / "scripts" / "classify_items.py"


@pytest.fixture(scope="module")
def classify_items():
    spec = importlib.util.spec_from_file_location("classify_items_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_requests_the_index_key(classify_items):
    """The parser requires ``index``; the prompt must therefore ask for it by name."""
    prompt = classify_items._build_prompt([{"subject": "x"}], "criteria")
    assert '"index"' in prompt
    assert "MUST" in prompt


def test_parses_explicit_index_objects(classify_items):
    """The documented contract: one object per item carrying its bracketed index."""
    content = '[{"index": 0, "score": 7, "reason": "a"}, {"index": 1, "score": 2, "reason": "b"}]'
    assert classify_items._parse_scores(content, 2) == {
        0: {"index": 0, "score": 7, "reason": "a"},
        1: {"index": 1, "score": 2, "reason": "b"},
    }


@pytest.mark.parametrize(
    "content",
    [
        # bare score objects, no index key at all (the observed production shape)
        '[{"score": 9, "reason": "a"}, {"score": 1, "reason": "b"}, {"score": 4, "reason": "c"}]',
        # the model's id-as-ordinal variant, which is NOT a valid index
        '[{"id": 0, "score": 9}, {"id": 1, "score": 1}, {"id": 2, "score": 4}]',
    ],
)
def test_falls_back_to_positional_mapping(classify_items, content):
    """A full-length array without a usable ``index`` maps positionally instead of dropping the batch."""
    out = classify_items._parse_scores(content, 3)
    assert sorted(out) == [0, 1, 2], "a full-length array must never parse to empty"
    assert [out[i]["score"] for i in range(3)] == [9, 1, 4]


def test_partial_array_stays_empty(classify_items):
    """The fallback must not invent data: a short array is still unparseable."""
    assert classify_items._parse_scores('[{"score": 9}]', 3) == {}


def test_out_of_range_index_is_rejected(classify_items):
    assert classify_items._parse_scores('[{"index": 5, "score": 9}]', 2) == {}


def test_bool_index_is_not_an_int(classify_items):
    """``True`` is an ``int`` in Python; it must not be accepted as an index."""
    assert classify_items._parse_scores('[{"index": true, "score": 9}]', 2) == {}


def test_garbage_input_is_empty_not_an_exception(classify_items):
    assert classify_items._parse_scores("not json at all", 2) == {}
    assert classify_items._parse_scores("", 2) == {}
    assert classify_items._parse_scores('{"index": 0}', 2) == {}
