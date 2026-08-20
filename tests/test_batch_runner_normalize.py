"""Tests for batch_runner tool-stat normalization — deterministic field order.

The per-record ``tool_stats`` / ``tool_error_counts`` dicts must have a stable
key order across processes so the JSONL shards combine into a single Arrow/Parquet
schema when loaded with HuggingFace ``datasets``. Workers are spawned (fresh
``PYTHONHASHSEED`` each), so iterating an unordered ``set`` would emit shards with
different struct field orders and break the combined load.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_runner import (
    ALL_POSSIBLE_TOOLS,
    _normalize_tool_stats,
    _normalize_tool_error_counts,
)


def test_all_possible_tools_is_ordered_and_sorted():
    """ALL_POSSIBLE_TOOLS must be an ordered, sorted sequence (not a bare set)."""
    assert isinstance(ALL_POSSIBLE_TOOLS, tuple)
    assert list(ALL_POSSIBLE_TOOLS) == sorted(ALL_POSSIBLE_TOOLS)
    # Still usable for membership in the trajectory-combine filter.
    assert all(tool in ALL_POSSIBLE_TOOLS for tool in ALL_POSSIBLE_TOOLS)


def test_normalize_tool_stats_key_order_is_deterministic():
    """Output key order is the sorted canonical order, regardless of input order."""
    expected_known = sorted(ALL_POSSIBLE_TOOLS)

    sample = next(iter(ALL_POSSIBLE_TOOLS))
    raw = {sample: {"count": 3, "success": 2, "failure": 1}}

    normalized = _normalize_tool_stats(raw)

    # Every known tool is present and the leading keys follow the canonical order.
    known_keys = [k for k in normalized if k in set(ALL_POSSIBLE_TOOLS)]
    assert known_keys == expected_known
    # The populated tool keeps its values; the rest default to zeros.
    assert normalized[sample] == {"count": 3, "success": 2, "failure": 1}
    other = next(t for t in ALL_POSSIBLE_TOOLS if t != sample)
    assert normalized[other] == {"count": 0, "success": 0, "failure": 0}


def test_normalize_tool_stats_independent_of_input_dict_order():
    """Same set of tools in different input orders yields identical key order."""
    a, b = ALL_POSSIBLE_TOOLS[0], ALL_POSSIBLE_TOOLS[-1]
    forward = _normalize_tool_stats({a: {"count": 1, "success": 1, "failure": 0},
                                     b: {"count": 2, "success": 2, "failure": 0}})
    reverse = _normalize_tool_stats({b: {"count": 2, "success": 2, "failure": 0},
                                     a: {"count": 1, "success": 1, "failure": 0}})
    assert list(forward.keys()) == list(reverse.keys())


def test_normalize_tool_stats_appends_unexpected_tools():
    """Hallucinated tool names are preserved after the canonical block."""
    raw = {"definitely_not_a_real_tool": {"count": 1, "success": 0, "failure": 1}}
    normalized = _normalize_tool_stats(raw)
    assert "definitely_not_a_real_tool" in normalized
    canonical = list(ALL_POSSIBLE_TOOLS)
    assert list(normalized.keys())[: len(canonical)] == canonical


def test_normalize_tool_error_counts_key_order_is_deterministic():
    """Error-count normalization shares the same deterministic ordering."""
    sample = next(iter(ALL_POSSIBLE_TOOLS))
    normalized = _normalize_tool_error_counts({sample: 4})
    known_keys = [k for k in normalized if k in set(ALL_POSSIBLE_TOOLS)]
    assert known_keys == sorted(ALL_POSSIBLE_TOOLS)
    assert normalized[sample] == 4
