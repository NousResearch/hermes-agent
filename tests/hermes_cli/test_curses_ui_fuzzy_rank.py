"""Tests for the ranked fuzzy scorer used by the searchable curses pickers."""
from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _fuzzy_score,
    _handle_active_search_key,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343




def test_scorer_ranks_prefix_and_boundary_matches_higher():
    """Contiguous / word-boundary matches outrank scattered ones; non-subsequences never match."""
    assert _fuzzy_score("gpt-4o", "gpt") > _fuzzy_score("gpt-4o", "g4o")
    assert _fuzzy_score("claude-sonnet-4", "sonnet") > _fuzzy_score("claude-sonnet-4", "clad snnt")
    assert _fuzzy_score("gpt-4o", "zzz") is None


def test_contiguous_match_outranks_an_earlier_greedy_scatter():
    """``opus`` must rank the Opus row first: the leftmost greedy pass scattered it over
    ``anthropic``/``claude`` and put claude-opus BELOW coder-plus, command-a-plus and sonnet."""
    items = ["qwen/qwen3-coder-plus", "cohere/command-a-plus", "anthropic/claude-sonnet-5",
             "anthropic/claude-opus-5.5"]
    assert _filter_indices(items, "opus")[0] == 3


def test_esc_clears_query_and_signals_changed():
    # Esc during active search clears the filter (restores full list) and
    # signals `changed` so the driver resets scroll/cursor.
    search = _SearchState(active=True, query="gpt")
    handled, confirm, changed = _handle_active_search_key(_FakeCurses, 27, search)
    assert (handled, confirm, changed) == (True, False, True)
    assert search.active is False
    assert search.query == ""

    # Esc with no query: still stops search, but nothing changed.
    search2 = _SearchState(active=True, query="")
    assert _handle_active_search_key(_FakeCurses, 27, search2) == (True, False, False)










