"""Tests for the ranked fuzzy scorer used by the searchable curses pickers."""
from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _fuzzy_score,
    _handle_active_search_key,
    _is_boundary,
    _token_score,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343


def test_fuzzy_score_matches_subsequence():
    assert _fuzzy_score("gpt-4o", "g4o") is not None
    assert _fuzzy_score("gpt-4o", "4o") is not None
    assert _fuzzy_score("gpt-4o", "o4g") is None
    assert _fuzzy_score("gpt-4o", "xyz") is None


def test_scorer_matches_typescript_reference():
    """Score parity with ui-tui/web fuzzy.ts. These exact values are produced
    by the TS fuzzyScoreMulti for the same inputs (verified via a cross-language
    harness); keep the Python port byte-identical so all three surfaces rank
    consistently. If you change the scoring constants, update the TS copies too.
    """
    cases = {
        ("gpt-4o", "g4o"): 15.94,
        ("gpt-4o", "gpt"): 28.94,
        ("claude-sonnet-4", "sonnet"): 33.85,
        ("claude-sonnet-4", "clad snnt"): 30.70,
        ("GptO", "gpto"): 57.96,  # camelCase boundary on the original-case 'O'
    }
    for (label, query), expected in cases.items():
        score = _fuzzy_score(label, query)
        assert score is not None
        assert round(score, 2) == expected, f"{label!r}/{query!r}: {score} != {expected}"


def test_is_boundary_camelcase_and_separators():
    assert _is_boundary("gpt-4o", 0) is True       # start
    assert _is_boundary("gpt-4o", 4) is True        # after '-'
    assert _is_boundary("gpt-4o", 2) is False       # mid-word
    assert _is_boundary("GptO", 3) is True          # lower->upper transition


def test_token_score_takes_orig_and_lower():
    # Exact match (lower == token) earns the +20 bonus over a prefix.
    exact = _token_score("sonnet", "sonnet", "sonnet")
    prefix = _token_score("sonnet-x", "sonnet-x", "sonnet")
    assert exact is not None and prefix is not None
    assert exact > prefix


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


def test_high_byte_keys_ignored():
    # Bytes 128-255 must NOT append Latin-1 mojibake to the query.
    search = _SearchState(active=True, query="ab")
    handled, _, changed = _handle_active_search_key(_FakeCurses, 200, search)
    assert (handled, changed) == (False, False)
    assert search.query == "ab"


def test_fuzzy_score_empty_query_is_zero():
    assert _fuzzy_score("anything", "") == 0
    assert _fuzzy_score("anything", "   ") == 0


def test_fuzzy_score_prefix_beats_scattered():
    prefix = _fuzzy_score("gpt-4o-mini", "gpt")
    scattered = _fuzzy_score("a-g-p-t", "gpt")
    assert prefix is not None and scattered is not None
    assert prefix > scattered


def test_fuzzy_score_exact_and_shorter_rank_higher():
    exact = _fuzzy_score("sonnet", "sonnet")
    longer = _fuzzy_score("sonnet-extended", "sonnet")
    assert exact is not None and longer is not None
    # Same prefix match, but the shorter id wins on the length tiebreak.
    assert exact > longer


def test_filter_indices_ranks_best_first():
    models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku", "o1-preview"]

    # g4o matches both gpt-4o variants; the shorter exact-ish one ranks first.
    ranked = _filter_indices(models, "g4o")
    assert [models[i] for i in ranked] == ["gpt-4o", "gpt-4o-mini"]

    # son4 surfaces the sonnet model.
    assert [models[i] for i in _filter_indices(models, "son4")] == ["claude-sonnet-4"]

    # Multi-token AND.
    assert [models[i] for i in _filter_indices(models, "clad snnt")] == ["claude-sonnet-4"]

    # No match drops everything.
    assert _filter_indices(models, "zzz") == []


def test_filter_indices_blank_query_preserves_order():
    models = ["b", "a", "c"]
    assert _filter_indices(models, "") == [0, 1, 2]
    assert _filter_indices(models, "   ") == [0, 1, 2]


def test_filter_indices_stable_for_equal_scores():
    # Identical labels score identically; original order is the tiebreak.
    items = ["ab", "ab", "ab"]
    assert _filter_indices(items, "ab") == [0, 1, 2]
