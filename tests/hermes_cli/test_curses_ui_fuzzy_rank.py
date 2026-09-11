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










