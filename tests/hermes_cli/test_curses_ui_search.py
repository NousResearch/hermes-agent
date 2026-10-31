from hermes_cli.curses_ui import (
    _SearchState,
    _filter_indices,
    _fuzzy_score,
    _handle_active_search_key,
    _move_filtered_cursor,
    _reconcile_cursor,
)


class _FakeCurses:
    KEY_BACKSPACE = 263
    KEY_DOWN = 258
    KEY_ENTER = 343




def test_reconcile_cursor_moves_to_first_visible_match():
    assert _reconcile_cursor([2, 4], 0) == (2, 0)
    assert _reconcile_cursor([2, 4], 4) == (4, 1)




def test_active_search_consumes_query_editing_and_confirm_keys():
    search = _SearchState(active=True, query="op")

    assert _handle_active_search_key(_FakeCurses, ord("u"), search) == (True, False, True)
    assert search.query == "opu"

    assert _handle_active_search_key(_FakeCurses, _FakeCurses.KEY_ENTER, search) == (
        True,
        True,
        False,
    )


def test_fuzzy_score_folds_separators_like_the_shared_ts_scorer():
    """Parity guard for apps/shared/src/fuzzy.test.ts: the same pairs, the same verdicts."""
    assert _fuzzy_score("gpt-4o", "gpt.4o") is not None
    assert _fuzzy_score("claude-3-opus", "claude_3") is not None
    assert _fuzzy_score("qwen3.8-flash", "qwen3-8") is not None


def test_fuzzy_score_finds_vendor_prefixed_ids():
    """A `/` between owner and model folds like the other separators."""
    assert _fuzzy_score("Qwen/Qwen3.8-Flash", "Qwen-3.8") is not None
    assert _fuzzy_score("deepseek-ai/DeepSeek-V4-Flash", "deepseek-v4-flash") is not None
    assert _fuzzy_score("Qwen/Qwen3.8-Flash", "llama") is None
