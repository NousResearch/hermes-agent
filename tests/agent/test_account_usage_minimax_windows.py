"""Regression tests for the MiniMax M-Plan usage fetcher.

Live payload captured 2026-11-03 from https://api.minimax.io/v1/token_plan/remains,
authenticated with the SAME OAuth bearer used for inference (scope
``group_id profile model.completion``) - no admin API key required:

    {"base_resp": {"status_code": 0, "status_msg": "success"},
     "model_remains": [
       {"model_name": "general",
        "current_interval_remaining_percent": 84, "end_time": 1793750400000,
        "current_weekly_remaining_percent": 95, "weekly_end_time": 1794182400000,
        "current_interval_usage_count": 0, "current_weekly_usage_count": 0},
       {"model_name": "video",
        "current_interval_remaining_percent": 100, "end_time": 1793750400000,
        "current_weekly_remaining_percent": 100, "weekly_end_time": 1794182400000,
        "current_interval_total_count": 3, "current_weekly_total_count": 21}]}

Two things worth pinning: the endpoint reports the PERCENT REMAINING (so the used
percent is ``100 - remaining``), and the window bounds are epoch MILLISECONDS, not
seconds (the generic ``_parse_dt`` treats a bare number as seconds).
"""

from agent import account_usage


def _payload():
    return {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "model_remains": [
            {
                "model_name": "general",
                "current_interval_remaining_percent": 84, "end_time": 1793750400000,
                "current_weekly_remaining_percent": 95, "weekly_end_time": 1794182400000,
                "current_interval_usage_count": 0, "current_weekly_usage_count": 0,
            },
            {
                "model_name": "video",
                "current_interval_remaining_percent": 100, "end_time": 1793750400000,
                "current_weekly_remaining_percent": 100, "weekly_end_time": 1794182400000,
                "current_interval_total_count": 3, "current_weekly_total_count": 21,
            },
        ],
    }


def _fetch_with(payload):
    """Call the fetcher with a stubbed transport (works under pytest and standalone)."""
    original = account_usage._get_json
    account_usage._get_json = lambda *args, **kwargs: payload
    try:
        return account_usage._fetch_minimax_account_usage(api_key="test-token")
    finally:
        account_usage._get_json = original


def test_minimax_remaining_percent_becomes_used_percent():
    snap = _fetch_with(_payload())
    assert snap is not None
    by_label = {w.label: w for w in snap.windows}
    assert by_label["5-hour (general)"].used_percent == 16.0, "84% remaining == 16% used"
    assert by_label["Weekly (general)"].used_percent == 5.0, "95% remaining == 5% used"
    assert by_label["5-hour (video)"].used_percent == 0.0
    assert by_label["Weekly (video)"].used_percent == 0.0


def test_minimax_window_bounds_are_read_as_milliseconds():
    snap = _fetch_with(_payload())
    assert snap is not None
    by_label = {w.label: w for w in snap.windows}
    # 1793750400000 ms == 2026-11-04T00:00:00Z
    interval_reset = by_label["5-hour (general)"].reset_at
    weekly_reset = by_label["Weekly (general)"].reset_at
    assert interval_reset is not None
    assert weekly_reset is not None
    assert interval_reset.timestamp() == 1793750400.0, (
        "epoch ms must be divided by 1000 before _parse_dt"
    )
    assert weekly_reset.timestamp() == 1794182400.0


def test_minimax_fetcher_is_registered_for_the_oauth_provider():
    assert account_usage._USAGE_FETCHERS.get("minimax-oauth") is account_usage._fetch_minimax_account_usage


def test_minimax_snapshot_carries_the_raw_payload():
    snap = _fetch_with(_payload())
    assert snap is not None
    assert snap.source == "token_plan_remains"
    assert snap.raw is not None
    assert snap.raw["base_resp"]["status_code"] == 0


def test_minimax_missing_percent_fields_are_skipped():
    payload = {"base_resp": {"status_code": 0}, "model_remains": [{"model_name": "general"}]}
    snap = _fetch_with(payload)
    assert snap is not None
    assert not snap.windows, "a window with no percent must not be fabricated"
