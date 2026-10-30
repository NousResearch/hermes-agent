"""The MiniMax token-plan quota reader (the provider plugin's ``fetch_account_usage`` hook).

The wire is undocumented, so these tests pin the four properties that are easy to get
backwards: HTTP 200 always, in-band ``base_resp.status_code``, percentages that are what
REMAINS, and epoch milliseconds.
"""

import sys
from datetime import UTC, datetime

import pytest

from providers import get_provider_profile


def _plugin_module():
    """The loaded plugin package.

    Bundled model-provider plugins are registered under ``plugins.model_providers.<name>``
    by discovery (the on-disk directory is ``model-providers``), so the module is reached
    through ``sys.modules`` after the profile resolves rather than by a plain import.
    """
    profile = get_provider_profile("minimax")
    return sys.modules[type(profile).__module__]


# Discovery registers the plugin (and its quota submodule) under
# ``plugins.model_providers.minimax``; resolving the profile first is what puts it there.
quota = sys.modules[f"{_plugin_module().__name__}.quota"]


# A live-shaped success payload: the ``general`` (text/coding) bucket on a 5-hour window
# and a 7-day weekly window, plus the ``video`` bucket on its own 24-hour window.
LIVE = {
    "model_remains": [
        {
            "start_time": 1785164400000,
            "end_time": 1785182400000,
            "remains_time": 1877492,
            "current_interval_total_count": 0,
            "current_interval_usage_count": 0,
            "model_name": "general",
            "current_weekly_total_count": 0,
            "current_weekly_usage_count": 0,
            "weekly_start_time": 1785110400000,
            "weekly_end_time": 1785715200000,
            "weekly_remains_time": 534677492,
            "current_interval_status": 1,
            "current_interval_remaining_percent": 99,
            "current_weekly_status": 1,
            "current_weekly_remaining_percent": 99,
        },
        {
            "start_time": 1785110400000,
            "end_time": 1785196800000,
            "remains_time": 16277492,
            "current_interval_total_count": 3,
            "current_interval_usage_count": 0,
            "model_name": "video",
            "current_weekly_total_count": 21,
            "current_weekly_usage_count": 0,
            "weekly_start_time": 1785110400000,
            "weekly_end_time": 1785715200000,
            "weekly_remains_time": 534677492,
            "current_interval_status": 1,
            "current_interval_remaining_percent": 100,
            "current_weekly_status": 1,
            "current_weekly_remaining_percent": 100,
        },
    ],
    "base_resp": {"status_code": 0, "status_msg": "success"},
}


class TestParseQuota:
    def test_reads_the_text_bucket_and_its_weekly_window(self):
        """The text/coding pool is what a coding session spends against; the video pool
        is metered separately and must not stand in for it."""
        windows = quota.parse_quota(LIVE)

        assert [w.label for w in windows] == ["5h limit", "weekly limit"]
        assert [w.short_label for w in windows] == ["5h", "wk"]

    def test_inverts_remaining_percent_into_used(self):
        """The API reports what is LEFT; Hermes counts what was SPENT. Echoing the
        remaining percent here would show a nearly-full plan as nearly-exhausted."""
        windows = quota.parse_quota(LIVE)

        assert windows[0].used_percent == pytest.approx(1.0)
        assert windows[1].used_percent == pytest.approx(1.0)

    def test_remaining_percent_drives_the_used_percent(self):
        payload = {**LIVE, "model_remains": [{**LIVE["model_remains"][0],
                                               "current_interval_remaining_percent": 14,
                                               "current_weekly_remaining_percent": 46}]}

        windows = quota.parse_quota(payload)

        assert windows[0].used_percent == pytest.approx(86.0)
        assert windows[1].used_percent == pytest.approx(54.0)

    def test_reset_comes_from_end_time_in_milliseconds(self):
        """The reset instant is the window's end in epoch MILLISECONDS."""
        windows = quota.parse_quota(LIVE)

        assert windows[0].reset_at == datetime.fromtimestamp(1785182400000 / 1000, tz=UTC)
        assert windows[1].reset_at == datetime.fromtimestamp(1785715200000 / 1000, tz=UTC)

    def test_rejects_in_band_auth_failure(self):
        """Auth failures arrive as HTTP 200 with the real status in the envelope — the
        HTTP status must never be read as success."""
        for status in (1004, 2049):
            payload = {"base_resp": {"status_code": status, "status_msg": "invalid api key"}}

            assert quota.parse_quota(payload) == []

    def test_no_rows_yields_no_windows(self):
        payload = {"model_remains": [], "base_resp": {"status_code": 0, "status_msg": "success"}}

        assert quota.parse_quota(payload) == []

    def test_video_only_payload_has_no_five_hour_window(self):
        """MiniMax meters video on the weekly window alone; showing it a 5-hour limit
        would name a window the plan does not have."""
        payload = {"model_remains": [LIVE["model_remains"][1]],
                   "base_resp": {"status_code": 0, "status_msg": "success"}}

        windows = quota.parse_quota(payload)

        assert [w.label for w in windows] == ["weekly limit"]

    def test_missing_percent_leaves_that_window_out(self):
        payload = {"model_remains": [{k: v for k, v in LIVE["model_remains"][0].items()
                                      if k != "current_interval_remaining_percent"}],
                   "base_resp": {"status_code": 0, "status_msg": "success"}}

        windows = quota.parse_quota(payload)

        assert [w.label for w in windows] == ["weekly limit"]

    def test_out_of_range_percent_is_clamped(self):
        payload = {"model_remains": [{**LIVE["model_remains"][0],
                                       "current_interval_remaining_percent": 140,
                                       "current_weekly_remaining_percent": -5}],
                   "base_resp": {"status_code": 0, "status_msg": "success"}}

        windows = quota.parse_quota(payload)

        assert windows[0].used_percent == pytest.approx(0.0)
        assert windows[1].used_percent == pytest.approx(100.0)


class TestQuotaOrigin:
    def test_strips_the_anthropic_route(self):
        """The plan API answers at the API root, not under the profile's base URL path."""
        assert quota.quota_origin("https://api.minimax.io/anthropic") == "https://api.minimax.io"

    def test_keeps_the_china_host(self):
        assert quota.quota_origin("https://api.minimaxi.com/anthropic") == "https://api.minimaxi.com"

    def test_empty_base_url_has_no_origin(self):
        assert quota.quota_origin("") == ""


class TestProfileHook:
    def test_profile_exposes_the_account_usage_hook(self):
        """``has_account_usage`` detects a provider by this override, so the whole
        generic surface (the picker chip, ``/usage``) lights up from the plugin alone."""
        from agent.account_usage_cache import has_account_usage
        from providers.base import ProviderProfile

        profile = get_provider_profile("minimax")

        assert type(profile).fetch_account_usage is not ProviderProfile.fetch_account_usage
        assert has_account_usage("minimax") is True

    def test_hook_returns_a_snapshot_with_both_windows(self, monkeypatch):
        profile = get_provider_profile("minimax")
        monkeypatch.setattr(_plugin_module(), "fetch_quota",
                            lambda token, origin: quota.parse_quota(LIVE))

        snapshot = profile.fetch_account_usage(api_key="test-key")

        assert snapshot is not None
        assert snapshot.provider == "minimax"
        assert [w.label for w in snapshot.windows] == ["5h limit", "weekly limit"]

    def test_hook_returns_none_without_a_key(self, monkeypatch):
        """No credential means no plan readout — never a fabricated one."""
        profile = get_provider_profile("minimax")

        def _must_not_fetch(token, origin):
            raise AssertionError("must not fetch without a key")

        monkeypatch.setattr(_plugin_module(), "fetch_quota", _must_not_fetch)

        snapshot = profile.fetch_account_usage(api_key="")

        assert snapshot is None or not snapshot.windows
