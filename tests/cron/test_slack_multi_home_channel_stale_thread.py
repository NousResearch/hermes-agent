"""Regression test: multi-value SLACK_HOME_CHANNEL must not defeat
_origin_thread_is_stale's membership check.

Root cause: _origin_thread_is_stale compared origin['chat_id'] against the
RAW SLACK_HOME_CHANNEL value with plain string equality. When that env var
holds a comma-separated list of channel ids (a supported multi-channel
operator setup), the raw CSV string can never equal any single channel id,
so the heuristic always returned False -- causing deliver=slack:<home-chan>
cron jobs to wrongly inherit the origin thread instead of posting flat.
"""

from cron.scheduler_delivery import _origin_thread_is_stale


def test_stale_thread_detected_with_multi_channel_home(monkeypatch):
    monkeypatch.setenv(
        "SLACK_HOME_CHANNEL",
        "C0B4ERS7V3N,C0ASPB877T6,G0102DHM1J9,C05LXFKPLTH",
    )
    origin = {
        "platform": "slack",
        "chat_id": "C0ASPB877T6",
        "thread_id": "1790621436.519279",
    }
    assert _origin_thread_is_stale(origin) is True


def test_non_home_chat_thread_is_not_stale(monkeypatch):
    monkeypatch.setenv(
        "SLACK_HOME_CHANNEL",
        "C0B4ERS7V3N,C0ASPB877T6,G0102DHM1J9,C05LXFKPLTH",
    )
    origin = {
        "platform": "slack",
        "chat_id": "C9999999999",
        "thread_id": "123.456",
    }
    assert _origin_thread_is_stale(origin) is False


def test_no_thread_id_is_not_stale(monkeypatch):
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C0ASPB877T6")
    origin = {"platform": "slack", "chat_id": "C0ASPB877T6", "thread_id": None}
    assert _origin_thread_is_stale(origin) is False


def test_single_value_home_channel_still_works(monkeypatch):
    """Backward-compat: single-channel configs behaved correctly before
    this fix and must continue to."""
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C0ASPB877T6")
    origin = {
        "platform": "slack",
        "chat_id": "C0ASPB877T6",
        "thread_id": "1790621436.519279",
    }
    assert _origin_thread_is_stale(origin) is True
