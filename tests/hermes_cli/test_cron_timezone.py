"""Tests for the dashboard cron timezone helper (`_cron_timezone_name`).

The cron timeline re-derives occurrence series client-side, but must match the
*configured Hermes timezone* the scheduler evaluates recurrence in (croniter
from `hermes_time.now()`), not the browser's own zone. The
`GET /api/cron/timezone` endpoint exposes that IANA name; its resolution lives
in `_cron_timezone_name` (pure — tested directly, matching the codebase's
preference for testing pure helpers as data rather than through the auth gate).
"""

from zoneinfo import ZoneInfo

from hermes_cli.web_routers.cron import _cron_timezone_name


def test_cron_timezone_name_returns_iana(monkeypatch):
    """A configured timezone surfaces as its IANA name for the timeline."""
    monkeypatch.setattr(
        "hermes_time.get_timezone", lambda: ZoneInfo("Europe/Amsterdam")
    )
    assert _cron_timezone_name() == "Europe/Amsterdam"


def test_cron_timezone_name_none_when_unset(monkeypatch):
    """No configured timezone (server-local) -> None so the client falls back
    to the browser's local zone."""
    monkeypatch.setattr("hermes_time.get_timezone", lambda: None)
    assert _cron_timezone_name() is None
