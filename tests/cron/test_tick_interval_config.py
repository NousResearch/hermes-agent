"""``cron.tick_interval_seconds`` drives the built-in ticker, and the interval the ticker ran
with is what ``hermes cron status`` judges heartbeat staleness against."""
import threading
import time
from unittest.mock import patch

import pytest


def _write_tick_interval(home, raw):
    (home / "config.yaml").write_text(f"cron:\n  tick_interval_seconds: {raw}\n", encoding="utf-8")


def test_configured_interval_reaches_ticker_heartbeat_and_status():
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler, resolve_tick_interval
    from hermes_cli.cron import _ticker_age_is_fresh
    from hermes_constants import get_hermes_home

    configured = jobs.TICKER_INTERVAL_SECONDS * 5
    _write_tick_interval(get_hermes_home(), configured)

    assert resolve_tick_interval() == configured

    # Started like the gateway and Desktop start it: no explicit interval.
    stop = threading.Event()
    with patch("cron.scheduler.tick", lambda **_kw: None):
        ticker = threading.Thread(target=InProcessCronScheduler().start, args=(stop,), daemon=True)
        ticker.start()
        deadline = time.monotonic() + 10
        while jobs.get_ticker_heartbeat_age() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        stop.set()
        ticker.join(timeout=10)

    assert jobs.get_ticker_interval_seconds() == configured
    # A gap longer than several default intervals but shorter than one configured interval is a
    # ticker sleeping as configured, not a dead one.
    gap = jobs.TICKER_INTERVAL_SECONDS * 4
    assert gap < configured
    assert _ticker_age_is_fresh(gap)
    assert not _ticker_age_is_fresh(configured * 4)


@pytest.mark.parametrize("raw", ["0", "-30", "soon", "null"])
def test_unusable_interval_never_spins_or_disables_the_ticker(raw):
    import cron.jobs as jobs
    from cron.scheduler_provider import _MIN_TICK_INTERVAL_SECONDS, resolve_tick_interval
    from hermes_constants import get_hermes_home

    _write_tick_interval(get_hermes_home(), raw)

    interval = resolve_tick_interval()
    assert interval >= _MIN_TICK_INTERVAL_SECONDS
    assert interval in (_MIN_TICK_INTERVAL_SECONDS, jobs.TICKER_INTERVAL_SECONDS)
