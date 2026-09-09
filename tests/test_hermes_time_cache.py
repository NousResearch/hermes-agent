"""Regression: ``hermes_time.get_timezone()`` must not publish a timezone under a
cache identity that changed while it was doing config I/O.

``get_timezone()`` captures the cache identity, resolves the zone name outside the
lock (slow config read), then publishes ``(name, tz)`` keyed to that identity. If
a profile / ``HERMES_TIMEZONE`` switch lands during the resolve — a real scenario
now that the tier/pin system gives gateway and dashboard genuinely concurrent
profile contexts — the pre-switch identity's slot gets poisoned with another
profile's zone (or that zone is returned for the wrong profile). The fix re-checks
the identity after resolving and retries instead of publishing a mismatched pair.
"""

import os

import pytest
from zoneinfo import ZoneInfo

import hermes_time


@pytest.fixture(autouse=True)
def _clean_tz_cache(monkeypatch):
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()
    os.environ.pop("HERMES_TIMEZONE", None)


def test_identity_shift_during_resolve_is_not_published_stale(monkeypatch):
    """First resolve flips the identity (a profile switch) and hands back the OLD
    profile's zone. That result must be discarded and retried, never published
    under the pre-switch identity."""
    pre_switch_identity = hermes_time._timezone_cache_identity()
    assert pre_switch_identity[0] == "config"  # no HERMES_TIMEZONE yet

    real_resolve = hermes_time._resolve_timezone_name
    calls = []

    def flipping_resolve():
        calls.append(1)
        if len(calls) > 5:
            raise AssertionError("get_timezone() did not converge - retry loop spun")
        if len(calls) == 1:
            # A concurrent profile switch lands mid-resolution...
            monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
            # ...but this call already read the OLD profile's config.
            return "Asia/Tokyo"
        return real_resolve()

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", flipping_resolve)

    tz = hermes_time.get_timezone()

    # Retried against the identity that actually applied at publish time.
    assert isinstance(tz, ZoneInfo)
    assert str(tz) == "America/New_York"

    # The pre-switch identity was never populated, and the stale zone was never
    # cached under any identity.
    assert pre_switch_identity not in hermes_time._tz_cache
    assert all(k[0] != "config" for k in hermes_time._tz_cache)
    assert all(
        name != "Asia/Tokyo" for name, _tz in hermes_time._tz_cache.values()
    ), hermes_time._tz_cache


def test_stable_identity_still_caches_once_and_hits(monkeypatch):
    """No shift: the resolved zone is cached under the right identity and the
    second call is served from cache (same object, resolver not re-run)."""
    monkeypatch.setenv("HERMES_TIMEZONE", "Europe/London")
    identity = hermes_time._timezone_cache_identity()

    calls = []
    real_resolve = hermes_time._resolve_timezone_name

    def counting_resolve():
        calls.append(1)
        return real_resolve()

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", counting_resolve)

    first = hermes_time.get_timezone()
    second = hermes_time.get_timezone()

    assert str(first) == "Europe/London"
    assert first is second
    assert len(calls) == 1, "second call should be a cache hit, not a re-resolve"
    assert hermes_time._tz_cache[identity] == ("Europe/London", first)
