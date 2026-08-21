"""hermes_time zone cache: bounded TTL, not whole-process-lifetime (#88220).

The cache used to live forever once resolved, so a long-running process kept
the zone it happened to boot with even after ``timezone:`` changed in
config.yaml — the divergence behind the 8-hour cron shift. It now re-checks
the configured name at most once per ``_TZ_CACHE_TTL_SECONDS``, and only
rebuilds the ``ZoneInfo`` when the name actually changed.
"""

from __future__ import annotations

import pytest

import hermes_time


@pytest.fixture(autouse=True)
def _isolate_timezone_cache():
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def test_get_timezone_reuses_cache_within_ttl(monkeypatch):
    calls = []

    def fake_resolve():
        calls.append(1)
        return "Asia/Shanghai"

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", fake_resolve)
    fake_now = [1000.0]
    monkeypatch.setattr(hermes_time.time, "monotonic", lambda: fake_now[0])

    first = hermes_time.get_timezone()
    fake_now[0] += hermes_time._TZ_CACHE_TTL_SECONDS - 1
    second = hermes_time.get_timezone()

    assert first is second
    assert len(calls) == 1


def test_get_timezone_re_resolves_after_ttl_expires(monkeypatch):
    calls = []

    def fake_resolve():
        calls.append(1)
        return "Asia/Shanghai"

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", fake_resolve)
    fake_now = [1000.0]
    monkeypatch.setattr(hermes_time.time, "monotonic", lambda: fake_now[0])

    hermes_time.get_timezone()
    fake_now[0] += hermes_time._TZ_CACHE_TTL_SECONDS + 1
    hermes_time.get_timezone()

    assert len(calls) == 2


def test_get_timezone_rebuilds_zoneinfo_only_when_name_changes(monkeypatch):
    names = ["Asia/Shanghai", "Asia/Shanghai", "America/New_York"]

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", lambda: names.pop(0))
    fake_now = [1000.0]
    monkeypatch.setattr(hermes_time.time, "monotonic", lambda: fake_now[0])

    first = hermes_time.get_timezone()
    fake_now[0] += hermes_time._TZ_CACHE_TTL_SECONDS + 1
    second = hermes_time.get_timezone()
    fake_now[0] += hermes_time._TZ_CACHE_TTL_SECONDS + 1
    third = hermes_time.get_timezone()

    assert first is second  # same name re-resolved -> same ZoneInfo object
    assert str(first) == "Asia/Shanghai"
    assert third is not second
    assert str(third) == "America/New_York"


def test_get_timezone_name_reflects_ttl_refresh(monkeypatch):
    names = ["Asia/Shanghai", "America/New_York"]

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", lambda: names.pop(0))
    fake_now = [1000.0]
    monkeypatch.setattr(hermes_time.time, "monotonic", lambda: fake_now[0])

    assert hermes_time.get_timezone_name() == "Asia/Shanghai"
    fake_now[0] += hermes_time._TZ_CACHE_TTL_SECONDS + 1
    assert hermes_time.get_timezone_name() == "America/New_York"


def test_reset_cache_forces_immediate_re_resolution(monkeypatch):
    calls = []

    def fake_resolve():
        calls.append(1)
        return "Asia/Shanghai"

    monkeypatch.setattr(hermes_time, "_resolve_timezone_name", fake_resolve)
    fake_now = [1000.0]
    monkeypatch.setattr(hermes_time.time, "monotonic", lambda: fake_now[0])

    hermes_time.get_timezone()
    hermes_time.reset_cache()
    hermes_time.get_timezone()

    assert len(calls) == 2
