"""The model picker shows subscription usage before the wall, without ever waiting on a usage API.

Usage windows come from the provider's API, so ``model.options`` must read them from cache and
refresh in the background: a slow or hung usage endpoint can't be allowed to stall the picker.
"""

import threading
import time
from datetime import datetime, timedelta, timezone, UTC

from agent import account_usage
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_cache import cached_account_usage
from hermes_cli.inventory import build_model_options_payload, load_picker_context


def _snapshot(used: float) -> AccountUsageSnapshot:
    reset = datetime.now(UTC) + timedelta(hours=2)
    return AccountUsageSnapshot(
        provider="openrouter", source="test", fetched_at=datetime.now(UTC),
        windows=(AccountUsageWindow(label="Current session", used_percent=used, reset_at=reset),))


def test_picker_reads_cached_usage_and_refreshes_without_waiting(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    release, called = threading.Event(), threading.Event()
    answers = iter([_snapshot(92.0)])

    def slow_usage_api(_base_url, _api_key):
        called.set()
        release.wait(10)
        return next(answers)

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openrouter", slow_usage_api)
    ctx = load_picker_context()

    started = time.monotonic()
    first = next(r for r in build_model_options_payload(ctx)["providers"] if r["slug"] == "openrouter")
    assert time.monotonic() - started < 5, "the picker waited on the usage API"
    assert "usage" not in first, "nothing cached yet, so nothing to show"
    assert called.wait(5), "opening the picker must start a background usage refresh"

    release.set()
    deadline = time.monotonic() + 5
    while cached_account_usage("openrouter") is None and time.monotonic() < deadline:
        time.sleep(0.05)

    row = next(r for r in build_model_options_payload(ctx)["providers"] if r["slug"] == "openrouter")
    (window,) = row["usage"]["windows"]
    assert window["label"] == "Current session" and window["used_percent"] == 92.0
    assert datetime.fromisoformat(window["resets_at"]) > datetime.now(UTC)
