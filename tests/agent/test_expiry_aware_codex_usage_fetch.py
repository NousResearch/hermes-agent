from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from agent import account_usage
from agent.credential_pool import CodexAccountUsageWindows, select_expiry_aware_entry


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    account_usage.clear_codex_expiry_aware_usage_cache()
    monkeypatch.setattr(account_usage.time, "time", lambda: 100.0)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, calls, payload=None, error=None, timeout=None):
        self.calls = calls
        self.payload = payload
        self.error = error
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url, headers):
        self.calls.append((url, headers, self.timeout))
        if self.error:
            raise self.error
        return _Response(self.payload)


def _entry(entry_id="one", account_id="acct-one"):
    return SimpleNamespace(
        id=entry_id,
        provider="openai-codex",
        runtime_api_key="test-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        extra={"account_id": account_id},
    )


def _payload(*, primary_used=10, weekly_used=20, weekly_seconds=604800):
    return {
        "rate_limit": {
            "primary_window": {"used_percent": primary_used, "reset_at": 500.0, "limit_window_seconds": 18000},
            "secondary_window": {"used_percent": weekly_used, "reset_at": 10_000.0, "limit_window_seconds": weekly_seconds},
        }
    }


def _patch_http(monkeypatch, calls, payload=None, error=None):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout, **_kwargs: _Client(calls, payload=payload, error=error, timeout=timeout),
    )


def test_fetch_codex_expiry_aware_usage_uses_entry_account_and_bounded_http(monkeypatch):
    calls = []
    _patch_http(monkeypatch, calls, _payload())

    usage = account_usage.fetch_codex_expiry_aware_usage(_entry())

    assert usage is not None
    assert usage.account_id == "acct-one"
    assert usage.short_window_remaining_fraction == 0.9
    assert usage.remaining_fraction == 0.8
    assert calls == [
        (
            "https://chatgpt.com/backend-api/wham/usage",
            {
                "Authorization": "Bearer test-token",
                "Accept": "application/json",
                "User-Agent": "codex-cli",
                "ChatGPT-Account-Id": "acct-one",
            },
            5.0,
        )
    ]


@pytest.mark.parametrize(
    "entry,payload",
    [
        (_entry(account_id=""), _payload()),
        (_entry(), _payload(weekly_seconds=0)),
        (_entry(), _payload(primary_used="invalid")),
    ],
)
def test_fetch_codex_expiry_aware_usage_rejects_unverifiable_data(monkeypatch, entry, payload):
    calls = []
    _patch_http(monkeypatch, calls, payload)

    assert account_usage.fetch_codex_expiry_aware_usage(entry) is None


def test_fetch_codex_expiry_aware_usage_does_not_reuse_other_account_cache(monkeypatch):
    calls = []
    _patch_http(monkeypatch, calls, _payload())
    account_usage.clear_codex_expiry_aware_usage_cache()

    one = account_usage.fetch_codex_expiry_aware_usage(_entry("one", "acct-one"))
    two = account_usage.fetch_codex_expiry_aware_usage(_entry("two", "acct-two"))

    assert one is not None and two is not None
    assert one.account_id == "acct-one"
    assert two.account_id == "acct-two"
    assert len(calls) == 2
    assert calls[1][1]["ChatGPT-Account-Id"] == "acct-two"


def test_fetch_codex_expiry_aware_usage_refetches_after_cache_freshness_limit(monkeypatch):
    calls = []
    clock = {"now": 100.0}
    monkeypatch.setattr(account_usage.time, "time", lambda: clock["now"])
    _patch_http(monkeypatch, calls, _payload())

    first = account_usage.fetch_codex_expiry_aware_usage(_entry())
    clock["now"] = 161.0
    second = account_usage.fetch_codex_expiry_aware_usage(_entry())

    assert first is not None and second is not None
    assert len(calls) == 2


def test_fetch_codex_expiry_aware_usage_refetches_after_clock_moves_back(monkeypatch):
    calls = []
    clock = {"now": 100.0}
    monkeypatch.setattr(account_usage.time, "time", lambda: clock["now"])
    _patch_http(monkeypatch, calls, _payload())

    first = account_usage.fetch_codex_expiry_aware_usage(_entry())
    clock["now"] = 99.0
    second = account_usage.fetch_codex_expiry_aware_usage(_entry())

    assert first is not None and second is not None
    assert len(calls) == 2


def test_fetch_codex_expiry_aware_usage_fails_open_on_timeout(monkeypatch):
    calls = []
    _patch_http(monkeypatch, calls, error=httpx.TimeoutException("timed out"))

    assert account_usage.fetch_codex_expiry_aware_usage(_entry()) is None
    assert calls[0][2] == 5.0


def test_expiry_aware_rejects_exhausted_short_or_weekly_window():
    entry = _entry()
    base = dict(account_id="acct-one", observed_at=1_000.0, reset_at=10_000.0, remaining_fraction=0.8)

    assert select_expiry_aware_entry(
        [entry], lambda _: CodexAccountUsageWindows(**base, short_window_remaining_fraction=0.0), now=1_001.0
    ) is None
    assert select_expiry_aware_entry(
        [entry], lambda _: CodexAccountUsageWindows(**{**base, "remaining_fraction": 0.0}, short_window_remaining_fraction=0.8), now=1_001.0
    ) is None


def test_expiry_aware_ranks_by_explicit_weekly_reset_not_short_window():
    first, second = _entry("first", "acct-first"), _entry("second", "acct-second")
    usage = {
        "first": CodexAccountUsageWindows(
            account_id="acct-first", observed_at=1_000.0, reset_at=9_000.0, remaining_fraction=0.9,
            short_window_remaining_fraction=0.9,
        ),
        "second": CodexAccountUsageWindows(
            account_id="acct-second", observed_at=1_000.0, reset_at=8_000.0, remaining_fraction=0.9,
            short_window_remaining_fraction=0.9,
        ),
    }

    assert select_expiry_aware_entry([first, second], lambda entry: usage[entry.id], now=1_001.0) is second


def test_expiry_aware_rejects_usage_bound_to_a_different_entry_account():
    entry = _entry(account_id="acct-one")
    usage = CodexAccountUsageWindows(
        account_id="acct-other", observed_at=1_000.0, reset_at=10_000.0,
        remaining_fraction=0.8, short_window_remaining_fraction=0.8,
    )

    assert select_expiry_aware_entry([entry], lambda _: usage, now=1_001.0) is None


def test_expiry_aware_rejects_stale_usage_even_if_quota_remains():
    entry = _entry()
    usage = CodexAccountUsageWindows(
        account_id="acct-one", observed_at=1_000.0, reset_at=10_000.0,
        remaining_fraction=0.8, short_window_remaining_fraction=0.8,
    )

    assert select_expiry_aware_entry([entry], lambda _: usage, now=1_301.0) is None
