from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent import account_usage


class _RateLimited(RuntimeError):
    def __init__(self, retry_after=None, *, header=None):
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.response = SimpleNamespace(headers={"Retry-After": header} if header else {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (120, 120.0),
        ("45", 45.0),
        (timedelta(seconds=30), 30.0),
    ],
)
def test_retry_after_seconds_supports_numeric_and_timedelta(value, expected):
    assert account_usage.retry_after_seconds(_RateLimited(value)) == expected


def test_retry_after_seconds_supports_http_date():
    now = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)
    exc = _RateLimited(header="Wed, 02 Jan 2030 03:06:00 GMT")

    assert account_usage.retry_after_seconds(exc, now=now) == 120.0


def test_fetch_outcome_preserves_provider_retry_after(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_fetch_anthropic_account_usage",
        lambda: (_ for _ in ()).throw(_RateLimited(timedelta(minutes=10))),
    )

    outcome = account_usage.fetch_account_usage_outcome("anthropic")

    assert outcome.snapshot is None
    assert outcome.retry_after_seconds == 600.0
    assert outcome.failed is True
    assert account_usage.fetch_account_usage("anthropic") is None
