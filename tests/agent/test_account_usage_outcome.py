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


def test_openrouter_credits_success_key_rate_limit_preserves_partial_snapshot(monkeypatch):
    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def get(self, url, *, headers):
            request = account_usage.httpx.Request("GET", url)
            if url.endswith("/credits"):
                return account_usage.httpx.Response(
                    200,
                    json={"data": {"total_credits": 10, "total_usage": 3}},
                    request=request,
                )
            return account_usage.httpx.Response(
                429,
                headers={"Retry-After": "120"},
                request=request,
            )

    monkeypatch.setattr(
        account_usage,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "or-key",
            "base_url": "https://openrouter.example/api/v1",
        },
    )
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kwargs: _Client())

    outcome = account_usage.fetch_account_usage_outcome("openrouter")

    assert outcome.failed is True
    assert outcome.retry_after_seconds == 120.0
    assert outcome.snapshot is not None
    assert outcome.snapshot.details == ("Credits balance: $7.00",)
    assert outcome.snapshot.windows == ()


@pytest.mark.asyncio
async def test_openrouter_rate_limit_outcome_is_used_by_controller_before_deadline(
    tmp_path,
):
    from gateway.account_usage_presence import (
        AccountUsagePresenceCapabilities,
        AccountUsagePresenceController,
    )
    from gateway.config import AccountUsagePresenceConfig

    class _Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = _Clock()
    calls = []

    def fetch(provider):
        calls.append(provider)
        if len(calls) == 1:
            return account_usage.AccountUsageFetchOutcome(
                snapshot=account_usage.AccountUsageSnapshot(
                    provider="openrouter",
                    source="credits_api",
                    fetched_at=datetime.now(timezone.utc),
                    windows=(
                        account_usage.AccountUsageWindow(
                            label="API key quota", used_percent=25
                        ),
                    ),
                    details=("Credits balance: $7.00",),
                ),
                retry_after_seconds=120.0,
                failed=True,
            )
        return account_usage.AccountUsageSnapshot(
            provider="openrouter",
            source="credits_api",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                account_usage.AccountUsageWindow(label="API key quota", used_percent=30),
            ),
        )

    class _Adapter:
        def __init__(self):
            self.account_usage_presence_capabilities = AccountUsagePresenceCapabilities(
                activity=True
            )
            self.applied = []

        async def apply_account_usage_presence(self, payload, baseline):
            self.applied.append(payload)
            return True

    adapter = _Adapter()
    controller = AccountUsagePresenceController(
        AccountUsagePresenceConfig.from_dict(
            {
                "enabled": True,
                "provider": "openrouter",
                "platforms": ["discord"],
            }
        ),
        lambda: {"discord": adapter},
        fetcher=fetch,
        state_path=tmp_path / "journal.json",
        monotonic=clock,
    )

    await controller.refresh_once()
    clock.now = 119
    await controller.refresh_once()

    assert calls == ["openrouter"]
    assert len(adapter.applied) == 2
    assert adapter.applied[-1].cached is True
