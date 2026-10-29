"""Regression tests for Anthropic's OAuth usage percentage units."""

import httpx
import pytest

from agent import account_usage


@pytest.mark.parametrize(
    ("utilization", "expected_percent"),
    [
        (1.0, 1.0),
        (0.5, 0.5),
        (7.0, 7.0),
    ],
)
def test_anthropic_utilization_is_already_percentage(
    monkeypatch: pytest.MonkeyPatch,
    utilization: float,
    expected_percent: float,
) -> None:
    """Anthropic returns utilization on a 0..100 scale, including values <= 1."""
    payload = {"seven_day": {"utilization": utilization}}
    real_client = httpx.Client

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(handle),
            **kwargs,
        ),
    )

    snapshot = account_usage._fetch_anthropic_account_usage(
        api_key="sk-ant-oat01-test-token"
    )

    assert snapshot is not None
    assert snapshot.windows[0].used_percent == expected_percent
