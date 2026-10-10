"""Regression coverage for shared OAuth device-code polling."""

from __future__ import annotations

import httpx
import pytest


class _SequenceClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._responses = iter(payloads)

    def post(self, url: str, **_kwargs: object) -> httpx.Response:
        payload = next(self._responses)
        status_code = 200 if "access_token" in payload else 400
        return httpx.Response(
            status_code,
            json=payload,
            request=httpx.Request("POST", url),
        )


@pytest.mark.parametrize(
    ("provider", "poll_interval", "expected_sleep"),
    [("nous", 5, 6), ("xai", 5, 10), ("xai", 30, 30)],
)
def test_device_code_slow_down_adds_five_seconds_for_shared_provider_paths(
    monkeypatch: pytest.MonkeyPatch, provider: str, poll_interval: int, expected_sleep: int,
) -> None:
    """RFC 8628 requires slow_down to add five seconds to future polls."""
    import hermes_cli.auth_device_flow as device_flow

    sleeps: list[int] = []
    monkeypatch.setattr(device_flow.time, "sleep", sleeps.append)
    client = _SequenceClient([
        {"error": "slow_down"},
        {"access_token": "access", "refresh_token": "refresh"},
    ])

    if provider == "nous":
        from hermes_cli.auth import _poll_for_token

        _poll_for_token(
            client=client,  # type: ignore[arg-type]
            portal_base_url="https://portal.example.test",
            client_id="hermes-cli",
            device_code="device-code",
            expires_in=60,
            poll_interval=poll_interval,
        )
    else:
        from hermes_cli.auth_xai import _xai_oauth_poll_device_token

        _xai_oauth_poll_device_token(
            client=client,  # type: ignore[arg-type]
            token_endpoint="https://auth.x.ai/oauth2/token",
            device_code="device-code",
            expires_in=60,
            poll_interval=poll_interval,
        )

    assert sleeps == [expected_sleep]
