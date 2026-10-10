"""#121254: RFC 8628 3.5 — each ``slow_down`` answer grows the poll interval by 5 s
for this and all subsequent requests (the shared device-code loop grew it by 1 s)."""

from __future__ import annotations

import time

from hermes_cli.auth_device_flow import _poll_device_token_generic


class _Resp:
    status_code = 400
    headers: dict = {}

    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _run(script: list, poll_interval: int = 1) -> tuple[dict, list[float]]:
    sleeps: list[float] = []
    calls = iter(script)

    def post():
        return next(calls)

    real_sleep = time.sleep
    time.sleep = sleeps.append
    try:
        result = _poll_device_token_generic(
            post,
            expires_in=60,
            poll_interval=poll_interval,
            validate_success=lambda payload: None,
            on_non_json_error=lambda _r: RuntimeError("non-JSON error response"),
            on_error=lambda _r, payload: RuntimeError(str(payload.get("error"))),
            on_timeout=lambda: TimeoutError("device code expired"),
        )
    finally:
        time.sleep = real_sleep
    return result, sleeps


def test_success_polls_without_sleeping():
    result, sleeps = _run([_Resp(200, {"access_token": "tok"})])
    assert sleeps == []
    assert result["access_token"] == "tok"


def test_pending_then_slow_down_waits_interval_then_interval_plus_five():
    result, sleeps = _run([
        _Resp(400, {"error": "authorization_pending"}),
        _Resp(400, {"error": "slow_down"}),
        _Resp(200, {"access_token": "tok"}),
    ])
    assert sleeps == [1.0, 6.0], (
        f"RFC 8628 3.5 requires the interval to grow by 5s after slow_down: {sleeps}")
    assert result["access_token"] == "tok"


def test_slow_down_growth_stays_at_the_cap():
    _, sleeps = _run([
        _Resp(400, {"error": "slow_down"}),
        _Resp(400, {"error": "slow_down"}),
        _Resp(200, {"access_token": "tok"}),
    ], poll_interval=28)
    assert sleeps == [30.0, 30.0], f"interval must cap at 30s: {sleeps}"
