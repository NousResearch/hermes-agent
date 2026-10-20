"""The shared RFC 8628 poll loop backs off slow_down by the mandated 5s.

``_poll_device_token_generic`` used to grow the poll interval by only 1s on
``slow_down``. A server that answers ``slow_down`` is already rate-limiting the
client, so a +1s back-off keeps it above the server's limit and every subsequent
poll draws ``slow_down`` again (or a fatal 429). RFC 8628 §3.5 requires the
interval to increase by 5s "for this and all subsequent requests"; these tests
pin that contract for both flows that share the loop (Nous and xAI).
"""

import httpx
import pytest

from hermes_cli import auth_device_flow


def _response(status, payload):
    request = httpx.Request("POST", "https://portal.example/api/oauth/token")
    return httpx.Response(status, json=payload, request=request)


def _poll(scripted_errors, *, poll_interval, monkeypatch):
    """Run the shared loop over a scripted error sequence ending in success;
    return the (real) sleep durations it chose."""
    steps = [_response(400, {"error": code}) for code in scripted_errors]
    steps.append(_response(200, {"access_token": "token"}))

    def _post():
        return steps.pop(0)

    sleeps = []
    monkeypatch.setattr(auth_device_flow.time, "sleep", sleeps.append)
    token = auth_device_flow._poll_device_token_generic(
        _post, expires_in=60, poll_interval=poll_interval,
        validate_success=lambda payload: None,
        on_non_json_error=lambda response: RuntimeError("non-JSON error"),
        on_error=lambda response, error_payload: RuntimeError(str(error_payload.get("error"))),
        on_timeout=lambda: TimeoutError("timed out"))
    assert token == {"access_token": "token"}
    return sleeps


@pytest.mark.parametrize(("poll_interval", "expected"), [(1, [6, 6]), (5, [10, 10])])
def test_slow_down_grows_the_interval_by_five_seconds_and_holds_it(
        monkeypatch, poll_interval, expected):
    """slow_down must add 5s — not 1s — and the grown cadence applies to every
    following request, so a rate-limited client actually falls below the limit."""
    assert _poll(["slow_down", "authorization_pending"], poll_interval=poll_interval,
                 monkeypatch=monkeypatch) == expected


def test_consecutive_slow_downs_keep_adding_five_seconds(monkeypatch):
    """Each slow_down adds another 5s on top of the grown interval."""
    assert _poll(["slow_down", "slow_down"], poll_interval=1,
                 monkeypatch=monkeypatch) == [6, 11]


def test_slow_down_interval_is_capped_at_thirty_seconds(monkeypatch):
    """A server-directed interval near the cap grows to exactly 30s, never beyond."""
    assert _poll(["slow_down"], poll_interval=28,
                 monkeypatch=monkeypatch) == [30]


def test_authorization_pending_keeps_the_ungrown_interval(monkeypatch):
    """Only slow_down grows the cadence; plain authorization_pending polls at the
    server-directed interval."""
    assert _poll(["authorization_pending", "authorization_pending"], poll_interval=3,
                 monkeypatch=monkeypatch) == [3, 3]
