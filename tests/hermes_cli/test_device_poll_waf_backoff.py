"""Regression tests: edge/WAF non-JSON error responses must not abort an in-flight
device-code login.

Vercel fronts the Nous Portal and answers rate-limited clients with a text/plain
403 (``x-vercel-mitigated: deny``) or 429 — no JSON body, so the response can never
carry ``authorization_pending``/``slow_down``. Before the fix, the generic
device-token poll loop hit ``response.raise_for_status()`` on the first such
response and killed a login the user might still be approving in the browser.

The loop now treats non-JSON 403/408/429/5xx as transient: it backs off
(honoring ``Retry-After``, doubling capped at 60s) and keeps polling until the
device code expires. Non-JSON statuses outside that set still abort, and JSON
OAuth errors keep their caller-specific error contract.
"""

import httpx
import pytest

from hermes_cli import auth_device_flow as adf

_REQ = httpx.Request("POST", "https://portal.example/api/oauth/token")


def _ok(payload=None):
    return httpx.Response(200, request=_REQ, json=payload or {"access_token": "tok"})


def _non_json(status, headers=None):
    return httpx.Response(status, request=_REQ, headers=headers or {}, text="edge mitigation")


def _post_returning(*responses):
    seq = iter(responses)

    def post():
        return next(seq)

    return post


def _poll(post, *, expires_in=600, poll_interval=5):
    return adf._poll_device_token_generic(
        post, expires_in=expires_in, poll_interval=poll_interval,
        validate_success=lambda payload: None,
        on_non_json_error=lambda response: RuntimeError("non-JSON error response"),
        on_error=lambda response, payload: RuntimeError(f"oauth:{payload.get('error')}"),
        on_timeout=lambda: TimeoutError("device code expired"))


def test_waf_statuses_back_off_and_keep_polling(monkeypatch):
    sleeps = []
    monkeypatch.setattr(adf.time, "sleep", lambda s: sleeps.append(s))
    post = _post_returning(
        _non_json(403, headers={"x-vercel-mitigated": "deny"}),
        _non_json(429),
        _non_json(503),
        _ok({"access_token": "late-token"}))

    result = _poll(post)

    assert result == {"access_token": "late-token"}
    assert sleeps == [10, 20, 40]  # doubling from interval 5, floor 5, cap 60


def test_retry_after_header_is_honored(monkeypatch):
    sleeps = []
    monkeypatch.setattr(adf.time, "sleep", lambda s: sleeps.append(s))
    post = _post_returning(
        _non_json(429, headers={"retry-after": "7"}),
        _ok())

    _poll(post)

    assert sleeps == [7]


def test_backoff_doubles_but_caps_at_60(monkeypatch):
    sleeps = []
    monkeypatch.setattr(adf.time, "sleep", lambda s: sleeps.append(s))
    post = _post_returning(*([_non_json(429)] * 6), _ok())

    _poll(post)

    assert sleeps == [10, 20, 40, 60, 60, 60]


def test_non_json_status_outside_waf_set_still_aborts(monkeypatch):
    monkeypatch.setattr(adf.time, "sleep", lambda s: None)

    with pytest.raises(httpx.HTTPStatusError):
        _poll(_post_returning(_non_json(400)))


def test_json_oauth_error_contract_unchanged(monkeypatch):
    monkeypatch.setattr(adf.time, "sleep", lambda s: None)
    denied = httpx.Response(400, request=_REQ, json={"error": "access_denied"})

    with pytest.raises(RuntimeError, match="oauth:access_denied"):
        _poll(_post_returning(denied))


def test_deadline_still_ends_the_login(monkeypatch):
    # Fake clock: first reading starts the loop, the second is past the deadline,
    # so a persistently mitigated endpoint ends with the caller's timeout error.
    readings = iter([1000.0, 1000.0 + 10 * 365 * 24 * 3600])
    monkeypatch.setattr(adf.time, "monotonic", lambda: next(readings))
    monkeypatch.setattr(adf.time, "sleep", lambda s: None)

    with pytest.raises(TimeoutError, match="device code expired"):
        _poll(lambda: _non_json(403), expires_in=5)
