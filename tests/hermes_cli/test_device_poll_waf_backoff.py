"""Regression tests: edge/WAF non-JSON error responses must not abort an in-flight
device-code login.

Vercel fronts the Nous Portal and answers rate-limited clients with a text/plain
403 (``x-vercel-mitigated: deny``) or 429 — no JSON body, so the response can never
carry ``authorization_pending``/``slow_down``. Before the fix, the generic
device-token poll loop hit ``response.raise_for_status()`` on the first such
response and killed a login the user might still be approving in the browser.

The loop now treats non-JSON 408/429/5xx and an ``x-vercel-mitigated`` 403 as
transient: it backs off (honoring ``Retry-After``, capped at 60s and at the
device-code deadline) and keeps polling until the code expires, then returns to
the server's polling interval once the Portal answers with OAuth JSON again.
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


def _fake_clock(monkeypatch):
    clock = [1000.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(adf.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adf.time, "sleep", sleep)
    return sleeps


def test_recovers_after_edge_blocked_responses(monkeypatch):
    sleeps = _fake_clock(monkeypatch)
    pending = httpx.Response(400, request=_REQ, json={"error": "authorization_pending"})
    post = _post_returning(
        _non_json(403, headers={"x-vercel-mitigated": "deny"}),
        _non_json(429, headers={"retry-after": "3600"}),
        _non_json(503),
        pending, pending,
        _ok({"access_token": "late-token"}))

    result = _poll(post, poll_interval=5)

    assert result == {"access_token": "late-token"}
    assert all(s <= 60 for s in sleeps)
    assert sleeps[-2:] == [5, 5]  # back to the server interval once OAuth JSON returns


def test_persistent_block_ends_at_deadline_without_oversleeping(monkeypatch):
    sleeps = _fake_clock(monkeypatch)

    with pytest.raises(TimeoutError, match="device code expired"):
        _poll(lambda: _non_json(429, headers={"retry-after": "3600"}), expires_in=300)

    assert sum(sleeps) <= 300
