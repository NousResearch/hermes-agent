"""The shared Nous/xAI device-code poll loop survives transient transport blips.

``_poll_device_token_generic`` (consumed by the CLI Nous login, the dashboard/Desktop Nous
poller, the anon sign-in flow, and the xAI login) used to call ``post()`` with zero exception
handling: a single dropped connection (SSL EOF, ConnectError) aborted the whole device-code
flow, wasting a browser approval the user may have already completed. This mirrors the same
fix already shipped for the Codex device-login poll (``auth_codex._codex_login_post``,
#114610) by reusing its transient-transport classifier, so a network blip is retried a bounded
number of times while a real (non-transport) error still fails immediately.
"""

import ssl

import httpx
import pytest

from hermes_cli import auth_device_flow


_SSL_EOF_MESSAGE = (
    "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)")


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _ScriptedPost:
    """Each call pops the next scripted step: a ``BaseException`` is raised, else returned."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _poll(post, **overrides):
    kwargs = dict(
        expires_in=30, poll_interval=0,
        validate_success=lambda payload: None,
        on_non_json_error=lambda _r: RuntimeError("non-json"),
        on_error=lambda _r, payload: RuntimeError(f"error: {payload}"),
        on_timeout=lambda: TimeoutError("timed out"),
    )
    kwargs.update(overrides)
    return auth_device_flow._poll_device_token_generic(post, **kwargs)


def test_poll_survives_transport_blips_but_not_other_errors(monkeypatch):
    monkeypatch.setattr(auth_device_flow.time, "sleep", lambda *_: None)
    success = _Response(200, {"access_token": "tok"})

    post = _ScriptedPost(
        [ssl.SSLEOFError(8, _SSL_EOF_MESSAGE), httpx.ConnectError("connection reset"), success])
    assert _poll(post) == {"access_token": "tok"}
    assert post.calls == 3

    bug = ValueError("decode bug")
    post = _ScriptedPost([bug, success])
    with pytest.raises(ValueError) as excinfo:
        _poll(post)
    assert excinfo.value is bug
    assert post.calls == 1  # a non-transport exception is a bug, never retried


def test_poll_caps_transient_retries_and_reraises_original(monkeypatch):
    monkeypatch.setattr(auth_device_flow.time, "sleep", lambda *_: None)
    exc = ssl.SSLEOFError(8, _SSL_EOF_MESSAGE)
    post = _ScriptedPost([exc] * 5)

    with pytest.raises(ssl.SSLEOFError) as excinfo:
        _poll(post)
    assert excinfo.value is exc
    assert post.calls == 3  # bounded, not an endless retry


def test_poll_still_handles_authorization_pending_between_transient_retries(monkeypatch):
    """The per-call transient-retry loop must not disturb the outer RFC 8628 poll semantics."""
    monkeypatch.setattr(auth_device_flow.time, "sleep", lambda *_: None)
    pending = _Response(400, {"error": "authorization_pending"})
    success = _Response(200, {"access_token": "tok"})

    post = _ScriptedPost([pending, httpx.ConnectError("connection reset"), success])
    assert _poll(post) == {"access_token": "tok"}
    assert post.calls == 3
