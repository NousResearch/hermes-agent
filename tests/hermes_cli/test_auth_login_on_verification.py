"""Out-of-band verification callback on every device/browser login flow.

``_nous_device_code_login(on_verification=)`` already lets a caller whose stdout is not a terminal
(the TUI gateway's JSON-RPC pipe, a GUI wrapper) render the verification link while the flow polls.
The codex (device + browser), xAI and MiniMax flows take the same callback with the same contract:
fired with ``(url, user_code)`` after the instructions print and BEFORE the flow starts waiting, and
a raising callback never aborts the login. Network steps are stubbed at the flows' own seams.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as codex_mod
import hermes_cli.auth_codex_browser as codex_browser_mod
import hermes_cli.auth_minimax as minimax_mod
import hermes_cli.auth_xai as xai_mod
from hermes_cli.auth_codex import _codex_device_code_login
from hermes_cli.auth_codex_browser import _codex_browser_login
from hermes_cli.auth_minimax import _minimax_oauth_login
from hermes_cli.auth_xai import _xai_oauth_device_code_login


def _codex_device(monkeypatch, events):
    monkeypatch.setattr(codex_mod, "_codex_request_device_code", lambda issuer, client_id: {
        "user_code": "UC-CODEX", "device_auth_id": "d-1", "interval": 1})

    def _poll(*_a, **_k):
        events.append("wait")
        return {"authorization_code": "ac", "code_verifier": "cv"}
    monkeypatch.setattr(codex_mod, "_codex_poll_authorization_code", _poll)
    monkeypatch.setattr(codex_mod, "_codex_exchange_authorization_code",
                        lambda *_a, **_k: {"access_token": "at", "refresh_token": "rt"})
    return lambda cb: _codex_device_code_login(on_verification=cb), (
        "https://auth.openai.com/codex/device", "UC-CODEX")


def _codex_browser(monkeypatch, events):
    monkeypatch.setattr(codex_browser_mod, "secrets", SimpleNamespace(token_urlsafe=lambda _n: "ST"))
    monkeypatch.setattr(codex_browser_mod, "_make_loopback_callback_handler", lambda *_a, **_k: (None, {}))
    monkeypatch.setattr(codex_browser_mod, "_bind_loopback_callback_server",
                        lambda *_a, **_k: SimpleNamespace(server_address=("127.0.0.1", 1455)))
    monkeypatch.setattr(codex_browser_mod, "_print_loopback_ssh_hint", lambda *_a, **_k: None)

    def _serve(*_a, **_k):
        events.append("wait")
        return {"state": "ST", "code": "AC"}
    monkeypatch.setattr(codex_browser_mod, "_serve_loopback_callback", _serve)
    monkeypatch.setattr(codex_browser_mod, "_codex_browser_exchange_code",
                        lambda *_a, **_k: {"access_token": "at", "refresh_token": "rt"})
    seen = {}
    real_url = codex_browser_mod._codex_browser_authorize_url

    def _authorize_url(**kwargs):
        seen["url"] = real_url(**kwargs)
        return seen["url"]
    monkeypatch.setattr(codex_browser_mod, "_codex_browser_authorize_url", _authorize_url)
    return (lambda cb: _codex_browser_login(open_browser=False, on_verification=cb),
            SimpleNamespace(resolve=lambda: (seen["url"], "")))


def _xai(monkeypatch, events):
    monkeypatch.setattr(auth_mod, "_xai_oauth_discovery", lambda _t: {"token_endpoint": "https://x.test/token"})
    monkeypatch.setattr(xai_mod, "_xai_oauth_request_device_code", lambda _client, **_k: {
        "verification_uri": "https://x.test/device", "verification_uri_complete": "https://x.test/device?c=UC-XAI",
        "user_code": "UC-XAI", "device_code": "dc", "expires_in": 60, "interval": 1})

    def _poll(*_a, **_k):
        events.append("wait")
        return {"access_token": "at", "refresh_token": "rt"}
    monkeypatch.setattr(auth_mod, "_xai_oauth_poll_device_token", _poll)
    return (lambda cb: _xai_oauth_device_code_login(open_browser=False, on_verification=cb),
            ("https://x.test/device?c=UC-XAI", "UC-XAI"))


def _minimax(monkeypatch, events):
    monkeypatch.setattr(auth_mod, "_minimax_request_user_code", lambda *_a, **_k: {
        "verification_uri": "https://mm.test/verify", "user_code": "UC-MM", "expired_in": 60, "interval": 2000})

    def _poll(*_a, **_k):
        events.append("wait")
        return {"access_token": "at", "refresh_token": "rt", "expired_in": 3600}
    monkeypatch.setattr(minimax_mod, "_minimax_poll_token", _poll)
    monkeypatch.setattr(auth_mod, "_minimax_save_auth_state", lambda _state: events.append("save"))
    return (lambda cb: _minimax_oauth_login(open_browser=False, on_verification=cb, persist=False),
            ("https://mm.test/verify", "UC-MM"))


@pytest.mark.parametrize("flow", [_codex_device, _codex_browser, _xai, _minimax],
                         ids=["codex-device", "codex-browser", "xai", "minimax"])
def test_verification_fires_once_before_waiting_and_a_raising_callback_is_contained(monkeypatch, flow):
    events: list = []
    run, expected = flow(monkeypatch, events)

    def _callback(url, code):
        events.append(("verify", url, code))
        raise RuntimeError("consumer bug must not abort the login")

    state = run(_callback)
    expected = expected.resolve() if hasattr(expected, "resolve") else expected

    assert events[:2] == [("verify", *expected), "wait"]
    assert state  # the flow completed after the callback raised


def test_minimax_persist_false_returns_state_without_saving_it(monkeypatch):
    events: list = []
    run, _expected = _minimax(monkeypatch, events)

    state = run(None)

    assert state["access_token"] == "at" and "save" not in events
    # Positive control: the default still persists, so the assertion above is about ``persist``.
    _minimax_oauth_login(open_browser=False)
    assert events.count("save") == 1
