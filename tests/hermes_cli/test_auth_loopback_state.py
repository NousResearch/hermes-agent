"""Loopback OAuth logins refuse a non-ASCII callback ``state`` as an ordinary state mismatch.

``hmac.compare_digest`` raises TypeError on a non-ASCII ``str``, and the loopback redirect's query is
whatever reached the listener, so a forged ``state=%C3%A9`` must get the same refusal as a forged ASCII
one. Drives the real loopback listener; only the system browser is replaced by a client that hits the
redirect URI with a forged state.
"""

from __future__ import annotations

import threading
import urllib.request
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

from hermes_cli.auth_constants import AuthError


def _no_token_exchange(*_args, **_kwargs):
    pytest.fail("a forged state must never reach the token endpoint")


def _codex_login(monkeypatch):
    from hermes_cli import auth_codex_browser as mod

    monkeypatch.setattr(mod, "CODEX_BROWSER_CALLBACK_PORT", 0)  # ephemeral; production is 1455
    monkeypatch.setattr(mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(mod, "_codex_browser_exchange_code", _no_token_exchange)
    return lambda: mod._codex_browser_login(timeout_seconds=5)


def _pkce_plugin_login(monkeypatch):
    from hermes_cli import auth_oauth_pkce_plugin as mod

    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(mod, "_post_token", _no_token_exchange)
    cfg = mod.OAuthPKCEConfig(client_id="hermes-example", authorize_url="https://idp.example/authorize",
                              token_url="https://idp.example/token", timeout_seconds=5)
    return lambda: mod.login("example-pkce", cfg)


@pytest.mark.parametrize("start", [_codex_login, _pkce_plugin_login], ids=["codex", "pkce_plugin"])
def test_non_ascii_callback_state_is_an_ordinary_state_mismatch(monkeypatch, start):
    login = start(monkeypatch)

    def refusal_code(forged_state: str) -> str:
        def browser(url):
            redirect_uri = parse_qs(urlparse(url).query)["redirect_uri"][0]
            callback = f"{redirect_uri}?{urlencode({'code': 'AC-1', 'state': forged_state})}"
            threading.Thread(target=lambda: urllib.request.urlopen(callback, timeout=5).read(), daemon=True).start()
            return True

        monkeypatch.setattr("webbrowser.open", browser)
        with pytest.raises(AuthError) as excinfo:
            login()
        return excinfo.value.code

    ascii_code = refusal_code("forged")
    assert ascii_code.endswith("state_mismatch")
    assert refusal_code("forgé") == ascii_code
