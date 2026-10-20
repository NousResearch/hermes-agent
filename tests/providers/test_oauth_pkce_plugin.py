"""Invariants for the declarative OAuth PKCE helper an out-of-tree provider plugin plugs into
``ProviderProfile.auth_handler`` / ``refresh_credential``."""

from __future__ import annotations

import json
import re
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import pytest

import hermes_cli.auth
import providers
from hermes_cli import auth_oauth_pkce_plugin as pkce
from hermes_cli.auth_constants import AuthError
from providers import register_provider
from providers.base import ProviderProfile
from tests.providers.fake_pkce_idp import FakeIdP

PROVIDER = "example-pkce"


@pytest.fixture
def idp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # A local session unless a test says otherwise: run over SSH, these tests would take the paste path.
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: False)
    server = FakeIdP().start()
    yield server
    server.stop()


class PastingUser:
    """Hermes runs where it cannot open a browser; this user reads the printed authorize URL and answers each
    prompt in turn: a string as typed, an exception raised by ``input``, or a callable given the URL."""

    def __init__(self, capsys):
        self.capsys, self.answers, self.prompts, self.printed = capsys, [], [], ""

    def authorize_url(self) -> str:
        return re.search(r"^  (https?://\S+)$", self.printed, re.MULTILINE).group(1)

    def redirect_uri(self) -> str:
        return parse_qs(urlparse(self.authorize_url()).query)["redirect_uri"][0]

    def input(self, prompt: str = "") -> str:
        assert threading.current_thread() is threading.main_thread()  # no reader is left on stdin after sign-in
        self.prompts.append(prompt)
        self.printed += self.capsys.readouterr().out
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer(self.authorize_url()) if callable(answer) else answer


@pytest.fixture
def pasting(idp, monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: True)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: False)
    monkeypatch.setattr(pkce.webbrowser, "open", Mock(side_effect=AssertionError("no browser on this machine")))
    user = PastingUser(capsys)
    monkeypatch.setattr("builtins.input", user.input)
    return user


def _port_is_closed(redirect_uri: str) -> bool:
    try:
        socket.create_connection(("127.0.0.1", urlparse(redirect_uri).port), timeout=1).close()
    except OSError:
        return True
    return False


def _listener_stopped(redirect_uri: str) -> bool:
    """The callback port is closed and the thread that served it has ended."""
    alive = any(t.name == "oauth-pkce-callback" and t.is_alive() for t in threading.enumerate())
    return _port_is_closed(redirect_uri) and not alive


def _config(idp: FakeIdP, **overrides) -> pkce.OAuthPKCEConfig:
    base = pkce.OAuthPKCEConfig(client_id="hermes-example", authorize_url=f"{idp.base}/authorize",
                                token_url=f"{idp.base}/token", scopes=("inference",), timeout_seconds=5)
    return replace(base, **overrides)


def _browser_hits(url: str) -> bool:
    # A browser loads the page on its own, after open() returns; the fake IdP 302s straight to the loopback
    # callback, which answers once login() serves it.
    threading.Thread(target=lambda: urlopen(url, timeout=10).close(), daemon=True).start()
    return True


def test_pkce_handler_add_status_refresh_logout_against_fake_idp(idp, monkeypatch, tmp_path, request):
    from agent.credential_pool import load_pool

    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler, refresh = pkce.pkce_auth_handler(cfg), pkce.pkce_refresh_credential(cfg)
    # Registered like a real plugin so the credential pool finds ``refresh_credential`` through the seam.
    register_provider(ProviderProfile(name=PROVIDER, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                      auth_handler=handler, refresh_credential=refresh))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(PROVIDER, None),
                                  hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None)))
    args = SimpleNamespace(provider=PROVIDER, no_browser=False)

    assert handler("add", args) is True
    exchange = idp.token_requests[-1]
    assert exchange["grant_type"] == "authorization_code" and exchange["code_verifier"]
    assert exchange["redirect_uri"].startswith("http://127.0.0.1:")
    rows = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER]
    assert [(r["auth_type"], r["source"], r["oauth_pkce"]["client_id"]) for r in rows] == [
        ("oauth", pkce.POOL_SOURCE, "hermes-example")]
    assert rows[0]["refresh_token"] in idp.refresh_tokens and rows[0]["expires_at_ms"]
    assert handler("status", args) is True
    assert handler("refresh", args) is False  # declined → the pool's generic refresh owns it

    # Pool-driven rotation through the real refresh path; the old single-use refresh token is gone.
    before = rows[0]["refresh_token"]
    pool = load_pool(PROVIDER)
    rotated = pool.try_refresh_matching(credential_id=rows[0]["id"])
    assert rotated is not None and rotated.refresh_token != before and rotated.access_token != rows[0]["access_token"]
    assert idp.token_requests[-1] == {"grant_type": "refresh_token", "refresh_token": before, "scope": "inference",
                                      "client_id": "hermes-example"}
    assert before not in idp.refresh_tokens
    # A peer that already rotated on disk is adopted without spending the refresh token again.
    stale = load_pool(PROVIDER).entries()[0]
    stale.refresh_token = before
    assert refresh(stale)["refresh_token"] == rotated.refresh_token
    assert idp.token_requests[-1]["refresh_token"] == before  # no new POST

    assert handler("logout", args) is True
    assert load_pool(PROVIDER).entries() == []

    # Negative: a callback whose state is not ours is rejected and never reaches the token endpoint.
    idp.override_state = "forged"
    posts = len(idp.token_requests)
    with pytest.raises(AuthError) as excinfo:
        handler("add", args)
    assert excinfo.value.code == "oauth_state_mismatch" and len(idp.token_requests) == posts


def test_token_url_off_authorize_allowlist_refused_before_any_request(idp, monkeypatch):
    from hermes_cli.auth_constants import httpx

    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("token endpoint must not be contacted")))
    cfg = _config(idp, token_url="https://token.attacker.example/token")
    with pytest.raises(AuthError) as login_err:
        pkce.login(PROVIDER, cfg, open_browser=False)
    entry = SimpleNamespace(provider=PROVIDER, id="abc123", refresh_token="rt", access_token="at", expires_at_ms=None)
    with pytest.raises(AuthError) as refresh_err:
        pkce.pkce_refresh_credential(cfg)(entry)
    assert {login_err.value.code, refresh_err.value.code} == {"oauth_token_host_rejected"}
    with pytest.raises(AuthError, match="HTTPS"):
        pkce.validate_config(PROVIDER, _config(idp, authorize_url="http://auth.example.com/authorize"))


def test_spent_refresh_token_is_grant_dead_and_marks_the_pool_row_dead(idp, monkeypatch, tmp_path, request):
    """RFC 6749 invalid_grant on refresh is terminal. The pool must DEAD the row, not EXHAUST it."""
    from agent.credential_pool import STATUS_DEAD, load_pool

    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler, refresh = pkce.pkce_auth_handler(cfg), pkce.pkce_refresh_credential(cfg)
    register_provider(ProviderProfile(name=PROVIDER, auth_type="oauth_external",
                                      base_url="https://example.invalid/v1",
                                      auth_handler=handler, refresh_credential=refresh))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(PROVIDER, None),
                                  hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None)))
    args = SimpleNamespace(provider=PROVIDER, no_browser=False)
    assert handler("add", args) is True
    rows = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER]
    spent = rows[0]["refresh_token"]
    idp.refresh_tokens.discard(spent)

    with pytest.raises(AuthError) as excinfo:
        refresh(SimpleNamespace(provider=PROVIDER, id=rows[0]["id"], refresh_token=spent,
                                access_token=rows[0]["access_token"], expires_at_ms=None))
    assert excinfo.value.code == "invalid_grant"

    pool = load_pool(PROVIDER)
    result = pool.try_refresh_matching(credential_id=rows[0]["id"])
    assert result is None or result.last_status == STATUS_DEAD
    disk = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"][PROVIDER][0]
    assert disk["last_status"] == STATUS_DEAD


def test_alias_login_stores_the_row_under_the_canonical_profile_name(idp, monkeypatch, tmp_path, request):
    """hermes auth add <alias> must write the pool under ProviderProfile.name, not the typed alias."""
    from agent.credential_pool import load_pool

    alias = "example-pkce-alias"
    monkeypatch.setattr(pkce.webbrowser, "open", _browser_hits)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    cfg = _config(idp)
    handler = pkce.pkce_auth_handler(cfg)
    register_provider(ProviderProfile(
        name=PROVIDER, aliases=(alias,), auth_type="oauth_external",
        base_url="https://example.invalid/v1",
        auth_handler=handler, refresh_credential=pkce.pkce_refresh_credential(cfg)))
    request.addfinalizer(lambda: (
        providers._REGISTRY.pop(PROVIDER, None),
        providers._ALIASES.pop(alias, None),
        hermes_cli.auth.PROVIDER_REGISTRY.pop(PROVIDER, None),
        hermes_cli.auth.PROVIDER_REGISTRY.pop(alias, None)))
    args = SimpleNamespace(provider=alias, no_browser=False)
    assert handler("add", args) is True
    pool = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]
    assert PROVIDER in pool and alias not in pool
    assert handler("status", args) is True
    assert load_pool(PROVIDER).entries()[0].provider == PROVIDER


def test_browser_on_another_machine_signs_in_with_the_pasted_address(pasting, idp):
    landed = []

    def copy_from_the_address_bar(authorize_url: str) -> str:
        landed.append(idp.landing_address(authorize_url))
        return landed[0]

    pasting.answers.append(copy_from_the_address_bar)
    tokens = pkce.login(PROVIDER, _config(idp))

    exchange = idp.token_requests[-1]
    assert exchange["grant_type"] == "authorization_code" and exchange["code_verifier"]
    assert exchange["redirect_uri"] == landed[0].split("?", 1)[0] == pasting.redirect_uri()
    assert tokens["access_token"] and tokens["refresh_token"] in idp.refresh_tokens
    assert pasting.prompts == ["Redirect address: "]
    printed = pasting.printed + pasting.capsys.readouterr().out
    assert "paste" in printed and exchange["code"] not in printed  # Hermes does not print the address back
    assert _listener_stopped(pasting.redirect_uri())


def test_paste_prompt_asks_again_until_the_address_carries_a_code(pasting, idp):
    pasting.answers += ["", "https://example.invalid/somewhere-else",
                        lambda url: "?" + idp.landing_address(url).split("?", 1)[1]]  # the query part works too
    assert pkce.login(PROVIDER, _config(idp))["access_token"]
    assert len(pasting.prompts) == 3 and len(idp.token_requests) == 1
    assert _listener_stopped(pasting.redirect_uri())


def test_pasted_address_from_another_sign_in_is_refused_before_the_token_endpoint(pasting, idp):
    idp.override_state = "forged"
    pasting.answers.append(idp.landing_address)
    with pytest.raises(AuthError) as excinfo:
        pkce.login(PROVIDER, _config(idp))
    assert excinfo.value.code == "oauth_state_mismatch" and idp.token_requests == []
    assert _listener_stopped(pasting.redirect_uri())


def test_an_ssh_tunnel_still_completes_the_sign_in_on_an_empty_line(pasting, idp):
    def browser_through_the_tunnel(authorize_url: str) -> str:
        with urlopen(authorize_url, timeout=5) as page:  # the IdP 302s to the loopback, which the tunnel reaches
            assert b"received" in page.read()
        return ""

    pasting.answers.append(browser_through_the_tunnel)
    assert pkce.login(PROVIDER, _config(idp))["access_token"]
    assert [r["grant_type"] for r in idp.token_requests] == ["authorization_code"]
    assert _listener_stopped(pasting.redirect_uri())


def test_an_empty_line_waits_for_the_callback_the_loopback_is_still_writing(pasting, idp, monkeypatch):
    """Enter lands while the loopback handler has stored the code but not yet the state: the sign-in must read
    the finished callback, not a half-written copy."""
    from hermes_cli import auth_device_flow

    halfway, finish = threading.Event(), threading.Event()

    class HeldQuery(dict):
        def get(self, key, default=None):
            if key == "state":  # the handler stores code, then state: hold it in between
                halfway.set()
                finish.wait(timeout=5)
            return super().get(key, default)

    monkeypatch.setattr(auth_device_flow, "parse_qs", lambda query: HeldQuery(parse_qs(query)))

    def enter_while_the_callback_is_half_written(authorize_url: str) -> str:
        threading.Thread(target=lambda: urlopen(authorize_url, timeout=10).close(), daemon=True).start()
        assert halfway.wait(timeout=5)
        threading.Timer(0.2, finish.set).start()  # the handler finishes after the prompt has returned
        return ""

    pasting.answers.append(enter_while_the_callback_is_half_written)
    assert pkce.login(PROVIDER, _config(idp))["access_token"]
    assert _listener_stopped(pasting.redirect_uri())


def test_closed_input_at_the_paste_prompt_fails_and_frees_the_port(pasting, idp):
    pasting.answers.append(EOFError())
    with pytest.raises(AuthError) as excinfo:
        pkce.login(PROVIDER, _config(idp))
    assert excinfo.value.code == "oauth_no_code" and idp.token_requests == []
    assert _listener_stopped(pasting.redirect_uri())


def test_ctrl_c_at_the_paste_prompt_cancels_and_frees_the_port(pasting, idp):
    pasting.answers.append(KeyboardInterrupt())
    with pytest.raises(SystemExit) as excinfo:
        pkce.login(PROVIDER, _config(idp))
    assert excinfo.value.code == 130 and idp.token_requests == []
    assert "Login cancelled." in pasting.capsys.readouterr().out and _listener_stopped(pasting.redirect_uri())


@pytest.mark.parametrize("forged", [False, True], ids=["this-sign-in", "forged-state"])
def test_a_pasted_refusal_ends_the_sign_in_as_denied(pasting, idp, forged):
    """An address carrying ``error`` ends the sign-in as the loopback does; nothing is exchanged either way."""
    def refused(authorize_url: str) -> str:
        state = "forged" if forged else parse_qs(urlparse(authorize_url).query)["state"][0]
        return "?" + urlencode({"error": "access_denied", "state": state})

    pasting.answers.append(refused)
    with pytest.raises(AuthError) as excinfo:
        pkce.login(PROVIDER, _config(idp))
    assert excinfo.value.code == "oauth_authorization_denied" and idp.token_requests == []
    assert _listener_stopped(pasting.redirect_uri())


@pytest.mark.parametrize("launch", [Mock(return_value=False), Mock(side_effect=OSError("no browser"))],
                         ids=["open-returned-false", "open-raised"])
def test_a_browser_that_does_not_start_falls_back_to_pasting(pasting, idp, monkeypatch, launch):
    """Hermes thought it could open a browser here, but none started: the user opens the URL elsewhere."""
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: False)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(pkce.webbrowser, "open", launch)
    pasting.answers.append(idp.landing_address)
    assert pkce.login(PROVIDER, _config(idp))["access_token"]
    assert launch.call_count == 1 and pasting.prompts == ["Redirect address: "]
    assert _listener_stopped(pasting.redirect_uri())


_DENIED = {"code": None, "state": None, "error": "access_denied", "error_description": None}


@pytest.mark.parametrize(("remote", "graphical", "open_browser", "opened", "asks"), [
    (False, True, True, True, False),  # a desktop: the browser opens and the loopback answers
    (False, True, False, None, False),  # --no-browser on a desktop: the user opens the URL on this machine
    (False, True, True, False, True),  # a desktop browser that did not start
    (False, False, True, None, True),  # no graphical browser on this machine
    (True, True, True, True, True),  # SSH with X11 forwarding: it asks; Enter takes what the loopback received
    (True, False, False, None, True),  # SSH, a cloud shell
])
def test_which_sign_ins_ask_for_the_address(idp, monkeypatch, remote, graphical, open_browser, opened, asks):
    monkeypatch.setattr("hermes_cli.auth_device_flow._is_remote_session", lambda: remote)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: graphical)
    launch = Mock(return_value=opened)
    monkeypatch.setattr(pkce.webbrowser, "open", launch)
    waited = []

    def waiter(name):
        def wait(server, *args, **kwargs):
            waited.append(name)
            server.server_close()
            return dict(_DENIED)
        return wait

    monkeypatch.setattr(pkce, "_pasted_callback", waiter("paste"))
    monkeypatch.setattr("hermes_cli.auth_device_flow._serve_loopback_callback", waiter("loopback"))
    with pytest.raises(AuthError, match="access_denied"):
        pkce.login(PROVIDER, _config(idp), open_browser=open_browser)
    assert waited == (["paste"] if asks else ["loopback"])
    assert launch.call_count == (1 if open_browser and graphical else 0)
