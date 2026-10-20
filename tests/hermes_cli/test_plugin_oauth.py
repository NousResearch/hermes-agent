"""Plugin-declared OAuth clients: Hermes signs in, stores and refreshes the grant; plugin code never holds a token.

A real plugin directory under the test HERMES_HOME declares ``oauth:``. ``FakeIdP`` plays the authorization
server (PKCE, rotating refresh tokens) and a loopback stub plays the API the bearer is meant for.
"""

from __future__ import annotations

import importlib.util
import json
import os
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from hermes_cli import auth_oauth_pkce_plugin as pkce
from hermes_cli import plugin_oauth
from hermes_cli import plugin_provider_requests as ppr
from hermes_cli.auth_constants import AuthError
from tests.providers.fake_pkce_idp import FakeIdP

CLIENT_ID = "voice-client.apps.example"
SECRET = "canary-client-secret-7f3a91"
SCOPES = ["https://api.example.com/auth/calendar", "https://api.example.com/auth/docs"]
KEY = "voice:google"


class _Api:
    """Loopback stand-in for the third-party API; records every request it receives."""

    def __init__(self, status=200, location=""):
        self.seen: list[dict] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                api.seen.append({"path": self.path, "headers": dict(self.headers), "body": self.rfile.read(length)})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            do_GET = do_POST = do_PATCH = _answer

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = ("http", "127.0.0.1", self.server.server_address[1])
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _entry(idp: FakeIdP, **overrides) -> dict:
    entry = {"id": "google", "authorize_url": f"{idp.base}/authorize", "token_url": f"{idp.base}/token",
             "scopes": SCOPES, "client_id_env": "VOICE_CLIENT_ID", "client_secret_env": "VOICE_CLIENT_SECRET",
             "hosts": ["api.example.com"], "authorize_params": {"access_type": "offline", "prompt": "consent"}}
    return {**entry, **overrides}


def _plugin(name: str = "voice", oauth: list | None = None, display: str | None = None, **fields):
    """Write ``plugins/<name>`` (manifest plus a module that calls the helper) and import it like a plugin.
    *display* is a manifest ``name`` other than the directory; *fields* are further manifest keys."""
    root = _home() / "plugins" / name
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"name": display or name, "version": "1.0.0", "description": "test plugin",
                "requires_env": [{"name": "VOICE_CLIENT_ID"}, {"name": "VOICE_CLIENT_SECRET"}], **fields}
    if oauth is not None:
        manifest["oauth"] = oauth
    (root / "plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")  # JSON is valid YAML
    (root / "api.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def call(provider, url, method='POST', **kwargs):\n"
        "    return credentialed_provider_request(provider, method, url, **kwargs)\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"hermes_plugins_{name}_api", root / "api.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _auth(action: str, provider: str = KEY):
    args = SimpleNamespace(provider=provider, no_browser=False, timeout=None)
    return plugin_oauth.handle_auth_action(action, args, provider)


def _grant(key: str = KEY) -> dict | None:
    store = json.loads((_home() / "auth.json").read_text(encoding="utf-8-sig"))
    return (store.get("plugin_oauth") or {}).get(key)


def _set_grant(key: str = KEY, **fields) -> None:
    path = _home() / "auth.json"
    store = json.loads(path.read_text(encoding="utf-8-sig"))
    store["plugin_oauth"][key].update(fields)
    path.write_text(json.dumps(store), encoding="utf-8")


def _refreshes(idp: FakeIdP) -> int:
    return sum(r["grant_type"] == "refresh_token" for r in idp.token_requests)


@pytest.fixture
def idp(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("VOICE_CLIENT_SECRET", SECRET)
    server = FakeIdP().start(tls_dir=tmp_path / "idp-ca")  # https: plugin OAuth endpoints are https-only
    trust = ssl.create_default_context(cafile=str(server.ca_pem))
    monkeypatch.setattr("hermes_cli.auth._default_verify", lambda: trust)
    server.opened = []

    def browser(url: str) -> bool:
        # A real browser runs beside Hermes; the fake IdP 302s straight to the loopback callback.
        server.opened.append(url)
        threading.Thread(target=lambda: urlopen(url, timeout=10, context=trust).close(), daemon=True).start()
        return True

    monkeypatch.setattr(pkce.webbrowser, "open", browser)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    yield server
    server.stop()


@pytest.fixture
def api(monkeypatch):
    """Test-only: the stub joins the declared hosts' origins; a second stub stays outside them."""
    allowed, other = _Api(), _Api()
    declared = plugin_oauth._origins
    monkeypatch.setattr(plugin_oauth, "_origins", lambda client: declared(client) | {allowed.origin})
    yield allowed, other
    allowed.close()
    other.close()


def _signed_in(idp: FakeIdP, **entry_overrides):
    plugin = _plugin(oauth=[_entry(idp, **entry_overrides)])
    assert _auth("add") is True
    return plugin


def test_sign_in_stores_the_grant_then_status_refresh_and_logout(idp, capsys):
    from hermes_cli import auth_commands

    _plugin(oauth=[_entry(idp)])
    args = SimpleNamespace(provider=KEY, no_browser=False, timeout=None)
    auth_commands.auth_add_command(args)

    assert "access_type=offline" in idp.opened[-1] and "prompt=consent" in idp.opened[-1]
    exchange = idp.token_requests[-1]
    assert exchange["grant_type"] == "authorization_code" and exchange["code_verifier"]
    assert (exchange["client_id"], exchange["client_secret"]) == (CLIENT_ID, SECRET)
    grant = _grant()
    assert grant["refresh_token"] in idp.refresh_tokens and grant["access_token"] and grant["expires_at_ms"]
    assert SECRET not in (_home() / "auth.json").read_text(encoding="utf-8-sig")

    auth_commands.auth_status_command(args)
    auth_commands.auth_refresh_command(args)
    assert _refreshes(idp) == 1
    auth_commands.auth_logout_command(args)
    assert _grant() is None
    out = capsys.readouterr()
    assert "voice:google: logged in" in out.out and SECRET not in out.out + out.err


def test_bearer_reaches_the_declared_host_and_never_the_plugin(idp, api):
    allowed, _ = api
    plugin = _signed_in(idp)
    token = _grant()["access_token"]

    result = plugin.call("google", allowed.url + "/v1/items?x=1", json={"title": "t"},
                         headers={"X-Plugin": "1", "authorization": "Bearer plugin-made"})

    assert result.status == 200 and result.json() == {"ok": True}
    seen = allowed.seen[0]
    assert seen["path"] == "/v1/items?x=1" and json.loads(seen["body"]) == {"title": "t"}
    assert seen["headers"]["Authorization"] == f"Bearer {token}" and seen["headers"]["X-Plugin"] == "1"
    assert token not in repr(result) and token not in json.dumps(result.headers)


def test_foreign_origin_userinfo_and_redirect_never_carry_the_token(idp, api, monkeypatch):
    allowed, other = api
    plugin = _signed_in(idp)
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.call("google", other.url + "/steal")
    host = allowed.url.split("//", 1)[1]
    for url in (f"http://user@{host}/x", f"http://{host}@127.0.0.1:{other.origin[2]}/x"):
        with pytest.raises(PermissionError):
            plugin.call("google", url)

    redirecting = _Api(status=302, location=other.url + "/steal")
    declared = plugin_oauth._origins
    monkeypatch.setattr(plugin_oauth, "_origins", lambda client: declared(client) | {redirecting.origin})
    try:
        assert plugin.call("google", redirecting.url + "/v1").status == 302
    finally:
        redirecting.close()
    assert other.seen == [] and allowed.seen == []


def test_undeclared_id_another_plugins_grant_and_non_plugin_callers_are_refused(idp, api):
    allowed, _ = api
    _signed_in(idp)
    with pytest.raises(ValueError, match="google"):
        _plugin("bare").call("google", allowed.url)
    twin = _plugin("twin", oauth=[_entry(idp)])  # the same declaration, but no grant of its own
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add twin:google"):
        twin.call("google", allowed.url)
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_request("google", "GET", allowed.url)
    assert allowed.seen == []


def test_own_oauth_id_wins_over_a_core_provider_of_the_same_name(idp, api, monkeypatch):
    allowed, _ = api
    plugin = _signed_in(idp)

    def core_resolver(url):
        raise AssertionError("a core provider's credential must not answer the plugin's own oauth id")

    monkeypatch.setitem(ppr._PROVIDERS, "google", (frozenset({allowed.origin}), core_resolver))
    assert plugin.call("google", allowed.url).status == 200
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {_grant()['access_token']}"


def test_not_signed_in_names_the_command_and_sends_nothing(idp, api):
    allowed, _ = api
    plugin = _plugin(oauth=[_entry(idp)])
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add voice:google"):
        plugin.call("google", allowed.url)
    assert allowed.seen == [] and idp.token_requests == []


def test_expiring_token_is_refreshed_once_and_the_rotation_kept(idp, api):
    allowed, _ = api
    plugin = _signed_in(idp)
    old_refresh = _grant()["refresh_token"]
    _set_grant(expires_at_ms=int(time.time() * 1000) + 60_000)  # inside the 120 s margin

    plugin.call("google", allowed.url)
    plugin.call("google", allowed.url)

    assert [r for r in idp.token_requests if r["grant_type"] == "refresh_token"] == [{
        "grant_type": "refresh_token", "refresh_token": old_refresh, "client_id": CLIENT_ID, "client_secret": SECRET}]
    grant = _grant()
    assert grant["refresh_token"] != old_refresh and grant["refresh_token"] in idp.refresh_tokens
    assert [s["headers"]["Authorization"] for s in allowed.seen] == [f"Bearer {grant['access_token']}"] * 2


def test_refresh_without_a_new_refresh_token_keeps_the_old_one(idp, api):
    """Google answers a refresh with an access token only; the stored refresh token must survive it."""
    allowed, _ = api
    plugin = _signed_in(idp)
    old_refresh = _grant()["refresh_token"]
    _set_grant(expires_at_ms=0)
    idp.token_answer = (200, {"access_token": "at-google-style", "expires_in": 3599, "token_type": "Bearer"})

    assert plugin.call("google", allowed.url).status == 200

    grant = _grant()
    assert (grant["access_token"], grant["refresh_token"]) == ("at-google-style", old_refresh)
    assert allowed.seen[-1]["headers"]["Authorization"] == "Bearer at-google-style"


def test_a_redirect_from_the_token_endpoint_is_a_failure_not_a_token(idp, api):
    allowed, _ = api
    redirect = (302, {"access_token": "at-from-a-redirect", "refresh_token": "rt-x", "expires_in": 3600})
    _plugin(oauth=[_entry(idp)])
    idp.token_answer = redirect
    with pytest.raises(SystemExit, match="HTTP 302"):
        _auth("add")
    assert not (_home() / "auth.json").exists() or _grant() is None

    idp.token_answer = None
    plugin = _signed_in(idp)
    _set_grant(expires_at_ms=0)
    before = _grant()
    idp.token_answer = redirect
    with pytest.raises(AuthError) as excinfo:
        plugin.call("google", allowed.url)
    assert excinfo.value.code == "oauth_refresh_failed" and _grant() == before and allowed.seen == []


def test_two_threads_with_an_expiring_token_refresh_once(idp, api):
    allowed, _ = api
    plugin = _signed_in(idp)
    _set_grant(expires_at_ms=0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: plugin.call("google", allowed.url).status, range(2)))

    assert statuses == [200, 200] and _refreshes(idp) == 1 and "dead" not in _grant()


def test_live_token_is_sent_while_another_thread_holds_the_auth_lock(idp, api):
    from hermes_cli.auth import _auth_store_lock

    allowed, _ = api
    plugin = _signed_in(idp)
    held, release = threading.Event(), threading.Event()

    def holder():
        with _auth_store_lock():
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(5)
        started = time.monotonic()
        assert plugin.call("google", allowed.url).status == 200
        assert time.monotonic() - started < 5
    finally:
        release.set()
        thread.join(10)


def test_revoked_grant_is_marked_dead_and_stops_touching_the_network(idp, api, capsys):
    allowed, _ = api
    plugin = _signed_in(idp)
    idp.refresh_tokens.clear()  # the user removed the app at the IdP
    _set_grant(expires_at_ms=0)
    with pytest.raises(ppr.ProviderNotSignedIn, match="revoked"):
        plugin.call("google", allowed.url)
    posts = len(idp.token_requests)
    with pytest.raises(ppr.ProviderNotSignedIn, match="revoked"):
        plugin.call("google", allowed.url)

    assert len(idp.token_requests) == posts and allowed.seen == [] and _grant()["dead"] == "invalid_grant"
    capsys.readouterr()
    assert _auth("status") is True and "revoked" in capsys.readouterr().out


def test_rejected_client_is_an_auth_error_that_leaves_the_grant(idp, api):
    allowed, _ = api
    plugin = _signed_in(idp)
    _set_grant(expires_at_ms=0)
    idp.token_answer = (401, {"error": "invalid_client", "error_description": "The OAuth client was not found."})
    for _ in range(2):
        with pytest.raises(AuthError) as excinfo:
            plugin.call("google", allowed.url)
        assert excinfo.value.code == "invalid_client"
    assert "dead" not in _grant() and _refreshes(idp) == 2 and allowed.seen == []


def test_missing_client_env_fails_before_any_request(idp, api, monkeypatch):
    allowed, _ = api
    plugin = _signed_in(idp)
    _set_grant(expires_at_ms=0)
    monkeypatch.delenv("VOICE_CLIENT_SECRET")
    posts = len(idp.token_requests)
    with pytest.raises(AuthError) as excinfo:
        plugin.call("google", allowed.url)
    assert excinfo.value.code == "oauth_client_secret_missing" and "VOICE_CLIENT_SECRET" in str(excinfo.value)
    assert len(idp.token_requests) == posts and allowed.seen == []


@pytest.mark.parametrize("change", [
    {"token_url": "https://collect.example.net/token"},
    {"hosts": ["api.example.com", "collect.example.net"]},
    {"scopes": SCOPES[:1]},
    {"client_id_env": "OTHER_CLIENT_ID"},
    {"client_secret_env": "OTHER_CLIENT_SECRET"},
    {"authorize_url": "https://login.example.net/authorize"},
    {"authorize_params": {"access_type": "online"}},
])
def test_entry_changed_since_sign_in_is_not_used(idp, api, change):
    allowed, _ = api
    _signed_in(idp)
    changed = _plugin(oauth=[_entry(idp, **change)])  # a later version of the same plugin
    posts = len(idp.token_requests)
    with pytest.raises(ppr.ProviderNotSignedIn, match="changed"):
        changed.call("google", allowed.url)
    assert len(idp.token_requests) == posts and allowed.seen == []


def test_partial_consent_saves_nothing_and_names_what_is_missing(idp):
    _plugin(oauth=[_entry(idp)])
    idp.granted_scope = SCOPES[0]
    with pytest.raises(SystemExit, match=SCOPES[1]):
        _auth("add")
    assert not (_home() / "auth.json").exists() or _grant() is None

    idp.granted_scope = " ".join(SCOPES)
    assert _auth("add") is True and _grant()["refresh_token"]


def test_grant_belongs_to_one_profile(idp, api, monkeypatch, tmp_path):
    allowed, _ = api
    _signed_in(idp)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
    with pytest.raises(ppr.ProviderNotSignedIn):
        _plugin(oauth=[_entry(idp)]).call("google", allowed.url)
    assert allowed.seen == []


def test_client_secret_stays_out_of_config_repr_logs_and_errors(idp, api, caplog, capsys):
    allowed, _ = api
    plugin = _signed_in(idp)
    manifest = json.loads((_home() / "plugins" / "voice" / "plugin.yaml").read_text(encoding="utf-8-sig"))
    client = plugin_oauth.parse_oauth_clients(manifest)[0][0]
    assert SECRET not in repr(plugin_oauth._pkce_config(client, KEY))

    _set_grant(expires_at_ms=0)
    idp.token_answer = (500, {"error": "server_error", "error_description": SECRET})
    with pytest.raises(AuthError) as excinfo:
        plugin.call("google", allowed.url)
    out = capsys.readouterr()
    assert excinfo.value.code == "oauth_refresh_failed"
    assert SECRET not in str(excinfo.value) + caplog.text + out.out + out.err


def test_custom_prefix_never_routes_to_a_plugin_grant(idp):
    _plugin("custom", oauth=[_entry(idp, id="foo")])
    assert _auth("add", "custom:foo") is None and idp.token_requests == []


def test_other_auth_store_writes_keep_the_plugin_grant(idp):
    from hermes_cli.auth import _save_active_provider_state

    _signed_in(idp)
    before = _grant()
    _save_active_provider_state("nous", {"access_token": "unrelated"})
    assert _grant() == before


def test_install_notice_and_repin_delta_name_the_oauth_entry(idp, tmp_path):
    from hermes_cli.plugins_cmd_catalog import plugin_surface, surface_delta, surface_delta_lines

    manifest = {"name": "voice", "oauth": [_entry(idp)]}
    notice = ppr.requires_auth_notice(manifest)
    assert "google" in notice and "api.example.com" in notice and "hermes auth add voice:google" in notice

    widened = {"name": "voice", "oauth": [_entry(idp, hosts=["api.example.com", "collect.example.net"])]}
    lines = surface_delta_lines(surface_delta(plugin_surface(manifest, tmp_path), plugin_surface(widened, tmp_path)))
    assert len(lines) == 1 and lines[0].startswith("your OAuth sign-in for") and "collect.example.net" in lines[0]

    installed = plugin_surface(manifest, tmp_path)
    moved = {"name": "voice", "oauth": [_entry(idp, authorize_url="https://signin.example.net/authorize")]}
    lines = surface_delta_lines(surface_delta(installed, plugin_surface(moved, tmp_path)))
    assert len(lines) == 1 and "signin.example.net" in lines[0]

    same_host = {"name": "voice", "oauth": [_entry(idp, authorize_url=f"{idp.base}/v2/authorize")]}
    assert surface_delta(installed, plugin_surface(same_host, tmp_path)) == {}


def test_the_notice_command_finds_a_plugin_whose_name_differs_from_its_directory(idp, api):
    allowed, _ = api
    plugin = _plugin("voice-dir", oauth=[_entry(idp)], display="gemini-voice")
    manifest = json.loads((_home() / "plugins" / "voice-dir" / "plugin.yaml").read_text(encoding="utf-8-sig"))
    assert "hermes auth add gemini-voice:google" in ppr.requires_auth_notice(manifest)

    assert _auth("add", "gemini-voice:google") is True
    assert _grant("voice-dir:google")["refresh_token"] in idp.refresh_tokens
    assert plugin.call("google", allowed.url).status == 200


# ── Security invariants (review of #139985) ─────────────────────────────────────────────────────────


def _reachable_text(exc: BaseException) -> str:
    """Everything plugin code can read off a caught exception: repr, chain, httpx request, frame locals."""
    import traceback

    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append("".join(traceback.TracebackException.from_exception(exc, capture_locals=True).format()))
        try:
            request = exc.request  # httpx errors carry the request they sent
        except (AttributeError, RuntimeError):
            request = None
        if request is not None:
            parts += [repr(dict(request.headers)), request.content.decode("utf-8", "replace")]
        exc = exc.__cause__ or exc.__context__
    return "\n".join(parts)


def test_errors_raised_to_the_plugin_carry_no_token_refresh_token_or_secret(idp, api):
    """A failed API call or refresh hands the plugin an exception; nothing in it may lead back to a credential."""
    allowed, _ = api
    plugin = _signed_in(idp)
    token = _grant()["access_token"]
    allowed.close()  # the API host is down: httpx raises with the request it tried to send
    exc = _caught(lambda: plugin.call("google", allowed.url + "/v1"))
    assert type(exc).__name__ == "ConnectError" and token not in _reachable_text(exc)

    refresh = _grant()["refresh_token"]
    _set_grant(expires_at_ms=0)
    idp.stop()  # the token endpoint is down mid-refresh
    exc = _caught(lambda: plugin.call("google", allowed.url + "/v1"))
    assert isinstance(exc, AuthError) and exc.code == "oauth_refresh_failed"
    text = _reachable_text(exc)
    assert refresh not in text and SECRET not in text

    import httpx  # a core resolver (Codex) whose own refresh POST fails raises httpx with that request attached

    def resolver(url):
        httpx.post(f"http://127.0.0.1:{allowed.origin[2]}/token", data={"refresh_token": "rt-core-canary"}, timeout=2)

    _ppr_providers = dict(ppr._PROVIDERS)
    ppr._PROVIDERS["core-x"] = (frozenset({allowed.origin}), resolver)
    try:
        bare = _plugin("bare-core")
        (_home() / "plugins" / "bare-core" / "plugin.yaml").write_text(json.dumps(
            {"name": "bare-core", "version": "1", "requires_auth": ["core-x"]}), encoding="utf-8")
        exc = _caught(lambda: bare.call("core-x", allowed.url + "/v1"))
    finally:
        ppr._PROVIDERS.clear()
        ppr._PROVIDERS.update(_ppr_providers)
    assert type(exc).__name__ == "ConnectError" and "rt-core-canary" not in _reachable_text(exc)


def _caught(fn) -> BaseException:
    try:
        fn()
    except Exception as exc:  # what plugin code would catch
        return exc
    raise AssertionError("expected an error")


def test_revoked_grant_keeps_no_tokens_and_uninstall_drops_the_grant(idp, api):
    allowed, _ = api
    plugin = _signed_in(idp)
    idp.refresh_tokens.clear()
    _set_grant(expires_at_ms=0)
    with pytest.raises(ppr.ProviderNotSignedIn, match="revoked"):
        plugin.call("google", allowed.url)
    assert set(_grant()) == {"dead", "fingerprint"}

    from hermes_cli.plugins_cmd_remove import _remove_user_plugin

    # Broken manifest after sign-in: logout still removes the grant.
    assert _auth("add") is True and _grant()["access_token"]
    manifest = _home() / "plugins" / "voice" / "plugin.yaml"
    good = manifest.read_text(encoding="utf-8-sig")
    manifest.write_text(good.replace('"hosts": ["api.example.com"]', '"hosts": ["http://bad"]'), encoding="utf-8")
    assert _auth("logout") is True and _grant() is None
    manifest.write_text(good, encoding="utf-8")

    # Uninstall drops the grant, also when the manifest name differs from the directory it is stored under.
    _plugin("voice-dir", oauth=[_entry(idp)], display="gemini-voice")
    assert _auth("add", "gemini-voice:google") is True and _grant("voice-dir:google")
    _remove_user_plugin(_home() / "plugins", "gemini-voice", _home() / "plugins" / "voice-dir")
    assert _grant("voice-dir:google") is None  # a later plugin under that key does not inherit the sign-in


def test_sign_in_is_refused_where_nobody_can_answer_the_browser(idp, monkeypatch):
    _plugin(oauth=[_entry(idp)])
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    with pytest.raises(SystemExit, match="your own terminal"):
        _auth("add")
    assert idp.token_requests == [] and idp.opened == []


def _registered_like_a_fresh_process(monkeypatch) -> None:
    """``hermes_cli.config`` feeds every platform manifest's requires_env/optional_env into OPTIONAL_ENV_VARS at
    import; this process imported it before the test wrote its plugins, so run that pass again."""
    from hermes_cli import config

    monkeypatch.setattr(config, "OPTIONAL_ENV_VARS", dict(config.OPTIONAL_ENV_VARS))
    config._inject_platform_plugin_env_vars()


def test_a_platform_plugin_signs_in_with_the_client_variables_it_declares(idp, monkeypatch):
    """A ``kind: platform`` plugin lists its client variables in requires_env, as catalog admission demands, and
    that list also feeds the config form: they stay its own, not variables Hermes uses itself."""
    from hermes_cli.config_env_routing import is_registered_env_name

    _plugin(oauth=[_entry(idp)], kind="platform")
    _registered_like_a_fresh_process(monkeypatch)
    assert is_registered_env_name("VOICE_CLIENT_ID") and is_registered_env_name("VOICE_CLIENT_SECRET")

    assert _auth("add") is True and _grant()["refresh_token"] in idp.refresh_tokens


def test_an_update_that_moves_a_platform_plugins_token_host_asks_first(idp, monkeypatch, tmp_path):
    """The installed copy and its next version both list the client in the re-pin consent surface, so a moved
    token host shows up as added."""
    from hermes_cli.plugins_cmd_catalog import plugin_surface, surface_delta, surface_delta_lines

    _plugin(oauth=[_entry(idp)], kind="platform")
    _registered_like_a_fresh_process(monkeypatch)
    installed = json.loads((_home() / "plugins" / "voice" / "plugin.yaml").read_text(encoding="utf-8"))
    moved = {**installed, "version": "1.1.0", "oauth": [_entry(idp, token_url="https://tokens.example.net/token")]}

    lines = surface_delta_lines(surface_delta(plugin_surface(installed, tmp_path), plugin_surface(moved, tmp_path)))
    assert len(lines) == 1 and lines[0].startswith("your OAuth sign-in for") and "tokens.example.net" in lines[0]


@pytest.mark.parametrize("name", [
    "API_SERVER_KEY", "FEISHU_ENCRYPT_KEY", "VOICE_HOME_CHANNEL", "NTFY_TOKEN", "CHAT_BOT_TOKEN",
], ids=["core", "extra", "hidden-setup-suffix", "bundled-platform", "another-plugins-undeclared"])
def test_a_platform_plugin_still_cannot_take_a_variable_hermes_uses_itself(idp, monkeypatch, name):
    """Hermes would POST the value to the token URL the plugin names. Listing a Hermes variable in the plugin's
    own requires_env does not make it the plugin's, and another plugin's variable needs at least that."""
    _plugin("chat", requires_env=[{"name": "CHAT_BOT_TOKEN"}], kind="platform")
    declared = [] if name == "CHAT_BOT_TOKEN" else [{"name": name}]
    _plugin(oauth=[_entry(idp, client_secret_env=name)], kind="platform",
            requires_env=[{"name": "VOICE_CLIENT_ID"}, *declared])
    _registered_like_a_fresh_process(monkeypatch)

    with pytest.raises(ValueError, match=f"{name} is a variable Hermes uses itself"):
        plugin_oauth._declared_clients(_home() / "plugins" / "voice")
    assert idp.token_requests == []


@pytest.mark.parametrize("requires_env", [1, True, {"VOICE_CLIENT_ID": {}, "VOICE_CLIENT_SECRET": {}}],
                         ids=["number", "bool", "mapping"])
def test_a_requires_env_that_is_not_a_list_declares_nothing(idp, monkeypatch, requires_env):
    """The install notice, the consent surface, admission and sign-in all parse the manifest: a requires_env of
    another shape leaves the registered client variables undeclared, a reported problem rather than a crash."""
    _plugin(oauth=[_entry(idp)], kind="platform")
    _registered_like_a_fresh_process(monkeypatch)
    manifest = {"name": "voice", "kind": "platform", "requires_env": requires_env, "oauth": [_entry(idp)]}

    clients, problems = plugin_oauth.parse_oauth_clients(manifest)
    assert clients == [] and any("VOICE_CLIENT_ID is a variable Hermes uses itself" in p for p in problems)
