"""Regression tests for Google Workspace OAuth setup.

These tests cover the headless/manual auth-code flow where the browser step and
code exchange happen in separate process invocations.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


class FakeCredentials:
    from_authorized_user_file_impl = None

    def __init__(self, payload=None):
        self._payload = payload or {
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/contacts.readonly",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/documents.readonly",
            ],
        }
        self.valid = True
        self.expired = False
        self.refresh_token = self._payload.get("refresh_token", "refresh-token")
        self.refresh_error = None

    def to_json(self):
        return json.dumps(self._payload)

    def refresh(self, _request):
        if self.refresh_error is not None:
            raise self.refresh_error
        self.valid = True
        self.expired = False

    @classmethod
    def from_authorized_user_file(cls, path, scopes=None):
        if cls.from_authorized_user_file_impl is not None:
            return cls.from_authorized_user_file_impl(path, scopes=scopes)
        return cls()


class FakeFlow:
    created = []
    default_state = "generated-state"
    default_verifier = "generated-code-verifier"
    credentials_payload = None
    fetch_error = None

    def __init__(
        self,
        client_secrets_file,
        scopes,
        *,
        redirect_uri=None,
        state=None,
        code_verifier=None,
        autogenerate_code_verifier=False,
    ):
        self.client_secrets_file = client_secrets_file
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        self.state = state
        self.code_verifier = code_verifier
        self.autogenerate_code_verifier = autogenerate_code_verifier
        self.authorization_kwargs = None
        self.fetch_token_calls = []
        self.credentials = FakeCredentials(self.credentials_payload)

        if autogenerate_code_verifier and not self.code_verifier:
            self.code_verifier = self.default_verifier
        if not self.state:
            self.state = self.default_state

    @classmethod
    def reset(cls):
        cls.created = []
        cls.default_state = "generated-state"
        cls.default_verifier = "generated-code-verifier"
        cls.credentials_payload = None
        cls.fetch_error = None

    @classmethod
    def from_client_secrets_file(cls, client_secrets_file, scopes, **kwargs):
        inst = cls(client_secrets_file, scopes, **kwargs)
        cls.created.append(inst)
        return inst

    def authorization_url(self, **kwargs):
        self.authorization_kwargs = kwargs
        return f"https://auth.example/authorize?state={self.state}", self.state

    def fetch_token(self, **kwargs):
        self.fetch_token_calls.append(kwargs)
        if self.fetch_error:
            raise self.fetch_error


class FakeHttpError(Exception):
    def __init__(self, status, body):
        super().__init__(body)
        self.resp = SimpleNamespace(status=status)


@pytest.fixture
def setup_module(monkeypatch, tmp_path):
    FakeFlow.reset()
    FakeCredentials.from_authorized_user_file_impl = None

    google_auth_module = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    google_auth_module.flow = flow_module
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", google_auth_module)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    google_module = types.ModuleType("google")
    google_auth_pkg = types.ModuleType("google.auth")
    google_auth_exceptions = types.ModuleType("google.auth.exceptions")

    class FakeRefreshError(Exception):
        pass

    google_auth_exceptions.RefreshError = FakeRefreshError
    google_auth_transport = types.ModuleType("google.auth.transport")
    google_auth_transport_requests = types.ModuleType("google.auth.transport.requests")

    class FakeRequest:
        pass

    google_auth_transport_requests.Request = FakeRequest
    google_auth_transport.requests = google_auth_transport_requests
    google_auth_pkg.exceptions = google_auth_exceptions
    google_auth_pkg.transport = google_auth_transport

    google_oauth2_pkg = types.ModuleType("google.oauth2")
    google_oauth2_credentials = types.ModuleType("google.oauth2.credentials")
    google_oauth2_credentials.Credentials = FakeCredentials
    google_oauth2_pkg.credentials = google_oauth2_credentials

    googleapiclient_module = types.ModuleType("googleapiclient")
    googleapiclient_discovery = types.ModuleType("googleapiclient.discovery")
    googleapiclient_errors = types.ModuleType("googleapiclient.errors")
    googleapiclient_errors.HttpError = FakeHttpError

    def _default_build(*_args, **_kwargs):
        raise AssertionError("googleapiclient.discovery.build not stubbed for this test")

    googleapiclient_discovery.build = _default_build
    googleapiclient_module.discovery = googleapiclient_discovery
    googleapiclient_module.errors = googleapiclient_errors

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth_pkg)
    monkeypatch.setitem(sys.modules, "google.auth.exceptions", google_auth_exceptions)
    monkeypatch.setitem(sys.modules, "google.auth.transport", google_auth_transport)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", google_auth_transport_requests)
    monkeypatch.setitem(sys.modules, "google.oauth2", google_oauth2_pkg)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", google_oauth2_credentials)
    monkeypatch.setitem(sys.modules, "googleapiclient", googleapiclient_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", googleapiclient_discovery)
    monkeypatch.setitem(sys.modules, "googleapiclient.errors", googleapiclient_errors)

    spec = importlib.util.spec_from_file_location("google_workspace_setup_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_ensure_deps", lambda: None)
    monkeypatch.setattr(module, "CLIENT_SECRET_PATH", tmp_path / "google_client_secret.json")
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "google_token.json")
    monkeypatch.setattr(module, "PENDING_AUTH_PATH", tmp_path / "google_oauth_pending.json", raising=False)

    client_secret = {
        "installed": {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    module.CLIENT_SECRET_PATH.write_text(json.dumps(client_secret))
    return module


def _make_fake_creds(*, valid=True, expired=False, refresh_token="refresh-token", payload=None, refresh_error=None):
    creds = FakeCredentials(payload=payload)
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.refresh_error = refresh_error
    return creds


class TestGetAuthUrl:
    def test_persists_state_and_code_verifier_for_later_exchange(self, setup_module, capsys):
        setup_module.get_auth_url()

        out = capsys.readouterr().out.strip()
        assert out == "https://auth.example/authorize?state=generated-state"

        saved = json.loads(setup_module.PENDING_AUTH_PATH.read_text())
        assert saved["state"] == "generated-state"
        assert saved["code_verifier"] == "generated-code-verifier"

        flow = FakeFlow.created[-1]
        assert flow.autogenerate_code_verifier is True
        assert flow.authorization_kwargs == {"access_type": "offline", "prompt": "consent"}


class TestExchangeAuthCode:
    def test_reuses_saved_pkce_material_for_plain_code(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        setup_module.exchange_auth_code("4/test-auth-code")

        flow = FakeFlow.created[-1]
        assert flow.state == "saved-state"
        assert flow.code_verifier == "saved-verifier"
        assert flow.fetch_token_calls == [{"code": "4/test-auth-code"}]
        saved = json.loads(setup_module.TOKEN_PATH.read_text())
        assert saved["token"] == "access-token"
        assert saved["type"] == "authorized_user"
        assert not setup_module.PENDING_AUTH_PATH.exists()

    def test_extracts_code_from_redirect_url_and_checks_state(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        setup_module.exchange_auth_code(
            "http://localhost:1/?code=4/extracted-code&state=saved-state&scope=gmail"
        )

        flow = FakeFlow.created[-1]
        assert flow.fetch_token_calls == [{"code": "4/extracted-code"}]

    def test_passes_scopes_from_redirect_url_to_flow(self, setup_module):
        """Callback URL carries space-delimited scope list; Flow must receive it (not full SCOPES)."""
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )
        g1 = "https://www.googleapis.com/auth/gmail.readonly"
        g2 = "https://www.googleapis.com/auth/calendar"
        from urllib.parse import quote

        scope_q = quote(f"{g1} {g2}", safe="")
        setup_module.exchange_auth_code(
            f"http://localhost:1/?code=4/extracted-code&state=saved-state&scope={scope_q}"
        )
        flow = FakeFlow.created[-1]
        assert flow.scopes == [g1, g2]

    def test_rejects_state_mismatch(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code(
                "http://localhost:1/?code=4/extracted-code&state=wrong-state"
            )

        out = capsys.readouterr().out
        assert "state mismatch" in out.lower()
        assert not setup_module.TOKEN_PATH.exists()

    def test_requires_pending_auth_session(self, setup_module, capsys):
        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "run --auth-url first" in out.lower()
        assert not setup_module.TOKEN_PATH.exists()

    def test_keeps_pending_auth_session_when_exchange_fails(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )
        FakeFlow.fetch_error = Exception("invalid_grant: Missing code verifier")

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "token exchange failed" in out.lower()
        assert setup_module.PENDING_AUTH_PATH.exists()
        assert not setup_module.TOKEN_PATH.exists()

    def test_accepts_narrower_scopes_with_warning(self, setup_module, capsys):
        """Partial scopes are accepted with a warning (gws migration: v2.0)."""
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "***", "scopes": setup_module.SCOPES}))
        FakeFlow.credentials_payload = {
            "token": "***",
            "refresh_token": "***",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        }

        setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "warning" in out.lower()
        assert "missing" in out.lower()
        # Token is saved (partial scopes accepted)
        assert setup_module.TOKEN_PATH.exists()
        # Pending auth is cleaned up
        assert not setup_module.PENDING_AUTH_PATH.exists()


class TestCheckAuth:
    def test_extract_oauth_error_code_uses_structured_refresh_error(self, setup_module):
        refresh_error = sys.modules["google.auth.exceptions"].RefreshError(
            "misleading invalid_client text",
            {"error": "invalid_grant"},
        )

        assert setup_module._extract_oauth_error_code(refresh_error) == "invalid_grant"

    def test_extract_oauth_error_code_does_not_match_compound_codes(self, setup_module):
        assert setup_module._extract_oauth_error_code(Exception("invalid_grant_type: nope")) == ""

    def test_check_auth_refresh_disabled_client_prints_guidance(self, setup_module, capsys):
        refresh_error = sys.modules["google.auth.exceptions"].RefreshError(
            "disabled_client: The OAuth client was disabled.",
            {"error": "disabled_client", "error_description": "The OAuth client was disabled."},
        )
        creds = _make_fake_creds(valid=False, expired=True, refresh_error=refresh_error)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        assert setup_module.check_auth() is False

        out = capsys.readouterr().out
        assert "OAUTH_CLIENT_DISABLED:" in out
        assert "accounts.google.com/signin/recovery" in out

    def test_check_auth_refresh_invalid_grant_prints_reauth(self, setup_module, capsys):
        refresh_error = sys.modules["google.auth.exceptions"].RefreshError(
            "invalid_grant: Token has been expired or revoked.",
            {"error": "invalid_grant", "error_description": "Token has been expired or revoked."},
        )
        creds = _make_fake_creds(valid=False, expired=True, refresh_error=refresh_error)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        assert setup_module.check_auth() is False

        out = capsys.readouterr().out
        assert "TOKEN_REVOKED:" in out
        assert "Re-run setup" in out

    def test_check_auth_refresh_generic_error_falls_through(self, setup_module, capsys):
        creds = _make_fake_creds(valid=False, expired=True, refresh_error=Exception("something else"))
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        assert setup_module.check_auth() is False

        out = capsys.readouterr().out
        assert "REFRESH_FAILED:" in out
        assert "OAUTH_CLIENT_DISABLED" not in out
        assert "TOKEN_REVOKED" not in out


class TestCheckAuthLive:
    def test_check_auth_live_success(self, setup_module, monkeypatch, capsys):
        creds = _make_fake_creds(valid=True, expired=False)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        class _FakeCalendarList:
            def list(self, **_kwargs):
                return self

            def execute(self):
                return {"items": []}

        class _FakeService:
            def calendarList(self):
                return _FakeCalendarList()

        monkeypatch.setattr(sys.modules["googleapiclient.discovery"], "build", lambda *_a, **_k: _FakeService())

        assert setup_module.check_auth_live() is True

        out = capsys.readouterr().out
        assert "LIVE_CHECK_OK:" in out
        assert "AUTHENTICATED:" not in out

    def test_check_auth_live_disabled_client(self, setup_module, monkeypatch, capsys):
        creds = _make_fake_creds(valid=True, expired=False)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        class _FakeCalendarList:
            def list(self, **_kwargs):
                return self

            def execute(self):
                raise sys.modules["google.auth.exceptions"].RefreshError(
                    "disabled_client: OAuth client disabled",
                    {"error": "disabled_client"},
                )

        class _FakeService:
            def calendarList(self):
                return _FakeCalendarList()

        monkeypatch.setattr(sys.modules["googleapiclient.discovery"], "build", lambda *_a, **_k: _FakeService())

        assert setup_module.check_auth_live() is False

        out = capsys.readouterr().out
        assert "LIVE_CHECK_FAILED:" in out
        assert "OAuth client or account disabled" in out

    def test_check_auth_live_partial_scope_should_not_panic(self, setup_module, monkeypatch, capsys):
        creds = _make_fake_creds(valid=True, expired=False)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        class _FakeCalendarList:
            def list(self, **_kwargs):
                return self

            def execute(self):
                raise FakeHttpError(
                    403,
                    "HttpError 403 when requesting https://www.googleapis.com/calendar/v3/users/me/calendarList "
                    "returned \"Request had insufficient authentication scopes. "
                    "[reason: ACCESS_TOKEN_SCOPE_INSUFFICIENT]\"",
                )

        class _FakeService:
            def calendarList(self):
                return _FakeCalendarList()

        monkeypatch.setattr(sys.modules["googleapiclient.discovery"], "build", lambda *_a, **_k: _FakeService())

        assert setup_module.check_auth_live() is True

        out = capsys.readouterr().out
        assert "LIVE_CHECK_PARTIAL:" in out
        assert "disabled" not in out.lower()

    def test_check_auth_live_generic_403_still_fails(self, setup_module, monkeypatch, capsys):
        creds = _make_fake_creds(valid=True, expired=False)
        FakeCredentials.from_authorized_user_file_impl = lambda _path, scopes=None: creds
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "x", "refresh_token": "r"}))

        class _FakeCalendarList:
            def list(self, **_kwargs):
                return self

            def execute(self):
                raise FakeHttpError(403, "HttpError 403: access forbidden by policy")

        class _FakeService:
            def calendarList(self):
                return _FakeCalendarList()

        monkeypatch.setattr(sys.modules["googleapiclient.discovery"], "build", lambda *_a, **_k: _FakeService())

        assert setup_module.check_auth_live() is False

        out = capsys.readouterr().out
        assert "LIVE_CHECK_FAILED: HTTP 403" in out
        assert "LIVE_CHECK_PARTIAL" not in out


class TestHermesConstantsFallback:
    """Tests for _hermes_home.py fallback when hermes_constants is unavailable."""

    HELPER_PATH = (
        Path(__file__).resolve().parents[2]
        / "skills/productivity/google-workspace/scripts/_hermes_home.py"
    )

    def _load_helper(self, monkeypatch):
        """Load _hermes_home.py with hermes_constants blocked."""
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        spec = importlib.util.spec_from_file_location("_hermes_home_test", self.HELPER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_fallback_uses_hermes_home_env_var(self, monkeypatch, tmp_path):
        """When hermes_constants is missing, HERMES_HOME comes from env var."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-hermes"))
        module = self._load_helper(monkeypatch)
        assert module.get_hermes_home() == tmp_path / "custom-hermes"

    def test_fallback_defaults_to_dot_hermes(self, monkeypatch):
        """When hermes_constants is missing and HERMES_HOME unset, default to ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        module = self._load_helper(monkeypatch)
        assert module.get_hermes_home() == Path.home() / ".hermes"

    def test_fallback_ignores_empty_hermes_home(self, monkeypatch):
        """Empty/whitespace HERMES_HOME is treated as unset."""
        monkeypatch.setenv("HERMES_HOME", "  ")
        module = self._load_helper(monkeypatch)
        assert module.get_hermes_home() == Path.home() / ".hermes"

    def test_fallback_display_hermes_home_shortens_path(self, monkeypatch):
        """Fallback display_hermes_home() uses ~/ shorthand like the real one."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        module = self._load_helper(monkeypatch)
        assert module.display_hermes_home() == "~/.hermes"

    def test_fallback_display_hermes_home_profile_path(self, monkeypatch):
        """Fallback display_hermes_home() handles profile paths under ~/."""
        monkeypatch.setenv("HERMES_HOME", str(Path.home() / ".hermes/profiles/coder"))
        module = self._load_helper(monkeypatch)
        assert module.display_hermes_home() == "~/.hermes/profiles/coder"

    def test_fallback_display_hermes_home_custom_path(self, monkeypatch):
        """Fallback display_hermes_home() returns full path for non-home locations."""
        monkeypatch.setenv("HERMES_HOME", "/opt/hermes-custom")
        module = self._load_helper(monkeypatch)
        assert module.display_hermes_home() == "/opt/hermes-custom"

    def test_delegates_to_hermes_constants_when_available(self):
        """When hermes_constants IS importable, _hermes_home delegates to it."""
        spec = importlib.util.spec_from_file_location(
            "_hermes_home_happy", self.HELPER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        import hermes_constants
        assert module.get_hermes_home is hermes_constants.get_hermes_home
        assert module.display_hermes_home is hermes_constants.display_hermes_home
