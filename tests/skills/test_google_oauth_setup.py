"""Regression tests for Google Workspace OAuth setup.

These tests cover the headless/manual auth-code flow where the browser step and
code exchange happen in separate process invocations.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


class FakeCredentials:
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

    def to_json(self):
        return json.dumps(self._payload)


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


@pytest.fixture
def setup_module(monkeypatch, tmp_path):
    FakeFlow.reset()

    google_auth_module = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    google_auth_module.flow = flow_module
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", google_auth_module)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

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
        assert module.get_default_hermes_root is hermes_constants.get_default_hermes_root


class TestClientSecretDefaultRoot:
    """Regression tests for anchoring the shared OAuth client secret.

    Sibling of commit ``fff056144`` (google_chat): the host-wide
    ``google_client_secret.json`` must be visible to a gateway running under a
    named profile (``HERMES_HOME=<root>/profiles/<name>``), so it resolves to
    the default Hermes root unless a profile-local copy is present.
    """

    SCRIPT_PATH = (
        Path(__file__).resolve().parents[2]
        / "skills/productivity/google-workspace/scripts/setup.py"
    )

    def _load_setup(self, monkeypatch):
        """Load setup.py with hermes_constants blocked so the stdlib
        ``get_default_hermes_root`` fallback (the path this fix adds) runs."""
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        # setup.py imports google_auth_oauthlib lazily inside _ensure_deps, so
        # module import itself only needs _hermes_home on sys.path (added by
        # the script's own bootstrap).
        spec = importlib.util.spec_from_file_location(
            "google_workspace_setup_secret_test", self.SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_client_secret_shared_across_profiles(self, monkeypatch, tmp_path):
        """A profile gateway resolves the client secret at the default root."""
        root = tmp_path / ".hermes"
        (root / "profiles" / "bot1").mkdir(parents=True)
        (root / "google_client_secret.json").write_text("{}")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "bot1"))

        module = self._load_setup(monkeypatch)
        assert module._client_secret_path() == root / "google_client_secret.json"

    def test_profile_local_client_secret_takes_precedence(self, monkeypatch, tmp_path):
        """A profile-local copy wins over the default-root secret when present."""
        root = tmp_path / ".hermes"
        profile = root / "profiles" / "bot1"
        profile.mkdir(parents=True)
        (root / "google_client_secret.json").write_text("{}")
        (profile / "google_client_secret.json").write_text("{}")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(profile))

        module = self._load_setup(monkeypatch)
        assert module._client_secret_path() == profile / "google_client_secret.json"

    def test_default_root_used_when_no_profile_local(self, monkeypatch, tmp_path):
        """With no profile-local copy, the resolver returns the default-root path."""
        root = tmp_path / ".hermes"
        profile = root / "profiles" / "bot1"
        profile.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(profile))

        module = self._load_setup(monkeypatch)
        assert module._client_secret_path() == root / "google_client_secret.json"

    def test_profile_local_directory_does_not_shadow_default_root(
        self, monkeypatch, tmp_path
    ):
        """A *directory* named google_client_secret.json must not win precedence.

        ``exists()`` matches directories too, so a same-named directory in the
        profile dir would shadow the real default-root secret and break the
        OAuth flow at file-open time. The resolver gates on ``is_file()``, so it
        skips the directory and falls back to the default-root file.
        """
        root = tmp_path / ".hermes"
        profile = root / "profiles" / "bot1"
        profile.mkdir(parents=True)
        # Real secret lives at the default root.
        (root / "google_client_secret.json").write_text("{}")
        # A *directory* of the same name sits in the profile dir.
        (profile / "google_client_secret.json").mkdir()

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(profile))

        module = self._load_setup(monkeypatch)
        assert module._client_secret_path() == root / "google_client_secret.json"
