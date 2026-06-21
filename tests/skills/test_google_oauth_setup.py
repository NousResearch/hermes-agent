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
    monkeypatch.setattr(module, "LAST_AUTH_URL_PATH", tmp_path / "google_oauth_last_url.txt", raising=False)

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

    def test_scopes_auth_url_to_requested_services_and_outputs_json(self, setup_module, capsys):
        setup_module.get_auth_url(services="calendar", output_format="json")

        payload = json.loads(capsys.readouterr().out)
        assert payload["auth_url"] == "https://auth.example/authorize?state=generated-state"
        assert payload["services"] == ["calendar"]
        assert payload["scopes"] == ["https://www.googleapis.com/auth/calendar"]
        assert payload["pending_path"] == str(setup_module.PENDING_AUTH_PATH)
        assert payload["last_auth_url_path"] == str(setup_module.LAST_AUTH_URL_PATH)

        saved = json.loads(setup_module.PENDING_AUTH_PATH.read_text())
        assert saved["requested_scopes"] == ["https://www.googleapis.com/auth/calendar"]
        assert saved["services"] == ["calendar"]
        assert setup_module.LAST_AUTH_URL_PATH.read_text() == payload["auth_url"]

        flow = FakeFlow.created[-1]
        assert flow.scopes == ["https://www.googleapis.com/auth/calendar"]

    def test_main_accepts_documented_auth_url_flags(self, setup_module, monkeypatch):
        calls = []
        monkeypatch.setattr(
            setup_module,
            "get_auth_url",
            lambda services=None, output_format="text": calls.append((services, output_format)),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["setup.py", "--auth-url", "--services", "calendar", "--format", "json"],
        )

        setup_module.main()

        assert calls == [("calendar", "json")]


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

    def test_json_failure_returns_fresh_auth_url_for_same_services(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "requested_scopes": ["https://www.googleapis.com/auth/calendar"],
                    "services": ["calendar"],
                }
            )
        )
        FakeFlow.fetch_error = Exception("invalid_grant: code expired")

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code", output_format="json")

        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "token_exchange_failed"
        assert "invalid_grant" in payload["message"]
        assert payload["fresh_auth_url"] == "https://auth.example/authorize?state=generated-state"
        assert payload["services"] == ["calendar"]
        assert payload["scopes"] == ["https://www.googleapis.com/auth/calendar"]
        assert setup_module.LAST_AUTH_URL_PATH.read_text() == payload["fresh_auth_url"]
        assert json.loads(setup_module.PENDING_AUTH_PATH.read_text())["state"] == "generated-state"

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

    def test_saves_json_success_for_requested_services(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "requested_scopes": ["https://www.googleapis.com/auth/calendar"],
                    "services": ["calendar"],
                }
            )
        )
        FakeFlow.credentials_payload = {
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": ["https://www.googleapis.com/auth/calendar"],
        }

        setup_module.exchange_auth_code("4/test-auth-code", output_format="json")

        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "authenticated"
        assert payload["token_path"] == str(setup_module.TOKEN_PATH)
        assert payload["services"] == ["calendar"]
        assert payload["scopes"] == ["https://www.googleapis.com/auth/calendar"]

        saved = json.loads(setup_module.TOKEN_PATH.read_text())
        assert saved["services"] == ["calendar"]
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


def _load_setup_module(monkeypatch):
    """Load setup.py without stubbing _ensure_deps (for install_deps tests)."""
    spec = importlib.util.spec_from_file_location(
        "google_workspace_setup_installdeps_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _force_deps_missing(monkeypatch):
    """Make `import googleapiclient` / `import google_auth_oauthlib` fail so
    install_deps() proceeds past its early-return short-circuit."""
    for name in ("googleapiclient", "google_auth_oauthlib"):
        monkeypatch.setitem(sys.modules, name, None)


class TestInstallDeps:
    """Tests for install_deps() interpreter/installer selection.

    Regression coverage for the Hermes Docker image, whose venv is built with
    `uv sync` and ships without pip — `sys.executable -m pip install` fails
    with `No module named pip`, so install_deps() must fall back to uv.
    """

    def test_returns_early_when_already_installed(self, monkeypatch):
        """If both libs import, no installer subprocess runs at all."""
        module = _load_setup_module(monkeypatch)
        # Don't force-missing: real test env has the libs importable. Guard
        # against any subprocess being spawned.
        calls = []
        monkeypatch.setattr(
            module.subprocess, "check_call", lambda *a, **k: calls.append(a)
        )
        # google_auth_oauthlib may not be installed in the test env; only run
        # this assertion when the early-return path is actually reachable.
        try:
            import googleapiclient  # noqa: F401
            import google_auth_oauthlib  # noqa: F401
        except ImportError:
            pytest.skip("Google libs not installed in test env")
        assert module.install_deps() is True
        assert calls == []

    def test_uses_pip_when_available(self, monkeypatch):
        """When pip works, install_deps succeeds via pip and never calls uv."""
        module = _load_setup_module(monkeypatch)
        _force_deps_missing(monkeypatch)

        recorded = []

        def fake_check_call(cmd, **kwargs):
            recorded.append(cmd)
            # pip path is the first attempt — succeed.
            return 0

        which_calls = []
        monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
        monkeypatch.setattr(
            module.shutil, "which", lambda name: which_calls.append(name)
        )

        assert module.install_deps() is True
        assert recorded[0][:3] == [module.sys.executable, "-m", "pip"]
        # Control: uv must NOT be consulted when pip succeeds.
        assert which_calls == []

    def test_falls_back_to_uv_when_pip_missing(self, monkeypatch):
        """No pip → uv pip install --python <interpreter> is used."""
        module = _load_setup_module(monkeypatch)
        _force_deps_missing(monkeypatch)

        recorded = []

        def fake_check_call(cmd, **kwargs):
            recorded.append(cmd)
            if cmd[:3] == [module.sys.executable, "-m", "pip"]:
                raise module.subprocess.CalledProcessError(1, cmd)
            return 0  # uv invocation succeeds

        monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
        monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/local/bin/uv")

        assert module.install_deps() is True
        assert len(recorded) == 2
        uv_cmd = recorded[1]
        assert uv_cmd[0] == "/usr/local/bin/uv"
        assert uv_cmd[1:5] == ["pip", "install", "--python", module.sys.executable]
        for pkg in module.REQUIRED_PACKAGES:
            assert pkg in uv_cmd

    def test_returns_false_when_no_pip_and_no_uv(self, monkeypatch, capsys):
        """No pip AND no uv → failure, with the [google] extra hint printed."""
        module = _load_setup_module(monkeypatch)
        _force_deps_missing(monkeypatch)

        def fake_check_call(cmd, **kwargs):
            raise module.subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
        monkeypatch.setattr(module.shutil, "which", lambda name: None)

        assert module.install_deps() is False
        out = capsys.readouterr().out
        assert "hermes-agent[google]" in out

    def test_returns_false_when_uv_fallback_also_fails(self, monkeypatch, capsys):
        """uv present but its install fails → failure surfaced (not swallowed)."""
        module = _load_setup_module(monkeypatch)
        _force_deps_missing(monkeypatch)

        def fake_check_call(cmd, **kwargs):
            raise module.subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(module.subprocess, "check_call", fake_check_call)
        monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/local/bin/uv")

        assert module.install_deps() is False
        out = capsys.readouterr().out
        assert "via uv" in out
