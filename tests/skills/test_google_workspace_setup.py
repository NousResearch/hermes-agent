"""Security-floor tests for the Google Workspace runtime installer."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)

CONTACTS_READONLY = "https://www.googleapis.com/auth/contacts.readonly"
CONTACTS_WRITE = "https://www.googleapis.com/auth/contacts"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_WRITE = "https://www.googleapis.com/auth/drive"
CALENDAR_READONLY = "https://www.googleapis.com/auth/calendar.readonly"
CALENDAR_WRITE = "https://www.googleapis.com/auth/calendar"
GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND = "https://www.googleapis.com/auth/gmail.send"
GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"


class FakeCredentials:
    def __init__(self, payload, granted_scopes=None):
        self._payload = payload
        self.granted_scopes = granted_scopes

    def to_json(self):
        return json.dumps(self._payload)


class FakeFlow:
    created = []
    credentials_payload = None
    granted_scopes = None
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
        self.state = state or "generated-state"
        self.code_verifier = code_verifier or "generated-code-verifier"
        self.autogenerate_code_verifier = autogenerate_code_verifier
        self.authorization_kwargs = None
        self.fetch_token_calls = []
        payload = self.credentials_payload or {
            "token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": list(scopes),
        }
        self.credentials = FakeCredentials(payload, self.granted_scopes)

    @classmethod
    def reset(cls):
        cls.created = []
        cls.credentials_payload = None
        cls.granted_scopes = None
        cls.fetch_error = None

    @classmethod
    def from_client_secrets_file(cls, client_secrets_file, scopes, **kwargs):
        flow = cls(client_secrets_file, scopes, **kwargs)
        cls.created.append(flow)
        return flow

    def authorization_url(self, **kwargs):
        self.authorization_kwargs = kwargs
        query = urlencode({"state": self.state, **kwargs})
        return f"https://auth.example/authorize?{query}", self.state

    def fetch_token(self, **kwargs):
        self.fetch_token_calls.append(kwargs)
        if self.fetch_error:
            raise self.fetch_error


@pytest.fixture()
def setup_module(monkeypatch, tmp_path):
    FakeFlow.reset()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    google_auth_module = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    google_auth_module.flow = flow_module
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", google_auth_module)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_ensure_deps", lambda: None)
    module.CLIENT_SECRET_PATH.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                }
            }
        ),
        encoding="utf-8",
    )
    return module


def _write_pending(module, *, state="saved-state"):
    module.PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": "saved-code-verifier",
                "redirect_uri": module.REDIRECT_URI,
            }
        ),
        encoding="utf-8",
    )


def _new_token(scopes):
    return {
        "token": "new-access-token",
        "refresh_token": "new-refresh-token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": scopes,
    }


def test_stale_google_transitives_are_reported_missing(setup_module, monkeypatch):
    installed = {
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(setup_module, "_distribution_version", fake_version)

    assert setup_module._missing_required_packages() == [
        "google-auth==2.55.1",
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ]


def test_installer_repairs_stale_transitives(setup_module, monkeypatch):
    states = iter(
        [
            [
                "google-auth==2.55.1",
                "httplib2==0.32.0",
                "pyasn1==0.6.4",
            ],
            [],
        ]
    )
    monkeypatch.setattr(
        setup_module,
        "_missing_required_packages",
        lambda: next(states),
    )
    calls = []
    monkeypatch.setattr(
        setup_module.subprocess,
        "check_call",
        lambda argv, **kwargs: calls.append(argv),
    )

    assert setup_module.install_deps() is True
    assert calls == [
        [
            setup_module.sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "google-auth==2.55.1",
            "httplib2==0.32.0",
            "pyasn1==0.6.4",
        ]
    ]


def test_contacts_write_satisfies_contacts_readonly(setup_module):
    missing = setup_module._missing_scopes_from_payload(
        {"scopes": [CONTACTS_WRITE]}
    )
    assert CONTACTS_READONLY not in missing


@pytest.mark.parametrize(
    ("broader", "narrower"),
    [
        (GMAIL_MODIFY, GMAIL_READONLY),
        (GMAIL_MODIFY, GMAIL_SEND),
        (DRIVE_WRITE, DRIVE_READONLY),
        (CALENDAR_WRITE, CALENDAR_READONLY),
    ],
)
def test_known_broader_scopes_satisfy_narrower_capabilities(
    setup_module, broader, narrower
):
    effective = setup_module._effective_scope_capabilities([broader])
    assert narrower in effective


def test_write_to_readonly_downgrade_preserves_existing_bytes(
    setup_module, capsys
):
    old_bytes = (
        b'{"token":"old-access-token","scopes":["'
        + CONTACTS_WRITE.encode()
        + b'"]}\n'
    )
    setup_module.TOKEN_PATH.write_bytes(old_bytes)
    _write_pending(setup_module)
    FakeFlow.credentials_payload = _new_token([CONTACTS_READONLY])

    with pytest.raises(SystemExit) as exc:
        setup_module.exchange_auth_code("new-auth-code")

    assert exc.value.code == 1
    assert setup_module.TOKEN_PATH.read_bytes() == old_bytes
    assert not setup_module.PENDING_AUTH_PATH.exists()
    assert "SCOPE_DOWNGRADE_BLOCKED" in capsys.readouterr().out


def test_readonly_to_write_upgrade_saves(setup_module):
    setup_module.TOKEN_PATH.write_text(
        json.dumps({"token": "old-access-token", "scopes": [CONTACTS_READONLY]}),
        encoding="utf-8",
    )
    _write_pending(setup_module)
    FakeFlow.credentials_payload = _new_token([CONTACTS_WRITE])

    setup_module.exchange_auth_code("new-auth-code")

    saved = json.loads(setup_module.TOKEN_PATH.read_text(encoding="utf-8"))
    assert saved["scopes"] == [CONTACTS_WRITE]
    assert not setup_module.PENDING_AUTH_PATH.exists()


def test_partial_token_without_existing_token_saves_with_warning(
    setup_module, capsys
):
    _write_pending(setup_module)
    FakeFlow.credentials_payload = _new_token([DRIVE_READONLY])

    setup_module.exchange_auth_code("new-auth-code")

    assert setup_module.TOKEN_PATH.exists()
    assert "WARNING" in capsys.readouterr().out
    assert not setup_module.PENDING_AUTH_PATH.exists()


def test_explicit_flag_permits_scope_downgrade(setup_module, monkeypatch):
    setup_module.TOKEN_PATH.write_text(
        json.dumps({"token": "old-access-token", "scopes": [CONTACTS_WRITE]}),
        encoding="utf-8",
    )
    _write_pending(setup_module)
    FakeFlow.credentials_payload = _new_token([CONTACTS_READONLY])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup.py",
            "--auth-code",
            "new-auth-code",
            "--allow-scope-downgrade",
        ],
    )
    setup_module.main()

    saved = json.loads(setup_module.TOKEN_PATH.read_text(encoding="utf-8"))
    assert saved["scopes"] == [CONTACTS_READONLY]


def test_scope_downgrade_flag_is_invalid_without_auth_code(
    setup_module, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["setup.py", "--auth-url", "--allow-scope-downgrade"],
    )

    with pytest.raises(SystemExit) as exc:
        setup_module.main()

    assert exc.value.code == 2
    assert "only valid with --auth-code" in capsys.readouterr().err


def test_callback_scopes_build_flow_and_granted_scopes_are_saved(setup_module):
    _write_pending(setup_module)
    FakeFlow.credentials_payload = _new_token([CONTACTS_READONLY])
    FakeFlow.granted_scopes = [CONTACTS_WRITE]
    callback = (
        "http://localhost:1/?code=new-auth-code&state=saved-state&scope="
        + CONTACTS_READONLY
    )

    setup_module.exchange_auth_code(callback)

    assert FakeFlow.created[-1].scopes == [CONTACTS_READONLY]
    saved = json.loads(setup_module.TOKEN_PATH.read_text(encoding="utf-8"))
    assert saved["scopes"] == [CONTACTS_WRITE]


def test_state_mismatch_consumes_pending_session(setup_module, capsys):
    _write_pending(setup_module)

    with pytest.raises(SystemExit) as exc:
        setup_module.exchange_auth_code(
            "http://localhost:1/?code=secret-code&state=wrong-secret-state"
        )

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "state mismatch" in output.lower()
    assert "secret-code" not in output
    assert "wrong-secret-state" not in output
    assert not setup_module.PENDING_AUTH_PATH.exists()

    with pytest.raises(SystemExit) as retry_exc:
        setup_module.exchange_auth_code("another-code")
    assert retry_exc.value.code == 1
    assert "run --auth-url first" in capsys.readouterr().out.lower()


def test_exchange_failure_redacts_secrets_and_requires_new_url(
    setup_module, capsys
):
    _write_pending(setup_module)
    FakeFlow.fetch_error = RuntimeError(
        "auth-code-secret refresh-token-secret client-secret"
    )

    with pytest.raises(SystemExit) as exc:
        setup_module.exchange_auth_code("auth-code-secret")

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "auth-code-secret" not in output
    assert "refresh-token-secret" not in output
    assert "client-secret" not in output
    assert "run --auth-url" in output.lower()
    assert not setup_module.PENDING_AUTH_PATH.exists()


def test_auth_url_enables_incremental_authorization(setup_module, capsys):
    setup_module.get_auth_url()

    auth_url = capsys.readouterr().out.strip()
    query = parse_qs(urlparse(auth_url).query)
    assert query["include_granted_scopes"] == ["true"]
    assert FakeFlow.created[-1].authorization_kwargs[
        "include_granted_scopes"
    ] == "true"
