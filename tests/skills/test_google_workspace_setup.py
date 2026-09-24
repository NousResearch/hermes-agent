"""Security-floor tests for the Google Workspace runtime installer."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_google_transitives_are_reported_missing(setup_module, monkeypatch):
    """Absent or version-drifted packages are reported; exact matches are not."""
    specs = list(setup_module.REQUIRED_PACKAGES)
    assert len(specs) >= 3
    installed = {spec.partition("==")[0]: spec.partition("==")[2] for spec in specs}
    stale, missing = specs[0], specs[1]
    installed[stale.partition("==")[0]] = "0.0.0-stale"
    del installed[missing.partition("==")[0]]

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(setup_module, "_distribution_version", fake_version)

    assert setup_module._missing_required_packages() == [stale, missing]


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


def test_selected_services_limit_oauth_scopes(setup_module):
    assert hasattr(setup_module, "_scopes_for_services"), (
        "The documented --services option must select OAuth scopes"
    )

    scopes = setup_module._scopes_for_services("email,calendar")

    assert scopes == [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/calendar",
    ]
    with pytest.raises(setup_module.argparse.ArgumentTypeError):
        setup_module._scopes_for_services("all,typo")


def test_private_json_write_enforces_owner_only_mode(setup_module, tmp_path):
    assert hasattr(setup_module, "_write_private_json"), (
        "Google credential state must be written with owner-only permissions"
    )
    path = tmp_path / "credentials.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)

    setup_module._write_private_json(path, {"token": "secret"}, mode=0o600)

    assert json.loads(path.read_text(encoding="utf-8")) == {"token": "secret"}
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_auth_code_file_is_consumed_after_success(setup_module, tmp_path, monkeypatch):
    callback_path = tmp_path / "oauth-callback.txt"
    callback_path.write_text("http://localhost:1/?code=secret", encoding="utf-8")
    received = []
    monkeypatch.setattr(setup_module, "exchange_auth_code", received.append)

    assert hasattr(setup_module, "exchange_auth_code_file"), (
        "OAuth callback credentials must not require command-line arguments"
    )
    setup_module.exchange_auth_code_file(str(callback_path))

    assert received == ["http://localhost:1/?code=secret"]
    assert not callback_path.exists()


def test_live_check_refreshes_without_assuming_calendar_scope(setup_module, monkeypatch):
    refreshed = []
    writes = []

    class FakeCredentials:
        refresh_token = "refresh"

        def refresh(self, request):
            refreshed.append(request)

        def to_json(self):
            return json.dumps({"token": "new", "refresh_token": "refresh"})

    credentials_module = types.ModuleType("google.oauth2.credentials")
    setattr(
        credentials_module,
        "Credentials",
        types.SimpleNamespace(from_authorized_user_file=lambda path: FakeCredentials()),
    )
    requests_module = types.ModuleType("google.auth.transport.requests")
    setattr(requests_module, "Request", object)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)
    monkeypatch.setattr(setup_module, "check_auth", lambda quiet: True)
    monkeypatch.setattr(
        setup_module,
        "_write_private_json",
        lambda path, payload, **kwargs: writes.append((path, payload, kwargs)),
    )

    assert setup_module.check_auth_live() is True
    assert len(refreshed) == 1
    assert writes[0][1]["token"] == "new"
    assert writes[0][2]["mode"] == 0o600
