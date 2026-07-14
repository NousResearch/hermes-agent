"""Tests for Google Workspace named auth contexts."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "skills/productivity/google-workspace/scripts"
AUTH_CONTEXTS_PATH = SCRIPT_DIR / "auth_contexts.py"
API_PATH = SCRIPT_DIR / "google_api.py"
SETUP_PATH = SCRIPT_DIR / "setup.py"


def load_module(name: str, path: Path, monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auth_contexts(monkeypatch, tmp_path):
    return load_module("auth_contexts_test", AUTH_CONTEXTS_PATH, monkeypatch, tmp_path)


class TestContextNames:
    @pytest.mark.parametrize("name", ["default", "gmail-readonly", "workspace.writer", "x_1"])
    def test_accepts_path_safe_names(self, auth_contexts, name):
        assert auth_contexts.validate_context_name(name) == name

    @pytest.mark.parametrize("name", ["", ".", "..", "../x", "x/y", "x\\y", " space"])
    def test_rejects_unsafe_names(self, auth_contexts, name):
        with pytest.raises(ValueError):
            auth_contexts.validate_context_name(name)


class TestScopeResolution:
    def test_gmail_readonly_is_least_privilege(self, auth_contexts):
        assert auth_contexts.resolve_scopes(services="gmail-readonly") == [
            "https://www.googleapis.com/auth/gmail.readonly"
        ]

    def test_workspace_writer_services(self, auth_contexts):
        assert auth_contexts.resolve_scopes(services="drive,calendar,docs,sheets") == [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
        ]

    def test_explicit_scopes_override_services(self, auth_contexts):
        assert auth_contexts.resolve_scopes(
            services="all",
            scopes="https://www.googleapis.com/auth/calendar.readonly,https://www.googleapis.com/auth/drive.file",
        ) == [
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/drive.file",
        ]

    def test_unknown_service_fails(self, auth_contexts):
        with pytest.raises(ValueError):
            auth_contexts.resolve_scopes(services="gmail-readonly,photos")


class TestStoreAndDefaults:
    def test_context_store_round_trip(self, auth_contexts):
        auth_contexts.set_client_secret("gmail-readonly", {"installed": {"client_id": "id"}}, account_hint="mail@example.com")
        auth_contexts.set_token_payload(
            "gmail-readonly",
            {"token": "tok", "scopes": [auth_contexts.GMAIL_READONLY]},
            services=["gmail-readonly"],
            requested_scopes=[auth_contexts.GMAIL_READONLY],
        )
        data = json.loads(auth_contexts.store_path().read_text())
        assert data["contexts"]["gmail-readonly"]["account_hint"] == "mail@example.com"
        assert auth_contexts.get_token_payload("gmail-readonly")["token"] == "tok"

    def test_default_context_routing(self, auth_contexts):
        auth_contexts.set_client_secret("workspace-writer", {"installed": {"client_id": "id"}})
        auth_contexts.set_default_for_services("workspace-writer", "drive,calendar")
        assert auth_contexts.default_context_for_service("drive") == "workspace-writer"
        assert auth_contexts.default_context_for_service("calendar") == "workspace-writer"
        assert auth_contexts.default_context_for_service("gmail") == "default"

    def test_legacy_default_context_reads_old_files(self, auth_contexts):
        auth_contexts.legacy_token_path().write_text(json.dumps({"token": "legacy"}))
        auth_contexts.legacy_client_secret_path().write_text(json.dumps({"installed": {"client_id": "id"}}))
        assert auth_contexts.get_token_payload("default")["token"] == "legacy"
        assert auth_contexts.materialize_token_file("default") == auth_contexts.legacy_token_path()


class TestCommandScopeGate:
    def test_allows_readonly_gmail_read(self, auth_contexts):
        auth_contexts.set_token_payload("gmail-readonly", {"token": "tok", "scopes": [auth_contexts.GMAIL_READONLY]})
        auth_contexts.assert_command_allowed("gmail-readonly", "gmail", "search")

    def test_blocks_gmail_send_with_readonly_context(self, auth_contexts):
        auth_contexts.set_token_payload("gmail-readonly", {"token": "tok", "scopes": [auth_contexts.GMAIL_READONLY]})
        with pytest.raises(PermissionError):
            auth_contexts.assert_command_allowed("gmail-readonly", "gmail", "send")

    def test_gmail_reply_requires_send_and_read_scopes(self, auth_contexts):
        auth_contexts.set_token_payload(
            "send-only",
            {"token": "tok", "scopes": [auth_contexts.GMAIL_SEND]},
        )
        with pytest.raises(PermissionError):
            auth_contexts.assert_command_allowed("send-only", "gmail", "reply")

        auth_contexts.set_token_payload(
            "send-and-read",
            {
                "token": "tok",
                "scopes": [auth_contexts.GMAIL_SEND, auth_contexts.GMAIL_READONLY],
            },
        )
        auth_contexts.assert_command_allowed("send-and-read", "gmail", "reply")

    def test_blocks_named_context_when_scope_metadata_missing(self, auth_contexts):
        auth_contexts.set_token_payload("unknown", {"token": "tok"})
        with pytest.raises(PermissionError):
            auth_contexts.assert_command_allowed("unknown", "gmail", "send")

    def test_auth_store_is_private(self, auth_contexts):
        auth_contexts.set_token_payload("gmail-readonly", {"token": "tok", "scopes": [auth_contexts.GMAIL_READONLY]})
        mode = auth_contexts.store_path().stat().st_mode & 0o777
        assert mode == 0o600

    def test_delete_token_removes_materialized_cache_file(self, auth_contexts):
        auth_contexts.set_token_payload("gmail-readonly", {"token": "tok", "scopes": [auth_contexts.GMAIL_READONLY]})
        path = auth_contexts.materialize_token_file("gmail-readonly")
        assert path.exists()
        auth_contexts.delete_token("gmail-readonly")
        assert not path.exists()

    def test_named_store_migrates_and_preserves_legacy_default(self, auth_contexts):
        auth_contexts.legacy_token_path().write_text('{"token": "legacy"}')
        auth_contexts.legacy_client_secret_path().write_text(
            '{"installed": {"client_id": "legacy"}}'
        )
        auth_contexts.legacy_pending_path().write_text('{"state": "legacy"}')

        auth_contexts.set_client_secret(
            "named",
            {"installed": {"client_id": "named"}},
        )

        assert auth_contexts.get_token_payload("default")["token"] == "legacy"
        assert auth_contexts.get_client_secret("default")["installed"]["client_id"] == "legacy"
        assert auth_contexts.get_pending_auth("default")["state"] == "legacy"
        assert auth_contexts.context_exists("default")
        assert "default" in auth_contexts.list_contexts()

        store = auth_contexts.load_store()
        assert store["legacy_default_migrated"] is True
        stored_default = store["contexts"]["default"]
        assert stored_default["token"]["token"] == "legacy"
        assert stored_default["client_secret"]["installed"]["client_id"] == "legacy"
        assert stored_default["pending_auth"]["state"] == "legacy"

    def test_store_backed_default_remains_authoritative_after_migration(self, auth_contexts):
        auth_contexts.legacy_token_path().write_text('{"token": "legacy"}')
        auth_contexts.set_client_secret(
            "named",
            {"installed": {"client_id": "named"}},
        )
        auth_contexts.set_token_payload(
            "default",
            {"token": "store", "scopes": [auth_contexts.GMAIL_READONLY]},
        )
        auth_contexts.legacy_token_path().write_text('{"token": "stale"}')

        assert auth_contexts.get_token_payload("default")["token"] == "store"

    def test_delete_migrated_default_token_does_not_restore_legacy(self, auth_contexts):
        auth_contexts.legacy_token_path().write_text('{"token": "legacy"}')
        auth_contexts.legacy_pending_path().write_text('{"state": "pending"}')
        auth_contexts.set_client_secret(
            "named",
            {"installed": {"client_id": "named"}},
        )
        assert auth_contexts.get_token_payload("default")["token"] == "legacy"

        auth_contexts.delete_token("default")

        assert auth_contexts.get_token_payload("default") == {}
        assert auth_contexts.get_pending_auth("default") == {}
        assert not auth_contexts.legacy_token_path().exists()
        assert not auth_contexts.legacy_pending_path().exists()
        stored_default = auth_contexts.load_store()["contexts"]["default"]
        assert "token" not in stored_default
        assert "pending_auth" not in stored_default

    def test_clear_migrated_default_pending_does_not_restore_legacy(self, auth_contexts):
        auth_contexts.legacy_token_path().write_text('{"token": "legacy"}')
        auth_contexts.legacy_pending_path().write_text('{"state": "pending"}')
        auth_contexts.set_client_secret(
            "named",
            {"installed": {"client_id": "named"}},
        )
        assert auth_contexts.get_pending_auth("default")["state"] == "pending"

        auth_contexts.clear_pending_auth("default")

        assert auth_contexts.get_pending_auth("default") == {}
        assert auth_contexts.get_token_payload("default")["token"] == "legacy"
        assert not auth_contexts.legacy_pending_path().exists()
        assert "pending_auth" not in auth_contexts.load_store()["contexts"]["default"]

    def test_full_drive_implies_read(self, auth_contexts):
        auth_contexts.set_token_payload("drive", {"token": "tok", "scopes": [auth_contexts.DRIVE]})
        auth_contexts.assert_command_allowed("drive", "drive", "search")


def test_live_check_uses_scope_neutral_oauth_refresh(monkeypatch, tmp_path, capsys):
    setup_script = load_module("google_workspace_setup_live_test", SETUP_PATH, monkeypatch, tmp_path)
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")
    monkeypatch.setattr(setup_script, "check_auth", lambda **_kwargs: True)
    monkeypatch.setattr(setup_script, "_token_file", lambda _context: token_file)
    monkeypatch.setattr(
        setup_script,
        "_token_payload",
        lambda _context: {
            "token": "old",
            "refresh_token": "refresh",
            "scopes": [setup_script.gauth.GMAIL_SEND],
        },
    )
    saved = {}
    monkeypatch.setattr(
        setup_script,
        "_set_token_payload",
        lambda context, payload, **_kwargs: saved.update(
            {"context": context, "payload": payload}
        ),
    )

    class FakeCredentials:
        refresh_token = "refresh"

        @classmethod
        def from_authorized_user_file(cls, filename):
            assert filename == str(token_file)
            return cls()

        def refresh(self, _request):
            self.refreshed = True

        def to_json(self):
            assert self.refreshed
            return json.dumps(
                {
                    "token": "refreshed",
                    "refresh_token": "refresh",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "client_id": "client",
                    "client_secret": "secret",
                }
            )

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    setattr(credentials_module, "Credentials", FakeCredentials)
    auth_module = types.ModuleType("google.auth")
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    setattr(requests_module, "Request", lambda: object())
    googleapiclient_module = types.ModuleType("googleapiclient")
    discovery_module = types.ModuleType("googleapiclient.discovery")

    def reject_service_specific_call(*_args, **_kwargs):
        raise AssertionError("live check must not require a Calendar or other service scope")

    setattr(discovery_module, "build", reject_service_specific_call)
    for name, module in {
        "google": google_module,
        "google.oauth2": oauth2_module,
        "google.oauth2.credentials": credentials_module,
        "google.auth": auth_module,
        "google.auth.transport": transport_module,
        "google.auth.transport.requests": requests_module,
        "googleapiclient": googleapiclient_module,
        "googleapiclient.discovery": discovery_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert setup_script.check_auth_live("gmail-send") is True
    assert saved["context"] == "gmail-send"
    assert saved["payload"]["token"] == "refreshed"
    assert saved["payload"]["scopes"] == [setup_script.gauth.GMAIL_SEND]
    assert "LIVE_CHECK_OK" in capsys.readouterr().out
