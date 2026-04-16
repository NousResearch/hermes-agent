"""Tests for the ``hermes gmail-push`` CLI surface."""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gmail_push as gp


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _make_args(**kwargs):
    defaults = {
        "gmail_push_action": None,
        "account": "",
        "client_secret": "",
        "auth_code": "",
        "redirect_uri": "",
        "topic": "",
        "subscription": "",
        "public_url": "",
        "audience": "",
        "service_account_email": "",
        "host": "",
        "port": 0,
        "path": "",
        "label_ids": "",
        "renew_every_hours": 0,
        "include_html": False,
        "max_body_chars": 0,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class _FakeCredentials:
    granted_scopes = gp.GMAIL_PUSH_OAUTH_SCOPES

    def to_json(self):
        return json.dumps(
            {
                "token": "token-123",
                "refresh_token": "refresh-123",
                "scopes": gp.GMAIL_PUSH_OAUTH_SCOPES,
            }
        )


class _FakeFlow:
    last_instance = None

    def __init__(self, client_secret_file, scopes, redirect_uri=None, autogenerate_code_verifier=False):
        self.client_secret_file = client_secret_file
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        self.code_verifier = "verifier"
        self.credentials = _FakeCredentials()
        self.fetched_code = None
        _FakeFlow.last_instance = self

    @classmethod
    def from_client_secrets_file(cls, client_secret_file, scopes, **kwargs):
        return cls(client_secret_file, scopes, **kwargs)

    def authorization_url(self, access_type="offline", prompt="consent"):
        return ("https://accounts.google.com/o/oauth2/auth", "state-123")

    def fetch_token(self, code):
        self.fetched_code = code


def test_setup_saves_profile_scoped_config_and_token(tmp_path, monkeypatch, capsys):
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "abc",
                    "client_secret": "def",
                    "redirect_uris": ["http://localhost"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gp, "GOOGLE_OAUTHLIB_AVAILABLE", True)
    monkeypatch.setattr(gp, "Flow", _FakeFlow)

    gp.gmail_push_command(
        _make_args(
            gmail_push_action="setup",
            account="reader@example.com",
            client_secret=str(client_secret),
            auth_code="http://localhost/?code=code-123&state=state-123",
            topic="projects/demo/topics/hermes-gmail-push",
            subscription="hermes-gmail-push",
            public_url="https://example.com/gmail-push",
            service_account_email="push-auth@example.iam.gserviceaccount.com",
        )
    )

    config = gp._load_gmail_push_block()
    token_path = Path(config["extra"]["oauth"]["token_path"])
    assert config["enabled"] is True
    assert config["extra"]["account"] == "reader@example.com"
    assert config["extra"]["push_auth"]["audience"] == "https://example.com/gmail-push"
    assert token_path.exists()
    token_payload = json.loads(token_path.read_text(encoding="utf-8"))
    assert token_payload["token"] == "token-123"
    assert _FakeFlow.last_instance.fetched_code == "code-123"
    out = capsys.readouterr().out
    assert "Gmail push configured" in out
    assert "Start ingestion with: hermes gateway run" in out


def test_status_reads_saved_state(tmp_path, capsys):
    block = {
        "enabled": True,
        "extra": {
            "account": "reader@example.com",
            "topic": "projects/demo/topics/hermes-gmail-push",
            "subscription": "hermes-gmail-push",
            "endpoint": {
                "host": "0.0.0.0",
                "port": 8645,
                "path": "/gmail-push",
                "public_url": "https://example.com/gmail-push",
            },
            "oauth": {
                "client_secret_path": str(tmp_path / "client_secret.json"),
                "token_path": str(tmp_path / "integrations" / "gmail_push" / "reader@example.com" / "token.json"),
            },
            "push_auth": {
                "service_account_email": "push-auth@example.iam.gserviceaccount.com",
                "audience": "https://example.com/gmail-push",
            },
            "state": {
                "path": str(tmp_path / "integrations" / "gmail_push" / "reader@example.com" / "state.json"),
            },
        },
    }
    gp._save_gmail_push_block(block)
    state_path = Path(block["extra"]["state"]["path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "account": "reader@example.com",
                "last_history_id": "777",
                "watch_expiration_ms": 12345,
                "last_watch_renewed_at": "2026-04-15T12:00:00Z",
                "last_notification_at": "2026-04-15T13:00:00Z",
                "last_error": None,
                "last_successful_pubsub_message_id": "pubsub-42",
                "degraded": False,
            }
        ),
        encoding="utf-8",
    )

    gp.gmail_push_command(_make_args(gmail_push_action="status"))
    out = capsys.readouterr().out
    assert "Gmail push status" in out
    assert "reader@example.com" in out
    assert "777" in out
    assert "pubsub-42" in out


def test_renew_and_resync_route_to_adapter(monkeypatch, capsys):
    class FakeAdapter:
        async def refresh_watch_now(self):
            return {"historyId": "renew-1", "expiration": "1000"}

        async def rebaseline(self):
            return {"historyId": "rebased-2", "expiration": "2000"}

    monkeypatch.setattr(gp, "_make_cli_adapter", lambda require_enabled=True: FakeAdapter())

    gp.gmail_push_command(_make_args(gmail_push_action="renew"))
    renew_out = capsys.readouterr().out
    assert "Watch renewed" in renew_out
    assert "renew-1" in renew_out

    gp.gmail_push_command(_make_args(gmail_push_action="resync"))
    resync_out = capsys.readouterr().out
    assert "Baseline history cursor reset" in resync_out
    assert "rebased-2" in resync_out


def test_test_command_prints_health_issues(monkeypatch, capsys):
    class FakeAdapter:
        async def run_health_check(self):
            return {
                "ok": False,
                "issues": ["OAuth token not found"],
                "account": "reader@example.com",
                "endpoint": {"host": "0.0.0.0", "port": 8645, "path": "/gmail-push"},
            }

    monkeypatch.setattr(gp, "_make_cli_adapter", lambda require_enabled=False: FakeAdapter())
    gp.gmail_push_command(_make_args(gmail_push_action="test"))
    out = capsys.readouterr().out
    assert "Gmail push health check" in out
    assert "OAuth token not found" in out


def test_main_parser_wires_gmail_push_subcommand():
    import hermes_cli.main as main_mod

    source = inspect.getsource(main_mod)
    assert '"gmail-push"' in source
    assert "cmd_gmail_push" in source
