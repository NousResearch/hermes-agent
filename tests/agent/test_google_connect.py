"""Google wizard backend: real files in a temp HERMES_HOME, the skill's scripts replaced by a fake runner."""

import json
from datetime import datetime, timezone

import pytest

from agent import google_connect as gc

CLIENT = {"installed": {"client_id": "id.apps.googleusercontent.com", "client_secret": "s3cret"}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_status_reads_files_only_and_reports_granted_services(home):
    assert gc.status() == {"client_secret": False, "token": False, "connected": False, "services": [], "can_send_mail": False}

    (home / gc.TOKEN_FILE).write_text(json.dumps({"scopes": [
        "https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/calendar"]}), encoding="utf-8")
    got = gc.status()

    assert got["connected"] and got["services"] == ["calendar", "gmail.readonly"] and got["can_send_mail"] is False


def test_client_file_is_validated_before_it_is_stored(home):
    for bad in ("not json", json.dumps({"foo": 1}), json.dumps({"installed": {"client_id": "x"}})):
        with pytest.raises(ValueError):
            gc.save_client_secret(bad)
    assert not (home / gc.CLIENT_FILE).exists()

    assert gc.save_client_secret(json.dumps(CLIENT)) == {"ok": True}
    assert json.loads((home / gc.CLIENT_FILE).read_text())["installed"]["client_id"].startswith("id.")


def test_auth_url_installs_missing_libraries_once_then_returns_the_link():
    calls = []

    def run(script, args, timeout):
        calls.append(args[0])
        if calls.count("--auth-url") == 1 and args == ["--auth-url"]:
            return 1, "", "ModuleNotFoundError: No module named 'google_auth_oauthlib'"
        if args == ["--install-deps"]:
            return 0, "installed", ""
        return 0, "noise\nhttps://accounts.google.com/o/oauth2/auth?x=1\n", ""

    assert gc.auth_url(run) == "https://accounts.google.com/o/oauth2/auth?x=1"
    assert calls == ["--auth-url", "--install-deps", "--auth-url"]


def test_a_rejected_code_is_an_error_with_the_reason_not_a_silent_success():
    with pytest.raises(RuntimeError, match="expired"):
        gc.exchange_code("bad", lambda s, a, t: (1, "ERROR: Token exchange failed: code expired", ""))

    assert gc.exchange_code("good", lambda s, a, t: (0, "OK: Authenticated. Token saved", "")) == {"ok": True, "warning": None}


def test_verify_reads_back_events_and_unread_mail_and_survives_one_service_failing():
    def run(script, args, timeout):
        if script == "setup.py":
            return 0, "LIVE_CHECK_OK: Real API call succeeded.", ""
        if args[0] == "calendar":
            return 0, json.dumps([{"summary": "Spotkanie z Anną", "start": "2026-10-01T09:00:00+02:00"}]), ""
        return 1, "", "gmail scope missing"

    got = gc.verify(run, now=datetime(2026, 9, 30, tzinfo=timezone.utc))

    assert got["ok"] and got["events"] == [{"summary": "Spotkanie z Anną", "start": "2026-10-01T09:00:00+02:00"}]
    assert got["unread"] == [] and got["calendar_ok"] is True and got["gmail_ok"] is False


def test_verify_fails_closed_when_the_live_check_does():
    got = gc.verify(lambda s, a, t: (1, "LIVE_CHECK_FAILED: disabled_client", ""))

    assert got["ok"] is False and "disabled_client" in got["reason"]
