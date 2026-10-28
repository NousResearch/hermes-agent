"""google-workspace setup.py honours the documented --services/--format contract."""

from __future__ import annotations

import importlib.util
import json
import urllib.parse
from pathlib import Path
from unittest.mock import Mock

import pm
import pytest

SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)
GMAIL_CALENDAR = {
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
}


@pytest.fixture()
def setup_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup_services", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_services_select_scopes_and_reject_unknown_names(setup_module):
    assert set(setup_module._scopes_for_services("email,calendar")) == GMAIL_CALENDAR
    assert set(setup_module._scopes_for_services(None)) == set(setup_module.SCOPES)
    with pytest.raises(ValueError, match="calender"):
        setup_module._scopes_for_services("gmail,calender")


def test_auth_url_json_requests_only_the_chosen_services(setup_module, monkeypatch, capsys, tmp_path):
    pytest.importorskip("google_auth_oauthlib")
    monkeypatch.setattr(pm, "ensure_import", Mock())
    secret = {
        "installed": {
            "client_id": "fake-id.apps.googleusercontent.com",
            "client_secret": "fake-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    setup_module.CLIENT_SECRET_PATH.write_text(json.dumps(secret), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["setup.py", "--auth-url", "--services", "email,calendar", "--format", "json"])
    with pytest.raises(SystemExit) as exit_info:
        setup_module.main()

    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(payload["auth_url"]).query)
    assert set(query["scope"][0].split()) == GMAIL_CALENDAR
    assert (tmp_path / "google_oauth_last_url.txt").read_text(encoding="utf-8").strip() == payload["auth_url"]


def test_check_json_keeps_status_lines_off_stdout(setup_module, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["setup.py", "--check", "--format", "json"])
    with pytest.raises(SystemExit) as exit_info:
        setup_module.main()

    out, err = capsys.readouterr()
    assert exit_info.value.code == 1
    assert json.loads(out)["status"] == "unauthenticated"
    assert "NOT_AUTHENTICATED" in err


def test_contacts_only_live_probe_names_the_people_resource(setup_module, monkeypatch):
    pytest.importorskip("googleapiclient")
    from googleapiclient.http import HttpRequest

    setup_module.TOKEN_PATH.write_text(json.dumps({
        "type": "authorized_user",
        "client_id": "fake-id.apps.googleusercontent.com",
        "client_secret": "fake-secret",
        "refresh_token": "fake-refresh",
        "token": "fake-access",
        "scopes": ["https://www.googleapis.com/auth/contacts.readonly"],
    }), encoding="utf-8")
    sent = []
    monkeypatch.setattr(HttpRequest, "execute", lambda self, *a, **k: sent.append(self.uri) or {})

    setup_module._probe_service("contacts")

    assert len(sent) == 1 and "/people/me/connections" in sent[0]
