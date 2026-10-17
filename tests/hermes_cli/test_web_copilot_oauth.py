"""Regression for PR #32481: dashboard logins must reflect Copilot runtime auth."""

import pytest
from fastapi.testclient import TestClient

from hermes_cli import copilot_auth
from hermes_cli.web_server import _SESSION_TOKEN, app


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("source", ["gh auth token", *copilot_auth.COPILOT_ENV_VARS])
def test_copilot_dashboard_reflects_runtime_token_source(tmp_path, monkeypatch, source):
    """Exercise the real status/credential chain, including a local gh subprocess."""
    token = "gho_dashboard_regression_secret"
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/sh\n[ "$1" = auth ] && [ "$2" = token ] || exit 1\n'
        f"printf '%s\\n' '{token}'\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    for var in copilot_auth.COPILOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    if source != "gh auth token":
        monkeypatch.setenv(source, token)
    monkeypatch.setattr(copilot_auth, "_gh_cli_candidates", lambda: [str(gh)])
    monkeypatch.setattr(copilot_auth, "_gh_cli_token_cache", None)
    # Keep only the remote token exchange outside this local integration test.
    monkeypatch.setattr(copilot_auth, "get_copilot_api_token", lambda raw: (raw, None))

    client = TestClient(app)
    response = client.get("/api/providers/oauth", headers={"X-Hermes-Session-Token": _SESSION_TOKEN})
    assert response.status_code == 200, response.text
    cards = [p for p in response.json()["providers"] if p["id"] == "copilot"]
    assert len(cards) == 1
    card = cards[0]
    assert card["flow"] == "external"
    assert card["cli_command"] == "gh auth login"
    assert card["disconnectable"] is False
    assert card["status"]["logged_in"] is True
    assert card["status"]["source_label"] == source
    assert card["status"]["token_preview"] == f"…{token[-6:]}"
    assert token not in response.text


@pytest.mark.parametrize("token", [None, "ghp_unsupported_classic_pat", "invalid-token"])
def test_copilot_dashboard_keeps_external_login_when_no_usable_token(monkeypatch, token):
    for var in copilot_auth.COPILOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", lambda: token)
    headers = {"X-Hermes-Session-Token": _SESSION_TOKEN}

    client = TestClient(app)
    response = client.get("/api/providers/oauth", headers={"X-Hermes-Session-Token": _SESSION_TOKEN})
    assert response.status_code == 200, response.text
    cards = [p for p in response.json()["providers"] if p["id"] == "copilot"]
    assert len(cards) == 1
    card = cards[0]
    assert card["status"]["logged_in"] is False
    assert card["status"]["token_preview"] is None
    assert card["cli_command"] == "gh auth login"

    start = client.post("/api/providers/oauth/copilot/start", headers=headers)
    assert start.status_code == 400, start.text
    assert card["cli_command"] in start.json()["detail"]
    disconnect = client.delete("/api/providers/oauth/copilot", headers=headers)
    assert disconnect.status_code == 400, disconnect.text
    assert "cannot be disconnected automatically" in disconnect.json()["detail"]
