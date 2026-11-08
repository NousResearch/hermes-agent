"""Accounts Connected must follow Claude Code token validity, not token presence."""

from hermes_cli.web_server import _SESSION_TOKEN, app
from fastapi.testclient import TestClient

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def test_expired_claude_code_token_is_not_connected(monkeypatch):
    """A present but expired access token must not mark the account Connected."""
    from agent import anthropic_credentials

    monkeypatch.setattr(
        anthropic_credentials,
        "read_claude_code_credentials",
        lambda: {"accessToken": "stale-token", "expiresAt": 1},
    )

    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}

    assert providers["claude-code"]["status"]["logged_in"] is False


def test_windows_claude_code_removal_command_is_unambiguous_powershell():
    """PowerShell must not see ``rm -f``: ``-f`` is an ambiguous Remove-Item parameter."""
    from hermes_cli.web_routers.oauth import _oauth_provider_disconnect_command

    command = _oauth_provider_disconnect_command(
        {"id": "claude-code", "flow": "external"}, platform="win32"
    )

    assert command is not None
    assert "Remove-Item" in command
    assert "-LiteralPath" in command
    assert "-Force" in command
    assert "rm -f" not in command
    assert " -f " not in command and not command.strip().endswith(" -f")


def test_disconnect_is_not_success_when_nothing_was_cleared(monkeypatch):
    """A no-op clear must not be a 200 the client can toast as removed."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr(auth_mod, "clear_provider_auth", lambda _provider: False)

    resp = client.delete("/api/providers/oauth/nous", headers=HEADERS)

    assert resp.status_code == 409, resp.text
    assert resp.json().get("ok") is not True


def test_disconnect_failure_does_not_echo_the_store_error(monkeypatch):
    from hermes_cli import auth as auth_mod

    secret = "auth store is read-only: /secret/hermes-home"

    def fail_clear(_provider):
        raise OSError(secret)

    monkeypatch.setattr(auth_mod, "clear_provider_auth", fail_clear)

    resp = client.delete("/api/providers/oauth/nous", headers=HEADERS)

    assert resp.status_code == 500, resp.text
    assert secret not in resp.text

