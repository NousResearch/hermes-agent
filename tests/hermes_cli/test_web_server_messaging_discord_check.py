"""Desktop checks the saved Discord token without crossing profile boundaries."""

import socket
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.testclient import TestClient

from plugins.platforms.discord.onboarding import INVITE_PERMISSIONS
from tests.fakes.platforms.discord_standin import APP_ID, BOT_USERNAME, TOKEN, DiscordStandin


@pytest.fixture
def discord_check(monkeypatch, _isolate_hermes_home):
    from hermes_cli import profiles
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home
    from tools import discord_tool

    root = get_hermes_home()
    home = root / "profiles" / "discord_test"
    home.mkdir(parents=True)
    for directory in (root, home):
        (directory / "config.yaml").write_text("{}\n", encoding="utf-8")
        (directory / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    standin = DiscordStandin().start()
    monkeypatch.setattr(discord_tool, "DISCORD_API_BASE", standin.api_base)
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    try:
        yield client, home, standin
    finally:
        client.close()
        standin.stop()


def _check(client):
    return client.post("/api/messaging/discord/check", params={"profile": "discord_test"})


@pytest.mark.parametrize("intents", [True, False])
def test_saved_profile_token_returns_safe_check(discord_check, monkeypatch, intents):
    client, home, standin = discord_check
    (home / ".env").write_text(f"DISCORD_BOT_TOKEN={TOKEN}\n", encoding="utf-8")
    original = standin._r_get_app

    def application(params):
        status, payload = original(params)
        return status, {**payload, "flags": payload["flags"] if intents else 0,
                        "approximate_guild_count": 0}

    monkeypatch.setattr(standin, "_r_get_app", application)
    response = _check(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_id"] == APP_ID
    assert payload["bot_name"] == BOT_USERNAME
    assert payload["owners"] == [{"id": "1", "username": "owner"}]
    assert payload["message_content"] is intents
    assert payload["server_members"] is intents
    assert payload["server_count"] == 0
    assert payload["bot_settings_url"] == f"https://discord.com/developers/applications/{APP_ID}/bot"
    query = parse_qs(urlparse(payload["invite_url"]).query)
    assert query == {"client_id": [APP_ID], "permissions": [str(INVITE_PERMISSIONS)],
                     "scope": ["bot applications.commands"], "integration_type": ["0"]}
    assert TOKEN not in response.text


def test_rejected_token_is_inline_safe_error(discord_check):
    client, home, _standin = discord_check
    rejected = "invalid.standin.token"
    (home / ".env").write_text(f"DISCORD_BOT_TOKEN={rejected}\n", encoding="utf-8")
    response = _check(client)
    assert response.status_code == 400
    assert "rejected" in response.json()["detail"]
    assert rejected not in response.text


@pytest.mark.parametrize("root_token", [False, True])
def test_missing_profile_token_never_checks_root_bot(discord_check, monkeypatch, root_token):
    from tools import discord_tool

    client, _home, _standin = discord_check
    if root_token:
        monkeypatch.setenv("DISCORD_BOT_TOKEN", TOKEN)
    monkeypatch.setattr(discord_tool, "_discord_request", lambda *a, **kw: pytest.fail("no outbound request"))
    response = _check(client)
    assert response.status_code == 400
    assert response.json()["detail"] == "Save a Discord bot token first."


def test_unreachable_discord_is_safe_gateway_error(discord_check, monkeypatch):
    from tools import discord_tool

    client, home, _standin = discord_check
    (home / ".env").write_text(f"DISCORD_BOT_TOKEN={TOKEN}\n", encoding="utf-8")
    # A bound socket that isn't listening is a deterministic refused connection.
    with socket.socket() as dead_port:
        dead_port.bind(("127.0.0.1", 0))
        monkeypatch.setattr(discord_tool, "DISCORD_API_BASE", f"http://127.0.0.1:{dead_port.getsockname()[1]}")
        response = _check(client)
    assert response.status_code == 502
    assert "Couldn't reach Discord" in response.json()["detail"]
    assert TOKEN not in response.text


@pytest.mark.parametrize("error_kind", ["header", "discord"])
def test_exception_text_never_reaches_response(discord_check, monkeypatch, error_kind):
    from plugins.platforms.discord import onboarding
    from tools.discord_tool import DiscordAPIError

    client, home, _standin = discord_check
    (home / ".env").write_text(f"DISCORD_BOT_TOKEN={TOKEN}\n", encoding="utf-8")

    def fail(_token):
        raise ValueError(TOKEN) if error_kind == "header" else DiscordAPIError(503, TOKEN)

    monkeypatch.setattr(onboarding, "check_bot_token", fail)
    response = _check(client)
    assert response.status_code == (400 if error_kind == "header" else 502)
    assert TOKEN not in response.text


@pytest.mark.parametrize(("status", "body"), [
    (200, b"<html>sign in to the network</html>"),
    (200, b"[]"),
    (200, b'{"id": "1", "flags": "abc"}'),
    (204, b""),
])
def test_non_json_discord_body_is_unreachable_not_a_bad_token(discord_check, monkeypatch, status, body):
    """A captive portal or proxy answering 2xx with something other than an application is not a bad token."""
    import http.server
    import threading
    from tools import discord_tool

    class Portal(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            if body:
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    client, home, _standin = discord_check
    (home / ".env").write_text(f"DISCORD_BOT_TOKEN={TOKEN}\n", encoding="utf-8")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Portal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(discord_tool, "DISCORD_API_BASE", f"http://127.0.0.1:{server.server_address[1]}")
        response = _check(client)
    finally:
        server.shutdown()
        server.server_close()
    assert response.status_code == 502
    assert "Couldn't reach Discord" in response.json()["detail"]
    assert TOKEN not in response.text
