"""OAuth control commands against a loopback token endpoint."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from hermes_cli import auth_commands
from hermes_cli.auth import read_credential_pool, write_credential_pool


@pytest.fixture(autouse=True)
def isolated_external_auth_stores(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))


def _rows():
    return [dict(id=f"row{i}", label=f"account{i}", source="manual:device_code",
                 auth_type="oauth", access_token=f"fixture-access-{i}",
                 refresh_token=f"fixture-refresh-{i}", priority=i,
                 last_status="exhausted", last_status_at=time.time(),
                 last_error_code=429, last_error_reset_at=time.time()+3600)
            for i in range(2)]


@pytest.mark.parametrize("status", [200, 503, 401])
def test_refresh_uses_target_grant_and_preserves_sibling(monkeypatch, status):
    from hermes_cli import auth_codex
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()))
            body = ({"access_token": "fixture-new-access", "refresh_token": "fixture-new-refresh"}
                    if status == 200 else {"error": "invalid_grant" if status == 401 else "unavailable"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(auth_codex, "CODEX_OAUTH_TOKEN_URL", f"http://127.0.0.1:{server.server_port}/token")
    from agent.credential_pool import PooledCredential
    rows = [PooledCredential.from_dict("openai-codex", row).to_dict() for row in _rows()]
    write_credential_pool("openai-codex", rows)
    before = read_credential_pool("openai-codex")
    try:
        args = SimpleNamespace(provider="openai-codex", target="row1")
        if status == 200:
            auth_commands.auth_refresh_command(args)
        else:
            with pytest.raises(SystemExit, match="Refresh failed"):
                auth_commands.auth_refresh_command(args)
        after = {e["id"]: e for e in read_credential_pool("openai-codex")}
        assert requests == [{"grant_type": ["refresh_token"], "refresh_token": ["fixture-refresh-1"],
                             "client_id": [auth_codex.CODEX_OAUTH_CLIENT_ID]}]
        assert after["row0"] == before[0], (after["row0"], before[0])
        target = after["row1"]
        if status == 200:
            assert target["access_token"] == "fixture-new-access"
            assert target["refresh_token"] == "fixture-new-refresh"
            assert target.get("last_error_reset_at") is None
            assert target["last_status"] == "ok"
        else:
            # Manual grants remain in the pool on terminal failure; only
            # singleton-seeded grants are removed by the existing quarantine.
            assert target["last_status"] == "exhausted"
            assert target["access_token"] == before[1]["access_token"]
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def test_add_priority_places_reauthenticated_row_in_multi_entry_pool(monkeypatch):
    rows = _rows()
    rows[1]["source"] = "device_code"
    write_credential_pool("nous", rows)
    monkeypatch.setattr(auth_commands.auth_mod, "_read_shared_nous_state", lambda: None)
    monkeypatch.setattr(auth_commands.auth_mod, "_nous_device_code_login", lambda **_kwargs: {
        "access_token": "fixture-renewed", "refresh_token": "fixture-renewed-refresh",
        "agent_key": "fixture-agent-key", "expires_at": time.time() + 3600,
    })
    auth_commands.auth_add_command(SimpleNamespace(
        provider="nous", auth_type="oauth", priority=0, label="reauthenticated"))
    entries = read_credential_pool("nous")
    assert [e["id"] for e in entries] == ["row1", "row0"]
    assert entries[0]["priority"] == 0


def test_refresh_rejects_ambiguous_and_non_oauth_targets():
    rows = _rows()
    write_credential_pool("openai-codex", rows)
    with pytest.raises(SystemExit, match="pass an index"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="openai-codex", target=None))
    with pytest.raises(SystemExit, match="No credential matching"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="openai-codex", target="missing"))
    rows[0].update(auth_type="api_key", source="manual")
    write_credential_pool("openrouter", rows[:1])
    with pytest.raises(SystemExit, match="not a refreshable"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="openrouter", target=None))
    # Nous's resolver refreshes only its singleton, never an independent pool grant.
    write_credential_pool("nous", _rows())
    before = read_credential_pool("nous")
    with pytest.raises(SystemExit, match="not a refreshable"):
        auth_commands.auth_refresh_command(SimpleNamespace(provider="nous", target="row0"))
    assert read_credential_pool("nous") == before
