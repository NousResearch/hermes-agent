"""Exercise the real dashboard router, HTTP client and profile-local files."""
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import config as cfg
from hermes_cli.web_routers import messaging


@pytest.fixture
def onboarding(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = {"counter": 0, "acks": 0, "fail_ack": False, "cancelled": [], "status": "ready"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path.endswith("/ack"):
                assert self.headers["Authorization"] == "Bearer poll-secret"
                state["acks"] += 1
                self.respond({"ok": True}, 502 if state["fail_ack"] else 200)
            else:
                state["counter"] += 1
                self.respond({"pairing_id": f"pair-{state['counter']}", "poll_token": "poll-secret",
                              "expires_at": datetime.fromtimestamp(time.time()+2, timezone.utc).isoformat(),
                              "deep_link": "https://t.me/Manager?start=pair_example", "suggested_username": "suggested_bot"})

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer poll-secret"
            if state["status"] == "cancelled":
                self.respond({"status": "cancelled"}, 410)
                return
            self.respond({"status": state["status"], "token": "123456:" + "A"*35,
                          "bot_username": "renamed_bot", "owner_user_id": 42, "requires_ack": True})

        def do_DELETE(self):
            state["cancelled"].append(self.path)
            self.respond({"ok": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TELEGRAM_ONBOARDING_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(messaging, "_restart_gateway_after", lambda *a, **k: {"restart_started": False, "restart_error": "test supervisor"})
    app = FastAPI()
    app.include_router(messaging.router)
    with TestClient(app) as client:
        yield client, home, state
    server.shutdown()
    server.server_close()
    thread.join()


def test_ready_survives_creation_expiry_and_restart_then_save_is_retryable(onboarding, monkeypatch):
    from hermes_cli import telegram_onboarding_store as store
    client, home, state = onboarding
    response = client.post("/api/messaging/telegram/onboarding/start", json={})
    assert response.status_code == 200, response.text
    start = response.json()
    endpoint = f"/api/messaging/telegram/onboarding/{start['pairing_id']}"
    assert "poll_token" not in start
    # Real persistence is the only source: no in-memory session dictionary.
    with store.lock:
        waiting = store.load(start["pairing_id"])
        waiting.expires_at_ts = time.time()-1
        store.save(start["pairing_id"], waiting)
    ready = client.get(endpoint)
    assert ready.status_code == 200
    assert ready.json()["owner_user_id"] == "42"
    assert datetime.fromisoformat(ready.json()["expires_at"]).timestamp() > time.time()+1700
    assert store.record_path(start["pairing_id"]).stat().st_mode & 0o777 == 0o600
    assert "token" not in ready.json()
    state["fail_ack"] = True
    first = client.post(endpoint+"/apply", json={"allowed_user_ids": ["42"]})
    assert first.status_code == 200
    assert first.json()["needs_restart"] is True
    assert cfg.load_env()["TELEGRAM_ALLOWED_USERS"] == "42"
    assert store.load(start["pairing_id"]).bot_token is None
    # A lost response followed by a later manual edit must not be overwritten by retry.
    cfg.save_env_value("TELEGRAM_ALLOWED_USERS", "99")
    state["fail_ack"] = False
    assert client.post(endpoint+"/apply", json={"allowed_user_ids":["42"]}).json() == first.json()
    assert cfg.load_env()["TELEGRAM_ALLOWED_USERS"] == "99"
    assert state["acks"] == 2
    assert store.load(start["pairing_id"]).poll_token == ""
    # A new attempt can be cancelled at the Worker, discarded locally, and replaced.
    response = client.post("/api/messaging/telegram/onboarding/start", json={})
    assert response.status_code == 200, response.text
    fresh = response.json()["pairing_id"]
    assert client.delete(f"/api/messaging/telegram/onboarding/{fresh}").status_code == 200
    assert state["cancelled"] == [f"/v1/telegram/pairings/{fresh}"]
    assert client.get(f"/api/messaging/telegram/onboarding/{fresh}").status_code == 404
    assert client.post("/api/messaging/telegram/onboarding/start", json={}).status_code == 200


def test_superseded_remote_attempt_cannot_apply_a_cached_token(onboarding):
    client, home, state = onboarding
    start = client.post("/api/messaging/telegram/onboarding/start", json={}).json()
    endpoint = f"/api/messaging/telegram/onboarding/{start['pairing_id']}"
    assert client.get(endpoint).json()["status"] == "ready"
    state["status"] = "cancelled"
    assert client.post(endpoint+"/apply", json={"allowed_user_ids": ["42"]}).status_code == 410
    assert "TELEGRAM_BOT_TOKEN" not in cfg.load_env()
    assert state["acks"] == 0
    assert client.get(endpoint).status_code == 404
    assert client.post("/api/messaging/telegram/onboarding/start", json={}).status_code == 200


@pytest.mark.parametrize("failure", ["allowlist", "enabled", "receipt"])
def test_failed_save_restores_previous_settings_and_leaves_retryable_pairing(onboarding, monkeypatch, failure):
    from hermes_cli import telegram_onboarding_store as store
    client, home, state = onboarding
    cfg.save_env_value("TELEGRAM_BOT_TOKEN", "previous-token")
    cfg.save_env_value("TELEGRAM_ALLOWED_USERS", "7")
    cfg.save_env_value("UNRELATED", "preserve-me")
    (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: false\n    custom: keep\nmodel:\n  default: keep-model\n")
    before_config = cfg.require_readable_config_before_write()
    response = client.post("/api/messaging/telegram/onboarding/start", json={})
    assert response.status_code == 200, response.text
    start = response.json()
    endpoint = f"/api/messaging/telegram/onboarding/{start['pairing_id']}"
    assert client.get(endpoint).status_code == 200
    with monkeypatch.context() as patch:
        if failure == "allowlist":
            original = cfg.save_env_value
            def fail(key, value):
                original(key, value)
                if key == "TELEGRAM_ALLOWED_USERS" and value == "42":
                    raise OSError("injected failure after replace")
            patch.setattr(cfg, "save_env_value", fail)
        elif failure == "enabled":
            original = cfg.atomic_config_write
            def fail(*args, **kwargs):
                original(*args, **kwargs)
                raise OSError("injected failure after config replace")
            patch.setattr(cfg, "atomic_config_write", fail)
        else:
            patch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("receipt failed")))
        result = client.post(endpoint+"/apply", json={"allowed_user_ids":["42"]})
        assert result.status_code == 500
    assert cfg.load_env() == {"TELEGRAM_BOT_TOKEN":"previous-token", "TELEGRAM_ALLOWED_USERS":"7", "UNRELATED":"preserve-me"}
    assert cfg.require_readable_config_before_write() == before_config
    assert state["acks"] == 0
    assert client.get(endpoint).json()["status"] == "ready"
    assert client.post(endpoint+"/apply", json={"allowed_user_ids":["42"]}).status_code == 200
    assert cfg.load_env()["TELEGRAM_ALLOWED_USERS"] == "42"
