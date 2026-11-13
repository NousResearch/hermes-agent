"""An OAuth sign-in belongs to the gateway that issued it, never to whatever a connection id names later.

Contracts: a token set is bound to the URL it was issued for; after the entry is re-pointed, or removed
and its label (so its id) re-used at another address, no bearer is sent and no refresh token is posted
anywhere; a sign-in survives edits that keep the dial (a rename).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_agent.hermes_cli import gateway_connections as store
from hermes_agent.hermes_cli import gateway_connections_oauth as oauth
from hermes_agent.hermes_cli import gateway_connections_reach as reach


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def host_b():
    """A real HTTP listener standing in for the gateway the entry is re-pointed at."""
    seen: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.rfile.read(int(self.headers.get("content-length", 0))).decode()))
            self.send_response(401)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


def _oauth(label: str, url: str, **extra) -> dict:
    return store.save_connection({"kind": "remote", "label": label, "url": url, "authMode": "oauth", **extra})[0]


def test_a_re_pointed_or_re_added_connection_never_carries_the_old_gateways_sign_in(home):
    saved = _oauth("Mac mini", "https://a.example")
    store.write_tokens(saved["id"], {"accessToken": "AT-for-A", "refreshToken": "RT-for-A",
                                     "expiresAt": int(time.time()) + 3600})
    _oauth("Mac mini renamed", "https://a.example", id=saved["id"])
    assert reach.resolve_target(saved["id"])[1]["Authorization"] == "Bearer AT-for-A"

    _oauth("Mac mini renamed", "https://b.example", id=saved["id"])
    with pytest.raises(oauth.NotSignedIn):
        reach.resolve_target(saved["id"])

    store.write_tokens(saved["id"], {"accessToken": "AT-for-B", "refreshToken": "RT-for-B",
                                     "expiresAt": int(time.time()) + 3600})
    store.remove_connection(saved["id"])
    again = _oauth("Mac mini", "https://c.example")
    assert again["id"] == saved["id"]
    with pytest.raises(oauth.NotSignedIn):
        reach.resolve_target(again["id"])
    assert json.loads(store.tokens_path().read_text())["tokens"] == {}


def test_a_set_whose_url_differs_from_the_entry_is_never_refreshed_there(home, host_b):
    url_b, seen = host_b
    saved = _oauth("Box", "https://a.example")
    store.write_tokens(saved["id"], {"accessToken": "AT-A", "refreshToken": "RT-of-host-A", "expiresAt": 1})
    # Another writer (an older build, a hand edit) moved the URL without clearing the tokens file.
    doc = json.loads(store.store_path().read_text())
    doc["connections"][1]["url"] = url_b
    store.store_path().write_text(json.dumps(doc))

    with pytest.raises(oauth.NotSignedIn):
        oauth.ensure_access_token(saved["id"])
    assert seen == []


def test_a_refresh_answered_after_the_entry_was_re_pointed_is_not_bound_to_the_new_host(home, monkeypatch):
    """A re-point landing after the refresh's last read but before its write must not bind host A's
    rotated tokens to host B (the write re-checks the issuer under the tokens lock)."""
    saved = _oauth("Box", "https://a.example")
    store.write_tokens(saved["id"], {"accessToken": "AT-A", "refreshToken": "RT-A", "expiresAt": 1})

    class Rotated:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "AT-A-rotated", "refresh_token": "RT-A2", "expires_at": time.time() + 3600}

    monkeypatch.setattr(oauth, "_post_json", lambda *_args, **_kw: Rotated)
    real_read, reads = store.dial_snapshot, []

    def read_then_re_point(connection_id):
        found = real_read(connection_id)
        reads.append(connection_id)
        if len(reads) == 2:  # the refresh's post-rotation check passed; another writer re-points now
            store.save_connection({"id": connection_id, "kind": "remote", "label": "Box",
                                   "url": "https://b.example", "authMode": "oauth"})
        return found

    monkeypatch.setattr(store, "dial_snapshot", read_then_re_point)
    with pytest.raises(oauth.NotSignedIn):
        oauth.ensure_access_token(saved["id"])
    monkeypatch.setattr(store, "dial_snapshot", real_read)
    assert store.read_tokens(saved["id"]) is None
    assert "AT-A-rotated" not in store.tokens_path().read_text()


@pytest.fixture
def listener():
    """Real HTTP listeners recording the credentials each one receives."""
    servers = []

    def start():
        received: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", received

    yield start
    for server in servers:
        server.shutdown()


@pytest.mark.parametrize("auth_mode", ["oauth", "token"])
def test_a_re_point_during_resolution_never_sends_the_new_hosts_credential_to_the_old_one(
        home, listener, monkeypatch, auth_mode):
    import httpx

    url_a, at_a = listener()
    url_b, at_b = listener()
    fields = {"authMode": "oauth"} if auth_mode == "oauth" else {"token": "TOKEN_A"}
    saved = store.save_connection({"kind": "remote", "label": "Box", "url": url_a, **fields})[0]
    if auth_mode == "oauth":
        store.write_tokens(saved["id"], {"accessToken": "TOKEN_A", "refreshToken": "R",
                                         "expiresAt": int(time.time()) + 3600}, issued_by=url_a)
    real_entry = reach._entry

    def read_then_re_point(connection_id):
        found = real_entry(connection_id)
        # Another writer re-points the id at B and stores B's credential before reach reads it.
        store.save_connection({"id": connection_id, "kind": "remote", "label": "Box", "url": url_b,
                               **({"authMode": "oauth"} if auth_mode == "oauth" else {"token": "TOKEN_B"})})
        if auth_mode == "oauth":
            store.write_tokens(connection_id, {"accessToken": "TOKEN_B", "refreshToken": "R",
                                               "expiresAt": int(time.time()) + 3600}, issued_by=url_b)
        return found

    monkeypatch.setattr(reach, "_entry", read_then_re_point)
    base_url, headers = reach.resolve_target(saved["id"])
    httpx.get(f"{base_url}/api/sessions", headers=headers, trust_env=False)

    assert not any("TOKEN_B" in " ".join(h.values()) for h in at_a)
    assert (base_url, len(at_b)) == (url_b, 1)
