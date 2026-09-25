"""A contract-faithful stub of the config plane's agent routes (contract.md §7.1, §7.2, §8) plus an
OAuth2 client-credentials token endpoint, for driving the remote backend without gg.

Levels are collapsed to one ``upper`` document (everything above the profile) plus per-profile
``values``; locks are ``upper_locks``. Effective = deep_merge(upper, profile) — enough for the
agent, which never resolves the chain itself (design §2.3).
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugins.config_backends.remote.diff import write_check
from plugins.config_backends.remote.paths import decode, encode
from plugins.config_backends.remote.values import secret_literal_path

TOKEN = "plane-token-1"
CLIENT_ID = "agent-client"
CLIENT_SECRET = "agent-secret"
INSTANCE = "inst-0001"


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class StubPlane:
    def __init__(self) -> None:
        self.upper: Dict[str, Any] = {}
        self.upper_locks: List[Dict[str, str]] = []
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.fail_status: Optional[int] = None   # every /self request returns this
        self.requests: List[Dict[str, Any]] = []
        self.token_requests = 0
        self.home: Optional[Path] = None  # the test's HERMES_HOME, set by the fixture
        self.lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> "StubPlane":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    # --- model ---

    def profile(self, name: str) -> Dict[str, Any]:
        return self.profiles.setdefault(name, {"values": {}, "version": 0, "writer": None})

    def effective(self, name: str) -> Dict[str, Any]:
        prof = self.profile(name)
        body = {
            "instanceId": INSTANCE, "profile": name,
            "config": deep_merge(self.upper, prof["values"]),
            "locks": copy.deepcopy(self.upper_locks),
            "provenance": {},
            "levels": [{"kind": "tenant", "version": 1, "writerConfigVersion": None},
                       {"kind": "profile", "version": prof["version"], "writerConfigVersion": prof["writer"]}],
            "profileVersion": prof["version"], "recordVersion": 1,
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        body["etag"] = f'"{digest}"'
        return body

    def patches(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "PATCH"]

    # --- HTTP ---

    def _handler(self):
        plane = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args) -> None:  # noqa: A002 — base signature
                pass

            def _send(self, status: int, body: Optional[Dict[str, Any]] = None, etag: Optional[str] = None):
                raw = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                if etag:
                    self.send_header("ETag", etag)
                self.end_headers()
                self.wfile.write(raw)

            def _body(self) -> Dict[str, Any]:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_POST(self):
                if self.path != "/token":
                    return self._send(404, {"error": "not_found"})
                n = int(self.headers.get("Content-Length") or 0)
                form = urllib.parse.parse_qs(self.rfile.read(n).decode())
                plane.token_requests += 1
                if (form.get("grant_type") != ["client_credentials"] or form.get("client_id") != [CLIENT_ID]
                        or form.get("client_secret") != [CLIENT_SECRET]):
                    return self._send(401, {"error": "invalid_client"})
                return self._send(200, {"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3600})

            def _route(self, method: str):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/v1/config/self":
                    return self._send(404, {"error": "not_found"})
                profile = urllib.parse.parse_qs(parsed.query).get("profile", ["default"])[0]
                body = self._body() if method == "PATCH" else None
                with plane.lock:
                    plane.requests.append({
                        "method": method, "profile": profile, "body": body,
                        "auth": self.headers.get("Authorization"),
                        "instance": self.headers.get("X-Hermes-Instance-Id"),
                        "if_none_match": self.headers.get("If-None-Match")})
                    if plane.fail_status is not None:
                        return self._send(plane.fail_status, {"error": "unavailable", "message": "stub outage"})
                    if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                        return self._send(401, {"error": "unauthorized", "message": "bad token"})
                    if self.headers.get("X-Hermes-Instance-Id") != INSTANCE:
                        return self._send(403, {"error": "config_agent_unknown", "message": "unknown instance"})
                    if method == "GET":
                        eff = plane.effective(profile)
                        if self.headers.get("If-None-Match") == eff["etag"]:
                            return self._send(304, None, eff["etag"])
                        return self._send(200, eff, eff["etag"])
                    return self._patch(profile, body or {})

            def _patch(self, profile: str, body: Dict[str, Any]):
                prof = plane.profile(profile)
                if body.get("expectedVersion") != prof["version"]:
                    return self._send(409, {"error": "config_version_conflict", "message": "stale",
                                            "currentVersion": prof["version"]})
                sets = {decode(k): v for k, v in (body.get("set") or {}).items()}
                unsets = [decode(k) for k in body.get("unset") or []]
                locks = [(decode(lk["path"]), lk["level"]) for lk in plane.upper_locks]
                refused = write_check(sets, unsets, locks)
                if refused is not None:
                    path = encode(refused[0])
                    return self._send(403, {"error": "config_key_locked", "path": path, "lockedBy": refused[1],
                                            "message": f"{path} is locked by {refused[1]}"})
                for p, v in sets.items():
                    if secret_literal_path(p, v) is not None:
                        return self._send(400, {"error": "config_secret_literal", "message": "secret literal"})
                values = prof["values"]
                for p, v in sets.items():
                    node = values
                    for seg in p[:-1]:
                        if not isinstance(node.get(seg), dict):
                            node[seg] = {}
                        node = node[seg]
                    node[p[-1]] = copy.deepcopy(v)
                for p in unsets:
                    node = values
                    for seg in p[:-1]:
                        node = node.get(seg) if isinstance(node, dict) else None
                    if isinstance(node, dict):
                        node.pop(p[-1], None)
                prof["version"] += 1
                prof["writer"] = body.get("writerConfigVersion")
                eff = plane.effective(profile)
                return self._send(200, eff, eff["etag"])

            def do_GET(self):
                self._route("GET")

            def do_PATCH(self):
                self._route("PATCH")

        return Handler


def remote_env(plane: StubPlane) -> Dict[str, str]:
    return {
        "HERMES_CONFIG_BACKEND": "remote",
        "HERMES_CONFIG_REMOTE_URL": plane.url,
        "HERMES_CONFIG_INSTANCE_ID": INSTANCE,
        "GATEWAY_RELAY_IDP_TOKEN_URL": f"{plane.url}/token",
        "GATEWAY_RELAY_IDP_CLIENT_ID": CLIENT_ID,
        "GATEWAY_RELAY_IDP_CLIENT_SECRET": CLIENT_SECRET,
    }
