"""A contract-faithful stub of the secrets plane's agent route (``/v1/secrets/self``), plus the
level writes other callers make (tenant, owner, instance) and an OAuth2 client-credentials token
endpoint, served over real HTTP on loopback for driving the remote secret store.

Model: per (instance, profile) a profile level; per instance an instance level and its owner; per
owner subject an owner level; one tenant level. Each level holds env entries ``{name: (value,
version)}`` and auth entries ``{provider: (record, version)}`` and a document version. A read
resolves per name the highest level defining it (profile > instance > owner > tenant). Reserved
names (the vendored ``secrets-reserved-names.json``) are refused on every write, and so is a name
outside contract §2's rules (``secret_name_invalid``, as gateway-gateway's ``patch.ts``).

It is a test double, not server conformance evidence; the vendored connector fixtures
(``test_remote_secrets_conformance.py``) and the connector's own suite are that evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

TOKEN = "plane-token-1"
CLIENT_ID = "agent-client"
CLIENT_SECRET = "agent-secret"
INSTANCE = "inst-0001"
WORKLOAD_TOKEN = "workload-token-1"
LEVELS = ("profile", "instance", "owner", "tenant")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
AUTH_ID = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
VENDOR = (Path(__file__).resolve().parents[3] / "plugins" / "secret_store" / "remote" / "vendor" / "conformance")


def _reserved() -> dict[str, Any]:
    return json.loads((VENDOR / "secrets-reserved-names.json").read_text(encoding="utf-8-sig"))


def _level() -> dict[str, Any]:
    return {"env": {}, "auth": {}, "version": 0}


class StubSecretsPlane:
    def __init__(self) -> None:
        self.tenant = _level()
        self.owners: dict[str, dict[str, Any]] = {}
        self.instances: dict[str, dict[str, Any]] = {INSTANCE: {"owner": "principal:alice", "level": _level()}}
        self.profiles: dict[tuple[str, str], dict[str, Any]] = {}
        self.claims: dict[str, str] = {}      # turn claim -> instance it names (hosted turns)
        self.fail_status: Optional[int] = None
        self.conflict_next_patch = 0           # bump the profile level before the next N PATCHes
        self.before_get_reply: Optional[Any] = None   # called after a GET's view is taken, before its reply
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        reserved = _reserved()
        self._exact = set(reserved["globalEnvExact"]) | set(reserved["bootstrapEnvExact"]) \
            | set(reserved["bootstrapContract"]["envExact"])
        self._prefixes = tuple(reserved["globalEnvPrefixes"]) + tuple(reserved["bootstrapEnvPrefixes"])
        self._any_case = {n.upper() for n in reserved["launchOnlyAnyCase"]}
        self._auth_reserved = set(reserved["bootstrapContract"]["authProviders"])
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> StubSecretsPlane:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    # --- model --------------------------------------------------------------------------------

    def profile(self, name: str, instance: str = INSTANCE) -> dict[str, Any]:
        return self.profiles.setdefault((instance, name), _level())

    def owner_level(self, subject: str) -> dict[str, Any]:
        return self.owners.setdefault(subject, _level())

    def env_reserved(self, name: str) -> bool:
        return name in self._exact or name.startswith(self._prefixes) or name.upper() in self._any_case

    def view(self, profile: str, instance: str = INSTANCE) -> dict[str, Any]:
        inst = self.instances[instance]
        chain = [("profile", self.profile(profile, instance)), ("instance", inst["level"]),
                 ("owner", self.owners.get(inst["owner"], _level())), ("tenant", self.tenant)]
        env: dict[str, Any] = {}
        auth: dict[str, Any] = {}
        for level, doc in reversed(chain):
            for name, (value, version) in doc["env"].items():
                env[name] = {"value": value, "level": level, "version": version}
            for rid, (record, version) in doc["auth"].items():
                auth[rid] = {"record": copy.deepcopy(record), "level": level, "version": version}
        body = {"env": dict(sorted(env.items())), "auth": dict(sorted(auth.items())),
                "profileVersion": self.profile(profile, instance)["version"]}
        body["version"] = '"' + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32] + '"'
        return body

    def set_level(self, doc: dict[str, Any], env: Optional[dict[str, str]] = None,
                  unset: tuple[str, ...] = (), auth: Optional[dict[str, Any]] = None) -> None:
        """A write by another caller (a person's editor, NAS, a tenant admin), bumping versions."""
        with self.lock:
            for name, value in (env or {}).items():
                doc["env"][name] = (value, doc["env"].get(name, (None, 0))[1] + 1)
            for name in unset:
                doc["env"].pop(name, None)
            for rid, record in (auth or {}).items():
                if record is None:
                    doc["auth"].pop(rid, None)
                else:
                    doc["auth"][rid] = (copy.deepcopy(record), doc["auth"].get(rid, (None, 0))[1] + 1)
            doc["version"] += 1

    def patches(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "PATCH"]

    # --- HTTP ---------------------------------------------------------------------------------

    def _handler(self):
        plane = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args) -> None:
                pass

            def _send(self, status: int, body: Optional[dict[str, Any]] = None, etag: Optional[str] = None):
                raw = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                if etag:
                    self.send_header("ETag", etag)
                self.end_headers()
                self.wfile.write(raw)

            def _body(self) -> dict[str, Any]:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_POST(self):
                if self.path != "/token":
                    return self._send(404, {"error": "not_found", "message": "no route"})
                n = int(self.headers.get("Content-Length") or 0)
                form = urllib.parse.parse_qs(self.rfile.read(n).decode())
                if form.get("client_id") != [CLIENT_ID] or form.get("client_secret") != [CLIENT_SECRET]:
                    return self._send(401, {"error": "invalid_client"})
                return self._send(200, {"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3600})

            def _caller(self) -> Optional[str]:
                """The instance this request acts for, or None (refused)."""
                claim = self.headers.get("X-Hermes-Turn-Claim")
                if claim is not None:
                    if self.headers.get("Authorization") != f"Bearer {WORKLOAD_TOKEN}":
                        return None
                    return plane.claims.get(claim)
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return None
                instance = self.headers.get("X-Hermes-Instance-Id")
                return instance if instance in plane.instances else None

            def _route(self, method: str):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/v1/secrets/self":
                    return self._send(404, {"error": "not_found", "message": "no route"})
                profile = urllib.parse.parse_qs(parsed.query).get("profile", ["default"])[0]
                body = self._body() if method == "PATCH" else None
                with plane.lock:
                    plane.requests.append({
                        "method": method, "profile": profile, "body": copy.deepcopy(body),
                        "auth": self.headers.get("Authorization"),
                        "instance": self.headers.get("X-Hermes-Instance-Id"),
                        "claim": self.headers.get("X-Hermes-Turn-Claim"),
                        "if_none_match": self.headers.get("If-None-Match")})
                    if plane.fail_status is not None:
                        return self._send(plane.fail_status, {"error": "unavailable", "message": "stub outage"})
                    if (self.headers.get("Authorization") or "").strip() in ("", "Bearer"):
                        return self._send(401, {"error": "unauthenticated", "message": "no credential"})
                    instance = self._caller()
                    if instance is None:
                        return self._send(403, {"error": "secret_agent_unknown", "message": "unknown agent"})
                    if method != "GET":
                        return self._patch(profile, instance, body or {})
                    view = plane.view(profile, instance)
                # The GET's view is taken; a test may hold the reply here (the plane unlocked, so a
                # PATCH can land meanwhile) to deliver a response that is older than a later write's.
                gate = plane.before_get_reply
                if gate is not None:
                    gate()
                if self.headers.get("If-None-Match") == view["version"]:
                    return self._send(304, None, view["version"])
                return self._send(200, view, view["version"])

            def _patch(self, profile: str, instance: str, body: dict[str, Any]):
                if plane.conflict_next_patch > 0:
                    plane.conflict_next_patch -= 1
                    plane.profile(profile, instance)["version"] += 1   # another writer won the race
                prof = plane.profile(profile, instance)
                sets, unsets, auth = body.get("set") or {}, body.get("unset") or [], body.get("auth") or {}
                for kind, name, rule in [*(("env", n, ENV_NAME) for n in [*sets, *unsets]),
                                         *(("auth", r, AUTH_ID) for r in auth)]:
                    if not rule.fullmatch(name):
                        return self._send(400, {"error": "secret_name_invalid", "message": "invalid name",
                                                "kind": kind, "name": name})
                for name in [*sets, *unsets]:
                    if plane.env_reserved(name):
                        return self._send(400, {"error": "secret_name_reserved", "message": "reserved name",
                                                "kind": "env", "name": name})
                for rid in auth:
                    if rid in plane._auth_reserved:
                        return self._send(400, {"error": "secret_name_reserved", "message": "reserved name",
                                                "kind": "auth", "name": rid})
                if (sets or unsets) and "expectedProfileVersion" not in body:
                    return self._send(400, {"error": "secret_request_invalid", "message": "missing version"})
                if "expectedProfileVersion" in body and body["expectedProfileVersion"] != prof["version"]:
                    return self._send(409, {"error": "secret_version_conflict", "message": "stale",
                                            "profileVersion": prof["version"]})
                for rid, entry in auth.items():
                    current = prof["auth"].get(rid, (None, 0))[1]
                    if entry.get("expectedVersion") != current:
                        return self._send(409, {"error": "secret_version_conflict", "message": "stale",
                                                "provider": rid, "version": current})
                for name, value in sets.items():
                    prof["env"][name] = (value, prof["version"] + 1)
                for name in unsets:
                    prof["env"].pop(name, None)
                for rid, entry in auth.items():
                    if entry.get("record") is None:
                        prof["auth"].pop(rid, None)
                    else:
                        prof["auth"][rid] = (copy.deepcopy(entry["record"]), prof["version"] + 1)
                prof["version"] += 1
                view = plane.view(profile, instance)
                return self._send(200, view, view["version"])

            def do_GET(self):
                self._route("GET")

            def do_PATCH(self):
                self._route("PATCH")

        return Handler


def remote_env(plane: StubSecretsPlane) -> dict[str, str]:
    """The deployment for the remote secret store against *plane* (on-prem IdP credential)."""
    return {
        "HERMES_SECRET_STORE": "remote",
        "HERMES_SECRETS_REMOTE_URL": plane.url,
        "HERMES_CONFIG_INSTANCE_ID": INSTANCE,
        "GATEWAY_RELAY_IDP_TOKEN_URL": f"{plane.url}/token",
        "GATEWAY_RELAY_IDP_CLIENT_ID": CLIENT_ID,
        "GATEWAY_RELAY_IDP_CLIENT_SECRET": CLIENT_SECRET,
    }
