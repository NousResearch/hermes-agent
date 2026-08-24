#!/usr/bin/env python3
"""Credential-free live OAuth/MCP persistence probe (no production mocks).

Run with the repository's Python environment:
  python evals/desktop_mcp_oauth/backend_http_fixture.py --repo . --output receipt.json

The parent creates an ephemeral HOME and re-execs with an environment allowlist.
Only the redacted receipt survives; token files are verified then deleted with HOME.
This exercises production session handlers, not Electron or gateway transport.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit


class Provider(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), ProviderHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.events = []
        self.clients = {}
        self.codes = {}
        self.tokens = set()
        self.pkce_verified = 0


class ProviderHandler(BaseHTTPRequestHandler):
    server: Provider

    def log_message(self, format, *args):
        pass  # Never log callback queries, codes or bearer headers.

    def reply(self, status, payload=None, headers=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        self.server.events.append(["GET", path])
        if path.startswith("/.well-known/oauth-protected-resource"):
            return self.reply(200, {"resource": self.server.origin + "/mcp",
                                    "authorization_servers": [self.server.origin],
                                    "scopes_supported": ["tools:read"]})
        if path.startswith("/.well-known/oauth-authorization-server"):
            return self.reply(200, {
                "issuer": self.server.origin,
                "authorization_endpoint": self.server.origin + "/authorize",
                "token_endpoint": self.server.origin + "/token",
                "registration_endpoint": self.server.origin + "/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "code_challenge_methods_supported": ["S256"],
                "scopes_supported": ["tools:read"],
            })
        if path == "/authorize":
            query = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
            client = self.server.clients.get(query.get("client_id"))
            if (not client or query.get("redirect_uri") not in client["redirect_uris"]
                    or query.get("code_challenge_method") != "S256"):
                return self.reply(400, {"error": "invalid_request"})
            code = secrets.token_urlsafe(24)
            self.server.codes[code] = query
            target = query["redirect_uri"] + "?" + urlencode({"code": code, "state": query["state"]})
            return self.reply(302, headers={"Location": target})
        self.reply(405 if path == "/mcp" else 404)

    def do_POST(self):  # noqa: N802
        path = urlsplit(self.path).path
        self.server.events.append(["POST", path])
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/register":
            client = json.loads(body)
            client["client_id"] = secrets.token_urlsafe(16)
            client["token_endpoint_auth_method"] = "none"
            self.server.clients[client["client_id"]] = client
            return self.reply(201, client)
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
            auth = self.server.codes.pop(form.get("code"), None)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(
                form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
            if (not auth or challenge != auth["code_challenge"]
                    or form.get("redirect_uri") != auth["redirect_uri"]
                    or form.get("client_id") != auth["client_id"]):
                return self.reply(400, {"error": "invalid_grant"})
            self.server.pkce_verified += 1
            token = secrets.token_urlsafe(32)
            self.server.tokens.add(token)
            return self.reply(200, {"access_token": token, "token_type": "Bearer",
                                    "expires_in": 3600, "scope": "tools:read",
                                    "refresh_token": secrets.token_urlsafe(32)})
        if path != "/mcp":
            return self.reply(404)
        if self.headers.get("Authorization", "").removeprefix("Bearer ") not in self.server.tokens:
            return self.reply(401, {"error": "unauthorized"}, {"WWW-Authenticate":
                f'Bearer resource_metadata="{self.server.origin}/.well-known/oauth-protected-resource"'})
        request = json.loads(body)
        method = request["method"]
        self.server.events.append(["MCP", method])
        if "id" not in request:
            return self.reply(202)
        results = {
            "initialize": {"protocolVersion": request.get("params", {}).get("protocolVersion", "2025-03-26"),
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "local-oauth-fixture", "version": "1"}},
            "tools/list": {"tools": [{"name": "fixture_ping", "description": "Local fixture ping",
                                       "inputSchema": {"type": "object", "properties": {}}}]},
            "ping": {},
        }
        if method not in results:
            return self.reply(200, {"jsonrpc": "2.0", "id": request["id"],
                                    "error": {"code": -32601, "message": "Method not found"}})
        self.reply(200, {"jsonrpc": "2.0", "id": request["id"], "result": results[method]})

    def do_DELETE(self):  # noqa: N802
        self.reply(200)


class CallbackServer(ThreadingHTTPServer):
    callback: dict[str, str]


class CallbackHandler(BaseHTTPRequestHandler):
    server: CallbackServer

    def log_message(self, format, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.server.callback = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Local fixture callback captured")


@contextlib.contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def eventually(predicate, label, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(label)


def run_probe(repo, receipt):
    sys.path.insert(0, str(repo))
    logging.disable(logging.CRITICAL)
    import httpx
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tui_gateway import mcp_oauth_sessions as sessions
    from tools.mcp_oauth import HermesTokenStorage

    owner = Path(os.environ["HERMES_HOME"])
    other = owner.parent / "other-profile"
    other.mkdir()
    checks = receipt["checks"]

    def check(name, result):
        checks[name] = bool(result)
        if not result:
            raise AssertionError(name)

    with serving(Provider()) as provider, serving(CallbackServer(
            ("127.0.0.1", 0), CallbackHandler)) as callback, httpx.Client(trust_env=False) as browser:
        cfg = {"url": provider.origin + "/mcp", "auth": "oauth",
               "oauth": {"cimd": False, "scope": "tools:read", "timeout": 30}}
        redirect = f"http://127.0.0.1:{callback.server_port}/callback"

        def begin(name, remote=True):
            return sessions.start_flow(str(owner), name, cfg, url_timeout=25,
                                       client_redirect_uri=redirect if remote else None)

        def done(flow, name):
            sid = flow["session_id"]
            eventually(lambda: sessions._sessions[sid]["flow"].worker_done, "worker completed")
            return sessions.poll_flow(sid, name)

        flow = begin("positive")
        sid = flow["session_id"]
        check("remote_does_not_bind_backend_listener", sessions._sessions[sid]["httpd"] is None)
        check("initial_poll_pending", sessions.poll_flow(sid, "positive")["status"] == "pending")
        check("wrong_state_rejected", not sessions.deliver_callback_flow(
            sid, "positive", code="invalid", state="invalid")["ok"])
        check("wrong_state_keeps_pending", sessions.poll_flow(sid, "positive")["status"] == "pending")
        check("wrong_server_poll_rejected", sessions.poll_flow(sid, "wrong")["status"] == "error")
        check("wrong_server_callback_rejected", not sessions.deliver_callback_flow(
            sid, "wrong", code="invalid", state="invalid")["ok"])
        check("wrong_owner_cancel_rejected", not sessions.cancel_flow(sid, "positive", str(other))["ok"])
        check("wrong_owner_did_not_cancel", sessions.poll_flow(sid, "positive")["status"] == "pending")
        response = browser.get(flow["auth_url"], follow_redirects=True)
        check("real_http_callback_received", response.status_code == 200 and bool(callback.callback.get("code")))
        captured = callback.callback
        check("correct_callback_accepted", sessions.deliver_callback_flow(sid, "positive", **captured)["ok"])
        result = done(flow, "positive")
        receipt["positive_status"] = result["status"]
        check("positive_approved", result["status"] == "approved")
        check("real_mcp_tools_discovered", any(t["name"] == "fixture_ping" for t in result["tools"]))
        check("callback_replay_rejected", not sessions.deliver_callback_flow(sid, "positive", **captured)["ok"])
        token_path = owner / "mcp-tokens" / "positive.json"
        stored = json.loads(token_path.read_text(encoding="utf-8"))
        check("disk_token_matches_provider", stored["access_token"] in provider.tokens)
        check("disk_refresh_token_and_absolute_expiry", bool(stored.get("refresh_token")) and stored["expires_at"] > time.time())
        check("token_file_private", token_path.stat().st_mode & 0o777 == 0o600)
        reloaded = asyncio.run(HermesTokenStorage("positive", hermes_home=owner).get_tokens())
        check("fresh_storage_reload", reloaded.access_token in provider.tokens)
        check("wrong_profile_has_no_token", not (other / "mcp-tokens" / "positive.json").exists())
        check("config_saved_to_owner", "positive" in (owner / "config.yaml").read_text(encoding="utf-8"))
        exchanges = provider.pkce_verified
        cold = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--repo", str(repo), "--output", os.devnull, "--cold-probe"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=40, check=False)
        check("fresh_process_authenticated_mcp_from_disk", cold.returncode == 0)
        check("fresh_process_did_not_reauthorize", provider.pkce_verified == exchanges)

        flow = begin("cancelled")
        sid = flow["session_id"]
        exchanges = provider.pkce_verified
        check("owner_cancel_accepted", sessions.cancel_flow(sid, "cancelled", str(owner))["ok"])
        check("cancel_worker_terminated", done(flow, "cancelled")["status"] == "error")
        check("cancel_did_not_exchange_or_persist", provider.pkce_verified == exchanges and
              not (owner / "mcp-tokens" / "cancelled.json").exists())
        check("callback_after_cancel_rejected", not sessions.deliver_callback_flow(
            sid, "cancelled", code="invalid", state="invalid")["ok"])

        flow = begin("loopback", remote=False)
        check("backend_listener_http_success", browser.get(flow["auth_url"], follow_redirects=True).status_code == 200)
        check("backend_listener_approved", done(flow, "loopback")["status"] == "approved")
        check("backend_listener_closed", eventually(
            lambda: sessions._sessions[flow["session_id"]]["httpd"] is None,
            "backend listener shutdown completed"))

        # Even possession of valid session/state cannot cross the resolved home.
        flow = begin("owner_boundary")
        browser.get(flow["auth_url"], follow_redirects=True)
        override = set_hermes_home_override(other)
        try:
            foreign_poll = sessions.poll_flow(flow["session_id"], "owner_boundary")
            foreign_callback = sessions.deliver_callback_flow(flow["session_id"], "owner_boundary", **callback.callback)
        finally:
            reset_hermes_home_override(override)
        receipt["ownership_boundary"] = {
            "cancel_owner_enforced": checks["wrong_owner_cancel_rejected"],
            "cross_profile_poll_rejected": foreign_poll["status"] == "error",
            "cross_profile_callback_rejected_with_session_and_state": not foreign_callback["ok"],
        }
        check("wrong_owner_poll_rejected", foreign_poll["status"] == "error" and "auth_url" not in foreign_poll)
        check("wrong_owner_callback_rejected", not foreign_callback["ok"])
        check("owner_callback_after_foreign_attempt_accepted", sessions.deliver_callback_flow(
            flow["session_id"], "owner_boundary", **callback.callback)["ok"])
        check("boundary_probe_worker_completed", done(flow, "owner_boundary")["status"] == "approved")
        check("boundary_exchange_still_persists_only_to_owner", (owner / "mcp-tokens" / "owner_boundary.json").exists()
              and not (other / "mcp-tokens" / "owner_boundary.json").exists())
        receipt["http_events"] = provider.events
        receipt["pkce_exchanges_verified"] = provider.pkce_verified
        check("real_discovery_dcr_token_http", all(any(path == target for _, path in provider.events)
              for target in ["/.well-known/oauth-protected-resource", "/register", "/token"]))
        receipt["persisted_files_verified_then_removed"] = sorted(p.name for p in (owner / "mcp-tokens").iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if args.cold_probe:
        sys.path.insert(0, str(repo))
        logging.disable(logging.CRITICAL)
        from hermes_cli.mcp_config import _get_mcp_servers, _probe_single_server
        from tools.mcp_oauth import suppress_interactive_oauth
        with suppress_interactive_oauth():
            tools = _probe_single_server("positive", _get_mcp_servers()["positive"])
        return 0 if any(name == "fixture_ping" for name, _ in tools) else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.isolated_worker:
        with tempfile.TemporaryDirectory(prefix="hermes-oauth-http-") as temp:
            home = Path(temp)
            (home / ".hermes").mkdir()
            env = {"PATH": os.environ.get("PATH", ""), "HOME": temp,
                   "HERMES_HOME": str(home / ".hermes"), "LANG": "C.UTF-8", "TZ": "UTC",
                   "PYTHONNOUSERSITE": "1"}
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--repo", str(repo), "--output", str(output), "--isolated-worker"],
                env=env, cwd=temp, stdin=subprocess.DEVNULL, timeout=180, check=False)
        print(json.dumps({"receipt": str(output), "exit_code": completed.returncode, "isolated_home_removed": True}))
        return completed.returncode
    receipt = {"checks": {}, "repo": str(repo), "fidelity":
               "Real production OAuth session functions + real HTTP discovery/DCR/PKCE/token/MCP/callback + real disk persistence; no production mocks; not gateway RPC transport or Electron",
               "isolated_home": True, "credentials": "ephemeral local fixture only; no token material in receipt"}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            run_probe(repo, receipt)
        receipt["status"] = "passed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        receipt["failed_check"] = next((k for k, v in receipt["checks"].items() if not v), None)
        # Assertions are our fixed labels, not provider bodies or secret-bearing URLs.
        if isinstance(exc, AssertionError):
            receipt["assertion"] = str(exc)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
