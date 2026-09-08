"""Local RFC 8628 wire fixture; no external credentials or provider claims.

Run: python evals/mcp_device_flow.py --repo /path/to/checkout
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


@contextmanager
def oauth_fixture(mode="success"):
    wire = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, headers=None):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            wire.append({"path": self.path, "method": "GET"})
            if self.path == "/mcp":
                if self.headers.get("Authorization") == "Bearer fixture-access":
                    return self.reply(200, {"authenticated": True})
                return self.reply(401, {}, {"WWW-Authenticate": f'Bearer resource_metadata="{base}/prm"'})
            if self.path == "/prm" or "oauth-protected-resource" in self.path:
                return self.reply(200, {"resource": base + ("/wrong" if mode == "resource" else "/mcp"),
                                        "authorization_servers": [base]})
            if "oauth-authorization-server" in self.path:
                metadata = {"issuer": base + ("/wrong" if mode == "issuer" else ""),
                            "authorization_endpoint": base + "/authorize", "token_endpoint": base + "/token",
                            "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"],
                            "grant_types_supported": [DEVICE_GRANT, "refresh_token"]}
                if mode != "unsupported":
                    metadata["device_authorization_endpoint"] = base + "/device"
                if mode != "preregistered":
                    metadata["registration_endpoint"] = base + "/register"
                else:
                    metadata.pop("authorization_endpoint")
                return self.reply(200, metadata)
            self.reply(404, {})

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.loads(raw) if "json" in self.headers.get("Content-Type", "") else {
                key: value[0] for key, value in parse_qs(raw.decode()).items()}
            wire.append({"path": self.path, "method": "POST", "data": data, "at": time.monotonic()})
            if self.path == "/register":
                if DEVICE_GRANT not in data.get("grant_types", []):
                    return self.reply(400, {"error": "invalid_client_metadata"})
                return self.reply(201, {**data, "client_id": "fixture-client"})
            if self.path == "/device":
                return self.reply(200, {"device_code": "fixture-device-secret", "user_code": "TEST-CODE",
                                        "verification_uri": base + "/verify", "interval": 0.01,
                                        "expires_in": 0.1 if mode == "expiry" else 30})
            if self.path == "/token":
                polls = sum(row["path"] == "/token" for row in wire)
                if mode == "preregistered" and data.get("client_secret") != "fixture-client-secret":
                    return self.reply(401, {"error": "invalid_client"})
                if mode in {"denied", "expiry", "malformed"}:
                    if mode == "malformed":
                        return self.reply(200, {"access_token": {"secret": "fixture-device-secret"}})
                    return self.reply(400, {"error": "access_denied" if mode == "denied" else "authorization_pending",
                                            "error_description": "fixture-device-secret MUST NOT BE PRINTED"})
                if polls <= 2 and mode == "success":
                    return self.reply(400, {"error": "authorization_pending" if polls == 1 else "slow_down"})
                return self.reply(200, {"access_token": "fixture-access", "refresh_token": "fixture-refresh",
                                        "token_type": "Bearer", "expires_in": 3600})
            if self.path == "/mcp":
                if self.headers.get("Authorization") != "Bearer fixture-access":
                    return self.reply(401, {}, {"WWW-Authenticate": f'Bearer resource_metadata="{base}/prm"'})
                method = data.get("method")
                if "id" not in data:
                    return self.reply(202, {})
                result = {"initialize": {"protocolVersion": data.get("params", {}).get("protocolVersion"),
                                           "capabilities": {"tools": {}},
                                           "serverInfo": {"name": "fixture", "version": "1"}},
                          "tools/list": {"tools": [{"name": "fixture_echo", "description": "Fixture tool",
                                                     "inputSchema": {"type": "object", "properties": {}}}]}}
                return self.reply(200, {"jsonrpc": "2.0", "id": data["id"], "result": result.get(method, {})})
            self.reply(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield base, wire
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def run_cli(repo, mode):
    with tempfile.TemporaryDirectory(prefix="hermes-device-wire-") as directory, oauth_fixture(mode) as (base, wire):
        home = Path(directory)
        oauth = {"flow": "device", "cimd": False, "scope": "fixture.read", "timeout": 15}
        if mode == "preregistered":
            oauth.update(client_id="fixture-client", client_secret="fixture-client-secret")
        config = {"mcp_servers": {"fixture": {"url": base + "/mcp", "auth": "oauth", "oauth": oauth}}}
        (home / "config.yaml").write_text(json.dumps(config))
        previous = {}
        if mode == "persistence":
            token_dir = home / "mcp-tokens"
            token_dir.mkdir()
            previous = {"fixture.json": '{"access_token":"old-fixture","token_type":"Bearer"}',
                        "fixture.client.json": '{"client_id":"old-client"}',
                        "fixture.meta.json": '{"issuer":"https://old.example"}'}
            for filename, value in previous.items():
                (token_dir / filename).write_text(value)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("HERMES_") and not any(part in key for part in ("API_KEY", "TOKEN", "SECRET"))}
        env.update(HOME=str(home), HERMES_HOME=str(home), PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
        command = ["reauth", "fixture"] if mode == "preregistered" else ["login", "fixture", "--flow", "device"]
        argv = [sys.executable, "-m", "hermes_cli.main", "mcp", *command]
        if mode == "persistence":
            # Inject a filesystem write error after real registration/metadata writes.
            argv = [sys.executable, "-c", '''
from tools import mcp_oauth
write = mcp_oauth._write_json
def fail_token(path, data):
    if path.name == "fixture.json":
        raise OSError("fixture disk failure")
    return write(path, data)
mcp_oauth._write_json = fail_token
from hermes_cli.main import main
main()
''', "mcp", *command]
        result = subprocess.run(argv,
                                cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=40)
        token_path = home / "mcp-tokens" / "fixture.json"
        refresh_output = None
        if token_path.exists() and not previous:
            tokens = json.loads(token_path.read_text())
            tokens["expires_at"] = time.time() - 60
            token_path.write_text(json.dumps(tokens))
            refreshed = subprocess.run([sys.executable, "-m", "hermes_cli.main", "mcp", "test", "fixture"],
                                       cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
            refresh_output = refreshed.stdout + refreshed.stderr
        return {"mode": mode, "returncode": result.returncode, "output": result.stdout + result.stderr,
                "token_persisted": token_path.exists(), "refresh_output": refresh_output, "wire": wire,
                "state_preserved": all((home / "mcp-tokens" / k).read_text() == v for k, v in previous.items())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mode", default="success")
    args = parser.parse_args()
    print(json.dumps(run_cli(args.repo, args.mode), indent=2))
