"""Local fake OAuth IdP on 127.0.0.1 for the PKCE plugin helper (tests + live evidence).

/authorize  → 302 to redirect_uri?code=…&state=…  (records code_challenge)
/token      → grant_type=authorization_code: verifies S256(code_verifier) == challenge, issues tokens
              grant_type=refresh_token: rotates (old refresh token becomes invalid → 400)
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse


class FakeIdP:
    def __init__(self) -> None:
        self.codes: dict[str, str] = {}          # code -> code_challenge
        self.refresh_tokens: set[str] = set()
        self.token_requests: list[dict] = []
        self.issued = 0
        idp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: A003
                return

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/authorize":
                    self.send_response(404); self.end_headers(); return
                q = parse_qs(parsed.query)
                assert q["code_challenge_method"] == ["S256"], q
                code = secrets.token_urlsafe(16)
                idp.codes[code] = q["code_challenge"][0]
                state = idp.override_state if idp.override_state is not None else q["state"][0]
                self.send_response(302)
                self.send_header("Location", f"{q['redirect_uri'][0]}?{urlencode({'code': code, 'state': state})}")
                self.end_headers()

            def do_POST(self):  # noqa: N802
                if urlparse(self.path).path != "/token":
                    self.send_response(404); self.end_headers(); return
                body = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
                form = {k: v[0] for k, v in body.items()}
                idp.token_requests.append(form)
                grant = form.get("grant_type")
                if grant == "authorization_code":
                    challenge = idp.codes.pop(form.get("code", ""), None)
                    digest = hashlib.sha256(form.get("code_verifier", "").encode()).digest()
                    if challenge is None or base64.urlsafe_b64encode(digest).decode().rstrip("=") != challenge:
                        return self._json(400, {"error": "invalid_grant"})
                elif grant == "refresh_token":
                    if form.get("refresh_token") not in idp.refresh_tokens:
                        return self._json(400, {"error": "invalid_grant"})
                    idp.refresh_tokens.discard(form["refresh_token"])
                else:
                    return self._json(400, {"error": "unsupported_grant_type"})
                idp.issued += 1
                refresh = f"rt-{idp.issued}-{secrets.token_hex(4)}"
                idp.refresh_tokens.add(refresh)
                self._json(200, {"access_token": f"at-{idp.issued}-{secrets.token_hex(4)}",
                                 "refresh_token": refresh, "expires_in": 3600, "token_type": "Bearer"})

            def _json(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.override_state: str | None = None
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def start(self) -> "FakeIdP":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


PLUGIN_TEMPLATE = '''
from providers import register_provider
from providers.base import ProviderProfile
from hermes_cli.auth_oauth_pkce_plugin import OAuthPKCEConfig, pkce_auth_handler, pkce_refresh_credential

cfg = OAuthPKCEConfig(client_id="hermes-example", authorize_url="{base}/authorize", token_url="{base}/token",
                      scopes=("inference",), timeout_seconds=20)
register_provider(ProviderProfile(name="example-pkce", auth_type="oauth_external",
                                  base_url="https://example.invalid/v1", fallback_models=("example-model",),
                                  auth_handler=pkce_auth_handler(cfg), refresh_credential=pkce_refresh_credential(cfg)))
'''
