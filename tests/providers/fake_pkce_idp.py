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
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.codes: dict[str, str] = {}          # code -> code_challenge
        self.refresh_tokens: set[str] = set()
        self.token_requests: list[dict] = []
        self.issued = 0
        idp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_GET(self):
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

            def do_POST(self):
                if urlparse(self.path).path != "/token":
                    self.send_response(404); self.end_headers(); return
                body = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
                form = {k: v[0] for k, v in body.items()}
                idp.token_requests.append(form)
                if idp.token_answer is not None:
                    return self._json(*idp.token_answer)
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
                payload = {"access_token": f"at-{idp.issued}-{secrets.token_hex(4)}",
                           "refresh_token": refresh, "expires_in": 3600, "token_type": "Bearer"}
                if idp.granted_scope is not None:
                    payload["scope"] = idp.granted_scope
                self._json(200, payload)

            def _json(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.override_state: str | None = None
        self.granted_scope: str | None = None  # ``scope`` echoed in token responses (None = omitted)
        self.token_answer: tuple[int, dict] | None = None  # (status, JSON) answered to every token POST instead
        self.server = HTTPServer((host, port), Handler)
        self.base = f"http://{host}:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def start(self, *, tls_dir=None, name: str | None = None) -> FakeIdP:
        """Serve; with *tls_dir*, over HTTPS with a throwaway CA written there (``self.ca_pem`` to trust).
        *name* is the DNS name the certificate (and ``self.base``) use instead of the bound address."""
        if tls_dir is not None:
            import ssl
            from pathlib import Path

            from tests.fakes.providers.oauth_token_server import make_test_ca

            host, port = self.server.server_address[:2]
            ca = make_test_ca(Path(tls_dir), [name or host])
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(ca.leaf_pem, ca.leaf_key)
            self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
            self.base = f"https://{name or host}" + ("" if port == 443 else f":{port}")
            self.ca_pem = ca.ca_pem
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def landing_address(self, authorize_url: str) -> str:
        """The address a browser ends on after consent: this IdP's 302 ``Location``, not followed. A user
        copies it from the address bar when the redirect cannot reach Hermes."""
        import http.client
        import ssl

        parsed = urlparse(authorize_url)
        conn = (http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=5,
                                            context=ssl.create_default_context(cafile=str(self.ca_pem)))
                if parsed.scheme == "https" else http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5))
        try:
            conn.request("GET", f"{parsed.path}?{parsed.query}")
            response = conn.getresponse()
            assert response.status == 302, response.status
            return response.getheader("Location")
        finally:
            conn.close()


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
