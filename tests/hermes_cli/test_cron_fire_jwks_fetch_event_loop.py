"""A Chronos fire arriving at the dashboard must not stop it from answering other requests.

The dashboard verifies the NAS-minted fire token before forwarding the fire to the gateway. With
a cold key cache the verifier fetches the JWKS document from the portal; the dashboard serves every
page, API call and the embedded terminal from one event loop, which must keep answering while that
fetch is in flight.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import uvicorn

from tests.waiting import wait_until

AUDIENCE = "agent:inst-test"
ISSUER = "https://portal.test"


@pytest.fixture
def dashboard():
    """The real dashboard app served by uvicorn on loopback, as ``hermes dashboard`` serves it."""
    from hermes_cli.web_server import app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error", timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_until(lambda: server.started or not thread.is_alive(), what="uvicorn to start")
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


@pytest.fixture
def signing_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def portal(signing_key):
    """A JWKS endpoint that holds each response until the test releases it."""
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(signing_key.public_key()))
    jwks = json.dumps({"keys": [{**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}]}).encode()
    requested, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.set()
            release.wait(30)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(jwks)))
            self.end_headers()
            self.wfile.write(jwks)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/.well-known/jwks.json", requested, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_dashboard_answers_while_a_fire_token_waits_for_the_jwks_fetch(dashboard, portal, signing_key):
    import jwt

    from hermes_cli.config import load_config, save_config

    jwks_url, jwks_requested, release_jwks = portal
    cfg = load_config()
    cfg.setdefault("cron", {})["chronos"] = {"expected_audience": AUDIENCE, "nas_jwks_url": jwks_url, "portal_url": ISSUER}
    save_config(cfg)
    now = int(time.time())
    token = jwt.encode({"aud": AUDIENCE, "iss": ISSUER, "purpose": "cron_fire", "iat": now, "exp": now + 300},
                       signing_key, algorithm="RS256", headers={"kid": "k1"})
    fire: dict = {}

    def nas_fires():
        with httpx.Client(base_url=dashboard, timeout=60) as client:
            fire["response"] = client.post("/api/cron/fire", json={"job_id": "no-such-job"},
                                           headers={"Authorization": f"Bearer {token}"})

    nas = threading.Thread(target=nas_fires, daemon=True)
    nas.start()
    try:
        assert jwks_requested.wait(10), "the fire never reached the JWKS fetch"
        with httpx.Client(base_url=dashboard, timeout=5) as client:
            try:
                health = client.get("/api/health")
            except httpx.TimeoutException:
                pytest.fail("GET /api/health got no answer while POST /api/cron/fire waited for the JWKS fetch")
        assert health.status_code == 200
    finally:
        release_jwks.set()
        nas.join(60)
    assert fire["response"].status_code == 200
    assert fire["response"].json() == {"status": "gone", "job_id": "no-such-job"}
