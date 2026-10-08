"""The dashboard fire webhook verifies with the launch home's Chronos settings.

The hosting control plane writes ``cron.chronos.*`` into the launch home only, and the gateway
re-verifies every forwarded fire against those same launch-home settings. The dashboard must not
pick them up from whatever profile scope the request happens to run in (or from the job's
profile): that profile has no JWKS on a hosted instance and every fire would be refused.

Real RS256 signing and the real verifier. Regression for #69715.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from starlette.requests import Request

import hermes_cli.web_server_cron as _web_server_cron
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

AUD = "agent:inst-dashboard"
ISS = "https://portal.example.test"


@pytest.fixture(scope="module")
def rsa_keys():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return priv, pub


def _mint(priv: str) -> str:
    import jwt

    now = int(time.time())
    return jwt.encode(
        {"aud": AUD, "iss": ISS, "purpose": "cron_fire", "iat": now, "nbf": now - 5, "exp": now + 300},
        priv,
        algorithm="RS256",
    )


@pytest.fixture
def homes(rsa_keys):
    _, pub = rsa_keys
    root = Path(os.environ["HERMES_HOME"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(json.dumps({
        "cron": {"provider": "chronos", "chronos": {
            "portal_url": ISS, "expected_audience": AUD, "nas_jwks_url": pub,
            "callback_url": "https://agent.example.test"}},
    }), encoding="utf-8")
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(json.dumps({"model": {"default": "x"}}), encoding="utf-8")
    return root, worker


def _request(token: str, body: dict) -> Request:
    payload = json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http", "method": "POST", "path": "/api/cron/fire", "query_string": b"",
        "headers": [(b"authorization", f"Bearer {token}".encode()), (b"content-type", b"application/json")],
    }
    return Request(scope, receive)


def test_settings_come_from_launch_home_under_a_profile_scope(homes, rsa_keys):
    from plugins.cron_providers.chronos.verify import fire_verification_settings

    _, pub = rsa_keys
    _root, worker = homes
    token = set_hermes_home_override(str(worker))
    try:
        settings = fire_verification_settings()
    finally:
        reset_hermes_home_override(token)
    assert settings == {"expected_audience": AUD, "jwks_or_key": pub, "issuer": ISS}


def test_dashboard_webhook_verifies_with_launch_home_under_profile_scope(homes, rsa_keys, monkeypatch):
    from hermes_cli.web_routers.cron import cron_fire_webhook

    priv, _ = rsa_keys
    _root, worker = homes
    forwarded = []

    async def fake_forward(profile, job_id, auth):
        forwarded.append((profile, job_id))
        return 202, {"status": "accepted"}

    monkeypatch.setattr(_web_server_cron, "_find_cron_job_profile", lambda jid: "worker")
    monkeypatch.setattr(_web_server_cron, "_forward_cron_fire_to_gateway", fake_forward)

    async def run():
        # The request runs inside the worker profile's scope (the job's profile).
        token = set_hermes_home_override(str(worker))
        try:
            return await cron_fire_webhook(_request(_mint(priv), {"job_id": "job-worker"}))
        finally:
            reset_hermes_home_override(token)

    resp = asyncio.run(run())
    assert resp.status_code == 202, resp.body
    assert forwarded == [("worker", "job-worker")]
