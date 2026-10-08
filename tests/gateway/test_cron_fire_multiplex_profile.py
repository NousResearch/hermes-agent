"""Chronos fire tokens for a non-default profile on a multiplexed gateway.

The hosting control plane writes ``cron.chronos.*`` (audience, JWKS URL, issuer) into the launch
home's config.yaml only. A fire for a non-default profile reaches the gateway as
``/p/<profile>/api/cron/fire``; the profile-prefix middleware enters that profile's scope before
the handler runs, so a plain ``load_config()`` returned the profile's config, which has no JWKS,
and every fire for that profile was refused with 401. The fire-token identity belongs to the
instance, so the handler must read it from the launch home whatever scope it runs in.

Real RS256 signing and the real verifier; only the cron provider is a spy. Regression for #69715.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope as ss
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

AUD = "agent:inst-multiplex"
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


class _SpyProvider:
    def __init__(self):
        self.fired = []

    def claim_fire(self, job_id):
        return {"id": job_id, "execution_id": f"exec-{job_id}"}

    def fire_claimed(self, job, *, adapters=None, loop=None):
        self.fired.append(job["id"])
        return True


@pytest.fixture
def homes(rsa_keys):
    """Launch home seeded like a hosted instance; a profile whose config has no chronos keys."""
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


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _multiplexed_app(monkeypatch, root: Path, worker: Path) -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True), adapters={})
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", root), ("worker", worker)],
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: root if name == "default" else worker,
    )
    ss.set_multiplex_active(True)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/api/cron/fire", adapter._handle_cron_fire)
    app.router.add_post("/p/{profile}/api/cron/fire", adapter._handle_cron_fire)
    return app


async def _wait_fired(spy: _SpyProvider) -> None:
    for _ in range(100):
        if spy.fired:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_profile_prefixed_fire_verifies_against_launch_home(homes, rsa_keys, monkeypatch):
    root, worker = homes
    priv, _ = rsa_keys
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    app = _multiplexed_app(monkeypatch, root, worker)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/worker/api/cron/fire",
            headers={"Authorization": f"Bearer {_mint(priv)}"},
            json={"job_id": "job-worker"},
        )
        assert resp.status == 202, await resp.text()
        await _wait_fired(spy)
    assert spy.fired == ["job-worker"]


@pytest.mark.asyncio
async def test_profile_config_cannot_replace_launch_home_identity(homes, rsa_keys, monkeypatch):
    """A profile's own cron.chronos values never decide what the instance accepts."""
    root, worker = homes
    priv, _ = rsa_keys
    (worker / "config.yaml").write_text(json.dumps({
        "cron": {"chronos": {"expected_audience": "agent:someone-else", "nas_jwks_url": "",
                             "portal_url": "https://evil.example.test"}},
    }), encoding="utf-8")
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    app = _multiplexed_app(monkeypatch, root, worker)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/worker/api/cron/fire",
            headers={"Authorization": f"Bearer {_mint(priv)}"},
            json={"job_id": "job-worker"},
        )
        assert resp.status == 202, await resp.text()
        await _wait_fired(spy)
        forged = await cli.post(
            "/p/worker/api/cron/fire",
            headers={"Authorization": "Bearer not-a-jwt"},
            json={"job_id": "job-worker"},
        )
        assert forged.status == 401
    assert spy.fired == ["job-worker"]
