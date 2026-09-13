"""Chronos callbacks keep profile authentication across the scheduler lifecycle."""

import threading
import time
from pathlib import Path

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import web_server_cron as cron_web
from plugins.cron_providers.chronos import ChronosCronScheduler
from plugins.cron_providers.chronos import verify as fire_verify


@pytest.fixture
def fire_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    homes = {"default": root, "work": root / "profiles" / "work"}
    for profile, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"cron": {"chronos": {
            "expected_audience": f"agent:{profile}",
            "portal_url": f"https://{profile}.example",
            "nas_jwks_url": public_key,
            "callback_url": "https://agent.example/api/cron/fire",
        }}}), encoding="utf-8")

    def token(profile="work", **overrides):
        claims = {"aud": f"agent:{profile}", "iss": f"https://{profile}.example",
                  "exp": int(time.time()) + 600, "purpose": "cron_fire"}
        return jwt.encode({**claims, **overrides}, key, algorithm="RS256")

    def create(profile="work"):
        with cron_web._cron_store_scope(homes[profile]) as jobs:
            return jobs.create_job(prompt="inspect status", schedule="every 1h")

    forwarded = []

    async def forward(profile, job_id, authorization):
        forwarded.append((profile, job_id, authorization))
        return 202, {"status": "accepted", "job_id": job_id}

    monkeypatch.setattr(cron_web, "_forward_cron_fire_to_gateway", forward)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    return homes, token, create, forwarded


def test_profile_fire_auth_survives_arm_reconcile_and_cancel(fire_profiles):
    homes, token, create, forwarded = fire_profiles
    new_job, existing_job, failed_job = create(), create(), create()
    provider = ChronosCronScheduler()
    issued_token = token()

    with TestClient(web_server.app) as client:
        def fire(job_id):
            # The public dashboard runs in its own default-profile context.
            with cron_web._cron_store_scope(homes["default"]):
                return client.post("/api/cron/fire", json={"job_id": job_id},
                                   headers={"Authorization": f"Bearer {issued_token}"})

        class NasClient:
            def __init__(self):
                self.armed = {existing_job["id"]: existing_job["next_run_at"]}
                self.provisions = []

            def provision(self, *, job_id, fire_at, **kwargs):
                # NAS can call back before its provision response reaches Hermes.
                self.provisions.append(job_id)
                assert fire(job_id).status_code == 202
                if job_id == failed_job["id"]:
                    raise TimeoutError("provision response lost")
                self.armed[job_id] = fire_at

            def list_armed(self):
                return [{"job_id": job_id, "fire_at": fire_at}
                        for job_id, fire_at in self.armed.items()]

            def cancel(self, *, job_id):
                self.armed.pop(job_id, None)

        nas = NasClient()
        provider._client = nas
        with cron_web._cron_store_scope(homes["work"]):
            provider.register_job(new_job)
            with pytest.raises(TimeoutError, match="provision response lost"):
                provider.register_job(failed_job)
        assert fire(failed_job["id"]).status_code == 202

        # A cold upgrade discovers already-armed NAS jobs without provisioning again.
        provider = ChronosCronScheduler()
        provider._client = nas
        with cron_web._cron_store_scope(homes["work"]):
            provider.reconcile()
        assert fire(existing_job["id"]).status_code == 202
        assert existing_job["id"] not in nas.provisions

        with cron_web._cron_store_scope(homes["work"]) as jobs:
            jobs.remove_job(new_job["id"])
            provider.reconcile()
        before = list(forwarded)
        response = fire(new_job["id"])
        assert response.status_code == 200
        assert response.json()["status"] == "gone"
        assert new_job["id"] not in nas.armed
        assert forwarded == before
        assert {profile for profile, _, _ in forwarded} == {"work"}
        assert all(auth == f"Bearer {issued_token}" for _, _, auth in forwarded)


@pytest.mark.asyncio
async def test_fire_verifies_owner_off_loop_and_rejects_invalid_tokens(fire_profiles, monkeypatch):
    homes, token, create, forwarded = fire_profiles
    job = create()
    provider = ChronosCronScheduler()
    provider._client = type("NasClient", (), {"provision": lambda self, **kwargs: None})()
    with cron_web._cron_store_scope(homes["work"]):
        provider.register_job(job)

    loop_thread = threading.get_ident()
    verification_threads = []
    verified = []
    scans = []
    real_scan = cron_web._find_cron_job_profile

    def verify(**kwargs):
        verification_threads.append(threading.get_ident())
        claims = fire_verify.verify_nas_fire_token(**kwargs)
        if claims is not None:
            verified.append(kwargs["token"])
        return claims

    def scan(job_id):
        assert verified, "an unauthenticated callback must not scan profile job stores"
        scans.append(job_id)
        return real_scan(job_id)

    monkeypatch.setattr(fire_verify, "get_fire_verifier", lambda: verify)
    monkeypatch.setattr(cron_web, "_find_cron_job_profile", scan)
    transport = httpx.ASGITransport(app=web_server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        consumed = []

        async def oversized_stream():
            for chunk in range(128):
                consumed.append(chunk)
                yield b" " * 1024

        response = await client.post("/api/cron/fire", content=oversized_stream(),
                                     headers={"Authorization": "Bearer forged"})
        assert response.request.headers["transfer-encoding"] == "chunked"
        assert response.status_code == 413
        assert 0 < len(consumed) < 128
        assert verification_threads == scans == forwarded == []
        consumed.clear()
        response = await client.post("/api/cron/fire", content=oversized_stream())
        assert response.status_code == 401
        assert consumed == verification_threads == scans == forwarded == []
        for bearer, expected_status in (("forged", 401), (token("default"), 400)):
            response = await client.post("/api/cron/fire", content=b"{",
                                         headers={"Authorization": f"Bearer {bearer}"})
            assert response.status_code == expected_status
        verified.clear()

        for bearer in ("forged", token(aud="agent:other"), token(iss="https://other.example")):
            response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                         headers={"Authorization": f"Bearer {bearer}"})
            assert response.status_code == 401
            assert scans == []
        for job_id in ("../work/config.yaml", "x" * 256, ["work"], {"profile": "work"}):
            response = await client.post("/api/cron/fire", json={"job_id": job_id},
                                         headers={"Authorization": "Bearer forged"})
            assert response.status_code == 401
            assert scans == []
        assert forwarded == []
        # A default-profile token cannot authorize a work-profile job via fallback.
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token('default')}"})
        assert response.status_code == 401
        assert forwarded == []
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 202

        from cron.chronos_fire_profiles import record_cron_fire_profile_hint

        default_job = create("default")
        # A stale hint can recover only through another configured verifier accepting
        # the token, followed by authentication against the actual job owner.
        for stale_profile in ("work", "removed-profile"):
            record_cron_fire_profile_hint(default_job["id"], stale_profile)
            response = await client.post("/api/cron/fire", json={"job_id": default_job["id"]},
                                         headers={"Authorization": f"Bearer {token('default')}"})
            assert response.status_code == 202
        before = list(forwarded)
        (homes["work"] / "config.yaml").unlink()
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 401
        assert forwarded == before
    assert verification_threads and loop_thread not in verification_threads
    assert [(profile, job_id) for profile, job_id, _ in forwarded] == [
        ("work", job["id"]), ("default", default_job["id"]), ("default", default_job["id"]),
    ]
