"""Chronos callbacks keep profile authentication across the scheduler lifecycle."""

import asyncio
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

from cron import chronos_fire_profiles as fire_catalogs
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

    def token(profile="work", signing_key=None, **overrides):
        claims = {"aud": f"agent:{profile}", "iss": f"https://{profile}.example",
                  "exp": int(time.time()) + 600, "purpose": "cron_fire"}
        return jwt.encode({**claims, **overrides}, signing_key or key, algorithm="RS256")

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


def test_profile_fire_auth_survives_arm_reconcile_and_cancel(fire_profiles, monkeypatch):
    homes, token, create, forwarded = fire_profiles
    new_job, existing_job, failed_job = create(), create(), create()
    provider = ChronosCronScheduler()
    issued_token = token()
    build_started, release_build = threading.Event(), threading.Event()
    config_read, release_config = threading.Event(), threading.Event()
    builds = []
    real_load = cron_web._load_cron_config_for_profile

    def load(profile):
        cfg = real_load(profile)
        if profile == "work" and threading.get_ident() in builds and not config_read.is_set():
            config_read.set()
            assert release_config.wait(5), "config read was never released"
        return cfg

    monkeypatch.setattr(cron_web, "_load_cron_config_for_profile", load)
    real_build = fire_catalogs._build_cron_fire_catalog

    def blocked_build(root):
        builds.append(threading.get_ident())
        build_started.set()
        assert release_build.wait(5), "catalog build was never released"
        return real_build(root)

    monkeypatch.setattr(fire_catalogs, "_build_cron_fire_catalog", blocked_build)
    catalog = fire_catalogs.get_cron_fire_catalog()

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
        # An armed job can wake the dashboard before its scheduler reconciles.
        cold_response = fire(existing_job["id"])
        assert cold_response.status_code == 503
        assert int(cold_response.headers["Retry-After"]) > 0
        assert build_started.wait(5)
        try:
            for _ in range(3):
                assert fire(existing_job["id"]).status_code == 503
            assert len(builds) == 1  # concurrent retries coalesce into one discovery
            assert forwarded == []
        finally:
            release_build.set()
        # Change issuer AFTER the builder read work's config. An old selector must
        # never be published with the new fingerprint; retry the inconsistent build.
        assert config_read.wait(5)
        try:
            config_path = homes["work"] / "config.yaml"
            cfg = yaml.safe_load(config_path.read_text())
            cfg["cron"]["chronos"]["portal_url"] = "https://upgraded-work.example"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            issued_token = token(iss="https://upgraded-work.example")
        finally:
            release_config.set()
        assert catalog.ready.wait(5)
        cold_response = fire(existing_job["id"])
        # An older gateway can also arm a new job after the dashboard started.
        legacy_job = create()
        nas.armed[legacy_job["id"]] = legacy_job["next_run_at"]
        legacy_response = fire(legacy_job["id"])
        assert [cold_response.status_code, legacy_response.status_code] == [202, 202]
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
    verification_settings = []
    verified = []
    builds = []
    real_build = fire_catalogs._build_cron_fire_catalog

    def build(root):
        builds.append(threading.get_ident())
        return real_build(root)

    monkeypatch.setattr(fire_catalogs, "_build_cron_fire_catalog", build)
    catalog = fire_catalogs.get_cron_fire_catalog()
    scans = []
    real_scan = cron_web._find_cron_job_profile

    def verify(**kwargs):
        verification_threads.append(threading.get_ident())
        verification_settings.append((kwargs["expected_audience"], kwargs["issuer"], kwargs["jwks_or_key"]))
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
        for bearer in (token(exp=int(time.time()) - 120), token(purpose="agent"), token(exp=None)):
            response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                         headers={"Authorization": f"Bearer {bearer}"})
            assert response.status_code == 401
        assert builds == []  # malformed tokens never discover profiles
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 503
        assert await asyncio.to_thread(catalog.ready.wait, 5)
        assert scans == forwarded == []

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

        default_job = create("default")
        response = await client.post("/api/cron/fire", json={"job_id": default_job["id"]},
                                     headers={"Authorization": f"Bearer {token('default')}"})
        assert response.status_code == 202
        assert len(builds) == 1  # wrong selectors/signatures do not rebuild a fresh catalog

        # Config can be loaded before dotenv, then acquire/rotate its audience without
        # changing YAML. Exercise the same contract for user config and managed overlay.
        from hermes_cli.env_loader import load_hermes_dotenv

        env_forwarded = []
        config_path = homes["work"] / "config.yaml"
        original_config = config_path.read_text()
        managed_home = homes["default"].parent / "managed"
        managed_home.mkdir()
        for source in ("user", "managed"):
            with monkeypatch.context() as env_patch:
                env_patch.setenv("CHRONOS_TEST_AUDIENCE", "")
                env_patch.delenv("CHRONOS_TEST_AUDIENCE")
                if source == "user":
                    env_cfg = yaml.safe_load(original_config)
                    env_cfg["cron"]["chronos"]["expected_audience"] = "${CHRONOS_TEST_AUDIENCE}"
                    config_path.write_text(yaml.safe_dump(env_cfg), encoding="utf-8")
                    changed_path = config_path
                else:
                    changed_path = managed_home / "config.yaml"
                    changed_path.write_text(yaml.safe_dump({"cron": {"chronos": {
                        "expected_audience": "${env:CHRONOS_TEST_AUDIENCE}",
                    }}}), encoding="utf-8")
                    env_patch.setenv("HERMES_MANAGED_DIR", str(managed_home))
                response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                             headers={"Authorization": f"Bearer {token()}"})
                assert response.status_code == 503
                assert await asyncio.to_thread(catalog.ready.wait, 5)
                yaml_stat = changed_path.stat()
                for audience in (f"agent:{source}-dotenv", f"agent:{source}-rotated"):
                    (homes["default"] / ".env").write_text(
                        f"CHRONOS_TEST_AUDIENCE={audience}\n", encoding="utf-8",
                    )
                    load_hermes_dotenv(hermes_home=homes["default"])
                    # Another consumer may already have refreshed the loader's own cache.
                    assert cron_web._load_cron_config_for_profile("work")["cron"]["chronos"]["expected_audience"] == audience
                    assert changed_path.stat().st_mtime_ns == yaml_stat.st_mtime_ns
                    response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                                 headers={"Authorization": f"Bearer {token(aud=audience)}"})
                    assert response.status_code == 503
                    assert await asyncio.to_thread(catalog.ready.wait, 5)
                    response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                                 headers={"Authorization": f"Bearer {token(aud=audience)}"})
                    assert response.status_code == 202
                    env_forwarded.append(("work", job["id"]))
            config_path.write_text(original_config, encoding="utf-8")
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 503
        assert await asyncio.to_thread(catalog.ready.wait, 5)

        # Key rotation takes effect immediately, even while selectors are cached.
        rotated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        config_path = homes["work"] / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        cfg["cron"]["chronos"]["nas_jwks_url"] = rotated_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rotated_token = token(signing_key=rotated_key)
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {rotated_token}"})
        assert response.status_code == 202
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 503  # refresh changed candidates before a final rejection
        assert await asyncio.to_thread(catalog.ready.wait, 5)
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token()}"})
        assert response.status_code == 401

        # New selectors/profile configs are discovered immediately from real file changes,
        # independently of job creation or scheduler warmup.
        cfg["cron"]["chronos"].update(
            portal_url="https://new-work.example", expected_audience="agent:work-updated",
        )
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        sibling = homes["work"].parent / "work-copy"
        sibling.mkdir()
        (sibling / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        new_token = token(signing_key=rotated_key, iss="https://new-work.example", aud="agent:work-updated")
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {new_token}"})
        assert response.status_code == 503
        assert await asyncio.to_thread(catalog.ready.wait, 5)
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {new_token}"})
        assert response.status_code == 202
        verification_settings.clear()
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {token(iss='https://new-work.example', aud='agent:work-updated')}"})
        assert response.status_code == 401
        assert len(verification_settings) == len(set(verification_settings))
        # A new profile can reuse an existing selector with a DIFFERENT key. Failed
        # old candidates must check freshness just like an unknown selector does.
        homes["after-start"] = homes["work"].parent / "after-start"
        homes["after-start"].mkdir()
        added_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cfg["cron"]["chronos"]["nas_jwks_url"] = added_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        (homes["after-start"] / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        added_job = create("after-start")
        added_token = token("after-start", signing_key=added_key,
                            iss="https://new-work.example", aud="agent:work-updated")
        response = await client.post("/api/cron/fire", json={"job_id": added_job["id"]},
                                     headers={"Authorization": f"Bearer {added_token}"})
        assert response.status_code == 503
        assert await asyncio.to_thread(catalog.ready.wait, 5)
        response = await client.post("/api/cron/fire", json={"job_id": added_job["id"]},
                                     headers={"Authorization": f"Bearer {added_token}"})
        assert response.status_code == 202
        before = list(forwarded)
        (homes["work"] / "config.yaml").unlink()
        response = await client.post("/api/cron/fire", json={"job_id": job["id"]},
                                     headers={"Authorization": f"Bearer {new_token}"})
        assert response.status_code == 401
        assert forwarded == before
        from hermes_constants import mark_named_profile_deleted

        mark_named_profile_deleted(homes["after-start"])
        response = await client.post("/api/cron/fire", json={"job_id": added_job["id"]},
                                     headers={"Authorization": f"Bearer {added_token}"})
        assert response.status_code == 503
        assert await asyncio.to_thread(catalog.ready.wait, 5)
        response = await client.post("/api/cron/fire", json={"job_id": added_job["id"]},
                                     headers={"Authorization": f"Bearer {added_token}"})
        assert response.status_code == 401
        assert forwarded == before
    assert builds and loop_thread not in builds
    assert verification_threads and loop_thread not in verification_threads
    assert [(profile, job_id) for profile, job_id, _ in forwarded] == [
        ("work", job["id"]), ("default", default_job["id"]), *env_forwarded,
        ("work", job["id"]), ("work", job["id"]), ("after-start", added_job["id"]),
    ]
