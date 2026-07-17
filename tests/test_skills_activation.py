"""
Tests for the Layer-2 skills activation API (spec 309, plan 02-01).

Covers POST /v1/skills/activations, DELETE /v1/skills/activations/{skillId},
and GET /v1/skills?scope=managed against the NET-NEW ``SkillsActivationService``
(``tools/skills_activation.py``) plus the ``api_server.activation.patch``
extensions to the existing aiohttp API server.

RED mechanics: ``tools.skills_activation`` does not exist until Task 3's
patch/module land. Every test imports it LAZILY via ``_build_service()``
(never at module scope), so all named tests below REGISTER at collection
time and each FAILS independently (via ``ModuleNotFoundError`` or an
``AttributeError``/assertion against the not-yet-patched server) rather
than aborting collection entirely.

House conventions followed (see tests/gateway/test_api_server.py:393-431,
681-731 and tests/conftest.py:327-398): ``@pytest.mark.asyncio``,
``TestClient(TestServer(app))``, a real ``APIServerAdapter`` built via a
synthetic API key, and the autouse ``_hermetic_environment`` fixture that
isolates ``HERMES_HOME`` to a fresh per-test tmp dir.
"""

import asyncio
import hashlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _IdempotencyCache,
    cors_middleware,
    security_headers_middleware,
)
from hermes_constants import get_hermes_home

DEFAULT_SCOPE = {
    "tenantId": "deployment-local",
    "companyId": "deployment-local",
    "agentId": "deployment-local",
}


# ---------------------------------------------------------------------------
# Shared helpers — no ledger/validation/handler logic is defined here; every
# helper only constructs fixtures, requests, or reads real files/DB rows.
# ---------------------------------------------------------------------------


def _hash_of(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_adapter(api_key: str = "sk-secret") -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    return APIServerAdapter(config)


def _build_service(home, check_auth, *, scope=None, idem_cache=None):
    """Lazy import — tools.skills_activation does not exist at RED time."""
    from tools.skills_activation import SkillsActivationService

    return SkillsActivationService(
        home=Path(home),
        check_auth=check_auth,
        scope=dict(scope) if scope is not None else dict(DEFAULT_SCOPE),
        idem_cache=idem_cache if idem_cache is not None else _IdempotencyCache(),
    )


def _build_app(adapter: APIServerAdapter, service) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    adapter._skills_activation_service = service
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_post("/v1/skills/activations", service.handle_activate)
    app.router.add_delete("/v1/skills/activations/{skill_id}", service.handle_deactivate)
    return app


def _hermetic_home() -> Path:
    return get_hermes_home()


def _auth_headers(key: str = "sk-secret") -> dict:
    return {"Authorization": f"Bearer {key}"}


def _activation_body(skill_id: str, content: str, *, idempotency_key: str, version: str = None) -> dict:
    skill = {"skillId": skill_id, "contentHash": _hash_of(content), "content": content}
    if version is not None:
        skill["version"] = version
    return {"contractVersion": "1", "idempotencyKey": idempotency_key, "skill": skill}


def _seed_native(skills_dir: Path, name: str) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\nname: {name}\ndescription: native skill {name}.\n---\n\n# {name}\n\nBody.\n")
    return skill_md


def _managed_row_count(home: Path) -> int:
    db_path = home / "skills_activation.db"
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM managed_skills").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auth + scope (REQ-04, REQ-09)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthenticated_post_401_no_state_change():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Brand voice.\n\nBody."
    body = _activation_body("brand-voice", content, idempotency_key="key-unauth")

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body)
        assert resp.status == 401

        resp2 = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers("wrong-key"))
        assert resp2.status == 401

        resp3 = await cli.delete("/v1/skills/activations/brand-voice")
        assert resp3.status == 401

    assert _managed_row_count(home) == 0
    assert not (home / "skills" / "brand-voice").exists()


@pytest.mark.asyncio
async def test_unauthenticated_managed_inventory_401():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/skills", params={"scope": "managed"})
        assert resp.status == 401


@pytest.mark.asyncio
async def test_wrong_agent_assert_403_zero_state():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Wrong scope test.\n\nBody."

    async with TestClient(TestServer(app)) as cli:
        for field in ("agentId", "tenantId", "companyId"):
            body = _activation_body(f"scope-test-{field.lower()}", content, idempotency_key=f"key-{field}")
            body[field] = "not-the-configured-value"
            resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
            assert resp.status == 403, field
            data = await resp.json()
            assert data["error"]["code"] == "wrong_scope"

    assert _managed_row_count(home) == 0
    assert not any((home / "skills").glob("scope-test-*"))


@pytest.mark.asyncio
async def test_body_scope_never_derives_ledger_scope():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Scope derivation test.\n\nBody."

    async with TestClient(TestServer(app)) as cli:
        body1 = _activation_body("scope-derive-skill", content, idempotency_key="key-1")
        resp1 = await cli.post("/v1/skills/activations", json=body1, headers=_auth_headers())
        assert resp1.status == 201
        data1 = await resp1.json()
        assert data1["ledgerEntry"]["scope"] == DEFAULT_SCOPE

        body2 = _activation_body("scope-derive-skill", content, idempotency_key="key-2")
        body2["agentId"] = DEFAULT_SCOPE["agentId"]  # matching assert — accepted, still deployment scope
        resp2 = await cli.post("/v1/skills/activations", json=body2, headers=_auth_headers())
        assert resp2.status == 201
        data2 = await resp2.json()
        assert data2["ledgerEntry"]["scope"] == DEFAULT_SCOPE


# ---------------------------------------------------------------------------
# Filesystem-path rejection (REQ-04)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filesystem_path_skillid_422():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Path skillId test.\n\nBody."

    async with TestClient(TestServer(app)) as cli:
        for bad_id in ("../etc/passwd", "a/b", "a\\b", "a..b"):
            body = {
                "contractVersion": "1",
                "idempotencyKey": "key-path",
                "skill": {"skillId": bad_id, "contentHash": _hash_of(content), "content": content},
            }
            resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
            assert resp.status == 422, bad_id

    assert _managed_row_count(home) == 0


@pytest.mark.asyncio
async def test_path_field_in_body_422():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Path field test.\n\nBody."
    body = _activation_body("path-field-skill", content, idempotency_key="key-pf")
    body["skill"]["filePath"] = "/etc/passwd"

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp.status == 422

    assert _managed_row_count(home) == 0
    assert not (home / "skills" / "path-field-skill").exists()


# ---------------------------------------------------------------------------
# Integrity + versioning (REQ-05 server clauses)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integrity_mismatch_409_no_partial_state():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Integrity mismatch test.\n\nReal body."
    wrong_hash = _hash_of("this is different content entirely")
    body = {
        "contractVersion": "1",
        "idempotencyKey": "key-integrity",
        "skill": {"skillId": "integrity-skill", "contentHash": wrong_hash, "content": content},
    }

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp.status == 409
        data = await resp.json()
        assert data["error"]["code"] == "integrity_mismatch"

        resp2 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
        data2 = await resp2.json()
        assert data2["data"] == []

    assert _managed_row_count(home) == 0
    assert not (home / "skills" / "integrity-skill").exists()


@pytest.mark.asyncio
async def test_unknown_contract_version_refused_422():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Contract version test.\n\nBody."
    body = _activation_body("version-skill", content, idempotency_key="key-v")
    body["contractVersion"] = "2"

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp.status == 422
        data = await resp.json()
        assert data["error"]["code"] == "unsupported_contract_version"

    assert _managed_row_count(home) == 0


# ---------------------------------------------------------------------------
# Activation ack + observed inventory (REQ-05, REQ-06)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activation_201_ack_ledger_and_observed():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Brand voice.\n\nBody text for the activation ack test."
    content_hash = _hash_of(content)
    body = _activation_body("brand-voice", content, idempotency_key="key-activate")

    with patch("tools.skills_tool.SKILLS_DIR", home / "skills"):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
            assert resp.status == 201
            data = await resp.json()
            assert data["ack"]["skillId"] == "brand-voice"
            assert data["ack"]["contentHash"] == content_hash
            assert data["ack"]["reloaded"] is True
            assert "activatedAt" in data["ack"]
            assert data["ledgerEntry"]["managedBy"] == "paperclip-skill-contract/v1"
            assert data["ledgerEntry"]["scope"] == DEFAULT_SCOPE

            resp2 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
            data2 = await resp2.json()
            managed = {e["name"]: e for e in data2["data"]}
            assert managed["brand-voice"]["contentHash"] == content_hash

            resp3 = await cli.get("/v1/skills", headers=_auth_headers())
            data3 = await resp3.json()
            base_names = {e["name"] for e in data3["data"]}
            assert "brand-voice" in base_names


@pytest.mark.asyncio
async def test_managed_inventory_excludes_remote_native():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    _seed_native(home / "skills", "native-tool")

    with patch("tools.skills_tool.SKILLS_DIR", home / "skills"):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/skills", headers=_auth_headers())
            data = await resp.json()
            assert "native-tool" in {e["name"] for e in data["data"]}

            resp2 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
            data2 = await resp2.json()
            assert "native-tool" not in {e["name"] for e in data2["data"]}


# ---------------------------------------------------------------------------
# Cleanup / rollback / native-ownership (REQ-06)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_native_403_not_managed():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    skill_md = _seed_native(home / "skills", "native-del")
    before = skill_md.read_text()

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete("/v1/skills/activations/native-del", headers=_auth_headers())
        assert resp.status == 403
        data = await resp.json()
        assert data["error"]["code"] == "not_managed"

    assert skill_md.read_text() == before


@pytest.mark.asyncio
async def test_delete_missing_404():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete("/v1/skills/activations/nonexistent-skill", headers=_auth_headers())
        assert resp.status == 404


@pytest.mark.asyncio
async def test_clean_removes_managed_only():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    native_md = _seed_native(home / "skills", "native-keep")
    native_before = native_md.read_text()
    content = "# Managed clean test.\n\nBody."
    body = _activation_body("managed-clean", content, idempotency_key="key-clean")

    with patch("tools.skills_tool.SKILLS_DIR", home / "skills"):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
            assert resp.status == 201

            resp2 = await cli.delete(
                "/v1/skills/activations/managed-clean", params={"mode": "clean"}, headers=_auth_headers()
            )
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["ack"]["removed"] is True

            resp3 = await cli.get("/v1/skills", headers=_auth_headers())
            data3 = await resp3.json()
            base_names = {e["name"] for e in data3["data"]}
            assert "native-keep" in base_names
            assert "managed-clean" not in base_names

    assert native_md.read_text() == native_before
    assert not (home / "skills" / "managed-clean").exists()
    assert _managed_row_count(home) == 0


# ---------------------------------------------------------------------------
# Idempotency + concurrency (REQ-07 server primitives, INT-004)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_replay_returns_original_ack_200():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Replay test.\n\nBody."
    body = _activation_body("replay-skill", content, idempotency_key="replay-key")

    async with TestClient(TestServer(app)) as cli:
        resp1 = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp1.status == 201
        data1 = await resp1.json()

        resp2 = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp2.status == 200
        data2 = await resp2.json()
        assert data2["ack"] == data1["ack"]

    assert _managed_row_count(home) == 1


@pytest.mark.asyncio
async def test_idempotency_key_conflict_different_body_409():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content1 = "# Conflict v1.\n\nBody one."
    content2 = "# Conflict v2, genuinely different.\n\nBody two."
    body1 = _activation_body("conflict-skill", content1, idempotency_key="conflict-key")
    body2 = _activation_body("conflict-skill", content2, idempotency_key="conflict-key")

    async with TestClient(TestServer(app)) as cli:
        resp1 = await cli.post("/v1/skills/activations", json=body1, headers=_auth_headers())
        assert resp1.status == 201

        resp2 = await cli.post("/v1/skills/activations", json=body2, headers=_auth_headers())
        assert resp2.status == 409
        data2 = await resp2.json()
        assert data2["error"]["code"] == "idempotency_conflict"

        resp3 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
        data3 = await resp3.json()
        row = next(e for e in data3["data"] if e["name"] == "conflict-skill")
        assert row["contentHash"] == _hash_of(content1)


@pytest.mark.asyncio
async def test_concurrent_same_key_single_execution():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Concurrent same key.\n\nBody."
    body = _activation_body("concurrent-skill", content, idempotency_key="concurrent-key")

    call_count = {"n": 0}
    original_write = service._write_skill_file_atomic

    def counting_write(skill_id, content_):
        call_count["n"] += 1
        return original_write(skill_id, content_)

    service._write_skill_file_atomic = counting_write

    async with TestClient(TestServer(app)) as cli:
        responses = await asyncio.gather(
            *[cli.post("/v1/skills/activations", json=body, headers=_auth_headers()) for _ in range(5)]
        )
        payloads = [await r.json() for r in responses]

    acks = [p["ack"] for p in payloads]
    assert all(a == acks[0] for a in acks)
    assert call_count["n"] == 1
    assert _managed_row_count(home) == 1


@pytest.mark.asyncio
async def test_concurrent_distinct_keys_converge_single_row():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Distinct keys, same content.\n\nBody."
    body_a = _activation_body("distinct-skill", content, idempotency_key="key-a")
    body_b = _activation_body("distinct-skill", content, idempotency_key="key-b")

    async with TestClient(TestServer(app)) as cli:
        resp_a, resp_b = await asyncio.gather(
            cli.post("/v1/skills/activations", json=body_a, headers=_auth_headers()),
            cli.post("/v1/skills/activations", json=body_b, headers=_auth_headers()),
        )
        assert resp_a.status in (200, 201)
        assert resp_b.status in (200, 201)

    assert _managed_row_count(home) == 1
    conn = sqlite3.connect(str(home / "skills_activation.db"))
    row = conn.execute(
        "SELECT content_hash, previous_state_json FROM managed_skills WHERE skill_id = ?",
        ("distinct-skill",),
    ).fetchone()
    conn.close()
    assert row[0] == _hash_of(content)
    assert row[1] is None


# ---------------------------------------------------------------------------
# Rollback semantics (REQ-07 server primitives, spec AC-007)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_restores_prior_managed_set():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content_v1 = "# Rollback v1.\n\nBody v1."
    content_v2 = "# Rollback v2, genuinely different.\n\nBody v2."
    body_v1 = _activation_body("rollback-skill", content_v1, idempotency_key="key-v1")
    body_v2 = _activation_body("rollback-skill", content_v2, idempotency_key="key-v2")

    async with TestClient(TestServer(app)) as cli:
        resp1 = await cli.post("/v1/skills/activations", json=body_v1, headers=_auth_headers())
        assert resp1.status == 201

        resp2 = await cli.post("/v1/skills/activations", json=body_v2, headers=_auth_headers())
        assert resp2.status == 201

        resp3 = await cli.delete(
            "/v1/skills/activations/rollback-skill", params={"mode": "rollback"}, headers=_auth_headers()
        )
        assert resp3.status == 200

        resp4 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
        data4 = await resp4.json()
        row = next(e for e in data4["data"] if e["name"] == "rollback-skill")
        assert row["contentHash"] == _hash_of(content_v1)

    assert (home / "skills" / "rollback-skill" / "SKILL.md").read_text() == content_v1


@pytest.mark.asyncio
async def test_rollback_initial_activation_removes_entry():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Initial activation only.\n\nBody."
    body1 = _activation_body("initial-skill", content, idempotency_key="key-first")
    body2 = _activation_body("initial-skill", content, idempotency_key="key-second")

    with patch("tools.skills_tool.SKILLS_DIR", home / "skills"):
        async with TestClient(TestServer(app)) as cli:
            resp1 = await cli.post("/v1/skills/activations", json=body1, headers=_auth_headers())
            assert resp1.status == 201
            data1 = await resp1.json()
            assert data1["ledgerEntry"]["previousState"] is None

            # Same content, new idempotencyKey — must be a no-op refresh, never
            # a fabricated previousState.
            resp2 = await cli.post("/v1/skills/activations", json=body2, headers=_auth_headers())
            assert resp2.status == 201
            data2 = await resp2.json()
            assert data2["ledgerEntry"]["previousState"] is None

            resp3 = await cli.delete(
                "/v1/skills/activations/initial-skill", params={"mode": "rollback"}, headers=_auth_headers()
            )
            assert resp3.status == 200

            resp4 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
            data4 = await resp4.json()
            assert all(e["name"] != "initial-skill" for e in data4["data"])

    assert not (home / "skills" / "initial-skill").exists()
    assert _managed_row_count(home) == 0


# ---------------------------------------------------------------------------
# Cross-tenant isolation (REQ-09 two-server proof)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_deployments_zero_cross_leakage(tmp_path):
    home_a = tmp_path / "deployment-a"
    home_b = tmp_path / "deployment-b"
    (home_a / "skills").mkdir(parents=True)
    (home_b / "skills").mkdir(parents=True)

    adapter_a = _make_adapter(api_key="key-a")
    adapter_b = _make_adapter(api_key="key-b")
    service_a = _build_service(
        home_a, adapter_a._check_auth,
        scope={"tenantId": "tenant-a", "companyId": "tenant-a", "agentId": "tenant-a"},
    )
    service_b = _build_service(
        home_b, adapter_b._check_auth,
        scope={"tenantId": "tenant-b", "companyId": "tenant-b", "agentId": "tenant-b"},
    )
    app_a = _build_app(adapter_a, service_a)
    app_b = _build_app(adapter_b, service_b)

    content = "# Cross-tenant test.\n\nBody."
    body = _activation_body("cross-skill", content, idempotency_key="cross-key")

    async with TestClient(TestServer(app_b)) as cli_b:
        resp_post = await cli_b.post("/v1/skills/activations", json=body, headers=_auth_headers("key-a"))
        assert resp_post.status == 401
        resp_del = await cli_b.delete("/v1/skills/activations/cross-skill", headers=_auth_headers("key-a"))
        assert resp_del.status == 401
        resp_get = await cli_b.get(
            "/v1/skills", params={"scope": "managed"}, headers=_auth_headers("key-a")
        )
        assert resp_get.status == 401

    async with TestClient(TestServer(app_a)) as cli_a:
        resp_ok = await cli_a.post("/v1/skills/activations", json=body, headers=_auth_headers("key-a"))
        assert resp_ok.status == 201

    async with TestClient(TestServer(app_b)) as cli_b2:
        resp_b_managed = await cli_b2.get(
            "/v1/skills", params={"scope": "managed"}, headers=_auth_headers("key-b")
        )
        data_b = await resp_b_managed.json()
        assert data_b["data"] == []

    assert not (home_b / "skills" / "cross-skill").exists()
    assert _managed_row_count(home_a) == 1
    assert _managed_row_count(home_b) == 0


@pytest.mark.asyncio
async def test_native_skill_activation_409_conflict():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    native_md = _seed_native(home / "skills", "native-conflict")
    before = native_md.read_text()
    content = "# Hijack attempt.\n\nBody."
    body = _activation_body("native-conflict", content, idempotency_key="key-hijack")

    with patch("tools.skills_tool.SKILLS_DIR", home / "skills"):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
            assert resp.status == 409
            data = await resp.json()
            assert data["error"]["code"] == "native_conflict"

            resp2 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
            data2 = await resp2.json()
            assert data2["data"] == []

    assert native_md.read_text() == before
    assert _managed_row_count(home) == 0


# ---------------------------------------------------------------------------
# Hostile-input DELETE handling (adjudication round 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_delete_mode_422():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Mode validation test.\n\nBody."
    body = _activation_body("mode-skill", content, idempotency_key="key-mode")

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp.status == 201

        resp_bad_mode = await cli.delete(
            "/v1/skills/activations/mode-skill", params={"mode": "purge"}, headers=_auth_headers()
        )
        assert resp_bad_mode.status == 422

        resp_dup_mode = await cli.request(
            "DELETE",
            "/v1/skills/activations/mode-skill?mode=clean&mode=rollback",
            headers=_auth_headers(),
        )
        assert resp_dup_mode.status == 422

        resp_traversal = await cli.delete("/v1/skills/activations/a..b", headers=_auth_headers())
        assert resp_traversal.status == 422

    assert (home / "skills" / "mode-skill" / "SKILL.md").exists()
    assert _managed_row_count(home) == 1


@pytest.mark.asyncio
async def test_replay_after_clean_reexecutes():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content = "# Replay after clean.\n\nBody."
    body = _activation_body("replay-clean-skill", content, idempotency_key="replay-clean-key")

    async with TestClient(TestServer(app)) as cli:
        resp1 = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp1.status == 201

        resp_clean = await cli.delete(
            "/v1/skills/activations/replay-clean-skill", params={"mode": "clean"}, headers=_auth_headers()
        )
        assert resp_clean.status == 200

        resp2 = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp2.status == 201  # fresh execution, NOT a stale 200 replay

    assert (home / "skills" / "replay-clean-skill" / "SKILL.md").exists()
    assert _managed_row_count(home) == 1


@pytest.mark.asyncio
async def test_refresh_ledger_failure_restores_prior_file():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)
    content_v1 = "# Ledger-failure v1.\n\nBody v1."
    content_v2 = "# Ledger-failure v2, genuinely different.\n\nBody v2."
    body_v1 = _activation_body("ledger-fail-skill", content_v1, idempotency_key="key-lf1")
    body_v2 = _activation_body("ledger-fail-skill", content_v2, idempotency_key="key-lf2")

    async with TestClient(TestServer(app)) as cli:
        resp1 = await cli.post("/v1/skills/activations", json=body_v1, headers=_auth_headers())
        assert resp1.status == 201

        original_upsert = service._upsert_managed_row
        call_state = {"raised": False}

        def failing_upsert(*args, **kwargs):
            if not call_state["raised"]:
                call_state["raised"] = True
                raise RuntimeError("simulated ledger failure")
            return original_upsert(*args, **kwargs)

        service._upsert_managed_row = failing_upsert

        resp2 = await cli.post("/v1/skills/activations", json=body_v2, headers=_auth_headers())
        assert resp2.status >= 400

    assert (home / "skills" / "ledger-fail-skill" / "SKILL.md").read_text() == content_v1
    conn = sqlite3.connect(str(home / "skills_activation.db"))
    row = conn.execute(
        "SELECT content_hash, previous_state_json FROM managed_skills WHERE skill_id = ?",
        ("ledger-fail-skill",),
    ).fetchone()
    conn.close()
    assert row[0] == _hash_of(content_v1)
    assert row[1] is None


# ---------------------------------------------------------------------------
# Capability advert + real patched wiring (contract §Versioning)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_advertises_activation():
    home = _hermetic_home()
    adapter = _make_adapter()
    service = _build_service(home, adapter._check_auth)
    app = _build_app(adapter, service)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities", headers=_auth_headers())
        assert resp.status == 200
        data = await resp.json()
        features = data["features"]
        assert isinstance(features, dict)
        assert features.get("skills_api") is True
        assert features.get("skills_activation/v1") is True


@pytest.mark.asyncio
async def test_patched_server_wiring_real_adapter():
    home = _hermetic_home()
    adapter = _make_adapter()
    app = web.Application(
        middlewares=[mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    )
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/skills", adapter._handle_skills)

    # Real construction path — service built by the adapter itself from
    # get_hermes_home()/env scope + its own _check_auth + a fresh
    # _IdempotencyCache (patch-added method; AttributeError pre-patch).
    adapter._register_skills_activation(app.router)

    content = "# Wired via the real patch.\n\nBody."
    body = _activation_body("wired-skill", content, idempotency_key="key-wired")

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/skills/activations", json=body, headers=_auth_headers())
        assert resp.status == 201

        resp2 = await cli.get("/v1/skills", params={"scope": "managed"}, headers=_auth_headers())
        assert resp2.status == 200
        data2 = await resp2.json()
        assert "wired-skill" in {e["name"] for e in data2["data"]}

    # Source-level smoke (flagged for Phase-4 live re-verification): connect()
    # must actually call the patch-added registration method — not just make
    # it callable in isolation.
    import inspect
    from gateway.platforms import api_server as api_server_module

    source = inspect.getsource(api_server_module.APIServerAdapter.connect)
    assert "_register_skills_activation" in source
