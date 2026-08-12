"""Tests for the safe administrative control-plane API (/v1/admin/*)."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def temp_hermes_home(monkeypatch, tmp_path):
    """Fixture providing a real isolated temp HERMES_HOME filesystem."""
    home_dir = tmp_path / ".hermes"
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / "profiles").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: home_dir)
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home_dir)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: home_dir / "profiles")
    return home_dir


@pytest_asyncio.fixture
async def create_api_client(temp_hermes_home):
    """Helper to spin up an APIServerAdapter TestClient with given config."""
    clients = []

    async def _make_client(admin_config_rw=True, key="test-secret-key"):
        extra = {
            "key": key,
            "admin_config_rw": admin_config_rw,
        }
        config = PlatformConfig(enabled=True, extra=extra)
        adapter = APIServerAdapter(config)

        # Setup route table & app
        routes = adapter._http_route_table()
        app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
        app["api_server_adapter"] = adapter

        for method, path, handler in routes:
            if method == "GET":
                app.router.add_get(path, handler)
            elif method == "POST":
                app.router.add_post(path, handler)
            elif method == "PUT":
                app.router.add_put(path, handler)
            elif method == "PATCH":
                app.router.add_patch(path, handler)
            elif method == "DELETE":
                app.router.add_delete(path, handler)

        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        clients.append(client)
        return client, key

    yield _make_client

    for client in clients:
        await client.close()


OWNER_HEADERS = {
    "X-Hermes-Owner-Managed-By": "control_plane",
    "X-Hermes-Owner-Tenant-Id": "tenant-a",
    "X-Hermes-Owner-Resource-Id": "res-1",
}


@pytest.mark.asyncio
async def test_admin_config_rw_off_by_default_and_capabilities(create_api_client):
    """Verify capability is false by default and capabilities advertises admin endpoints only when enabled."""
    client_off, key = await create_api_client(admin_config_rw=False)
    headers = {"Authorization": f"Bearer {key}"}

    # Capabilities when feature is OFF
    resp = await client_off.get("/v1/capabilities", headers=headers)
    assert resp.status == 200
    caps = await resp.json()
    assert caps["features"]["admin_config_rw"] is False
    assert "admin_profiles" not in caps["endpoints"]

    # Admin route fails closed (403) when feature is OFF
    resp = await client_off.get("/v1/admin/profiles", headers={**headers, **OWNER_HEADERS})
    assert resp.status == 403
    err = await resp.json()
    assert "disabled" in json.dumps(err).lower()

    # Capabilities when feature is ON
    client_on, key_on = await create_api_client(admin_config_rw=True)
    headers_on = {"Authorization": f"Bearer {key_on}"}
    resp_on = await client_on.get("/v1/capabilities", headers=headers_on)
    assert resp_on.status == 200
    caps_on = await resp_on.json()
    assert caps_on["features"]["admin_config_rw"] is True
    assert "admin_profiles" in caps_on["endpoints"]


@pytest.mark.asyncio
async def test_bearer_auth_on_every_admin_route(create_api_client):
    """Verify bearer auth is enforced on all admin routes."""
    client, key = await create_api_client(admin_config_rw=True)

    routes = [
        ("GET", "/v1/admin/profiles"),
        ("PUT", "/v1/admin/profiles/test-profile"),
        ("GET", "/v1/admin/profiles/test-profile"),
        ("DELETE", "/v1/admin/profiles/test-profile"),
        ("GET", "/v1/admin/profiles/test-profile/skills"),
        ("PUT", "/v1/admin/profiles/test-profile/skills/test-skill"),
        ("GET", "/v1/admin/profiles/test-profile/skills/test-skill"),
        ("DELETE", "/v1/admin/profiles/test-profile/skills/test-skill"),
        ("GET", "/v1/admin/profiles/test-profile/files"),
        ("PUT", "/v1/admin/profiles/test-profile/files/SOUL.md"),
        ("GET", "/v1/admin/profiles/test-profile/files/SOUL.md"),
        ("DELETE", "/v1/admin/profiles/test-profile/files/SOUL.md"),
    ]

    for method, path in routes:
        # No header
        resp = await client.request(method, path)
        assert resp.status == 401, f"Expected 401 for unauthenticated {method} {path}"

        # Bad key
        resp_bad = await client.request(method, path, headers={"Authorization": "Bearer wrong-key"})
        assert resp_bad.status == 401, f"Expected 401 for bad key {method} {path}"


@pytest.mark.asyncio
async def test_ownership_tuple_required_for_all_endpoints(create_api_client):
    """Verify missing ownership tuple returns 400 missing_ownership on all profile and child endpoints."""
    client, key = await create_api_client(admin_config_rw=True)
    bearer_only = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    routes = [
        ("GET", "/v1/admin/profiles"),
        ("PUT", "/v1/admin/profiles/noowner"),
        ("GET", "/v1/admin/profiles/noowner"),
        ("DELETE", "/v1/admin/profiles/noowner"),
        ("GET", "/v1/admin/profiles/noowner/skills"),
        ("PUT", "/v1/admin/profiles/noowner/skills/skill1"),
        ("GET", "/v1/admin/profiles/noowner/skills/skill1"),
        ("DELETE", "/v1/admin/profiles/noowner/skills/skill1"),
        ("GET", "/v1/admin/profiles/noowner/files"),
        ("PUT", "/v1/admin/profiles/noowner/files/SOUL.md"),
        ("GET", "/v1/admin/profiles/noowner/files/SOUL.md"),
        ("DELETE", "/v1/admin/profiles/noowner/files/SOUL.md"),
    ]

    for method, path in routes:
        payload = {"content": "data"} if method in ("PUT", "POST") else None
        resp = await client.request(method, path, headers=bearer_only, json=payload)
        assert resp.status == 400, f"Expected 400 for missing ownership on {method} {path}"
        data = await resp.json()
        assert data["error"]["code"] == "missing_ownership"


@pytest.mark.asyncio
async def test_profile_desired_state_display_name_soul_user_context(create_api_client, temp_hermes_home):
    """Verify profile PUT/GET handles display_name, soul, user_context, excludes path, and updates digest."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    create_payload = {
        "managed_by": "control_plane",
        "tenant_id": "tenant-a",
        "resource_id": "res-1",
        "display_name": "Alpha Display",
        "description": "Primary test profile",
        "soul": "You are Alpha.",
        "user_context": "User likes Python.",
    }

    # 1. Create profile
    resp = await client.put("/v1/admin/profiles/alpha", headers=headers, json=create_payload)
    assert resp.status == 201
    data = await resp.json()
    assert data["name"] == "alpha"
    assert data["display_name"] == "Alpha Display"
    assert data["description"] == "Primary test profile"
    assert data["soul"] == "You are Alpha."
    assert data["user_context"] == "User likes Python."
    assert "path" not in data  # Absolute path must be removed
    assert data["digest"].startswith("sha256:")

    # Verify disk files
    prof_dir = temp_hermes_home / "profiles" / "alpha"
    assert (prof_dir / "SOUL.md").read_text(encoding="utf-8") == "You are Alpha."
    assert (prof_dir / "memories" / "USER.md").read_text(encoding="utf-8") == "User likes Python."

    # 2. GET profile reads back exact values and matching digest
    get_resp = await client.get("/v1/admin/profiles/alpha", headers=headers)
    assert get_resp.status == 200
    get_data = await get_resp.json()
    assert get_data["display_name"] == "Alpha Display"
    assert get_data["soul"] == "You are Alpha."
    assert get_data["user_context"] == "User likes Python."
    assert get_data["digest"] == data["digest"]
    assert "path" not in get_data


@pytest.mark.asyncio
async def test_profile_collection_lists_tenant_resources_with_optional_resource_filter(create_api_client):
    """Collection reconciliation spans a tenant; resource_id is an optional exact filter."""
    client, key = await create_api_client(admin_config_rw=True)
    base_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Hermes-Owner-Managed-By": "control_plane",
        "X-Hermes-Owner-Tenant-Id": "tenant-a",
    }

    for name, resource_id in (("tenant-prof-a", "res-a"), ("tenant-prof-b", "res-b")):
        response = await client.put(
            f"/v1/admin/profiles/{name}",
            headers={**base_headers, "X-Hermes-Owner-Resource-Id": resource_id},
            json={},
        )
        assert response.status == 201

    tenant_list = await client.get("/v1/admin/profiles", headers=base_headers)
    assert tenant_list.status == 200
    assert [item["name"] for item in (await tenant_list.json())["data"]] == [
        "tenant-prof-a",
        "tenant-prof-b",
    ]

    filtered = await client.get(
        "/v1/admin/profiles",
        headers={**base_headers, "X-Hermes-Owner-Resource-Id": "res-b"},
    )
    assert filtered.status == 200
    assert [item["name"] for item in (await filtered.json())["data"]] == ["tenant-prof-b"]


@pytest.mark.asyncio
async def test_cloning_forbidden(create_api_client, temp_hermes_home):
    """Verify clone_from, clone_config, and clone_all are rejected with 400."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    for clone_key in ("clone_from", "clone_config", "clone_all"):
        payload = {
            "managed_by": "control_plane",
            "tenant_id": "tenant-a",
            "resource_id": "res-1",
            clone_key: "default" if clone_key == "clone_from" else True,
        }
        resp = await client.put("/v1/admin/profiles/clonetest", headers=headers, json=payload)
        assert resp.status == 400, f"Expected 400 when passing {clone_key}"
        data = await resp.json()
        assert data["error"]["code"] in ("cloning_forbidden", "invalid_request")


@pytest.mark.asyncio
async def test_manifest_digest_persistence_and_get_readonly(create_api_client, temp_hermes_home):
    """Verify spec_digest is persisted in manifest on PUT and GET does not write to disk."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    payload = {
        "managed_by": "control_plane",
        "tenant_id": "tenant-a",
        "resource_id": "res-1",
        "description": "Persistence check",
    }

    put_resp = await client.put("/v1/admin/profiles/persistprof", headers=headers, json=payload)
    assert put_resp.status == 201
    put_data = await put_resp.json()
    digest = put_data["digest"]

    manifest_file = temp_hermes_home / "profiles" / "persistprof" / ".control_plane_manifest.json"
    manifest_disk = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_disk["spec_digest"] == digest

    mtime_before = manifest_file.stat().st_mtime_ns
    get_resp = await client.get("/v1/admin/profiles/persistprof", headers=headers)
    assert get_resp.status == 200
    get_data = await get_resp.json()
    assert get_data["digest"] == digest
    mtime_after = manifest_file.stat().st_mtime_ns

    # GET must be strictly read-only
    assert mtime_before == mtime_after


@pytest.mark.asyncio
async def test_etag_concurrency_and_revision_idempotency(create_api_client):
    """Verify ETag header, If-Match enforcement, and idempotent repeat PUT."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    payload = {
        "managed_by": "control_plane",
        "tenant_id": "tenant-a",
        "resource_id": "res-1",
        "description": "ETag test",
    }

    # 1. PUT returns ETag header
    resp = await client.put("/v1/admin/profiles/etagprof", headers=headers, json=payload)
    assert resp.status == 201
    etag = resp.headers.get("ETag")
    assert etag is not None
    data = await resp.json()
    raw_digest = data["digest"]
    assert etag in (raw_digest, f'"{raw_digest}"')

    # 2. GET returns matching ETag header
    get_resp = await client.get("/v1/admin/profiles/etagprof", headers=headers)
    assert get_resp.status == 200
    assert get_resp.headers.get("ETag") == etag

    # 3. If-Match with valid ETag succeeds
    if_match_headers = {**headers, "If-Match": etag}
    update_resp = await client.put(
        "/v1/admin/profiles/etagprof",
        headers=if_match_headers,
        json={**payload, "description": "Updated ETag test"},
    )
    assert update_resp.status == 200
    new_data = await update_resp.json()
    assert new_data["revision"] == 2

    # 4. If-Match with invalid ETag fails with 412
    bad_match_headers = {**headers, "If-Match": '"sha256:0000000000000000000000000000000000000000000000000000000000000000"'}
    fail_resp = await client.put(
        "/v1/admin/profiles/etagprof",
        headers=bad_match_headers,
        json={**payload, "description": "Should fail"},
    )
    assert fail_resp.status == 412

    # 5. If-Match with revision string ("2") fails with 412 (never treat revision as valid ETag)
    rev_match_headers = {**headers, "If-Match": "2"}
    fail_rev_resp = await client.put(
        "/v1/admin/profiles/etagprof",
        headers=rev_match_headers,
        json={**payload, "description": "Should fail"},
    )
    assert fail_rev_resp.status == 412

    # 6. Repeat identical PUT is idempotent, returns 200, preserves revision & ETag
    repeat_resp = await client.put(
        "/v1/admin/profiles/etagprof",
        headers=headers,
        json={**payload, "description": "Updated ETag test"},
    )
    assert repeat_resp.status == 200
    repeat_data = await repeat_resp.json()
    assert repeat_data["revision"] == 2
    assert repeat_data["digest"] == new_data["digest"]


@pytest.mark.asyncio
async def test_stale_if_match_blocks_profile_skill_and_file_deletes(create_api_client):
    """DELETE must enforce the same digest precondition as PUT before mutating resources."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    profile_create = await client.put(
        "/v1/admin/profiles/delete-etag-profile",
        headers=headers,
        json={"description": "v1"},
    )
    stale_profile_digest = (await profile_create.json())["digest"]
    profile_update = await client.put(
        "/v1/admin/profiles/delete-etag-profile",
        headers={**headers, "If-Match": stale_profile_digest},
        json={"description": "v2"},
    )
    assert profile_update.status == 200
    profile_delete = await client.delete(
        "/v1/admin/profiles/delete-etag-profile",
        headers={**headers, "If-Match": stale_profile_digest},
    )
    assert profile_delete.status == 412
    assert (await client.get("/v1/admin/profiles/delete-etag-profile", headers=headers)).status == 200

    await client.put("/v1/admin/profiles/delete-etag-child", headers=headers, json={})
    skill_v1 = "---\nname: guarded-skill\ndescription: v1\n---\nDo v1\n"
    skill_create = await client.put(
        "/v1/admin/profiles/delete-etag-child/skills/guarded-skill",
        headers=headers,
        json={"content": skill_v1},
    )
    stale_skill_digest = (await skill_create.json())["digest"]
    skill_update = await client.put(
        "/v1/admin/profiles/delete-etag-child/skills/guarded-skill",
        headers={**headers, "If-Match": stale_skill_digest},
        json={"content": skill_v1 + "updated\n"},
    )
    assert skill_update.status == 200
    skill_delete = await client.delete(
        "/v1/admin/profiles/delete-etag-child/skills/guarded-skill",
        headers={**headers, "If-Match": stale_skill_digest},
    )
    assert skill_delete.status == 412
    assert (
        await client.get(
            "/v1/admin/profiles/delete-etag-child/skills/guarded-skill",
            headers=headers,
        )
    ).status == 200

    file_create = await client.put(
        "/v1/admin/profiles/delete-etag-child/files/context/guarded.md",
        headers=headers,
        json={"content": "v1"},
    )
    stale_file_digest = (await file_create.json())["digest"]
    file_update = await client.put(
        "/v1/admin/profiles/delete-etag-child/files/context/guarded.md",
        headers={**headers, "If-Match": stale_file_digest},
        json={"content": "v2"},
    )
    assert file_update.status == 200
    file_delete = await client.delete(
        "/v1/admin/profiles/delete-etag-child/files/context/guarded.md",
        headers={**headers, "If-Match": stale_file_digest},
    )
    assert file_delete.status == 412
    assert (
        await client.get(
            "/v1/admin/profiles/delete-etag-child/files/context/guarded.md",
            headers=headers,
        )
    ).status == 200


@pytest.mark.asyncio
async def test_child_skill_and_file_metadata_persistence(create_api_client):
    """Verify skill and file revisions and digests are persisted and read back after restart."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    # Setup profile
    await client.put("/v1/admin/profiles/childmeta", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})

    # 1. Create skill
    skill_content = "---\nname: my-skill\ndescription: test\n---\n# Skill\n"
    s_resp = await client.put("/v1/admin/profiles/childmeta/skills/my-skill", headers=headers, json={"content": skill_content})
    assert s_resp.status in (200, 201)
    s_data = await s_resp.json()
    assert s_data["revision"] == 1
    s_digest = s_data["digest"]

    # Read back skill via GET -> exact same revision and digest
    s_get = await client.get("/v1/admin/profiles/childmeta/skills/my-skill", headers=headers)
    assert s_get.status == 200
    s_get_data = await s_get.json()
    assert s_get_data["revision"] == 1
    assert s_get_data["digest"] == s_digest

    # 2. Update skill -> revision becomes 2
    s_update = await client.put(
        "/v1/admin/profiles/childmeta/skills/my-skill",
        headers={**headers, "If-Match": s_digest},
        json={"content": skill_content + "\n# Updated\n"},
    )
    assert s_update.status == 200
    s_up_data = await s_update.json()
    assert s_up_data["revision"] == 2

    # 3. Create file
    f_resp = await client.put("/v1/admin/profiles/childmeta/files/context/custom.txt", headers=headers, json={"content": "File text"})
    assert f_resp.status == 201
    f_data = await f_resp.json()
    assert f_data["revision"] == 1
    f_digest = f_data["digest"]

    # Read back file via GET -> exact same revision and digest
    f_get = await client.get("/v1/admin/profiles/childmeta/files/context/custom.txt", headers=headers)
    assert f_get.status == 200
    assert f_get.headers.get("ETag") == f'"{f_digest}"'
    f_get_data = await f_get.json()
    assert f_get_data["revision"] == 1
    assert f_get_data["digest"] == f_digest


@pytest.mark.asyncio
async def test_skills_prompt_cache_invalidation(create_api_client):
    """Verify clear_skills_system_prompt_cache is invoked on skill create/update/delete."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    await client.put("/v1/admin/profiles/cacheprof", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})

    with patch("agent.prompt_builder.clear_skills_system_prompt_cache") as mock_clear:
        # Create
        r1 = await client.put("/v1/admin/profiles/cacheprof/skills/s1", headers=headers, json={"content": "---\nname: s1\ndescription: d\n---\n# S1"})
        assert r1.status in (200, 201)
        assert mock_clear.called

        mock_clear.reset_mock()
        # Update
        r2 = await client.put("/v1/admin/profiles/cacheprof/skills/s1", headers=headers, json={"content": "---\nname: s1\ndescription: d2\n---\n# S1 V2"})
        assert r2.status == 200
        assert mock_clear.called

        mock_clear.reset_mock()
        # Delete
        r3 = await client.delete("/v1/admin/profiles/cacheprof/skills/s1", headers=headers)
        assert r3.status == 200
        assert mock_clear.called


@pytest.mark.asyncio
async def test_managed_child_scope_and_unmanaged_preservation(create_api_client, temp_hermes_home):
    """Verify only managed children are listed/accessible and unmanaged children are preserved on delete."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    # 1. Create managed profile via API
    await client.put("/v1/admin/profiles/scopetest", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})

    prof_dir = temp_hermes_home / "profiles" / "scopetest"

    # Add unmanaged skill directly on filesystem
    unmanaged_skill_dir = prof_dir / "skills" / "human-skill"
    unmanaged_skill_dir.mkdir(parents=True, exist_ok=True)
    (unmanaged_skill_dir / "SKILL.md").write_text("---\nname: human-skill\ndescription: h\n---\n# Human", encoding="utf-8")

    # Add unmanaged file directly on filesystem
    (prof_dir / "memories" / "USER.md").write_text("Pre-existing human user memory", encoding="utf-8")

    # Create managed skill via API
    await client.put("/v1/admin/profiles/scopetest/skills/api-skill", headers=headers, json={"content": "---\nname: api-skill\ndescription: a\n---\n# API Skill"})

    # Create managed file via API
    await client.put("/v1/admin/profiles/scopetest/files/SOUL.md", headers=headers, json={"content": "API Soul"})

    # List skills -> returns ONLY api-skill
    skills_resp = await client.get("/v1/admin/profiles/scopetest/skills", headers=headers)
    assert skills_resp.status == 200
    skills_data = await skills_resp.json()
    skill_slugs = [s["skill_slug"] for s in skills_data["data"]]
    assert skill_slugs == ["api-skill"]

    # List files -> returns ONLY SOUL.md
    files_resp = await client.get("/v1/admin/profiles/scopetest/files", headers=headers)
    assert files_resp.status == 200
    files_data = await files_resp.json()
    file_paths = [f["path"] for f in files_data["data"]]
    assert file_paths == ["SOUL.md"]

    # GET unmanaged skill -> 404
    g_skill = await client.get("/v1/admin/profiles/scopetest/skills/human-skill", headers=headers)
    assert g_skill.status == 404

    # GET unmanaged file -> 404
    g_file = await client.get("/v1/admin/profiles/scopetest/files/memories/USER.md", headers=headers)
    assert g_file.status == 404

    # DELETE unmanaged skill -> 404 and file remains on disk
    d_skill = await client.delete("/v1/admin/profiles/scopetest/skills/human-skill", headers=headers)
    assert d_skill.status == 404
    assert (unmanaged_skill_dir / "SKILL.md").exists()

    # DELETE unmanaged file -> 404 and file remains on disk
    d_file = await client.delete("/v1/admin/profiles/scopetest/files/memories/USER.md", headers=headers)
    assert d_file.status == 404
    assert (prof_dir / "memories" / "USER.md").exists()

    # DELETE profile with unmanaged resources -> 409 profile_not_empty without mutating disk
    del_prof = await client.delete("/v1/admin/profiles/scopetest", headers=headers)
    assert del_prof.status == 409
    assert (await del_prof.json())["error"]["code"] == "profile_not_empty"
    assert (prof_dir / "SOUL.md").exists()
    assert (prof_dir / "skills" / "api-skill").exists()
    assert (prof_dir / "skills" / "human-skill").exists()
    assert (prof_dir / "memories" / "USER.md").exists()


@pytest.mark.asyncio
async def test_profile_deletion_safety_and_symlinks(create_api_client, temp_hermes_home):
    """Verify default profile protected, unmanaged protected, active profile 409, and symlinks not followed."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    # 1. Delete default -> 403
    d_def = await client.delete("/v1/admin/profiles/default", headers=headers)
    assert d_def.status == 403

    # 2. Delete unmanaged -> 409
    unmanaged_dir = temp_hermes_home / "profiles" / "unmanaged_p"
    unmanaged_dir.mkdir(parents=True, exist_ok=True)
    d_unm = await client.delete("/v1/admin/profiles/unmanaged_p", headers=headers)
    assert d_unm.status == 409
    assert unmanaged_dir.exists()

    # 3. Create managed profile and add an external symlink inside it
    await client.put("/v1/admin/profiles/symtest", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})
    sym_dir = temp_hermes_home / "profiles" / "symtest"
    outside_target = temp_hermes_home / "outside_secret.txt"
    outside_target.write_text("secret", encoding="utf-8")

    try:
        (sym_dir / "symlink_file").symlink_to(outside_target)
    except OSError:
        pass

    # Mark symlink_file as a managed file in manifest so preflight scan passes
    manifest_path = sym_dir / ".control_plane_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["files"]["symlink_file"] = {"managed": True, "revision": 1, "digest": "sha256:dummy"}
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    d_sym = await client.delete("/v1/admin/profiles/symtest", headers=headers)
    assert d_sym.status == 200
    # Outside file must remain untouched
    assert outside_target.exists()


@pytest.mark.asyncio
async def test_request_payload_bounds(create_api_client):
    """Verify payload size bounds and field length bounds."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    # Oversized field length -> 400
    huge_name = "a" * 300
    resp = await client.put(
        f"/v1/admin/profiles/{huge_name}",
        headers=headers,
        json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"},
    )
    assert resp.status == 400


@pytest.mark.asyncio
async def test_path_safety_strict_codes(create_api_client):
    """Verify exact 400 for absolute paths, exact 403 for traversal / forbidden files / workspace."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    await client.put("/v1/admin/profiles/pathtest", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})

    # 1. Absolute path -> 400
    r_abs = await client.put("/v1/admin/profiles/pathtest/files//etc/passwd", headers=headers, json={"content": "x"})
    assert r_abs.status == 400

    # 2. Traversal -> 403
    r_trav = await client.put("/v1/admin/profiles/pathtest/files/context%2f..%2f..%2fetc/passwd", headers=headers, json={"content": "x"})
    assert r_trav.status == 403

    # 3. Forbidden file -> 403
    r_forb = await client.put("/v1/admin/profiles/pathtest/files/.env", headers=headers, json={"content": "x"})
    assert r_forb.status == 403

    r_man = await client.put("/v1/admin/profiles/pathtest/files/.control_plane_manifest.json", headers=headers, json={"content": "x"})
    assert r_man.status == 403

    # 4. workspace/ subtree -> 403 (removed from allowlist)
    r_work = await client.put("/v1/admin/profiles/pathtest/files/workspace/code.py", headers=headers, json={"content": "x"})
    assert r_work.status == 403

    # 5. Allowed path -> 200/201
    r_ok = await client.put("/v1/admin/profiles/pathtest/files/context/notes.txt", headers=headers, json={"content": "x"})
    assert r_ok.status in (200, 201)


@pytest.mark.asyncio
async def test_deterministic_sorting_and_post_routes(create_api_client):
    """Verify collection POST routes require name/slug and lists return sorted results."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        **OWNER_HEADERS,
    }

    # POST /v1/admin/profiles without name -> 400
    r_no_name = await client.post("/v1/admin/profiles", headers=headers, json={"managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})
    assert r_no_name.status == 400

    # POST /v1/admin/profiles with name -> 201
    r_post_p1 = await client.post("/v1/admin/profiles", headers=headers, json={"name": "z-prof", "managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})
    assert r_post_p1.status == 201

    r_post_p2 = await client.post("/v1/admin/profiles", headers=headers, json={"name": "a-prof", "managed_by": "control_plane", "tenant_id": "tenant-a", "resource_id": "res-1"})
    assert r_post_p2.status == 201

    # List profiles -> sorted by name: ["a-prof", "z-prof"]
    l_profs = await client.get("/v1/admin/profiles", headers=headers)
    assert l_profs.status == 200
    p_names = [p["name"] for p in (await l_profs.json())["data"]]
    assert p_names == ["a-prof", "z-prof"]

    # POST skill without name/slug -> 400
    r_no_skill = await client.post("/v1/admin/profiles/a-prof/skills", headers=headers, json={"content": "---\nname: x\n---\n# X"})
    assert r_no_skill.status == 400

    # POST skill with name -> 200/201
    r_skill1 = await client.post("/v1/admin/profiles/a-prof/skills", headers=headers, json={"name": "z-skill", "content": "---\nname: z-skill\ndescription: z\n---\n# Z"})
    assert r_skill1.status in (200, 201)

    r_skill2 = await client.post("/v1/admin/profiles/a-prof/skills", headers=headers, json={"name": "a-skill", "content": "---\nname: a-skill\ndescription: a\n---\n# A"})
    assert r_skill2.status in (200, 201)

    # List skills -> sorted by skill_slug: ["a-skill", "z-skill"]
    l_skills = await client.get("/v1/admin/profiles/a-prof/skills", headers=headers)
    assert l_skills.status == 200
    s_slugs = [s["skill_slug"] for s in (await l_skills.json())["data"]]
    assert s_slugs == ["a-skill", "z-skill"]


@pytest.mark.asyncio
async def test_canonical_profile_digest_and_drift(create_api_client, temp_hermes_home):
    """Verify GET computes actual digest/drift without writing, PUT reads back and updates applied_digest."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    # Create profile
    resp = await client.put("/v1/admin/profiles/driftprof", headers=headers, json={"soul": "Original soul"})
    assert resp.status == 201
    data = await resp.json()
    orig_digest = data["digest"]
    assert data["applied_digest"] == orig_digest
    assert data["drifted"] is False
    assert resp.headers.get("ETag") == f'"{orig_digest}"'

    # Manually mutate SOUL.md on disk (introducing drift)
    prof_dir = temp_hermes_home / "profiles" / "driftprof"
    (prof_dir / "SOUL.md").write_text("Drifted soul on disk", encoding="utf-8")

    # GET must compute actual digest from disk, set applied_digest to manifest spec_digest, and set drifted=True
    get_resp = await client.get("/v1/admin/profiles/driftprof", headers=headers)
    assert get_resp.status == 200
    get_data = await get_resp.json()
    assert get_data["digest"] != orig_digest
    assert get_data["applied_digest"] == orig_digest
    assert get_data["drifted"] is True
    assert get_resp.headers.get("ETag") == f'"{get_data["digest"]}"'

    # Successful PUT reads back actual state, updates manifest spec_digest to match
    put_resp = await client.put("/v1/admin/profiles/driftprof", headers=headers, json={"soul": "Updated soul via PUT"})
    assert put_resp.status == 200
    put_data = await put_resp.json()
    assert put_data["digest"] == put_data["applied_digest"]
    assert put_data["drifted"] is False


@pytest.mark.asyncio
async def test_soul_user_file_api_recomputes_profile_digest(create_api_client):
    """Verify direct PUT/DELETE of SOUL.md or memories/USER.md recomputes profile applied_digest."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/filedigestprof", headers=headers, json={})
    p1 = await (await client.get("/v1/admin/profiles/filedigestprof", headers=headers)).json()

    # PUT SOUL.md via file API
    f_put = await client.put("/v1/admin/profiles/filedigestprof/files/SOUL.md", headers=headers, json={"content": "New SOUL"})
    assert f_put.status in (200, 201)
    p2 = await (await client.get("/v1/admin/profiles/filedigestprof", headers=headers)).json()
    assert p2["digest"] != p1["digest"]
    assert p2["applied_digest"] == p2["digest"]
    assert p2["drifted"] is False

    # DELETE SOUL.md via file API
    f_del = await client.delete("/v1/admin/profiles/filedigestprof/files/SOUL.md", headers=headers)
    assert f_del.status == 200
    p3 = await (await client.get("/v1/admin/profiles/filedigestprof", headers=headers)).json()
    assert p3["digest"] != p2["digest"]
    assert p3["applied_digest"] == p3["digest"]
    assert p3["drifted"] is False


@pytest.mark.asyncio
async def test_clearing_desired_soul_and_user_context(create_api_client):
    """Verify updating soul or user_context to empty string keeps child metadata and content consistent."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    # Set non-empty initially
    await client.put("/v1/admin/profiles/cleartest", headers=headers, json={"soul": "Soul 1", "user_context": "User 1"})

    s_get1 = await client.get("/v1/admin/profiles/cleartest/files/SOUL.md", headers=headers)
    assert s_get1.status == 200
    assert (await s_get1.json())["revision"] == 1

    u_get1 = await client.get("/v1/admin/profiles/cleartest/files/memories/USER.md", headers=headers)
    assert u_get1.status == 200
    assert (await u_get1.json())["revision"] == 1

    # Update to empty strings
    update_resp = await client.put("/v1/admin/profiles/cleartest", headers=headers, json={"soul": "", "user_context": ""})
    assert update_resp.status == 200
    p_data = await update_resp.json()
    assert p_data["soul"] == ""
    assert p_data["user_context"] == ""

    # Child GET must return 200 with content="" and revision=2
    s_get2 = await client.get("/v1/admin/profiles/cleartest/files/SOUL.md", headers=headers)
    assert s_get2.status == 200
    s_data2 = await s_get2.json()
    assert s_data2["content"] == ""
    assert s_data2["revision"] == 2

    u_get2 = await client.get("/v1/admin/profiles/cleartest/files/memories/USER.md", headers=headers)
    assert u_get2.status == 200
    u_data2 = await u_get2.json()
    assert u_data2["content"] == ""
    assert u_data2["revision"] == 2


@pytest.mark.asyncio
async def test_no_swallowed_persistence_failures(create_api_client, temp_hermes_home):
    """Verify profile PUT fails 500 when disk write fails and manifest is not updated."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/failprof", headers=headers, json={"description": "Original"})
    manifest_path = temp_hermes_home / "profiles" / "failprof" / ".control_plane_manifest.json"
    manifest_before = manifest_path.read_text(encoding="utf-8")

    with patch("gateway.admin._atomic_write_file", side_effect=OSError("Disk failure")):
        resp = await client.put("/v1/admin/profiles/failprof", headers=headers, json={"soul": "New soul"})
        assert resp.status == 500

    # Manifest must not be updated
    manifest_after = manifest_path.read_text(encoding="utf-8")
    assert manifest_before == manifest_after


@pytest.mark.asyncio
async def test_no_adoption_of_unmanaged_children(create_api_client, temp_hermes_home):
    """Verify PUT skill or file on existing unmanaged disk target returns 409 and preserves bytes."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/unmanprof", headers=headers, json={})
    prof_dir = temp_hermes_home / "profiles" / "unmanprof"

    # Create unmanaged file on disk
    unmanaged_file = prof_dir / "context" / "secret.txt"
    unmanaged_file.parent.mkdir(parents=True, exist_ok=True)
    unmanaged_file.write_text("Unmanaged bytes", encoding="utf-8")

    f_put = await client.put("/v1/admin/profiles/unmanprof/files/context/secret.txt", headers=headers, json={"content": "New bytes"})
    assert f_put.status == 409
    assert (await f_put.json())["error"]["code"] == "unmanaged_resource_conflict"
    assert unmanaged_file.read_text(encoding="utf-8") == "Unmanaged bytes"

    # Create unmanaged skill on disk
    unmanaged_skill = prof_dir / "skills" / "unmanaged-skill" / "SKILL.md"
    unmanaged_skill.parent.mkdir(parents=True, exist_ok=True)
    unmanaged_skill.write_text("---\nname: unmanaged-skill\ndescription: unmanaged\n---\n# Unmanaged", encoding="utf-8")

    s_put = await client.put(
        "/v1/admin/profiles/unmanprof/skills/unmanaged-skill",
        headers=headers,
        json={"content": "---\nname: unmanaged-skill\ndescription: edited\n---\n# Edited"},
    )
    assert s_put.status == 409
    assert (await s_put.json())["error"]["code"] == "unmanaged_resource_conflict"
    assert unmanaged_skill.read_text(encoding="utf-8") == "---\nname: unmanaged-skill\ndescription: unmanaged\n---\n# Unmanaged"


@pytest.mark.asyncio
async def test_profile_deletion_preflight_and_unmanaged_blocking(create_api_client, temp_hermes_home):
    """Verify profile delete preflight blocks on unmanaged content and requires ownership for absent profiles."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/delpreflight", headers=headers, json={})
    prof_dir = temp_hermes_home / "profiles" / "delpreflight"

    # Add unmanaged user file
    unmanaged = prof_dir / "unknown_file.txt"
    unmanaged.write_text("user data", encoding="utf-8")

    del_resp = await client.delete("/v1/admin/profiles/delpreflight", headers=headers)
    assert del_resp.status == 409
    assert (await del_resp.json())["error"]["code"] == "profile_not_empty"
    assert prof_dir.exists()

    # Remove unmanaged file
    unmanaged.unlink()
    del_resp2 = await client.delete("/v1/admin/profiles/delpreflight", headers=headers)
    assert del_resp2.status == 200
    assert not prof_dir.exists()

    # Absent profile deletion without ownership headers fails 400
    del_absent_no_owner = await client.delete("/v1/admin/profiles/nonexistent", headers={"Authorization": f"Bearer {key}"})
    assert del_absent_no_owner.status == 400

    # Absent profile deletion with complete ownership returns 200
    del_absent_ok = await client.delete("/v1/admin/profiles/nonexistent", headers=headers)
    assert del_absent_ok.status == 200


@pytest.mark.asyncio
async def test_active_and_busy_profile_deletion_guard(create_api_client, monkeypatch):
    """Verify active request scope, running gateway process, or active runs block profile deletion with 409."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/activeprof", headers=headers, json={})
    from hermes_cli.profiles import get_profile_dir
    prof_dir = get_profile_dir("activeprof")

    # 1. Active in request scope via _api_request_profile
    class DummyContextVar:
        def get(self, default=None):
            return "activeprof"
        def set(self, val):
            return "token"
        def reset(self, token):
            pass

    with monkeypatch.context() as m:
        m.setattr("gateway.admin._api_request_profile", DummyContextVar())
        r1 = await client.delete("/v1/admin/profiles/activeprof", headers=headers)
        assert r1.status == 409
        assert (await r1.json())["error"]["code"] == "profile_active_conflict"

    # 2. Running gateway process reported by list_profiles()
    from hermes_cli.profiles import ProfileInfo
    def mock_list_profiles():
        return [ProfileInfo(name="activeprof", path=prof_dir, is_default=False, gateway_running=True)]

    with monkeypatch.context() as m:
        m.setattr("gateway.admin.list_profiles", mock_list_profiles)
        r2 = await client.delete("/v1/admin/profiles/activeprof", headers=headers)
        assert r2.status == 409
        assert (await r2.json())["error"]["code"] == "profile_active_conflict"

    # 3. Active run in APIServerAdapter
    adapter = client.app["api_server_adapter"]
    adapter._run_statuses["run-123"] = {"status": "running", "profile": "activeprof"}
    r3 = await client.delete("/v1/admin/profiles/activeprof", headers=headers)
    assert r3.status == 409
    assert (await r3.json())["error"]["code"] in ("profile_busy_conflict", "profile_active_conflict")
    adapter._run_statuses.pop("run-123", None)


@pytest.mark.asyncio
async def test_skill_validity_frontmatter_enforcement(create_api_client, temp_hermes_home):
    """Verify skill creation/update validates frontmatter name, description, and name matching slug."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/skillvalprof", headers=headers, json={})
    prof_dir = temp_hermes_home / "profiles" / "skillvalprof"

    # Missing description in frontmatter -> 400
    r1 = await client.put(
        "/v1/admin/profiles/skillvalprof/skills/my-skill",
        headers=headers,
        json={"content": "---\nname: my-skill\n---\n# Skill without desc"},
    )
    assert r1.status == 400
    assert not (prof_dir / "skills" / "my-skill").exists()

    # Frontmatter name does not match path slug -> 400
    r2 = await client.put(
        "/v1/admin/profiles/skillvalprof/skills/my-skill",
        headers=headers,
        json={"content": "---\nname: wrong-slug\ndescription: desc\n---\n# Mismatched"},
    )
    assert r2.status == 400
    assert not (prof_dir / "skills" / "my-skill").exists()

    # Valid skill -> 201
    r3 = await client.put(
        "/v1/admin/profiles/skillvalprof/skills/my-skill",
        headers=headers,
        json={"content": "---\nname: my-skill\ndescription: valid description\n---\n# Valid"},
    )
    assert r3.status == 201
    assert (prof_dir / "skills" / "my-skill" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_manifest_corruption_fails_closed(create_api_client, temp_hermes_home):
    """Verify corrupt manifest fails closed on write/read attempts and is not silently overwritten."""
    client, key = await create_api_client(admin_config_rw=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **OWNER_HEADERS}

    await client.put("/v1/admin/profiles/corruptprof", headers=headers, json={})
    prof_dir = temp_hermes_home / "profiles" / "corruptprof"
    manifest_path = prof_dir / ".control_plane_manifest.json"
    manifest_path.write_text("{corrupt json", encoding="utf-8")

    # GET profile fails
    g_resp = await client.get("/v1/admin/profiles/corruptprof", headers=headers)
    assert g_resp.status == 409

    # PUT profile fails and does not reset corrupt manifest
    p_resp = await client.put("/v1/admin/profiles/corruptprof", headers=headers, json={"description": "new desc"})
    assert p_resp.status == 409
    assert manifest_path.read_text(encoding="utf-8") == "{corrupt json"
