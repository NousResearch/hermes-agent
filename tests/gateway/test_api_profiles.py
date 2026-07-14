"""Tests for scoped profile contracts exposed by the canonical API server.

Covers GET/POST/PATCH/DELETE /api/profiles and GET/PUT /api/profiles/{name}/soul:
list/create/clone/rename/delete and persona (SOUL.md) read/write, all reusing
hermes_cli.profiles domain functions directly (no Dashboard HTTP proxy, no
active-profile mutation endpoint, no leaked filesystem paths/env values/
wrapper commands).
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.api_operator_auth import OperatorCredentialStore
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _profile_test_app(adapter):
    app = web.Application()
    app.router.add_get("/api/profiles", adapter._handle_list_profiles)
    app.router.add_post("/api/profiles", adapter._handle_create_profile)
    app.router.add_patch("/api/profiles/{name}", adapter._handle_patch_profile)
    app.router.add_delete("/api/profiles/{name}", adapter._handle_delete_profile)
    app.router.add_get("/api/profiles/{name}/soul", adapter._handle_get_profile_soul)
    app.router.add_put("/api/profiles/{name}/soul", adapter._handle_put_profile_soul)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _make_adapter(tmp_path, scopes=("profiles:read", "profiles:write")):
    """Return (adapter, headers) wired to a fresh credential store with the given scopes."""
    store = OperatorCredentialStore(tmp_path / "credentials.json")
    token = store.issue("phone", list(scopes)).token
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._operator_credentials = store
    return adapter, {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_profile_contract_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    store = OperatorCredentialStore(tmp_path / "credentials.json")
    token = store.issue("phone", ["profiles:read", "profiles:write"]).token
    headers = {"Authorization": f"Bearer {token}"}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._operator_credentials = store

    async with TestClient(TestServer(_profile_test_app(adapter))) as client:
        created = await client.post(
            "/api/profiles",
            headers=headers,
            json={"name": "coder", "clone_from": "default"},
        )
        listed = await client.get("/api/profiles", headers=headers)
        created_body = await created.json()
        listed_body = await listed.json()

    assert created.status == 201
    assert created_body["name"] == "coder"
    assert any(item["id"] == "coder" for item in listed_body["data"])
    assert not (tmp_path / "hermes" / "active_profile").exists()


class TestProfileList:
    @pytest.mark.asyncio
    async def test_list_never_returns_filesystem_paths_or_wrapper_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)
        monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *a, **kw: None)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "widget"})
            listed = await client.get("/api/profiles", headers=headers)
            body = await listed.json()

        assert listed.status == 200
        for item in body["data"]:
            serialized = str(item)
            assert "path" not in item
            assert "alias" not in item
            assert "wrapper" not in item
            assert str(tmp_path) not in serialized

    @pytest.mark.asyncio
    async def test_list_and_create_independent_of_existing_active_profile_file(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        hermes_home.mkdir(parents=True)
        active_profile = hermes_home / "active_profile"
        active_profile.write_text("ghost\n", encoding="utf-8")

        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            created = await client.post("/api/profiles", headers=headers, json={"name": "widget"})
            listed = await client.get("/api/profiles", headers=headers)
            body = await listed.json()

        assert created.status == 201
        assert active_profile.read_text(encoding="utf-8") == "ghost\n"
        for item in body["data"]:
            assert "active" not in item
            assert "is_active" not in item


class TestProfileCreate:
    @pytest.mark.asyncio
    async def test_create_rejects_traversal_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            response = await client.post(
                "/api/profiles", headers=headers, json={"name": "../../etc"}
            )

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_create_duplicate_name_is_conflict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            first = await client.post("/api/profiles", headers=headers, json={"name": "dup"})
            second = await client.post("/api/profiles", headers=headers, json={"name": "dup"})

        assert first.status == 201
        assert second.status == 409

    @pytest.mark.asyncio
    async def test_create_clone_source_missing_is_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            response = await client.post(
                "/api/profiles",
                headers=headers,
                json={"name": "coder", "clone_from": "does-not-exist"},
            )

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_create_does_not_create_wrapper_alias(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        wrapper_dir = tmp_path / "local_bin"
        monkeypatch.setattr("hermes_cli.profiles._get_wrapper_dir", lambda: wrapper_dir)
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            created = await client.post("/api/profiles", headers=headers, json={"name": "widget"})

        assert created.status == 201
        assert not (wrapper_dir / "widget").exists()


class TestProfileSoul:
    @pytest.mark.asyncio
    async def test_soul_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})

            initial = await client.get("/api/profiles/coder/soul", headers=headers)
            initial_body = await initial.json()

            write_headers = dict(headers, **{"If-Match": initial_body["revision"]})
            written = await client.put(
                "/api/profiles/coder/soul",
                headers=write_headers,
                json={"content": "You are a focused coding agent."},
            )
            written_body = await written.json()

            after = await client.get("/api/profiles/coder/soul", headers=headers)
            after_body = await after.json()

        assert initial.status == 200
        assert written.status == 200
        assert after_body["content"] == "You are a focused coding agent."
        assert after_body["exists"] is True
        assert written_body["revision"] != initial_body["revision"]
        assert after_body["revision"] == written_body["revision"]


class TestProfileScopeEnforcement:
    @pytest.mark.asyncio
    async def test_read_only_scope_cannot_create(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path, scopes=("profiles:read",))

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            response = await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            response_body = await response.json()

        assert response.status == 403
        assert response_body["error"]["code"] == "insufficient_scope"

    @pytest.mark.asyncio
    async def test_read_only_scope_cannot_rename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        write_adapter, write_headers = _make_adapter(tmp_path, scopes=("profiles:read", "profiles:write"))

        async with TestClient(TestServer(_profile_test_app(write_adapter))) as client:
            await client.post("/api/profiles", headers=write_headers, json={"name": "coder"})
            read_token = write_adapter._operator_credentials.issue("reader", ["profiles:read"]).token
            read_headers = {"Authorization": f"Bearer {read_token}"}
            response = await client.patch(
                "/api/profiles/coder",
                headers=dict(read_headers, **{"If-Match": "irrelevant"}),
                json={"new_name": "coder2"},
            )
            response_body = await response.json()

        assert response.status == 403
        assert response_body["error"]["code"] == "insufficient_scope"

    @pytest.mark.asyncio
    async def test_read_only_scope_cannot_delete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path, scopes=("profiles:read", "profiles:write"))

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            read_token = adapter._operator_credentials.issue("reader", ["profiles:read"]).token
            read_headers = {"Authorization": f"Bearer {read_token}"}
            response = await client.delete(
                "/api/profiles/coder", headers=dict(read_headers, **{"If-Match": "irrelevant"})
            )

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_read_only_scope_cannot_write_soul(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path, scopes=("profiles:read", "profiles:write"))

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            read_token = adapter._operator_credentials.issue("reader", ["profiles:read"]).token
            read_headers = {"Authorization": f"Bearer {read_token}"}
            response = await client.put(
                "/api/profiles/coder/soul",
                headers=dict(read_headers, **{"If-Match": "irrelevant"}),
                json={"content": "nope"},
            )

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_unrelated_scope_cannot_create(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path, scopes=("chat:read", "chat:write"))

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            response = await client.post("/api/profiles", headers=headers, json={"name": "coder"})

        assert response.status == 403


class TestProfileConcurrency:
    @pytest.mark.asyncio
    async def test_default_profile_delete_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            listed = await client.get("/api/profiles", headers=headers)
            listed_body = await listed.json()
            default_revision = next(
                item["revision"] for item in listed_body["data"] if item["id"] == "default"
            )
            response = await client.delete(
                "/api/profiles/default",
                headers=dict(headers, **{"If-Match": default_revision}),
            )

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_missing_if_match_returns_428_for_rename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            response = await client.patch(
                "/api/profiles/coder", headers=headers, json={"new_name": "coder2"}
            )
            response_body = await response.json()

        assert response.status == 428
        assert response_body["error"]["code"] == "if_match_required"

    @pytest.mark.asyncio
    async def test_missing_if_match_returns_428_for_delete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            response = await client.delete("/api/profiles/coder", headers=headers)

        assert response.status == 428

    @pytest.mark.asyncio
    async def test_missing_if_match_returns_428_for_soul_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            response = await client.put(
                "/api/profiles/coder/soul", headers=headers, json={"content": "hi"}
            )

        assert response.status == 428

    @pytest.mark.asyncio
    async def test_stale_revision_returns_412_and_performs_no_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            response = await client.patch(
                "/api/profiles/coder",
                headers=dict(headers, **{"If-Match": "rev-stale-does-not-match"}),
                json={"new_name": "coder2"},
            )
            response_body = await response.json()
            listed = await client.get("/api/profiles", headers=headers)
            listed_body = await listed.json()

        assert response.status == 412
        assert response_body["error"]["code"] == "revision_conflict"
        names = {item["id"] for item in listed_body["data"]}
        assert "coder" in names
        assert "coder2" not in names

    @pytest.mark.asyncio
    async def test_successful_rename_returns_new_revision(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            created = await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            created_body = await created.json()
            renamed = await client.patch(
                "/api/profiles/coder",
                headers=dict(headers, **{"If-Match": created_body["revision"]}),
                json={"new_name": "coder2"},
            )
            renamed_body = await renamed.json()

        assert renamed.status == 200
        assert renamed_body["id"] == "coder2"
        assert renamed_body["revision"] != created_body["revision"]

    @pytest.mark.asyncio
    async def test_successful_delete_returns_new_revision(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)
        monkeypatch.setattr("hermes_cli.profiles._cleanup_gateway_service", lambda *a, **kw: None)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            created = await client.post("/api/profiles", headers=headers, json={"name": "coder"})
            created_body = await created.json()
            deleted = await client.delete(
                "/api/profiles/coder",
                headers=dict(headers, **{"If-Match": created_body["revision"]}),
            )
            deleted_body = await deleted.json()
            listed = await client.get("/api/profiles", headers=headers)
            listed_body = await listed.json()

        assert deleted.status == 200
        assert deleted_body["deleted"] is True
        assert deleted_body["revision"] != created_body["revision"]
        assert "coder" not in {item["id"] for item in listed_body["data"]}


class TestProfileCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_advertise_exact_profile_endpoints(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        adapter, headers = _make_adapter(tmp_path)

        async with TestClient(TestServer(_profile_test_app(adapter))) as client:
            response = await client.get("/v1/capabilities", headers=headers)
            body = await response.json()

        endpoints = body["endpoints"]
        assert endpoints["profiles"] == {
            "method": "GET", "path": "/api/profiles",
            "required_scopes": ["profiles:read"], "profile_scoped": False,
        }
        assert endpoints["profile_create"] == {
            "method": "POST", "path": "/api/profiles",
            "required_scopes": ["profiles:write"], "profile_scoped": False,
        }
        assert endpoints["profile_update"] == {
            "method": "PATCH", "path": "/api/profiles/{name}",
            "required_scopes": ["profiles:write"], "profile_scoped": False,
        }
        assert endpoints["profile_delete"] == {
            "method": "DELETE", "path": "/api/profiles/{name}",
            "required_scopes": ["profiles:write"], "profile_scoped": False,
        }
        assert endpoints["profile_soul"] == {
            "method": "GET", "path": "/api/profiles/{name}/soul",
            "required_scopes": ["profiles:read"], "profile_scoped": False,
        }
        assert endpoints["profile_soul_update"] == {
            "method": "PUT", "path": "/api/profiles/{name}/soul",
            "required_scopes": ["profiles:write"], "profile_scoped": False,
        }
