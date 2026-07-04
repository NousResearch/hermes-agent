import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


AUTH_HEADERS = {"Authorization": "Bearer sk-secret"}


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))


def _create_import_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/imports/analyze", adapter._handle_import_analyze)
    app.router.add_post("/api/imports/{import_id}/confirm", adapter._handle_import_confirm)
    return app


def _strict_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "owners": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}},
                },
            },
            "horses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "owner_name": {"type": "string"},
                    },
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


@pytest.mark.asyncio
async def test_glys_import_analyze_empty_multipart_requires_files():
    adapter = _make_adapter()
    app = _create_import_app(adapter)
    form = FormData()
    form.add_field("install_id", "install-test", content_type="text/plain")

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/imports/analyze", data=form, headers=AUTH_HEADERS)
        body = await resp.json()

    assert resp.status == 400
    assert body["error"] == "GLYSAI_IMPORT_FILES_REQUIRED"


@pytest.mark.asyncio
async def test_glys_import_analyze_returns_strict_import_plan_and_no_cache_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    app = _create_import_app(adapter)
    final_plan = {
        "owners": [],
        "horses": [
            {
                "name": "Brume",
                "owner_name": "C. Martin",
                "raw_file_content": "horse_name,owner\nBrume,C. Martin",
                "unexpected_nested": "drop me",
            }
        ],
        "warnings": [{"kind": "review", "message": "A verifier"}],
        "raw_file_content": "horse_name,owner\nBrume,C. Martin",
        "unexpected_root": True,
    }
    run_agent = AsyncMock(return_value=({"final_response": json.dumps(final_plan)}, {"total_tokens": 12}))

    form = FormData()
    form.add_field("files", b"horse_name,owner\nBrume,C. Martin\n", filename="../secret/registre.csv", content_type="text/csv")
    form.add_field("install_id", "install-test")
    form.add_field("raw_file_persistence", "false")
    form.add_field("provider_file_retention", "delete_after_analysis")
    form.add_field("backend_transport_mode", "bounded_replayable_bytes")
    form.add_field("backend_replayable_bytes_required", "true")
    form.add_field("requested_schema_version", "test-v1")
    form.add_field("requested_schema", json.dumps(_strict_schema()))
    form.add_field("context", json.dumps({"license_key": "lic-should-not-be-in-prompt", "note": "ok"}))
    form.add_field("files_metadata", json.dumps([{"name": "registre.csv", "size": 33}]))

    with patch.object(adapter, "_run_agent", run_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/imports/analyze", data=form, headers=AUTH_HEADERS)
            body = await resp.json()

    assert resp.status == 200, body
    assert set(body.keys()) == {"importId", "proposal"}
    assert body["importId"].startswith("import_")
    proposal = body["proposal"]
    assert set(proposal.keys()) == {"owners", "horses", "warnings"}
    assert proposal["horses"] == [{"name": "Brume", "owner_name": "C. Martin"}]
    assert all(isinstance(item, str) for item in proposal["warnings"])
    assert "raw_file_content" not in json.dumps(proposal)
    assert "unexpected_root" not in proposal

    run_agent.assert_awaited_once()
    kwargs = run_agent.await_args.kwargs
    assert kwargs["persist"] is False
    assert kwargs["skip_memory"] is True
    assert kwargs["enabled_toolsets_override"] == []
    assert "../secret" not in kwargs["user_message"]
    assert "lic-should-not-be-in-prompt" not in kwargs["user_message"]

    cache_root = tmp_path / "cache" / "glys_imports"
    assert not [p for p in cache_root.rglob("*") if p.is_file()]


@pytest.mark.asyncio
async def test_glys_import_provider_error_has_no_usable_import_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    app = _create_import_app(adapter)
    run_agent = AsyncMock(side_effect=TimeoutError("provider timeout"))

    form = FormData()
    form.add_field("files", b"horse_name,owner\nBrume,C. Martin\n", filename="registre.csv", content_type="text/csv")
    form.add_field("requested_schema", json.dumps(_strict_schema()))

    with patch.object(adapter, "_run_agent", run_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/imports/analyze", data=form, headers=AUTH_HEADERS)
            body = await resp.json()

    assert resp.status == 502
    assert body["error"] == "GLYSAI_IMPORT_ANALYSIS_FAILED"
    assert "importId" not in body
    assert "import_id" not in body


@pytest.mark.asyncio
async def test_glys_import_confirm_deletes_spool_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    app = _create_import_app(adapter)
    spool = tmp_path / "cache" / "glys_imports" / "import_test"
    spool.mkdir(parents=True)
    (spool / "raw.tmp").write_text("must disappear", encoding="utf-8")

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/imports/import_test/confirm",
            json={"import_id": "import_test", "install_id": "i", "actor_scope": "owner", "delete_files": True},
            headers=AUTH_HEADERS,
        )
        body = await resp.json()
        resp2 = await cli.post(
            "/api/imports/import_test/confirm",
            json={"import_id": "import_test", "delete_files": True},
            headers=AUTH_HEADERS,
        )
        body2 = await resp2.json()

    assert resp.status == 200
    assert body == {"importId": "import_test", "deleted": True, "files_deleted": True}
    assert not spool.exists()
    assert resp2.status == 200
    assert body2 == {"importId": "import_test", "deleted": True, "files_deleted": True}


@pytest.mark.asyncio
async def test_glys_import_routes_require_bearer_auth():
    adapter = _make_adapter()
    app = _create_import_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/imports/analyze")
        body = await resp.json()

    assert resp.status == 401
    assert body["error"]["code"] == "invalid_api_key"
