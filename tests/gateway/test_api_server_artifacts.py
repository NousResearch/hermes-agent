"""Tests for the Hermes API server artifact workspace endpoint."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/artifacts", adapter._handle_list_artifacts)
    app.router.add_get("/api/artifacts/{artifact_path:.+}", adapter._handle_get_artifact)
    return app


@pytest.fixture
def artifact_adapter(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    reports = tmp_path / "reports"
    outputs.mkdir()
    reports.mkdir()
    (outputs / "report.md").write_text("# Weekly report\nReady for review.\n", encoding="utf-8")
    (outputs / ".env").write_text("API_KEY=secret", encoding="utf-8")
    (outputs / "video.mp4").write_bytes(b"\x00\x00\x00\x20ftypmp42")
    (outputs / "project").mkdir()
    (outputs / "project" / "index.html").write_text("<h1>Demo</h1>", encoding="utf-8")
    (reports / "daily.txt").write_text("Daily summary", encoding="utf-8")

    monkeypatch.setenv(
        "API_SERVER_ARTIFACT_ROOTS",
        f"outputs={outputs},reports={reports}",
    )
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))


class TestArtifactsEndpoint:
    @pytest.mark.asyncio
    async def test_requires_bearer_token(self, artifact_adapter):
        app = _create_app(artifact_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/artifacts")

        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_lists_virtual_roots_and_safe_files(self, artifact_adapter):
        app = _create_app(artifact_adapter)
        headers = {"Authorization": "Bearer sk-test"}

        async with TestClient(TestServer(app)) as cli:
            root_resp = await cli.get("/api/artifacts", headers=headers)
            assert root_resp.status == 200, await root_resp.text()
            root_body = await root_resp.json()

            outputs_resp = await cli.get("/api/artifacts?path=outputs", headers=headers)
            assert outputs_resp.status == 200, await outputs_resp.text()
            outputs_body = await outputs_resp.json()

        assert root_body["object"] == "hermes.artifacts.list"
        assert {item["path"] for item in root_body["items"]} == {"outputs", "reports"}

        paths = {item["path"] for item in outputs_body["items"]}
        assert "outputs/report.md" in paths
        assert "outputs/project" in paths
        assert "outputs/video.mp4" in paths
        assert "outputs/.env" not in paths

        report = next(item for item in outputs_body["items"] if item["path"] == "outputs/report.md")
        assert report["artifactType"] == "document"
        assert report["preview"].startswith("# Weekly report")

    @pytest.mark.asyncio
    async def test_rejects_traversal(self, artifact_adapter):
        app = _create_app(artifact_adapter)
        headers = {"Authorization": "Bearer sk-test"}

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/artifacts?path=outputs/../..", headers=headers)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_file_metadata_mode(self, artifact_adapter):
        app = _create_app(artifact_adapter)
        headers = {"Authorization": "Bearer sk-test"}

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/artifacts/outputs/report.md?metadata=1", headers=headers)
            assert resp.status == 200, await resp.text()
            body = await resp.json()

        assert body["kind"] == "file"
        assert body["name"] == "report.md"
        assert body["preview"].startswith("# Weekly report")
