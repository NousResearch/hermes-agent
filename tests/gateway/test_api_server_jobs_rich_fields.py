"""
Tests for the richer cron-job fields on the API server's /api/jobs surface.

The CLI (`hermes cron create/edit`) and the `cronjob` tool have always been able
to set change detection, run-to-run continuity, a narrowed toolset and a per-job
model pin. These tests cover accepting the same fields over REST, and — the
point of the change — that they are validated the SAME way, so a job created
over REST is indistinguishable from one created from the CLI.

Covers:
- create passes enabled_toolsets / continuity / context_from / monitor_* /
  model / provider / reasoning_effort through to create_job
- continuity is translated to the reserved "self" context_from ref
- context_from references are checked for existence ("self" excepted)
- monitor_script goes through the same path-containment validator as the CLI
- type errors return 400, not 500
- create_job's own ValueErrors (reasoning_effort grammar, monitor mutual
  exclusion) surface as 400
- update accepts the same fields, merges continuity onto the STORED
  context_from, and rejects a monitor_script + monitor_url pair
- the update allowlist still rejects unknown keys
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"

SAMPLE_JOB = {
    "id": "aabbccddeeff",
    "name": "test-job",
    "schedule": "*/5 * * * *",
    "prompt": "do something",
    "deliver": "local",
    "enabled": True,
}

OTHER_JOB = {"id": "112233445566", "name": "upstream-job"}

VALID_JOB_ID = "aabbccddeeff"
OTHER_JOB_ID = "112233445566"


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/jobs", adapter._handle_create_job)
    app.router.add_patch("/api/jobs/{job_id}", adapter._handle_update_job)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


def _get_job_stub(job_id):
    return {VALID_JOB_ID: SAMPLE_JOB, OTHER_JOB_ID: OTHER_JOB}.get(job_id)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreateJobRichFields:
    @pytest.mark.asyncio
    async def test_create_passes_rich_fields_through(self, adapter):
        """Every widened field reaches create_job with the CLI's shape."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(f"{_MOD}._cron_get", MagicMock(side_effect=_get_job_stub)):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "enabled_toolsets": ["web", "files"],
                    "context_from": [OTHER_JOB_ID],
                    "monitor_url": "https://example.invalid/feed",
                    "model": "claude-sonnet-4-6",
                    "provider": "anthropic",
                    "reasoning_effort": "low",
                })
                assert resp.status == 200
                kwargs = mock_create.call_args[1]
                assert kwargs["enabled_toolsets"] == ["web", "files"]
                assert kwargs["context_from"] == [OTHER_JOB_ID]
                assert kwargs["monitor_url"] == "https://example.invalid/feed"
                assert kwargs["model"] == "claude-sonnet-4-6"
                assert kwargs["provider"] == "anthropic"
                assert kwargs["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_continuity_becomes_the_self_context_ref(self, adapter):
        """continuity=true is sugar for context_from including "self"."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(f"{_MOD}._cron_get", MagicMock(side_effect=_get_job_stub)):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "continuity": True,
                })
                assert resp.status == 200
                assert mock_create.call_args[1]["context_from"] == ["self"]
                # "self" is never looked up — the job does not exist yet.

    @pytest.mark.asyncio
    async def test_continuity_false_leaves_external_refs(self, adapter):
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(f"{_MOD}._cron_get", MagicMock(side_effect=_get_job_stub)):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "context_from": [OTHER_JOB_ID, "self"],
                    "continuity": False,
                })
                assert resp.status == 200
                assert mock_create.call_args[1]["context_from"] == [OTHER_JOB_ID]

    @pytest.mark.asyncio
    async def test_unknown_context_from_job_is_rejected(self, adapter):
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(f"{_MOD}._cron_get", MagicMock(side_effect=_get_job_stub)):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "context_from": ["ffffffffffff"],
                })
                assert resp.status == 400
                assert "not found" in (await resp.json())["error"]
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_script_uses_the_cli_path_validator(self, adapter):
        """A traversal path is refused by the same helper the CLI runs."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(
                f"{_MOD}._cron_validate_script_path",
                MagicMock(return_value="Script path escapes ~/.hermes/scripts/"),
            ) as validator:
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "monitor_script": "../../etc/passwd",
                })
                assert resp.status == 400
                validator.assert_called_once_with("../../etc/passwd")
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitor_script_fails_closed_without_the_validator(self, adapter):
        """A missing helper rejects the field rather than skipping the check."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ), patch(f"{_MOD}._cron_validate_script_path", None):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "monitor_script": "watch.sh",
                })
                assert resp.status == 400
                mock_create.assert_not_called()

    @pytest.mark.parametrize("body_extra, expected", [
        ({"enabled_toolsets": "web"}, ["web"]),
        ({"enabled_toolsets": []}, None),
        ({"monitor_url": ""}, None),
    ])
    @pytest.mark.asyncio
    async def test_scalar_and_clearing_shapes(self, adapter, body_extra, expected):
        """A bare string is a one-element list; an empty value clears."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        field = next(iter(body_extra))
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    **body_extra,
                })
                assert resp.status == 200
                assert mock_create.call_args[1][field] == expected

    @pytest.mark.parametrize("body_extra", [
        {"continuity": "yes"},
        {"enabled_toolsets": [1, 2]},
        {"enabled_toolsets": 7},
        {"model": 42},
        {"monitor_url": ["a"]},
        {"context_from": {"job": "x"}},
    ])
    @pytest.mark.asyncio
    async def test_type_errors_are_400_not_500(self, adapter, body_extra):
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    **body_extra,
                })
                assert resp.status == 400
                mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_job_value_error_is_400(self, adapter):
        """reasoning_effort grammar and monitor exclusion are client errors."""
        app = _create_app(adapter)
        mock_create = MagicMock(
            side_effect=ValueError("Invalid reasoning_effort 'turbo'."),
        )
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                    "reasoning_effort": "turbo",
                })
                assert resp.status == 400
                assert "reasoning_effort" in (await resp.json())["error"]

    @pytest.mark.asyncio
    async def test_basic_create_is_unchanged(self, adapter):
        """A body with none of the new fields sends none of them."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_create", mock_create
            ):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                })
                assert resp.status == 200
                kwargs = mock_create.call_args[1]
                for field in APIServerAdapter._RICH_JOB_FIELDS:
                    assert field not in kwargs


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdateJobRichFields:
    @pytest.mark.asyncio
    async def test_update_passes_rich_fields_through(self, adapter):
        app = _create_app(adapter)
        mock_update = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ), patch(f"{_MOD}._cron_get", MagicMock(side_effect=_get_job_stub)):
                resp = await cli.patch(f"/api/jobs/{VALID_JOB_ID}", json={
                    "enabled_toolsets": ["web"],
                    "model": "claude-haiku-4-5",
                })
                assert resp.status == 200
                updates = mock_update.call_args[0][1]
                assert updates["enabled_toolsets"] == ["web"]
                assert updates["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_continuity_only_update_keeps_stored_refs(self, adapter):
        """Toggling continuity must not drop an external chaining ref."""
        app = _create_app(adapter)
        stored = dict(SAMPLE_JOB, context_from=[OTHER_JOB_ID])
        mock_update = MagicMock(return_value=stored)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ), patch(f"{_MOD}._cron_get", MagicMock(return_value=stored)):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}", json={"continuity": True},
                )
                assert resp.status == 200
                updates = mock_update.call_args[0][1]
                assert updates["context_from"] == [OTHER_JOB_ID, "self"]
                # `continuity` is sugar, never a stored field.
                assert "continuity" not in updates

    @pytest.mark.asyncio
    async def test_update_rejects_monitor_script_and_url_together(self, adapter):
        app = _create_app(adapter)
        stored = dict(SAMPLE_JOB, monitor_url="https://example.invalid/feed")
        mock_update = MagicMock(return_value=stored)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ), patch(f"{_MOD}._cron_get", MagicMock(return_value=stored)), patch(
                f"{_MOD}._cron_validate_script_path", MagicMock(return_value=None)
            ):
                resp = await cli.patch(f"/api/jobs/{VALID_JOB_ID}", json={
                    "monitor_script": "watch.sh",
                })
                assert resp.status == 400
                assert "mutually exclusive" in (await resp.json())["error"]
                mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_on_missing_job_is_404(self, adapter):
        app = _create_app(adapter)
        mock_update = MagicMock(return_value=None)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ), patch(f"{_MOD}._cron_get", MagicMock(return_value=None)):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}", json={"continuity": True},
                )
                assert resp.status == 404
                mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_still_rejects_unknown_fields(self, adapter):
        """Widening the allowlist must not open it."""
        app = _create_app(adapter)
        mock_update = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ):
                resp = await cli.patch(f"/api/jobs/{VALID_JOB_ID}", json={
                    "id": "deadbeefcafe",
                    "base_url": "https://attacker.invalid/v1",
                    "script": "../../etc/passwd",
                    "workdir": "/",
                    "no_agent": True,
                })
                assert resp.status == 400
                assert (await resp.json())["error"] == "No valid fields to update"
                mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_job_value_error_is_400(self, adapter):
        app = _create_app(adapter)
        mock_update = MagicMock(
            side_effect=ValueError("Cron job field(s) cannot be updated: id"),
        )
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), patch(
                f"{_MOD}._cron_update", mock_update
            ):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}", json={"name": "renamed"},
                )
                assert resp.status == 400
