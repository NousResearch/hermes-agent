"""
Tests for the Cron Jobs API endpoints on the API server adapter.

Covers:
- CRUD operations for cron jobs (list, create, get, update, delete)
- Pause / resume / run (trigger) actions
- Input validation (missing name, name too long, prompt too long, invalid repeat)
- Job ID validation (invalid hex)
- Auth enforcement (401 when API_SERVER_KEY is set)
- Cron module unavailability (501 when _CRON_AVAILABLE is False)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_JOB = {
    "id": "aabbccddeeff",
    "name": "test-job",
    "schedule": "*/5 * * * *",
    "prompt": "do something",
    "deliver": "local",
    "enabled": True,
}

VALID_JOB_ID = "aabbccddeeff"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app with jobs routes registered."""
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    # Register only job routes (plus health for sanity)
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/api/jobs", adapter._handle_list_jobs)
    app.router.add_post("/api/jobs", adapter._handle_create_job)
    app.router.add_get("/api/jobs/{job_id}", adapter._handle_get_job)
    app.router.add_patch("/api/jobs/{job_id}", adapter._handle_update_job)
    app.router.add_delete("/api/jobs/{job_id}", adapter._handle_delete_job)
    app.router.add_post("/api/jobs/{job_id}/pause", adapter._handle_pause_job)
    app.router.add_post("/api/jobs/{job_id}/resume", adapter._handle_resume_job)
    app.router.add_post("/api/jobs/{job_id}/run", adapter._handle_run_job)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# 1. test_list_jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_jobs(self, adapter):
        """GET /api/jobs returns job list."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_list", return_value=[SAMPLE_JOB]
            ):
                resp = await cli.get("/api/jobs")
                assert resp.status == 200
                data = await resp.json()
                assert "jobs" in data
                assert data["jobs"] == [SAMPLE_JOB]

    # -------------------------------------------------------------------
    # 2. test_list_jobs_include_disabled
    # -------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_jobs_include_disabled(self, adapter):
        """GET /api/jobs?include_disabled=true passes the flag."""
        app = _create_app(adapter)
        mock_list = MagicMock(return_value=[SAMPLE_JOB])
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_list", mock_list
            ):
                resp = await cli.get("/api/jobs?include_disabled=true")
                assert resp.status == 200
                mock_list.assert_called_once_with(include_disabled=True)

    @pytest.mark.asyncio
    async def test_list_jobs_default_excludes_disabled(self, adapter):
        """GET /api/jobs without flag passes include_disabled=False."""
        app = _create_app(adapter)
        mock_list = MagicMock(return_value=[])
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_list", mock_list
            ):
                resp = await cli.get("/api/jobs")
                assert resp.status == 200
                mock_list.assert_called_once_with(include_disabled=False)


# ---------------------------------------------------------------------------
# 3-7. test_create_job and validation
# ---------------------------------------------------------------------------

class TestCreateJob:
    @pytest.mark.asyncio
    async def test_create_job(self, adapter):
        """POST /api/jobs with valid body returns created job."""
        app = _create_app(adapter)
        mock_create = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_create", mock_create
            ):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == SAMPLE_JOB
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs["name"] == "test-job"
                assert call_kwargs["schedule"] == "*/5 * * * *"
                assert call_kwargs["prompt"] == "do something"

    @pytest.mark.asyncio
    async def test_create_job_missing_name(self, adapter):
        """POST /api/jobs without name returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "schedule": "*/5 * * * *",
                    "prompt": "do something",
                })
                assert resp.status == 400
                data = await resp.json()
                assert "name" in data["error"].lower() or "Name" in data["error"]

    @pytest.mark.asyncio
    async def test_create_job_name_too_long(self, adapter):
        """POST /api/jobs with name > 200 chars returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "name": "x" * 201,
                    "schedule": "*/5 * * * *",
                })
                assert resp.status == 400
                data = await resp.json()
                assert "200" in data["error"] or "Name" in data["error"]

    @pytest.mark.asyncio
    async def test_create_job_prompt_too_long(self, adapter):
        """POST /api/jobs with prompt > 5000 chars returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "prompt": "x" * 5001,
                })
                assert resp.status == 400
                data = await resp.json()
                assert "5000" in data["error"] or "Prompt" in data["error"]

    @pytest.mark.asyncio
    async def test_create_job_invalid_repeat(self, adapter):
        """POST /api/jobs with repeat=0 returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                    "schedule": "*/5 * * * *",
                    "repeat": 0,
                })
                assert resp.status == 400
                data = await resp.json()
                assert "repeat" in data["error"].lower() or "Repeat" in data["error"]

    @pytest.mark.asyncio
    async def test_create_job_missing_schedule(self, adapter):
        """POST /api/jobs without schedule returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "name": "test-job",
                })
                assert resp.status == 400
                data = await resp.json()
                assert "schedule" in data["error"].lower() or "Schedule" in data["error"]


# ---------------------------------------------------------------------------
# 8-10. test_get_job
# ---------------------------------------------------------------------------

class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_job(self, adapter):
        """GET /api/jobs/{id} returns job."""
        app = _create_app(adapter)
        mock_get = MagicMock(return_value=SAMPLE_JOB)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_get", mock_get
            ):
                resp = await cli.get(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == SAMPLE_JOB
                mock_get.assert_called_once_with(VALID_JOB_ID)

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, adapter):
        """GET /api/jobs/{id} returns 404 when job doesn't exist."""
        app = _create_app(adapter)
        mock_get = MagicMock(return_value=None)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_get", mock_get
            ):
                resp = await cli.get(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_job_invalid_id(self, adapter):
        """GET /api/jobs/{id} with non-hex id returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.get("/api/jobs/not-a-valid-hex!")
                assert resp.status == 400
                data = await resp.json()
                assert "Invalid" in data["error"]


# ---------------------------------------------------------------------------
# 11-12. test_update_job
# ---------------------------------------------------------------------------

class TestUpdateJob:
    @pytest.mark.asyncio
    async def test_update_job(self, adapter):
        """PATCH /api/jobs/{id} updates with whitelisted fields."""
        app = _create_app(adapter)
        updated_job = {**SAMPLE_JOB, "name": "updated-name"}
        mock_update = MagicMock(return_value=updated_job)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_update", mock_update
            ):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}",
                    json={"name": "updated-name", "schedule": "0 * * * *"},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == updated_job
                mock_update.assert_called_once()
                call_args = mock_update.call_args
                assert call_args[0][0] == VALID_JOB_ID
                sanitized = call_args[0][1]
                assert "name" in sanitized
                assert "schedule" in sanitized

    @pytest.mark.asyncio
    async def test_update_job_rejects_unknown_fields(self, adapter):
        """PATCH /api/jobs/{id} — only allowed fields pass through."""
        app = _create_app(adapter)
        updated_job = {**SAMPLE_JOB, "name": "new-name"}
        mock_update = MagicMock(return_value=updated_job)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_update", mock_update
            ):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}",
                    json={
                        "name": "new-name",
                        "evil_field": "malicious",
                        "__proto__": "hack",
                    },
                )
                assert resp.status == 200
                call_args = mock_update.call_args
                sanitized = call_args[0][1]
                assert "name" in sanitized
                assert "evil_field" not in sanitized
                assert "__proto__" not in sanitized

    @pytest.mark.asyncio
    async def test_update_job_no_valid_fields(self, adapter):
        """PATCH /api/jobs/{id} with only unknown fields returns 400."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.patch(
                    f"/api/jobs/{VALID_JOB_ID}",
                    json={"evil_field": "malicious"},
                )
                assert resp.status == 400
                data = await resp.json()
                assert "No valid fields" in data["error"]


# ---------------------------------------------------------------------------
# 13. test_delete_job
# ---------------------------------------------------------------------------

class TestDeleteJob:
    @pytest.mark.asyncio
    async def test_delete_job(self, adapter):
        """DELETE /api/jobs/{id} returns ok."""
        app = _create_app(adapter)
        mock_remove = MagicMock(return_value=True)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_remove", mock_remove
            ):
                resp = await cli.delete(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True
                mock_remove.assert_called_once_with(VALID_JOB_ID)

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, adapter):
        """DELETE /api/jobs/{id} returns 404 when job doesn't exist."""
        app = _create_app(adapter)
        mock_remove = MagicMock(return_value=False)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_remove", mock_remove
            ):
                resp = await cli.delete(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 404


# ---------------------------------------------------------------------------
# 14. test_pause_job
# ---------------------------------------------------------------------------

class TestPauseJob:
    @pytest.mark.asyncio
    async def test_pause_job(self, adapter):
        """POST /api/jobs/{id}/pause returns updated job."""
        app = _create_app(adapter)
        paused_job = {**SAMPLE_JOB, "enabled": False}
        mock_pause = MagicMock(return_value=paused_job)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_pause", mock_pause
            ):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/pause")
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == paused_job
                assert data["job"]["enabled"] is False
                mock_pause.assert_called_once_with(VALID_JOB_ID)


# ---------------------------------------------------------------------------
# 15. test_resume_job
# ---------------------------------------------------------------------------

class TestResumeJob:
    @pytest.mark.asyncio
    async def test_resume_job(self, adapter):
        """POST /api/jobs/{id}/resume returns updated job."""
        app = _create_app(adapter)
        resumed_job = {**SAMPLE_JOB, "enabled": True}
        mock_resume = MagicMock(return_value=resumed_job)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_resume", mock_resume
            ):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/resume")
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == resumed_job
                assert data["job"]["enabled"] is True
                mock_resume.assert_called_once_with(VALID_JOB_ID)


# ---------------------------------------------------------------------------
# 16. test_run_job
# ---------------------------------------------------------------------------

class TestRunJob:
    @pytest.mark.asyncio
    async def test_run_job(self, adapter):
        """POST /api/jobs/{id}/run returns triggered job."""
        app = _create_app(adapter)
        triggered_job = {**SAMPLE_JOB, "last_run": "2025-01-01T00:00:00Z"}
        mock_trigger = MagicMock(return_value=triggered_job)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_trigger", mock_trigger
            ):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/run")
                assert resp.status == 200
                data = await resp.json()
                assert data["job"] == triggered_job
                mock_trigger.assert_called_once_with(VALID_JOB_ID)


# ---------------------------------------------------------------------------
# 17. test_auth_required
# ---------------------------------------------------------------------------

class TestAuthRequired:
    @pytest.mark.asyncio
    async def test_auth_required_list_jobs(self, auth_adapter):
        """GET /api/jobs without API key returns 401 when key is set."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.get("/api/jobs")
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_required_create_job(self, auth_adapter):
        """POST /api/jobs without API key returns 401 when key is set."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.post("/api/jobs", json={
                    "name": "test", "schedule": "* * * * *",
                })
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_required_get_job(self, auth_adapter):
        """GET /api/jobs/{id} without API key returns 401 when key is set."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.get(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_required_delete_job(self, auth_adapter):
        """DELETE /api/jobs/{id} without API key returns 401."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", True):
                resp = await cli.delete(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_passes_with_valid_key(self, auth_adapter):
        """GET /api/jobs with correct API key succeeds."""
        app = _create_app(auth_adapter)
        mock_list = MagicMock(return_value=[])
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                APIServerAdapter, "_CRON_AVAILABLE", True
            ), patch.object(
                APIServerAdapter, "_cron_list", mock_list
            ):
                resp = await cli.get(
                    "/api/jobs",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 200


# ---------------------------------------------------------------------------
# 18. test_cron_unavailable
# ---------------------------------------------------------------------------

class TestCronUnavailable:
    @pytest.mark.asyncio
    async def test_cron_unavailable_list(self, adapter):
        """GET /api/jobs returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.get("/api/jobs")
                assert resp.status == 501
                data = await resp.json()
                assert "not available" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_cron_unavailable_create(self, adapter):
        """POST /api/jobs returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.post("/api/jobs", json={
                    "name": "test", "schedule": "* * * * *",
                })
                assert resp.status == 501

    @pytest.mark.asyncio
    async def test_cron_unavailable_get(self, adapter):
        """GET /api/jobs/{id} returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.get(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 501

    @pytest.mark.asyncio
    async def test_cron_unavailable_delete(self, adapter):
        """DELETE /api/jobs/{id} returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.delete(f"/api/jobs/{VALID_JOB_ID}")
                assert resp.status == 501

    @pytest.mark.asyncio
    async def test_cron_unavailable_pause(self, adapter):
        """POST /api/jobs/{id}/pause returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/pause")
                assert resp.status == 501

    @pytest.mark.asyncio
    async def test_cron_unavailable_resume(self, adapter):
        """POST /api/jobs/{id}/resume returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/resume")
                assert resp.status == 501

    @pytest.mark.asyncio
    async def test_cron_unavailable_run(self, adapter):
        """POST /api/jobs/{id}/run returns 501 when _CRON_AVAILABLE is False."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(APIServerAdapter, "_CRON_AVAILABLE", False):
                resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/run")
                assert resp.status == 501
