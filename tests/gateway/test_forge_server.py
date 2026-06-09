"""
tests/gateway/test_forge_server.py — Plan 048-E

Unit tests for gateway/forge_server.py.

Tests mock the NeonBackend pool, hermes_storage.get_backend(), and the 004-A
modules so no live Neon or S3 is required. Pattern follows
tests/self_improvement/test_feedback_capture.py (AsyncMock pool, _FakeRecord).

Test inventory:
  1.  test_auth_required_when_token_configured       — 401 when bearer absent
  2.  test_auth_passes_with_correct_bearer           — request proceeds past auth
  3.  test_auth_open_when_no_token_configured        — no bearer = open (local dev)
  4.  test_candidates_returns_rows                   — GET /forge/candidates returns list
  5.  test_candidates_with_since                     — since param forwarded to SQL
  6.  test_candidates_pool_unavailable_503           — non-Neon backend → 503
  7.  test_draft_missing_fields_400                  — missing tenant_id → 400
  8.  test_draft_no_pending_recommendation_404       — no pending rec → 404
  9.  test_draft_calls_generate_skill_draft          — happy path calls recommender
  10. test_score_uses_thumbs_rate_from_db            — skill_scores row → score from DB
  11. test_score_heuristic_pass                      — good draft → 0.95 pass
  12. test_score_heuristic_fail                      — bad draft → 0.40 fail
  13. test_promote_missing_fields_400                — missing skill_name → 400
  14. test_promote_calls_promote_skill_to_team       — happy path delegates to skills_scoped
  15. test_promote_internal_error_500                — promote raises → 500
  16. test_register_forge_routes                     — routes added to aiohttp app
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRecord(dict):
    """Dict subclass supporting asyncpg Record-style access."""
    def __getitem__(self, key):
        return super().__getitem__(key)


def _make_pool(rows=None):
    """Return a mock asyncpg pool that returns `rows` from conn.fetch / conn.fetchrow."""
    rows = rows or []
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.fetchrow = AsyncMock(return_value=rows[0] if rows else None)

    # Context manager for pool.acquire()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool, conn


def _make_neon_backend(pool):
    """Return a mock NeonBackend-like object wrapping the given pool."""
    backend = MagicMock()
    backend._require_pool = MagicMock(return_value=pool)
    # make isinstance(backend, NeonBackend) checks pass via spec if possible
    backend.__class__.__name__ = "NeonBackend"
    return backend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def forge_module():
    """Import gateway.forge_server fresh each test (avoids cached state)."""
    import importlib
    import gateway.forge_server as m
    importlib.reload(m)
    return m


@pytest.fixture()
def clear_forge_bearer(monkeypatch):
    """Ensure HERMES_FORGE_BEARER_TOKEN is unset (open mode) by default."""
    monkeypatch.delenv("HERMES_FORGE_BEARER_TOKEN", raising=False)


@pytest.fixture()
def set_forge_bearer(monkeypatch):
    monkeypatch.setenv("HERMES_FORGE_BEARER_TOKEN", "test-secret-token")


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestForgeAuth:
    def test_auth_required_when_token_configured(self, set_forge_bearer):
        """401 when Authorization header is absent and token is configured."""
        from gateway.forge_server import _check_forge_auth
        from aiohttp.web import Request
        from unittest.mock import MagicMock

        req = MagicMock(spec=Request)
        req.headers = {}
        result = _check_forge_auth(req)
        assert result is not None
        assert result.status == 401

    def test_auth_passes_with_correct_bearer(self, set_forge_bearer):
        """None (pass) when Authorization: Bearer <correct-token>."""
        from gateway.forge_server import _check_forge_auth
        from unittest.mock import MagicMock

        req = MagicMock()
        req.headers = {"Authorization": "Bearer test-secret-token"}
        result = _check_forge_auth(req)
        assert result is None

    def test_auth_open_when_no_token_configured(self, clear_forge_bearer):
        """None (pass) for any request when no token is configured."""
        from gateway.forge_server import _check_forge_auth
        from unittest.mock import MagicMock

        req = MagicMock()
        req.headers = {}
        result = _check_forge_auth(req)
        assert result is None

    def test_auth_rejects_wrong_bearer(self, set_forge_bearer):
        """401 when Authorization header has a wrong token."""
        from gateway.forge_server import _check_forge_auth
        from unittest.mock import MagicMock

        req = MagicMock()
        req.headers = {"Authorization": "Bearer wrong-token"}
        result = _check_forge_auth(req)
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# GET /forge/candidates
# ---------------------------------------------------------------------------


class TestForgeCandidates:
    @pytest.mark.asyncio
    async def test_candidates_returns_rows(self, clear_forge_bearer):
        """GET /forge/candidates returns list of candidate dicts."""
        rows = [
            _FakeRecord(
                tenant_id="t1",
                skill_name="my-skill",
                slack_ts="123.456",
                conversation_id="conv-1",
            )
        ]
        pool, _ = _make_pool(rows)
        backend = _make_neon_backend(pool)

        from unittest.mock import AsyncMock, MagicMock

        request = MagicMock()
        request.headers = {}
        request.rel_url.query = {}

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.get_backend", new=AsyncMock(return_value=backend)):
            from gateway.forge_server import handle_forge_candidates
            response = await handle_forge_candidates(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["skill_name"] == "my-skill"
        assert data[0]["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_candidates_pool_unavailable_503(self, clear_forge_bearer):
        """503 when the pool raises (not in saas mode)."""
        request = MagicMock()
        request.headers = {}
        request.rel_url.query = {}

        with patch(
            "gateway.forge_server._get_pool",
            side_effect=RuntimeError("NeonBackend required"),
        ):
            from gateway.forge_server import handle_forge_candidates
            response = await handle_forge_candidates(request)

        assert response.status == 503


# ---------------------------------------------------------------------------
# POST /forge/draft
# ---------------------------------------------------------------------------


class TestForgeDraft:
    @pytest.mark.asyncio
    async def test_draft_missing_fields_400(self, clear_forge_bearer):
        """400 when tenant_id is missing from the request body."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={"skill_name": "my-skill"})

        from gateway.forge_server import handle_forge_draft
        response = await handle_forge_draft(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_draft_no_pending_recommendation_404(self, clear_forge_bearer):
        """404 when no pending recommendation exists for the skill."""
        pool, conn = _make_pool([])  # fetchrow returns None (empty rows)
        conn.fetchrow = AsyncMock(return_value=None)

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={"tenant_id": "t1", "skill_name": "unknown-skill"}
        )

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.neon_backend._RLSTransaction", new=MagicMock(
                 return_value=_async_cm()
             )):
            from gateway.forge_server import handle_forge_draft
            response = await handle_forge_draft(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_draft_calls_generate_skill_draft(self, clear_forge_bearer):
        """Happy path: calls generate_skill_draft and returns content."""
        rec_row = _FakeRecord(id="rec-uuid-123")

        # Two fetchrow calls: recommendation lookup + content re-read.
        content_row = _FakeRecord(
            generated_skill_content="# My Skill\n\n## Description\nDoes things.",
            llm_cost_usd=0.0125,
        )
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(side_effect=[rec_row, content_row])

        generate_result = {
            "status": "generated",
            "skill_name": "my-skill",
            "draft_path": "/tmp/my-skill.md",
            "estimated_cost_usd": 0.0125,
        }

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={"tenant_id": "t1", "skill_name": "my-skill"}
        )

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.neon_backend._RLSTransaction", new=MagicMock(
                 return_value=_async_cm()
             )), \
             patch(
                 "hermes_agent.self_improvement.recommender.generate_skill_draft",
                 new=AsyncMock(return_value=generate_result),
             ):
            from gateway.forge_server import handle_forge_draft
            response = await handle_forge_draft(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert data["skill_name"] == "my-skill"
        assert data["content"] == "# My Skill\n\n## Description\nDoes things."
        assert data["cost_usd"] == pytest.approx(0.0125)


# ---------------------------------------------------------------------------
# POST /forge/score
# ---------------------------------------------------------------------------


class TestForgeScore:
    @pytest.mark.asyncio
    async def test_score_uses_thumbs_rate_from_db(self, clear_forge_bearer):
        """Returns thumbs_rate_30d from skill_scores when available."""
        score_row = _FakeRecord(thumbs_rate_30d=0.85)
        pool, conn = _make_pool([score_row])
        conn.fetchrow = AsyncMock(return_value=score_row)

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={
                "skill_name": "my-skill",
                "tenant_id": "t1",
                "content": "# My Skill\n\n## Description\nDoes things. " * 10,
            }
        )

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.neon_backend._RLSTransaction", new=MagicMock(
                 return_value=_async_cm()
             )):
            from gateway.forge_server import handle_forge_score
            response = await handle_forge_score(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert data["value"] == pytest.approx(0.85)
        assert data["passed"] is False  # 0.85 < 0.92 bar

    @pytest.mark.asyncio
    async def test_score_heuristic_pass(self, clear_forge_bearer):
        """Heuristic pass: good content → 0.95 / passed=True."""
        pool, conn = _make_pool([])
        conn.fetchrow = AsyncMock(return_value=None)

        content = "# My Skill\n\n## Description\n" + "Does things well. " * 20

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={"skill_name": "new-skill", "tenant_id": "t1", "content": content}
        )

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.neon_backend._RLSTransaction", new=MagicMock(
                 return_value=_async_cm()
             )):
            from gateway.forge_server import handle_forge_score
            response = await handle_forge_score(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert data["passed"] is True
        assert data["value"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_score_heuristic_fail(self, clear_forge_bearer):
        """Heuristic fail: minimal content → 0.40 / passed=False."""
        pool, conn = _make_pool([])
        conn.fetchrow = AsyncMock(return_value=None)

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={"skill_name": "new-skill", "tenant_id": "t1", "content": "short"}
        )

        with patch("gateway.forge_server._get_pool", new=AsyncMock(return_value=pool)), \
             patch("hermes_storage.neon_backend._RLSTransaction", new=MagicMock(
                 return_value=_async_cm()
             )):
            from gateway.forge_server import handle_forge_score
            response = await handle_forge_score(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert data["passed"] is False
        assert data["value"] == pytest.approx(0.40)
        assert "too short" in data["diagnostic"]


# ---------------------------------------------------------------------------
# POST /forge/promote
# ---------------------------------------------------------------------------


class TestForgePromote:
    @pytest.mark.asyncio
    async def test_promote_missing_fields_400(self, clear_forge_bearer):
        """400 when skill_name is absent."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(return_value={"tenant_id": "t1"})

        from gateway.forge_server import handle_forge_promote
        response = await handle_forge_promote(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_promote_calls_promote_skill_to_team(self, clear_forge_bearer):
        """Happy path: calls promote_skill_to_team and returns promoted=True."""
        promote_result = {"promoted": True, "notes": "copied personal→team"}

        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={
                "skill_name": "my-skill",
                "tenant_id": "t1",
                "platform": "slack",
                "team_id": "T0ABC123",
            }
        )

        with patch("gateway.forge_server._run_promote", new=AsyncMock(return_value=promote_result)):
            from gateway.forge_server import handle_forge_promote
            response = await handle_forge_promote(request)

        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert data["promoted"] is True
        assert data["skill_name"] == "my-skill"
        assert data["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_promote_internal_error_500(self, clear_forge_bearer):
        """500 when promote raises an unexpected exception."""
        request = MagicMock()
        request.headers = {}
        request.json = AsyncMock(
            return_value={"skill_name": "my-skill", "tenant_id": "t1"}
        )

        with patch(
            "gateway.forge_server._run_promote",
            side_effect=RuntimeError("S3 error"),
        ):
            from gateway.forge_server import handle_forge_promote
            response = await handle_forge_promote(request)

        assert response.status == 500


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestRegisterForgeRoutes:
    def test_register_forge_routes(self):
        """register_forge_routes() adds 4 routes to an aiohttp Application."""
        import aiohttp.web as web
        from gateway.forge_server import register_forge_routes

        app = web.Application()
        register_forge_routes(app)

        # Collect registered resource paths.
        paths = {str(r.get_info().get("path", "")) for r in app.router.resources()}
        assert "/forge/candidates" in paths
        assert "/forge/draft" in paths
        assert "/forge/score" in paths
        assert "/forge/promote" in paths


# ---------------------------------------------------------------------------
# Async context manager helper for _RLSTransaction mock
# ---------------------------------------------------------------------------


def _async_cm():
    """Return an async context manager that does nothing."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm
