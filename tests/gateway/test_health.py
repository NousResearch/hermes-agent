"""
tests/gateway/test_health.py — Unit tests for gateway/health.py

Phase E acceptance criteria tested:
  AC-1: GET /health returns {"status": "ok"} when all deps reachable
  AC-2: GET /health returns {"status": "degraded", "storage": "error"} when Neon unreachable
  AC-3: Non-SaaS mode returns {"status": "ok", "storage": "sqlite", "skills": "local"}
  AC-4: S3 unreachable → {"status": "degraded", "skills": "error"}
  AC-5: health_check does not raise — always returns a dict

All external dependencies (asyncpg, boto3, Secrets Manager) are mocked.
These tests run fully offline — no network, no AWS credentials required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run a coroutine in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Non-SaaS mode
# ---------------------------------------------------------------------------


class TestHealthNonSaas:
    """In local mode, health_check should always return ok without hitting cloud."""

    def test_local_mode_returns_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERMES_MODE", "local")
        from gateway.health import health_check

        result = _run(health_check())
        assert result["status"] == "ok"
        assert result["storage"] == "sqlite"
        assert result["skills"] == "local"

    def test_unset_mode_returns_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HERMES_MODE", raising=False)
        # Reload to pick up env change
        from gateway.health import health_check

        result = _run(health_check())
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# SaaS mode — Neon checks
# ---------------------------------------------------------------------------


class TestHealthNeonCheck:
    """Tests for the Neon PostgreSQL health check in SaaS mode."""

    def _patch_s3_ok(self) -> Any:
        """Patch S3 check to always succeed so we can isolate Neon."""
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}
        return patch("boto3.client", return_value=mock_s3)

    def test_neon_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_conn.close = AsyncMock()

        with (
            patch(
                "hermes_storage.neon_backend._resolve_dsn",
                return_value="postgresql://fake/db",
            ),
            patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn)),
            self._patch_s3_ok(),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "ok"
        assert result["storage"] == "neon"

    def test_neon_unreachable_reports_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-2: Neon unreachable → status=degraded, storage=error."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        with (
            patch(
                "hermes_storage.neon_backend._resolve_dsn",
                return_value="postgresql://fake/db",
            ),
            patch(
                "asyncpg.connect",
                new=AsyncMock(
                    side_effect=ConnectionRefusedError("Connection refused")
                ),
            ),
            self._patch_s3_ok(),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "degraded"
        assert result["storage"] == "error"
        assert "storage_error" in result.get("details", result)

    def test_neon_timeout_reports_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neon connection timeout → status=degraded."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        async def _slow_connect(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(100)  # Never resolves within check timeout

        with (
            patch(
                "hermes_storage.neon_backend._resolve_dsn",
                return_value="postgresql://fake/db",
            ),
            patch("asyncpg.connect", new=_slow_connect),
            patch(
                "gateway.health._CHECK_TIMEOUT",
                new=0.01,  # 10ms timeout for the test
            ),
            self._patch_s3_ok(),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "degraded"
        assert result["storage"] == "error"


# ---------------------------------------------------------------------------
# SaaS mode — S3 checks
# ---------------------------------------------------------------------------


class TestHealthS3Check:
    """Tests for the S3 skills bucket health check in SaaS mode."""

    def _patch_neon_ok(self) -> Any:
        """Patch Neon to always succeed so we can isolate S3."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_conn.close = AsyncMock()
        return patch("asyncpg.connect", new=AsyncMock(return_value=mock_conn))

    def _patch_dsn(self) -> Any:
        return patch(
            "hermes_storage.neon_backend._resolve_dsn",
            return_value="postgresql://fake/db",
        )

    def test_s3_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("S3_SKILLS_BUCKET", "hermes-saas-skills")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}

        with (
            self._patch_dsn(),
            self._patch_neon_ok(),
            patch("boto3.client", return_value=mock_s3),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "ok"
        assert result["skills"] == "s3"

    def test_s3_unreachable_reports_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-4: S3 unreachable → status=degraded, skills=error."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("S3_SKILLS_BUCKET", "hermes-saas-skills")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "NoSuchBucket"}},
            "HeadBucket",
        )

        with (
            self._patch_dsn(),
            self._patch_neon_ok(),
            patch("boto3.client", return_value=mock_s3),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "degraded"
        assert result["skills"] == "error"

    def test_s3_credentials_error_reports_degraded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NoCredentialsError → degraded (not a crash)."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("S3_SKILLS_BUCKET", "hermes-saas-skills")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        from botocore.exceptions import NoCredentialsError

        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = NoCredentialsError()

        with (
            self._patch_dsn(),
            self._patch_neon_ok(),
            patch("boto3.client", return_value=mock_s3),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "degraded"
        assert result["skills"] == "error"


# ---------------------------------------------------------------------------
# Both checks fail
# ---------------------------------------------------------------------------


class TestHealthBothFail:
    def test_both_fail_reports_degraded_with_both_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both Neon and S3 fail, report degraded with both errors."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        from botocore.exceptions import NoCredentialsError

        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = NoCredentialsError()

        with (
            patch(
                "hermes_storage.neon_backend._resolve_dsn",
                return_value="postgresql://fake/db",
            ),
            patch(
                "asyncpg.connect",
                new=AsyncMock(side_effect=OSError("unreachable")),
            ),
            patch("boto3.client", return_value=mock_s3),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert result["status"] == "degraded"
        assert result["storage"] == "error"
        assert result["skills"] == "error"
        # Both errors captured in details
        details = result.get("details", {})
        assert "storage_error" in details
        assert "skills_error" in details

    def test_health_check_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC-5: health_check must never raise — always returns a dict."""
        monkeypatch.setenv("HERMES_MODE", "saas")
        monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://fake/db")

        # Simulate a completely broken environment.
        with (
            patch(
                "hermes_storage.neon_backend._resolve_dsn",
                side_effect=RuntimeError("Secrets Manager down"),
            ),
            patch(
                "boto3.client",
                side_effect=Exception("boto3 init failed"),
            ),
        ):
            from gateway.health import health_check

            result = _run(health_check())

        assert isinstance(result, dict)
        assert "status" in result
