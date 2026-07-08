"""Production security tests for the API server adapter."""

import hashlib
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/models", adapter._handle_models)
    return app


def _request(*, method: str = "GET", headers: dict | None = None, cookies: dict | None = None):
    request = MagicMock()
    request.method = method
    request.headers = headers or {}
    request.cookies = cookies or {}
    return request


class TestBearerTokenRevocation:
    def test_revoked_token_sha256_rejects_otherwise_valid_api_key(self):
        token = "sk-valid-but-revoked"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": token,
                    "revoked_token_sha256": [digest],
                },
            )
        )

        result = adapter._check_auth(_request(headers={"Authorization": f"Bearer {token}"}))

        assert result is not None
        assert result.status == 401

    def test_revoked_token_sha256_accepts_comma_separated_sha256_prefixes(self):
        token = "sk-valid-but-revoked"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": token,
                    "revoked_token_sha256": f"sha256:{digest}, sha256:{'0' * 64}",
                },
            )
        )

        result = adapter._check_auth(_request(headers={"Authorization": f"Bearer {token}"}))

        assert result is not None
        assert result.status == 401

    def test_unrevoked_valid_bearer_key_still_passes(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-valid",
                    "revoked_token_sha256": ["0" * 64],
                },
            )
        )

        assert adapter._check_auth(_request(headers={"Authorization": "Bearer sk-valid"})) is None


class TestCookieCsrfProtection:
    def test_safe_cookie_authenticated_request_does_not_require_csrf(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-cookie",
                    "cookie_name": "hermes_api_token",
                },
            )
        )

        result = adapter._check_auth(_request(cookies={"hermes_api_token": "sk-cookie"}))

        assert result is None

    def test_mutating_cookie_authenticated_request_requires_csrf(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-cookie",
                    "cookie_name": "hermes_api_token",
                },
            )
        )

        result = adapter._check_auth(
            _request(method="POST", cookies={"hermes_api_token": "sk-cookie"})
        )

        assert result is not None
        assert result.status == 403

    def test_mutating_cookie_authenticated_request_accepts_double_submit_csrf(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-cookie",
                    "cookie_name": "hermes_api_token",
                    "csrf_header": "X-Hermes-CSRF",
                    "csrf_cookie": "hermes_csrf",
                },
            )
        )

        result = adapter._check_auth(
            _request(
                method="DELETE",
                headers={"X-Hermes-CSRF": "csrf-token"},
                cookies={"hermes_api_token": "sk-cookie", "hermes_csrf": "csrf-token"},
            )
        )

        assert result is None

    def test_bearer_authenticated_mutating_request_does_not_require_csrf(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-bearer",
                    "cookie_name": "hermes_api_token",
                },
            )
        )

        result = adapter._check_auth(
            _request(method="POST", headers={"Authorization": "Bearer sk-bearer"})
        )

        assert result is None


class TestCapabilitiesSecuritySurface:
    @pytest.mark.asyncio
    async def test_capabilities_reports_security_configuration_without_secrets(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "key": "sk-secret-value",
                    "cookie_name": "hermes_api_token",
                    "revoked_token_sha256": ["0" * 64],
                    "max_bearer_token_age_seconds": 900,
                },
            )
        )
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities", headers={"Authorization": "Bearer sk-secret-value"})
            assert resp.status == 200
            data = await resp.json()

        assert data["auth"] == {"type": "bearer", "required": True}
        assert data["security"] == {
            "bearer_auth_required": True,
            "token_revocation_supported": True,
            "revoked_token_count": 1,
            "cookie_auth_configured": True,
            "csrf_required_for_cookie_mutations": True,
            "recommended_max_bearer_token_age_seconds": 900,
        }
        assert "sk-secret-value" not in str(data)
