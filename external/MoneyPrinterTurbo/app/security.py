"""Authentication boundary for the Hermes-managed MoneyPrinter sidecar."""

from __future__ import annotations

import hmac
import os

from starlette.datastructures import Headers
from starlette.responses import JSONResponse


SIDECAR_TOKEN_HEADER = "X-Hermes-MoneyPrinter-Token"


def _protected_path(path: str) -> bool:
    return (
        path.startswith("/api/v1/")
        or path == "/api/v1"
        or path == "/tasks"
        or path.startswith("/tasks/")
    )


class ManagedSidecarAuthMiddleware:
    """Require the per-process token when Hermes launches this service.

    An upstream standalone launch without ``MONEYPRINTER_HERMES_TOKEN`` keeps
    its existing behavior. Hermes-managed launches always set the token and
    therefore fail closed for API and task-file access.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        managed_token = os.getenv("MONEYPRINTER_HERMES_TOKEN", "").strip()
        path = str(scope.get("path") or "")
        if scope.get("type") == "http" and managed_token and _protected_path(path):
            supplied = Headers(scope=scope).get(SIDECAR_TOKEN_HEADER, "")
            if not supplied or not hmac.compare_digest(supplied, managed_token):
                response = JSONResponse(
                    {
                        "status": 401,
                        "data": None,
                        "message": "invalid Hermes sidecar token",
                    },
                    status_code=401,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
