"""FastAPI app for the Telegram Mini App M2 read-only sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import secrets
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hermes_cli.config import get_env_value

from .auth import InitDataAuthError, TelegramMiniAppUser, VerifiedInitData, verify_init_data
from .status import build_status_snapshot


SESSION_COOKIE = "hermes_tma_session"
_ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _default_allowed_origins() -> set[str]:
    return {"http://127.0.0.1:5175", "http://localhost:5175"}


@dataclass
class MiniAppSettings:
    bot_token: str | None = None
    allowed_users: set[str] = field(default_factory=set)
    host: str = "127.0.0.1"
    port: int = 9120
    auth_ttl_seconds: int = 300
    future_skew_seconds: int = 60
    session_ttl_seconds: int = 3600
    cors_allowed_origins: set[str] = field(default_factory=_default_allowed_origins)
    now: Callable[[], int | float] = time.time

    def resolved_bot_token(self) -> str:
        return self.bot_token or get_env_value("TELEGRAM_BOT_TOKEN") or ""


@dataclass
class MiniAppSession:
    session_id: str
    user: TelegramMiniAppUser
    expires_at: datetime
    init_data_fingerprint: str


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, MiniAppSession] = {}

    def create(self, verified: VerifiedInitData, *, ttl_seconds: int, now: int | float) -> MiniAppSession:
        session = MiniAppSession(
            session_id=secrets.token_urlsafe(32),
            user=verified.user,
            expires_at=datetime.fromtimestamp(now, tz=timezone.utc) + timedelta(seconds=ttl_seconds),
            init_data_fingerprint=verified.fingerprint,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str | None, *, now: int | float) -> MiniAppSession | None:
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.expires_at <= datetime.fromtimestamp(now, tz=timezone.utc):
            self._sessions.pop(session_id, None)
            return None
        return session

    def delete(self, session_id: str | None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)


class TelegramAuthRequest(BaseModel):
    initData: str = ""


def _user_payload(user: TelegramMiniAppUser) -> dict[str, str]:
    payload = {"id": user.id}
    if user.username:
        payload["username"] = user.username
    if user.first_name:
        payload["first_name"] = user.first_name
    if user.last_name:
        payload["last_name"] = user.last_name
    return payload


def _host_only(host_header: str) -> str:
    value = (host_header or "").strip()
    if value.startswith("["):
        close = value.find("]")
        return value[1:close].lower() if close != -1 else value.strip("[]").lower()
    return value.rsplit(":", 1)[0].lower() if ":" in value else value.lower()


def _is_loopback_host(host: str) -> bool:
    return host.lower() in _ALLOWED_LOOPBACK_HOSTS


def _cors_headers(origin: str, settings: MiniAppSettings) -> dict[str, str]:
    if origin in settings.cors_allowed_origins:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        }
    return {}


def _safe_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def create_app(
    *,
    settings: MiniAppSettings | None = None,
    status_provider: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    settings = settings or MiniAppSettings()
    sessions = SessionStore()
    status_provider = status_provider or (lambda: build_status_snapshot(hermes_home_configured=True))
    app = FastAPI(title="Hermes Telegram Mini App", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def _host_origin_cors_guard(request: Request, call_next):
        origin = request.headers.get("origin", "")
        if not _is_loopback_host(settings.host):
            return _safe_error(503, "Mini App sidecar is local-only in M2")

        host_header = request.headers.get("host", "")
        host = _host_only(host_header)
        if not host or not _is_loopback_host(host):
            return _safe_error(400, "Invalid Host header")

        if request.method == "OPTIONS":
            headers = _cors_headers(origin, settings)
            if not headers:
                return Response(status_code=204)
            headers.update({
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "content-type",
            })
            return Response(status_code=204, headers=headers)

        if origin and origin not in settings.cors_allowed_origins:
            return _safe_error(403, "Origin is not allowed")

        response = await call_next(request)
        if origin:
            for key, value in _cors_headers(origin, settings).items():
                response.headers[key] = value
        return response

    def current_session(request: Request) -> MiniAppSession:
        session = sessions.get(request.cookies.get(SESSION_COOKIE), now=settings.now())
        if session is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return session

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "telegram-miniapp", "version": "m2"}

    @app.get("/readyz")
    def readyz():
        if not _is_loopback_host(settings.host):
            raise HTTPException(status_code=503, detail="Mini App sidecar is local-only in M2")
        return {"ok": True, "service": "telegram-miniapp"}

    @app.post("/api/auth/telegram")
    def auth_telegram(payload: TelegramAuthRequest, response: Response):
        try:
            verified = verify_init_data(
                payload.initData,
                bot_token=settings.resolved_bot_token(),
                allowed_users=settings.allowed_users,
                now=settings.now(),
                ttl_seconds=settings.auth_ttl_seconds,
                future_skew_seconds=settings.future_skew_seconds,
            )
        except InitDataAuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        session = sessions.create(verified, ttl_seconds=settings.session_ttl_seconds, now=settings.now())
        response.set_cookie(
            SESSION_COOKIE,
            session.session_id,
            max_age=settings.session_ttl_seconds,
            expires=settings.session_ttl_seconds,
            httponly=True,
            secure=False,
            samesite="lax",
            path="/api",
        )
        return {"ok": True, "user": _user_payload(session.user), "expires_at": session.expires_at.isoformat()}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        current_session(request)
        sessions.delete(request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE, path="/api")
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request):
        session = current_session(request)
        return {
            "authenticated": True,
            "user": _user_payload(session.user),
            "session_expires_at": session.expires_at.isoformat(),
        }

    @app.get("/api/status")
    def status(request: Request):
        current_session(request)
        return status_provider()

    return app
