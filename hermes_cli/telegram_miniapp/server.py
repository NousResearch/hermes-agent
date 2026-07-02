"""FastAPI app for the Telegram Mini App read-only sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import secrets
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hermes_cli.config import get_env_value

from .auth import InitDataAuthError, TelegramMiniAppUser, VerifiedInitData, verify_init_data
from .approvals import build_approvals_snapshot
from .capabilities import build_capabilities_snapshot
from .previews import build_logs_snapshot, build_sessions_snapshot
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
    public_smoke: bool = False
    public_base_url: str | None = None
    enable_actions: bool = False
    auth_rate_limit_per_minute: int = 10
    auth_global_limit: int = 50
    status_rate_limit_per_minute: int = 60
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


class RateLimiter:
    def __init__(self) -> None:
        # FastAPI runs sync handlers in a threadpool, so the shared counters
        # must be mutated under a lock.
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, int], int] = {}
        self._absolute_counts: dict[str, int] = {}

    def check(self, key: str, *, limit: int, now: int | float) -> bool:
        if limit <= 0:
            return False
        bucket = int(now // 60)
        with self._lock:
            # Only the current minute bucket is ever read; evict older buckets
            # so a long-lived loopback sidecar does not accumulate stale
            # counters.
            stale = [entry for entry in self._counts if entry[1] < bucket]
            for entry in stale:
                del self._counts[entry]
            counter_key = (key, bucket)
            count = self._counts.get(counter_key, 0) + 1
            self._counts[counter_key] = count
        return count <= limit

    def check_absolute(self, key: str, *, limit: int) -> bool:
        if limit <= 0:
            return False
        with self._lock:
            count = self._absolute_counts.get(key, 0) + 1
            self._absolute_counts[key] = count
        return count <= limit


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


def _host_authority(host_header: str) -> str:
    value = (host_header or "").strip().lower()
    if value.startswith("["):
        close = value.find("]")
        if close == -1:
            return value.strip("[]")
        host = value[1:close]
        suffix = value[close + 1:]
        return f"[{host}]{suffix}" if suffix else host
    return value


def _is_loopback_host(host: str) -> bool:
    return host.lower() in _ALLOWED_LOOPBACK_HOSTS


def _public_origin(settings: MiniAppSettings) -> str | None:
    if not settings.public_base_url:
        return None
    parsed = urlparse(settings.public_base_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    if parsed.query or parsed.fragment or parsed.params:
        return None
    if parsed.path not in ("", "/"):
        return None
    netloc = parsed.hostname.lower()
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return f"https://{netloc}"


def _public_host(settings: MiniAppSettings) -> str | None:
    origin = _public_origin(settings)
    return _host_only(origin.removeprefix("https://")) if origin else None


def _public_authority(settings: MiniAppSettings) -> str | None:
    origin = _public_origin(settings)
    return origin.removeprefix("https://") if origin else None


def _settings_ready(settings: MiniAppSettings) -> bool:
    if settings.public_smoke:
        origin = _public_origin(settings)
        if origin is None:
            return False
        if settings.cors_allowed_origins != {origin}:
            return False
        if not settings.allowed_users:
            return False
        if not settings.resolved_bot_token():
            return False
        if settings.enable_actions:
            return False
        return True
    return _is_loopback_host(settings.host)


def _host_allowed(host: str, settings: MiniAppSettings) -> bool:
    if settings.public_smoke:
        return bool(host) and host == _public_host(settings)
    return bool(host) and _is_loopback_host(host)


def _host_header_allowed(host_header: str, settings: MiniAppSettings) -> bool:
    if settings.public_smoke:
        return bool(host_header) and _host_authority(host_header) == _public_authority(settings)
    return _host_allowed(_host_only(host_header), settings)


def _origin_allowed_for_request(request: Request, origin: str, settings: MiniAppSettings) -> bool:
    if not settings.public_smoke:
        return not origin or origin in settings.cors_allowed_origins
    if origin == "null":
        return False
    if origin:
        return origin == _public_origin(settings)
    if request.method == "POST":
        return False
    if request.method == "GET" and request.url.path.startswith("/api/"):
        return request.headers.get("sec-fetch-site", "").lower() in {"same-origin", "none"}
    return True


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


def _apply_public_headers(response: Response, settings: MiniAppSettings) -> Response:
    if not settings.public_smoke:
        return response
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


def _apply_api_no_store(response: Response, settings: MiniAppSettings) -> Response:
    response.headers["Cache-Control"] = "no-store"
    return _apply_public_headers(response, settings)


def create_app(
    *,
    settings: MiniAppSettings | None = None,
    status_provider: Callable[[], dict[str, Any]] | None = None,
    approvals_provider: Callable[[], dict[str, Any]] | None = None,
    sessions_provider: Callable[[], dict[str, Any]] | None = None,
    logs_provider: Callable[[], dict[str, Any]] | None = None,
    capabilities_provider: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    settings = settings or MiniAppSettings()
    sessions = SessionStore()
    rate_limiter = RateLimiter()
    status_provider = status_provider or (lambda: build_status_snapshot(hermes_home_configured=True))
    approvals_provider = approvals_provider or build_approvals_snapshot
    sessions_provider = sessions_provider or build_sessions_snapshot
    logs_provider = logs_provider or build_logs_snapshot
    capabilities_provider = capabilities_provider or build_capabilities_snapshot
    app = FastAPI(title="Hermes Telegram Mini App", docs_url=None, redoc_url=None, openapi_url=None)

    def guarded_response(request: Request, response: Response) -> Response:
        if request.url.path.startswith("/api/"):
            return _apply_api_no_store(response, settings)
        return _apply_public_headers(response, settings)

    @app.middleware("http")
    async def _host_origin_cors_guard(request: Request, call_next):
        origin = request.headers.get("origin", "")
        if not _settings_ready(settings):
            return guarded_response(request, _safe_error(503, "Mini App sidecar is not ready"))

        host_header = request.headers.get("host", "")
        if not _host_header_allowed(host_header, settings):
            return guarded_response(request, _safe_error(400, "Invalid Host header"))

        if request.method == "OPTIONS":
            headers = _cors_headers(origin, settings)
            if not headers:
                return guarded_response(request, Response(status_code=204))
            headers.update({
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "content-type",
            })
            return guarded_response(request, Response(status_code=204, headers=headers))

        if not _origin_allowed_for_request(request, origin, settings):
            return guarded_response(request, _safe_error(403, "Origin is not allowed"))

        response = await call_next(request)
        if origin:
            for key, value in _cors_headers(origin, settings).items():
                response.headers[key] = value
        return guarded_response(request, response)

    def current_session(request: Request) -> MiniAppSession:
        session = sessions.get(request.cookies.get(SESSION_COOKIE), now=settings.now())
        if session is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return session

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": "telegram-miniapp", "version": "m3" if settings.public_smoke else "m2"}

    @app.get("/readyz")
    def readyz():
        if not _settings_ready(settings):
            raise HTTPException(status_code=503, detail="Mini App sidecar is not ready")
        return {"ok": True, "service": "telegram-miniapp"}

    @app.post("/api/auth/telegram")
    def auth_telegram(payload: TelegramAuthRequest, request: Request):
        host = _host_only(request.headers.get("host", ""))
        client_host = request.client.host if request.client else "unknown"
        if not rate_limiter.check(
            f"auth:{client_host}:{host}",
            limit=settings.auth_rate_limit_per_minute,
            now=settings.now(),
        ):
            raise HTTPException(status_code=429, detail="Too many requests")
        # The absolute lifetime cap is a blast-radius guard for short public
        # smoke runs only; on a long-lived loopback sidecar it would lock out
        # auth entirely once exhausted.
        if settings.public_smoke and not rate_limiter.check_absolute(
            "auth:global", limit=settings.auth_global_limit
        ):
            raise HTTPException(status_code=429, detail="Too many requests")
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
            if settings.public_smoke:
                raise HTTPException(status_code=401, detail="Unauthorized") from exc
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        session = sessions.create(verified, ttl_seconds=settings.session_ttl_seconds, now=settings.now())
        response = JSONResponse({"ok": True, "user": _user_payload(session.user), "expires_at": session.expires_at.isoformat()})
        response.set_cookie(
            SESSION_COOKIE,
            session.session_id,
            max_age=settings.session_ttl_seconds,
            expires=settings.session_ttl_seconds,
            httponly=True,
            secure=settings.public_smoke,
            samesite="none" if settings.public_smoke else "lax",
            path="/api",
        )
        return _apply_api_no_store(response, settings)

    @app.post("/api/logout")
    def logout(request: Request):
        current_session(request)
        sessions.delete(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/api")
        return _apply_api_no_store(response, settings)

    @app.get("/api/me")
    def me(request: Request):
        session = current_session(request)
        response = JSONResponse({
            "authenticated": True,
            "user": _user_payload(session.user),
            "session_expires_at": session.expires_at.isoformat(),
        })
        return _apply_api_no_store(response, settings)

    @app.get("/api/status")
    def status(request: Request):
        session = current_session(request)
        if settings.public_smoke:
            if not rate_limiter.check(
                f"status:{session.session_id}",
                limit=settings.status_rate_limit_per_minute,
                now=settings.now(),
            ):
                raise HTTPException(status_code=429, detail="Too many requests")
        snapshot = status_provider()
        if settings.public_smoke:
            snapshot = dict(snapshot)
            snapshot["miniapp"] = {"mode": "https-smoke", "actions_enabled": False, "public_exposure": True}
        return _apply_api_no_store(JSONResponse(snapshot), settings)

    @app.get("/api/capabilities")
    def capabilities(request: Request):
        session = current_session(request)
        if settings.public_smoke:
            if not rate_limiter.check(
                f"capabilities:{session.session_id}",
                limit=settings.status_rate_limit_per_minute,
                now=settings.now(),
            ):
                raise HTTPException(status_code=429, detail="Too many requests")
        snapshot = capabilities_provider()
        return _apply_api_no_store(JSONResponse(snapshot), settings)

    @app.get("/api/approvals")
    def approvals(request: Request):
        session = current_session(request)
        if settings.public_smoke:
            if not rate_limiter.check(
                f"approvals:{session.session_id}",
                limit=settings.status_rate_limit_per_minute,
                now=settings.now(),
            ):
                raise HTTPException(status_code=429, detail="Too many requests")
        snapshot = approvals_provider()
        return _apply_api_no_store(JSONResponse(snapshot), settings)

    @app.get("/api/sessions")
    def sessions_preview(request: Request):
        session = current_session(request)
        if settings.public_smoke:
            if not rate_limiter.check(
                f"sessions:{session.session_id}",
                limit=settings.status_rate_limit_per_minute,
                now=settings.now(),
            ):
                raise HTTPException(status_code=429, detail="Too many requests")
        snapshot = sessions_provider()
        return _apply_api_no_store(JSONResponse(snapshot), settings)

    @app.get("/api/logs")
    def logs_preview(request: Request):
        session = current_session(request)
        if settings.public_smoke:
            if not rate_limiter.check(
                f"logs:{session.session_id}",
                limit=settings.status_rate_limit_per_minute,
                now=settings.now(),
            ):
                raise HTTPException(status_code=429, detail="Too many requests")
        snapshot = logs_provider()
        return _apply_api_no_store(JSONResponse(snapshot), settings)

    return app
