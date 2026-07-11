"""ASGI runtime for the read-only Telegram Mini App."""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from hermes_constants import get_default_hermes_root, get_hermes_home

from .auth import (
    MAX_INIT_DATA_BYTES,
    PRIVATE_SESSION_RATE_LIMIT,
    SESSION_COOKIE,
    MiniAppAuth,
)
from .projection import (
    normalize_swarm_board,
    project_catalog,
    project_sessions,
    project_status,
)

ROOT = Path(__file__).resolve().parent
HERMES_HOME = get_hermes_home()
MEMORY_DIR = HERMES_HOME / "memories"
MAX_REQUEST_BODY_BYTES = 1024 * 1024
REQUEST_BODY_TIMEOUT_SECONDS = 5.0
MAX_CONCURRENT_REQUESTS = 32

_BASE_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": (
        "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
        "clipboard-read=(), clipboard-write=(self)"
    ),
    "Cache-Control": "no-store",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' https://telegram.org; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors https://web.telegram.org "
        "https://*.telegram.org; base-uri 'none'; object-src 'none'; form-action 'none'"
    ),
}

auth = MiniAppAuth(
    bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    allowed_users_raw=os.environ.get("TELEGRAM_MINI_APP_OWNER_IDS", ""),
    public_url=os.environ.get("TELEGRAM_MINI_APP_PUBLIC_URL", ""),
)


class RequestBodyLimitMiddleware:
    """Admit a bounded number of requests before buffering any body bytes."""

    def __init__(
        self,
        application,
        max_bytes: int,
        timeout_seconds: float,
        max_concurrent: int,
        admission_timeout_seconds: float = 0.25,
    ):
        self.application = application
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.admission_timeout_seconds = admission_timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrent)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.application(scope, receive, send)
            return
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self.admission_timeout_seconds
            )
        except TimeoutError:
            response = JSONResponse(
                status_code=503,
                content={"error": "Mini App is at its concurrency limit"},
                headers=_BASE_SECURITY_HEADERS,
            )
            await response(scope, receive, send)
            return

        try:
            buffered = []
            total = 0
            more_body = True
            deadline = asyncio.get_running_loop().time() + self.timeout_seconds
            while more_body:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    message = await asyncio.wait_for(
                        receive(), timeout=max(0, remaining)
                    )
                except TimeoutError:
                    response = JSONResponse(
                        status_code=408,
                        content={"error": "Request body timed out"},
                        headers=_BASE_SECURITY_HEADERS,
                    )
                    await response(scope, receive, send)
                    return
                buffered.append(message)
                if message.get("type") != "http.request":
                    break
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={"error": "Request body is too large"},
                        headers=_BASE_SECURITY_HEADERS,
                    )
                    await response(scope, receive, send)
                    return
                more_body = bool(message.get("more_body", False))

            async def replay_receive():
                if buffered:
                    return buffered.pop(0)
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.application(scope, replay_receive, send)
        finally:
            self._slots.release()


app = FastAPI(
    title="Hermes Telegram Mini App",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_bytes=MAX_REQUEST_BODY_BYTES,
    timeout_seconds=REQUEST_BODY_TIMEOUT_SECONDS,
    max_concurrent=MAX_CONCURRENT_REQUESTS,
)
app.mount(
    "/static", StaticFiles(directory=ROOT / "static", check_dir=False), name="static"
)


def _apply_security_headers(response: Response, request: Request) -> Response:
    for key, value in _BASE_SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    cacheable_static = (
        request.url.path.startswith("/static/")
        and request.method in {"GET", "HEAD"}
        and response.status_code in {200, 304}
    )
    response.headers["Cache-Control"] = (
        "public, max-age=31536000, immutable" if cacheable_static else "no-store"
    )
    return response


def _security_error(request: Request, status_code: int, message: str) -> Response:
    return _apply_security_headers(
        JSONResponse(status_code=status_code, content={"error": message}), request
    )


@app.middleware("http")
async def security_boundary(request: Request, call_next):
    init_data = request.headers.get("x-telegram-init-data", "")
    if len(init_data.encode()) > MAX_INIT_DATA_BYTES:
        return _security_error(request, 431, "Telegram initData header is too large")
    if init_data and request.url.path != "/api/auth/session":
        return _security_error(
            request, 401, "Telegram initData is accepted only by the session exchange"
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return _security_error(request, 413, "Request body is too large")
        except ValueError:
            return _security_error(request, 400, "Invalid Content-Length header")

    is_private = (
        request.url.path.startswith("/api/") and request.url.path != "/api/auth/session"
    )
    if is_private:
        record = auth.record(
            request.cookies.get(SESSION_COOKIE, ""),
            request.headers.get("user-agent", ""),
        )
        if not record:
            return _security_error(
                request, 401, "An authenticated Mini App session is required"
            )
        now = time.time()
        events = [stamp for stamp in record["request_events"] if now - stamp < 60]
        record["request_events"] = events
        if len(events) >= PRIVATE_SESSION_RATE_LIMIT:
            return _security_error(request, 429, "Mini App session rate limit exceeded")
        events.append(now)
        request.state.app_session = record

    response: Response = await call_next(request)
    return _apply_security_headers(response, request)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return _apply_security_headers(
        JSONResponse(status_code=exc.status_code, content={"error": exc.detail}),
        request,
    )


@app.get("/")
async def index():
    index_path = ROOT / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(503, "Mini App assets are not installed")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    return {"ok": True, "service": "hermes-telegram-mini-app"}


@app.post("/api/auth/session")
async def create_session(request: Request):
    return auth.exchange(request)


@app.delete("/api/auth/session")
async def delete_session(request: Request):
    return auth.logout(request)


@app.get("/api/me")
async def me(request: Request):
    record = getattr(request.state, "app_session", None)
    if not record:
        raise HTTPException(401, "An authenticated Mini App session is required")
    return {"ok": True, "user": record["user"], "allowed": True}


def _memory_file(path: Path) -> dict[str, Any]:
    target = "user" if path.name == "USER.md" else "memory"
    label = "User profile" if target == "user" else "Hermes memory"
    if not path.exists():
        return {
            "name": path.name,
            "path": f"~/.hermes/memories/{path.name}",
            "target": target,
            "label": label,
            "raw": "",
            "entries": [],
            "count": 0,
            "updated_at": None,
        }
    text = path.read_text(errors="ignore")
    modified = path.stat().st_mtime
    parts = [part.strip() for part in re.split(r"\n?§\n?", text) if part.strip()]
    entries = [
        {
            "id": f"{path.stem.lower()}-{index}",
            "target": target,
            "file": path.name,
            "text": item,
            "updated_at": modified,
        }
        for index, item in enumerate(parts, 1)
    ]
    return {
        "name": path.name,
        "path": f"~/.hermes/memories/{path.name}",
        "target": target,
        "label": label,
        "raw": text,
        "entries": entries,
        "count": len(entries),
        "updated_at": modified,
    }


@app.get("/api/memory")
async def memories():
    files = [
        _memory_file(MEMORY_DIR / "USER.md"),
        _memory_file(MEMORY_DIR / "MEMORY.md"),
    ]
    entries = [entry for item in files for entry in item["entries"]]
    return {
        "files": files,
        "memory": files[1]["entries"],
        "user": files[0]["entries"],
        "entries": entries,
        "count": len(entries),
    }


def _gateway_status_payload(home: Path = HERMES_HOME) -> dict[str, Any]:
    """Project the gateway's persisted runtime record without a network hop."""
    from gateway.status import get_runtime_status_running_pid, read_runtime_status
    from hermes_cli import __version__

    runtime = read_runtime_status(home / "gateway_state.json") or {}
    running = get_runtime_status_running_pid(runtime, expected_home=home) is not None
    platforms = runtime.get("platforms")
    if not isinstance(platforms, dict):
        platforms = {}
    return {
        "version": __version__,
        "gateway_running": running,
        "gateway_state": runtime.get("gateway_state")
        or ("running" if running else "stopped"),
        "gateway_platforms": platforms,
        "source": "gateway-runtime-status",
    }


def _profile_list() -> list[dict[str, Any]]:
    """Describe only the active profile; sibling profile homes remain masked."""
    default_root = get_default_hermes_root()
    active_home = HERMES_HOME.resolve()
    name = "default"
    try:
        relative = active_home.relative_to((default_root / "profiles").resolve())
        if len(relative.parts) == 1 and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", relative.parts[0]
        ):
            name = relative.parts[0]
    except ValueError:
        pass
    runtime = _gateway_status_payload(active_home)
    return [
        {
            "name": name,
            "profile": name,
            "active": True,
            "gateway": "connected" if runtime["gateway_running"] else "stopped",
        }
    ]


def _session_page(limit: int, offset: int) -> dict[str, Any]:
    """Read compact session rows through SQLite's read-only mode."""
    from hermes_state import SessionDB

    db_path = HERMES_HOME / "state.db"
    if not db_path.is_file():
        return {"sessions": [], "total": 0, "limit": limit, "offset": offset}
    database = SessionDB(db_path=db_path, read_only=True)
    try:
        rows = database.list_sessions_rich(
            limit=limit,
            offset=offset,
            order_by_last_active=True,
            compact_rows=True,
        )
        total = database.session_count(exclude_children=True)
    finally:
        database.close()
    return {"sessions": rows, "total": total, "limit": limit, "offset": offset}


def _latest_session() -> dict[str, Any]:
    rows = _session_page(1, 0)["sessions"]
    return rows[0] if rows else {}


def _kanban_board(include_archived: bool, tenant: str) -> dict[str, Any]:
    """Read the active Kanban database with a query-only SQLite connection."""
    from hermes_cli import kanban_db as kb

    db_path = kb.kanban_db_path()
    statuses = [
        "triage",
        "todo",
        "scheduled",
        "ready",
        "running",
        "blocked",
        "review",
        "done",
    ]
    if include_archived:
        statuses.append("archived")
    if not db_path.is_file():
        return {
            "columns": [{"name": status, "tasks": []} for status in statuses],
            "latest_event_id": 0,
            "now": int(time.time()),
        }
    connection = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=1.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        columns = []
        for task_status in statuses:
            tasks = []
            for task in kb.list_tasks(
                connection,
                status=task_status,
                tenant=tenant or None,
                include_archived=include_archived,
                limit=40,
            ):
                item = vars(task).copy()
                item["updated_at"] = (
                    item.get("completed_at")
                    or item.get("started_at")
                    or item.get("created_at")
                )
                tasks.append(item)
            columns.append({"name": task_status, "tasks": tasks})
        try:
            latest_event_id = int(
                connection.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM task_events"
                ).fetchone()[0]
            )
        except sqlite3.DatabaseError:
            latest_event_id = 0
    finally:
        connection.close()
    return {
        "columns": columns,
        "latest_event_id": latest_event_id,
        "now": int(time.time()),
    }


def _toolset_catalog() -> list[dict[str, Any]]:
    """Return built-in registry metadata without loading user configuration."""
    from toolsets import TOOLSETS

    items = [
        {
            "name": name,
            "description": definition.get("description", ""),
            "category": "built-in",
        }
        for name, definition in sorted(TOOLSETS.items())
    ]
    return project_catalog(items, "toolsets")


def _skill_catalog() -> list[dict[str, Any]]:
    """Scan local skill manifests only; do not load config or environment secrets."""
    import yaml

    skills_root = HERMES_HOME / "skills"
    if not skills_root.is_dir():
        return []
    items = []
    for skill_md in sorted(skills_root.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")[:4000]
        except OSError:
            continue
        metadata: dict[str, Any] = {}
        if text.startswith("---\n"):
            frontmatter, separator, _ = text[4:].partition("\n---\n")
            if separator:
                try:
                    loaded = yaml.safe_load(frontmatter)
                except yaml.YAMLError:
                    loaded = None
                if isinstance(loaded, dict):
                    metadata = loaded
        items.append({
            "name": str(metadata.get("name") or skill_md.parent.name)[:100],
            "description": str(metadata.get("description") or "")[:500],
            "category": str(metadata.get("category") or "local")[:100],
        })
    return project_catalog(items, "skills")


@app.get("/api/status")
async def status():
    return project_status(_gateway_status_payload())


@app.get("/api/swarm/board")
async def swarm_board(include_archived: bool = False, tenant: str = ""):
    return normalize_swarm_board(
        _kanban_board(include_archived, tenant[:100]), _profile_list()
    )


@app.get("/api/sessions")
async def sessions(limit: int = 20, offset: int = 0):
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    return project_sessions(_session_page(limit, offset))


@app.get("/api/tools/toolsets")
async def toolsets():
    return _toolset_catalog()


@app.get("/api/skills")
async def skills():
    return _skill_catalog()


@app.get("/api/live-usage")
async def live_usage(include_accounts: bool = False):
    """Return local session usage without provider or account network calls."""
    session = _latest_session()
    input_tokens = int(session.get("input_tokens") or 0)
    output_tokens = int(session.get("output_tokens") or 0)
    cache_read = int(session.get("cache_read_tokens") or 0)
    cache_write = int(session.get("cache_write_tokens") or 0)
    prompt_tokens = int(session.get("last_prompt_tokens") or 0)
    return {
        "session_id": session.get("id"),
        "model": session.get("model"),
        "provider": session.get("billing_provider"),
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(session.get("reasoning_tokens") or 0),
        "total_tokens": input_tokens + output_tokens + cache_read + cache_write,
        "api_calls": int(session.get("api_call_count") or 0),
        "cost": session.get("actual_cost_usd") or session.get("estimated_cost_usd"),
        "context": {
            "prompt_tokens": prompt_tokens,
            "context_length": 0,
            "percent": 0,
        },
        "account": None,
        "accounts": {},
        "accounts_requested": bool(include_accounts),
        "accounts_available": False,
        "updated_at": time.time(),
    }
