from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger

from . import db
from .config import VERSION
from .routes import admin, ask, health, keys, parse
from .tasks.cleanup_task import run_cleanup_loop

# --- structured JSON logging ---
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# --- request logging middleware ---
async def _log_request(request: Request, call_next):
    response = await call_next(request)
    api_key_id: str | None = None
    # Best-effort key resolution for logs (non-blocking)
    x_api_key = request.headers.get("X-API-Key")
    if x_api_key:
        try:
            async with db.connect() as conn:
                conn.row_factory = aiosqlite.Row
                row = await db.fetchone(conn, "SELECT id FROM api_keys WHERE key=?", (x_api_key,))
            api_key_id = row["id"] if row else None
        except Exception:
            pass
    try:
        async with db.connect() as conn:
            await conn.execute(
                "INSERT INTO request_log (api_key_id, endpoint, status_code) VALUES (?,?,?)",
                (api_key_id, request.url.path, response.status_code),
            )
            await conn.commit()
    except Exception:
        pass
    return response


# --- lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    logger.info("deepparser-api v%s starting", VERSION)
    cleanup_task = asyncio.create_task(run_cleanup_loop())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# --- app ---
app = FastAPI(
    title="DeepParser Developer API",
    version=VERSION,
    description="Parse DWG, Excel-embedded PDFs, and scanned tables without OCR soup.",
    lifespan=lifespan,
)

app.middleware("http")(_log_request)

app.include_router(health.router)
app.include_router(keys.router)
app.include_router(parse.router)
app.include_router(ask.router)
app.include_router(admin.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred.",
            "doc_url": "https://github.com/ysh145/hermes-agent/tree/main/deepparser",
        },
    )


def run() -> None:
    import uvicorn
    uvicorn.run("deepparser_api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
