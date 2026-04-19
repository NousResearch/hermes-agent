import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import Base, SessionLocal, engine
from .job_runner import process_ingestion_jobs as process_ingestion_jobs_core
from .migration_bootstrap import run_startup_schema_bootstrap
from .models import IngestionJob
from .routers import analytics, ask, audit, auth, documents, groups, handoff, integrations, policy_router

settings = get_settings()


def _ingestion_worker_loop() -> None:
    while True:
        try:
            with SessionLocal() as db:
                process_ingestion_jobs_core(db, limit=5)
        except Exception:
            # Keep daemon alive even on transient failures.
            pass

        time.sleep(max(1, int(settings.ingestion_worker_interval_seconds)))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # First create currently-known tables, then patch legacy sqlite schemas in-place.
    Base.metadata.create_all(bind=engine)
    run_startup_schema_bootstrap(engine)

    if settings.ingestion_worker_enabled:
        thread = threading.Thread(target=_ingestion_worker_loop, daemon=True, name="ingestion-worker")
        thread.start()

    yield


app = FastAPI(title=settings.app_name, version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    queued = 0
    retry = 0
    failed = 0
    try:
        with SessionLocal() as db:
            queued = db.query(IngestionJob).filter(IngestionJob.status == "queued").count()
            retry = db.query(IngestionJob).filter(IngestionJob.status == "retry").count()
            failed = db.query(IngestionJob).filter(IngestionJob.status == "failed").count()
    except Exception:
        pass

    return {
        "ok": True,
        "version": "2.2.0",
        "ingestion_worker_enabled": bool(settings.ingestion_worker_enabled),
        "ingestion_queue_depth": int(queued + retry),
        "ingestion_dead_letter_count": int(failed),
    }


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(groups.router)
app.include_router(policy_router.router)
app.include_router(ask.router)
app.include_router(handoff.router)
app.include_router(analytics.router)
app.include_router(audit.router)
app.include_router(integrations.router)
