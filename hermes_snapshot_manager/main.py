from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dataclasses import asdict, is_dataclass

from hermes_snapshot_manager.api import health, restore, settings as settings_api, snapshots
from hermes_snapshot_manager.core.config import load_settings
from hermes_snapshot_manager.core.display import format_bytes
from hermes_snapshot_manager.core.paths import build_paths, ensure_app_dirs
from hermes_snapshot_manager.services.restore_service import RestoreService
from hermes_snapshot_manager.services.scheduler_service import SchedulerService, describe_cron
from hermes_snapshot_manager.services.snapshot_service import SnapshotService

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

paths = build_paths()
ensure_app_dirs(paths)
snapshot_service = SnapshotService(paths)
restore_service = RestoreService(paths)
scheduler_service = SchedulerService(paths)


def _enrich_snapshot_detail(detail: dict) -> dict:
    metadata = detail.get("metadata") or {}
    snapshot = detail.get("snapshot") or {}
    original_bytes = snapshot.get("total_bytes") or metadata.get("total_bytes") or 0
    compressed_bytes = metadata.get("payload_archive_size")
    detail["display_stats"] = {
        "original_bytes": original_bytes,
        "original_bytes_human": format_bytes(original_bytes),
        "compressed_bytes": compressed_bytes,
        "compressed_bytes_human": format_bytes(compressed_bytes),
        "space_saved_bytes": metadata.get("space_saved_bytes"),
        "space_saved_bytes_human": format_bytes(metadata.get("space_saved_bytes")),
        "compression_ratio": metadata.get("compression_ratio"),
        "payload_format": metadata.get("payload_format", "-"),
    }
    return detail


def _base_context(*, title: str, current_path: str) -> dict:
    return {"title": title, "current_path": current_path}


def _status_tone(status: str | None) -> str:
    mapping = {
        "created": "info",
        "verified": "success",
        "completed": "success",
        "running": "running",
        "degraded": "warning",
        "failed": "danger",
        "stale": "warning",
        "success": "success",
    }
    return mapping.get((status or "").lower(), "neutral")


def _snapshot_card(snapshot) -> dict | None:
    if not snapshot:
        return None
    data = asdict(snapshot) if is_dataclass(snapshot) else dict(snapshot)
    data["total_bytes_human"] = format_bytes(data.get("total_bytes"))
    data["status_tone"] = _status_tone(data.get("status"))
    return data

app = FastAPI(title="Hermes Snapshot Manager", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(health.router)
app.include_router(snapshots.router)
app.include_router(restore.router)
app.include_router(settings_api.router)


@app.on_event("startup")
def startup_event() -> None:
    scheduler_service.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    scheduler_service.stop()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    snapshots_list = snapshot_service.list_snapshots()
    latest = snapshots_list[0] if snapshots_list else None
    history = restore_service.list_restore_history()
    latest_restore = history[0] if history else None
    settings = load_settings(paths)
    latest_snapshot = _snapshot_card(latest)
    if latest_snapshot and latest:
        latest_detail = _enrich_snapshot_detail(snapshot_service.get_snapshot(latest.id))
        latest_snapshot["compression_ratio"] = latest_detail["display_stats"].get("compression_ratio")
        latest_snapshot["space_saved_bytes_human"] = latest_detail["display_stats"].get("space_saved_bytes_human")
    context = {
        **_base_context(title="Dashboard · Hermes Snapshot Manager", current_path="/"),
        "snapshot_count": len(snapshots_list),
        "latest_snapshot": latest_snapshot,
        "latest_snapshots": [_snapshot_card(item) for item in snapshots_list[:5]],
        "latest_restore": latest_restore,
        "latest_restore_tone": _status_tone(None if not latest_restore else latest_restore.get("result")),
        "restore_count": len(history),
        "paths": paths,
        "settings": settings,
        "schedule_summary": {
            **scheduler_service.next_run_info(),
            "cron_description": describe_cron(settings.schedule_cron),
        },
    }
    return TEMPLATES.TemplateResponse(request, "dashboard.html", context)


@app.get("/snapshots", response_class=HTMLResponse)
def snapshots_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "snapshots.html",
        {**_base_context(title="Snapshots · Hermes Snapshot Manager", current_path="/snapshots"), "snapshots": [_snapshot_card(item) for item in snapshot_service.list_snapshots()]},
    )


@app.get("/snapshots/{snapshot_id}", response_class=HTMLResponse)
def snapshot_detail(request: Request, snapshot_id: str):
    detail = _enrich_snapshot_detail(snapshot_service.get_snapshot(snapshot_id))
    verification = snapshot_service.verify_snapshot(snapshot_id)
    diff = snapshot_service.diff_snapshot_to_current(snapshot_id)
    detail["snapshot"]["status_tone"] = _status_tone(detail["snapshot"].get("status"))
    return TEMPLATES.TemplateResponse(
        request,
        "snapshot_detail.html",
        {**_base_context(title=f"Snapshot {snapshot_id} · Hermes Snapshot Manager", current_path="/snapshots"), **detail, "verification": verification, "diff": diff},
    )


@app.get("/restores", response_class=HTMLResponse)
def restore_history_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "restore_history.html",
        {**_base_context(title="Restore History · Hermes Snapshot Manager", current_path="/restores"), "history": restore_service.list_restore_history()},
    )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    settings = load_settings(paths)
    schedule_summary = {
        **scheduler_service.next_run_info(),
        "cron_description": describe_cron(settings.schedule_cron),
        "exclude_patterns_text": "\n".join(settings.exclude_patterns),
    }
    return TEMPLATES.TemplateResponse(
        request,
        "settings.html",
        {**_base_context(title="Settings · Hermes Snapshot Manager", current_path="/settings"), "settings": settings, "schedule_summary": schedule_summary},
    )


def main() -> None:
    uvicorn.run("hermes_snapshot_manager.main:app", host="127.0.0.1", port=8876, reload=False)


if __name__ == "__main__":
    main()
