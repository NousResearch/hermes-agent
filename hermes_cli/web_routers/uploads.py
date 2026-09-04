"""General-purpose phone-to-tower file drop endpoints.

Lets the mobile chat UI (or any dashboard client) push a file straight
onto disk at a named, pre-configured destination — e.g. a screenshot from
a phone landing in a mygen ingest pending directory. Deliberately generic:
destinations are declared in config (``dashboard.upload_targets``), not
hardcoded, so new drop points can be added without a code change.

Auth: mounted under ``/api/`` like every other dashboard route, so it is
covered by the existing dashboard-auth middleware in web_server.py —
no separate gating needed here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from hermes_cli.config import load_config_readonly
from hermes_cli.kanban_db import _collision_free_path, _safe_attachment_name

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Hard cap independent of any per-target override — a phone photo is a few
# MB at most; this just stops a runaway/misconfigured client from filling
# the disk. Matches the kanban attachment cap. A target's own `max_mb`
# (config) can raise this per-destination (e.g. video needs far more room
# than a screenshot ever would) but never lower it below a sane floor.
_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
# Absolute ceiling regardless of per-target config — stops a typo'd
# max_mb (or a deliberately hostile config edit) from being able to fill
# the disk unbounded. 4GB comfortably covers a multi-minute 4K phone clip.
_UPLOAD_ABSOLUTE_MAX_BYTES = 4 * 1024 * 1024 * 1024


def _upload_targets() -> Dict[str, Dict[str, Any]]:
    """Read ``dashboard.upload_targets`` from config.

    Shape in config.yaml::

        dashboard:
          upload_targets:
            - id: mygen-screenshots
              label: "MyGen: screenshot inbox"
              path: "C:/projects/mygen/inbox/pending/screenshots"
              accept: "image/*"       # optional, informs the picker's file filter
              max_mb: 25              # optional, overrides the 25MB default cap

    Returns a dict keyed by ``id`` for O(1) lookup. Missing/malformed
    entries are skipped (logged), never crash the endpoint.
    """
    cfg = load_config_readonly()
    raw = ((cfg.get("dashboard") or {}).get("upload_targets")) or []
    targets: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, list):
        return targets
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("id")
        path = entry.get("path")
        if not tid or not path:
            continue
        max_bytes = _UPLOAD_MAX_BYTES
        raw_max_mb = entry.get("max_mb")
        if raw_max_mb is not None:
            try:
                max_bytes = min(
                    int(float(raw_max_mb) * 1024 * 1024), _UPLOAD_ABSOLUTE_MAX_BYTES
                )
            except (TypeError, ValueError):
                pass
        targets[str(tid)] = {
            "id": str(tid),
            "label": str(entry.get("label") or tid),
            "path": str(path),
            "accept": str(entry.get("accept") or "") or None,
            "max_bytes": max_bytes,
        }
    return targets


@router.get("/uploads/targets")
def list_upload_targets():
    """List configured upload destinations for the picker UI."""
    targets = _upload_targets()
    return {
        "targets": [
            {"id": t["id"], "label": t["label"], "accept": t["accept"]}
            for t in targets.values()
        ]
    }


@router.post("/uploads/{target_id}")
async def upload_to_target(target_id: str, file: UploadFile = File(...)):
    """Stream an uploaded file into a pre-configured destination directory.

    Mirrors the kanban attachment upload handler (streamed write, hard
    size cap, sanitised/collision-free filename) but writes to a plain
    directory on disk instead of a DB-tracked attachments store — the
    destination is just a drop point another workflow (e.g. mygen ingest)
    picks up later.
    """
    targets = _upload_targets()
    target = targets.get(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown upload target {target_id!r}")

    dest_dir = Path(target["path"])
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"cannot create destination: {exc}")

    try:
        safe_name = _safe_attachment_name(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    dest_path = _collision_free_path(dest_dir, safe_name)
    candidate = dest_path.name
    max_bytes = int(target.get("max_bytes") or _UPLOAD_MAX_BYTES)

    total = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds {max_bytes // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to store upload: {exc}")

    _log.info(
        "dashboard upload: target=%s file=%s bytes=%d dest=%s",
        target_id,
        candidate,
        total,
        dest_path,
    )

    return {
        "ok": True,
        "target": target_id,
        "filename": candidate,
        "size": total,
        "path": str(dest_path.resolve()),
    }
