"""projects.list handler (contract §4.7)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def list_projects_snapshot(
    params: Optional[Dict[str, Any]] = None,
    *,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return Hermes projects for Portal pickers.

    Missing ``projects.db`` → empty list with ``source: "none"`` (MVP simplicity).
    """
    params = params or {}
    include_archived = bool(params.get("include_archived", False))

    try:
        from hermes_cli.projects_db import connect_closing, list_projects, projects_db_path
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain_rpc projects: import failed: %s", exc)
        return {"projects": [], "source": "none"}

    path = db_path if db_path is not None else projects_db_path()
    if not path.is_file():
        return {"projects": [], "source": "none"}

    try:
        with connect_closing(db_path=path) as conn:
            projects = list_projects(conn, include_archived=include_archived)
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain_rpc projects: db read failed: %s", exc)
        return {"projects": [], "source": "none"}

    out: List[Dict[str, Any]] = []
    for p in projects:
        folders = []
        for f in p.folders or []:
            folders.append(
                {
                    "path": f.path,
                    "label": f.label,
                    "is_primary": bool(f.is_primary),
                }
            )
        out.append(
            {
                "id": p.id,
                "slug": p.slug,
                "name": p.name,
                "primary_path": p.primary_path,
                "folders": folders,
            }
        )
    return {"projects": out, "source": "hermes_projects_db"}
