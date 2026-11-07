"""Read-only validation of a Desktop Host Project binding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from hermes_cli import projects_db


@dataclass(frozen=True)
class VerifiedDesktopProject:
    """Project metadata loaded from the owning Hermes profile's database."""

    project_id: str
    primary_path: str
    folders: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "primary_path": self.primary_path,
        }


def _read_only_connection(db_path: Path) -> sqlite3.Connection | None:
    path = Path(db_path).expanduser()
    if not path.is_file():
        return None

    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    connection.row_factory = sqlite3.Row
    return connection


def validate_desktop_project(
    project_id: str | None,
    *,
    db_path: Path | str | None = None,
) -> VerifiedDesktopProject | None:
    """Resolve an exact, active Host Project id without creating or mutating its store."""

    requested = str(project_id or "").strip()
    if not requested:
        return None

    database = Path(db_path) if db_path is not None else projects_db.projects_db_path()
    connection = _read_only_connection(database)
    if connection is None:
        return None

    try:
        project = projects_db.get_project(connection, requested)
        if project is None or project.id != requested or project.archived:
            return None

        primary = str(project.primary_path or "").strip()
        if not primary:
            primary = next(
                (folder.path for folder in project.folders
                 if folder.is_primary and str(folder.path or "").strip()),
                "",
            )
        if not primary and project.folders:
            primary = str(project.folders[0].path or "").strip()
        primary = projects_db._normalize_path(primary) if primary else ""
        if not primary:
            return None

        folders = [primary]
        for folder in project.folders:
            normalized = projects_db._normalize_path(folder.path)
            if normalized and normalized not in folders:
                folders.append(normalized)

        return VerifiedDesktopProject(project.id, primary, tuple(folders))
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def prepare_desktop_project_request(
    params: dict[str, Any],
    *,
    db_path: Path | str,
) -> tuple[dict[str, Any], VerifiedDesktopProject | None]:
    """Replace a Desktop-supplied cwd with the owning profile's verified project root.

    A deliberate cwd may accompany the project only when it names the same normalized path. An
    inherited cwd is presentation state and cannot override the Host Project's root.
    """
    prepared = dict(params)
    project_id = prepared.get("desktop_project_id")
    if project_id is None:
        return prepared, None
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("desktop_project_id must be a non-empty Host Project id")
    if prepared.get("source") != "gui":
        raise ValueError("desktop_project_id is only valid for a GUI session")
    cwd_explicit = prepared.get("cwd_explicit", False)
    if type(cwd_explicit) is not bool:
        raise ValueError("cwd_explicit must be a boolean")

    project = validate_desktop_project(project_id, db_path=db_path)
    if project is None:
        raise ValueError("desktop_project_id does not resolve to an active Host Project")

    client_cwd = str(prepared.get("cwd") or "").strip()
    if cwd_explicit and client_cwd:
        requested_path = os.path.normcase(projects_db._normalize_path(client_cwd))
        project_path = os.path.normcase(project.primary_path)
        if requested_path != project_path:
            raise ValueError("explicit cwd conflicts with the verified Host Project root")

    prepared["cwd"] = project.primary_path
    prepared["cwd_explicit"] = True
    return prepared, project


def validate_session_desktop_project(policy, authority) -> VerifiedDesktopProject | None:
    """Revalidate a frozen Desktop Project binding against its owning profile on each turn."""
    request = json.loads(policy.request_json)
    project_id = request.get("desktop_project_id")
    if project_id is None:
        return None
    if policy.source != "gui" or authority is None:
        raise ValueError("the saved Desktop Project has no GUI profile authority")

    db_path = Path(authority.db.db_path).resolve().parent / "projects.db"
    project = validate_desktop_project(project_id, db_path=db_path)
    if project is None:
        raise ValueError("the saved Desktop Project is missing, archived, or invalid")
    if os.path.normcase(project.primary_path) != os.path.normcase(
        projects_db._normalize_path(policy.cwd)
    ):
        raise ValueError("the saved Desktop Project root changed; create a new session")
    return project
