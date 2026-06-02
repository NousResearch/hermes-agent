"""Workspace model and store.

A workspace groups sessions by deployment context (local project, Discord
category, Feishu group, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

Entrypoint = Literal["feishu", "discord", "web", "cli", "mac_app"]

DEFAULT_WORKSPACE_ID = "hermes-local"
DEFAULT_WORKSPACE_NAME = "Local"
DEFAULT_WORKSPACE_ENTRYPOINT: Entrypoint = "cli"


@dataclass(frozen=True, slots=True)
class Workspace:
    workspace_id: str
    name: str
    entrypoint: Entrypoint = DEFAULT_WORKSPACE_ENTRYPOINT
    external_source_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "entrypoint": self.entrypoint,
            "external_source_id": self.external_source_id,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Workspace:
        return Workspace(
            workspace_id=str(data.get("workspace_id") or DEFAULT_WORKSPACE_ID),
            name=str(data.get("name") or DEFAULT_WORKSPACE_NAME),
            entrypoint=str(data.get("entrypoint") or DEFAULT_WORKSPACE_ENTRYPOINT),
            external_source_id=data.get("external_source_id") or None,
            created_at=str(data.get("created_at") or ""),
        )

    @staticmethod
    def make_default() -> Workspace:
        """Return the singleton default workspace for legacy compatibility."""
        return Workspace(
            workspace_id=DEFAULT_WORKSPACE_ID,
            name=DEFAULT_WORKSPACE_NAME,
            entrypoint=DEFAULT_WORKSPACE_ENTRYPOINT,
        )


_ws_store: dict[str, Workspace] = {}
_ws_lock = threading.Lock()


def _workspace_store_path() -> Path:
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "data" / "workspaces.json"


def load_workspaces() -> dict[str, Workspace]:
    path = _workspace_store_path()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Workspace.from_dict(v) for k, v in raw.items()}


def save_workspaces(workspaces: dict[str, Workspace]) -> None:
    path = _workspace_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in workspaces.items()}, f, ensure_ascii=False, indent=2)


def get_workspace(workspace_id: str) -> Workspace | None:
    with _ws_lock:
        if not _ws_store:
            _ws_store.update(load_workspaces())
        return _ws_store.get(workspace_id)


def put_workspace(workspace: Workspace) -> None:
    with _ws_lock:
        _ws_store[workspace.workspace_id] = workspace
        save_workspaces(_ws_store)
