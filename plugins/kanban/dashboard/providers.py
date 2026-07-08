"""Provider helpers for the Kanban dashboard.

The native Kanban board is still backed by ``hermes_cli.kanban_db``.  This
module adds a small external read-provider seam so the dashboard can render
task-shaped projections owned by another system without copying those rows
into Hermes' execution database.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from fastapi import HTTPException

from hermes_cli.config import load_config


BOARD_COLUMNS: list[str] = [
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done",
]

_VALID_COLUMNS = set(BOARD_COLUMNS + ["archived"])


@dataclass(frozen=True)
class ProviderSelection:
    name: str
    kind: str
    config: dict[str, Any]

    @property
    def is_native(self) -> bool:
        return self.kind == "native"


def resolve_provider(provider: Optional[str]) -> ProviderSelection:
    """Resolve a dashboard provider name from config.

    ``native`` is always available and remains the default. External providers
    are configured under ``dashboard.kanban.providers``:

    dashboard:
      kanban:
        providers:
          gloops:
            type: external_http
            base_url: http://localhost:3000/api/hermes-kanban
            token_env: GLOOPS_HERMES_KANBAN_TOKEN
    """
    name = (provider or "native").strip() or "native"
    if name == "native":
        return ProviderSelection(name="native", kind="native", config={})

    cfg = load_config()
    dash_cfg = cfg.get("dashboard") or {}
    kanban_cfg = dash_cfg.get("kanban") if isinstance(dash_cfg, dict) else {}
    providers = (
        kanban_cfg.get("providers")
        if isinstance(kanban_cfg, dict)
        else None
    )
    if not isinstance(providers, dict) or name not in providers:
        raise HTTPException(status_code=404, detail=f"kanban provider {name!r} not configured")

    raw = providers.get(name)
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail=f"kanban provider {name!r} config must be an object")
    kind = str(raw.get("type") or raw.get("kind") or "external_http").strip()
    if kind != "external_http":
        raise HTTPException(status_code=400, detail=f"unsupported kanban provider type {kind!r}")
    if not str(raw.get("base_url") or "").strip():
        raise HTTPException(status_code=400, detail=f"kanban provider {name!r} missing base_url")
    return ProviderSelection(name=name, kind=kind, config=dict(raw))


class ExternalHttpTaskProvider:
    """Read-only provider backed by an external HTTP projection API."""

    def __init__(self, selection: ProviderSelection):
        self.selection = selection
        base = str(selection.config.get("base_url") or "").strip()
        self.base_url = base.rstrip("/")
        self.timeout = float(selection.config.get("timeout_seconds") or 5.0)
        self.token_env = str(selection.config.get("token_env") or "").strip() or None

    def get_board(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = self._get_json("/board", params=params)
        return self._normalize_board(payload)

    def get_task(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = self._get_json(f"/tasks/{quote(task_id, safe='')}", params=params)
        return self._normalize_task_detail(payload)

    def _get_json(self, path: str, *, params: Optional[dict[str, Any]] = None) -> Any:
        cleaned = {
            k: v
            for k, v in (params or {}).items()
            if v is not None and v != ""
        }
        query = f"?{urlencode(cleaned, doseq=True)}" if cleaned else ""
        url = f"{self.base_url}{path}{query}"
        headers = {"Accept": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env, "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read(10 * 1024 * 1024)
        except HTTPError as exc:
            detail = exc.read(1024).decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise HTTPException(status_code=502, detail=f"external kanban provider returned {exc.code}: {detail}")
        except URLError as exc:
            raise HTTPException(status_code=502, detail=f"external kanban provider unavailable: {exc.reason}")
        except TimeoutError:
            raise HTTPException(status_code=504, detail="external kanban provider timed out")
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"external kanban provider returned invalid JSON: {exc}")

    def _normalize_board(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="external kanban board payload must be an object")
        raw_columns = payload.get("columns")
        if not isinstance(raw_columns, list):
            raise HTTPException(status_code=502, detail="external kanban board payload missing columns")

        grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in BOARD_COLUMNS}
        if bool(payload.get("include_archived")) or any(
            isinstance(c, dict) and c.get("name") == "archived" for c in raw_columns
        ):
            grouped["archived"] = []

        for column in raw_columns:
            if not isinstance(column, dict):
                continue
            name = str(column.get("name") or "todo")
            if name not in _VALID_COLUMNS:
                name = "todo"
            if name == "archived" and name not in grouped:
                grouped[name] = []
            raw_tasks = column.get("tasks") or []
            if not isinstance(raw_tasks, list):
                raw_tasks = []
            for task in raw_tasks:
                if not isinstance(task, dict):
                    continue
                normalized = self._normalize_task(task, default_status=name)
                grouped.setdefault(normalized["status"], []).append(normalized)

        return {
            "columns": [{"name": name, "tasks": grouped.get(name, [])} for name in grouped.keys()],
            "tenants": _string_list(payload.get("tenants")),
            "assignees": _string_list(payload.get("assignees")),
            "latest_event_id": int(payload.get("latest_event_id") or 0),
            "now": int(payload.get("now") or time.time()),
            "provider": self.selection.name,
            "readonly": True,
        }

    def _normalize_task_detail(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("task"), dict):
            raise HTTPException(status_code=502, detail="external kanban task payload missing task")
        task = self._normalize_task(payload["task"])
        return {
            "task": task,
            "comments": _list(payload.get("comments")),
            "events": _list(payload.get("events")),
            "attachments": _list(payload.get("attachments")),
            "links": payload.get("links") if isinstance(payload.get("links"), dict) else {"parents": [], "children": []},
            "runs": _list(payload.get("runs")),
            "provider": self.selection.name,
            "readonly": True,
        }

    def _normalize_task(self, task: dict[str, Any], *, default_status: str = "todo") -> dict[str, Any]:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            raise HTTPException(status_code=502, detail="external kanban task missing id")
        status = str(task.get("status") or default_status)
        if status not in _VALID_COLUMNS:
            status = "todo"
        out = dict(task)
        out["id"] = task_id
        out["status"] = status
        out["readonly"] = True
        out["external_provider"] = out.get("external_provider") or self.selection.name
        return out


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
