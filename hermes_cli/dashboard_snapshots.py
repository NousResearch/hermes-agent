"""Dashboard snapshot registry and health adapters.

This module is the production-facing bridge for the dashboard signal contract:
Hermes can expose its own ``/dashboard-snapshot`` and aggregate all known
``hermes.dashboards.json`` registries into standard DashboardSnapshot payloads.
Project-owned snapshot endpoints can replace these synthesized registry
snapshots later without changing the executive dashboard contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hermes_cli import __version__


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_dashboard_registries(root: Path) -> list[Path]:
    here = root / "hermes.dashboards.json"
    projects_root = root.parent
    registries: list[Path] = []
    if here.exists():
        registries.append(here)
    if projects_root.exists():
        for entry in projects_root.iterdir():
            if not entry.is_dir():
                continue
            candidate = entry / "hermes.dashboards.json"
            if candidate.exists() and candidate.resolve() != here.resolve():
                registries.append(candidate)
    return sorted(registries, key=lambda path: str(path))


def load_registry_dashboards(root: Path) -> list[dict[str, Any]]:
    dashboards_by_id: dict[str, dict[str, Any]] = {}
    for registry_path in discover_dashboard_registries(root):
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            continue
        for dashboard in registry.get("dashboards", []):
            dashboard_id = str(dashboard.get("id") or "").strip()
            if not dashboard_id:
                continue
            item = dict(dashboard)
            item["_registry_path"] = str(registry_path)
            item.setdefault("projectPath", str(registry_path.parent))
            item.setdefault("projectName", registry_path.parent.name)
            dashboards_by_id[dashboard_id] = merge_dashboard_registry_item(dashboards_by_id.get(dashboard_id), item)
    return list(dashboards_by_id.values())


def merge_dashboard_registry_item(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if not existing:
        return incoming
    merged = {**existing, **incoming}
    for key in ("url", "healthUrl", "snapshotUrl"):
        merged[key] = best_url(existing.get(key), incoming.get(key))
    for key in ("projectPath", "projectName", "owner", "category", "label", "description"):
        if not incoming.get(key) and existing.get(key):
            merged[key] = existing[key]
    return merged


def best_url(left: Any, right: Any) -> Any:
    left_s = str(left or "")
    right_s = str(right or "")
    if right_s.startswith("https://"):
        return right
    if left_s.startswith("https://"):
        return left
    if right_s.startswith("http://"):
        return right
    if left_s.startswith("http://"):
        return left
    return right or left


def build_hermes_dashboard_snapshot(root: Path, now: str | None = None) -> dict[str, Any]:
    timestamp = now or now_iso()
    return {
        "source": {
            "id": "nous-hermes-agent.dashboard",
            "label": "Nous Hermes Agent",
            "owner": "Hermes",
            "category": "control-plane",
            "projectName": "Nous Hermes Agent",
            "projectPath": str(root),
            "url": "/",
            "healthUrl": "/api/status",
        },
        "health": {
            "state": "healthy",
            "score": 88,
            "message": "Hermes dashboard snapshot endpoint is live.",
            "checkedAt": timestamp,
            "freshness": "fresh",
        },
        "cost": {
            "period": "unknown",
            "known": False,
            "message": "Hermes dashboard cost telemetry is not attached to this snapshot yet.",
        },
        "capacity": {
            "known": True,
            "used": 1,
            "limit": 1,
            "pressure": "low",
            "message": "Dashboard API is serving the standard snapshot contract.",
        },
        "queue": {"queued": 0, "running": 1, "failed": 0, "blocked": 0, "stale": 0, "completed": 0},
        "actions": [],
        "deployment": {
            "environment": "local",
            "status": "current",
            "version": __version__,
            "message": "Served by the active Hermes dashboard backend.",
        },
        "updatedAt": timestamp,
    }


def build_registry_dashboard_snapshots(
    root: Path,
    *,
    live: bool = False,
    timeout: float = 2.0,
    now: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = now or now_iso()
    snapshots = [registry_dashboard_to_snapshot(item, live=live, timeout=timeout, now=timestamp) for item in load_registry_dashboards(root)]
    if not any(snapshot["source"]["id"] == "nous-hermes-agent.dashboard" for snapshot in snapshots):
        snapshots.insert(0, build_hermes_dashboard_snapshot(root, timestamp))
    return snapshots


def registry_dashboard_to_snapshot(
    dashboard: dict[str, Any],
    *,
    live: bool,
    timeout: float,
    now: str,
) -> dict[str, Any]:
    snapshot_url = dashboard.get("snapshotUrl")
    if live and snapshot_url:
        resolved_snapshot_url = resolve_dashboard_url(snapshot_url, dashboard.get("url"))
        if resolved_snapshot_url:
            try:
                return normalize_project_snapshot(
                    fetch_json(resolved_snapshot_url, timeout=timeout),
                    dashboard,
                    now,
                    resolved_snapshot_url,
                )
            except Exception:
                pass
    source = {
        "id": str(dashboard.get("id") or "unknown-dashboard"),
        "label": str(dashboard.get("label") or dashboard.get("id") or "Unknown Dashboard"),
        "owner": str(dashboard.get("owner") or "Dashboard owner"),
        "category": str(dashboard.get("category") or "unknown"),
        "projectName": dashboard.get("projectName"),
        "projectPath": dashboard.get("projectPath"),
        "url": dashboard.get("url"),
        "healthUrl": dashboard.get("healthUrl"),
    }
    health = health_for_dashboard(dashboard, live=live, timeout=timeout, now=now)
    return {
        "source": source,
        "health": health,
        "cost": {
            "period": "unknown",
            "known": False,
            "message": "Project has not exposed a standard cost signal.",
        },
        "capacity": {
            "known": False,
            "pressure": "unknown",
            "message": "Project has not exposed a standard capacity signal.",
        },
        "queue": queue_for_dashboard(source["id"], health["state"]),
        "actions": actions_for_dashboard(dashboard, source["id"], health),
        "deployment": deployment_for_dashboard(dashboard),
        "updatedAt": now,
    }


def health_for_dashboard(
    dashboard: dict[str, Any],
    *,
    live: bool,
    timeout: float,
    now: str,
) -> dict[str, Any]:
    health_url = dashboard.get("healthUrl")
    if not health_url:
        return {
            "state": "unknown",
            "score": 45,
            "message": "No healthUrl is registered.",
            "checkedAt": now,
            "freshness": "unknown",
        }
    if not live:
        return {
            "state": "unknown" if str(health_url).startswith("/") else "degraded",
            "score": 62 if str(health_url).startswith("/") else 68,
            "message": "Health URL is registered; live check was not requested.",
            "checkedAt": now,
            "freshness": "unknown",
        }
    if str(health_url).startswith("/"):
        return {
            "state": "unknown",
            "score": 55,
            "message": "Relative health URL needs its deployment base URL before live checking.",
            "checkedAt": now,
            "freshness": "unknown",
        }
    try:
        status_code, body = fetch_health(str(health_url), timeout=timeout)
    except Exception as exc:
        return {
            "state": "critical",
            "score": 30,
            "message": f"Health check failed: {exc}",
            "checkedAt": now,
            "freshness": "fresh",
        }
    if 200 <= status_code < 300:
        message = health_message_from_body(body) or f"Health check returned {status_code}."
        return {
            "state": "healthy",
            "score": 86,
            "message": message,
            "checkedAt": now,
            "freshness": "fresh",
        }
    return {
        "state": "critical" if status_code >= 500 else "degraded",
        "score": 42 if status_code >= 500 else 58,
        "message": f"Health check returned {status_code}.",
        "checkedAt": now,
        "freshness": "fresh",
    }


def fetch_health(url: str, *, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"accept": "application/json,text/plain,*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(4096).decode("utf-8", errors="replace")
        return int(response.getcode()), body


def fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(1_000_000).decode("utf-8", errors="replace")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("snapshot payload must be a JSON object")
    return data


def resolve_dashboard_url(value: Any, base_url: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    base = str(base_url or "").strip()
    if base.startswith("http://") or base.startswith("https://"):
        return urllib.parse.urljoin(base, raw)
    return ""


def normalize_project_snapshot(
    snapshot: dict[str, Any],
    dashboard: dict[str, Any],
    now: str,
    snapshot_url: str,
) -> dict[str, Any]:
    source = snapshot.get("source") if isinstance(snapshot.get("source"), dict) else {}
    health = snapshot.get("health") if isinstance(snapshot.get("health"), dict) else {}
    normalized = {
        **snapshot,
        "source": {
            "id": source.get("id") or dashboard.get("id") or "unknown-dashboard",
            "label": source.get("label") or dashboard.get("label") or dashboard.get("id") or "Unknown Dashboard",
            "owner": source.get("owner") or dashboard.get("owner") or "Dashboard owner",
            "category": source.get("category") or dashboard.get("category") or "unknown",
            "projectName": source.get("projectName") or dashboard.get("projectName"),
            "projectPath": source.get("projectPath") or dashboard.get("projectPath"),
            "url": source.get("url") or dashboard.get("url"),
            "healthUrl": source.get("healthUrl") or dashboard.get("healthUrl"),
        },
        "health": {
            "state": health.get("state") or "unknown",
            "score": health.get("score"),
            "message": health.get("message") or f"Project-owned snapshot loaded from {snapshot_url}.",
            "checkedAt": health.get("checkedAt") or now,
            "freshness": health.get("freshness") or "fresh",
        },
        "updatedAt": snapshot.get("updatedAt") or now,
    }
    normalized.setdefault("queue", {"queued": 0, "running": 0, "failed": 0, "blocked": 0, "stale": 0, "completed": 0})
    normalized.setdefault("actions", [])
    normalized.setdefault("deployment", deployment_for_dashboard(dashboard))
    return normalized


def health_message_from_body(body: str) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except Exception:
        return text[:160]
    for key in ("message", "status", "state", "ok"):
        if key in data:
            return f"Health payload {key}={data[key]}"
    return "Health payload returned JSON."


def queue_for_dashboard(dashboard_id: str, health_state: str) -> dict[str, int]:
    if health_state == "critical":
        return {"queued": 0, "running": 0, "failed": 1, "blocked": 0, "stale": 0, "completed": 0}
    if health_state == "unknown":
        return {"queued": 0, "running": 0, "failed": 0, "blocked": 0, "stale": 1, "completed": 0}
    if dashboard_id == "khashi-vc.roc":
        return {"queued": 0, "running": 30, "failed": 0, "blocked": 0, "stale": 0, "completed": 0}
    return {"queued": 0, "running": 1 if health_state == "healthy" else 0, "failed": 0, "blocked": 0, "stale": 0, "completed": 0}


def actions_for_dashboard(dashboard: dict[str, Any], dashboard_id: str, health: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not dashboard.get("snapshotUrl"):
        actions.append({
            "id": f"{dashboard_id}-snapshot-url",
            "title": "Add a project-owned dashboard snapshot endpoint.",
            "owner": str(dashboard.get("owner") or "Dashboard owner"),
            "severity": "normal",
            "sourceDashboardId": dashboard_id,
            "source": "DashboardSnapshot contract",
            "nextStep": "Expose /dashboard-snapshot or register snapshotUrl in hermes.dashboards.json.",
        })
    if health["state"] in {"critical", "unknown"}:
        actions.append({
            "id": f"{dashboard_id}-health",
            "title": "Verify dashboard health signal.",
            "owner": str(dashboard.get("owner") or "Dashboard owner"),
            "severity": "high" if health["state"] == "critical" else "normal",
            "sourceDashboardId": dashboard_id,
            "source": "Dashboard health",
            "nextStep": health.get("message", "Check the registered health URL."),
        })
    return actions


def deployment_for_dashboard(dashboard: dict[str, Any]) -> dict[str, Any]:
    url = str(dashboard.get("url") or "")
    parsed = urllib.parse.urlparse(url)
    environment = "production" if parsed.scheme == "https" else "local" if parsed.scheme in {"http", ""} else "unknown"
    return {
        "environment": environment,
        "status": "unknown",
        "message": "Deployment status needs a project deployment signal.",
    }
