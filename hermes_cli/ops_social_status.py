"""Read-only social platform status data for the Jenny Ops dashboard.

This module intentionally reads a local JSON status snapshot only. It does not
call YouTube, Meta, TikTok, or any other external platform API, and it does not
write files, schedule jobs, change privacy, upload, delete, or mutate tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

DEFAULT_SOCIAL_PLATFORMS: List[Dict[str, Any]] = [
    {
        "platform": "YouTube",
        "published": None,
        "scheduled": None,
        "issues_private": "Queue reset / private check needed",
        "readiness": "Canonical upload engine, but live counts require a read-only sync.",
        "source": "Default dashboard status; no local sync file found.",
        "status": "needs_sync",
    },
    {
        "platform": "Facebook",
        "published": None,
        "scheduled": None,
        "issues_private": "Legacy old-style queue blocked",
        "readiness": "Native Reels path exists; use only approved current-quality packages.",
        "source": "Default dashboard status; no local sync file found.",
        "status": "needs_sync",
    },
    {
        "platform": "Instagram",
        "published": None,
        "scheduled": "0 known scheduler",
        "issues_private": "Immediate publish only / API readiness check",
        "readiness": "Do not call scheduling ready until a real scheduler and token check exist.",
        "source": "Default dashboard status; no local sync file found.",
        "status": "needs_sync",
    },
    {
        "platform": "TikTok",
        "published": 0,
        "scheduled": 0,
        "issues_private": "Onboarding/API not ready",
        "readiness": "Format support is not posting readiness; OAuth/app review remains gated.",
        "source": "Default dashboard status; no local sync file found.",
        "status": "blocked",
    },
]


def social_status_path() -> Path:
    """Return the profile-local social platform status snapshot path."""

    return get_hermes_home() / "state" / "ops-center" / "social-platform-status.json"


def social_status_history_path() -> Path:
    """Return the profile-local manual snapshot history JSONL path."""

    return get_hermes_home() / "state" / "ops-center" / "social-platform-status-history.jsonl"


def _display_count(value: Any) -> str:
    if value is None:
        return "Needs sync"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    return text or "Needs sync"


def _normalize_status(value: Any) -> str:
    text = str(value or "needs_sync").strip().lower().replace(" ", "_").replace("-", "_")
    allowed = {"ok", "needs_review", "blocked", "not_connected", "needs_sync"}
    return text if text in allowed else "needs_review"


def _normalize_platform(raw: Dict[str, Any], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = dict(default or {})
    base.update(raw or {})
    platform = str(base.get("platform") or "Unknown").strip() or "Unknown"
    return {
        "platform": platform,
        "published": _display_count(base.get("published")),
        "scheduled": _display_count(base.get("scheduled")),
        "issues_private": str(base.get("issues_private") or base.get("issuesPrivate") or "Needs sync").strip() or "Needs sync",
        "readiness": str(base.get("readiness") or "Read-only status only; no platform action performed.").strip(),
        "source": str(base.get("source") or "Local status snapshot").strip(),
        "status": _normalize_status(base.get("status")),
        "last_checked_at": base.get("last_checked_at"),
    }


def _merge_defaults(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {str(item.get("platform", "")).lower(): item for item in items if item.get("platform")}
    merged: List[Dict[str, Any]] = []
    for default in DEFAULT_SOCIAL_PLATFORMS:
        key = str(default["platform"]).lower()
        merged.append(_normalize_platform(by_name.pop(key, {}), default))
    for extra in by_name.values():
        merged.append(_normalize_platform(extra))
    return merged


def read_social_platform_status(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read and normalize the local social platform status snapshot.

    Missing files return conservative defaults. Invalid JSON also returns defaults
    with a warning instead of failing the dashboard, because this is an
    observability panel and must not become an execution path.
    """

    status_file = path or social_status_path()
    base: Dict[str, Any] = {
        "ok": True,
        "mode": "local_read_only",
        "path": str(status_file),
        "updated_at": None,
        "warning": None,
        "platforms": _merge_defaults([]),
    }

    if not status_file.exists():
        return base

    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact JSON error varies
        base["warning"] = f"Could not read local status snapshot: {exc}"
        return base

    platforms = data.get("platforms", data if isinstance(data, list) else [])
    if not isinstance(platforms, list):
        base["warning"] = "Local status snapshot has no list-valued platforms field."
        return base

    base["updated_at"] = data.get("updated_at") if isinstance(data, dict) else None
    base["source"] = data.get("source") if isinstance(data, dict) else None
    base["platforms"] = _merge_defaults([item for item in platforms if isinstance(item, dict)])
    return base


def _history_summary(platforms: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"ok": 0, "needs_review": 0, "blocked": 0, "not_connected": 0, "needs_sync": 0}
    for item in platforms:
        status = _normalize_status(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _append_history_event(snapshot: Dict[str, Any], history_path: Path) -> None:
    """Append a compact local-only audit event for manual status snapshots."""

    platforms = snapshot.get("platforms") if isinstance(snapshot, dict) else []
    if not isinstance(platforms, list):
        platforms = []
    event = {
        "timestamp": snapshot.get("updated_at"),
        "source": snapshot.get("source"),
        "mode": "manual_local_only",
        "platform_count": len(platforms),
        "status_counts": _history_summary([item for item in platforms if isinstance(item, dict)]),
        "platforms": [
            {
                "platform": item.get("platform"),
                "published": item.get("published"),
                "scheduled": item.get("scheduled"),
                "status": item.get("status"),
                "last_checked_at": item.get("last_checked_at"),
            }
            for item in platforms
            if isinstance(item, dict)
        ],
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def read_social_platform_history(path: Optional[Path] = None, *, limit: int = 8) -> Dict[str, Any]:
    """Read compact local-only manual snapshot history from JSONL."""

    history_file = path or social_status_history_path()
    safe_limit = max(1, min(int(limit or 8), 50))
    result: Dict[str, Any] = {
        "ok": True,
        "mode": "local_read_only",
        "path": str(history_file),
        "warning": None,
        "events": [],
    }
    if not history_file.exists():
        return result
    try:
        lines = history_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # pragma: no cover - file read errors vary
        result["warning"] = f"Could not read local status history: {exc}"
        return result

    events: List[Dict[str, Any]] = []
    for line in lines[-safe_limit:]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            events.append(item)
    result["events"] = list(reversed(events))
    return result


def _storage_platform(raw: Dict[str, Any], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a safe storage shape for the local manual snapshot."""

    normalized = _normalize_platform(raw, default)
    return {
        "platform": normalized["platform"],
        "published": normalized["published"],
        "scheduled": normalized["scheduled"],
        "issues_private": normalized["issues_private"],
        "readiness": normalized["readiness"],
        "source": normalized["source"],
        "status": normalized["status"],
        "last_checked_at": normalized.get("last_checked_at"),
    }


def write_manual_social_platform_status(
    payload: Dict[str, Any],
    path: Optional[Path] = None,
    *,
    history_path: Optional[Path] = None,
    record_history: bool = True,
) -> Dict[str, Any]:
    """Write a local manual social-platform status snapshot.

    This is intentionally local-only. It writes JSON under ``$HERMES_HOME/state``
    and never calls platform APIs, touches tokens, schedules jobs, uploads,
    deletes, or changes privacy/public state.
    """

    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    raw_platforms = payload.get("platforms")
    if not isinstance(raw_platforms, list):
        raise ValueError("Payload must include a list-valued platforms field.")
    if len(raw_platforms) > 20:
        raise ValueError("Too many platform rows; maximum is 20.")

    platforms = [item for item in raw_platforms if isinstance(item, dict)]
    if len(platforms) != len(raw_platforms):
        raise ValueError("Each platform row must be an object.")

    now = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "updated_at": str(payload.get("updated_at") or now),
        "source": str(payload.get("source") or "manual-dashboard-snapshot").strip() or "manual-dashboard-snapshot",
        "mode": "manual_local_only",
        "platforms": [_storage_platform(item) for item in _merge_defaults(platforms)],
    }

    status_file = path or social_status_path()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(status_file)
    if record_history:
        _append_history_event(snapshot, history_path or social_status_history_path())
    return read_social_platform_status(status_file)


def cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manage local-only Jenny Ops social platform status snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("read", help="Print normalized local social platform status JSON")
    history_parser = sub.add_parser("history", help="Print compact local manual snapshot history JSON")
    history_parser.add_argument("--limit", type=int, default=8, help="Maximum history events to show")
    write_parser = sub.add_parser("write", help="Write a manual local-only snapshot from a JSON file")
    write_parser.add_argument("--json-file", required=True, help="Path to a JSON payload with a platforms list")
    args = parser.parse_args(argv)

    if args.command == "read":
        print(json.dumps(read_social_platform_status(), indent=2, sort_keys=True))
        return 0
    if args.command == "history":
        print(json.dumps(read_social_platform_history(limit=args.limit), indent=2, sort_keys=True))
        return 0
    if args.command == "write":
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
        print(json.dumps(write_manual_social_platform_status(payload), indent=2, sort_keys=True))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main(sys.argv[1:]))
