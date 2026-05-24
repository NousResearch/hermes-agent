"""Local YouTube dashboard queue storage.

This module is intentionally local-state only.  It does not import or call the
YouTube API and it never stores credentials.  Publishing/upload side effects
belong behind a later explicit approval-gated integration.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hermes_constants import get_hermes_home
from utils import atomic_json_write

SCHEMA_VERSION = 1
VALID_CHANNELS = {"scripturedepth", "newslish"}
VALID_FORMATS = {"short", "long_form", "clip"}
VALID_STATUSES = {
    "idea",
    "metadata",
    "assets",
    "review",
    "ready",
    "scheduled_local",
    "published_manual",
    "archived",
}
VALID_VISIBILITIES = {"private", "unlisted", "scheduled"}
VALID_RISKS = {"low", "medium", "high"}
VALID_REVIEW_STATUSES = {"needs_review", "approved", "changes_requested", "rejected"}
MUTABLE_FIELDS = {
    "title",
    "description",
    "format",
    "status",
    "visibility",
    "scheduled_for",
    "timezone",
    "playlist",
    "tags",
    "source_refs",
    "asset_paths",
    "checks",
    "risk",
    "owner",
    "notes",
    "review_status",
    "reviewer",
    "review_notes",
}

MANIFEST_FIELDS = [
    "channel_id",
    "title",
    "format",
    "status",
    "owner",
    "description",
    "visibility",
    "scheduled_for",
    "timezone",
    "playlist",
    "tags",
    "source_refs",
    "risk",
    "notes",
    "review_status",
    "reviewer",
    "review_notes",
    "video_path",
    "thumbnail_path",
    "captions_path",
]

JSON_MANIFEST_ALLOWED_FIELDS = set(MANIFEST_FIELDS) | MUTABLE_FIELDS | {
    "id",
    "channel_id",
    "created_at",
    "updated_at",
    "external",
    "missing",
    "blocked",
    "schema_version",
}
CSV_MANIFEST_ALLOWED_FIELDS = set(MANIFEST_FIELDS) | {"id"}


_STORAGE_LOCK = threading.RLock()
_ID_SAFE_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_CHANNELS = [
    {
        "id": "scripturedepth",
        "name": "ScriptureDepth",
        "handle": "@scripturedepth",
        "default_visibility": "private",
        "playlist": "Shorts / Bible Study",
        "cadence": "1-2 Shorts/day after review",
        "voice": "reverent, clear, Bible-study useful",
        "guardrail": "Scripture claims need source/reference check before schedule.",
    },
    {
        "id": "newslish",
        "name": "Newslish",
        "handle": "@newslishapp",
        "default_visibility": "private",
        "playlist": "Explainers / Updates",
        "cadence": "1-4 day cadence; timely when evidence is fresh",
        "voice": "concise, news-style, SEO/answer optimized",
        "guardrail": "Source URLs required; no unsourced news claims.",
    },
]

DEFAULT_CHECKS = {
    "video_file": False,
    "thumbnail": False,
    "title": False,
    "description": False,
    "captions": False,
    "sources_or_scripture_refs": False,
    "fact_check": False,
    "human_approval": False,
}

DEFAULT_ASSET_PATHS: dict[str, str | None] = {
    "video": None,
    "thumbnail": None,
    "captions": None,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def storage_dir() -> Path:
    return get_hermes_home() / "youtube"


def queue_path() -> Path:
    return storage_dir() / "queue.json"


def audit_path() -> Path:
    return storage_dir() / "audit.jsonl"


def _ensure_storage_dir() -> None:
    path = storage_dir()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_iso(),
        "channels": deepcopy(DEFAULT_CHANNELS),
        "items": [],
    }


def _write_state(state: Dict[str, Any]) -> None:
    _ensure_storage_dir()
    state["updated_at"] = now_iso()
    atomic_json_write(queue_path(), state)
    try:
        os.chmod(queue_path(), 0o600)
    except OSError:
        pass


def load_state() -> Dict[str, Any]:
    with _STORAGE_LOCK:
        path = queue_path()
        if not path.exists():
            state = _empty_state()
            _write_state(state)
            return state
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"YouTube queue state is invalid JSON: {exc}") from exc
        return _normalize_state(raw)


def _normalize_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(raw or {})
    state.setdefault("schema_version", SCHEMA_VERSION)
    state.setdefault("updated_at", now_iso())
    state["channels"] = deepcopy(DEFAULT_CHANNELS)
    items = state.get("items")
    if not isinstance(items, list):
        items = []
    state["items"] = [_derive_item(item) for item in items if isinstance(item, dict)]
    return state


def _channel_playlist(channel_id: str) -> str:
    for channel in DEFAULT_CHANNELS:
        if channel["id"] == channel_id:
            return str(channel["playlist"])
    return ""


def _slugify(text: str) -> str:
    slug = _ID_SAFE_RE.sub("-", text.lower()).strip("-")[:40]
    return slug or "video"


def _new_item_id(channel_id: str, title: str) -> str:
    return f"{channel_id}-{_slugify(title)}-{uuid.uuid4().hex[:8]}"


def _validate_str(value: Any, field: str, *, max_len: int, required: bool = False) -> str:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{field} is required")
    if len(value) > max_len:
        raise ValueError(f"{field} must be <= {max_len} characters")
    return value


def _validate_enum(value: Any, field: str, allowed: set[str], default: str | None = None) -> str:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"{field} is required")
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in allowed:
        raise ValueError(f"invalid {field}: {value}")
    return normalized


def _validate_str_list(value: Any, field: str, *, max_items: int, max_len: int) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{field} must have <= {max_items} items")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} items must be strings")
        text = item.strip()
        if text:
            if len(text) > max_len:
                raise ValueError(f"{field} items must be <= {max_len} characters")
            out.append(text)
    return out


def _validate_checks(value: Any) -> dict[str, bool]:
    checks = dict(DEFAULT_CHECKS)
    if value in (None, ""):
        return checks
    if not isinstance(value, dict):
        raise ValueError("checks must be an object")
    for key, raw in value.items():
        if key not in DEFAULT_CHECKS:
            raise ValueError(f"unknown check: {key}")
        if not isinstance(raw, bool):
            raise ValueError(f"checks.{key} must be a boolean")
        checks[key] = raw
    return checks


def _validate_asset_paths(value: Any) -> dict[str, str | None]:
    paths = dict(DEFAULT_ASSET_PATHS)
    if value in (None, ""):
        return paths
    if not isinstance(value, dict):
        raise ValueError("asset_paths must be an object")
    for key, raw in value.items():
        if key not in DEFAULT_ASSET_PATHS:
            raise ValueError(f"unknown asset path: {key}")
        if raw in (None, ""):
            paths[key] = None
        elif isinstance(raw, str):
            text = raw.strip()
            if len(text) > 500:
                raise ValueError(f"asset_paths.{key} is too long")
            paths[key] = text
        else:
            raise ValueError(f"asset_paths.{key} must be a string or null")
    return paths


def _is_valid_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _asset_path_blocker(key: str, raw_path: Any) -> str | None:
    if not raw_path:
        return f"asset_paths.{key} is required"
    path_text = str(raw_path).strip()
    if not path_text:
        return f"asset_paths.{key} is required"
    path = Path(path_text).expanduser()
    if not path.exists():
        return f"asset_paths.{key} does not exist: {path_text}"
    if not path.is_file():
        return f"asset_paths.{key} is not a readable file: {path_text}"
    if not os.access(path, os.R_OK):
        return f"asset_paths.{key} is not readable: {path_text}"
    return None


def _validate_schedule_fields(item: Dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if item.get("visibility") != "scheduled":
        return blockers
    scheduled_for = item.get("scheduled_for")
    timezone_name = item.get("timezone")
    if not scheduled_for:
        blockers.append("scheduled visibility requires scheduled_for")
    else:
        try:
            datetime.fromisoformat(str(scheduled_for).replace("Z", "+00:00"))
        except ValueError:
            blockers.append("scheduled_for must be an ISO datetime")
    if not timezone_name:
        blockers.append("scheduled visibility requires timezone")
    else:
        try:
            ZoneInfo(str(timezone_name))
        except ZoneInfoNotFoundError:
            blockers.append("timezone must be a valid IANA timezone")
    return blockers


def _structural_readiness_blockers(item: Dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    checks = item.get("checks") or {}
    assets = item.get("asset_paths") or {}
    title = str(item.get("title") or "").strip()
    description = str(item.get("description") or "").strip()

    if not title or title == "Untitled video":
        blockers.append("title text is required")
    if not description:
        blockers.append("description text is required")

    required_assets = ["video", "thumbnail"]
    if item.get("format") == "short":
        required_assets.append("captions")
    for key in required_assets:
        blocker = _asset_path_blocker(key, assets.get(key))
        if blocker:
            blockers.append(blocker)

    if checks.get("video_file") and not assets.get("video"):
        blockers.append("video_file check requires asset_paths.video")
    if checks.get("thumbnail") and not assets.get("thumbnail"):
        blockers.append("thumbnail check requires asset_paths.thumbnail")
    if checks.get("captions") and item.get("format") == "short" and not assets.get("captions"):
        blockers.append("captions check requires asset_paths.captions for Shorts")

    blockers.extend(_validate_schedule_fields(item))

    if item.get("channel_id") == "newslish":
        refs = item.get("source_refs") or []
        invalid = [ref for ref in refs if not _is_valid_url(ref)]
        if invalid:
            blockers.append("Newslish source_refs must be http(s) URLs")
    return blockers


def _derive_missing(item: Dict[str, Any]) -> list[str]:
    checks = item.get("checks") or {}
    missing: list[str] = []
    common = ["video_file", "thumbnail", "title", "description", "human_approval"]
    if item.get("format") == "short":
        common.append("captions")
    for key in common:
        if not checks.get(key):
            missing.append(key)
    if item.get("channel_id") == "scripturedepth" and not checks.get("sources_or_scripture_refs"):
        missing.append("scripture/source refs")
    if item.get("channel_id") == "newslish":
        if not checks.get("sources_or_scripture_refs"):
            missing.append("source URLs")
        if not checks.get("fact_check"):
            missing.append("fact check")
    return missing


def _derive_risk(item: Dict[str, Any]) -> str:
    if item.get("risk") in VALID_RISKS:
        return item["risk"]
    missing = _derive_missing(item)
    if "fact check" in missing or "scripture/source refs" in missing:
        return "high"
    if missing:
        return "medium"
    return "low"


def _derive_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(raw)
    channel_id = _validate_enum(item.get("channel_id"), "channel_id", VALID_CHANNELS, "scripturedepth")
    title = _validate_str(item.get("title"), "title", max_len=180, required=True) if item.get("title") else "Untitled video"
    item["id"] = _validate_str(item.get("id"), "id", max_len=120) or _new_item_id(channel_id, title)
    item["channel_id"] = channel_id
    item["title"] = title
    item["description"] = _validate_str(item.get("description"), "description", max_len=5000)
    item["format"] = _validate_enum(item.get("format"), "format", VALID_FORMATS, "short")
    item["status"] = _validate_enum(item.get("status"), "status", VALID_STATUSES, "idea")
    item["visibility"] = _validate_enum(item.get("visibility"), "visibility", VALID_VISIBILITIES, "private")
    item["scheduled_for"] = _validate_str(item.get("scheduled_for"), "scheduled_for", max_len=120) or None
    item["timezone"] = _validate_str(item.get("timezone"), "timezone", max_len=80) or None
    item["playlist"] = _validate_str(item.get("playlist"), "playlist", max_len=160) or _channel_playlist(channel_id)
    item["tags"] = _validate_str_list(item.get("tags"), "tags", max_items=30, max_len=80)
    item["source_refs"] = _validate_str_list(item.get("source_refs"), "source_refs", max_items=30, max_len=500)
    item["asset_paths"] = _validate_asset_paths(item.get("asset_paths"))
    item["checks"] = _validate_checks(item.get("checks"))
    item["risk"] = _validate_enum(item.get("risk"), "risk", VALID_RISKS, _derive_risk(item))
    item["owner"] = _validate_str(item.get("owner"), "owner", max_len=80) or "Hermes"
    item["notes"] = _validate_str(item.get("notes"), "notes", max_len=2000)
    item["review_status"] = _validate_enum(item.get("review_status"), "review_status", VALID_REVIEW_STATUSES, "needs_review")
    item["reviewer"] = _validate_str(item.get("reviewer"), "reviewer", max_len=80)
    item["review_notes"] = _validate_str(item.get("review_notes"), "review_notes", max_len=4000)
    raw_external = item.get("external")
    external = raw_external if isinstance(raw_external, dict) else {}
    item["external"] = {
        "youtube_video_id": external.get("youtube_video_id"),
        "published_at": external.get("published_at"),
    }
    item.setdefault("created_at", now_iso())
    item["updated_at"] = item.get("updated_at") or now_iso()
    item["missing"] = _derive_missing(item)
    item["blocked"] = bool(item["missing"]) or item["risk"] == "high"
    return item


def _validate_ready_transition(item: Dict[str, Any]) -> None:
    if item.get("status") in {"ready", "scheduled_local", "published_manual"}:
        if _derive_missing(item):
            raise ValueError("item cannot be marked ready/scheduled while required checks are missing")
        structural_blockers = _structural_readiness_blockers(item)
        if structural_blockers:
            raise ValueError(f"item cannot be marked ready/scheduled: {structural_blockers[0]}")
        if item.get("review_status") != "approved":
            raise ValueError("item cannot be marked ready/scheduled until review_status is approved")


def _split_manifest_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [part.strip() for part in str(value).replace("\n", ";").split(";") if part.strip()]


def _reject_unknown_manifest_fields(row: Dict[str, Any], allowed: set[str], *, row_number: int | None = None) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        prefix = f"row {row_number}: " if row_number is not None else ""
        raise ValueError(f"{prefix}unknown manifest field(s): {', '.join(unknown)}")


def _manifest_row_to_item(row: Dict[str, Any], *, allowed_fields: set[str] = JSON_MANIFEST_ALLOWED_FIELDS, row_number: int | None = None) -> Dict[str, Any]:
    _reject_unknown_manifest_fields(row, allowed_fields, row_number=row_number)
    asset_paths = _validate_asset_paths(row.get("asset_paths")) if "asset_paths" in row else dict(DEFAULT_ASSET_PATHS)
    for source_key, target_key in (
        ("video_path", "video"),
        ("thumbnail_path", "thumbnail"),
        ("captions_path", "captions"),
    ):
        raw = row.get(source_key)
        if raw:
            asset_paths[target_key] = str(raw).strip()

    checks = {}
    if row.get("title"):
        checks["title"] = True
    if row.get("description"):
        checks["description"] = True
    if asset_paths.get("video"):
        checks["video_file"] = True
    if asset_paths.get("thumbnail"):
        checks["thumbnail"] = True
    if asset_paths.get("captions"):
        checks["captions"] = True
    if row.get("source_refs"):
        checks["sources_or_scripture_refs"] = True

    raw_checks = row.get("checks")
    row_checks: Dict[str, Any] = raw_checks if isinstance(raw_checks, dict) else {}
    item = {
        "id": row.get("id"),
        "channel_id": row.get("channel_id"),
        "title": row.get("title"),
        "description": row.get("description", ""),
        "format": row.get("format", "short"),
        "status": row.get("status", "idea"),
        "visibility": row.get("visibility", "private"),
        "scheduled_for": row.get("scheduled_for"),
        "timezone": row.get("timezone"),
        "playlist": row.get("playlist"),
        "tags": _split_manifest_list(row.get("tags")),
        "source_refs": _split_manifest_list(row.get("source_refs")),
        "asset_paths": asset_paths,
        "checks": {**checks, **row_checks},
        "risk": row.get("risk", "medium"),
        "owner": row.get("owner", "Hermes"),
        "notes": row.get("notes", ""),
        "review_status": row.get("review_status", "needs_review"),
        "reviewer": row.get("reviewer", ""),
        "review_notes": row.get("review_notes", ""),
        "created_at": row.get("created_at"),
    }
    return {key: value for key, value in item.items() if value is not None}


def _parse_manifest(content: str, fmt: str) -> list[Dict[str, Any]]:
    normalized = fmt.strip().lower()
    if normalized == "json":
        raw = json.loads(content)
        if isinstance(raw, dict):
            raw_items = raw.get("items")
        else:
            raw_items = raw
        if not isinstance(raw_items, list):
            raise ValueError("JSON manifest must be an array or an object with items[]")
        rows = []
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"row {index}: JSON manifest items must be objects")
            rows.append(_manifest_row_to_item(item, row_number=index))
        return rows
    if normalized == "csv":
        reader = csv.DictReader(io.StringIO(content))
        if reader.fieldnames is None:
            raise ValueError("CSV manifest must include a header row")
        headers = {str(field).strip() for field in reader.fieldnames if field}
        _reject_unknown_manifest_fields({field: None for field in headers}, CSV_MANIFEST_ALLOWED_FIELDS)
        rows = []
        for index, row in enumerate(reader, start=1):
            clean = {str(key).strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items() if key}
            rows.append(_manifest_row_to_item(clean, allowed_fields=CSV_MANIFEST_ALLOWED_FIELDS, row_number=index))
        return rows
    raise ValueError("manifest format must be json or csv")


def manifest_template(fmt: str = "csv") -> Dict[str, Any]:
    rows = [
        {
            "channel_id": "scripturedepth",
            "title": "Psalm 23 hope in one minute",
            "format": "short",
            "status": "idea",
            "owner": "Hermes",
            "description": "Working description goes here.",
            "visibility": "private",
            "scheduled_for": "",
            "timezone": "Europe/Oslo",
            "playlist": "Shorts / Bible Study",
            "tags": "Bible;Psalm 23;Shorts",
            "source_refs": "Psalm 23",
            "risk": "medium",
            "notes": "Needs human approval before upload.",
            "review_status": "needs_review",
            "reviewer": "",
            "review_notes": "",
            "video_path": "",
            "thumbnail_path": "",
            "captions_path": "",
        },
        {
            "channel_id": "newslish",
            "title": "What changed today in one minute",
            "format": "short",
            "status": "idea",
            "owner": "Hermes",
            "description": "Working description goes here.",
            "visibility": "private",
            "scheduled_for": "",
            "timezone": "Europe/Oslo",
            "playlist": "Explainers / Updates",
            "tags": "news;explainer;shorts",
            "source_refs": "https://example.com/source",
            "risk": "medium",
            "notes": "Source and fact-check required before ready.",
            "review_status": "needs_review",
            "reviewer": "",
            "review_notes": "",
            "video_path": "",
            "thumbnail_path": "",
            "captions_path": "",
        },
    ]
    normalized = fmt.strip().lower()
    if normalized == "json":
        return {"format": "json", "content": json.dumps({"items": rows}, indent=2, ensure_ascii=False)}
    if normalized != "csv":
        raise ValueError("template format must be json or csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return {"format": "csv", "content": output.getvalue()}


def export_manifest(fmt: str = "json", include_archived: bool = False) -> Dict[str, Any]:
    state = dashboard_state()
    items = [item for item in state["items"] if include_archived or item.get("status") != "archived"]
    normalized = fmt.strip().lower()
    if normalized == "json":
        return {
            "format": "json",
            "content": json.dumps({"schema_version": SCHEMA_VERSION, "exported_at": now_iso(), "items": items}, indent=2, ensure_ascii=False),
        }
    if normalized != "csv":
        raise ValueError("export format must be json or csv")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    for item in items:
        assets = item.get("asset_paths") or {}
        writer.writerow({
            "channel_id": item.get("channel_id", ""),
            "title": item.get("title", ""),
            "format": item.get("format", ""),
            "status": item.get("status", ""),
            "owner": item.get("owner", ""),
            "description": item.get("description", ""),
            "visibility": item.get("visibility", ""),
            "scheduled_for": item.get("scheduled_for") or "",
            "timezone": item.get("timezone") or "",
            "playlist": item.get("playlist", ""),
            "tags": ";".join(item.get("tags") or []),
            "source_refs": ";".join(item.get("source_refs") or []),
            "risk": item.get("risk", ""),
            "notes": item.get("notes", ""),
            "review_status": item.get("review_status", "needs_review"),
            "reviewer": item.get("reviewer", ""),
            "review_notes": item.get("review_notes", ""),
            "video_path": assets.get("video") or "",
            "thumbnail_path": assets.get("thumbnail") or "",
            "captions_path": assets.get("captions") or "",
        })
    return {"format": "csv", "content": output.getvalue()}


def import_manifest(content: str, fmt: str, request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rows = _parse_manifest(content, fmt)
    if len(rows) > 500:
        raise ValueError("manifest import is limited to 500 rows")
    created: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    with _STORAGE_LOCK:
        state = load_state()
        existing_ids = {str(item.get("id")) for item in state.get("items", []) if item.get("id")}
        existing_titles = {
            (str(item.get("channel_id") or "").strip().lower(), str(item.get("title") or "").strip().lower())
            for item in state.get("items", [])
            if item.get("status") != "archived" and item.get("title")
        }
        for index, row in enumerate(rows, start=1):
            try:
                item = _derive_item({**row, "created_at": row.get("created_at") or now_iso()})
                item_id = str(item.get("id") or "")
                title_key = (str(item.get("channel_id") or "").strip().lower(), str(item.get("title") or "").strip().lower())
                if item_id in seen_ids or item_id in existing_ids:
                    raise ValueError(f"duplicate queue item id: {item_id}")
                if title_key in seen_titles or title_key in existing_titles:
                    raise ValueError(f"duplicate queue title for channel {item['channel_id']}: {item['title']}")
                _validate_ready_transition(item)
            except Exception as exc:
                errors.append({"row": index, "title": row.get("title"), "error": str(exc)})
                continue
            seen_ids.add(str(item["id"]))
            seen_titles.add((str(item.get("channel_id") or "").strip().lower(), str(item.get("title") or "").strip().lower()))
            created.append(item)
        if created:
            state["items"] = created + state["items"]
            _write_state(state)
            _append_audit(
                "queue.batch_import",
                item_id=None,
                channel_id=None,
                changes={"created_ids": [item["id"] for item in created], "errors": len(errors)},
                request=request,
            )
    return {"created": created, "errors": errors, "created_count": len(created), "error_count": len(errors)}


def _append_audit(action: str, *, item_id: str | None, channel_id: str | None, changes: Dict[str, Any], request: Dict[str, Any] | None = None) -> None:
    _ensure_storage_dir()
    event = {
        "id": uuid.uuid4().hex,
        "at": now_iso(),
        "actor": "dashboard",
        "action": action,
        "item_id": item_id,
        "channel_id": channel_id,
        "changes": changes,
        "request": request or {},
    }
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    path = audit_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _changed(before: Dict[str, Any] | None, after: Dict[str, Any]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for key, value in after.items():
        if key in {"updated_at", "missing", "blocked"}:
            continue
        old = None if before is None else before.get(key)
        if old != value:
            changes[key] = [old, value]
    return changes


def _summarize(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    rows = list(items)
    return {
        "total": len([i for i in rows if i.get("status") != "archived"]),
        "ready": len([i for i in rows if i.get("status") in {"ready", "scheduled_local"}]),
        "blocked": len([i for i in rows if i.get("status") != "archived" and i.get("blocked")]),
        "needs_approval": len([i for i in rows if i.get("status") != "archived" and not (i.get("checks") or {}).get("human_approval")]),
        "review_changes_requested": len([i for i in rows if i.get("status") != "archived" and i.get("review_status") == "changes_requested"]),
        "review_rejected": len([i for i in rows if i.get("status") != "archived" and i.get("review_status") == "rejected"]),
        "archived": len([i for i in rows if i.get("status") == "archived"]),
    }


def dashboard_state() -> Dict[str, Any]:
    state = load_state()
    items = [_derive_item(item) for item in state.get("items", [])]
    return {
        "channels": state["channels"],
        "items": items,
        "summary": _summarize(items),
        "capabilities": {
            "local_queue": True,
            "youtube_publish": False,
            "youtube_analytics": False,
        },
        "updated_at": state.get("updated_at"),
        "schema_version": state.get("schema_version", SCHEMA_VERSION),
    }


def create_item(data: Dict[str, Any], request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with _STORAGE_LOCK:
        state = load_state()
        item = _derive_item({
            "channel_id": data.get("channel_id"),
            "title": data.get("title"),
            "description": data.get("description", ""),
            "format": data.get("format", "short"),
            "status": data.get("status", "idea"),
            "visibility": data.get("visibility", "private"),
            "scheduled_for": data.get("scheduled_for"),
            "timezone": data.get("timezone"),
            "playlist": data.get("playlist") or _channel_playlist(str(data.get("channel_id", "scripturedepth"))),
            "tags": data.get("tags", []),
            "source_refs": data.get("source_refs", []),
            "asset_paths": data.get("asset_paths", {}),
            "checks": data.get("checks", {}),
            "risk": data.get("risk", "medium"),
            "owner": data.get("owner", "Hermes"),
            "notes": data.get("notes", ""),
            "review_status": data.get("review_status", "needs_review"),
            "reviewer": data.get("reviewer", ""),
            "review_notes": data.get("review_notes", ""),
            "created_at": now_iso(),
        })
        _validate_ready_transition(item)
        state["items"].insert(0, item)
        _write_state(state)
        _append_audit("queue.create", item_id=item["id"], channel_id=item["channel_id"], changes=_changed(None, item), request=request)
        return item


def patch_item(item_id: str, updates: Dict[str, Any], request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with _STORAGE_LOCK:
        state = load_state()
        for idx, item in enumerate(state["items"]):
            if item.get("id") != item_id:
                continue
            before = _derive_item(item)
            unknown = set(updates) - MUTABLE_FIELDS
            if unknown:
                raise ValueError(f"unsupported update fields: {', '.join(sorted(unknown))}")
            candidate = {**before, **updates, "updated_at": now_iso()}
            candidate = _derive_item(candidate)
            _validate_ready_transition(candidate)
            state["items"][idx] = candidate
            _write_state(state)
            _append_audit("queue.patch", item_id=item_id, channel_id=candidate["channel_id"], changes=_changed(before, candidate), request=request)
            return candidate
        raise KeyError(item_id)


def archive_item(item_id: str, request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return patch_item(item_id, {"status": "archived"}, request=request)


def bulk_update(item_ids: list[str], updates: Dict[str, Any], request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not item_ids:
        raise ValueError("item_ids is required")
    if len(item_ids) > 200:
        raise ValueError("bulk update is limited to 200 items")
    unknown = set(updates) - MUTABLE_FIELDS
    if unknown:
        raise ValueError(f"unsupported update fields: {', '.join(sorted(unknown))}")
    id_set = set(item_ids)
    updated: list[Dict[str, Any]] = []
    errors: list[Dict[str, str]] = []
    with _STORAGE_LOCK:
        state = load_state()
        found: set[str] = set()
        for idx, item in enumerate(state["items"]):
            item_id = str(item.get("id", ""))
            if item_id not in id_set:
                continue
            found.add(item_id)
            before = _derive_item(item)
            try:
                candidate = _derive_item({**before, **updates, "updated_at": now_iso()})
                _validate_ready_transition(candidate)
            except Exception as exc:
                errors.append({"item_id": item_id, "error": str(exc)})
                continue
            state["items"][idx] = candidate
            updated.append(candidate)
            _append_audit("queue.bulk_patch.item", item_id=item_id, channel_id=candidate["channel_id"], changes=_changed(before, candidate), request=request)
        for missing_id in sorted(id_set - found):
            errors.append({"item_id": missing_id, "error": "Queue item not found"})
        if updated:
            _write_state(state)
            _append_audit(
                "queue.bulk_patch",
                item_id=None,
                channel_id=None,
                changes={"updated_ids": [item["id"] for item in updated], "errors": len(errors), "updates": updates},
                request=request,
            )
    return {"updated": updated, "errors": errors, "updated_count": len(updated), "error_count": len(errors)}


def bulk_archive(item_ids: list[str], request: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return bulk_update(item_ids, {"status": "archived"}, request=request)


def get_item(item_id: str) -> Dict[str, Any]:
    for item in dashboard_state()["items"]:
        if item.get("id") == item_id:
            return item
    raise KeyError(item_id)


def _channel_config(channel_id: str) -> Dict[str, Any]:
    for channel in DEFAULT_CHANNELS:
        if channel["id"] == channel_id:
            return deepcopy(channel)
    return {}


def _readiness_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    derived = _derive_item(item)
    checks = derived.get("checks") or {}
    blockers: list[str] = []
    warnings: list[str] = []
    channel_rule_checks: list[Dict[str, Any]] = []

    if derived.get("status") == "archived":
        blockers.append("item is archived")
    if derived.get("review_status") != "approved":
        blockers.append("review_status must be approved")
    if derived.get("risk") == "high":
        blockers.append("risk is high")
    for missing in derived.get("missing") or []:
        blockers.append(f"missing required check: {missing}")
    blockers.extend(_structural_readiness_blockers(derived))

    if derived.get("visibility") != "private":
        warnings.append("private-first upload is preferred until OAuth and rollback paths are proven")
    if not derived.get("tags"):
        warnings.append("tags are empty")
    if not derived.get("playlist"):
        warnings.append("playlist is empty")
    assets = derived.get("asset_paths") or {}
    if checks.get("video_file") and not assets.get("video"):
        warnings.append("video_file check is marked done but asset_paths.video is empty")
    if checks.get("thumbnail") and not assets.get("thumbnail"):
        warnings.append("thumbnail check is marked done but asset_paths.thumbnail is empty")

    if derived.get("channel_id") == "scripturedepth":
        scripture_refs_ok = bool(derived.get("source_refs")) and checks.get("sources_or_scripture_refs")
        channel_rule_checks.extend([
            {"id": "scripture_refs", "label": "Scripture/source references present", "ok": scripture_refs_ok},
            {"id": "motion_uniqueness", "label": "Human confirms fresh motion/no reused static/icon/fake-text clips", "ok": bool(checks.get("human_approval"))},
            {"id": "shorts_metadata", "label": "Shorts-ready title/description/captions", "ok": bool(derived.get("title") and derived.get("description") and checks.get("title") and checks.get("description") and (derived.get("format") != "short" or checks.get("captions")))},
        ])
        if not scripture_refs_ok:
            blockers.append("ScriptureDepth requires scripture/source references")
        if derived.get("format") != "short":
            warnings.append("ScriptureDepth lane is currently optimized for Shorts")
    elif derived.get("channel_id") == "newslish":
        source_urls_ok = bool(derived.get("source_refs")) and all(_is_valid_url(ref) for ref in (derived.get("source_refs") or [])) and checks.get("sources_or_scripture_refs")
        fact_check_ok = bool(checks.get("fact_check"))
        channel_rule_checks.extend([
            {"id": "source_urls", "label": "Source URLs present", "ok": source_urls_ok},
            {"id": "fact_check", "label": "Fact-check complete", "ok": fact_check_ok},
            {"id": "seo_metadata", "label": "SEO/answer-oriented title and description", "ok": bool(derived.get("title") and derived.get("description") and checks.get("title") and checks.get("description"))},
        ])
        if not source_urls_ok:
            blockers.append("Newslish requires source URLs")
        if not fact_check_ok:
            blockers.append("Newslish requires fact check")

    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    return {
        "item_id": derived["id"],
        "ready": len(blockers) == 0,
        "publish_enabled": False,
        "youtube_side_effects_enabled": False,
        "blockers": blockers,
        "warnings": warnings,
        "required_checks": derived.get("missing") or [],
        "channel_rule_checks": channel_rule_checks,
        "review_status": derived.get("review_status"),
        "risk": derived.get("risk"),
        "capability_reason": "YouTube publishing is disabled until OAuth, quota, audit, and final approval gates are implemented.",
    }


def publish_readiness(item_id: str) -> Dict[str, Any]:
    return _readiness_for_item(get_item(item_id))


def publish_plan(item_id: str) -> Dict[str, Any]:
    item = get_item(item_id)
    readiness = _readiness_for_item(item)
    channel = _channel_config(str(item.get("channel_id")))
    assets = item.get("asset_paths") or {}
    payload_preview = {
        "channel_id": item.get("channel_id"),
        "channel_name": channel.get("name", item.get("channel_id")),
        "channel_handle": channel.get("handle"),
        "title": item.get("title"),
        "description": item.get("description"),
        "tags": item.get("tags") or [],
        "playlist": item.get("playlist"),
        "visibility": item.get("visibility"),
        "scheduled_for": item.get("scheduled_for"),
        "timezone": item.get("timezone"),
        "video_path": assets.get("video"),
        "thumbnail_path": assets.get("thumbnail"),
        "captions_path": assets.get("captions"),
        "source_refs": item.get("source_refs") or [],
        "category": "Education" if item.get("channel_id") == "scripturedepth" else "News & Politics",
        "made_for_kids": False,
    }
    return {
        "item_id": item["id"],
        "generated_at": now_iso(),
        "publish_enabled": False,
        "youtube_api_call_allowed": False,
        "side_effects": [],
        "readiness": readiness,
        "payload_preview": payload_preview,
        "safety_note": "Dry run only. No YouTube API upload, schedule, metadata update, or publish action is executed by this endpoint.",
    }


def read_audit(limit: int = 100, item_id: str | None = None) -> Dict[str, Any]:
    path = audit_path()
    if not path.exists():
        return {"events": []}
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item_id and event.get("item_id") != item_id:
            continue
        events.append(event)
    events = events[-max(1, min(limit, 500)):]
    events.reverse()
    return {"events": events}
