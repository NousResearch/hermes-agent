from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

JOB_HUNT_DIR = Path("/home/kensei/job-hunt")
QUEUE_PATH = JOB_HUNT_DIR / "cover-letter-queue.json"
COVER_LETTERS_DIR = JOB_HUNT_DIR / "cover-letters"
JOB_SCRIPTS_DIR = JOB_HUNT_DIR / "scripts"
_ACTIVE_QUEUE_STATUSES = frozenset({"queued", "needs_manual_jd", "delivery_failed"})
_ALLOWED_MODES = frozenset({"template", "ai"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_job_url(url: str) -> str:
    value = str(url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Job URL must be a full http(s) URL.")
    return value


def _normalise_mode(mode: str) -> str:
    value = str(mode or "template").strip().lower()
    if value not in _ALLOWED_MODES:
        raise ValueError("Mode must be 'template' or 'ai'.")
    return value


def _load_queue() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    try:
        data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _save_queue(items: list[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _find_active_queue_item(items: list[dict[str, Any]], url: str) -> dict[str, Any] | None:
    for item in reversed(items):
        if item.get("url") == url and item.get("status") in _ACTIVE_QUEUE_STATUSES:
            return item
    return None


def queue_cover_letter_job(
    url: str,
    *,
    mode: str,
    jd_text: str | None = None,
    requested_via: str = "gateway",
) -> tuple[dict[str, Any], bool]:
    validated_url = _validate_job_url(url)
    selected_mode = _normalise_mode(mode)
    items = _load_queue()
    existing = _find_active_queue_item(items, validated_url)
    reused_existing = existing is not None
    item = existing or {}

    item["url"] = validated_url
    item["mode"] = selected_mode
    item["queued_at"] = _utc_now_iso()
    item["status"] = "queued"
    item["requested_via"] = requested_via
    item.pop("error", None)
    item.pop("delivered_at", None)
    item.pop("cover_letter_path", None)

    if jd_text and jd_text.strip():
        item["jd_text"] = jd_text.strip()

    if not reused_existing:
        items.append(item)

    _save_queue(items)
    return dict(item), reused_existing


def attach_manual_job_description(
    url: str,
    jd_text: str,
    *,
    requested_via: str = "gateway",
    default_mode: str = "template",
) -> tuple[dict[str, Any], bool]:
    cleaned_jd = str(jd_text or "").strip()
    if not cleaned_jd:
        raise ValueError("Job description text cannot be empty.")

    validated_url = _validate_job_url(url)
    items = _load_queue()
    existing = _find_active_queue_item(items, validated_url)
    created_fresh = existing is None
    selected_mode = _normalise_mode(existing.get("mode", default_mode) if existing else default_mode)

    item, _ = queue_cover_letter_job(
        validated_url,
        mode=selected_mode,
        jd_text=cleaned_jd,
        requested_via=requested_via,
    )
    return item, created_fresh


def _job_tracker_module():
    scripts_dir = str(JOB_SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("job_tracker")


def review_cover_letter(cover_letter_ref: str, approved: bool) -> dict[str, str] | None:
    tracker = _job_tracker_module()
    return tracker.update_review_state(cover_letter_ref, approved)


def resolve_cover_letter_path(cover_letter_ref: str) -> Path:
    raw = str(cover_letter_ref or "").strip()
    if not raw:
        raise ValueError("Cover letter reference cannot be empty.")

    if raw.startswith("/home/"):
        candidate = Path(raw)
    elif raw.startswith("/job-hunt/"):
        candidate = Path("/home/kensei") / raw.lstrip("/")
    else:
        if raw != Path(raw).name:
            raise ValueError("Cover letter reference must be a filename or a /job-hunt path.")
        candidate = COVER_LETTERS_DIR / raw

    resolved = candidate.expanduser().resolve(strict=False)
    cover_dir = COVER_LETTERS_DIR.resolve(strict=False)
    if not resolved.is_relative_to(cover_dir):
        raise ValueError("Cover letter path must stay inside /home/kensei/job-hunt/cover-letters.")
    return resolved
