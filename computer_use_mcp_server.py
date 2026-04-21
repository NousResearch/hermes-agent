"""Hermes-adapted Codex-style computer-use MCP server.

This is not a direct embed of OpenAI's proprietary computer-use plugin.
Instead, it exposes a compatible-ish MCP surface backed by Hermes' local
macOS computer_control backend where possible, and returns explicit
"not implemented" results for actions Hermes does not support yet.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - runtime import guard
    FastMCP = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home
from tools.computer_control_tool import computer_control, frontmost_window_state


def _decode(payload: str) -> dict[str, Any]:
    data = json.loads(payload)
    assert isinstance(data, dict)
    return data


_SESSION_ID = f"session-{uuid.uuid4().hex}"
_APP_SESSIONS: dict[str, dict[str, Any]] = {}
_APP_SESSIONS_LOCK = threading.RLock()
_APPROVALS_LOCK = threading.RLock()


def _normalize_app_name(app_name: str | None) -> str:
    return str(app_name or "").strip()


def _session_key(app_name: str | None) -> str:
    normalized = _normalize_app_name(app_name) or "desktop"
    return normalized.casefold()


def _approval_store_path() -> Path:
    root = get_hermes_home() / "computer-use"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ComputerUseAppApprovals.json"


def _optional_dict(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = record.get(key)
    if isinstance(value, dict):
        return dict(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounds_close(left: Any, right: Any, *, position_tolerance: int = 16, size_tolerance: int = 16) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    lx = _coerce_int(left.get("x"))
    ly = _coerce_int(left.get("y"))
    lw = _coerce_int(left.get("width"))
    lh = _coerce_int(left.get("height"))
    rx = _coerce_int(right.get("x"))
    ry = _coerce_int(right.get("y"))
    rw = _coerce_int(right.get("width"))
    rh = _coerce_int(right.get("height"))
    if None in (lx, ly, lw, lh, rx, ry, rw, rh):
        return False
    return (
        abs(lx - rx) <= position_tolerance
        and abs(ly - ry) <= position_tolerance
        and abs(lw - rw) <= size_tolerance
        and abs(lh - rh) <= size_tolerance
    )


def _frontmost_states_match(base: dict[str, Any], candidate: dict[str, Any]) -> bool:
    base_bundle = str(base.get("bundle_id") or "").strip()
    candidate_bundle = str(candidate.get("bundle_id") or "").strip()
    if base_bundle and candidate_bundle and base_bundle != candidate_bundle:
        return False

    base_app = _normalize_app_name(base.get("app_name"))
    candidate_app = _normalize_app_name(candidate.get("app_name"))
    if base_app and candidate_app and base_app.casefold() != candidate_app.casefold():
        return False

    base_window_id = _coerce_int(base.get("window_id"))
    candidate_window_id = _coerce_int(candidate.get("window_id"))
    if base_window_id is not None and candidate_window_id is not None:
        return base_window_id == candidate_window_id

    base_title = str(base.get("window_title") or "").strip()
    candidate_title = str(candidate.get("window_title") or "").strip()
    if base_title and candidate_title and base_title != candidate_title:
        return False

    base_bounds = base.get("window_bounds")
    candidate_bounds = candidate.get("window_bounds")
    bounds_match = _bounds_close(base_bounds, candidate_bounds)
    if isinstance(base_bounds, dict) and isinstance(candidate_bounds, dict):
        return bounds_match

    if base_title and candidate_title and base_title == candidate_title:
        return True

    return bounds_match


def _session_payload(record: dict[str, Any]) -> dict[str, Any]:
    cursor = dict(record.get("virtual_cursor") or {"x": None, "y": None, "detached": True, "visible": True})
    overlay_path = str(record.get("overlay_screenshot_path", "") or "")
    session_state_path = str(record.get("session_state_path", "") or "")
    pending_pointer_action = _optional_dict(record, "pending_pointer_action")
    last_pointer_action_result = _optional_dict(record, "last_pointer_action_result")
    return {
        "app_name": record["app_name"],
        "app_session_id": record["app_session_id"],
        "active": bool(record.get("active")),
        "approved": bool(record.get("approved")),
        "approval_scope": str(record.get("temporary_approval_scope") or ("always" if record.get("approved") else "")),
        "virtual_cursor": cursor,
        "bundle_id": record.get("bundle_id", ""),
        "process_id": record.get("process_id"),
        "window_id": record.get("window_id"),
        "window_bounds": record.get("window_bounds"),
        "window_title": record.get("window_title", ""),
        "accessibility_tree": record.get("accessibility_tree") or [],
        "screenshot_path": record.get("screenshot_path", ""),
        "overlay_screenshot_path": overlay_path,
        "overlay_media_tag": _media_tag_for_path(overlay_path),
        "session_state_path": session_state_path,
        "pending_pointer_action": pending_pointer_action,
        "last_pointer_action_result": last_pointer_action_result,
    }


def _fresh_virtual_cursor() -> dict[str, Any]:
    return {"x": None, "y": None, "detached": True, "visible": True}


_PENDING_POINTER_CLAIM_TTL_SECONDS = 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


_PIXEL_CURSOR_SCALE = 4
_PIXEL_CURSOR_COLORS = {
    "1": "#0f172a",
    "2": "#8b5cf6",
    "3": "#c4b5fd",
    "4": "#22d3ee",
}
_PIXEL_CURSOR_ROWS = [
    "1.......",
    "12......",
    "123.....",
    "1232....",
    "12322...",
    "123222..",
    "1232222.",
    "12333321",
    "12324...",
    "121.21..",
    "1...21..",
    "....11..",
]


def _pixel_cursor_draw_args(x: int, y: int) -> list[str]:
    scale = _PIXEL_CURSOR_SCALE
    draw_args = [
        "-fill", "rgba(139,92,246,0.14)",
        "-draw", f"rectangle {x - scale},{y - scale} {x + scale - 1},{y + scale - 1}",
        "-fill", "rgba(34,211,238,0.16)",
        "-draw", f"rectangle {x + (6 * scale)},{y + (7 * scale)} {x + (7 * scale) - 1},{y + (8 * scale) - 1}",
    ]
    for row_index, row in enumerate(_PIXEL_CURSOR_ROWS):
        for col_index, token in enumerate(row):
            color = _PIXEL_CURSOR_COLORS.get(token)
            if not color:
                continue
            left = x + (col_index * scale)
            top = y + (row_index * scale)
            right = left + scale - 1
            bottom = top + scale - 1
            draw_args.extend([
                "-fill", color,
                "-draw", f"rectangle {left},{top} {right},{bottom}",
            ])
    return draw_args


def _overlay_root() -> Path:
    root = get_hermes_home() / "computer-use" / "overlay-previews"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_state_root() -> Path:
    root = get_hermes_home() / "computer-use" / "session-state"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _overlay_preview_path(session: dict[str, Any]) -> Path:
    app_session_id = str(session.get("app_session_id") or uuid.uuid4().hex)
    return _overlay_root() / f"{app_session_id}-cursor-preview.png"


def _session_state_path(session: dict[str, Any]) -> Path:
    app_session_id = str(session.get("app_session_id") or uuid.uuid4().hex)
    return _session_state_root() / f"{app_session_id}.json"


def _media_tag_for_path(path: str | Path | None) -> str:
    raw = str(path or "").strip()
    if not raw or " " in raw:
        return ""
    return f"MEDIA:{raw}"


def _clear_overlay_preview(session: dict[str, Any]) -> None:
    raw = str(session.get("overlay_screenshot_path") or "").strip()
    if raw:
        try:
            path = Path(os.path.normpath(str(Path(raw).expanduser().absolute())))
            managed_root = Path(os.path.normpath(str(_overlay_root().expanduser().absolute())))
            if path.exists() and (path == managed_root or managed_root in path.parents):
                path.unlink()
        except (OSError, RuntimeError):
            pass
    session["overlay_screenshot_path"] = ""


def _clear_session_state(session: dict[str, Any]) -> None:
    raw = str(session.get("session_state_path") or "").strip()
    if raw:
        try:
            path = Path(os.path.normpath(str(Path(raw).expanduser().absolute())))
            managed_root = Path(os.path.normpath(str(_session_state_root().expanduser().absolute())))
            if path.exists() and (path == managed_root or managed_root in path.parents):
                path.unlink()
        except (OSError, RuntimeError):
            pass
    session["session_state_path"] = ""


def _write_session_state(session: dict[str, Any]) -> str:
    state_path = _session_state_path(session)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    session["session_state_path"] = str(state_path)
    payload = {
        "session_id": _SESSION_ID,
        **_session_payload(session),
    }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return str(state_path)


def _sync_session_artifacts(session: dict[str, Any]) -> str:
    _sync_virtual_cursor_overlay(session)
    return _write_session_state(session)


def _record_pending_pointer_action(session: dict[str, Any], action_type: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "action_id": f"ptr-{uuid.uuid4().hex}",
        "action_type": action_type,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    session["pending_pointer_action"] = payload
    session["pending_pointer_claim_token"] = ""
    session["last_pointer_action_result"] = None
    return payload


def _sync_virtual_cursor_overlay(session: dict[str, Any]) -> str:
    cursor = session.get("virtual_cursor") or {}
    screenshot_path = Path(str(session.get("screenshot_path") or "")).expanduser()
    if not screenshot_path.exists():
        _clear_overlay_preview(session)
        return ""

    x = cursor.get("x")
    y = cursor.get("y")
    if x is None or y is None:
        _clear_overlay_preview(session)
        return ""

    magick = shutil.which("magick") or shutil.which("convert")
    if not magick:
        _clear_overlay_preview(session)
        return ""

    overlay_path = _overlay_preview_path(session)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ix = int(x)
        iy = int(y)
    except (TypeError, ValueError):
        _clear_overlay_preview(session)
        return ""

    draw_args = _pixel_cursor_draw_args(ix, iy)
    session["overlay_screenshot_path"] = str(overlay_path)

    try:
        subprocess.run([magick, str(screenshot_path), *draw_args, str(overlay_path)], check=True, capture_output=True, text=True)
    except Exception:
        _clear_overlay_preview(session)
        return ""

    session["overlay_screenshot_path"] = str(overlay_path)
    return str(overlay_path)


def _ensure_app_session(app_name: str | None, *, active: bool) -> dict[str, Any]:
    normalized = _normalize_app_name(app_name) or "desktop"
    key = _session_key(normalized)
    existing = _APP_SESSIONS.get(key)
    approved = _is_app_approved(normalized)
    if existing and (not active or existing.get("active")):
        existing["app_name"] = normalized
        existing["approved"] = approved or bool(existing.get("temporary_approval_scope"))
        existing.setdefault("approval_name", normalized)
        existing.setdefault("approval_bundle_id", "")
        existing.setdefault("approval_bundle_path", "")
        existing.setdefault("temporary_approval_scope", "")
        existing.setdefault("virtual_cursor", _fresh_virtual_cursor())
        return existing
    record = {
        "app_name": normalized,
        "app_session_id": f"app-{uuid.uuid4().hex}",
        "active": active,
        "approved": approved,
        "approval_name": normalized,
        "approval_bundle_id": "",
        "approval_bundle_path": "",
        "temporary_approval_scope": "",
        "accessibility_tree": [],
        "virtual_cursor": _fresh_virtual_cursor(),
    }
    _APP_SESSIONS[key] = record
    return record


def _find_session(*, app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any] | None:
    normalized = _normalize_app_name(app_name)
    if normalized:
        return _APP_SESSIONS.get(_session_key(normalized))
    wanted = str(app_session_id or "").strip()
    if wanted:
        for record in _APP_SESSIONS.values():
            if record.get("app_session_id") == wanted:
                return record
    return None


def _active_sessions() -> list[dict[str, Any]]:
    records = [_session_payload(record) for record in _APP_SESSIONS.values() if record.get("active")]
    return sorted(records, key=lambda item: item["app_name"].casefold())


def _normalize_bundle_path(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except Exception:
        try:
            return str(Path(text).expanduser().absolute())
        except Exception:
            return text


def _approval_entry_label(entry: dict[str, str]) -> str:
    label = _normalize_app_name(entry.get("approval_name"))
    if label:
        return label
    bundle_id = _normalize_app_name(entry.get("bundle_id"))
    if bundle_id:
        return bundle_id
    bundle_path = _normalize_bundle_path(entry.get("bundle_path"))
    if bundle_path:
        return Path(bundle_path).stem or bundle_path
    return ""


def _run_capture(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def _first_nonempty_line(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


def _parse_mdls_value(stdout: str) -> str:
    line = _first_nonempty_line(stdout)
    if not line:
        return ""
    if "=" in line:
        line = line.split("=", 1)[1].strip()
    if line in {"(null)", "null", "<null>"}:
        return ""
    if len(line) >= 2 and line[0] == line[-1] == '"':
        line = line[1:-1]
    return line.strip()


def _resolve_bundle_id_for_app(app_name: str, *, timeout: float = 5.0) -> str:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return ""
    proc = _run_capture(
        [
            "osascript",
            "-e", "on run argv",
            "-e", "set appRef to item 1 of argv",
            "-e", "return id of application appRef",
            "-e", "end run",
            normalized,
        ],
        timeout=timeout,
    )
    return _normalize_app_name(_first_nonempty_line(proc.stdout if proc else ""))


def _resolve_bundle_id_for_path(bundle_path: str, *, timeout: float = 5.0) -> str:
    normalized_path = _normalize_bundle_path(bundle_path)
    if not normalized_path:
        return ""
    proc = _run_capture(["mdls", "-name", "kMDItemCFBundleIdentifier", normalized_path], timeout=timeout)
    return _normalize_app_name(_parse_mdls_value(proc.stdout if proc else ""))


def _resolve_bundle_path_via_spotlight(bundle_id: str, *, timeout: float = 5.0) -> str:
    normalized_bundle_id = _normalize_app_name(bundle_id)
    if not normalized_bundle_id:
        return ""
    proc = _run_capture(["mdfind", f'kMDItemCFBundleIdentifier == "{normalized_bundle_id}"'], timeout=timeout)
    if not proc:
        return ""
    for raw_path in (proc.stdout or "").splitlines():
        candidate = _normalize_bundle_path(raw_path)
        if not candidate or not candidate.endswith(".app"):
            continue
        candidate_bundle_id = _resolve_bundle_id_for_path(candidate, timeout=timeout)
        if candidate_bundle_id and candidate_bundle_id.casefold() == normalized_bundle_id.casefold():
            return candidate
    return ""


def _resolve_bundle_path_for_app(app_name: str, *, timeout: float = 5.0) -> str:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return ""
    proc = _run_capture(
        [
            "osascript",
            "-e", "on run argv",
            "-e", "set appRef to item 1 of argv",
            "-e", "return POSIX path of (path to application appRef)",
            "-e", "end run",
            normalized,
        ],
        timeout=timeout,
    )
    return _normalize_bundle_path(_first_nonempty_line(proc.stdout if proc else ""))


def _resolve_approval_target(app_name: str) -> dict[str, str] | None:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return None
    bundle_id = _resolve_bundle_id_for_app(normalized)
    bundle_path = _resolve_bundle_path_via_spotlight(bundle_id)
    if not bundle_path:
        bundle_path = _resolve_bundle_path_for_app(normalized)
    if not bundle_id and bundle_path:
        bundle_id = _resolve_bundle_id_for_path(bundle_path)
    if not bundle_id and not bundle_path:
        return None
    return {
        "approval_name": normalized,
        "bundle_id": bundle_id,
        "bundle_path": bundle_path,
    }


def _normalize_approval_entry(raw_entry: Any) -> dict[str, str] | None:
    if isinstance(raw_entry, str):
        raw_entry = {"approval_name": raw_entry}
    if not isinstance(raw_entry, dict):
        return None
    label = _normalize_app_name(
        raw_entry.get("approval_name") or raw_entry.get("app_name") or raw_entry.get("bundle_name")
    )
    bundle_id = _normalize_app_name(raw_entry.get("bundle_id"))
    bundle_path = _normalize_bundle_path(raw_entry.get("bundle_path"))
    if label and (not bundle_id or not bundle_path):
        resolved = _resolve_approval_target(label)
        if resolved:
            if not bundle_id:
                bundle_id = _normalize_app_name(resolved.get("bundle_id"))
            if not bundle_path:
                bundle_path = _normalize_bundle_path(resolved.get("bundle_path"))
    if not label:
        if bundle_path:
            label = Path(bundle_path).stem or bundle_path
        else:
            label = bundle_id
    if not label and not bundle_id and not bundle_path:
        return None
    return {
        "approval_name": label,
        "bundle_id": bundle_id,
        "bundle_path": bundle_path,
    }


def _approval_entry_sort_key(entry: dict[str, str]) -> tuple[str, str, str]:
    return (
        _approval_entry_label(entry).casefold(),
        _normalize_app_name(entry.get("bundle_id")).casefold(),
        _normalize_bundle_path(entry.get("bundle_path")).casefold(),
    )


def _approval_entry_identity_key(entry: dict[str, str]) -> tuple[str, str, str]:
    return (
        _normalize_app_name(entry.get("bundle_id")).casefold(),
        _normalize_bundle_path(entry.get("bundle_path")).casefold(),
        _approval_entry_label(entry).casefold(),
    )


def _load_approval_entries() -> list[dict[str, str]]:
    with _APPROVALS_LOCK:
        path = _approval_store_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return []
        if isinstance(payload, dict):
            raw_entries = payload.get("approved_app_entries")
            if not isinstance(raw_entries, list):
                raw_entries = payload.get("approved_apps") or payload.get("approvedBundleIdentifiers") or []
        elif isinstance(payload, list):
            raw_entries = payload
        else:
            raw_entries = []
        entries: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw_entry in raw_entries:
            normalized = _normalize_approval_entry(raw_entry)
            if not normalized:
                continue
            key = _approval_entry_identity_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            entries.append(normalized)
        entries.sort(key=_approval_entry_sort_key)
        return entries


def _save_approval_entries(entries: list[dict[str, str]]) -> None:
    with _APPROVALS_LOCK:
        path = _approval_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for entry in entries:
            normalized = _normalize_approval_entry(entry)
            if not normalized:
                continue
            key = _approval_entry_identity_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        cleaned.sort(key=_approval_entry_sort_key)
        payload = {
            "approved_apps": [_approval_entry_label(entry) for entry in cleaned],
            "approved_app_entries": cleaned,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load_approved_apps() -> list[str]:
    return [_approval_entry_label(entry) for entry in _load_approval_entries()]


def _save_approved_apps(apps: list[str]) -> None:
    _save_approval_entries([{"approval_name": app} for app in apps])


def _approval_entry_matches_request(entry: dict[str, str], requested_app: str | None) -> bool:
    wanted = _normalize_app_name(requested_app)
    if not wanted:
        return False
    bundle_id = _normalize_app_name(entry.get("bundle_id"))
    bundle_path = _normalize_bundle_path(entry.get("bundle_path"))
    label = _approval_entry_label(entry)
    candidates = [label.casefold()]
    if bundle_id:
        candidates.append(bundle_id.casefold())
    if bundle_path:
        candidates.append(bundle_path.casefold())
    return wanted.casefold() in candidates


def _find_approval_entry(requested_app: str | None) -> dict[str, str] | None:
    for entry in _load_approval_entries():
        if _approval_entry_matches_request(entry, requested_app):
            return entry
    return None


def _is_app_approved(app_name: str | None) -> bool:
    return _find_approval_entry(app_name) is not None


def _frontmost_bundle_id(frontmost: dict[str, Any]) -> str:
    return _normalize_app_name(frontmost.get("bundle_id"))


def _frontmost_bundle_path(frontmost: dict[str, Any]) -> str:
    return _normalize_bundle_path(frontmost.get("bundle_path"))


def _approval_entry_matches_frontmost(entry: dict[str, str], frontmost: dict[str, Any]) -> bool:
    entry_bundle = _normalize_app_name(entry.get("bundle_id"))
    entry_path = _normalize_bundle_path(entry.get("bundle_path"))
    frontmost_bundle = _frontmost_bundle_id(frontmost)
    frontmost_path = _frontmost_bundle_path(frontmost)
    matched = False
    if entry_bundle:
        if not frontmost_bundle or entry_bundle.casefold() != frontmost_bundle.casefold():
            return False
        matched = True
    if entry_path and frontmost_path:
        if entry_path.casefold() != frontmost_path.casefold():
            return False
        matched = True
    return matched


def _approval_entry_fully_matches_frontmost(entry: dict[str, str], frontmost: dict[str, Any]) -> bool:
    entry_bundle = _normalize_app_name(entry.get("bundle_id"))
    entry_path = _normalize_bundle_path(entry.get("bundle_path"))
    frontmost_bundle = _frontmost_bundle_id(frontmost)
    frontmost_path = _frontmost_bundle_path(frontmost)
    matched = False
    if entry_bundle:
        if not frontmost_bundle or entry_bundle.casefold() != frontmost_bundle.casefold():
            return False
        matched = True
    if entry_path:
        if not frontmost_path or entry_path.casefold() != frontmost_path.casefold():
            return False
        matched = True
    return matched


def _frontmost_approved_entry(frontmost: dict[str, Any], requested_app: str | None = None) -> dict[str, str] | None:
    if requested_app:
        requested_entry = _find_approval_entry(requested_app)
        if requested_entry and _approval_entry_matches_frontmost(requested_entry, frontmost):
            return requested_entry
        return None
    for entry in _load_approval_entries():
        if _approval_entry_matches_frontmost(entry, frontmost):
            return entry
    return None


def _frontmost_approved_identity(frontmost: dict[str, Any], requested_app: str | None = None) -> str | None:
    entry = _frontmost_approved_entry(frontmost, requested_app=requested_app)
    if not entry:
        return None
    return _approval_entry_label(entry)


def list_active_sessions_impl() -> dict[str, Any]:
    with _APP_SESSIONS_LOCK:
        active_sessions = _active_sessions()
    return {
        "success": True,
        "session_id": _SESSION_ID,
        "active_sessions": active_sessions,
    }


def list_pending_pointer_actions_impl() -> dict[str, Any]:
    with _APP_SESSIONS_LOCK:
        pending_actions = [
            _session_payload(record)
            for record in _APP_SESSIONS.values()
            if record.get("active") and _optional_dict(record, "pending_pointer_action")
        ]
    pending_actions.sort(key=lambda item: item["app_name"].casefold())
    return {
        "success": True,
        "session_id": _SESSION_ID,
        "pending_actions": pending_actions,
    }


def _current_active_session() -> dict[str, Any] | None:
    active_records = [record for record in _APP_SESSIONS.values() if record.get("active")]
    if not active_records:
        return None
    return active_records[-1]


def _resolve_session(app_session_id: str | None = None) -> dict[str, Any] | None:
    wanted = str(app_session_id or "").strip()
    if wanted:
        record = _find_session(app_session_id=wanted)
        if record and record.get("active"):
            return record
        return None
    return _current_active_session()


def _session_required_error(action: str) -> dict[str, Any]:
    return {
        "success": False,
        "session_required": True,
        "session_id": _SESSION_ID,
        "error": f"No active app session. Call get_app_state(app_name=...) before {action}.",
    }


def _multiple_active_sessions_error(action: str) -> dict[str, Any]:
    return {
        "success": False,
        "session_required": True,
        "multiple_active_sessions": True,
        "active_sessions": _active_sessions(),
        "session_id": _SESSION_ID,
        "error": f"Multiple active app sessions found. Pass app_session_id before {action}.",
    }


def _session_approval_required_error(action: str, session: dict[str, Any]) -> dict[str, Any]:
    app_name = str(session.get("app_name") or "this app")
    return {
        "success": False,
        "approval_required": True,
        "approved": False,
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "approved_apps": _load_approved_apps(),
        "approval_store_path": str(_approval_store_path()),
        "error": f"{app_name} is not approved for Hermes computer-use yet. Approve it before {action}.",
    }


def _resolve_keyboard_session(action: str, app_session_id: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    wanted = str(app_session_id or "").strip()
    with _APP_SESSIONS_LOCK:
        if wanted:
            session = _find_session(app_session_id=wanted)
            if not session or not session.get("active"):
                return None, _session_required_error(action)
        else:
            active_records = [record for record in _APP_SESSIONS.values() if record.get("active")]
            if not active_records:
                return None, _session_required_error(action)
            if len(active_records) > 1:
                return None, _multiple_active_sessions_error(action)
            session = active_records[0]
        if not session.get("approved"):
            return None, _session_approval_required_error(action, session)
        return session, None


def _preview_pointer_response(action: str, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "supported": False,
        "preview_only": True,
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "error": f"{action} is not implemented in the Hermes computer-use adapter yet.",
    }


def _pending_pointer_action_conflict(action: str, session: dict[str, Any]) -> dict[str, Any]:
    pending = _optional_dict(session, "pending_pointer_action")
    action_id = str((pending or {}).get("action_id") or "").strip()
    return {
        "success": False,
        "action_pending": True,
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "error": (
            f"Cannot queue {action} while pointer action {action_id or 'unknown'} is still pending. "
            "Report the helper result first."
        ),
    }


def stop_app_session_impl(app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any]:
    with _APP_SESSIONS_LOCK:
        record = _find_session(app_name=app_name, app_session_id=app_session_id)
        if not record:
            return {
                "success": False,
                "stopped": False,
                "session_id": _SESSION_ID,
                "error": "No matching app session was found.",
            }
        was_active = bool(record.get("active"))
        record["active"] = False
        record["temporary_approval_scope"] = ""
        if not _find_approval_entry(record.get("approval_name") or record.get("app_name")):
            record["approved"] = False
            record["accessibility_tree"] = []
        _clear_overlay_preview(record)
        _clear_session_state(record)
        return {
            "success": True,
            "stopped": was_active,
            "session_id": _SESSION_ID,
            **_session_payload(record),
        }


def list_approved_apps_impl() -> dict[str, Any]:
    return {
        "success": True,
        "approved_apps": _load_approved_apps(),
        "approval_store_path": str(_approval_store_path()),
    }


def approve_app_impl(app_name: str) -> dict[str, Any]:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return {"success": False, "error": "app_name is required"}
    resolved = _resolve_approval_target(normalized)
    if not resolved:
        return {
            "success": False,
            "app_name": normalized,
            "error": f"Could not resolve a stable app identity for {normalized}.",
        }
    with _APPROVALS_LOCK:
        old_entries = _load_approval_entries()
        replaced_entries = [entry for entry in old_entries if _approval_entry_matches_request(entry, normalized)]
        entries = [entry for entry in old_entries if not _approval_entry_matches_request(entry, normalized)]
        entries.append(resolved)
        _save_approval_entries(entries)
        approved_apps = _load_approved_apps()
        with _APP_SESSIONS_LOCK:
            for session in _APP_SESSIONS.values():
                session_label = _normalize_app_name(session.get("approval_name") or session.get("app_name"))
                if session_label.casefold() != normalized.casefold():
                    continue
                if _session_matches_approval_entry(session, resolved):
                    continue
                session["approved"] = False
                session["accessibility_tree"] = []
                if replaced_entries:
                    session["approval_bundle_id"] = _normalize_app_name(session.get("approval_bundle_id"))
                    session["approval_bundle_path"] = _normalize_bundle_path(session.get("approval_bundle_path"))
                if session.get("active") or session.get("session_state_path"):
                    _sync_session_artifacts(session)
    return {
        "success": True,
        "app_name": normalized,
        "approved_apps": approved_apps,
        "approval_store_path": str(_approval_store_path()),
    }


def _session_matches_approval_entry(session: dict[str, Any], approval_entry: dict[str, str]) -> bool:
    session_label = _normalize_app_name(session.get("approval_name") or session.get("app_name"))
    session_bundle = _normalize_app_name(session.get("approval_bundle_id") or session.get("bundle_id"))
    session_path = _normalize_bundle_path(session.get("approval_bundle_path") or session.get("bundle_path"))
    approval_label = _approval_entry_label(approval_entry)
    approval_bundle = _normalize_app_name(approval_entry.get("bundle_id"))
    approval_path = _normalize_bundle_path(approval_entry.get("bundle_path"))
    matched = False
    if approval_bundle or session_bundle:
        if not approval_bundle or not session_bundle or approval_bundle.casefold() != session_bundle.casefold():
            return False
        matched = True
    if approval_path or session_path:
        if not approval_path or not session_path or approval_path.casefold() != session_path.casefold():
            return False
        matched = True
    if matched:
        return True
    return bool(approval_label and session_label and approval_label.casefold() == session_label.casefold())


def _session_matches_removed_approval(session: dict[str, Any], removed_entry: dict[str, str]) -> bool:
    return _session_matches_approval_entry(session, removed_entry)


def _temporary_approval_entry(session: dict[str, Any]) -> dict[str, str] | None:
    scope = str(session.get("temporary_approval_scope") or "").strip().lower()
    if scope not in {"once", "session"}:
        return None
    entry = {
        "approval_name": _normalize_app_name(session.get("approval_name") or session.get("app_name")),
        "bundle_id": _normalize_app_name(session.get("approval_bundle_id") or session.get("bundle_id")),
        "bundle_path": _normalize_bundle_path(session.get("approval_bundle_path") or session.get("bundle_path")),
    }
    if not any(entry.values()):
        return None
    return entry


def _session_authorization_entry(session: dict[str, Any]) -> dict[str, str] | None:
    current_entry = _find_approval_entry(session.get("approval_name") or session.get("app_name"))
    if current_entry and _session_matches_approval_entry(session, current_entry):
        return current_entry
    return _temporary_approval_entry(session)


def grant_temporary_app_approval_impl(
    app_name: str,
    app_session_id: str | None = None,
    scope: str = "once",
) -> dict[str, Any]:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return {"success": False, "error": "app_name is required"}
    normalized_scope = str(scope or "once").strip().lower()
    if normalized_scope not in {"once", "session"}:
        return {"success": False, "error": "scope must be 'once' or 'session'"}
    resolved = _resolve_approval_target(normalized)
    with _APPROVALS_LOCK:
        with _APP_SESSIONS_LOCK:
            session = _find_session(app_session_id=app_session_id) if app_session_id else None
            session = session or _ensure_app_session(normalized, active=False)
            session["approval_name"] = normalized
            session["temporary_approval_scope"] = normalized_scope
            session["approved"] = True
            if resolved:
                session["approval_bundle_id"] = _normalize_app_name(resolved.get("bundle_id"))
                session["approval_bundle_path"] = _normalize_bundle_path(resolved.get("bundle_path"))
            else:
                session.setdefault("approval_bundle_id", "")
                session.setdefault("approval_bundle_path", "")
            if session.get("active") or session.get("session_state_path"):
                _sync_session_artifacts(session)
            return {
                "success": True,
                "scope": normalized_scope,
                "session_id": _SESSION_ID,
                **_session_payload(session),
            }


def _consume_temporary_once_approval(session: dict[str, Any]) -> None:
    if str(session.get("temporary_approval_scope") or "").strip().lower() != "once":
        return
    session["temporary_approval_scope"] = ""
    if not _find_approval_entry(session.get("approval_name") or session.get("app_name")):
        session["approved"] = False
        session["accessibility_tree"] = []
    if session.get("active") or session.get("session_state_path"):
        _sync_session_artifacts(session)


def revoke_app_impl(app_name: str) -> dict[str, Any]:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return {"success": False, "error": "app_name is required"}
    with _APPROVALS_LOCK:
        entries = _load_approval_entries()
        removed = [entry for entry in entries if _approval_entry_matches_request(entry, normalized)]
        remaining = [entry for entry in entries if not _approval_entry_matches_request(entry, normalized)]
        _save_approval_entries(remaining)
        approved_apps = _load_approved_apps()
        with _APP_SESSIONS_LOCK:
            for session in _APP_SESSIONS.values():
                if removed and not any(_session_matches_removed_approval(session, entry) for entry in removed):
                    continue
                if not removed:
                    session_identity = _normalize_app_name(session.get("approval_name") or session.get("app_name"))
                    if session_identity.casefold() != normalized.casefold():
                        continue
                session["approved"] = False
                session["accessibility_tree"] = []
                if session.get("active") or session.get("session_state_path"):
                    _sync_session_artifacts(session)
    return {
        "success": True,
        "app_name": normalized,
        "approved_apps": approved_apps,
        "approval_store_path": str(_approval_store_path()),
    }


def _running_apps() -> list[str]:
    script = 'tell application "System Events" to get name of every application process whose background only is false'
    out = subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
    raw = out.stdout.strip()
    if not raw:
        return []
    return sorted({part.strip() for part in raw.split(",") if part.strip()})


def _installed_apps(limit: int = 100) -> list[str]:
    roots = [Path("/Applications"), Path.home() / "Applications"]
    names: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for app in sorted(root.glob("*.app")):
            names.add(app.stem)
            if len(names) >= limit:
                return sorted(names)
    return sorted(names)


def list_apps_impl(limit: int = 100) -> dict[str, Any]:
    running = sorted(_running_apps())
    installed = _installed_apps(limit=limit)
    return {
        "success": True,
        "running_apps": running,
        "installed_apps": installed,
    }


def get_app_state_impl(app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any]:
    session = None
    requested_app = _normalize_app_name(app_name)
    wanted_session_id = str(app_session_id or "").strip()
    if wanted_session_id:
        session = _find_session(app_session_id=wanted_session_id)
        if not session:
            return _session_required_error("getting app state")
        requested_app = _normalize_app_name(session.get("approval_name") or session.get("app_name"))
    elif requested_app:
        existing_session = _find_session(app_name=requested_app)
        if existing_session and str(existing_session.get("temporary_approval_scope") or "").strip():
            session = existing_session

    requested_entry = _find_approval_entry(requested_app) if requested_app else None
    temporary_entry = _temporary_approval_entry(session) if session else None
    if requested_app and not requested_entry and temporary_entry is None:
        pending_session = session or _ensure_app_session(requested_app, active=False)
        pending_session["approved"] = False
        pending_session["accessibility_tree"] = []
        if requested_app:
            pending_session["approval_name"] = requested_app
            pending_session["approval_bundle_id"] = ""
            pending_session["approval_bundle_path"] = ""
        return {
            "success": False,
            "approval_required": True,
            "approved": False,
            "app_name": requested_app,
            "session_id": _SESSION_ID,
            **_session_payload(pending_session),
            "approved_apps": _load_approved_apps(),
            "approval_store_path": str(_approval_store_path()),
            "error": f"{requested_app} is not approved for Hermes computer-use yet.",
        }

    if requested_app:
        activated = _decode(computer_control(action="activate_app", app_name=requested_app))
        if activated.get("error"):
            return {"success": False, "error": activated["error"]}

    frontmost = _decode(computer_control(action="frontmost_app"))
    if frontmost.get("error"):
        return {"success": False, "error": frontmost["error"]}

    screenshot_kwargs: dict[str, Any] = {"action": "screenshot"}
    frontmost_window_id = frontmost.get("window_id")
    if frontmost_window_id is not None:
        screenshot_kwargs["window_id"] = frontmost_window_id
    screenshot = _decode(computer_control(**screenshot_kwargs))
    if screenshot.get("error") and frontmost_window_id is not None:
        screenshot = _decode(computer_control(action="screenshot"))
    if screenshot.get("error"):
        return {"success": False, "error": screenshot["error"]}

    frontmost_app_name = _normalize_app_name(frontmost.get("app_name"))
    frontmost_bundle_id = _normalize_app_name(frontmost.get("bundle_id"))
    frontmost_bundle_path = _normalize_bundle_path(frontmost.get("bundle_path"))
    raw_accessibility_tree: list[dict[str, Any]] = []
    accessibility_state: dict[str, Any] | None = None
    with _APPROVALS_LOCK:
        matched_entry = _frontmost_approved_entry(frontmost, requested_app=requested_app)
        approved_entry = matched_entry if matched_entry and _approval_entry_fully_matches_frontmost(matched_entry, frontmost) else None
        temporary_entry = _temporary_approval_entry(session) if session else None
        temporary_approved = bool(temporary_entry and _approval_entry_fully_matches_frontmost(temporary_entry, frontmost))
        authorization_entry = approved_entry or temporary_entry
        approved_identity = _approval_entry_label(authorization_entry) if authorization_entry else None
        if authorization_entry is not None and (approved_entry is not None or temporary_approved):
            try:
                accessibility_state = frontmost_window_state(include_accessibility=True, require_helper_success=True)
            except Exception as e:
                return {"success": False, "error": f"Failed to capture accessibility snapshot: {e}"}
            if (
                isinstance(accessibility_state, dict)
                and _frontmost_states_match(frontmost, accessibility_state)
                and _approval_entry_fully_matches_frontmost(authorization_entry, frontmost)
                and _approval_entry_fully_matches_frontmost(authorization_entry, accessibility_state)
            ):
                candidate_tree = accessibility_state.get("accessibility_tree")
                if isinstance(candidate_tree, list):
                    raw_accessibility_tree = candidate_tree
        approved_now = approved_entry is not None or temporary_approved
        accessibility_tree = raw_accessibility_tree if approved_now else []
        with _APP_SESSIONS_LOCK:
            session = session or _ensure_app_session(frontmost_app_name or requested_app, active=True)
            session["active"] = True
            session["app_name"] = frontmost_app_name or requested_app or session.get("app_name", "desktop")
            session["approval_name"] = approved_identity or requested_app or session.get("approval_name") or session["app_name"]
            session["approval_bundle_id"] = _normalize_app_name((authorization_entry or {}).get("bundle_id"))
            session["approval_bundle_path"] = _normalize_bundle_path((authorization_entry or {}).get("bundle_path"))
            session["approved"] = approved_now
            session["bundle_id"] = frontmost_bundle_id
            session["bundle_path"] = frontmost_bundle_path
            session["process_id"] = frontmost.get("process_id")
            session["window_id"] = frontmost.get("window_id")
            session["window_bounds"] = frontmost.get("window_bounds")
            session["window_title"] = frontmost.get("window_title", "")
            session["accessibility_tree"] = accessibility_tree
            session["screenshot_path"] = screenshot.get("path", "")
            session.setdefault("virtual_cursor", _fresh_virtual_cursor())
            _sync_session_artifacts(session)
    note = (
        "Hermes adapter currently returns screenshot + frontmost window metadata + a best-effort window accessibility tree."
        if accessibility_tree else
        "Hermes adapter currently returns screenshot + frontmost window metadata, but not a full accessibility tree."
    )
    response = {
        "success": True,
        "app_name": frontmost_app_name,
        "bundle_id": frontmost_bundle_id,
        "bundle_path": frontmost_bundle_path,
        "process_id": frontmost.get("process_id"),
        "window_id": frontmost.get("window_id"),
        "window_bounds": frontmost.get("window_bounds"),
        "window_title": frontmost.get("window_title", ""),
        "screenshot_path": screenshot.get("path", ""),
        "media_tag": screenshot.get("media_tag"),
        "accessibility_tree": accessibility_tree,
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "approval_required": False,
        "approved_apps": _load_approved_apps(),
        "approval_store_path": str(_approval_store_path()),
        "note": note,
    }
    if temporary_approved and approved_entry is None:
        with _APP_SESSIONS_LOCK:
            _consume_temporary_once_approval(session)
    return response


def type_text_impl(text: str, app_session_id: str | None = None) -> dict[str, Any]:
    wanted = str(app_session_id or "").strip()
    with _APPROVALS_LOCK:
        with _APP_SESSIONS_LOCK:
            if wanted:
                session = _find_session(app_session_id=wanted)
                if not session or not session.get("active"):
                    return _session_required_error("typing")
            else:
                active_records = [record for record in _APP_SESSIONS.values() if record.get("active")]
                if not active_records:
                    return _session_required_error("typing")
                if len(active_records) > 1:
                    return _multiple_active_sessions_error("typing")
                session = active_records[0]
            authorization_entry = _session_authorization_entry(session)
            if not session.get("approved") or not authorization_entry:
                session["approved"] = False
                session["accessibility_tree"] = []
                return _session_approval_required_error("typing", session)
            result = _decode(computer_control(action="keystroke", text=text))
            response = {"success": not bool(result.get("error")), **result, **_session_payload(session), "session_id": _SESSION_ID}
            if response["success"]:
                _consume_temporary_once_approval(session)
            return response


def press_key_impl(key: str, modifiers: list[str] | None = None, app_session_id: str | None = None) -> dict[str, Any]:
    wanted = str(app_session_id or "").strip()
    with _APPROVALS_LOCK:
        with _APP_SESSIONS_LOCK:
            if wanted:
                session = _find_session(app_session_id=wanted)
                if not session or not session.get("active"):
                    return _session_required_error("pressing keys")
            else:
                active_records = [record for record in _APP_SESSIONS.values() if record.get("active")]
                if not active_records:
                    return _session_required_error("pressing keys")
                if len(active_records) > 1:
                    return _multiple_active_sessions_error("pressing keys")
                session = active_records[0]
            authorization_entry = _session_authorization_entry(session)
            if not session.get("approved") or not authorization_entry:
                session["approved"] = False
                session["accessibility_tree"] = []
                return _session_approval_required_error("pressing keys", session)
            result = _decode(computer_control(action="keystroke", key=key, modifiers=modifiers or []))
            response = {"success": not bool(result.get("error")), **result, **_session_payload(session), "session_id": _SESSION_ID}
            if response["success"]:
                _consume_temporary_once_approval(session)
            return response


def _unsupported(action: str) -> dict[str, Any]:
    return {
        "success": False,
        "supported": False,
        "error": f"{action} is not implemented in the Hermes computer-use adapter yet.",
    }


def _record_local_pointer_result(session: dict[str, Any], *, action_type: str, status: str,
                                 x: int | None = None, y: int | None = None,
                                 error: str | None = None, **fields: Any) -> dict[str, Any]:
    payload = {
        "action_id": f"ptr-local-{uuid.uuid4().hex}",
        "action_type": action_type,
        "status": status,
        "reported_by": "local_backend",
    }
    if x is not None:
        payload["x"] = x
    if y is not None:
        payload["y"] = y
    if error:
        payload["error"] = error
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    session["pending_pointer_action"] = None
    session["pending_pointer_claim_token"] = ""
    session["last_pointer_action_result"] = payload
    return payload


def click_impl(*, index: int | None = None, x: int | None = None, y: int | None = None,
               button: str = "left", click_count: int = 1, app_session_id: str | None = None) -> dict[str, Any]:
    wanted = str(app_session_id or "").strip()
    with _APPROVALS_LOCK:
        with _APP_SESSIONS_LOCK:
            if wanted:
                session = _find_session(app_session_id=wanted)
                if not session or not session.get("active"):
                    return _session_required_error("clicking")
            else:
                active_records = [record for record in _APP_SESSIONS.values() if record.get("active")]
                if not active_records:
                    return _session_required_error("clicking")
                if len(active_records) > 1:
                    return _multiple_active_sessions_error("clicking")
                session = active_records[0]
            authorization_entry = _session_authorization_entry(session)
            if not session.get("approved") or not authorization_entry:
                session["approved"] = False
                session["accessibility_tree"] = []
                return _session_approval_required_error("clicking", session)
            if _optional_dict(session, "pending_pointer_action"):
                return _pending_pointer_action_conflict("click", session)
            cursor = session.setdefault("virtual_cursor", _fresh_virtual_cursor())
            if x is not None:
                cursor["x"] = x
            if y is not None:
                cursor["y"] = y
            if cursor.get("x") is None or cursor.get("y") is None:
                return {
                    "success": False,
                    "session_id": _SESSION_ID,
                    **_session_payload(session),
                    "error": "click requires x and y coordinates",
                }
            result = _decode(computer_control(
                action="click",
                x=int(cursor["x"]),
                y=int(cursor["y"]),
                button=button,
                click_count=click_count,
            ))
            if result.get("error"):
                _record_local_pointer_result(
                    session,
                    action_type="click",
                    status="failed",
                    x=int(cursor["x"]),
                    y=int(cursor["y"]),
                    button=button,
                    click_count=click_count,
                    index=index,
                    error=str(result.get("error") or "click failed"),
                )
                _sync_session_artifacts(session)
                return {"success": False, **result, **_session_payload(session), "session_id": _SESSION_ID}
            _record_local_pointer_result(
                session,
                action_type="click",
                status="completed",
                x=int(cursor["x"]),
                y=int(cursor["y"]),
                button=button,
                click_count=click_count,
                index=index,
            )
            _sync_session_artifacts(session)
            response = {
                "success": True,
                "clicked": True,
                **result,
                **_session_payload(session),
                "session_id": _SESSION_ID,
            }
            _consume_temporary_once_approval(session)
            return response


def perform_secondary_action_impl(index: int, action_name: str) -> dict[str, Any]:
    return _unsupported("perform_secondary_action")


def scroll_impl(index: int | None = None, x: int | None = None, y: int | None = None,
                delta_y: int = 0, app_session_id: str | None = None) -> dict[str, Any]:
    session = _resolve_session(app_session_id=app_session_id)
    if not session:
        return _session_required_error("scrolling")
    with _APP_SESSIONS_LOCK:
        if not session.get("active"):
            return _session_required_error("scrolling")
        if _optional_dict(session, "pending_pointer_action"):
            return _pending_pointer_action_conflict("scroll", session)
        cursor = session.setdefault("virtual_cursor", _fresh_virtual_cursor())
        if x is not None:
            cursor["x"] = x
        if y is not None:
            cursor["y"] = y
        _record_pending_pointer_action(
            session,
            "scroll",
            x=cursor.get("x"),
            y=cursor.get("y"),
            index=index,
            delta_y=delta_y,
        )
        _sync_session_artifacts(session)
        response = _preview_pointer_response("scroll", session)
        response["delta_y"] = delta_y
        return response


def drag_impl(start_x: int, start_y: int, end_x: int, end_y: int, app_session_id: str | None = None) -> dict[str, Any]:
    session = _resolve_session(app_session_id=app_session_id)
    if not session:
        return _session_required_error("dragging")
    with _APP_SESSIONS_LOCK:
        if not session.get("active"):
            return _session_required_error("dragging")
        if _optional_dict(session, "pending_pointer_action"):
            return _pending_pointer_action_conflict("drag", session)
        cursor = session.setdefault("virtual_cursor", _fresh_virtual_cursor())
        cursor["x"] = end_x
        cursor["y"] = end_y
        _record_pending_pointer_action(
            session,
            "drag",
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )
        _sync_session_artifacts(session)
        response = _preview_pointer_response("drag", session)
        response["drag_path"] = {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}
        return response


def claim_pending_pointer_action_impl(*, app_session_id: str, action_id: str, worker_id: str | None = None) -> dict[str, Any]:
    wanted_session_id = str(app_session_id or "").strip()
    if not wanted_session_id:
        current = _current_active_session()
        payload = _session_payload(current) if current else {}
        return {
            "success": False,
            "app_session_required": True,
            "session_id": _SESSION_ID,
            **payload,
            "error": "app_session_id is required",
        }

    session = _resolve_session(app_session_id=wanted_session_id)
    if not session:
        return _session_required_error("claiming pointer actions")

    wanted_action_id = str(action_id or "").strip()
    if not wanted_action_id:
        return {
            "success": False,
            "session_id": _SESSION_ID,
            **_session_payload(session),
            "error": "action_id is required",
        }

    claimer = str(worker_id or "").strip()
    if not claimer:
        return {
            "success": False,
            "worker_required": True,
            "session_id": _SESSION_ID,
            **_session_payload(session),
            "error": "worker_id is required",
        }

    with _APP_SESSIONS_LOCK:
        if not session.get("active"):
            return _session_required_error("claiming pointer actions")
        pending = session.get("pending_pointer_action")
        if not isinstance(pending, dict):
            return {
                "success": False,
                "action_required": True,
                "session_id": _SESSION_ID,
                **_session_payload(session),
                "error": "No pending pointer action is available to claim.",
            }

        if str(pending.get("action_id") or "").strip() != wanted_action_id:
            return {
                "success": False,
                "action_mismatch": True,
                "session_id": _SESSION_ID,
                "expected_action_id": pending.get("action_id"),
                "received_action_id": wanted_action_id,
                **_session_payload(session),
                "error": "Pending pointer action does not match the claimed action_id.",
            }

        existing_claimer = str(pending.get("claimed_by") or "").strip()
        claim_expires_at = _parse_iso_datetime(pending.get("claim_expires_at"))
        now = _utc_now()
        if existing_claimer:
            claim_is_active = claim_expires_at is None or claim_expires_at > now
            if claim_is_active:
                return {
                    "success": False,
                    "action_claimed": True,
                    "session_id": _SESSION_ID,
                    "claimed_by": existing_claimer,
                    **_session_payload(session),
                    "error": f"Pending pointer action is already claimed by {existing_claimer}.",
                }

        pending["claimed_by"] = claimer
        pending["claimed_at"] = now.isoformat()
        pending["claim_expires_at"] = (now + timedelta(seconds=_PENDING_POINTER_CLAIM_TTL_SECONDS)).isoformat()
        claim_token = f"claim-{uuid.uuid4().hex}"
        session["pending_pointer_claim_token"] = claim_token
        _sync_session_artifacts(session)

        return {
            "success": True,
            "claimed": True,
            "claim_token": claim_token,
            "session_id": _SESSION_ID,
            **_session_payload(session),
        }


def report_pointer_action_result_impl(*, app_session_id: str, action_id: str, status: str,
                                      claim_token: str | None = None,
                                      x: int | None = None, y: int | None = None,
                                      error: str | None = None) -> dict[str, Any]:
    wanted_session_id = str(app_session_id or "").strip()
    if not wanted_session_id:
        current = _current_active_session()
        payload = _session_payload(current) if current else {}
        return {
            "success": False,
            "app_session_required": True,
            "session_id": _SESSION_ID,
            **payload,
            "error": "app_session_id is required",
        }

    session = _resolve_session(app_session_id=wanted_session_id)
    if not session:
        return _session_required_error("reporting pointer action results")

    wanted_action_id = str(action_id or "").strip()
    if not wanted_action_id:
        return {
            "success": False,
            "session_id": _SESSION_ID,
            **_session_payload(session),
            "error": "action_id is required",
        }

    provided_claim_token = str(claim_token or "").strip()
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"completed", "failed"}:
        return {
            "success": False,
            "session_id": _SESSION_ID,
            **_session_payload(session),
            "error": "status must be 'completed' or 'failed'",
        }

    with _APP_SESSIONS_LOCK:
        if not session.get("active"):
            return _session_required_error("reporting pointer action results")
        pending = session.get("pending_pointer_action")
        if not isinstance(pending, dict):
            return {
                "success": False,
                "action_required": True,
                "session_id": _SESSION_ID,
                **_session_payload(session),
                "error": "No pending pointer action is waiting for a helper result.",
            }

        if str(pending.get("action_id") or "").strip() != wanted_action_id:
            return {
                "success": False,
                "action_mismatch": True,
                "session_id": _SESSION_ID,
                "expected_action_id": pending.get("action_id"),
                "received_action_id": wanted_action_id,
                **_session_payload(session),
                "error": "Pending pointer action does not match the reported action_id.",
            }

        existing_claimer = str(pending.get("claimed_by") or "").strip()
        required_claim_token = str(session.get("pending_pointer_claim_token") or "").strip()
        claim_expires_at = _parse_iso_datetime(pending.get("claim_expires_at"))
        now = _utc_now()
        if existing_claimer and claim_expires_at is not None and claim_expires_at <= now:
            return {
                "success": False,
                "claim_expired": True,
                "session_id": _SESSION_ID,
                "claimed_by": existing_claimer,
                **_session_payload(session),
                "error": f"Pending pointer action claim has expired for {existing_claimer}; reclaim it before reporting the result.",
            }
        if existing_claimer and not required_claim_token:
            return {
                "success": False,
                "action_claimed": True,
                "session_id": _SESSION_ID,
                "claimed_by": existing_claimer,
                **_session_payload(session),
                "error": f"Pending pointer action is claimed by {existing_claimer}; a valid claim_token is required to report the result.",
            }
        if existing_claimer and provided_claim_token != required_claim_token:
            return {
                "success": False,
                "action_claimed": True,
                "session_id": _SESSION_ID,
                "claimed_by": existing_claimer,
                **_session_payload(session),
                "error": f"Pending pointer action is claimed by {existing_claimer}; a valid claim_token is required to report the result.",
            }

        cursor = session.setdefault("virtual_cursor", _fresh_virtual_cursor())
        if x is not None:
            cursor["x"] = x
        if y is not None:
            cursor["y"] = y

        result_payload = {
            **pending,
            "status": normalized_status,
        }
        if existing_claimer:
            result_payload["reported_by"] = existing_claimer
        if x is not None:
            result_payload["x"] = x
        if y is not None:
            result_payload["y"] = y
        if error:
            result_payload["error"] = error

        session["pending_pointer_action"] = None
        session["pending_pointer_claim_token"] = ""
        session["last_pointer_action_result"] = result_payload
        _sync_session_artifacts(session)
        return {
            "success": True,
            "reported": True,
            "status": normalized_status,
            "session_id": _SESSION_ID,
            **_session_payload(session),
        }


def set_value_impl(index: int, value: str) -> dict[str, Any]:
    return _unsupported("set_value")


mcp = FastMCP("hermes-computer-use-adapter") if FastMCP else None

if mcp:
    @mcp.tool()
    def list_apps(limit: int = 100) -> dict[str, Any]:
        """List running apps and a sample of installed apps visible to the Hermes adapter."""
        return list_apps_impl(limit=limit)

    @mcp.tool()
    def list_active_sessions() -> dict[str, Any]:
        """List currently active Hermes computer-use app sessions."""
        return list_active_sessions_impl()

    @mcp.tool()
    def list_pending_pointer_actions() -> dict[str, Any]:
        """List active app sessions that currently have unresolved pending pointer actions for a helper."""
        return list_pending_pointer_actions_impl()

    @mcp.tool()
    def stop_app_session(app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any]:
        """Stop an active Hermes computer-use app session by app name or app session id."""
        return stop_app_session_impl(app_name=app_name, app_session_id=app_session_id)

    @mcp.tool()
    def list_approved_apps() -> dict[str, Any]:
        """List apps explicitly approved for Hermes computer-use control."""
        return list_approved_apps_impl()

    @mcp.tool()
    def approve_app(app_name: str) -> dict[str, Any]:
        """Approve an app for Hermes computer-use control and persist the allowlist locally."""
        return approve_app_impl(app_name)

    @mcp.tool()
    def grant_temporary_app_approval(app_name: str, app_session_id: str | None = None, scope: str = "once") -> dict[str, Any]:
        """Grant a non-persistent temporary approval for a specific app session or requested app."""
        return grant_temporary_app_approval_impl(app_name=app_name, app_session_id=app_session_id, scope=scope)

    @mcp.tool()
    def revoke_app(app_name: str) -> dict[str, Any]:
        """Revoke a previously approved app from Hermes computer-use control."""
        return revoke_app_impl(app_name)

    @mcp.tool()
    def get_app_state(app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any]:
        """Activate an app if requested, then return a fresh screenshot and frontmost window metadata."""
        return get_app_state_impl(app_name=app_name, app_session_id=app_session_id)

    @mcp.tool()
    def type_text(text: str, app_session_id: str | None = None) -> dict[str, Any]:
        """Type literal text via Hermes' computer-control backend."""
        return type_text_impl(text, app_session_id=app_session_id)

    @mcp.tool()
    def press_key(key: str, modifiers: list[str] | None = None, app_session_id: str | None = None) -> dict[str, Any]:
        """Press one key or key combination via Hermes' computer-control backend."""
        return press_key_impl(key, modifiers or [], app_session_id=app_session_id)

    @mcp.tool()
    def click(index: int | None = None, x: int | None = None, y: int | None = None,
              button: str = "left", click_count: int = 1, app_session_id: str | None = None) -> dict[str, Any]:
        """Reserved for future pointer support. Returns an explicit unsupported result for now."""
        return click_impl(index=index, x=x, y=y, button=button, click_count=click_count, app_session_id=app_session_id)

    @mcp.tool()
    def perform_secondary_action(index: int, action_name: str) -> dict[str, Any]:
        """Reserved for future accessibility action support."""
        return perform_secondary_action_impl(index=index, action_name=action_name)

    @mcp.tool()
    def scroll(index: int | None = None, x: int | None = None, y: int | None = None,
               delta_y: int = 0, app_session_id: str | None = None) -> dict[str, Any]:
        """Reserved for future scroll support."""
        return scroll_impl(index=index, x=x, y=y, delta_y=delta_y, app_session_id=app_session_id)

    @mcp.tool()
    def drag(start_x: int, start_y: int, end_x: int, end_y: int, app_session_id: str | None = None) -> dict[str, Any]:
        """Reserved for future drag support."""
        return drag_impl(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y, app_session_id=app_session_id)

    @mcp.tool()
    def claim_pending_pointer_action(app_session_id: str, action_id: str, worker_id: str) -> dict[str, Any]:
        """Helper-facing bridge: claim an unresolved pending pointer action for a specific app session."""
        return claim_pending_pointer_action_impl(app_session_id=app_session_id, action_id=action_id, worker_id=worker_id)

    @mcp.tool()
    def report_pointer_action_result(app_session_id: str, action_id: str, status: str,
                                     claim_token: str | None = None,
                                     x: int | None = None, y: int | None = None,
                                     error: str | None = None) -> dict[str, Any]:
        """Helper-facing bridge: report the execution result of a pending pointer action for an app session."""
        return report_pointer_action_result_impl(
            app_session_id=app_session_id,
            action_id=action_id,
            status=status,
            claim_token=claim_token,
            x=x,
            y=y,
            error=error,
        )

    @mcp.tool()
    def set_value(index: int, value: str) -> dict[str, Any]:
        """Reserved for future settable accessibility elements."""
        return set_value_impl(index=index, value=value)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    if not mcp:
        raise SystemExit("FastMCP is unavailable. Install the mcp package first.")
    mcp.run()
