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
from tools.computer_control_tool import computer_control


def _decode(payload: str) -> dict[str, Any]:
    data = json.loads(payload)
    assert isinstance(data, dict)
    return data


_SESSION_ID = f"session-{uuid.uuid4().hex}"
_APP_SESSIONS: dict[str, dict[str, Any]] = {}
_APP_SESSIONS_LOCK = threading.RLock()


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
        "virtual_cursor": cursor,
        "bundle_id": record.get("bundle_id", ""),
        "process_id": record.get("process_id"),
        "window_id": record.get("window_id"),
        "window_bounds": record.get("window_bounds"),
        "window_title": record.get("window_title", ""),
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
        existing["approved"] = approved
        existing.setdefault("virtual_cursor", _fresh_virtual_cursor())
        return existing
    record = {
        "app_name": normalized,
        "app_session_id": f"app-{uuid.uuid4().hex}",
        "active": active,
        "approved": approved,
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


def _load_approved_apps() -> list[str]:
    path = _approval_store_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if isinstance(payload, dict):
        raw_apps = payload.get("approved_apps") or payload.get("approvedBundleIdentifiers") or []
    elif isinstance(payload, list):
        raw_apps = payload
    else:
        raw_apps = []
    return sorted({_normalize_app_name(app) for app in raw_apps if _normalize_app_name(app)})


def _save_approved_apps(apps: list[str]) -> None:
    path = _approval_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"approved_apps": sorted({_normalize_app_name(app) for app in apps if _normalize_app_name(app)})}))


def _is_app_approved(app_name: str | None) -> bool:
    wanted = _normalize_app_name(app_name).casefold()
    if not wanted:
        return False
    return any(app.casefold() == wanted for app in _load_approved_apps())


def _frontmost_identity_candidates(frontmost: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    def _append(raw_value: Any) -> None:
        normalized = _normalize_app_name(raw_value)
        if not normalized:
            return
        if any(existing.casefold() == normalized.casefold() for existing in candidates):
            return
        candidates.append(normalized)

    _append(frontmost.get("app_name"))
    _append(frontmost.get("bundle_name"))
    _append(frontmost.get("bundle_id"))
    return candidates


def _frontmost_matches_requested(requested_app: str | None, frontmost: dict[str, Any]) -> bool:
    wanted = _normalize_app_name(requested_app)
    if not wanted:
        return False
    return any(candidate.casefold() == wanted.casefold() for candidate in _frontmost_identity_candidates(frontmost))


def _frontmost_is_approved(frontmost: dict[str, Any], requested_app: str | None = None) -> bool:
    if requested_app:
        return _is_app_approved(requested_app) and _frontmost_matches_requested(requested_app, frontmost)
    return any(_is_app_approved(candidate) for candidate in _frontmost_identity_candidates(frontmost))


def _frontmost_approved_identity(frontmost: dict[str, Any], requested_app: str | None = None) -> str | None:
    wanted = _normalize_app_name(requested_app)
    if wanted and _is_app_approved(wanted) and _frontmost_matches_requested(wanted, frontmost):
        return wanted
    for candidate in _frontmost_identity_candidates(frontmost):
        if _is_app_approved(candidate):
            return candidate
    return None


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
    apps = _load_approved_apps()
    if not any(existing.casefold() == normalized.casefold() for existing in apps):
        apps.append(normalized)
        _save_approved_apps(apps)
    return {
        "success": True,
        "app_name": normalized,
        "approved_apps": _load_approved_apps(),
        "approval_store_path": str(_approval_store_path()),
    }


def revoke_app_impl(app_name: str) -> dict[str, Any]:
    normalized = _normalize_app_name(app_name)
    if not normalized:
        return {"success": False, "error": "app_name is required"}
    remaining = [app for app in _load_approved_apps() if app.casefold() != normalized.casefold()]
    _save_approved_apps(remaining)
    return {
        "success": True,
        "app_name": normalized,
        "approved_apps": remaining,
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
        if not session or not session.get("active"):
            return _session_required_error("getting app state")
        requested_app = _normalize_app_name(session.get("approval_name") or session.get("app_name"))

    if requested_app and not _is_app_approved(requested_app):
        pending_session = session or _ensure_app_session(requested_app, active=False)
        pending_session["approved"] = False
        if requested_app:
            pending_session["approval_name"] = requested_app
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
    approved_identity = _frontmost_approved_identity(frontmost, requested_app=requested_app)
    session = session or _ensure_app_session(frontmost_app_name or requested_app, active=True)
    session["active"] = True
    session["app_name"] = frontmost_app_name or requested_app or session.get("app_name", "desktop")
    session["approval_name"] = approved_identity or requested_app or session.get("approval_name") or session["app_name"]
    session["approved"] = approved_identity is not None
    session["bundle_id"] = str(frontmost.get("bundle_id") or "")
    session["process_id"] = frontmost.get("process_id")
    session["window_id"] = frontmost.get("window_id")
    session["window_bounds"] = frontmost.get("window_bounds")
    session["window_title"] = frontmost.get("window_title", "")
    session["screenshot_path"] = screenshot.get("path", "")
    session.setdefault("virtual_cursor", _fresh_virtual_cursor())
    _sync_session_artifacts(session)
    return {
        "success": True,
        "app_name": frontmost_app_name,
        "bundle_id": str(frontmost.get("bundle_id") or ""),
        "process_id": frontmost.get("process_id"),
        "window_id": frontmost.get("window_id"),
        "window_bounds": frontmost.get("window_bounds"),
        "window_title": frontmost.get("window_title", ""),
        "screenshot_path": screenshot.get("path", ""),
        "media_tag": screenshot.get("media_tag"),
        "accessibility_tree": [],
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "approval_required": False,
        "approved_apps": _load_approved_apps(),
        "approval_store_path": str(_approval_store_path()),
        "note": "Hermes adapter currently returns screenshot + frontmost window metadata, but not a full accessibility tree.",
    }


def type_text_impl(text: str, app_session_id: str | None = None) -> dict[str, Any]:
    session, error = _resolve_keyboard_session("typing", app_session_id=app_session_id)
    if error:
        return error
    result = _decode(computer_control(action="keystroke", text=text))
    return {"success": not bool(result.get("error")), **result, **_session_payload(session), "session_id": _SESSION_ID}


def press_key_impl(key: str, modifiers: list[str] | None = None, app_session_id: str | None = None) -> dict[str, Any]:
    session, error = _resolve_keyboard_session("pressing keys", app_session_id=app_session_id)
    if error:
        return error
    result = _decode(computer_control(action="keystroke", key=key, modifiers=modifiers or []))
    return {"success": not bool(result.get("error")), **result, **_session_payload(session), "session_id": _SESSION_ID}


def _unsupported(action: str) -> dict[str, Any]:
    return {
        "success": False,
        "supported": False,
        "error": f"{action} is not implemented in the Hermes computer-use adapter yet.",
    }


def click_impl(*, index: int | None = None, x: int | None = None, y: int | None = None,
               button: str = "left", click_count: int = 1, app_session_id: str | None = None) -> dict[str, Any]:
    session = _resolve_session(app_session_id=app_session_id)
    if not session:
        return _session_required_error("clicking")
    with _APP_SESSIONS_LOCK:
        if not session.get("active"):
            return _session_required_error("clicking")
        if _optional_dict(session, "pending_pointer_action"):
            return _pending_pointer_action_conflict("click", session)
        cursor = session.setdefault("virtual_cursor", _fresh_virtual_cursor())
        if x is not None:
            cursor["x"] = x
        if y is not None:
            cursor["y"] = y
        _record_pending_pointer_action(
            session,
            "click",
            x=cursor.get("x"),
            y=cursor.get("y"),
            index=index,
            button=button,
            click_count=click_count,
        )
        _sync_session_artifacts(session)
        return _preview_pointer_response("click", session)


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
