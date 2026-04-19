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
import uuid
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


def _normalize_app_name(app_name: str | None) -> str:
    return str(app_name or "").strip()


def _session_key(app_name: str | None) -> str:
    normalized = _normalize_app_name(app_name) or "desktop"
    return normalized.casefold()


def _approval_store_path() -> Path:
    root = get_hermes_home() / "computer-use"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ComputerUseAppApprovals.json"


def _session_payload(record: dict[str, Any]) -> dict[str, Any]:
    cursor = dict(record.get("virtual_cursor") or {"x": None, "y": None, "detached": True, "visible": True})
    overlay_path = str(record.get("overlay_screenshot_path", "") or "")
    session_state_path = str(record.get("session_state_path", "") or "")
    pending_pointer_action = record.get("pending_pointer_action")
    if isinstance(pending_pointer_action, dict):
        pending_pointer_action = dict(pending_pointer_action)
    else:
        pending_pointer_action = None
    return {
        "app_name": record["app_name"],
        "app_session_id": record["app_session_id"],
        "active": bool(record.get("active")),
        "approved": bool(record.get("approved")),
        "virtual_cursor": cursor,
        "window_title": record.get("window_title", ""),
        "screenshot_path": record.get("screenshot_path", ""),
        "overlay_screenshot_path": overlay_path,
        "overlay_media_tag": _media_tag_for_path(overlay_path),
        "session_state_path": session_state_path,
        "pending_pointer_action": pending_pointer_action,
    }


def _fresh_virtual_cursor() -> dict[str, Any]:
    return {"x": None, "y": None, "detached": True, "visible": True}


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


def list_active_sessions_impl() -> dict[str, Any]:
    return {
        "success": True,
        "session_id": _SESSION_ID,
        "active_sessions": _active_sessions(),
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


def _preview_pointer_response(action: str, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": False,
        "supported": False,
        "preview_only": True,
        "session_id": _SESSION_ID,
        **_session_payload(session),
        "error": f"{action} is not implemented in the Hermes computer-use adapter yet.",
    }


def stop_app_session_impl(app_name: str | None = None, app_session_id: str | None = None) -> dict[str, Any]:
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
        requested_app = _normalize_app_name(session.get("app_name"))

    if requested_app and not _is_app_approved(requested_app):
        pending_session = _ensure_app_session(requested_app, active=False)
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

    screenshot = _decode(computer_control(action="screenshot"))
    if screenshot.get("error"):
        return {"success": False, "error": screenshot["error"]}

    frontmost_app_name = _normalize_app_name(frontmost.get("app_name"))
    session = session or _ensure_app_session(frontmost_app_name or requested_app, active=True)
    session["active"] = True
    session["app_name"] = frontmost_app_name or requested_app or session.get("app_name", "desktop")
    session["approved"] = _is_app_approved(session["app_name"])
    session["window_title"] = frontmost.get("window_title", "")
    session["screenshot_path"] = screenshot.get("path", "")
    session.setdefault("virtual_cursor", _fresh_virtual_cursor())
    _sync_session_artifacts(session)
    return {
        "success": True,
        "app_name": frontmost_app_name,
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
    session = _resolve_session(app_session_id=app_session_id)
    if not session:
        return _session_required_error("typing")
    result = _decode(computer_control(action="keystroke", text=text))
    return {"success": not bool(result.get("error")), **result, **_session_payload(session), "session_id": _SESSION_ID}


def press_key_impl(key: str, modifiers: list[str] | None = None, app_session_id: str | None = None) -> dict[str, Any]:
    session = _resolve_session(app_session_id=app_session_id)
    if not session:
        return _session_required_error("pressing keys")
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
    def set_value(index: int, value: str) -> dict[str, Any]:
        """Reserved for future settable accessibility elements."""
        return set_value_impl(index=index, value=value)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    if not mcp:
        raise SystemExit("FastMCP is unavailable. Install the mcp package first.")
    mcp.run()
