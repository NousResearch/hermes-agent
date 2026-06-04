"""Hermes-adapted Codex-style computer-use MCP server.

This is not a direct embed of OpenAI's proprietary computer-use plugin.
Instead, it exposes a compatible-ish MCP surface backed by Hermes' local
macOS computer_control backend where possible, and returns explicit
"not implemented" results for actions Hermes does not support yet.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - runtime import guard
    FastMCP = None  # type: ignore[assignment]

from tools.computer_control_tool import computer_control


def _decode(payload: str) -> dict[str, Any]:
    data = json.loads(payload)
    assert isinstance(data, dict)
    return data


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


def get_app_state_impl(app_name: str | None = None) -> dict[str, Any]:
    if app_name:
        activated = _decode(computer_control(action="activate_app", app_name=app_name))
        if activated.get("error"):
            return {"success": False, "error": activated["error"]}

    frontmost = _decode(computer_control(action="frontmost_app"))
    if frontmost.get("error"):
        return {"success": False, "error": frontmost["error"]}

    screenshot = _decode(computer_control(action="screenshot"))
    if screenshot.get("error"):
        return {"success": False, "error": screenshot["error"]}

    return {
        "success": True,
        "app_name": frontmost.get("app_name", ""),
        "window_title": frontmost.get("window_title", ""),
        "screenshot_path": screenshot.get("path", ""),
        "media_tag": screenshot.get("media_tag"),
        "accessibility_tree": [],
        "note": "Hermes adapter currently returns screenshot + frontmost window metadata, but not a full accessibility tree.",
    }


def type_text_impl(text: str) -> dict[str, Any]:
    result = _decode(computer_control(action="keystroke", text=text))
    return {"success": not bool(result.get("error")), **result}


def press_key_impl(key: str, modifiers: list[str] | None = None) -> dict[str, Any]:
    result = _decode(computer_control(action="keystroke", key=key, modifiers=modifiers or []))
    return {"success": not bool(result.get("error")), **result}


def _unsupported(action: str) -> dict[str, Any]:
    return {
        "success": False,
        "supported": False,
        "error": f"{action} is not implemented in the Hermes computer-use adapter yet.",
    }


def click_impl(*, index: int | None = None, x: int | None = None, y: int | None = None,
               button: str = "left", click_count: int = 1) -> dict[str, Any]:
    return _unsupported("click")


def perform_secondary_action_impl(index: int, action_name: str) -> dict[str, Any]:
    return _unsupported("perform_secondary_action")


def scroll_impl(index: int | None = None, x: int | None = None, y: int | None = None,
                delta_y: int = 0) -> dict[str, Any]:
    return _unsupported("scroll")


def drag_impl(start_x: int, start_y: int, end_x: int, end_y: int) -> dict[str, Any]:
    return _unsupported("drag")


def set_value_impl(index: int, value: str) -> dict[str, Any]:
    return _unsupported("set_value")


mcp = FastMCP("hermes-computer-use-adapter") if FastMCP else None

if mcp:
    @mcp.tool()
    def list_apps(limit: int = 100) -> dict[str, Any]:
        """List running apps and a sample of installed apps visible to the Hermes adapter."""
        return list_apps_impl(limit=limit)

    @mcp.tool()
    def get_app_state(app_name: str | None = None) -> dict[str, Any]:
        """Activate an app if requested, then return a fresh screenshot and frontmost window metadata."""
        return get_app_state_impl(app_name=app_name)

    @mcp.tool()
    def type_text(text: str) -> dict[str, Any]:
        """Type literal text via Hermes' computer-control backend."""
        return type_text_impl(text)

    @mcp.tool()
    def press_key(key: str, modifiers: list[str] | None = None) -> dict[str, Any]:
        """Press one key or key combination via Hermes' computer-control backend."""
        return press_key_impl(key, modifiers or [])

    @mcp.tool()
    def click(index: int | None = None, x: int | None = None, y: int | None = None,
              button: str = "left", click_count: int = 1) -> dict[str, Any]:
        """Reserved for future pointer support. Returns an explicit unsupported result for now."""
        return click_impl(index=index, x=x, y=y, button=button, click_count=click_count)

    @mcp.tool()
    def perform_secondary_action(index: int, action_name: str) -> dict[str, Any]:
        """Reserved for future accessibility action support."""
        return perform_secondary_action_impl(index=index, action_name=action_name)

    @mcp.tool()
    def scroll(index: int | None = None, x: int | None = None, y: int | None = None,
               delta_y: int = 0) -> dict[str, Any]:
        """Reserved for future scroll support."""
        return scroll_impl(index=index, x=x, y=y, delta_y=delta_y)

    @mcp.tool()
    def drag(start_x: int, start_y: int, end_x: int, end_y: int) -> dict[str, Any]:
        """Reserved for future drag support."""
        return drag_impl(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)

    @mcp.tool()
    def set_value(index: int, value: str) -> dict[str, Any]:
        """Reserved for future settable accessibility elements."""
        return set_value_impl(index=index, value=value)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    if not mcp:
        raise SystemExit("FastMCP is unavailable. Install the mcp package first.")
    mcp.run()
