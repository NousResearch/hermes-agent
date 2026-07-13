"""Dashboard-facing remote browser bridge.

This module intentionally wraps the existing ``agent-browser`` CLI instead of
adding a new core model tool.  The dashboard uses it as a human hand-off
surface: the user can log in to sites inside the same server-side browser that
Hermes can later inspect/drive, without pasting passwords into chat.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SESSION = "hermes-dashboard-remote"
DEFAULT_VIEWPORT = "1280 720"


@dataclass
class BrowserCommandResult:
    ok: bool
    data: Dict[str, Any]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


def _agent_browser_path() -> str:
    path = shutil.which("agent-browser")
    if path:
        return path
    # Hermes installs a bundled Node toolchain under HERMES_HOME on many hosts.
    try:
        from hermes_cli.config import get_hermes_home

        candidate = get_hermes_home() / "node_modules" / ".bin" / "agent-browser"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return "agent-browser"


def _env(session: str = DEFAULT_SESSION) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AGENT_BROWSER_SESSION", session)
    # The dashboard often runs on Linux servers without Chromium's usable
    # sandbox.  This mirrors the existing browser-tool operational workaround.
    args = env.get("AGENT_BROWSER_ARGS", "")
    if "--no-sandbox" not in args:
        env["AGENT_BROWSER_ARGS"] = (args + ",--no-sandbox").strip(",")
    env.setdefault("AGENT_BROWSER_SCREENSHOT_FORMAT", "png")
    return env


def _run_agent_browser(*args: str, session: str = DEFAULT_SESSION, timeout: int = 60) -> BrowserCommandResult:
    cmd = [_agent_browser_path(), *args, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            env=_env(session),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return BrowserCommandResult(False, {}, "agent-browser is not installed")
    except subprocess.TimeoutExpired:
        return BrowserCommandResult(False, {}, f"agent-browser timed out after {timeout}s")

    raw = (proc.stdout or "").strip()
    parsed: Dict[str, Any]
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}

    if proc.returncode != 0:
        return BrowserCommandResult(False, parsed.get("data") or {}, parsed.get("error") or proc.stderr.strip() or raw)
    if parsed.get("success") is False:
        return BrowserCommandResult(False, parsed.get("data") or {}, str(parsed.get("error") or "agent-browser command failed"))
    return BrowserCommandResult(True, parsed.get("data") or {}, "")


def status(session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    session_result = _run_agent_browser("session", session=session, timeout=20)
    url_result = _run_agent_browser("get", "url", session=session, timeout=20)
    title_result = _run_agent_browser("get", "title", session=session, timeout=20)
    return {
        "session": session,
        "available": shutil.which(_agent_browser_path()) is not None or Path(_agent_browser_path()).exists(),
        "connected": url_result.ok,
        "url": str(url_result.data.get("url") or "") if url_result.ok else "",
        "title": str(title_result.data.get("title") or "") if title_result.ok else "",
        "error": "" if url_result.ok else url_result.error,
        "agent_browser_session": session_result.data,
    }


def open_url(url: str, session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    target = (url or "").strip()
    if not target:
        return BrowserCommandResult(False, {}, "URL is required").to_dict()
    result = _run_agent_browser("set", "viewport", *DEFAULT_VIEWPORT.split(), session=session, timeout=30)
    if not result.ok:
        # Non-fatal: opening the page is more important than viewport setup.
        pass
    opened = _run_agent_browser("open", target, session=session, timeout=90)
    return opened.to_dict() | {"status": status(session)}


def screenshot(session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = _run_agent_browser("screenshot", str(tmp_path), session=session, timeout=45)
        if not result.ok:
            return result.to_dict()
        data = tmp_path.read_bytes()
        return {
            "ok": True,
            "data": {
                "image_data_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
                "captured_at": time.time(),
                "status": status(session),
            },
            "error": "",
        }
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def click(x: int, y: int, session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    move = _run_agent_browser("mouse", "move", str(int(x)), str(int(y)), session=session, timeout=20)
    if not move.ok:
        return move.to_dict()
    down = _run_agent_browser("mouse", "down", session=session, timeout=20)
    if not down.ok:
        return down.to_dict()
    up = _run_agent_browser("mouse", "up", session=session, timeout=20)
    return up.to_dict()


def type_text(text: str, session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    # Use argv (not shell) so sensitive input never passes through shell parsing.
    return _run_agent_browser("keyboard", "type", text or "", session=session, timeout=60).to_dict()


def press_key(key: str, session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    return _run_agent_browser("press", (key or "").strip() or "Enter", session=session, timeout=30).to_dict()


def scroll(dy: int, session: str = DEFAULT_SESSION) -> Dict[str, Any]:
    return _run_agent_browser("mouse", "wheel", str(int(dy)), session=session, timeout=30).to_dict()
