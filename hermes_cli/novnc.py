"""``hermes novnc`` implementation — noVNC + websockify lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import socket
import time
import threading
from pathlib import Path
from typing import Optional

NOVNC_DIR = Path.home() / "noVNC"
PID_FILE = Path.home() / ".hermes" / "novnc.pid"
LOG_FILE = Path.home() / ".hermes" / "logs" / "novnc.log"
DEFAULT_WEB_ROOT = NOVNC_DIR


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _check_vnc_reachable(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        return s.connect_ex((host, port)) == 0


def _find_websockify() -> list[str]:
    """Return cmd prefix to invoke websockify (prefer exe, fallback to module)."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        hermes_venv = Path(local_app_data) / "hermes" / "hermes-agent" / "venv"
    else:
        hermes_venv = Path.home() / "AppData" / "Local" / "hermes" / "hermes-agent" / "venv"
    candidates = [
        hermes_venv / "Scripts" / "websockify.exe",
        hermes_venv / "Scripts" / "websockify",
        Path.home() / ".local" / "bin" / "websockify",
    ]
    for c in candidates:
        if c.is_file():
            return [str(c)]
    # Fallback: python -m websockify
    return [sys.executable, "-m", "websockify"]


def _save_state(pid: int, port: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps({"pid": pid, "port": port}))


def _read_state() -> tuple[Optional[int], int]:
    """Return (pid, port).  Plain-text pid files are migrated automatically."""
    if not PID_FILE.is_file():
        return None, 6080
    try:
        raw = PID_FILE.read_text().strip()
        if raw.startswith("{"):
            data = json.loads(raw)
            return data.get("pid"), data.get("port", 6080)
        # legacy plain-text pid
        return int(raw), 6080
    except (ValueError, OSError, json.JSONDecodeError):
        return None, 6080


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def _apply_auth_patch(password: str) -> None:
    """Patch noVNC vnc.html + ui.js to auto-submit VNC password."""
    vnc_html = NOVNC_DIR / "vnc.html"
    ui_js = NOVNC_DIR / "app" / "ui.js"

    if not vnc_html.is_file() or not ui_js.is_file():
        print("! Cannot apply auto-auth patch: vnc.html or app/ui.js not found", file=sys.stderr)
        return

    html_content = vnc_html.read_text(encoding="utf-8")
    if f'value="{password}"' in html_content:
        print("  Auto-auth patch already applied")
        return

    html_patched = html_content.replace(
        '<div id="noVNC_password_block">',
        '<div id="noVNC_password_block" style="display:none">',
    ).replace(
        'id="noVNC_password_input" type="password">',
        f'id="noVNC_password_input" type="password" value="{password}">',
    )
    vnc_html.write_text(html_patched, encoding="utf-8")

    js_content = ui_js.read_text(encoding="utf-8")
    if "Auto-fill password" in js_content:
        return

    js_needle = "credentials(e) {"
    js_patch = """credentials(e) {
    // Auto-fill password and auto-submit (injected by hermes novnc)
    const pwInput = document.getElementById("noVNC_password_input");
    if (pwInput && pwInput.value) {
        const username = document.getElementById("noVNC_username_input").value;
        UI.rfb.sendCredentials({ username: username, password: pwInput.value });
        UI.reconnectPassword = pwInput.value;
        pwInput.value = "";
        document.getElementById('noVNC_credentials_dlg')
            .classList.remove('noVNC_open');
        return;
    }
"""
    if js_needle in js_content:
        js_content = js_content.replace(js_needle, js_patch, 1)
        ui_js.write_text(js_content, encoding="utf-8")
        print(f"  Auto-auth patch applied (password: {'*' * len(password)})")
    else:
        print("! Could not find credentials() method in app/ui.js — patch skipped", file=sys.stderr)


def cmd_novnc_start(
    port: int = 6080,
    vnc_host: str = "127.0.0.1",
    vnc_port: int = 5900,
    password: Optional[str] = None,
) -> None:
    """Start a noVNC session."""
    if not NOVNC_DIR.is_dir() or not (NOVNC_DIR / "vnc.html").is_file():
        print(
            f"! noVNC not found at {NOVNC_DIR}. Clone it first:\n"
            f"  git clone --depth 1 https://github.com/novnc/noVNC.git \"{NOVNC_DIR}\"",
            file=sys.stderr,
        )
        sys.exit(1)

    existing_pid, existing_port = _read_state()
    if existing_pid and _is_running(existing_pid):
        print(f"  noVNC already running (PID {existing_pid}, port {existing_port})")
        if password:
            _apply_auth_patch(password)
        print(f"  → http://localhost:{existing_port}/vnc.html")
        return

    # Pre-flight checks
    if _is_port_in_use("127.0.0.1", port):
        print(
            f"! Port {port} is already in use on this machine. "
            f"Try a different port: --port {port + 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _check_vnc_reachable(vnc_host, vnc_port):
        print(
            f"! Cannot reach VNC server at {vnc_host}:{vnc_port}.\n"
            f"  Make sure the VNC server is running and reachable before starting noVNC.",
            file=sys.stderr,
        )
        sys.exit(1)

    websockify_cmd = _find_websockify()
    target = f"{vnc_host}:{vnc_port}"
    cmd = [*websockify_cmd, "--web", str(DEFAULT_WEB_ROOT), str(port), target]

    print(f"  Starting websockify on port {port} (→ {target})")
    print(f"  Web root: {DEFAULT_WEB_ROOT}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_FILE.open("w", encoding="utf-8")
    startup_info = None
    if sys.platform == "win32":
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        startupinfo=startup_info,
    )
    _save_state(proc.pid, port)
    print(f"  Started (PID {proc.pid})")

    # Wait briefly for startup, polling log for errors
    time.sleep(1.5)
    if proc.poll() is not None:
        log_handle.flush()
        recent_log = LOG_FILE.read_text(encoding="utf-8").strip()
        print(f"! websockify exited immediately (code {proc.returncode})", file=sys.stderr)
        if recent_log:
            print("  Log output:", file=sys.stderr)
            for line in recent_log.splitlines()[-8:]:
                print(f"    {line}", file=sys.stderr)
        else:
            print(
                "  Is the VNC server running? Is the port already in use?",
                file=sys.stderr,
            )
        PID_FILE.unlink(missing_ok=True)
        sys.exit(1)

    if password:
        _apply_auth_patch(password)

    import webbrowser
    url = f"http://localhost:{port}/vnc.html"
    webbrowser.open(url)
    print(f"  → {url}")


def cmd_novnc_stop() -> None:
    """Stop the running noVNC session."""
    pid, _ = _read_state()
    if pid is None:
        print("  No noVNC PID file found")
        return

    if not _is_running(pid):
        print(f"  noVNC (PID {pid}) is not running")
        PID_FILE.unlink(missing_ok=True)
        return

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, 15)
            time.sleep(0.5)
            if _is_running(pid):
                os.kill(pid, 9)
        print(f"  Stopped noVNC (PID {pid})")
    except ProcessLookupError:
        print(f"  Process {pid} already exited")
    PID_FILE.unlink(missing_ok=True)


def cmd_novnc_status() -> None:
    """Check noVNC session status."""
    pid, port = _read_state()
    if pid is None:
        print("  noVNC is not running (no PID file)")
        return

    if _is_running(pid):
        print(f"  noVNC is running (PID {pid})")
        print(f"  → http://localhost:{port}/vnc.html")
    else:
        print(f"  noVNC is not running (stale PID {pid})")
        PID_FILE.unlink(missing_ok=True)


def novnc_command(args) -> None:
    """Dispatch ``hermes novnc <subcommand>``."""
    sub = getattr(args, "novnc_command", None)
    if sub is None:
        print("Usage: hermes novnc {start|stop|status}")
        print("  start   Launch websockify + open browser")
        print("  stop    Kill the running websockify process")
        print("  status  Show PID and URL if running")
        sys.exit(0)
    if sub == "start":
        cmd_novnc_start(
            port=args.port,
            vnc_host=args.vnc_host,
            vnc_port=args.vnc_port,
            password=args.password,
        )
    elif sub == "stop":
        cmd_novnc_stop()
    elif sub == "status":
        cmd_novnc_status()
    else:
        print(f"Unknown subcommand: {sub}", file=sys.stderr)
        sys.exit(1)
