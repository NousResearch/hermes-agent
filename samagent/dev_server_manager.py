"""Managed Local Dev Server Controller (`samagent/dev_server_manager.py`).

Lets the SamAgent Local Platform start, stop, hot-restart, health-check, and stream logs from the
generated application's standalone local HTTP server (`python3 app/main.py --serve --host 0.0.0.0 --port 3000`)
so the user can test development changes from VS Code live before production deployment.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, Optional


_MANAGED_DEV_PROC: Dict[str, Any] = {
    "proc": None,
    "workspace": None,
    "port": 3000,
    "log_file": Path("/tmp/samagent-local-dev-3000.log"),
}


def check_dev_server_health(port: int = 3000) -> Dict[str, Any]:
    """Probe http://127.0.0.1:<port>/healthz to verify if the local dev server is live."""
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=0.8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return {"running": True, "port": port, "url": f"http://127.0.0.1:{port}", "health": data}
    except Exception:
        pass
    return {"running": False, "port": port, "url": f"http://127.0.0.1:{port}", "health": None}


def get_dev_server_status(project_dir: Path, port: int = 3000) -> Dict[str, Any]:
    health = check_dev_server_health(port=port)
    log_path: Path = _MANAGED_DEV_PROC["log_file"]
    log_tail = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-25:])
        except Exception:
            pass
    proc: Optional[subprocess.Popen] = _MANAGED_DEV_PROC.get("proc")
    pid = proc.pid if (proc is not None and proc.poll() is None) else None
    return {
        "running": health["running"],
        "pid": pid,
        "port": port,
        "url": health["url"],
        "workspace": str(Path(project_dir).resolve()),
        "log_tail": log_tail,
    }


def stop_dev_server() -> Dict[str, Any]:
    proc: Optional[subprocess.Popen] = _MANAGED_DEV_PROC.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _MANAGED_DEV_PROC["proc"] = None
    return {"stopped": True}


def start_or_restart_dev_server(project_dir: Path, port: int = 3000) -> Dict[str, Any]:
    """Start (or hot-restart) `python3 app/main.py --serve --host 0.0.0.0 --port <port>`."""
    root = Path(project_dir).resolve()
    main_py = root / "app" / "main.py"
    if not main_py.exists():
        return {"running": False, "error": f"Missing {main_py}"}

    # If a managed process is already running, stop it first for a clean hot-restart
    stop_dev_server()

    # If an external process is already serving port 3000 and healthy, report it
    existing = check_dev_server_health(port=port)
    if existing["running"]:
        return get_dev_server_status(root, port=port)

    log_path: Path = _MANAGED_DEV_PROC["log_file"]
    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(main_py), "--serve", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _MANAGED_DEV_PROC["proc"] = proc
    _MANAGED_DEV_PROC["workspace"] = str(root)
    _MANAGED_DEV_PROC["port"] = port

    # Wait up to 1.5s for healthz
    for _ in range(15):
        if check_dev_server_health(port=port)["running"]:
            break
        time.sleep(0.1)

    return get_dev_server_status(root, port=port)
