#!/usr/bin/env python3
"""Windows-friendly prototype launcher for Hermes配置宝典 / Setup Codex.

Intended PyInstaller name: Hermes配置宝典.exe

The launcher is deliberately conservative:
- binds the dashboard to 127.0.0.1 by default;
- never passes --host 0.0.0.0 or --insecure;
- only starts `hermes dashboard` when /api/status is not already reachable;
- opens /setup-codex in the user's browser after the dashboard responds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9119
STATUS_PATH = "/api/status"
SETUP_CODEX_PATH = "/setup-codex"


@dataclass
class ProbeResult:
    ok: bool
    detail: str


def _url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def probe_dashboard(host: str, port: int, timeout: float = 1.5) -> ProbeResult:
    try:
        with urllib.request.urlopen(_url(host, port, STATUS_PATH), timeout=timeout) as response:
            body = response.read(1024).decode("utf-8", errors="replace")
            if response.status == 200:
                try:
                    data = json.loads(body or "{}")
                    return ProbeResult(True, str(data.get("status") or "running"))
                except json.JSONDecodeError:
                    return ProbeResult(True, "running")
            return ProbeResult(False, f"HTTP {response.status}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ProbeResult(False, str(exc))


def start_dashboard(host: str, port: int) -> subprocess.Popen:
    hermes = shutil.which("hermes")
    if not hermes:
        raise RuntimeError(
            "Hermes CLI was not found on PATH. Install Hermes or open a shell where `hermes --version` works."
        )
    cmd = [hermes, "dashboard", "--host", host, "--port", str(port)]
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)


def wait_until_ready(host: str, port: int, seconds: float) -> ProbeResult:
    deadline = time.monotonic() + seconds
    last = ProbeResult(False, "not checked")
    while time.monotonic() < deadline:
        last = probe_dashboard(host, port, timeout=1.0)
        if last.ok:
            return last
        time.sleep(0.4)
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open Hermes配置宝典 / Setup Codex safely")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Dashboard host; default is 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dashboard port; default is 9119")
    parser.add_argument("--wait", type=float, default=20.0, help="Seconds to wait for dashboard startup")
    parser.add_argument("--no-open", action="store_true", help="Start/probe only; do not open a browser")
    args = parser.parse_args(argv)

    if args.host != DEFAULT_HOST:
        print("Refusing non-localhost host. Setup Codex launcher only binds to 127.0.0.1 by default.", file=sys.stderr)
        return 2

    probe = probe_dashboard(args.host, args.port)
    if not probe.ok:
        print(f"Dashboard not reachable yet ({probe.detail}); starting Hermes dashboard...")
        try:
            start_dashboard(args.host, args.port)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        probe = wait_until_ready(args.host, args.port, args.wait)

    if not probe.ok:
        print(
            "Hermes dashboard did not become ready. Try manually: "
            f"hermes dashboard --host {DEFAULT_HOST} --port {args.port}",
            file=sys.stderr,
        )
        return 1

    target = _url(args.host, args.port, SETUP_CODEX_PATH)
    print(f"Opening {target}")
    if not args.no_open:
        webbrowser.open(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
