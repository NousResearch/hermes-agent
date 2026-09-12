#!/usr/bin/env python3
"""Headed Camoufox Playwright server on Xvfb. Bind WS to 127.0.0.1 only.

Optional HERMES_BROWSER_PROXY=socks5://PEER_WG_IP:1080 sends browser
public traffic out another WireGuard peer. Direct Camoufox() launches
must pass the same proxy or they leak the VPS WAN.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time

PORT = int(os.environ.get("CAMOUFOX_PORT", "9377"))
WS_PATH = os.environ.get("CAMOUFOX_WS_PATH", "camoufox")
DISPLAY_NUM = os.environ.get("DISPLAY_NUM", "98")
PROXY = os.environ.get("HERMES_BROWSER_PROXY", "").strip()

xvfb_proc = None


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(code)


def start_xvfb() -> None:
    global xvfb_proc
    probe = subprocess.run(
        ["xdpyinfo", f":{DISPLAY_NUM}"],
        capture_output=True,
        timeout=2,
    )
    if probe.returncode == 0:
        os.environ["DISPLAY"] = f":{DISPLAY_NUM}"
        print(f"Xvfb already running on :{DISPLAY_NUM}", flush=True)
        return
    xvfb_proc = subprocess.Popen(
        ["Xvfb", f":{DISPLAY_NUM}", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{DISPLAY_NUM}"
    time.sleep(1)
    print(f"Xvfb started on :{DISPLAY_NUM}", flush=True)


def camel_case(s: str) -> str:
    if len(s) < 2:
        return s
    c = "".join(x.capitalize() for x in s.lower().split("_"))
    return c[0].lower() + c[1:]


def apply_proxy(config: dict) -> None:
    if not PROXY:
        return
    if "0.0.0.0" in PROXY or PROXY.endswith("://*") or "://[" in PROXY:
        _die("HERMES_BROWSER_PROXY must be a mesh SOCKS URL, never 0.0.0.0")
    config["proxy"] = {"server": PROXY}
    prefs = config.setdefault("firefox_user_prefs", {})
    prefs["network.proxy.socks_remote_dns"] = True
    prefs["media.peerconnection.enabled"] = False
    print(f"browser proxy {PROXY}", flush=True)


def main() -> int:
    global xvfb_proc
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    start_xvfb()

    from camoufox.server import LAUNCH_SCRIPT, get_nodejs
    from camoufox.utils import launch_options
    from pathlib import Path

    config = launch_options(headless=False, geoip=True)
    config = {k: v for k, v in config.items() if v is not None}
    apply_proxy(config)
    config["port"] = PORT
    config["host"] = "127.0.0.1"
    config["ws_path"] = WS_PATH

    data = json.dumps({camel_case(k): v for k, v in config.items()}).encode()
    b64 = base64.b64encode(data).decode()
    nodejs = get_nodejs()
    env = {**os.environ, "DISPLAY": f":{DISPLAY_NUM}"}
    process = subprocess.Popen(
        [nodejs, str(LAUNCH_SCRIPT)],
        cwd=Path(nodejs).parent / "package",
        stdin=subprocess.PIPE,
        env=env,
    )
    if process.stdin:
        process.stdin.write(b64.encode())
        process.stdin.close()
    print(f"Camoufox WS ws://127.0.0.1:{PORT}/{WS_PATH}", flush=True)
    process.wait()
    if xvfb_proc:
        xvfb_proc.terminate()
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
