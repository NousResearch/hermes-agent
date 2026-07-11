#!/usr/bin/env python3
"""Recover a wedged local llama.cpp server after Hermes abandons a request.

The script first tries the llama.cpp slots cancel action.  Older builds may
expose `/slots` monitoring but not the cancel action; for those deployments,
`--kill-listener` can terminate the process listening on the recovered base
URL's port.  llama-swap deployments normally respawn the child server on the
next request.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _request(url: str, *, api_key: str = "", method: str = "GET", timeout: float = 5.0):
    req = urllib.request.Request(url, method=method)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else None


def _load_key(path: str) -> str:
    if not path:
        return os.getenv("LLAMA_SERVER_API_KEY", "").strip()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.readline().strip()
    except OSError:
        return ""


def _management_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.lower().endswith("/v1"):
        path = path[:-3]
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized).rstrip("/")


def _active_slot_ids(base_url: str, api_key: str, timeout: float) -> list[int]:
    data = _request(
        _management_base_url(base_url) + "/slots",
        api_key=api_key,
        timeout=timeout,
    )
    if not isinstance(data, list):
        return []
    slot_ids: list[int] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("is_processing"):
            try:
                slot_ids.append(int(item.get("id")))
            except (TypeError, ValueError):
                pass
    return slot_ids


def _cancel_slots(base_url: str, api_key: str, slot_ids: list[int], timeout: float) -> bool:
    cancelled = False
    management_base_url = _management_base_url(base_url)
    for slot_id in slot_ids:
        url = f"{management_base_url}/slots/{slot_id}?action=cancel"
        try:
            _request(url, api_key=api_key, method="POST", timeout=timeout)
            cancelled = True
        except urllib.error.HTTPError as exc:
            if exc.code == 501:
                return False
            raise
    return cancelled


def _listener_pids(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def _kill_pids(pids: list[int], grace: float) -> bool:
    if not pids:
        return False
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(grace, 0.0)
    while time.monotonic() < deadline:
        if not any(_pid_exists(pid) for pid in pids):
            return True
        time.sleep(0.2)
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for pid in pids:
        try:
            os.kill(pid, sigkill)
        except ProcessLookupError:
            pass
    return True


def _pid_exists(pid: int) -> bool:
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        return Path(f"/proc/{pid}").exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("HERMES_RECOVERY_BASE_URL", ""),
        help="llama.cpp base URL; defaults to HERMES_RECOVERY_BASE_URL",
    )
    parser.add_argument("--api-key-file", default=os.getenv("LLAMA_SERVER_API_KEY_FILE", ""))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--kill-listener", action="store_true")
    parser.add_argument("--kill-grace", type=float, default=5.0)
    args = parser.parse_args(argv)

    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        print("invalid or missing local base URL", file=sys.stderr)
        return 2
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        print("refusing to kill non-loopback listener", file=sys.stderr)
        return 2

    api_key = _load_key(args.api_key_file)
    try:
        active_slots = _active_slot_ids(args.base_url, api_key, args.timeout)
    except Exception as exc:
        print(f"slot probe failed: {type(exc).__name__}", file=sys.stderr)
        active_slots = []

    if active_slots:
        try:
            if _cancel_slots(args.base_url, api_key, active_slots, args.timeout):
                print(f"cancelled active slots: {len(active_slots)}")
                return 0
        except Exception as exc:
            print(f"slot cancel failed: {type(exc).__name__}", file=sys.stderr)

    if not args.kill_listener:
        print("no slot cancel performed; listener kill disabled")
        return 1 if active_slots else 0

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    pids = _listener_pids(port)
    if not pids:
        print("no listening process found")
        return 0
    if _kill_pids(pids, args.kill_grace):
        print(f"killed listener process count: {len(pids)}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
