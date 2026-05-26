"""Quick benchmark: subprocess eval vs supervisor-WS eval.

Runs both paths against the same live Chrome and prints a comparison table.
Not a pytest — a script you run manually for the PR description.

Usage:
    .venv/bin/python scripts/benchmark_browser_eval.py [--iterations N]
"""
from __future__ import annotations

import argparse
import platform
import socket
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import json

try:
    from hermes_cli.browser_connect import get_chrome_debug_candidates
except Exception:
    get_chrome_debug_candidates = None  # type: ignore[assignment]


class BenchmarkUnavailable(RuntimeError):
    """Raised when this host cannot expose or reach the live CDP endpoint."""


def _find_chrome() -> str:
    if get_chrome_debug_candidates is not None:
        try:
            candidates = get_chrome_debug_candidates(platform.system())
        except Exception:
            candidates = []
    else:
        candidates = []

    for candidate in candidates:
        if candidate:
            return candidate
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        p = shutil.which(c)
        if p:
            return p
    print("No Chrome binary found.", file=sys.stderr)
    sys.exit(1)


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return 9333


def _loopback_bind_probe() -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
        return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _start_chrome(port: int):
    ok, reason = _loopback_bind_probe()
    if not ok:
        raise BenchmarkUnavailable(
            "Loopback TCP bind is unavailable, so Chrome cannot expose the "
            f"CDP endpoint required by this live benchmark ({reason})."
        )
    if port == 0:
        port = _free_port()
    profile = tempfile.mkdtemp(prefix="hermes-bench-eval-")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as r:
                info = json.loads(r.read().decode())
                return proc, profile, info["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    shutil.rmtree(profile, ignore_errors=True)
    raise BenchmarkUnavailable("Chrome didn't expose CDP")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--port", type=int, default=9333)
    parser.add_argument(
        "--cdp-url",
        help="Attach to an existing browser WebSocket debugger URL instead of starting Chrome.",
    )
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Exit 0 with an unavailable notice when Chrome/CDP cannot run on this host.",
    )
    args = parser.parse_args()

    proc = None
    profile = None
    try:
        if args.cdp_url:
            cdp_url = args.cdp_url
        else:
            proc, profile, cdp_url = _start_chrome(args.port)
    except BenchmarkUnavailable as exc:
        print(f"browser_eval benchmark unavailable: {exc}", file=sys.stderr)
        if args.allow_unavailable:
            return
        sys.exit(3)

    supervisor_registry = None
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY as supervisor_registry

        # Warm up: start the supervisor, navigate to a page.
        supervisor = supervisor_registry.get_or_start(
            task_id="bench-eval", cdp_url=cdp_url
        )
        # Give it a moment to attach.
        time.sleep(1.0)

        # Sanity check: one eval over WS should succeed.
        sanity = supervisor.evaluate_runtime("1 + 1")
        if not sanity.get("ok") or sanity.get("result") != 2:
            print(f"sanity check failed: {sanity}", file=sys.stderr)
            sys.exit(2)

        # ── Bench 1: supervisor WS path ──────────────────────────────────
        ws_times: list[float] = []
        for _ in range(args.iterations):
            t0 = time.monotonic()
            out = supervisor.evaluate_runtime("1 + 1")
            t1 = time.monotonic()
            assert out.get("ok"), out
            ws_times.append((t1 - t0) * 1000)

        # ── Bench 2: agent-browser subprocess path ────────────────────────
        # Skip if agent-browser isn't installed — the WS bench still tells
        # us what we need.
        if shutil.which("agent-browser") is None and shutil.which("npx") is None:
            print("agent-browser CLI not found — skipping subprocess bench.")
            sub_times = []
        else:
            from tools.browser_tool import _run_browser_command, _last_session_key
            task_id = _last_session_key("bench-eval")
            sub_times = []
            for _ in range(args.iterations):
                t0 = time.monotonic()
                _run_browser_command(task_id, "eval", ["1 + 1"])
                t1 = time.monotonic()
                sub_times.append((t1 - t0) * 1000)

        def fmt(name: str, ts: list[float]) -> str:
            if not ts:
                return f"  {name:<40} (skipped)"
            mean = statistics.mean(ts)
            median = statistics.median(ts)
            mn, mx = min(ts), max(ts)
            return (
                f"  {name:<40} mean={mean:>7.2f}ms  median={median:>7.2f}ms  "
                f"min={mn:>7.2f}ms  max={mx:>7.2f}ms"
            )

        print()
        port = cdp_url.split("/devtools/", 1)[0].rsplit(":", 1)[-1]
        print(f"browser_eval benchmark — {args.iterations} iterations of `1 + 1` on CDP port {port}")
        print("-" * 90)
        print(fmt("supervisor WS (Runtime.evaluate)", ws_times))
        print(fmt("agent-browser subprocess (eval)", sub_times))
        if ws_times and sub_times:
            speedup = statistics.mean(sub_times) / statistics.mean(ws_times)
            print()
            print(f"Speedup: {speedup:.1f}x (mean)")

    finally:
        if supervisor_registry is not None:
            supervisor_registry.stop_all()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        if profile is not None:
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
