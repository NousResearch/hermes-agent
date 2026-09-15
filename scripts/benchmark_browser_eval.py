"""Quick benchmark: subprocess eval vs supervisor-WS eval.

Runs both paths against the same live Chrome and prints a comparison table.
Not a pytest — a script you run manually for the PR description.

Usage:
    .venv/bin/python scripts/benchmark_browser_eval.py [--iterations N]
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import json


def _find_chrome() -> str:
    for c in ("google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(c)
        if p:
            return p
    print("No Chrome binary found.", file=sys.stderr)
    sys.exit(1)


def _start_chrome(port: int):
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
    raise RuntimeError("Chrome didn't expose CDP")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--port", type=int, default=9333)
    args = parser.parse_args()

    proc, profile, cdp_url = _start_chrome(args.port)
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        # Warm up: start the supervisor, navigate to a page.
        supervisor = SUPERVISOR_REGISTRY.get_or_start(
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
        print(f"browser_eval benchmark — {args.iterations} iterations of `1 + 1`")
        print("-" * 90)
        print(fmt("supervisor WS (Runtime.evaluate)", ws_times))
        print(fmt("agent-browser subprocess (eval)", sub_times))
        if ws_times and sub_times:
            speedup = statistics.mean(sub_times) / statistics.mean(ws_times)
            print()
            print(f"Speedup: {speedup:.1f}x (mean)")

    finally:
        SUPERVISOR_REGISTRY.stop_all()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
