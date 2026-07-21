#!/usr/bin/env python3
"""Measure dashboard startup, request latency, and process-tree RSS.

Run from the repository root with its development environment active:

    python scripts/benchmark_dashboard_memory.py --mode full --runs 3
    python scripts/benchmark_dashboard_memory.py --mode light --runs 3

The benchmark drives the real CLI entrypoint and exercises the first page plus
status, session metadata/transcript, files, logs, and config. This catches lazy
imports and request-time growth that a startup-only RSS sample would miss.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import psutil


READY_RE = re.compile(r"HERMES_(?:DASHBOARD|BACKEND)_READY port=(\d+)")
MIB = 1024 * 1024


def _process_tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

    total = 0
    for process in processes:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += process.memory_info().rss
    return total


def _request(url: str, *, timeout: float, token: str) -> tuple[float, bytes]:
    started = time.perf_counter()
    request = Request(url, headers={"X-Hermes-Session-Token": token})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback benchmark
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
    return (time.perf_counter() - started) * 1000, body


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _run_once(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hermes-dashboard-bench-") as tmp:
        root = Path(tmp)
        home = root / "home"
        web_dist = root / "web-dist"
        workspace = root / "workspace"
        home.mkdir()
        web_dist.mkdir()
        workspace.mkdir()
        (web_dist / "assets").mkdir()
        (web_dist / "index.html").write_text(
            "<!doctype html><title>Hermes benchmark</title>", encoding="utf-8"
        )
        (workspace / "benchmark.txt").write_text(
            "Hermes dashboard memory benchmark\n", encoding="utf-8"
        )
        (home / "logs").mkdir()
        (home / "logs" / "agent.log").write_text(
            "INFO dashboard benchmark fixture\n", encoding="utf-8"
        )
        (home / "config.yaml").write_text(
            f"model: benchmark/model\nterminal:\n  cwd: {workspace}\n",
            encoding="utf-8",
        )
        from hermes_state import SessionDB

        database = SessionDB(db_path=home / "state.db")
        database.create_session(
            "dashboard-benchmark-session", "cli", model="benchmark/model"
        )
        database.append_message(
            "dashboard-benchmark-session",
            "user",
            "benchmark transcript message",
        )
        database.close()

        command = [
            args.python,
            "-m",
            "hermes_cli.main",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--no-open",
        ]
        if args.mode == "light":
            command.append("--light")

        env = {
            **os.environ,
            "HERMES_HOME": str(home),
            "HERMES_DASHBOARD_SESSION_TOKEN": "dashboard-benchmark-token",
            "HERMES_DASHBOARD_FILES_ROOT": str(workspace),
            "HERMES_WEB_DIST": str(web_dist),
            "PYTHONUNBUFFERED": "1",
        }
        token = env["HERMES_DASHBOARD_SESSION_TOKEN"]
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        lines: queue.Queue[str | None] = queue.Queue()
        output: list[str] = []
        stop_sampling = threading.Event()
        peak_rss = 0

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output.append(line.rstrip())
                lines.put(line)
            lines.put(None)

        def sample_memory() -> None:
            nonlocal peak_rss
            while not stop_sampling.wait(0.05):
                peak_rss = max(peak_rss, _process_tree_rss(process.pid))

        threading.Thread(target=read_output, daemon=True).start()
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        started = time.perf_counter()

        try:
            deadline = started + args.startup_timeout
            port = None
            while time.perf_counter() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"dashboard exited with {process.returncode}: {' | '.join(output[-12:])}"
                    )
                try:
                    line = lines.get(timeout=min(0.25, deadline - time.perf_counter()))
                except queue.Empty:
                    continue
                if line is None:
                    continue
                match = READY_RE.search(line)
                if match:
                    port = int(match.group(1))
                    break
            if port is None:
                raise TimeoutError(
                    f"dashboard did not announce readiness: {' | '.join(output[-12:])}"
                )

            startup_ms = (time.perf_counter() - started) * 1000
            time.sleep(args.settle_seconds)
            ready_rss = _process_tree_rss(process.pid)
            base_url = f"http://127.0.0.1:{port}"

            root_ms, root_body = _request(
                f"{base_url}/", timeout=args.request_timeout, token=token
            )
            time.sleep(args.settle_seconds)
            root_rss = _process_tree_rss(process.pid)

            status_ms, status_body = _request(
                f"{base_url}/api/status", timeout=args.request_timeout, token=token
            )
            json.loads(status_body)
            time.sleep(args.settle_seconds)
            status_rss = _process_tree_rss(process.pid)

            sessions_ms, sessions_body = _request(
                f"{base_url}/api/sessions?limit=20",
                timeout=args.request_timeout,
                token=token,
            )
            json.loads(sessions_body)
            time.sleep(args.settle_seconds)
            sessions_rss = _process_tree_rss(process.pid)

            workflow_started = time.perf_counter()
            daily_urls = (
                f"{base_url}/api/sessions/dashboard-benchmark-session",
                f"{base_url}/api/sessions/dashboard-benchmark-session/messages?limit=20",
                f"{base_url}/api/files",
                f"{base_url}/api/files/read?path=benchmark.txt",
                f"{base_url}/api/logs?file=agent&lines=100",
                f"{base_url}/api/config",
            )
            for url in daily_urls:
                _elapsed, body = _request(
                    url,
                    timeout=args.request_timeout,
                    token=token,
                )
                json.loads(body)
            workflow_ms = (time.perf_counter() - workflow_started) * 1000
            time.sleep(args.settle_seconds)
            daily_use_rss = _process_tree_rss(process.pid)

            repeated: list[float] = []
            for _ in range(args.requests):
                elapsed, _body = _request(
                    f"{base_url}/api/status",
                    timeout=args.request_timeout,
                    token=token,
                )
                repeated.append(elapsed)

            peak_rss = max(peak_rss, _process_tree_rss(process.pid))
            return {
                "mode": args.mode,
                "startup_ms": round(startup_ms, 2),
                "root_ms": round(root_ms, 2),
                "status_ms": round(status_ms, 2),
                "sessions_ms": round(sessions_ms, 2),
                "workflow_ms": round(workflow_ms, 2),
                "status_p95_ms": round(_p95(repeated), 2),
                "ready_rss_mib": round(ready_rss / MIB, 2),
                "root_rss_mib": round(root_rss / MIB, 2),
                "status_rss_mib": round(status_rss / MIB, 2),
                "sessions_rss_mib": round(sessions_rss / MIB, 2),
                "daily_use_rss_mib": round(daily_use_rss / MIB, 2),
                "peak_rss_mib": round(peak_rss / MIB, 2),
                "request_growth_mib": round((daily_use_rss - ready_rss) / MIB, 2),
                "root_bytes": len(root_body),
            }
        finally:
            stop_sampling.set()
            sampler.join(timeout=1)
            _terminate(process)


def _summary(results: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        key for key, value in results[0].items() if isinstance(value, int | float)
    ]
    return {
        key: round(statistics.median(float(result[key]) for result in results), 2)
        for key in numeric_keys
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "light"), required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--request-timeout", type=float, default=15)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.requests < 1:
        parser.error("--runs and --requests must be positive")

    results = [_run_once(args) for _ in range(args.runs)]
    payload = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "runs": results,
        "median": _summary(results),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
