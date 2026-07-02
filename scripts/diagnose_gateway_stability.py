#!/usr/bin/env python
"""No-kill Hermes Desktop/gateway stability diagnostic.

This script intentionally only reads logs/process state and probes local health
endpoints. It does not stop, restart, or mutate any Hermes process.

The diagnostic follows the common SRE/alerting model used by tools such as
Grafana/Prometheus/PagerDuty: keep historical/resolved events in the timeline,
but make the top-level PASS/WARN/BLOCKED recommendation from the current active
window. By default the current window starts at the latest local recovery marker
(Desktop cron scheduler start, dashboard ready, or WS accept); pass ``--since``
to choose an explicit ISO-like timestamp.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

PATTERNS = {
    "event_loop_stalled": re.compile(r"event loop stalled|loop stalled", re.I),
    "ws_write_slow": re.compile(r"ws write slow", re.I),
    "ready_send_failed": re.compile(r"ready frame send failed|ready_send_failed", re.I),
    "response_send_failed": re.compile(r"ws response send failed|send_failed_after_response", re.I),
    "runtime_check_failed_send": re.compile(r"setup\.runtime_check", re.I),
    "unicode_decode_error": re.compile(r"UnicodeDecodeError", re.I),
}
RECOVERY_MARKER_RE = re.compile(
    r"Desktop cron scheduler started|HERMES_DASHBOARD_READY|ws accepted", re.I
)
PORT_RE = re.compile(r"HERMES_DASHBOARD_READY\s+port=(\d+)")
TS_RE = re.compile(r"(?P<ts>20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:,\d+)?")
LOG_FILES = ["gui.log", "gateway.log", "gateway-stdio.log", "tui_gateway_crash.log", "errors.log"]


def _default_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _parse_ts(text: str) -> datetime | None:
    m = TS_RE.search(text)
    if not m:
        return None
    raw = m.group("ts").replace("T", " ")
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_since(value: str | None) -> datetime | None:
    if not value or value == "auto":
        return None
    value = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise SystemExit(f"Invalid --since timestamp: {value!r}")


def _tail_text(path: Path, max_bytes: int = 512_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return f"[read_error:{exc}]"


def _iter_log_lines(log_dir: Path):
    for name in LOG_FILES:
        text = _tail_text(log_dir / name)
        for line in text.splitlines():
            yield name, line, _parse_ts(line)


def _latest_recovery_marker(log_dir: Path) -> datetime | None:
    latest: datetime | None = None
    for _name, line, ts in _iter_log_lines(log_dir):
        if ts and RECOVERY_MARKER_RE.search(line):
            latest = ts if latest is None or ts > latest else latest
    return latest


def _count_patterns(log_dir: Path, *, since: datetime | None = None) -> dict[str, int]:
    counts = {k: 0 for k in PATTERNS}
    for _name, line, ts in _iter_log_lines(log_dir):
        if since is not None and (ts is None or ts < since):
            continue
        for key, rx in PATTERNS.items():
            counts[key] += len(rx.findall(line))
    return counts


def _event_timeline(log_dir: Path, *, since: datetime | None) -> dict:
    states: dict[str, dict] = {
        key: {
            "status": "absent",
            "historical_count": 0,
            "current_count": 0,
            "first_seen": None,
            "last_seen": None,
        }
        for key in PATTERNS
    }
    recovery_markers: list[str] = []
    for name, line, ts in _iter_log_lines(log_dir):
        if ts and RECOVERY_MARKER_RE.search(line):
            recovery_markers.append(ts.isoformat(sep=" "))
        for key, rx in PATTERNS.items():
            hits = len(rx.findall(line))
            if not hits:
                continue
            state = states[key]
            state["historical_count"] += hits
            if ts is not None:
                iso = ts.isoformat(sep=" ")
                if state["first_seen"] is None or iso < state["first_seen"]:
                    state["first_seen"] = iso
                if state["last_seen"] is None or iso > state["last_seen"]:
                    state["last_seen"] = iso
                if since is not None and ts >= since:
                    state["current_count"] += hits
            elif since is None:
                state["current_count"] += hits
    for state in states.values():
        if state["current_count"]:
            state["status"] = "active"
        elif state["historical_count"]:
            state["status"] = "resolved"
        else:
            state["status"] = "absent"
    return {
        "current_window_start": since.isoformat(sep=" ") if since else None,
        "recovery_markers_seen": recovery_markers[-8:],
        "states": states,
        "active": [k for k, v in states.items() if v["status"] == "active"],
        "resolved": [k for k, v in states.items() if v["status"] == "resolved"],
    }


def _latest_dashboard_port(log_dir: Path) -> int | None:
    text = _tail_text(log_dir / "gui.log") + "\n" + _tail_text(log_dir / "gateway.log")
    matches = PORT_RE.findall(text)
    return int(matches[-1]) if matches else None


def _probe_health(port: int | None, timeout: float = 2.0) -> dict:
    if not port:
        return {"status": "unknown", "reason": "no HERMES_DASHBOARD_READY port found"}
    url = f"http://127.0.0.1:{port}/health"
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(200).decode("utf-8", errors="replace")
        return {
            "status": "ok",
            "url": url,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "body_preview": body,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "fail",
            "url": url,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": str(exc),
        }


def _process_snapshot() -> dict:
    terms = {
        "slash_worker": "tui_gateway.slash_worker",
        "dashboard_backend": "hermes_cli.main dashboard",
        "gateway": "gateway",
        "hermes_python": "python",
    }
    counts = {k: 0 for k in terms}
    samples: list[str] = []
    try:
        if os.name == "nt":
            try:
                raw = subprocess.check_output(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-Command",
                        "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=8,
                )
                commands = [line for line in raw.splitlines() if line.strip()]
            except Exception:
                raw = subprocess.check_output(
                    ["tasklist.exe", "/FO", "CSV"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                )
                commands = [line for line in raw.splitlines() if line.strip()]
        else:
            raw = subprocess.check_output(
                ["ps", "-eo", "pid=,args="],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            commands = raw.splitlines()
        for cmd in commands:
            low = cmd.lower()
            for key, needle in terms.items():
                if needle.lower() in low:
                    counts[key] += 1
            if any(n.lower() in low for n in ("hermes", "tui_gateway", "slash_worker")) and len(samples) < 8:
                samples.append(cmd[:240])
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
        return {"status": "unknown", "error": str(exc), "counts": counts, "samples": samples}
    return {"status": "ok", "counts": counts, "samples": samples}


def _recommend(current_counts: dict[str, int], health: dict, procs: dict) -> str:
    if health.get("status") == "fail" and (
        current_counts.get("ready_send_failed", 0)
        or current_counts.get("response_send_failed", 0)
    ):
        return "BLOCKED: backend health probe failed and active ready/send failures are present; collect logs before any restart."
    if current_counts.get("event_loop_stalled", 0) or current_counts.get("ws_write_slow", 0):
        return "WARN: current window has event-loop/WS stalls; avoid restart-first, reduce new worker intake and protect control-plane RPCs."
    if procs.get("counts", {}).get("slash_worker", 0) >= 4:
        return "WARN: no active WS stall in current window, but many slash workers are active; throttle interactive worker fanout before starting more heavy tasks."
    return "PASS: no active no-kill instability signal found in the current window. Historical resolved events are retained in timeline."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=_default_home(), help="Hermes home/profile directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--since",
        default="auto",
        help="Current-window start timestamp ('YYYY-MM-DD HH:MM:SS') or 'auto' for latest recovery marker.",
    )
    args = parser.parse_args(argv)

    home = args.home.expanduser().resolve()
    log_dir = home / "logs"
    explicit_since = _parse_since(args.since)
    auto_since = _latest_recovery_marker(log_dir) if args.since == "auto" else None
    current_since = explicit_since or auto_since
    historical_counts = _count_patterns(log_dir)
    current_counts = _count_patterns(log_dir, since=current_since)
    timeline = _event_timeline(log_dir, since=current_since)
    port = _latest_dashboard_port(log_dir)
    health = _probe_health(port)
    procs = _process_snapshot()
    report = {
        "home": str(home),
        "log_dir": str(log_dir),
        "dashboard_port": port,
        "health": health,
        "historical_log_counts": historical_counts,
        "current_window_counts": current_counts,
        "timeline": timeline,
        # Backward-compatible alias, but recommendations now use current_window_counts.
        "recent_log_counts": historical_counts,
        "process_snapshot": procs,
        "recommendation": _recommend(current_counts, health, procs),
        "safety": "read-only/no-kill/no-secret-dump",
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Hermes gateway/backend stability diagnostic")
        print(f"home: {report['home']}")
        print(f"dashboard_port: {port}")
        print(f"health: {health}")
        print(f"current_window_start: {timeline['current_window_start']}")
        print(f"current_window_counts: {current_counts}")
        print(f"resolved_timeline_events: {timeline['resolved']}")
        print(f"active_timeline_events: {timeline['active']}")
        print(f"process_counts: {procs.get('counts')}")
        print(f"recommendation: {report['recommendation']}")
        print(f"safety: {report['safety']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
