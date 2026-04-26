#!/usr/bin/env python3
"""Drive the Hermes TUI under HERMES_DEV_PERF and summarize the pipeline.

Usage:
  scripts/profile-tui.py [--session SID] [--hold KEY] [--seconds N] [--rate HZ]

Defaults: picks the session with the most messages, holds PageUp for 8s at
~30 Hz (matching xterm key-repeat), summarizes ~/.hermes/perf.log on exit.

The --tui build must exist (run `npm run build` in ui-tui first). This script
launches `node dist/entry.js` directly with HERMES_TUI_RESUME set so it
bypasses the hermes_cli wrapper — we want repeatable timing, not the CLI's
session-picker flow.

Environment overrides:
  HERMES_PERF_LOG     (default ~/.hermes/perf.log)
  HERMES_PERF_NODE    (default node from $PATH)
  HERMES_TUI_DIR      (default /home/bb/hermes-agent/ui-tui)

Exit code is 0 if the harness ran and parsed results, 2 if the TUI crashed
or produced no perf data (suggests HERMES_DEV_PERF wiring is broken).
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import signal
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TUI_DIR = Path(os.environ.get("HERMES_TUI_DIR", "/home/bb/hermes-agent/ui-tui"))
DEFAULT_LOG = Path(os.environ.get("HERMES_PERF_LOG", str(Path.home() / ".hermes" / "perf.log")))
DEFAULT_STATE_DB = Path.home() / ".hermes" / "state.db"

# Keystroke escape sequences.  Matches what xterm/VT220 send when the
# terminal has bracketed-paste disabled and the key-repeat handler fires.
KEYS = {
    "page_up": b"\x1b[5~",
    "page_down": b"\x1b[6~",
    "wheel_up": b"\x1b[M`!!",      # mouse wheel up (SGR-less) — best-effort
    "shift_up": b"\x1b[1;2A",
    "shift_down": b"\x1b[1;2B",
}


def pick_longest_session(db: Path) -> str:
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT id FROM sessions s ORDER BY "
        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) DESC LIMIT 1"
    ).fetchone()
    if not row:
        sys.exit(f"no sessions in {db}")
    return row[0]


def drain(fd: int, timeout: float) -> bytes:
    """Read whatever's available from fd within `timeout`, then return."""
    chunks = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r, _, _ = select.select([fd], [], [], max(0.0, end - time.monotonic()))
        if not r:
            break
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


def hold_key(fd: int, seq: bytes, seconds: float, rate_hz: int) -> int:
    """Write `seq` to fd at ~rate_hz for `seconds`. Returns keystrokes sent."""
    interval = 1.0 / max(1, rate_hz)
    end = time.monotonic() + seconds
    sent = 0
    while time.monotonic() < end:
        try:
            os.write(fd, seq)
            sent += 1
        except OSError:
            break
        # Drain stdout to keep the PTY buffer flowing; ignore content.
        drain(fd, 0)
        time.sleep(interval)
    return sent


def summarize(log: Path, since_ts_ms: int) -> dict[str, Any]:
    """Parse perf.log, keep only events newer than since_ts_ms, return stats."""
    react_events: list[dict[str, Any]] = []
    frame_events: list[dict[str, Any]] = []
    if not log.exists():
        return {"error": f"no log at {log}", "react": [], "frame": []}
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("ts", 0)) < since_ts_ms:
            continue
        src = row.get("src")
        if src == "react":
            react_events.append(row)
        elif src == "frame":
            frame_events.append(row)

    return {
        "react": react_events,
        "frame": frame_events,
    }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def format_report(data: dict[str, Any]) -> str:
    react = data.get("react") or []
    frames = data.get("frame") or []
    out = []

    out.append("═══ React Profiler ═══")
    if not react:
        out.append("  (no react events — HERMES_DEV_PERF wired? threshold too high?)")
    else:
        by_id: dict[str, list[float]] = {}
        for r in react:
            by_id.setdefault(r["id"], []).append(r["actualMs"])
        out.append(f"  {'pane':<14} {'count':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        for pid, ms in sorted(by_id.items(), key=lambda kv: -pct(kv[1], 0.99)):
            out.append(
                f"  {pid:<14} {len(ms):>6} {pct(ms,0.50):>8.2f} {pct(ms,0.95):>8.2f} "
                f"{pct(ms,0.99):>8.2f} {max(ms):>8.2f}"
            )

    out.append("")
    out.append("═══ Ink pipeline ═══")
    if not frames:
        out.append("  (no frame events — onFrame wiring broken?)")
    else:
        dur = [f["durationMs"] for f in frames]
        phases_present = any(f.get("phases") for f in frames)
        out.append(f"  frames captured: {len(frames)}")
        out.append(
            f"  durationMs  p50={pct(dur,0.50):.2f}  p95={pct(dur,0.95):.2f}  "
            f"p99={pct(dur,0.99):.2f}  max={max(dur):.2f}"
        )
        # Effective FPS during the run: frames / elapsed seconds.
        ts = sorted(f["ts"] for f in frames)
        if len(ts) >= 2:
            elapsed_s = (ts[-1] - ts[0]) / 1000.0
            fps = len(frames) / elapsed_s if elapsed_s > 0 else float("inf")
            out.append(f"  throughput: {len(frames)} frames / {elapsed_s:.2f}s = {fps:.1f} fps")

        if phases_present:
            fields = ["yoga", "renderer", "diff", "optimize", "write", "commit"]
            out.append("")
            out.append(f"  {'phase':<10} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   (ms)")
            for field in fields:
                vals = [f["phases"][field] for f in frames if f.get("phases")]
                if vals:
                    out.append(
                        f"  {field:<10} {pct(vals,0.50):>8.2f} {pct(vals,0.95):>8.2f} "
                        f"{pct(vals,0.99):>8.2f} {max(vals):>8.2f}"
                    )
            # Derived: sum of phases vs durationMs (reveals hidden time).
            sum_ps = [
                sum(f["phases"][k] for k in fields)
                for f in frames if f.get("phases")
            ]
            if sum_ps:
                dur_match = [f["durationMs"] for f in frames if f.get("phases")]
                deltas = [d - s for d, s in zip(dur_match, sum_ps)]
                out.append(
                    f"  {'dur-Σphases':<10} {pct(deltas,0.50):>8.2f} {pct(deltas,0.95):>8.2f} "
                    f"{pct(deltas,0.99):>8.2f} {max(deltas):>8.2f}   (unaccounted-for time)"
                )

            # Yoga counters
            visited = [f["phases"]["yogaVisited"] for f in frames if f.get("phases")]
            measured = [f["phases"]["yogaMeasured"] for f in frames if f.get("phases")]
            cache_hits = [f["phases"]["yogaCacheHits"] for f in frames if f.get("phases")]
            live = [f["phases"]["yogaLive"] for f in frames if f.get("phases")]
            out.append("")
            out.append("  Yoga counters (per frame):")
            for name, vals in (
                ("visited", visited),
                ("measured", measured),
                ("cacheHits", cache_hits),
                ("live", live),
            ):
                if vals:
                    out.append(f"    {name:<11} p50={pct(vals,0.5):.0f}  p99={pct(vals,0.99):.0f}  max={max(vals)}")

            # Patch counts — proxy for "how much changed each frame"
            patches = [f["phases"]["patches"] for f in frames if f.get("phases")]
            if patches:
                out.append(
                    f"  patches     p50={pct(patches,0.5):.0f}  p99={pct(patches,0.99):.0f}  "
                    f"max={max(patches)}  total={sum(patches)}"
                )
            optimized = [
                f["phases"].get("optimizedPatches", 0)
                for f in frames if f.get("phases")
            ]
            if any(optimized):
                out.append(
                    f"  optimized   p50={pct(optimized,0.5):.0f}  p99={pct(optimized,0.99):.0f}  "
                    f"max={max(optimized)}  total={sum(optimized)}"
                    f"  (ratio: {sum(optimized)/max(1,sum(patches)):.2f})"
                )

            # Write bytes + drain telemetry — the outer-terminal bottleneck gauge.
            bytes_written = [
                f["phases"].get("writeBytes", 0)
                for f in frames if f.get("phases")
            ]
            if any(bytes_written):
                total_b = sum(bytes_written)
                kb = total_b / 1024
                out.append(
                    f"  writeBytes  p50={pct(bytes_written,0.5):.0f}B  p99={pct(bytes_written,0.99):.0f}B  "
                    f"max={max(bytes_written)}B  total={kb:.1f}KB"
                )
            drains = [
                f["phases"].get("prevFrameDrainMs", 0)
                for f in frames if f.get("phases")
            ]
            if any(d > 0 for d in drains):
                nonzero = [d for d in drains if d > 0]
                out.append(
                    f"  drainMs     p50={pct(nonzero,0.5):.2f}  p95={pct(nonzero,0.95):.2f}  "
                    f"p99={pct(nonzero,0.99):.2f}  max={max(nonzero):.2f}   (terminal flush latency)"
                )
            backpressure = sum(1 for f in frames if f.get("phases", {}).get("backpressure"))
            if backpressure:
                out.append(
                    f"  backpressure: {backpressure}/{len(frames)} frames "
                    f"({100*backpressure/len(frames):.0f}%)   (Node stdout buffer full — terminal slow)"
                )

        # Flickers
        flicker_frames = [f for f in frames if f.get("flickers")]
        if flicker_frames:
            out.append("")
            out.append(f"  ⚠ flickers detected in {len(flicker_frames)} frames")
            reasons: dict[str, int] = {}
            for f in flicker_frames:
                for fl in f["flickers"]:
                    reasons[fl["reason"]] = reasons.get(fl["reason"], 0) + 1
            for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                out.append(f"    {reason}: {n}")

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", help="session id to resume (default: longest in db)")
    p.add_argument("--hold", default="page_up", choices=sorted(KEYS.keys()), help="key to hold")
    p.add_argument("--seconds", type=float, default=8.0, help="how long to hold the key")
    p.add_argument("--rate", type=int, default=30, help="keystrokes per second")
    p.add_argument("--warmup", type=float, default=3.0, help="seconds to wait after launch before input")
    p.add_argument("--threshold-ms", type=float, default=0.0, help="HERMES_DEV_PERF_MS (0 = capture all)")
    p.add_argument("--cols", type=int, default=120)
    p.add_argument("--rows", type=int, default=40)
    p.add_argument("--keep-log", action="store_true", help="don't wipe perf.log before run")
    p.add_argument("--tui-dir", default=str(DEFAULT_TUI_DIR))
    p.add_argument("--log", default=str(DEFAULT_LOG))
    args = p.parse_args()

    tui_dir = Path(args.tui_dir).resolve()
    entry = tui_dir / "dist" / "entry.js"
    if not entry.exists():
        sys.exit(f"{entry} missing — run `npm run build` in {tui_dir} first")

    sid = args.session or pick_longest_session(DEFAULT_STATE_DB)
    print(f"• session: {sid}")
    print(f"• hold: {args.hold} x {args.rate}Hz for {args.seconds}s after {args.warmup}s warmup")
    print(f"• terminal: {args.cols}x{args.rows}")

    log = Path(args.log)
    if not args.keep_log and log.exists():
        log.unlink()

    since_ms = int(time.time() * 1000)

    env = os.environ.copy()
    env["HERMES_DEV_PERF"] = "1"
    env["HERMES_DEV_PERF_MS"] = str(args.threshold_ms)
    env["HERMES_DEV_PERF_LOG"] = str(log)
    env["HERMES_TUI_RESUME"] = sid
    env["COLUMNS"] = str(args.cols)
    env["LINES"] = str(args.rows)
    env["TERM"] = env.get("TERM", "xterm-256color")
    # Ensure bracketed-paste doesn't intercept our PageUp writes.

    node = os.environ.get("HERMES_PERF_NODE", "node")

    # Fork under a PTY so the TUI enters alt-screen / raw-mode cleanly.
    pid, fd = pty.fork()
    if pid == 0:
        # Child: exec node.  PTY makes stdin/stdout/stderr all TTY.
        os.execvpe(node, [node, str(entry)], env)

    try:
        # Set initial PTY size via ioctl (TIOCSWINSZ).
        import fcntl, struct, termios
        winsize = struct.pack("HHHH", args.rows, args.cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

        print(f"• pid: {pid}  fd: {fd}")
        print(f"• warmup {args.warmup}s (drain startup output)…")
        drain(fd, args.warmup)

        print(f"• holding {args.hold}…")
        sent = hold_key(fd, KEYS[args.hold], args.seconds, args.rate)
        print(f"  sent {sent} keystrokes")

        # Small cooldown so trailing frames get written to the log.
        drain(fd, 0.5)
    finally:
        # Kill TUI cleanly.  SIGTERM first, SIGKILL if stubborn.
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                pid_done, _ = os.waitpid(pid, os.WNOHANG)
                if pid_done == pid:
                    break
                time.sleep(0.1)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    # Give the log a moment to flush.
    time.sleep(0.2)

    data = summarize(log, since_ms)
    print()
    print(format_report(data))

    if not data["react"] and not data["frame"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
