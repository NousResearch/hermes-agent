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


def key_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Flatten the report into a dict of scalar metrics for A/B diffing."""
    metrics: dict[str, float] = {}
    frames = data.get("frame") or []
    react = data.get("react") or []

    if frames:
        durs = [f["durationMs"] for f in frames]
        metrics["frames"] = len(frames)
        metrics["dur_p50"] = pct(durs, 0.50)
        metrics["dur_p95"] = pct(durs, 0.95)
        metrics["dur_p99"] = pct(durs, 0.99)
        metrics["dur_max"] = max(durs)

        ts = sorted(f["ts"] for f in frames)
        if len(ts) >= 2:
            elapsed = (ts[-1] - ts[0]) / 1000.0
            metrics["fps_throughput"] = len(frames) / elapsed if elapsed > 0 else 0.0
            # Interframe gaps distribution — complementary view to throughput:
            gaps = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
            if gaps:
                metrics["gap_p50_ms"] = pct(gaps, 0.50)
                metrics["gap_p99_ms"] = pct(gaps, 0.99)
                metrics["gaps_under_16ms"] = sum(1 for g in gaps if g < 16)
                metrics["gaps_over_200ms"] = sum(1 for g in gaps if g >= 200)

        for phase in ("renderer", "yoga", "diff", "write"):
            vals = [f["phases"][phase] for f in frames if f.get("phases")]
            if vals:
                metrics[f"{phase}_p99"] = pct(vals, 0.99)
                metrics[f"{phase}_max"] = max(vals)

        patches = [f["phases"]["patches"] for f in frames if f.get("phases")]
        if patches:
            metrics["patches_total"] = sum(patches)
            metrics["patches_p99"] = pct(patches, 0.99)

        optimized = [
            f["phases"].get("optimizedPatches", 0) for f in frames if f.get("phases")
        ]
        if any(optimized):
            metrics["optimized_total"] = sum(optimized)

        bytes_list = [
            f["phases"].get("writeBytes", 0) for f in frames if f.get("phases")
        ]
        if any(bytes_list):
            metrics["writeBytes_total"] = sum(bytes_list)

        drains = [
            f["phases"].get("prevFrameDrainMs", 0)
            for f in frames if f.get("phases")
        ]
        drain_nonzero = [d for d in drains if d > 0]
        if drain_nonzero:
            metrics["drain_p99"] = pct(drain_nonzero, 0.99)
            metrics["drain_max"] = max(drain_nonzero)

        bp = sum(1 for f in frames if f.get("phases", {}).get("backpressure"))
        metrics["backpressure_frames"] = bp

    if react:
        for pid in set(e["id"] for e in react):
            ms = [e["actualMs"] for e in react if e["id"] == pid]
            metrics[f"react_{pid}_p99"] = pct(ms, 0.99)
            metrics[f"react_{pid}_max"] = max(ms)

    return metrics


def format_diff(before: dict[str, float], after: dict[str, float]) -> str:
    """Render a side-by-side A/B comparison table."""
    keys = sorted(set(before) | set(after))
    lines = [f"{'metric':<28} {'before':>12} {'after':>12} {'delta':>12}  {'%':>6}"]
    lines.append("─" * 76)
    for k in keys:
        b = before.get(k, 0.0)
        a = after.get(k, 0.0)
        d = a - b
        pct_change = ((a / b) - 1) * 100 if b not in (0, 0.0) else float("inf") if a else 0

        # Flag improvements vs regressions. For _p99 / _max / _total / gaps_over /
        # patches / writeBytes / backpressure, LOWER is better.  For fps / gaps_under,
        # HIGHER is better.
        lower_is_better = any(
            token in k
            for token in (
                "p50",
                "p95",
                "p99",
                "_max",
                "_total",
                "gaps_over",
                "backpressure",
                "drain",
            )
        )
        higher_is_better = "fps_" in k or "gaps_under" in k
        mark = ""
        if d and not (lower_is_better or higher_is_better):
            mark = ""
        elif d < 0 and lower_is_better:
            mark = "↓"
        elif d > 0 and higher_is_better:
            mark = "↑"
        elif d > 0 and lower_is_better:
            mark = "↑"  # regression
        elif d < 0 and higher_is_better:
            mark = "↓"  # regression

        pct_str = "—" if pct_change == float("inf") else f"{pct_change:+6.1f}%"
        lines.append(
            f"{k:<28} {b:>12.2f} {a:>12.2f} {d:>+12.2f}  {pct_str} {mark}"
        )

    return "\n".join(lines)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
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

    # Pass through extra flags the TUI wrapper recognizes (e.g. --no-fullscreen).
    # Stored on args as `extra_flags` list.
    node = os.environ.get("HERMES_PERF_NODE", "node")
    node_args = [node, str(entry), *getattr(args, "extra_flags", [])]

    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(node, node_args, env)

    try:
        import fcntl, struct, termios
        winsize = struct.pack("HHHH", args.rows, args.cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

        print(f"• pid: {pid}  fd: {fd}")
        print(f"• warmup {args.warmup}s (drain startup output)…")
        drain(fd, args.warmup)

        print(f"• holding {args.hold}…")
        sent = hold_key(fd, KEYS[args.hold], args.seconds, args.rate)
        print(f"  sent {sent} keystrokes")

        drain(fd, 0.5)
    finally:
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

    time.sleep(0.2)
    return summarize(log, since_ms)


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
    p.add_argument("--save", metavar="LABEL",
                   help="save the final metrics as /tmp/perf-<LABEL>.json for later --compare")
    p.add_argument("--compare", metavar="LABEL",
                   help="diff against /tmp/perf-<LABEL>.json after running")
    p.add_argument("--loop", action="store_true",
                   help="watch for source changes, rebuild, rerun, and diff vs previous run")
    p.add_argument("--extra-flag", dest="extra_flags", action="append", default=[],
                   help="pass through to node dist/entry.js (repeatable)")
    args = p.parse_args()

    if args.loop:
        return loop_mode(args)

    # Single-shot path.
    data = run_once(args)
    print()
    print(format_report(data))

    metrics = key_metrics(data)

    if args.save:
        path = Path(f"/tmp/perf-{args.save}.json")
        path.write_text(json.dumps(metrics, indent=2))
        print(f"\n• saved: {path}")

    if args.compare:
        path = Path(f"/tmp/perf-{args.compare}.json")
        if not path.exists():
            print(f"\n⚠ no baseline at {path} — run with --save {args.compare} first")
        else:
            before = json.loads(path.read_text())
            print(f"\n═══ A/B diff vs /tmp/perf-{args.compare}.json ═══")
            print(format_diff(before, metrics))

    if not data["react"] and not data["frame"]:
        return 2
    return 0


def loop_mode(args: argparse.Namespace) -> int:
    """Watch source files, rebuild, rerun, print A/B diff against previous run.

    Keeps a rolling 'previous run' baseline in memory so each iteration
    reports delta vs the last one — visibility into whether the last
    edit moved the needle.  Press Ctrl+C to stop.
    """
    import subprocess

    tui_dir = Path(args.tui_dir).resolve()
    src_root = tui_dir / "src"
    pkg_root = tui_dir / "packages" / "hermes-ink" / "src"

    def collect_mtimes() -> dict[str, float]:
        mtimes: dict[str, float] = {}
        for root in (src_root, pkg_root):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.suffix in {".ts", ".tsx"} and "__tests__" not in str(path):
                    try:
                        mtimes[str(path)] = path.stat().st_mtime
                    except OSError:
                        pass
        return mtimes

    previous_metrics: dict[str, float] | None = None
    previous_mtimes = collect_mtimes()
    iteration = 0

    print(f"• loop mode — watching {src_root} + {pkg_root} for *.ts(x) changes")
    print("• edit any TS file, the harness rebuilds + reruns automatically")
    print("• Ctrl+C to stop\n")

    try:
        while True:
            iteration += 1
            print(f"\n{'═' * 76}")
            print(f"Iteration {iteration}  @ {time.strftime('%H:%M:%S')}")
            print("═" * 76)

            if iteration > 1:
                print("• rebuilding…")
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=tui_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print("✗ build failed:")
                    print(result.stdout[-2000:])
                    print(result.stderr[-2000:])
                    print("\n• waiting for source changes to retry…")
                    previous_mtimes = wait_for_change(previous_mtimes, collect_mtimes)
                    continue
                print("✓ build ok")

            data = run_once(args)
            metrics = key_metrics(data)

            print()
            print(format_report(data))

            if previous_metrics is not None:
                print(f"\n═══ A/B diff vs iteration {iteration - 1} ═══")
                print(format_diff(previous_metrics, metrics))

            previous_metrics = metrics

            print("\n• waiting for source changes…")
            previous_mtimes = wait_for_change(previous_mtimes, collect_mtimes)
    except KeyboardInterrupt:
        print("\n• loop stopped")
        return 0


def wait_for_change(prev: dict[str, float], collect) -> dict[str, float]:
    """Poll every 1s until a watched file's mtime changes. Debounced 500ms."""
    while True:
        time.sleep(1)
        current = collect()

        changed = [
            path for path, mtime in current.items() if prev.get(path) != mtime
        ]

        if changed:
            print(f"  ↻ {len(changed)} file(s) changed:")
            for path in changed[:5]:
                print(f"    {path}")
            # Debounce — editor save bursts can take ~500ms to settle
            time.sleep(0.5)
            return collect()


if __name__ == "__main__":
    sys.exit(main())
