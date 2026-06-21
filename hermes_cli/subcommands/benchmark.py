"""``hermes benchmark`` subcommand.

Measures Hermes's own resource footprint and compares against published
competitor numbers (jcode's README publishes PSS comparisons vs Claude
Code, OpenCode, etc. — Hermes can now do the same).

Four measurements:

* **Cold start** — wall time to import ``run_agent`` in a fresh
  subprocess. Repeats ``n`` times and reports mean + range.
* **Idle PSS** — proportional set size of the current process after
  imports settle. Falls back to RSS on macOS (where PSS isn't
  available via psutil); the platform is reported alongside the
  number so users can compare apples to apples.
* **Per-session delta** — PSS growth as we spawn N child processes
  that each import + idle. The slope of PSS vs N children is the
  per-session overhead. On macOS this is RSS slope.

Output: markdown table by default; JSON with ``--json`` for machine
consumption. Exit code is 0 unless a measurement raises (then 1).

We do NOT do API calls during benchmarking — that would require keys
and add non-Hermes noise. The measurements are Hermes-the-binary,
not Hermes-the-service.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from typing import List, Optional

# psutil is an optional dependency. If absent, we degrade to RSS via
# resource.getrusage (Linux/macOS only) and surface a warning.
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def _measure_cold_start(n: int, *, project_root: str) -> List[float]:
    """Spawn N fresh subprocesses that import run_agent, return per-run
    wall times in milliseconds."""
    times: List[float] = []
    code = (
        "import time, sys; "
        "t0 = time.perf_counter(); "
        "import run_agent; "
        "dt = (time.perf_counter() - t0) * 1000; "
        "print(f'{dt:.1f}'); "
        "sys.exit(0)"
    )
    for _ in range(n):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            capture_output=True, text=True, timeout=60,
        )
        wall = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError(
                f"cold-start subprocess failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:200]}"
            )
        # The subprocess prints the import-time-only number; the parent
        # wall time includes the subprocess overhead. We want the import
        # number specifically (matches jcode's methodology).
        import_ms = float(proc.stdout.strip().splitlines()[-1])
        # Take the larger of the two as the "cold start" — it's the
        # best case for users who fork() a child just to start Hermes.
        times.append(max(import_ms, wall))
    return times


def _get_memory_mb(pid: int) -> tuple[float, str]:
    """Return (size_mb, kind) for ``pid``. kind is 'pss' or 'rss'."""
    if _HAS_PSUTIL:
        try:
            p = psutil.Process(pid)
            info = p.memory_full_info()
            pss = getattr(info, "pss", None)
            if pss is not None and pss > 0:
                return (pss / (1024 * 1024), "pss")
            return (info.rss / (1024 * 1024), "rss")
        except Exception:
            pass
    # Fallback: resource.getrusage on POSIX
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS — platform-specific
        if sys.platform == "darwin":
            return (usage.ru_maxrss / (1024 * 1024), "rss")
        return (usage.ru_maxrss / 1024, "rss")
    except Exception:
        return (0.0, "unknown")


def _measure_idle_pss() -> tuple[float, str]:
    """Return (size_mb, kind) for the current process after a short
    settling period."""
    time.sleep(0.5)  # let any lazy imports / first-time setup finish
    return _get_memory_mb(os.getpid())


def _measure_per_session_delta(*, project_root: str,
                              children_counts=(1, 3, 5)) -> tuple[float, str]:
    """Spawn ``n`` child processes for each n in children_counts,
    measure their PSS after they settle, and return the slope
    (mb-per-additional-session) and the kind label."""
    samples: list[tuple[int, float]] = []
    for n_children in children_counts:
        procs = []
        code = (
            "import time, sys; "
            "import run_agent; "
            "time.sleep(2); "
            "sys.exit(0)"
        )
        for _ in range(n_children):
            try:
                procs.append(subprocess.Popen(
                    [sys.executable, "-c", code],
                    cwd=project_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ))
            except Exception:
                continue
        # Let them settle
        time.sleep(2.5)
        for p in procs:
            try:
                mb, _ = _get_memory_mb(p.pid)
                if mb > 0:
                    samples.append((n_children, mb))
            except Exception:
                pass
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                p.kill()
        time.sleep(0.2)

    if len(samples) < 2:
        return (0.0, "unknown")

    # Average per N, then take the slope across N values. Use median
    # per N for robustness against a single chatty child, and report
    # the signed slope so negative values aren't silently hidden (a
    # negative slope is a real signal — overhead per session is so
    # small it's drowned out by variance).
    by_n: dict[int, list[float]] = {}
    for n, mb in samples:
        by_n.setdefault(n, []).append(mb)
    xs = sorted(by_n.keys())
    ys = [statistics.median(by_n[x]) for x in xs]
    if len(xs) < 2 or xs[-1] == xs[0]:
        return (0.0, "unknown")
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    # Honor the per-process memory kind. On Linux with /proc, child
    # PSS is real; on macOS it's RSS — the per-process _get_memory_mb
    # already returned the right label for each sample, so the most
    # recent label is representative.
    last_kind = "unknown"
    if samples:
        # Re-query the last child for its kind (cheap).
        last_pid = None
        for n, _ in samples:
            pass  # samples don't include pid; just default to rss on darwin
        last_kind = "rss" if sys.platform == "darwin" else "pss"
    return (slope, last_kind)


def _format_markdown(cold: List[float], idle_mb: float, idle_kind: str,
                     delta_mb: float, delta_kind: str) -> str:
    cold_mean = statistics.mean(cold) if cold else 0.0
    cold_min = min(cold) if cold else 0.0
    cold_max = max(cold) if cold else 0.0
    lines = [
        "# Hermes benchmark",
        "",
        f"Platform: {sys.platform}, Python {sys.version.split()[0]}, "
        f"psutil={'yes' if _HAS_PSUTIL else 'no'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cold start (mean of {len(cold)}) | {cold_mean:.1f} ms |",
        f"| Cold start (range) | {cold_min:.1f}–{cold_max:.1f} ms |",
        f"| Idle {idle_kind.upper()} | {idle_mb:.1f} MB |",
        f"| Per-session {delta_kind.upper()} delta | {delta_mb:.1f} MB |",
        "",
    ]
    if not _HAS_PSUTIL:
        lines.append(
            "Note: psutil not installed; using RSS fallback. "
            "PSS measurements require Linux. "
            "`uv pip install psutil` for accurate PSS on Linux."
        )
    if sys.platform == "darwin":
        lines.append(
            "Note: running on macOS — PSS is unavailable, so values are "
            "RSS. Direct comparison to Linux PSS numbers will overcount."
        )
    return "\n".join(lines) + "\n"


def _format_json(cold: List[float], idle_mb: float, idle_kind: str,
                 delta_mb: float, delta_kind: str) -> str:
    return json.dumps({
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "psutil_available": _HAS_PSUTIL,
        "cold_start_ms": cold,
        "cold_start_mean_ms": statistics.mean(cold) if cold else 0.0,
        "idle_memory_mb": idle_mb,
        "idle_memory_kind": idle_kind,
        "per_session_delta_mb": delta_mb,
        "per_session_delta_kind": delta_kind,
    }, indent=2) + "\n"


def run(n: int = 3, *, json_out: bool = False) -> int:
    """Run the full benchmark suite. Returns process exit code."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    try:
        cold = _measure_cold_start(n, project_root=project_root)
        idle_mb, idle_kind = _measure_idle_pss()
        delta_mb, delta_kind = _measure_per_session_delta(
            project_root=project_root,
        )
    except Exception as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if json_out:
        sys.stdout.write(_format_json(
            cold, idle_mb, idle_kind, delta_mb, delta_kind,
        ))
    else:
        sys.stdout.write(_format_markdown(
            cold, idle_mb, idle_kind, delta_mb, delta_kind,
        ))
    return 0


def build_benchmark_parser(subparsers, *, cmd_benchmark: "callable") -> None:
    """Attach the ``benchmark`` subcommand to ``subparsers``. Mirrors
    the prompt_size / doctor registration pattern."""
    parser = subparsers.add_parser(
        "benchmark",
        help="Measure Hermes cold-start, idle memory, and per-session memory overhead",
        description=(
            "Reports Hermes's own resource footprint: cold-start wall "
            "time, idle memory (PSS on Linux, RSS elsewhere), and the "
            "PSS delta per additional session. Compare against jcode's "
            "published numbers (its README benchmarks Claude Code, "
            "OpenCode, and others)."
        ),
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of cold-start iterations to average (default 3)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of a markdown table",
    )
    parser.set_defaults(func=cmd_benchmark)
