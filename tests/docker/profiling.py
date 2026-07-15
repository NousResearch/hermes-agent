"""Profiling plugin for docker integration tests.

Activated by ``HERMES_DOCKER_TEST_PROFILE=1``. Instruments every
``subprocess.run`` call whose argv starts with ``docker`` to measure
wall-clock time, and also tracks ``time.sleep`` calls so we can see the
full wall-clock picture — including the invisible sleep time in polling
loops that doesn't show up as docker call duration.

Outputs:
  - JSON report at ``$HERMES_DOCKER_PROFILE_OUT`` (default:
    ``docker-test-profile-{pid}.json`` in the repo root — per-PID
    because run_tests_parallel.py spawns each test file in its own
    subprocess).
  - Console summary on stderr at session end.

The plugin is a no-op when the env var is not set — zero overhead on
normal test runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

_ACTIVE = bool(os.environ.get("HERMES_DOCKER_TEST_PROFILE"))

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TimedEvent:
    """Base for any timed event in a test's lifecycle."""

    duration_s: float
    timestamp: float  # monotonic


@dataclass
class DockerCall(TimedEvent):
    """One instrumented docker subprocess call."""

    argv: list[str]
    returncode: int


@dataclass
class SleepGap(TimedEvent):
    """A time.sleep() that occurred between docker calls.

    This captures the invisible "gap" time — the polling sleeps, the
    Python logic between assertions, etc. Each SleepGap is attributed to
    the test that was running when it occurred.
    """

    caller: str  # short description of who called sleep


@dataclass
class TestProfile:
    """Per-test accumulation of docker calls and sleep gaps."""

    name: str
    calls: list[DockerCall] = field(default_factory=list)
    sleeps: list[SleepGap] = field(default_factory=list)

    @property
    def total_docker_s(self) -> float:
        return sum(c.duration_s for c in self.calls)

    @property
    def total_sleep_s(self) -> float:
        return sum(s.duration_s for s in self.sleeps)

    @property
    def total_wall_s(self) -> float:
        """Docker time + sleep time — the test's visible wall-clock cost."""
        return self.total_docker_s + self.total_sleep_s

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def by_subcommand(self) -> dict[str, list[DockerCall]]:
        """Group calls by the first docker subcommand (run, exec, restart, ...)."""
        groups: dict[str, list[DockerCall]] = defaultdict(list)
        for c in self.calls:
            sub = c.argv[1] if len(c.argv) > 1 else "?"
            groups[sub].append(c)
        return groups


# ---------------------------------------------------------------------------
# Session-level collector
# ---------------------------------------------------------------------------


class ProfileCollector:
    """Singleton accumulator shared across the pytest session."""

    def __init__(self) -> None:
        self.tests: dict[str, TestProfile] = {}
        self.current: Optional[TestProfile] = None
        self._original_run: Any = None
        self._original_sleep: Any = None
        self._patched = False
        self._last_docker_end: float = 0.0

    def start_test(self, name: str) -> None:
        self.current = TestProfile(name=name)
        self.tests[name] = self.current
        self._last_docker_end = 0.0

    def end_test(self) -> None:
        self.current = None

    def record(self, call: DockerCall) -> None:
        if self.current is not None:
            self.current.calls.append(call)
        self._last_docker_end = call.timestamp + call.duration_s

    def record_sleep(self, gap: SleepGap) -> None:
        if self.current is not None:
            self.current.sleeps.append(gap)

    def install_patch(self) -> None:
        """Monkey-patch subprocess.run and time.sleep to capture timings."""
        if self._patched:
            return
        import subprocess

        self._original_run = subprocess.run
        self._original_sleep = time.sleep
        collector = self

        def timed_run(*args: Any, **kwargs: Any) -> Any:
            argv = list(args[0]) if args and args[0] else kwargs.get("args", [])
            is_docker = bool(argv) and argv[0] == "docker"
            if not is_docker:
                return collector._original_run(*args, **kwargs)
            t0 = time.monotonic()
            result = collector._original_run(*args, **kwargs)
            elapsed = time.monotonic() - t0
            rc = getattr(result, "returncode", -1)
            call = DockerCall(
                duration_s=round(elapsed, 4),
                timestamp=t0,
                argv=[str(a) for a in argv],
                returncode=rc,
            )
            collector.record(call)
            return result

        def timed_sleep(secs: float) -> None:
            # Only track sleeps that happen between docker calls within a test.
            # Short sleeps (< 0.05s) are probably just Python scheduling noise.
            if collector.current is None or secs < 0.05:
                return collector._original_sleep(secs)
            t0 = time.monotonic()
            collector._original_sleep(secs)
            elapsed = time.monotonic() - t0
            # Try to identify the caller for context
            import traceback
            stack = traceback.extract_stack(limit=4)
            caller = ""
            for frame in reversed(stack):
                fname = frame.filename
                if "conftest" in fname or "test_" in fname:
                    caller = f"{Path(fname).name}:{frame.lineno}"
                    break
            gap = SleepGap(
                duration_s=round(elapsed, 4),
                timestamp=t0,
                caller=caller,
            )
            collector.record_sleep(gap)

        subprocess.run = timed_run
        time.sleep = timed_sleep
        self._patched = True

    def uninstall_patch(self) -> None:
        if not self._patched:
            return
        import subprocess

        if self._original_run is not None:
            subprocess.run = self._original_run
        if self._original_sleep is not None:
            time.sleep = self._original_sleep
        self._patched = False

    def build_report(self) -> dict[str, Any]:
        """Build the JSON-serializable report dict."""
        report: dict[str, Any] = {
            "tests": [],
            "summary": {},
        }
        all_docker_time = 0.0
        all_sleep_time = 0.0
        all_call_count = 0
        all_sleep_count = 0
        subcmd_totals: dict[str, float] = defaultdict(float)
        subcmd_counts: dict[str, int] = defaultdict(int)

        for _name, tp in sorted(
            self.tests.items(), key=lambda x: x[1].total_wall_s, reverse=True
        ):
            by_sub = tp.by_subcommand()
            test_entry: dict[str, Any] = {
                "name": tp.name,
                "total_docker_s": round(tp.total_docker_s, 3),
                "total_sleep_s": round(tp.total_sleep_s, 3),
                "total_wall_s": round(tp.total_wall_s, 3),
                "call_count": tp.call_count,
                "sleep_count": len(tp.sleeps),
                "by_subcommand": {
                    sub: {
                        "count": len(calls),
                        "total_s": round(sum(c.duration_s for c in calls), 3),
                        "avg_s": round(
                            sum(c.duration_s for c in calls) / len(calls), 3
                        )
                        if calls
                        else 0,
                        "max_s": round(max(c.duration_s for c in calls), 3)
                        if calls
                        else 0,
                    }
                    for sub, calls in sorted(
                        by_sub.items(),
                        key=lambda x: sum(c.duration_s for c in x[1]),
                        reverse=True,
                    )
                },
                "calls": [
                    {
                        "type": "docker",
                        "argv": " ".join(c.argv[:8]),
                        "duration_s": c.duration_s,
                        "returncode": c.returncode,
                    }
                    for c in sorted(tp.calls, key=lambda x: x.duration_s, reverse=True)
                ],
                "sleeps": [
                    {
                        "type": "sleep",
                        "duration_s": s.duration_s,
                        "caller": s.caller,
                    }
                    for s in sorted(tp.sleeps, key=lambda x: x.duration_s, reverse=True)
                ],
            }
            report["tests"].append(test_entry)
            all_docker_time += tp.total_docker_s
            all_sleep_time += tp.total_sleep_s
            all_call_count += tp.call_count
            all_sleep_count += len(tp.sleeps)
            for sub, calls in by_sub.items():
                subcmd_totals[sub] += sum(c.duration_s for c in calls)
                subcmd_counts[sub] += len(calls)

        report["summary"] = {
            "total_tests": len(self.tests),
            "total_docker_s": round(all_docker_time, 3),
            "total_sleep_s": round(all_sleep_time, 3),
            "total_wall_s": round(all_docker_time + all_sleep_time, 3),
            "total_calls": all_call_count,
            "total_sleeps": all_sleep_count,
            "by_subcommand": {
                sub: {
                    "count": subcmd_counts[sub],
                    "total_s": round(subcmd_totals[sub], 3),
                    "avg_s": round(subcmd_totals[sub] / subcmd_counts[sub], 3)
                    if subcmd_counts[sub]
                    else 0,
                }
                for sub in sorted(
                    subcmd_totals, key=lambda s: subcmd_totals[s], reverse=True
                )
            },
        }
        return report

    def write_report(self, out_path: Path) -> None:
        """Write the JSON report."""
        out_path.write_text(
            json.dumps(self.build_report(), indent=2) + "\n"
        )

    def print_summary(self) -> None:
        """Print a human-readable summary to stderr."""
        if not self.tests:
            print("\n[docker-profile] No tests profiled.", file=sys.stderr)
            return

        print("\n" + "=" * 72, file=sys.stderr)
        print("[docker-profile] Docker + sleep timing breakdown", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        subcmd_totals: dict[str, float] = defaultdict(float)
        subcmd_counts: dict[str, int] = defaultdict(int)
        for tp in self.tests.values():
            for sub, calls in tp.by_subcommand().items():
                subcmd_totals[sub] += sum(c.duration_s for c in calls)
                subcmd_counts[sub] += len(calls)
        total_sleep = sum(tp.total_sleep_s for tp in self.tests.values())
        total_docker = sum(tp.total_docker_s for tp in self.tests.values())
        total_wall = total_docker + total_sleep

        print(
            f"\n  Total wall time: {total_wall:.1f}s"
            f" = {total_docker:.1f}s docker + {total_sleep:.1f}s sleep\n",
            file=sys.stderr,
        )
        print(
            f"  {'Category':<15} {'Count':>8} {'Total':>10} {'Avg':>8} {'%':>6}",
            file=sys.stderr,
        )
        print(
            f"  {'─' * 15} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 6}",
            file=sys.stderr,
        )
        for sub in sorted(
            subcmd_totals, key=lambda s: subcmd_totals[s], reverse=True
        ):
            t = subcmd_totals[sub]
            n = subcmd_counts[sub]
            pct = (t / total_wall * 100) if total_wall else 0
            print(
                f"  docker {sub:<8} {n:>8} {t:>9.1f}s {t / n:>7.2f}s {pct:>5.1f}%",
                file=sys.stderr,
            )
        # Sleep row
        sleep_count = sum(len(tp.sleeps) for tp in self.tests.values())
        pct = (total_sleep / total_wall * 100) if total_wall else 0
        print(
            f"  {'time.sleep':<15} {sleep_count:>8} {total_sleep:>9.1f}s"
            f" {total_sleep / sleep_count if sleep_count else 0:>7.2f}s {pct:>5.1f}%",
            file=sys.stderr,
        )

        # Top 10 slowest tests by WALL time
        print(
            f"\n  Top 10 slowest tests (by wall time = docker + sleep):\n",
            file=sys.stderr,
        )
        sorted_tests = sorted(
            self.tests.values(), key=lambda t: t.total_wall_s, reverse=True
        )
        for i, tp in enumerate(sorted_tests[:10], 1):
            print(
                f"  {i:>2}. {tp.total_wall_s:>6.1f}s "
                f"({tp.total_docker_s:.1f}s docker + {tp.total_sleep_s:.1f}s sleep) "
                f"{tp.call_count:>3} calls  ...{tp.name[-50:]}",
                file=sys.stderr,
            )

        print("\n" + "=" * 72, file=sys.stderr)


# ---------------------------------------------------------------------------
# Plugin hooks
# ---------------------------------------------------------------------------

_collector: Optional[ProfileCollector] = None


def _get_collector() -> ProfileCollector:
    global _collector
    if _collector is None:
        _collector = ProfileCollector()
    return _collector


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap each test call to track per-test docker operations."""
    if not _ACTIVE:
        yield
        return
    collector = _get_collector()
    collector.start_test(item.nodeid)
    yield
    collector.end_test()


def pytest_sessionstart(session):
    """Install the subprocess.run + time.sleep patches at session start."""
    if not _ACTIVE:
        return
    collector = _get_collector()
    collector.install_patch()


def pytest_sessionfinish(session, exitstatus):
    """Write the report and print the summary at session end."""
    if not _ACTIVE:
        return
    collector = _get_collector()
    collector.uninstall_patch()

    default_out = str(Path.cwd() / f"docker-test-profile-{os.getpid()}.json")
    out = os.environ.get("HERMES_DOCKER_PROFILE_OUT", default_out)
    out_path = Path(out)
    collector.write_report(out_path)
    collector.print_summary()
    print(f"\n[docker-profile] Report written to {out_path}", file=sys.stderr)
