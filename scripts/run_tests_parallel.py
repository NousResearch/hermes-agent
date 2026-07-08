#!/usr/bin/env python3
"""Per-file parallel test runner.

The minimum-viable replacement for pytest-xdist + a subprocess-isolation
plugin. Discovers test files under ``tests/`` (excluding integration/e2e
unless explicitly requested), then runs one ``python -m pytest <file>``
subprocess per file, with bounded parallelism (default: ``os.cpu_count()``).

Why per-file rather than per-test?
    Per-test spawn overhead (~250ms × 17k tests = 70min CPU minimum)
    swamped the actual work. Per-file spawn (~250ms × ~850 files = ~3.5min)
    fits in the budget while still giving every file a fresh Python
    interpreter — the only isolation boundary that actually matters
    (cross-file module-level state leakage was the original flake source;
    intra-file state is the test author's responsibility).

Why drop xdist entirely?
    xdist's persistent workers accumulate state across files, which is
    exactly the leakage we wanted to fix. xdist also adds complexity
    (loadfile vs loadscope, --max-worker-restart, internal control plane)
    that we don't need when the unit of work is "run pytest on one file".
    A subprocess.Popen pool gated by a semaphore is ~60 lines and does
    the job.

Usage:
    python scripts/run_tests_parallel.py [pytest_args...]

    Common pytest args pass through to each per-file pytest invocation
    (e.g. ``-q``, ``-v``, ``-x``, ``--tb=long``, ``-k 'pattern'``, ``--lf``)
    with no special separator — a bare ``-q`` "just works". Anything after
    a literal ``--`` is also passed through, and stacks with bare flags.

Environment:
    HERMES_TEST_WORKERS  Override worker count (default: os.cpu_count())
    HERMES_TEST_PATHS    Override discovery roots (colon-sep, default: 'tests')

Exit code: 0 if every file's pytest exited 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Tuple


# Default test discovery roots.
_DEFAULT_ROOTS = ["tests"]

# Directories to skip during discovery — these suites require real
# external services (a model gateway, a docker daemon with a prebuilt
# image, etc.) and are run in their own dedicated CI jobs:
#
#   tests/e2e/         — .github/workflows/tests.yml :: e2e job
#   tests/integration/ — historical; legacy --ignore flags
#   tests/docker/      — .github/workflows/docker.yml ::
#                        build-amd64 job (runs against the freshly-loaded
#                        nousresearch/hermes-agent:test image, via
#                        ``HERMES_TEST_IMAGE`` so the fixture skips
#                        rebuild). The full pytest-shard runner can't
#                        host these because the session-scoped
#                        ``built_image`` fixture would do a 3-7min
#                        ``docker build``,
#                        so the build is guaranteed to die in fixture
#                        setup. The dedicated job sidesteps both costs.
_SKIP_PARTS = {"integration", "e2e", "docker"}

# Per-file wall-clock cap. Override
# via --file-timeout or HERMES_TEST_FILE_TIMEOUT.
#
# Set to 300s (5 min) deliberately generous: the per-test subprocess
# isolation plugin spawns a fresh Python process per test, so a
# large-collection file pays N × (interpreter startup + import) of
# overhead before any test logic runs — and that overhead dilates under
# load on shared CI runners, producing false "no tests ran" timeouts on
# files that finish in ~100s on a quiet box. The Docker build matrix jobs
# take 7-10 min anyway, so this headroom costs nothing on total CI wall
# time while keeping a genuinely hung file bounded.
_DEFAULT_FILE_TIMEOUT_SECONDS = 300.0

# Duration cache: maps relative file paths to last-observed subprocess
# wall-clock seconds. Used by ``--slice`` to distribute files across
# CI jobs by estimated total time, so no one job gets all the slow files.
_DURATIONS_FILE = "test_durations.json"


def _approximately_count_tests(
    files: List[Path], repo_root: Path
) -> dict[Path, int]:
    """
    Make a decent estimate at individual tests per file.
    Running ``pytest --co -q`` is WAY too slow because it actually imports everything.

    Returns a mapping ``{file_path: test_count}``. Files with zero
    collected tests are omitted from the dict (not an error — e.g. the
    file only defines fixtures / conftest helpers).

    """

    results = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            contents = f.read()
        results[path] = contents.count("def test_")

    return results


def _discover_files(roots: List[Path]) -> List[Path]:
    """Return every ``test_*.py`` under the given roots (sorted).

    Roots may be directories (recursed for ``test_*.py``) or explicit
    ``.py`` files (included as-is, even if they don't match the
    ``test_*`` prefix — caller knows what they want).

    Exclude any file whose path contains a component in ``_SKIP_PARTS``,
    UNLESS the user explicitly named it as a root (in which case the
    user's intent overrides the skip filter). This makes
    ``scripts/run_tests.sh tests/docker/`` work locally the same way
    ``pytest tests/docker/`` does — the CI-level skip exists to keep
    the sharded matrix from blowing up, not to block targeted runs.
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            # Explicit file: include it as-is, skip the _SKIP_PARTS filter
            # since the user named it directly.
            real = root.resolve()
            if real not in seen:
                seen.add(real)
                out.append(root)
            continue
        # If the explicit root itself sits inside a skipped dir (e.g.
        # the user said ``tests/docker``), the user has overridden the
        # skip for that subtree. Compute the set of skip-parts the user
        # opted into, and only filter files whose path crosses a
        # skip-part *outside* that opt-in.
        root_skip_overrides = {
            part for part in root.parts if part in _SKIP_PARTS
        }
        effective_skips = _SKIP_PARTS - root_skip_overrides
        for path in root.rglob("test_*.py"):
            if any(part in effective_skips for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


def _kill_tree(proc: "subprocess.Popen", pgid: int | None = None) -> None:
    """Kill the pytest subprocess and every descendant it spawned.

    A test run can spin up uvicorn servers, async runtimes, or other
    long-running grandchildren that survive the pytest subprocess exit
    if we don't kill the whole tree. ``subprocess.Popen.kill()`` only
    targets the immediate child; grandchildren reparent to PID 1
    (Linux) / get adopted by services.exe (Windows) and leak.

    POSIX: the caller must pass ``pgid`` — the process group id captured
    immediately after Popen (via ``os.getpgid(proc.pid)``). We can't
    look it up here in the happy path because by the time we get
    called the leader process has already been reaped and its pid is
    gone from the kernel's process table, even though descendants in
    the group are still alive. SIGKILL'ing the captured pgid takes out
    everything in that group atomically.

    Windows: ``taskkill /F /T /PID`` walks the recorded ppid chain and
    terminates the whole tree, even when the root has already exited.

    Why not psutil: psutil walks the parent-child tree, but in the
    happy path the root has already been reaped so ``psutil.Process(pid)``
    can't find it; grandchildren reparented to PID 1 are also
    unreachable by tree walk at that point. The platform-native
    primitives (process groups / taskkill) handle both cases correctly
    without an extra abstraction layer.
    """
    if proc.pid is None:
        return

    if sys.platform == "win32":
        try:
            
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )  # windows-footgun: ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    else:
        # POSIX: kill the captured pgid. Local-import signal so the
        # SIGKILL attribute is never referenced on Windows.
        if pgid is not None:
            try:
                import signal as _signal
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Belt-and-suspenders: ensure subprocess.communicate() sees the exit.
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_one_file(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
) -> Tuple[Path, int, str, dict[str, int], float]:
    """Run ``python -m pytest <file> <pytest_args>`` in a fresh subprocess.

    Returns (file, returncode, captured_combined_output, summary_counts, subprocess_wall_seconds).

    ``summary_counts`` is the result of ``_parse_pytest_summary(output)`` —

    pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html):
        0 = all tests passed
        1 = some tests failed
        2 = test execution interrupted
        3 = internal error
        4 = pytest CLI usage error
        5 = no tests collected

    We treat exit 5 as a pass: it just means every test in the file was
    skipped or filtered by a marker (e.g. ``-m 'not integration'`` skips
    files where every test is marked integration). That's intentional and
    not a failure mode.

    On per-file timeout (``file_timeout`` seconds) or any other exception
    during ``communicate()``, we kill the whole process group / process
    tree so grandchildren (uvicorn servers, async runtimes, etc.) do not
    orphan onto PID 1. This outer timeout exists only to
    bound a pathologically slow or hung file as a whole.
    """
    cmd = [sys.executable, "-m", "pytest", str(file), *pytest_args]
    
    subproc_start = time.monotonic()
    # launch the pytest process
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # skipping writing bytecode because we're running a bunch of parallel python processes on the same code
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        # POSIX: place the child at the head of its own process group so
        # _kill_tree can SIGKILL the group atomically.
        # Windows: this maps to CREATE_NEW_PROCESS_GROUP in CPython 3.12+;
        # _kill_tree handles the Windows path via taskkill /F /T.
        start_new_session=True,
    )

    # Capture the pgid NOW, before the leader can exit and be reaped. Once
    # the leader is reaped, os.getpgid(proc.pid) raises ProcessLookupError
    # even though grandchildren in that group are still alive — defeating
    # the whole cleanup. None on Windows where the pgid concept doesn't apply.
    pgid: int | None = None
    if sys.platform != "win32":
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None

    try:
        output, _ = proc.communicate(timeout=file_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, pgid=pgid)
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            output = "(file timeout exceeded; output unavailable)"
        rc = 124  # de facto convention for "killed by timeout".
        output = (
            f"({file_timeout:.0f}s exceeded; "
            f"process tree SIGKILL'd)\n{output}"
        )
    except BaseException:
        # KeyboardInterrupt / runner crash — make sure no zombie
        # grandchildren outlive us.
        _kill_tree(proc, pgid=pgid)
        raise
    else:
        # Happy path: pytest exited on its own. Kill the group anyway in
        # case it left grandchildren behind; already-dead is a no-op.
        _kill_tree(proc, pgid=pgid)

        output +=  "\n"

    summary = _parse_pytest_summary(output)
    if rc == 5:
        # No tests collected — every test in the file was filtered out.
        # Treat as a pass (a correctly marker-filtered file SHOULD be a
        # no-op in the unit lane) BUT record the no-op instead of erasing
        # it: tag the summary so the aggregate verdict site can see that
        # this file collected zero tests. Without this tag, an exit-5 file
        # is coerced to rc=0, never enters `failures`, and a whole-suite
        # zero-collect run reports green ("green because it didn't run").
        rc = 0
        summary["noop_exit5"] = 1
    subproc_wall = time.monotonic() - subproc_start
    return file, rc, output, summary, subproc_wall


def _parse_pytest_summary(output: str) -> dict[str, int]:
    """Extract per-file test pass/fail/skip counts from pytest output.

    pytest prints a summary line like ``12 passed, 3 skipped, 1 failed in 2.1s``
    as the last non-empty line before the short test summary.  We scrape that
    line for the individual counts so the progress display can show test-level
    granularity instead of just file-level pass/fail.

    Returns a dict with keys ``passed``, ``failed``, ``skipped``, ``errors``,
    ``xfailed``, ``xpassed`` (only keys found in the output are present).
    """
    import re

    result: dict[str, int] = {}
    # Walk backwards from the end — the summary line is always near the tail.
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        # Match "N passed", "N failed", "N skipped", "N errors", "N xfailed", "N xpassed"
        for m in re.finditer(r"(\d+)\s+(passed|failed|skipped|errors|xfailed|xpassed)", line):
            result[m.group(2)] = int(m.group(1))
        # Also match "N error" (singular — pytest uses this sometimes).
        for m in re.finditer(r"(\d+)\s+error\b", line):
            result.setdefault("errors", result.get("errors", 0) + int(m.group(1)))
        if result:
            # Found the counts line — done.
            break
        # Stop at the short test summary header (if any) — everything above
        # that is individual failure details, not the counts line.
        if line.startswith("FAILED") or line.startswith("SHORT TEST SUMMARY"):
            break
    return result


def _format_file(file: Path, repo_root: Path) -> str:
    """Render a test-file path for display: strip the repo-root prefix
    when possible so output reads ``tests/acp/test_auth.py`` instead of
    ``/home/runner/work/hermes-agent/hermes-agent/tests/acp/test_auth.py``.

    Falls back to the absolute path for anything outside the repo root.
    """
    try:
        return str(file.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(file)


def _print_progress(
    tests_done: int,
    approx_total_tests: int,
    file: Path,
    rc: int,
    dur: float,
    repo_root: Path,
    tests_passed: int,
    tests_failed: int,
    test_counts: dict[Path, int],
    file_summary: dict[str, int] | None = None,
    subproc_wall: float | None = None,
) -> None:
    """Single-line live progress.

    When ``file_summary`` is provided (parsed from pytest output), the
    per-file parenthetical shows individual test pass/fail counts instead
    of just the total test count.

    ``subproc_wall`` is the actual subprocess wall-clock time (excluding
    queue-wait). When available, the display shows both the subprocess
    time and the queue-inclusive elapsed time.
    """
    status = "✓" if rc == 0 else "✗"
    pct = min((tests_done / approx_total_tests * 100), 100) if approx_total_tests else 0
    # Digit width for left-side counter padding (derived from total file count).
    fw = len(str(tests_passed + tests_failed))
    # Build per-file test count string.
    if file_summary:
        parts = []
        p = file_summary.get("passed", 0)
        f = file_summary.get("failed", 0)
        s = file_summary.get("skipped", 0)
        e = file_summary.get("errors", 0)
        if p:
            parts.append(f"{p}✓")
        if f:
            parts.append(f"{f}✗")
        if s:
            parts.append(f"{s}s")
        if e:
            parts.append(f"{e}e")
        # xfailed/xpassed are rare; include if present.
        xf = file_summary.get("xfailed", 0)
        xp = file_summary.get("xpassed", 0)
        if xf:
            parts.append(f"{xf}xf")
        if xp:
            parts.append(f"{xp}xp")
        test_str = " ".join(parts) + ", " if parts else ""
    else:
        n_tests = test_counts.get(file, 0)
        test_str = f"{n_tests} tests, " if n_tests else ""
    # Show subprocess time when available; fall back to queue-inclusive dur.
    if subproc_wall is not None:
        time_str = f"{subproc_wall:.1f}s"
    else:
        time_str = f"{dur:.1f}s"
    msg = (
        f"[{pct:5.1f}% | {tests_done:>5}/~{approx_total_tests}"
        f" | ✓{tests_passed:>{fw}} | ✗{tests_failed:>{fw}}] "
        f"{status} {_format_file(file, repo_root)} ({test_str}{time_str})"
    )
    # Truncate to terminal width if available (no clobbering ANSI lines).
    try:
        cols = os.get_terminal_size().columns
        if len(msg) > cols:
            msg = msg[: cols - 1] + "…"
    except OSError:
        pass
    print(msg, flush=True)


def _print_inline_failure(
    file: Path, output: str, repo_root: Path, pytest_passthrough: List[str]
) -> None:
    """Print a compact failure summary immediately when a file fails.

    Shows the tail of the pytest output (the failure section with stack
    traces) and a ready-to-run repro command, so the developer doesn't
    have to wait for the full run to finish before seeing what broke.
    """
    rel = _format_file(file, repo_root)
    # Build a repro command the developer can copy-paste.
    passthrough_str = " ".join(pytest_passthrough) if pytest_passthrough else ""
    repro = f"python -m pytest {rel}"
    if passthrough_str:
        repro += f" {passthrough_str}"

    # Grab just the failure lines (last ~30 lines of pytest output —
    # typically the FAILED summary + short test info).
    lines = output.rstrip().splitlines()
    tail = "\n".join(lines[-30:])

    print(flush=True)
    print(f"  ╔╍ Failed: {rel} ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    for line in tail.splitlines():
        print(f"  ║ {line}", flush=True)
    print(f"  ║", flush=True)
    print(f"  ║  Repro: {repro}", flush=True)
    print(f"  ╚╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍", flush=True)
    print(flush=True)


def _load_durations(repo_root: Path) -> dict[str, float]:
    """Read the duration cache from the repo root.

    Returns a dict mapping relative file paths (e.g.
    ``tests/tools/test_code_execution.py``) to wall-clock seconds from
    the last run. Missing or corrupt file → empty dict (safe fallback).
    """
    path = repo_root / _DURATIONS_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print("[ERROR] Failed to load json durations file! {e}")
        return {}


def _save_durations(
    file_times: List[Tuple[Path, float]],
    repo_root: Path,
) -> None:
    """Write the duration cache so future ``--slice`` runs can use it.

    Merges with any existing cache so entries from files not in the
    current run (e.g. from a different slice) are preserved. Keys are
    repo-relative paths so the cache is portable across checkouts
    and CI runners.
    """
    data: dict[str, float] = _load_durations(repo_root)
    for f, t in file_times:
        key = _format_file(f, repo_root)
        data[key] = round(t, 3)
    path = repo_root / _DURATIONS_FILE
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _compute_lpt_slices(
    files: List[Path],
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[List[Path]]:
    """Distribute files across N slices using LPT (Longest Processing Time first).

    Sorts files by estimated duration descending, then greedily assigns each
    file to the slice with the smallest accumulated time so far. This
    minimizes the makespan (max slice duration) and keeps CI jobs balanced.

    Files with no cached duration get a default estimate of 2.0s (roughly
    the P50 from profiling). This means first-time runs (no cache) still
    get reasonable distribution, and new files don't all land in one slice.

    Returns a list of N file-lists, one per slice (0-indexed).
    """
    if slice_count < 2:
        return [files]

    default_dur = 2.0
    file_durs: List[Tuple[Path, float]] = []
    for f in files:
        rel = _format_file(f, repo_root)
        dur = durations.get(rel, default_dur)
        file_durs.append((f, dur))

    # Sort longest first (LPT).
    file_durs.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment: for each file, add it to the slice with the
    # smallest current total.
    bucket_files: List[List[Path]] = [[] for _ in range(slice_count)]
    bucket_totals: List[float] = [0.0] * slice_count

    for f, dur in file_durs:
        min_idx = min(range(slice_count), key=lambda i: bucket_totals[i])
        bucket_files[min_idx].append(f)
        bucket_totals[min_idx] += dur

    return bucket_files


def _slice_files(
    files: List[Path],
    slice_index: int,
    slice_count: int,
    durations: dict[str, float],
    repo_root: Path,
) -> List[Path]:
    """Return the subset of *files* belonging to slice *slice_index*.

    Uses :func:`_compute_lpt_slices` for LPT distribution.

    ``slice_index`` is 1-indexed (1..slice_count) for ergonomics —
    ``--slice 1/4`` reads more naturally than ``--slice 0/4``.
    """
    if slice_count < 2:
        return files
    if not (1 <= slice_index <= slice_count):
        print(
            f"error: --slice index must be 1..{slice_count}, got {slice_index}",
            file=sys.stderr,
        )
        sys.exit(2)

    bucket_files = _compute_lpt_slices(files, slice_count, durations, repo_root)

    target = bucket_files[slice_index - 1]
    target_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0) for f in target
    )
    total_dur = sum(
        durations.get(_format_file(f, repo_root), 2.0)
        for bucket in bucket_files
        for f in bucket
    )
    print(
        f"Slice {slice_index}/{slice_count}: {len(target)} files "
        f"(~{target_dur:.0f}s estimated of {total_dur:.0f}s total)",
        flush=True,
    )

    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("HERMES_TEST_WORKERS") or (os.cpu_count() or 4) * 2),
        help="Parallel worker count (default: $HERMES_TEST_WORKERS or cpu_count*2)",
    )
    parser.add_argument(
        "--paths",
        default=os.environ.get("HERMES_TEST_PATHS", ":".join(_DEFAULT_ROOTS)),
        help="Colon-separated discovery roots (default: 'tests')",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Don't skip integration/ e2e/ during discovery",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(
            os.environ.get("HERMES_TEST_FILE_TIMEOUT", _DEFAULT_FILE_TIMEOUT_SECONDS)
        ),
        help=(
            "Per-file wall-clock cap in seconds. On timeout, the pytest "
            "subprocess and its full process tree are SIGKILL'd. "
            f"Default: {_DEFAULT_FILE_TIMEOUT_SECONDS}s ({round(_DEFAULT_FILE_TIMEOUT_SECONDS/60)} min), env: HERMES_TEST_FILE_TIMEOUT."
        ),
    )
    parser.add_argument(
        "--slice",
        metavar="I/N",
        help=(
            "Run only slice I of N (e.g. --slice 1/4). "
            "Files are distributed across slices using cached durations "
            "so each slice takes roughly equal wall time. "
            "Without a duration cache, files are distributed by count. "
            "Env: HERMES_TEST_SLICE (format: I/N)."
        ),
    )
    parser.add_argument(
        "--generate-slices",
        metavar="N",
        type=int,
        help=(
            "Discover test files, distribute them across N slices using "
            "LPT on cached durations, and print a JSON matrix to stdout "
            "then exit (no tests run). The JSON has the shape "
            "'{\"slices\": [{\"index\": 1, \"files\": [\"tests/foo.py\", ...]}, ...]}' "
            "so the CI generate job can feed it directly into a matrix."
        ),
    )
    parser.add_argument(
        "--files",
        metavar="LIST",
        help=(
            "Explicit colon-separated list of test files to run. Bypasses "
            "discovery entirely — used by CI matrix jobs that receive their "
            "file list from the generate job."
        ),
    )
    parser.add_argument(
        "--min-tests",
        metavar="N",
        type=int,
        default=None,
        help=(
            "No-op floor: fail the whole run RED if fewer than N tests were "
            "actually executed (passed+failed). OFF by default. Opt-in per "
            "invocation for a suite whose expected count the operator knows — "
            "catches a whole directory silently vanishing from discovery."
        ),
    )
    parser.add_argument(
        "--no-strict-noop",
        dest="strict_noop",
        action="store_false",
        default=True,
        help=(
            "Disable the hard RED on an EXPLICITLY-requested test file that "
            "executed 0 tests (--files entries / positional .py files). "
            "Whole-suite-zero stays a hard RED regardless; skip-storm ⚠ "
            "surfacing stays unconditional. Default: strict-noop ON."
        ),
    )
    parser.add_argument(
        "paths_positional",
        nargs="*",
        metavar="PATH",
        help=(
            "Restrict discovery to these paths (directories or .py files). "
            "Mutually exclusive with --paths. Anything after a literal '--' "
            "separator is passed through to each per-file pytest invocation."
        ),
    )
    # Split argv into "our flags + positional paths" vs "pytest passthrough".
    #
    # Two ways to pass args through to the per-file pytest invocation:
    #   1. Explicit ``--`` separator: everything after it goes to pytest.
    #   2. Bare pytest flags anywhere before ``--``: any token starting with
    #      ``-`` that isn't one of OUR options is routed to pytest, so a bare
    #      ``-q`` / ``-v`` / ``-x`` / ``--tb=long`` / ``-k expr`` "just works"
    #      without the developer remembering the ``--``. This matches the
    #      docstring's promise and pytest muscle-memory.
    #
    # The subtlety bare-flag routing must handle: value-taking pytest flags
    # given in space-separated form (``-k expr``, ``-m mark``, ``-p plugin``,
    # ``-o name=val``). Naively, ``expr`` would look like a positional path and
    # clobber discovery. We peel the following token along with such flags so
    # it never reaches our positional ``paths``. ``=``-joined forms
    # (``-k=expr``, ``--tb=long``) are self-contained and need no lookahead.
    OUR_FLAGS = {
        "-j", "--jobs", "--paths", "--include-integration",
        "--file-timeout", "--slice", "--generate-slices", "--files",
        "--min-tests", "--no-strict-noop",
    }
    # pytest short flags that consume the NEXT token as their value.
    PYTEST_VALUE_FLAGS = {"-k", "-m", "-p", "-o", "-c", "-r", "-W"}

    def _is_our_flag(tok: str) -> bool:
        # Match exact (``-j``, ``--paths``), ``=``-joined (``--paths=x``),
        # and attached short-value (``-j4``) forms of our own options.
        if tok in OUR_FLAGS:
            return True
        head = tok.split("=", 1)[0]
        if head in OUR_FLAGS:
            return True
        # Attached short value, e.g. ``-j4`` → ``-j``.
        if len(tok) > 2 and tok[:2] in OUR_FLAGS and not tok[1] == "-":
            return True
        return False

    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        before, explicit_passthrough = argv[:sep], argv[sep + 1 :]
    else:
        before, explicit_passthrough = argv, []

    our_args: List[str] = []
    bare_passthrough: List[str] = []
    i = 0
    while i < len(before):
        tok = before[i]
        if tok.startswith("-") and not _is_our_flag(tok):
            bare_passthrough.append(tok)
            # Pull the value token for space-separated value flags.
            if tok in PYTEST_VALUE_FLAGS and i + 1 < len(before):
                bare_passthrough.append(before[i + 1])
                i += 2
                continue
        else:
            our_args.append(tok)
        i += 1

    args = parser.parse_args(our_args)
    # Bare flags run before any explicit ``--`` passthrough so ordering is
    # intuitive (``run_tests.sh tests/foo.py -q -- --tb=long`` → ``-q --tb=long``).
    pytest_passthrough = bare_passthrough + explicit_passthrough

    # Parse --slice (or HERMES_TEST_SLICE) early so we can exit on bad input
    # before doing any expensive discovery.
    slice_raw = args.slice or os.environ.get("HERMES_TEST_SLICE")
    slice_index: int | None = None
    slice_count: int = 1
    if slice_raw:
        try:
            idx_s, count_s = slice_raw.split("/", 1)
            slice_index = int(idx_s)
            slice_count = int(count_s)
        except (ValueError, AttributeError):
            print(f"error: --slice must be I/N (e.g. 1/4), got: {slice_raw!r}", file=sys.stderr)
            sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent

    # --files: explicit file list from the CI generate job — skip discovery.
    # Track which files were *explicitly requested* (by --files or by a
    # positional .py path) vs. discovered by directory recursion. An
    # explicitly-requested file that collects 0 tests is a no-op the caller
    # did not intend (they asked for those specific tests) → hard RED under
    # --strict-noop. A default-discovery marker-filtered 0-collect file is
    # legitimate and only ⚠-surfaced.
    explicit_files: set[Path] = set()
    if args.files:
        files = [repo_root / f for f in args.files.split(":") if f.strip()]
        roots = []
        explicit_files = {f.resolve() for f in files}
    else:
        # Resolve discovery roots: positional path args override --paths if any
        # were supplied, otherwise --paths (which itself defaults to 'tests').
        if args.paths_positional:
            roots = [repo_root / p for p in args.paths_positional]
            # A positional that names a .py file directly is an explicit
            # request for that file's tests.
            explicit_files = {
                (repo_root / p).resolve()
                for p in args.paths_positional
                if str(p).endswith(".py")
            }
        else:
            roots = [repo_root / p for p in args.paths.split(":") if p]

        if args.include_integration:
            # Caller takes responsibility — typically used via explicit -k filter.
            global _SKIP_PARTS  # noqa: PLW0603 — config knob
            _SKIP_PARTS = set()

        files = _discover_files(roots)

    if not files:
        print(f"No test files to run", file=sys.stderr)
        return 1

    # --generate-slices: compute LPT distribution and emit JSON, then exit.
    if args.generate_slices is not None:
        durations = _load_durations(repo_root)
        slices = _compute_lpt_slices(
            files, args.generate_slices, durations, repo_root
        )
        matrix = {
            "slice": [
                {
                    "index": i + 1,
                    "files": ":".join(_format_file(f, repo_root) for f in bucket),
                }
                for i, bucket in enumerate(slices)
            ]
        }
        # Print to stdout so the CI step can capture it with $().
        print(json.dumps(matrix))
        return 0

    # Count individual tests per file
    test_counts = _approximately_count_tests(files, repo_root)
    approx_total_tests = sum(test_counts.values())

    # Apply slicing if requested — distribute files across CI jobs by
    # estimated duration so no one job gets all the slow files.
    if slice_index is not None:
        durations = _load_durations(repo_root)
        files = _slice_files(files, slice_index, slice_count, durations, repo_root)
        # Recount after slicing.
        test_counts = {f: test_counts[f] for f in files if f in test_counts}
        approx_total_tests = sum(test_counts.values())

    if roots:
        roots_str = [str(r.relative_to(repo_root)) if r.is_relative_to(repo_root) else str(r) for r in roots]
        print(
            f"Discovered {len(files)} test files (~{approx_total_tests} tests) under "
            f"{roots_str}; running with -j {args.jobs}",
            flush=True,
        )
    else:
        print(
            f"Running {len(files)} test files (~{approx_total_tests} tests) "
            f"with -j {args.jobs}",
            flush=True,
        )

    # Capture and print on completion (out-of-order is fine — keeps the
    # terminal clean rather than interleaving N parallel pytest outputs).
    failures: List[Tuple[Path, str, Dict[str, int]]] = []
    file_times: List[Tuple[Path, float]] = []  # (file, subprocess_wall) for distribution
    # Per-file summary for EVERY completed file (not just failures), so the
    # aggregate no-op gate can see exit-5→0-coerced zero-collect files that
    # never enter `failures`. Without this the silent no-op is invisible.
    all_summaries: List[Tuple[Path, Dict[str, int]]] = []
    started = time.monotonic()
    files_done = 0
    tests_done = 0
    pass_count = 0
    fail_count = 0
    tests_passed = 0
    tests_failed = 0
    lock = threading.Lock()

    def _on_done(file: Path, started_at: float, fut: "Future[Tuple[Path, int, str, dict[str, int], float]]") -> None:
        nonlocal files_done, tests_done, pass_count, fail_count, tests_passed, tests_failed
        n_tests = test_counts.get(file, 0)
        try:
            fpath, rc, output, summary, subproc_wall = fut.result()
        except Exception as exc:  # noqa: BLE001 — must always advance counter
            with lock:
                files_done += 1
                tests_done += n_tests
                fail_count += 1
                failures.append((file, f"runner crashed: {exc!r}", {}))
                _print_progress(
                    tests_done, approx_total_tests, file, 1,
                    time.monotonic() - started_at,
                    repo_root, tests_passed, tests_failed,
                    test_counts,
                    subproc_wall=0.0,
                )
            return
        with lock:
            files_done += 1
            tests_done += n_tests
            # Accumulate test-level counts from parsed summary.
            tests_passed += summary.get("passed", 0)
            tests_failed += summary.get("failed", 0)
            file_times.append((fpath, subproc_wall))
            all_summaries.append((fpath, summary))
            if rc == 0:
                pass_count += 1
            else:
                fail_count += 1
                failures.append((fpath, output, summary))
            _print_progress(
                tests_done, approx_total_tests, fpath, rc,
                time.monotonic() - started_at,
                repo_root, tests_passed, tests_failed,
                test_counts,
                file_summary=summary,
                subproc_wall=subproc_wall,
            )
            if rc != 0:
                _print_inline_failure(fpath, output, repo_root, pytest_passthrough)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures: List[Future] = []
        for file in files:
            t0 = time.monotonic()
            fut = pool.submit(
                _run_one_file, file, pytest_passthrough, repo_root, args.file_timeout
            )
            fut.add_done_callback(lambda f, file=file, t0=t0: _on_done(file, t0, f))
            futures.append(fut)
        # Block until everything's done. ThreadPoolExecutor.__exit__ waits
        # for all submitted work, but doing it explicitly here makes the
        # control flow obvious.
        for fut in futures:
            fut.result() if fut.exception() is None else None

    elapsed = time.monotonic() - started
    print()
    pct = min(100, (tests_done / approx_total_tests * 100)) if approx_total_tests else 0
    print(f"=== Summary: {len(files)} files, {tests_passed} tests passed, {tests_failed} failed ({pct:.0f}% complete) in {elapsed:.1f}s ({args.jobs} workers) ===")

    # Save durations for future --slice runs. Each slice writes its own
    # partial test_durations.json; a CI merge step joins them later.
    # Locally, _save_durations merges with any existing cache so entries
    # from previous runs aren't lost.
    if file_times:
        _save_durations(file_times, repo_root)
        print(f"  Durations cached to {_DURATIONS_FILE} ({len(file_times)} files)")

    # Per-file time distribution (throwaway diagnostic — shows how
    # subprocess time is distributed so we can see if startup dominates).
    if file_times:
        times = sorted([t for _, t in file_times])
        total_subproc = sum(times)
        median_t = times[len(times) // 2]
        p50 = median_t
        p90 = times[int(len(times) * 0.90)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[min(int(len(times) * 0.99), len(times) - 1)]
        max_t = times[-1]
        # How many files finish in <1s? That's roughly "just startup".
        fast = sum(1 for t in times if t < 1.0)
        fast_2s = sum(1 for t in times if t < 2.0)
        print()
        print(f"=== Per-file subprocess time distribution ===")
        print(f"  Files:   {len(times)}")
        print(f"  Total subprocess CPU-wall: {total_subproc:.1f}s  (runner wall: {elapsed:.1f}s, parallelism: {args.jobs}x)")
        print(f"  P50: {p50:.2f}s  P90: {p90:.2f}s  P95: {p95:.2f}s  P99: {p99:.2f}s  Max: {max_t:.2f}s")
        print(f"  <1s: {fast} files ({fast/len(times)*100:.0f}%)  <2s: {fast_2s} files ({fast_2s/len(times)*100:.0f}%)")
        # Top 10 slowest files — likely the ones dragging the run.
        slowest = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top 10 slowest:")
        for f, t in slowest:
            print(f"    {t:>6.2f}s  {_format_file(f, repo_root)}")

    had_failures = False
    if failures:
        print()
        print("=== Failure output ===")
        for file, output, _summary in failures:
            print()
            print(f"--- {_format_file(file, repo_root)} ---")
            print(output.rstrip())
        print()
        # Split: files with actual test failures vs non-zero exit for other reasons
        test_fail_files = [(f, s) for f, _o, s in failures if s.get("failed", 0) > 0]
        all_passed_but_nonzero = [(f, s) for f, _o, s in failures
                                  if s.get("failed", 0) == 0 and s.get("passed", 0) > 0]
        no_tests_ran = [(f, s) for f, _o, s in failures
                        if s.get("failed", 0) == 0 and s.get("passed", 0) == 0]
        if test_fail_files:
            total_tf = sum(s.get("failed", 0) for _, s in test_fail_files)
            print(f"=== {len(test_fail_files)} file{'s' if len(test_fail_files) != 1 else ''} with test failures ({total_tf} test{'s' if total_tf != 1 else ''} failed) ===")
            for file, s in test_fail_files:
                nf = s.get("failed", 0)
                print(f"  {_format_file(file, repo_root)}  ({nf} test{'s' if nf != 1 else ''} failed)")
        if all_passed_but_nonzero:
            print(f"=== {len(all_passed_but_nonzero)} file{'s' if len(all_passed_but_nonzero) != 1 else ''} where all tests passed but pytest exited non-zero (warnings-as-errors, hook failures, etc.) ===")
            for file, s in all_passed_but_nonzero:
                print(f"  {_format_file(file, repo_root)}  ({s.get('passed', 0)} passed)")
        if no_tests_ran:
            print(f"=== {len(no_tests_ran)} file{'s' if len(no_tests_ran) != 1 else ''} where no tests ran (collection/import error, timeout before collection, etc.) ===")
            for file, s in no_tests_ran:
                print(f"  {_format_file(file, repo_root)}")
        had_failures = True

    # ── No-op guard: count EXECUTED tests, don't trust exit codes ────────────
    # This is the single aggregate-verdict site (test-gate-honesty §5). It
    # catches the silent no-op: a suite that collected 0 tests reads "green"
    # only because nothing ran. Every no-op cause (missing dep → importorskip
    # storm, marker-emptied file, fixture-nuked collection, exit-5→0 coercion)
    # funnels through "0 executed" here.
    total_executed = tests_passed + tests_failed
    noop_red = _noop_guard(
        all_summaries=all_summaries,
        total_executed=total_executed,
        explicit_files=explicit_files,
        strict_noop=args.strict_noop,
        min_tests=args.min_tests,
        repo_root=repo_root,
    )

    if had_failures or noop_red:
        return 1

    return 0


def _noop_guard(
    *,
    all_summaries: List[Tuple[Path, Dict[str, int]]],
    total_executed: int,
    explicit_files: set[Path],
    strict_noop: bool,
    min_tests: int | None,
    repo_root: Path,
) -> bool:
    """Fail LOUD when a suite collected/executed zero tests.

    Returns True if the run should be RED for a no-op reason. Prints loud
    banners for each trip. The gates, in order of severity:

      1. Whole-suite-zero (hard RED, always): total executed == 0 across the
         entire invoked set is never legitimate — the Ace-named "green because
         it didn't run" failure.
      2. Explicitly-requested 0-collect (RED under --strict-noop): a file the
         caller named specifically (--files / positional .py) that executed 0
         tests is a no-op the caller did not intend. Scoped to explicit
         requests so a default-discovery marker-filtered file stays GREEN.
      3. --min-tests N floor (opt-in): fewer than N executed → RED.

    Skip-storms (skipped>0, passed==0, failed==0) are a loud ⚠ *surfacing*,
    not a gate — a dep-missing skip-storm stays visible even where the dep is
    genuinely optional, without false-reding an optional-dep environment.
    """
    def _executed(s: Dict[str, int]) -> int:
        return s.get("passed", 0) + s.get("failed", 0)

    red = False

    # ── ⚠ skip-storm surfacing (unconditional, never a gate) ──────────────
    skip_storms = [
        (f, s) for f, s in all_summaries
        if s.get("skipped", 0) > 0 and s.get("passed", 0) == 0 and s.get("failed", 0) == 0
    ]
    if skip_storms:
        print()
        print(f"⚠  {len(skip_storms)} file{'s' if len(skip_storms) != 1 else ''} skipped every test and executed none "
              f"(importorskip fired? missing optional dep?) — surfaced, not failed:")
        for f, s in skip_storms:
            print(f"  ⚠  {_format_file(f, repo_root)}  ({s.get('skipped', 0)} skipped, 0 run)")

    # ── Gate 1: whole-suite-zero (hard RED, always) ───────────────────────
    if total_executed == 0:
        print()
        print("=== NO TESTS EXECUTED — suite is a no-op (missing dep? bad selector? all filtered?) ===")
        print("    A run that executed 0 tests is RED, not green (test-gate-honesty §5).")
        red = True

    # ── Gate 2: explicitly-requested paths that collected 0 tests ─────────
    if explicit_files:
        noop_explicit = [
            (f, s) for f, s in all_summaries
            if f.resolve() in explicit_files and _executed(s) == 0
        ]
        if noop_explicit:
            print()
            verb = "RED" if strict_noop else "⚠ (--no-strict-noop: surfaced only)"
            print(f"=== {len(noop_explicit)} explicitly-requested file(s) collected 0 tests (no-op) — {verb} ===")
            for f, s in noop_explicit:
                tag = " [exit-5 no-op]" if s.get("noop_exit5") else ""
                print(f"    {_format_file(f, repo_root)}{tag}")
            if strict_noop:
                red = True

    # ── Gate 3: --min-tests floor (opt-in) ────────────────────────────────
    if min_tests is not None and total_executed < min_tests:
        print()
        print(f"=== EXECUTED {total_executed} tests, below --min-tests floor of {min_tests} — RED ===")
        red = True

    return red


if __name__ == "__main__":
    sys.exit(main())
