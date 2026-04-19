"""Subprocess sandbox for code-candidate fitness evaluation.

Running arbitrary LLM-generated code in-process is obviously unsafe —
an infinite loop stalls the evaluator, a recursive function blows the
stack, an ``open("/etc/passwd")`` reads files outside the experiment.
This module runs each candidate in a fresh ``python3`` subprocess with:

* **Wall-clock timeout** via ``subprocess.run(..., timeout=...)``.
* **CPU & memory rlimits** set in the child's ``preexec_fn`` — so an
  unbounded loop is killed even if the Python GIL prevents the parent's
  timer from firing on time. Only applied on POSIX; Windows skips them
  because ``resource.setrlimit`` is not available there.
* **Temp working directory** — the candidate can't see or mutate the
  experiment state; we pass any test files explicitly.
* **No network, by convention** — we don't actually block syscalls
  (that needs seccomp-bpf or firejail, out of scope). The skill
  documents that user-supplied code should not be trusted for network
  secrets regardless.

The fitness function calls :func:`run_candidate_code` (sync, for use
inside a thread-pool worker) or :func:`run_pytest_suite` (sync, returns
a pass fraction 0..1). Both return a :class:`SandboxResult`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:  # pragma: no cover — POSIX only
    import resource
    _HAS_RLIMIT = True
except ImportError:  # pragma: no cover — Windows
    resource = None  # type: ignore[assignment]
    _HAS_RLIMIT = False


@dataclass
class SandboxResult:
    """Outcome of one sandboxed execution.

    ``ok`` is True iff the process exited with code 0 within the limits.
    ``stdout`` and ``stderr`` are captured in full up to ~1 MB.
    ``timed_out`` is True only when the wall-clock killed the process
    (distinct from ``returncode != 0`` which could be a user exception).
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def _apply_rlimits(cpu_s: float, mem_mb: int) -> None:  # pragma: no cover
    """preexec_fn — install resource caps in the child, best-effort.

    Limits vary by platform: Linux enforces both CPU and virtual address
    space cleanly, but macOS rejects ``setrlimit(RLIMIT_AS, ...)`` with
    EINVAL in many cases, and BSD-family systems may lack it entirely.
    We apply what the kernel accepts and silently skip the rest — the
    wall-clock ``subprocess.run(timeout=...)`` in the parent is always a
    backstop.
    """
    if not _HAS_RLIMIT:
        return
    soft_cpu = max(1, int(cpu_s + 1))
    for limit_attr, value in (
        ("RLIMIT_CPU",  (soft_cpu, soft_cpu)),
        ("RLIMIT_AS",   (max(64, mem_mb) * 1024 * 1024,) * 2),
        ("RLIMIT_DATA", (max(64, mem_mb) * 1024 * 1024,) * 2),
        ("RLIMIT_CORE", (0, 0)),
    ):
        limit = getattr(resource, limit_attr, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, value)
        except (ValueError, OSError):
            # The kernel rejected this particular limit — carry on;
            # other limits and the parent's timeout still apply.
            pass


def _run_subprocess(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    cpu_s: float,
    mem_mb: int,
    env: Optional[dict] = None,
) -> SandboxResult:
    import time
    start = time.perf_counter()
    preexec = (lambda: _apply_rlimits(cpu_s, mem_mb)) if _HAS_RLIMIT else None
    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd),
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
            preexec_fn=preexec,  # type: ignore[arg-type]
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return SandboxResult(
            ok=False, returncode=-9,
            stdout=stdout[:1_000_000], stderr=stderr[:1_000_000],
            timed_out=True, duration_s=duration,
        )
    duration = time.perf_counter() - start
    return SandboxResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout[:1_000_000],
        stderr=proc.stderr[:1_000_000],
        timed_out=False,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def run_candidate_code(
    code: str,
    *,
    entry: str = "solution.py",
    extra_files: Optional[dict[str, str]] = None,
    argv: Sequence[str] = (),
    timeout_s: float = 10.0,
    cpu_s: float = 8.0,
    mem_mb: int = 256,
) -> SandboxResult:
    """Write *code* to a temp dir and execute it with ``python3``.

    ``extra_files`` maps relative path → content and is useful for test
    inputs or fixtures the candidate needs. ``argv`` is passed as
    trailing arguments to the candidate's entry point.
    """
    with tempfile.TemporaryDirectory(prefix="evolver-sbx-") as tmp:
        cwd = Path(tmp)
        (cwd / entry).write_text(code, encoding="utf-8")
        for rel, content in (extra_files or {}).items():
            path = cwd / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return _run_subprocess(
            [sys.executable, entry, *argv],
            cwd=cwd, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
        )


def run_pytest_suite(
    candidate_code: str,
    test_code: str,
    *,
    candidate_name: str = "solution.py",
    test_name: str = "test_solution.py",
    timeout_s: float = 30.0,
    cpu_s: float = 25.0,
    mem_mb: int = 512,
) -> tuple[SandboxResult, float]:
    """Run pytest on *test_code* against *candidate_code*.

    Returns ``(SandboxResult, pass_fraction)`` where ``pass_fraction`` is
    in ``[0.0, 1.0]`` on successful runs, derived from pytest's ``passed``
    and ``failed`` counts in the terse summary line. On timeout or crash
    the pass fraction is 0.
    """
    with tempfile.TemporaryDirectory(prefix="evolver-pytest-") as tmp:
        cwd = Path(tmp)
        (cwd / candidate_name).write_text(candidate_code, encoding="utf-8")
        (cwd / test_name).write_text(test_code, encoding="utf-8")
        result = _run_subprocess(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "--tb=no", "-p", "no:cacheprovider", test_name],
            cwd=cwd, timeout_s=timeout_s, cpu_s=cpu_s, mem_mb=mem_mb,
        )
    frac = _parse_pytest_summary(result.stdout) if not result.timed_out else 0.0
    return result, frac


def _parse_pytest_summary(stdout: str) -> float:
    """Return the pass fraction from pytest terse output.

    Recognises lines like ``"3 passed, 1 failed in 0.02s"``. When the
    summary is missing (collection error, no tests) we return 0.
    """
    import re
    passed = failed = 0
    for line in stdout.splitlines():
        if " passed" in line or " failed" in line:
            m_pass = re.search(r"(\d+) passed", line)
            m_fail = re.search(r"(\d+) failed", line)
            if m_pass:
                passed = max(passed, int(m_pass.group(1)))
            if m_fail:
                failed = max(failed, int(m_fail.group(1)))
    total = passed + failed
    return passed / total if total else 0.0
