"""The version stamp on the request path is BOUNDED, not merely time-limited.

The 605 s turn stall: the engineering-lead agent's MainThread sat in
``hermes_cli/version_info.py`` -> ``subprocess.communicate`` while
``agent/turn_api_request.build_api_request`` -> ``portal_tags.hermes_client_tag`` ->
``get_version_info()`` resolved the running version on the first request of a process. The turn
made no progress, ``agent.turn_liveness`` aborted it at 604 s, lease renewal stopped, and the run
hung until the 3600 s stale timeout.

``subprocess.run(timeout=3)`` was already there (since the pm rework) and did not bound it, because
on Windows it is not a hard bound: ``run()`` kills the process IT spawned and then re-reads the
inherited pipes with no timeout. Git for Windows starts the real git from a launcher
(``cmd/git.exe`` -> ``mingw64/bin/git.exe``), so a killed launcher leaves a grandchild holding
those handles and the second read never sees EOF.

These tests pin the two properties that make the stall impossible: the probe runner never waits on
a pipe (a hanging launcher is killed at its deadline), and one resolution cannot exceed a shared
wall-clock budget - the result is cached when complete and deliberately not cached when truncated.
"""

from __future__ import annotations

import sys
import time

import psutil

from hermes_cli import version_info as vi


def setup_function() -> None:
    # tests/conftest.py pre-seeds the cache so the suite never walks git; these tests need it cold.
    vi._reset_version_info_cache()


def test_a_hanging_git_launcher_cannot_outlive_its_probe_deadline(tmp_path):
    """The launcher spawns a grandchild that inherits stdout (Git for Windows' shape), then hangs."""
    pid_file = tmp_path / "grandchild.pid"
    wrapper = tmp_path / "fake_git.py"
    wrapper.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    try:
        assert vi._probe([sys.executable, str(wrapper)], tmp_path, 1.0) is None
        elapsed = time.monotonic() - started
        # A pipe-based read blocks here until the orphaned grandchild exits (the 605 s stall); the
        # file-backed runner returns at its deadline.
        assert elapsed < 4.0, f"probe took {elapsed:.1f}s for a 1s deadline"
    finally:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if pid_file.exists():
            grandchild = int(pid_file.read_text(encoding="utf-8").strip() or 0)
            if grandchild and psutil.pid_exists(grandchild):
                psutil.Process(grandchild).kill()


def test_a_slow_walk_is_bounded_and_not_cached_as_unknown(tmp_path, monkeypatch):
    """A cold/contended checkout spends one bounded window, not nine probes, and keeps trying."""
    calls: list[float] = []

    def slow_probe(argv, cwd, timeout, *, budget=None):
        calls.append(timeout)
        time.sleep(0.3)

    monkeypatch.setattr(vi, "_probe", slow_probe)
    monkeypatch.setattr(vi, "_RESOLVE_BUDGET_SECONDS", 1.0)
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(vi, "_resolve_repo_dir", lambda: tmp_path)

    started = time.monotonic()
    info = vi.get_version_info()
    elapsed = time.monotonic() - started

    assert (info.base_version, info.commit) == ("unknown", None)
    assert elapsed < 2.5, f"resolution spent {elapsed:.1f}s against a 1.0s budget"
    assert len(calls) < 9, f"budget did not stop the walk: {len(calls)} probes ran"
    # Truncated, so NOT cached as the process's identity: the next window retries.
    assert vi._cached_version_info is None
    assert vi._retry_after_monotonic is not None


def test_a_complete_walk_is_cached_so_requests_never_pay_it_twice(tmp_path, monkeypatch):
    calls: list[tuple] = []

    def counting_probe(argv, cwd, timeout, *, budget=None):
        calls.append(tuple(argv))

    monkeypatch.setattr(vi, "_probe", counting_probe)
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(vi, "_resolve_repo_dir", lambda: tmp_path)

    first = vi.get_version_info()
    after_first = len(calls)
    assert after_first > 0, "the walk probed nothing"
    second = vi.get_version_info()

    assert second is first
    assert len(calls) == after_first, "a cached identity re-walked git"
    assert vi._cached_version_info is first


def test_stderr_never_becomes_part_of_the_probe_value(tmp_path):
    """HEAD read stdout only; a shared sink let a git warning join the parsed identity."""
    script = (
        "import sys; "
        "sys.stdout.write('c0ffee1c0ffee1c0ffee1c0ffee1c0ffee1c0ff\\n'); "
        "sys.stderr.write('warning: refname HEAD is ambiguous.\\n')"
    )

    value = vi._probe([sys.executable, "-c", script], tmp_path, 10.0)

    assert value == "c0ffee1c0ffee1c0ffee1c0ffee1c0ffee1c0ff", (
        f"stderr leaked into the probe value: {value!r}")


def test_a_probe_killed_at_its_deadline_marks_the_budget_truncated(tmp_path):
    """The kill branch must say so: a truncated walk and a complete "unknown" are different answers."""
    budget = vi._Budget(60.0)  # seconds of shared budget left: only the per-probe deadline fires
    started = time.monotonic()

    value = vi._probe(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.5, budget=budget)
    elapsed = time.monotonic() - started

    assert value is None
    assert elapsed < 3.0, f"the killed probe cost {elapsed:.1f}s against a 0.5s deadline"
    assert budget.truncated is True, "a killed probe left the walk looking complete"


def test_a_truncated_walk_is_not_cached_as_the_process_identity(tmp_path, monkeypatch):
    """One killed probe is enough: answer the partial identity, then keep retrying.

    The incident shape: a contended disk blocks the first ``rev-parse`` past its deadline, every
    later probe is fast. Caching that answer would make the missing commit permanent for the
    process, which is exactly what the review called out.
    """
    calls: list[tuple] = []

    def probe_after_a_killed_rev_parse(argv, cwd, timeout, *, budget=None):
        calls.append(tuple(argv))
        if len(calls) == 1:
            # What the shipped TimeoutExpired branch does, without spending 3 s here.
            if budget is not None:
                budget.truncated = True
            return None  # ``rev-parse HEAD``: killed at its deadline, so no commit
        # Every later probe is fast, as it is after the contended moment passes.
        if "--short=7" in argv:
            return "c0ffee1"
        if argv[1:2] == ["branch"]:
            return "main"
        return None

    monkeypatch.setattr(vi, "_probe", probe_after_a_killed_rev_parse)
    monkeypatch.setattr(vi, "_resolve_stamp_file", lambda: None)
    monkeypatch.setattr(vi, "_resolve_repo_dir", lambda: tmp_path)

    info = vi.get_version_info()

    # The killed probe was the one that would have answered with the commit; the rest answered.
    assert info.commit is None
    assert info.branch == "main"
    assert len(calls) > 1, "the walk stopped at the killed probe; one kill is not the whole answer"
    # Truncated, so NOT cached: the next window retries instead of answering for the process life.
    assert vi._cached_version_info is None
    assert vi._retry_after_monotonic is not None
