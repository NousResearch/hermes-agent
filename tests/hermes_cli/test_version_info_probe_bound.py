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

    def slow_probe(argv, cwd, timeout):
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

    def counting_probe(argv, cwd, timeout):
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
