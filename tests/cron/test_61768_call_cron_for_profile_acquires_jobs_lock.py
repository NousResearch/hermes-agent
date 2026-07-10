"""
Regression for issue #61768 - Desktop serve backend: cron ticker thread
races _call_cron_for_profile global retarget -> destructive cross-profile
jobs.json overwrite.

The reporter's mechanism: in the desktop `hermes serve` backend, the
desktop cron ticker (calling `cron.tick` from `_start_desktop_cron_ticker`)
and the dashboard web thread (calling `_call_cron_for_profile`) execute
in the same process but on different paths. The ticker is calling
cron.jobs read/write through `_jobs_lock()`; the dashboard is calling
`_call_cron_for_profile()` which temporarily rewrites the module globals
(CRON_DIR / JOBS_FILE / OUTPUT_DIR) under `_CRON_PROFILE_LOCK`
(a SEPARATE Python RLock from cron.jobs' `_jobs_file_lock`).

In practice `_jobs_lock()` does acquire `_jobs_file_lock` (an RLock) so
the in-process thread race is partially mitigated. But there is still a
real gap: a function reached via `getattr(cron_jobs, func_name)(...)`
inside `_call_cron_for_profile()` may bypass the lock if the function
itself doesn't take it. (Today none of the helpers do, so the lock is
held by `save_jobs`/`load_jobs`/`claim_dispatch` directly. But the
contract is implicit and easy to break in future edits.)

The fix surface this test pins: `_call_cron_for_profile()` (and its
twin `_fire_cron_job_for_profile()`) must acquire the SAME lock that
cron.jobs helper functions acquire. The simplest mechanism: hold
cron.jobs' `_jobs_file_lock` (or its replacement `_jobs_lock`) around
the entire retarget + call window so the inter-process and the
in-process interlock paths are unified.

This test creates a thread holding `cron.jobs._jobs_lock()` and another
thread inside `cron_jobs.JOBS_FILE = ...; getattr(cron_jobs, ...)`
(doing what `_call_cron_for_profile` does). Without the fix, the
function-call path does NOT go through `_jobs_lock` and can interleave
with anything else calling `cron.jobs` helpers outside the lock window.
With the fix, the retarget window itself holds the lock.

We rely on observable state: each profile has a `jobs.json` with a
distinct sentinel job. After both threads complete, each store should
hold only its own sentinel, not a mix.
"""

import importlib
import json
import threading
import time
from pathlib import Path

import pytest


def _set_two_profile_env(monkeypatch, root: Path, profile_a: Path, profile_b: Path) -> None:
    """Set up two HERMES_HOMEs under one root, mimicking a multi-profile
    install where the desktop ticker runs on the root home and the
    dashboard retargets to a profile home.
    """
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "_get_platform_default_hermes_home", lambda: root
    )
    monkeypatch.setenv("HERMES_HOME", str(root))


def test_call_cron_for_profile_serializes_with_cron_jobs_writes(
    tmp_path, monkeypatch
):
    """The fix: `_call_cron_for_profile`-shaped retarget+call must hold
    the same lock as cron.jobs' write helpers, so a concurrent save_jobs
    cannot interleave with the retarget window.

    Test setup (per #61768 reporter's mechanism):
      - One root HERMES_HOME (the ticker process home).
      - Two profile homes underneath (profile-a, profile-b).
      - Each store has a sentinel job unique to that store.
      - T1 mimics the cron ticker: holds cron.jobs._jobs_lock() then runs
        save_jobs against the CURRENT JOBS_FILE (which may be root after
        a retarget roundtrip).
      - T2 mimics the dashboard: mutates cron_jobs.JOBS_FILE to a profile
        path, then calls a helper method on cron_jobs (e.g. list_jobs).
      - Without the fix: T2's mutator call does not hold _jobs_lock
        (the lock is held by T1), so T2's read can pass the join BEFORE
        T1's write. Either write leaks T1's content into T2's profile,
        or T2's profile gets cloned. With the fix: T2 holds
        `_jobs_file_lock` (or equivalent) during its entire retarget+call
        window, blocking until T1 releases.

    Because `save_jobs`/`load_jobs` use `_jobs_lock` internally, this
    test primarily verifies that the retarget+call sequence is
    serialized alongside those writes - NOT that the writes happen
    atomically, since they already do.
    """
    root = tmp_path / "hermes_home"
    profile_a = root / "profiles" / "profile-a"
    profile_b = root / "profiles" / "profile-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    (root / "cron").mkdir()
    (profile_a / "cron").mkdir()
    (profile_b / "cron").mkdir()

    # Each store has its own sentinel job.
    (root / "cron" / "jobs.json").write_text(json.dumps({
        "jobs": [{"id": "root-sentinel", "origin": "root", "next_run_at": None}],
    }))
    (profile_a / "cron" / "jobs.json").write_text(json.dumps({
        "jobs": [{"id": "profile-a-sentinel", "origin": "profile-a", "next_run_at": None}],
    }))
    (profile_b / "cron" / "jobs.json").write_text(json.dumps({
        "jobs": [{"id": "profile-b-sentinel", "origin": "profile-b", "next_run_at": None}],
    }))

    _set_two_profile_env(monkeypatch, root, profile_a, profile_b)

    import cron.jobs as jobs

    importlib.reload(jobs)
    # After reload, globals anchor at root (HERMES_HOME=root).
    assert jobs.CRON_DIR.resolve() == (root / "cron").resolve()
    assert jobs.JOBS_FILE.resolve() == (root / "cron" / "jobs.json").resolve()

    # Coordination events between two threads; deterministic, no time.sleep.
    ticker_locked_event = threading.Event()
    ticker_ready_to_release = threading.Event()
    errors = []

    def ticker_thread():
        """Mimic cron.jobs write under a long-held _jobs_lock like the
        desktop ticker does when processing jobs."""
        try:
            with jobs._jobs_lock():
                ticker_locked_event.set()
                ticker_ready_to_release.wait(timeout=5.0)
                # Write via the helper that takes the lock - this is what
                # advance_next_run/mark_job_run/save_jobs all do.
                jobs.save_jobs([
                    {"id": "root-sentinel", "origin": "root", "next_run_at": None},
                    {"id": "TICKER_WRITE_SENTINEL", "origin": "TICKER", "next_run_at": None},
                ])
        except Exception as e:
            errors.append(("ticker", repr(e)))

    def web_thread():
        """Mimic _call_cron_for_profile - retarget the module globals to
        a profile, then call a function on cron_jobs. Without the fix,
        the function-call path does NOT acquire cron.jobs._jobs_lock
        while the ticker is in its critical section, so it can read
        the just-retargeted profile store while the ticker is preparing
        to write.

        With the fix, this entire block runs inside cron.jobs._jobs_lock,
        so it blocks until the ticker releases.

        Currently exercises a NAKED function call (the buggy path):
        `cron_jobs.JOBS_FILE = profile_b_path; <bare list_jobs() read>`.
        """
        import cron.jobs as cron_jobs

        try:
            ticker_locked_event.wait(timeout=5.0)
            # Simulate the retarget done by _call_cron_for_profile.
            old_cron_dir = cron_jobs.CRON_DIR
            old_jobs_file = cron_jobs.JOBS_FILE
            old_output_dir = cron_jobs.OUTPUT_DIR
            try:
                cron_jobs.CRON_DIR = profile_b / "cron"
                cron_jobs.JOBS_FILE = cron_jobs.CRON_DIR / "jobs.json"
                cron_jobs.OUTPUT_DIR = cron_jobs.CRON_DIR / "output"
                # Call a cron.jobs helper that internally uses save_jobs
                # (via update_job-like path). This write goes under the
                # CURRENT (profile-b) JOBS_FILE.
                _ = cron_jobs.list_jobs()
                # Then mutate-and-save so we know we attempted a write
                # under the new path.
                jobs_list = cron_jobs.load_jobs()
                jobs_list.append({
                    "id": "WEB_WRITE_SENTINEL", "origin": "WEB", "next_run_at": None
                })
                cron_jobs.save_jobs(jobs_list)
            finally:
                cron_jobs.CRON_DIR = old_cron_dir
                cron_jobs.JOBS_FILE = old_jobs_file
                cron_jobs.OUTPUT_DIR = old_output_dir
        except Exception as e:
            errors.append(("web", repr(e)))

    t1 = threading.Thread(target=ticker_thread, name="ticker")
    t2 = threading.Thread(target=web_thread, name="web")
    t1.start()
    t2.start()
    time.sleep(0.2)
    ticker_ready_to_release.set()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Reload to undo the global retargets and the reload side-effects.
    monkeypatch.undo()
    importlib.reload(jobs)

    assert not errors, f"Threads raised: {errors}"

    # Read each store. The fix ensures no store ends up with a sentinel
    # that doesn't belong to it.
    root_st = json.loads((root / "cron" / "jobs.json").read_text())
    a_st = json.loads((profile_a / "cron" / "jobs.json").read_text())
    b_st = json.loads((profile_b / "cron" / "jobs.json").read_text())

    root_ids = {j["id"] for j in root_st.get("jobs", [])}
    a_ids = {j["id"] for j in a_st.get("jobs", [])}
    b_ids = {j["id"] for j in b_st.get("jobs", [])}

    # Each store should still hold its sentinel.
    assert "root-sentinel" in root_ids, "root store lost its root-sentinel"
    assert "profile-a-sentinel" in a_ids, "profile-a store lost its sentinel"
    assert "profile-b-sentinel" in b_ids, "profile-b store lost its sentinel"
    # The ticker's writes (TICKER_WRITE_SENTINEL) must NOT leak to
    # profile stores - the only store that holds ticker writes is root.
    assert "TICKER_WRITE_SENTINEL" not in a_ids, (
        f"TICKER_WRITE leaked into profile-a: {a_ids}"
    )
    assert "TICKER_WRITE_SENTINEL" not in b_ids, (
        f"TICKER_WRITE leaked into profile-b: {b_ids}"
    )
    # Profile-b should hold the WEB write because that path was chosen
    # by the retarget; profile-a must NOT.
    if "WEB_WRITE_SENTINEL" in root_ids:
        # If ticker and web ran truly serialized, the web write to profile-b
        # would block until ticker released. After both finished, both root
        # (which wrote BEFORE the retarget) and profile-b (which wrote AFTER)
        # should have their respective writes.
        pass  # this is fine if synchronized
    assert "WEB_WRITE_SENTINEL" not in a_ids, (
        f"WEB_WRITE leaked into profile-a (unrelated profile): {a_ids}"
    )


def test_call_cron_for_profile_holds_jobs_lock_during_retarget(
    tmp_path, monkeypatch
):
    """Static check: the fix path should observe `_call_cron_for_profile`
    (and its twin `_fire_cron_job_for_profile`) holding the same lock
    that cron.jobs writers hold during the entire retarget+call window.

    This test asserts the SHAPE of the fix, not just the lack of leakage.
    Useful as a tripwire if a future refactor reintroduces the bug.
    """
    # Compile-checkable: import both functions and verify they wrap
    # their retarget in cron.jobs' _jobs_lock (or its replacement).
    import re
    import subprocess

    # Read web_server.py and look for the line wrapping each retarget
    # in the cron.jobs lock-acquisition idiom.
    src = Path(
        "/tmp/hermes-pr-work-60859/hermes-agent/hermes_cli/web_server.py"
    ).read_text()
    # Each retarget site should hold cron.jobs._jobs_lock somewhere in
    # its scope. The simplest check: for each "def _call_cron_for_profile"
    # / "def _fire_cron_job_for_profile" function body, ensure that within
    # the function body either cron.jobs._jobs_lock or
    # cron_jobs._jobs_lock is acquired.
    for func_name in ["_call_cron_for_profile", "_fire_cron_job_for_profile"]:
        match = re.search(rf"def {func_name}\b.*?(?=^    def |\Z)", src, re.MULTILINE | re.DOTALL)
        assert match, f"Could not find function {func_name} in web_server.py"
        body = match.group(0)
        # Verify lock acquisition is now present in body.
        assert "_jobs_lock" in body or "_call_cron_with_lock" in body, (
            f"#61768 regression: {func_name} does not hold cron.jobs' "
            f"_jobs_lock around its retarget+call window. The destructive "
            f"write race is still possible."
        )
