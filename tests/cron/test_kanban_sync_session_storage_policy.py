"""Regression test: the kanban-sync policy when session storage is unavailable.

Policy decided in kanban task t_04103601 (incident sb-5e4384af, 2026-08-03:
kanban-sync failed 7 consecutive cron runs with a bare
``RuntimeError: [Errno 17] File exists: '/Users/craigrichardson/.hermes/sessions'``
because ``~/.hermes/sessions`` is a symlink into the Elements external volume
and the volume had dropped out, leaving the symlink dangling).

Evidence for the policy (see the task result for the full trace):

1. The kanban-sync cron job (job_id ff40656236ef) runs with
   ``no_agent: true`` and ``script: kanban-sync.sh`` (verified in
   ``~/.hermes/cron/jobs.json``). ``cron/scheduler.py::run_job`` short-circuits
   no_agent jobs BEFORE importing ``run_agent.AIAgent`` or constructing
   ``hermes_state.SessionDB`` — the comment at the branch reads "We check this
   BEFORE importing run_agent / constructing SessionDB so a pure-script tick
   never pays for the agent machinery it isn't going to use."
2. The script itself (``scripts/kanban-sync.sh`` in syntheos-system) is a bash
   stub around ``node scripts/kanban-sync.js``, which touches only
   ``~/.hermes/kanban.db`` and Supabase REST. Neither file references
   ``~/.hermes/sessions`` anywhere.
3. Therefore session persistence is not merely optional for kanban-sync — it
   is not on the execution path at all. Degradation is lossless by
   construction: there is no session-dependent operation to skip.
4. What the 2026-08-03 incident actually exposed was a *legibility* defect in
   the agent-era path: ``mkdir(exist_ok=True)`` on a dangling symlink raises
   an opaque EEXIST. That is now handled by ``gateway/session.py``
   (``SessionStorageUnavailableError`` / ``session_storage_health``), which
   fails fast with a clear message — the correct behavior there, because an
   agent-mode job that cannot persist its session would lose the entire run
   transcript. Tests for that legibility fix live in
   ``tests/gateway/test_session_storage_availability.py`` and
   ``tests/gateway/test_session_storage_health.py``.

This test pins the sync-side policy: a no_agent script job whose
``HERMES_HOME/sessions`` is a dangling symlink (target on an unmounted
volume, simulated entirely under tmp_path — the real volume is never touched)
must still run to completion, because nothing on that path reads or writes
session storage. If someone later moves the sync back to agent mode, or adds
a session dependency to the no_agent path, this test is the tripwire.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def test_no_agent_sync_job_runs_with_dangling_sessions_symlink(hermes_env, caplog):
    """kanban-sync policy: session storage is not on the no_agent path, so a
    dangling ~/.hermes/sessions symlink (backing volume unmounted) must not
    fail, skip, or alter the sync tick — the script runs and its output is
    delivered exactly as when storage is healthy. The storage failure must
    also be LEGIBLE: a WARNING naming the unavailable target, not a silently
    swallowed EEXIST (the pre-fix behavior hid it at DEBUG)."""
    import logging

    from cron.jobs import create_job
    from cron.scheduler import run_job

    # Simulate the incident condition entirely in the temp home: a sessions
    # symlink whose target lives on an "unmounted volume" that does not exist.
    unmounted_target = hermes_env / "unmounted_volume" / "sessions"
    assert not unmounted_target.exists()
    sessions_link = hermes_env / "sessions"
    sessions_link.symlink_to(unmounted_target)
    assert sessions_link.is_symlink()
    assert not sessions_link.exists()  # dangling — the incident precondition

    # A sync-shaped script: does its work and reports, touching only
    # kanban.db-class state (here, a marker file proving it executed).
    marker = hermes_env / "sync_ran.marker"
    script_path = hermes_env / "scripts" / "kanban-sync.sh"
    script_path.write_text(
        "#!/bin/bash\n"
        "echo 'Sync complete: 0 created, 0 updated, 0 pushed'\n"
        f"touch {marker}\n"
    )

    job = create_job(
        prompt=None, schedule="0 */4 * * *",
        script="kanban-sync.sh", no_agent=True, deliver="local",
    )
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        success, doc, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert "Sync complete" in final_response
    assert marker.exists()  # the sync actually ran, not silently skipped

    # The storage failure is legible: a WARNING-level diagnostic naming the
    # unavailable session-storage target, distinct from a bare EEXIST.
    warning_text = "\n".join(
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING
    )
    assert "Session storage unavailable" in warning_text
    assert str(unmounted_target) in warning_text

    # And the dangling symlink must be untouched: the no_agent path must never
    # "helpfully" materialize a fresh sessions tree over the missing mount
    # point (that is the worse failure session_storage_health guards against).
    assert sessions_link.is_symlink()
    assert not sessions_link.exists()
    assert not unmounted_target.exists()


def test_no_agent_sync_job_runs_with_no_sessions_entry_at_all(hermes_env):
    """Companion case: with no sessions path present at all, the no_agent sync
    tick completes. Note: ensure_hermes_home() (reached via load_config inside
    _get_script_timeout) legitimately creates a HEALTHY empty sessions dir as
    part of first-use home setup; the policy above only forbids materializing
    one over a dangling symlink / unmounted mount point."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    sessions_path = hermes_env / "sessions"
    assert not sessions_path.exists() and not sessions_path.is_symlink()

    script_path = hermes_env / "scripts" / "kanban-sync.sh"
    script_path.write_text("#!/bin/bash\necho 'Sync complete'\n")

    job = create_job(
        prompt=None, schedule="0 */4 * * *",
        script="kanban-sync.sh", no_agent=True, deliver="local",
    )
    success, doc, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert "Sync complete" in final_response
    # Healthy first-use creation is fine; it is a real directory on local
    # storage, not a shadow over an unmounted volume.
    assert sessions_path.is_dir() and not sessions_path.is_symlink()
