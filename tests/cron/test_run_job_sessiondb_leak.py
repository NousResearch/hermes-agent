"""
Regression tests for cron/scheduler.py SQLite session-store fd leak.

Issue #60859 - `run_job()` opens a `SessionDB()` connection eagerly at
the top of the LLM (non-`no_agent`) path but only `close()`s it inside
the trailing `finally` block. Several early-return paths (wake-gate,
prompt-injection block, no-output prerun) return BEFORE that finally,
so each occurrence silently leaks one open SQLite connection / fd.
A gateway ticking wake-gated cron jobs every 60s accumulates ~1 leaked
fd per tick; over hours it hits `EMFILE`.

These tests cover each of the three leaked early-return paths and the
fix described by the maintainer: "Constructing [SessionDB] inside [the]
`try` block makes the existing `finally` always close it, and means
the skip paths never open a connection at all."
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

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


class _CountingSessionDB:
    """Stand-in for hermes_state.SessionDB.

    Records method calls so a test can assert that close() was called
    after every reached-end of run_job. Trackable on the class so we
    don't need monkeypatch gymnastics.
    """

    instances = 0
    open_instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1
        type(self).open_instances += 1
        self.closed = False
        self.set_session_title_calls = []
        self.end_session_calls = []

    def close(self):
        self.closed = True
        type(self).open_instances -= 1

    def set_session_title(self, sid, title):
        self.set_session_title_calls.append((sid, title))

    def end_session(self, sid, reason):
        self.end_session_calls.append((sid, reason))


@pytest.fixture
def counting_session_db(monkeypatch):
    """Install _CountingSessionDB in place of hermes_state.SessionDB."""
    _CountingSessionDB.instances = 0
    _CountingSessionDB.open_instances = 0

    # Import hermes_state FIRST so it's in sys.modules; otherwise our
    # setitem below KeyErrors on the missing key. The import must come
    # AFTER the hermes_env fixture reloads hermes_constants etc.
    import hermes_state  # noqa: F401

    import sys
    monkeypatch.setitem(
        sys.modules["hermes_state"].__dict__,
        "SessionDB",
        _CountingSessionDB,
    )
    return _CountingSessionDB


def _make_wake_gate_false_job(hermes_env, prompt="say hi"):
    """Job whose prerun script returns {"wakeAgent": false}.

    Drives the wake-gate branch in scheduler.py: if the prerun script
    succeeds AND the gate says wake=false, run_job() returns early.
    """
    from cron.jobs import create_job
    script_path = hermes_env / "scripts" / "gate.sh"
    script_path.write_text('#!/bin/bash\necho \'{"wakeAgent": false}\'\n')
    return create_job(
        prompt=prompt,
        schedule="every 5m",
        script="gate.sh",
        deliver="local",
    )


def test_wake_gate_returns_does_not_leak_session_db(
    hermes_env, counting_session_db
):
    """Strong invariant: after run_job returns via wake-gate, NO
    SessionDB instance should still be open.

    Catches the leak regardless of whether the fix is "open it later"
    or "open it inside a try/finally" - what matters is that opened
    connections are closed.
    """
    from cron.scheduler import run_job, SILENT_MARKER

    job = _make_wake_gate_false_job(hermes_env)
    success, doc, final_response, error = run_job(job)

    assert success is True
    assert final_response == SILENT_MARKER
    assert error is None

    assert _CountingSessionDB.open_instances == 0, (
        f"SessionDB.open_instances={_CountingSessionDB.open_instances} "
        f"after wake-gate return; expected 0. "
        f"#60859: { _CountingSessionDB.open_instances } leaked fds."
    )


def test_wake_gate_does_not_open_session_db_at_all(
    hermes_env, counting_session_db
):
    """Strict invariant for the FIX described in #60859: the wake-gate
    path should open ZERO SessionDBs because it returns early before
    the only place that needs one.

    This is the strongest possible test - it catches the leak directly,
    not just "opened == closed".
    """
    from cron.scheduler import run_job, SILENT_MARKER

    job = _make_wake_gate_false_job(hermes_env)
    instances_before = _CountingSessionDB.instances

    success, doc, final_response, error = run_job(job)
    assert success is True
    assert final_response == SILENT_MARKER

    new_instances = _CountingSessionDB.instances - instances_before
    assert new_instances == 0, (
        f"SessionDB() was opened {new_instances} times on the wake-gate "
        "path. The fix (#60859) moves construction AFTER the wake-gate; "
        "the gate path should open ZERO connections."
    )


def test_no_agent_path_never_opens_session_db(
    hermes_env, counting_session_db
):
    """no_agent jobs short-circuit at the very top of run_job and never
    touch the agent path at all - they should never construct a
    SessionDB.
    """
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho hi\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deliver="local",
    )

    instances_before = _CountingSessionDB.instances
    success, doc, final_response, error = run_job(job)
    assert success is True

    new_instances = _CountingSessionDB.instances - instances_before
    assert new_instances == 0, (
        f"no_agent job opened {new_instances} SessionDBs; should be zero."
    )
