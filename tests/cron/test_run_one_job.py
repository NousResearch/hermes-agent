"""Characterization + unit tests for the `run_one_job` shared helper (Phase 4A).

`tick`'s per-job body (`_process_job`) is the execute → save → deliver → mark
sequence that fires ONE due job. Phase 4A extracts it into a module-level
`run_one_job(job, *, adapters=None, loop=None, verbose=False)` so the external
Chronos provider's `fire_due` can reuse the IDENTICAL body — no duplicated
correctness.

The first test characterizes the sequence as driven through `tick()` (proving
the extraction didn't change `tick`'s behavior); the rest unit-test the
extracted helper directly.
"""
import cron.scheduler as s


def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, silent_marker_in=None):
    """Patch the job pipeline primitives and record the call order."""
    calls = []

    def fake_run_job(job):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, trigger="scheduled"):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "advance_next_run", lambda jid: True)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_manual_trigger_is_tagged_and_excluded_from_repeat_budget(monkeypatch):
    """A job fired with trigger_source='manual' (via trigger_job/`hermes cron
    run`) tags the saved output header AND passes trigger='manual' to
    mark_job_run so the run is unambiguous and does not consume the repeat
    budget. Regression guard for the 2026-07-09 incident where manual debug
    re-runs looked identical to scheduler double-fires."""
    saved = {}
    marked = {}

    monkeypatch.setattr(
        s, "run_job",
        lambda job: (True, "# Cron Job: t\n\n**Run Time:** now\n\nbody", "resp", None),
    )
    monkeypatch.setattr(
        s, "save_job_output",
        lambda jid, out: saved.update(jid=jid, out=out) or f"/tmp/{jid}.txt",
    )
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, trigger="scheduled": marked.update(
            jid=jid, trigger=trigger
        ),
    )

    ok = s.run_one_job({"id": "jm", "name": "t", "trigger_source": "manual"})

    assert ok is True
    assert marked["trigger"] == "manual"
    assert "**Trigger:** manual" in saved["out"]
    # header line inserted immediately before Run Time, once
    assert saved["out"].count("**Trigger:**") == 1


def test_scheduled_fire_is_untagged(monkeypatch):
    """A normal scheduled fire (no trigger_source) leaves the output header
    unchanged and reports trigger='scheduled' — byte-identical to pre-change
    behavior for the common path."""
    saved = {}
    marked = {}
    monkeypatch.setattr(
        s, "run_job",
        lambda job: (True, "# Cron Job: t\n\n**Run Time:** now\n\nbody", "resp", None),
    )
    monkeypatch.setattr(
        s, "save_job_output",
        lambda jid, out: saved.update(out=out) or f"/tmp/{jid}.txt",
    )
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, trigger="scheduled": marked.update(
            trigger=trigger
        ),
    )

    s.run_one_job({"id": "js", "name": "t"})

    assert marked["trigger"] == "scheduled"
    assert "**Trigger:**" not in saved["out"]


def test_run_one_job_silent_skips_delivery(monkeypatch):
    """A [SILENT] final response saves output + marks the run but does NOT
    deliver."""
    calls = _patch_pipeline(monkeypatch, silent_marker_in="[SILENT]")

    s.run_one_job({"id": "j3", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "run_job" in kinds and "save" in kinds and "mark" in kinds
    assert "deliver" not in kinds


def test_run_one_job_empty_response_is_soft_failure(monkeypatch):
    """An empty final response marks the run as NOT ok (issue #8585)."""
    calls = _patch_pipeline(monkeypatch, final="   ")

    s.run_one_job({"id": "j4", "name": "t"})

    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j4", False)


def test_run_one_job_failed_job_delivers_error(monkeypatch):
    """A failed job still delivers (the error notice) and marks not-ok."""
    calls = _patch_pipeline(monkeypatch, success=False, final="", error="boom")

    s.run_one_job({"id": "j5", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "deliver" in kinds  # failures always deliver
    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j5", False)


def test_run_one_job_exception_marks_failure(monkeypatch):
    """If run_job raises, the helper marks the run failed and returns False
    rather than propagating."""
    def boom(job):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(s, "run_job", boom)
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, trigger="scheduled": marks.append((jid, ok)),
    )

    ok = s.run_one_job({"id": "j6", "name": "t"})

    assert ok is False
    assert marks == [("j6", False)]
