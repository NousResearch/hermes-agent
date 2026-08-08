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

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **_kw):
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
    monkeypatch.setattr(s, "advance_next_runs", lambda ids: 1)

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


def test_run_one_job_records_silent_as_distinct_terminal_status(monkeypatch):
    _patch_pipeline(monkeypatch, silent_marker_in="[SILENT]")
    marked = {}

    def fake_mark(jid, ok, err=None, delivery_error=None, status=None):
        marked.update(job_id=jid, success=ok, status=status)

    monkeypatch.setattr(s, "mark_job_run", fake_mark)

    assert s.run_one_job({"id": "silent-1", "name": "quiet"}) is True
    assert marked == {
        "job_id": "silent-1",
        "success": True,
        "status": "silent",
    }


def test_run_one_job_records_delivery_failure_in_last_status(monkeypatch):
    _patch_pipeline(monkeypatch)
    marked = {}
    monkeypatch.setattr(
        s, "_deliver_result", lambda *a, **k: "platform send failed: 502",
    )

    def fake_mark(jid, ok, err=None, delivery_error=None, status=None):
        marked.update(
            success=ok, delivery_error=delivery_error, status=status,
        )

    monkeypatch.setattr(s, "mark_job_run", fake_mark)

    job = {"id": "delivery-1", "name": "report", "deliver": "telegram:ops"}
    assert s.run_one_job(job) is True
    assert marked["success"] is True
    assert marked["delivery_error"] == "platform send failed: 502"
    assert marked["status"] == "delivery_failed"


def test_run_one_job_pages_when_failure_trips_circuit_breaker(monkeypatch):
    calls = _patch_pipeline(monkeypatch, success=False, error="timeout")
    delivered_content = []

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        delivered_content.append(content)
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, status=None):
        calls.append(("mark", jid, ok))
        return {
            "id": jid,
            "last_run_at": "2026-08-08T01:00:00+00:00",
            "circuit_breaker_tripped_at": "2026-08-08T01:00:00+00:00",
        }

    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)

    assert s.run_one_job({"id": "breaker-1", "name": "report"}) is True
    deliveries = [call for call in calls if call[0] == "deliver"]
    assert len(deliveries) == 2
    assert "paused after 3 consecutive failed runs" in delivered_content[-1]


def test_run_one_job_installs_secret_scope_under_multiplex(monkeypatch, tmp_path):
    """Regression: under profile isolation (multiplex active), run_one_job must
    execute run_job inside a profile secret scope so credential reads
    (resolve_runtime_provider -> get_secret) don't fail-close with
    UnscopedSecretError, and must tear the scope down afterward.

    Behavior contract: a scope is present during run_job and absent after,
    regardless of the concrete secret values.
    """
    from agent import secret_scope as ss

    # Point cron's home resolution at a profile whose .env carries a secret.
    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    scope_during_run = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    ss.set_multiplex_active(True)
    try:
        ok = s.run_one_job({"id": "j7", "name": "t"})
    finally:
        ss.set_multiplex_active(False)

    assert ok is True
    # Scope was installed during run_job and the profile secret resolved.
    assert scope_during_run["scope"] is not None
    assert scope_during_run["base_url"] == "https://openrouter.ai/api/v1"
    # And it was torn down after run_one_job returned (no leak).
    assert ss.current_secret_scope() is None
