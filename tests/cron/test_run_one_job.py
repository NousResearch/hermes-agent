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
import pytest

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
    monkeypatch.setattr(s, "claim_job_for_fire", lambda _job_id, **_kwargs: True)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_tick_skips_job_when_durable_fire_claim_is_lost(monkeypatch):
    """A manual/external fire that wins the shared CAS must exclude ticker."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "claim_job_for_fire", lambda _job_id: False)

    assert s.tick(verbose=False, sync=True) == 0
    assert calls == []


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_run_one_job_exception_delivers_failure_alert(monkeypatch):
    """An exception escaping the run body must not become a silent error row."""
    delivered = []
    marked = []
    finished = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j3"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            RuntimeError("Gemini HTTP 503 (UNAVAILABLE)")
        ),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    ok = s.run_one_job({"id": "j3", "name": "morning", "deliver": "telegram"})

    assert ok is False
    assert delivered == [
        ("j3", "⚠️ Cron 'morning' failed: Gemini HTTP 503 (UNAVAILABLE)")
    ]
    assert marked == [
        (("j3", False, "Gemini HTTP 503 (UNAVAILABLE)"), {"delivery_error": None})
    ]
    assert finished == [
        (
            ("exec-j3",),
            {
                "success": False,
                "error": "Gemini HTTP 503 (UNAVAILABLE)",
                "delivery_outcome": "delivered",
            },
        )
    ]


def test_run_one_job_exception_records_failure_alert_delivery_error(monkeypatch):
    """A failed fallback alert must populate last_delivery_error."""
    marked = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j4"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    monkeypatch.setattr(s, "_deliver_result", lambda *_a, **_kw: "send failed: 502")
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)

    assert s.run_one_job({"id": "j4", "deliver": "telegram"}) is False
    assert marked == [
        (("j4", False, "provider failed"), {"delivery_error": "send failed: 502"})
    ]


def _patch_escaped_failure(monkeypatch, delivered, *, exec_id, err):
    """Make run_job raise, and capture what the escape handler delivers."""
    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": exec_id})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError(err)),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append(content) or None,
    )
    monkeypatch.setattr(s, "mark_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)
    # Deterministic threshold: default 3, independent of the host config.
    monkeypatch.setattr(s, "load_config", lambda: {})


def test_escaped_failure_delivery_carries_the_streak_nudge(monkeypatch):
    """A repeatedly-failing job must be nudged even when it fails at the
    scheduler layer (#88655).

    ``mark_job_run`` increments ``failure_streak`` for an escaped failure just
    as it does for an agent failure, so the counter climbs either way. But the
    nudge that spends it was only composed on the normal delivery path, so a
    job that raises before the run body on every tick - a bad import from a
    half-applied update, a provider client that cannot construct - alerts
    forever and is never told it should be reviewed or paused. Nothing else
    surfaces the streak in chat.
    """
    delivered = []
    _patch_escaped_failure(
        monkeypatch, delivered, exec_id="exec-j5", err="cannot import name X"
    )

    ok = s.run_one_job(
        {
            "id": "j5",
            "name": "scout",
            "deliver": "telegram",
            "schedule": {"kind": "interval"},
            "failure_streak": 2,  # + this run = 3 = default threshold
        }
    )

    assert ok is False
    assert len(delivered) == 1
    assert "cannot import name X" in delivered[0]
    assert "failed 3 runs in a row" in delivered[0]
    assert "hermes cron pause scout" in delivered[0]


def test_escaped_failure_delivery_stays_quiet_below_the_threshold(monkeypatch):
    """The nudge is appended, not always-on: a first failure reads as before."""
    delivered = []
    _patch_escaped_failure(
        monkeypatch, delivered, exec_id="exec-j6", err="provider failed"
    )

    ok = s.run_one_job(
        {
            "id": "j6",
            "name": "scout",
            "deliver": "telegram",
            "schedule": {"kind": "interval"},
            "failure_streak": 0,
        }
    )

    assert ok is False
    assert delivered == ["⚠️ Cron 'scout' failed: provider failed"]


def test_run_one_job_exception_after_delivery_does_not_redeliver(monkeypatch):
    """Once delivery has been attempted, the outer handler must not send again."""
    delivered = []
    mark_calls = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j5"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )

    def fake_mark(*args, **kwargs):
        mark_calls.append((args, kwargs))
        if len(mark_calls) == 1:
            raise RuntimeError("bookkeeping boom")

    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)

    ok = s.run_one_job({"id": "j5", "name": "once", "deliver": "telegram"})

    assert ok is False
    assert delivered == [("j5", "final response")]
    assert mark_calls[0] == (("j5", True, None), {"delivery_error": None})
    assert mark_calls[1] == (
        ("j5", False, "bookkeeping boom"),
        {"delivery_error": None},
    )


def test_run_one_job_keyboard_interrupt_skips_delivery_and_reraises(monkeypatch):
    """Hard interrupts must not attempt failure delivery; they re-raise."""
    delivered = []
    marked = []
    finished = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j6"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        s.run_one_job({"id": "j6", "name": "interrupt", "deliver": "telegram"})

    assert delivered == []
    assert marked == [(("j6", False, "KeyboardInterrupt"), {})]
    assert finished == [
        (
            ("exec-j6",),
            {
                "success": False,
                "error": "KeyboardInterrupt",
                "delivery_outcome": "suppressed",
            },
        )
    ]


def test_run_one_job_installs_secret_scope_under_multiplex(monkeypatch, tmp_path):
    """Regression: under profile isolation (multiplex active), run_one_job must
    keep one profile secret scope active through execution and delivery so
    credential reads do not fail closed or fall through to another profile,
    then tear the scope down after the complete job lifecycle.

    Behavior contract: the same scope is present during run_job and
    _deliver_result, and no scope remains after run_one_job returns.
    """
    from agent import secret_scope as ss

    # Point cron's home resolution at a profile whose .env carries a secret.
    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    scope_during_run = {}
    scope_during_delivery = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    def fake_deliver(*args, **kwargs):
        scope_during_delivery["scope"] = ss.current_secret_scope()
        scope_during_delivery["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return None

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    ss.set_multiplex_active(True)
    try:
        ok = s.run_one_job({"id": "j7", "name": "t"})
    finally:
        ss.set_multiplex_active(False)

    assert ok is True
    # The same profile scope covered both execution and delivery.
    assert scope_during_run["scope"] is not None
    assert scope_during_run["base_url"] == "https://openrouter.ai/api/v1"
    assert scope_during_delivery["scope"] == scope_during_run["scope"]
    assert scope_during_delivery["base_url"] == "https://openrouter.ai/api/v1"
    # And it was torn down after the full lifecycle returned (no leak).
    assert ss.current_secret_scope() is None


