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

    def fake_run_job(job, *, defer_agent_teardown=None):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None):
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
    def boom(job, *, defer_agent_teardown=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(s, "run_job", boom)
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: marks.append((jid, ok)),
    )

    ok = s.run_one_job({"id": "j6", "name": "t"})

    assert ok is False
    assert marks == [("j6", False)]


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

    def fake_run_job(job, *, defer_agent_teardown=None):
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


def test_run_one_job_delivers_before_agent_teardown(monkeypatch):
    """Regression for #58720: the cron agent's async-resource teardown
    (agent.close + cleanup_stale_async_clients) MUST run AFTER delivery, not
    before. run_job defers teardown by appending the live agent to the holder
    list; run_one_job tears it down only after _deliver_result has run. If the
    order flips, delivery races a torn-down async client and dies with
    'cannot schedule new futures after interpreter shutdown'.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(job, *, defer_agent_teardown=None):
        order.append("run_job")
        # Mimic run_job's deferral contract: hand the live agent back so the
        # caller tears it down after delivery instead of in run_job's finally.
        assert defer_agent_teardown is not None, "run_one_job must defer teardown"
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def fake_deliver(job, content, adapters=None, loop=None):
        order.append("deliver")
        return None

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    # cleanup_stale_async_clients is imported lazily inside _teardown_cron_agent;
    # stub it so the teardown records its own marker without touching real caches.
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j8", "name": "t"})

    assert ok is True
    # Delivery must strictly precede agent teardown + stale-client reap.
    assert order == ["run_job", "deliver", "agent.close", "cleanup_stale"], order


def test_run_one_job_tears_down_deferred_agent_when_delivery_raises(monkeypatch):
    """Even if _deliver_result raises, the deferred agent is still torn down
    (no fd/client leak — #10200). Teardown lives in a finally around delivery.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(job, *, defer_agent_teardown=None):
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def boom_deliver(job, content, adapters=None, loop=None):
        order.append("deliver-raise")
        raise RuntimeError("send blew up")

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", boom_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j9", "name": "t"})

    assert ok is True  # delivery error is recorded, not propagated
    assert order == ["deliver-raise", "agent.close", "cleanup_stale"], order


def test_run_one_job_tears_down_deferred_agent_when_save_raises(monkeypatch):
    """#58720 W1: if save_job_output (or the [SILENT]/empty computation) raises
    AFTER run_job hands the agent back but BEFORE delivery, the deferred agent
    must still be torn down. The outer `except` would otherwise swallow the
    error and leak the agent (#10200). Teardown lives in a finally spanning
    save→deliver.
    """
    order = []

    class FakeAgent:
        def close(self):
            order.append("agent.close")

    def fake_run_job(job, *, defer_agent_teardown=None):
        defer_agent_teardown.append(FakeAgent())
        return (True, "out", "final response", None)

    def boom_save(jid, out):
        order.append("save-raise")
        raise RuntimeError("disk full")

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", boom_save)
    monkeypatch.setattr(s, "_deliver_result",
                        lambda *a, **k: order.append("deliver"))
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "cleanup_stale_async_clients",
                        lambda: order.append("cleanup_stale"))

    ok = s.run_one_job({"id": "j10", "name": "t"})

    # save raised → outer handler marks failure and returns False, but the
    # deferred agent was still torn down (no delivery, no leak).
    assert ok is False
    assert "deliver" not in order
    assert order == ["save-raise", "agent.close", "cleanup_stale"], order


def _patch_loop_pipeline(monkeypatch, job, final_response, deliver):
    """Install a stateful pipeline around the real loop evaluator."""
    monkeypatch.setattr(
        s,
        "run_job",
        lambda _job, *, defer_agent_teardown=None: (
            True,
            final_response,
            final_response,
            None,
        ),
    )
    monkeypatch.setattr(s, "save_job_output", lambda *_args: "/tmp/loop.txt")
    monkeypatch.setattr(s, "_deliver_result", deliver)
    monkeypatch.setattr(
        s,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "test", "chat_id": "1"}],
    )
    monkeypatch.setattr(s, "mark_job_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(s, "_run_loop_verify", lambda _job: None)
    monkeypatch.setattr(s, "_notify_provider_jobs_changed", lambda: None)

    import cron.jobs as jobs
    import hermes_cli.goals as goals

    def update_job(_job_id, updates):
        job.update(updates)
        return job

    monkeypatch.setattr(jobs, "update_job", update_job)
    monkeypatch.setattr(jobs, "pause_job", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_args, **_kwargs: ("continue", "still running", False, None),
    )


def test_loop_acknowledges_only_after_successful_delivery(monkeypatch):
    job = {
        "id": "loop-delivery",
        "name": "loop",
        "prompt": "check status",
        "loop": True,
        "loop_no_progress_threshold": 10,
        "loop_no_progress_count": 0,
    }
    delivery_results = iter(["send failed", None])
    deliveries = []

    def deliver(_job, content, adapters=None, loop=None):
        deliveries.append(content)
        return next(delivery_results)

    _patch_loop_pipeline(monkeypatch, job, "same result", deliver)

    assert s.run_one_job(job) is True
    assert job.get("loop_last_delivered_hash") is None

    assert s.run_one_job(job) is True
    assert deliveries == ["same result", "same result"]
    assert job["loop_last_delivered_hash"] == s._loop_response_hash("same result")


def test_loop_silent_response_is_never_acknowledged(monkeypatch):
    job = {
        "id": "loop-silent",
        "name": "loop",
        "prompt": "check quietly",
        "loop": True,
        "loop_no_progress_threshold": 10,
        "loop_no_progress_count": 0,
    }

    def unexpected_delivery(*_args, **_kwargs):
        raise AssertionError("[SILENT] loop responses must not be delivered")

    _patch_loop_pipeline(monkeypatch, job, "[SILENT]", unexpected_delivery)

    assert s.run_one_job(job) is True
    assert job.get("loop_last_delivered_hash") is None


def test_loop_skips_output_only_after_prior_successful_delivery(monkeypatch):
    response = "unchanged result"
    job = {
        "id": "loop-duplicate",
        "name": "loop",
        "prompt": "check status",
        "loop": True,
        "loop_no_progress_threshold": 10,
        "loop_no_progress_count": 0,
        "loop_last_output_hash": s._loop_response_hash(response),
        "loop_last_delivered_hash": s._loop_response_hash(response),
    }
    deliveries = []

    def deliver(_job, content, adapters=None, loop=None):
        deliveries.append(content)
        return None

    _patch_loop_pipeline(monkeypatch, job, response, deliver)

    assert s.run_one_job(job) is True
    assert deliveries == []


def test_loop_without_delivery_target_is_not_acknowledged(monkeypatch):
    job = {
        "id": "loop-local",
        "name": "loop",
        "prompt": "check locally",
        "loop": True,
        "loop_no_progress_threshold": 10,
        "loop_no_progress_count": 0,
        "deliver": "local",
    }

    _patch_loop_pipeline(monkeypatch, job, "local result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(s, "_resolve_delivery_targets", lambda _job: [])

    assert s.run_one_job(job) is True
    assert job.get("loop_last_delivered_hash") is None


def test_loop_pause_alert_is_delivered(monkeypatch):
    response = "unchanged result"
    job = {
        "id": "loop-alert",
        "name": "deploy-watch",
        "prompt": "check deploy status",
        "loop": True,
        "loop_no_progress_threshold": 1,
        "loop_no_progress_count": 0,
        "loop_last_output_hash": s._loop_response_hash(response),
        "repeat": {"completed": 2},
    }
    deliveries = []

    def deliver(_job, content, adapters=None, loop=None):
        deliveries.append(content)
        return None

    _patch_loop_pipeline(monkeypatch, job, response, deliver)

    assert s.run_one_job(job) is True
    assert len(deliveries) == 1
    assert "auto-paused" in deliveries[0]
    assert "Runs completed: 3" in deliveries[0]
    assert "/loop resume loop-alert" in deliveries[0]
