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

    def fake_run_job(job, *, verbose=False):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(
        jid,
        ok,
        err=None,
        delivery_error=None,
        *,
        expected_fire_claim_id=None,
    ):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "_run_job_in_killable_process", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "claim_job_for_fire_token", lambda jid: f"claim-{jid}")
    monkeypatch.setattr(s, "heartbeat_fire_claim", lambda *a, **k: True)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "claim_job_for_fire_token", lambda jid: f"claim-{jid}")
    monkeypatch.setattr(s, "heartbeat_fire_claim", lambda *a, **k: True)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


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

    def fake_run_job(job, *, verbose=False):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "_run_job_in_killable_process", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(s, "claim_job_for_fire_token", lambda jid: f"claim-{jid}")
    monkeypatch.setattr(s, "heartbeat_fire_claim", lambda *a, **k: True)

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


def test_tokenless_direct_call_must_acquire_fire_ownership(monkeypatch):
    """The shared callable must not expose an unclaimed execution bypass."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "claim_job_for_fire_token", lambda _jid: None)
    finished = []
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    ok = s.run_one_job({"id": "unowned", "name": "t"})

    assert ok is False
    assert calls == []
    assert finished[-1][1]["success"] is False
    assert "ownership" in finished[-1][1]["error"].lower()


def test_dispatch_rejection_releases_owned_fire_claim(monkeypatch):
    """A terminal one-shot cleanup cannot leave its universal claim live."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: False)
    released = []
    monkeypatch.setattr(
        s,
        "release_fire_claim",
        lambda job_id, *, expected_claim_id: released.append(
            (job_id, expected_claim_id)
        ),
    )
    monkeypatch.setattr(s, "finish_execution", lambda *a, **k: None)

    ok = s.run_one_job(
        {
            "id": "dispatch-complete",
            "name": "t",
            "execution_id": "exec-dispatch-complete",
            "_fire_claim_id": "owned-token",
        }
    )

    assert ok is True
    assert calls == []
    assert released == [("dispatch-complete", "owned-token")]


def test_fire_claim_storage_failure_finishes_ledger_without_running(monkeypatch):
    """Claim acquisition errors cannot strand a running ledger entry."""
    calls = _patch_pipeline(monkeypatch)

    def fail_claim(_job_id):
        raise RuntimeError("runtime store unavailable")

    monkeypatch.setattr(s, "claim_job_for_fire_token", fail_claim)
    finished = []
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    ok = s.run_one_job(
        {"id": "claim-error", "name": "t", "execution_id": "exec-claim-error"}
    )

    assert ok is False
    assert calls == []
    assert finished[-1][0] == "exec-claim-error"
    assert finished[-1][1]["success"] is False
    assert "runtime store unavailable" in finished[-1][1]["error"]


def test_claim_lost_after_run_suppresses_output_delivery_and_success(monkeypatch):
    """A stale worker must have no post-run side effects or successful ledger."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "heartbeat_fire_claim", lambda *a, **k: False)
    finished = []
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    ok = s.run_one_job(
        {
            "id": "lost-owner",
            "name": "t",
            "execution_id": "exec-lost",
            "_fire_claim_id": "old-token",
        }
    )

    assert ok is False
    assert calls == [("run_job", "lost-owner")]
    assert finished == [
        (
            "exec-lost",
            {
                "success": False,
                "error": "Fire-claim ownership was lost before completion.",
                "delivery_outcome": "suppressed",
            },
        )
    ]


def test_claim_lost_before_delivery_suppresses_external_side_effect(monkeypatch):
    """A token lost after output save must not deliver a stale response."""
    calls = _patch_pipeline(monkeypatch)
    heartbeats = iter([True, False])
    monkeypatch.setattr(
        s,
        "heartbeat_fire_claim",
        lambda *a, **k: next(heartbeats),
    )
    finished = []
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    ok = s.run_one_job(
        {
            "id": "lost-before-delivery",
            "name": "t",
            "execution_id": "exec-before-delivery",
            "_fire_claim_id": "old-token",
        }
    )

    assert ok is False
    assert [call[0] for call in calls] == ["run_job", "save"]
    assert finished[-1][1]["delivery_outcome"] == "suppressed"


def test_finalization_claim_loss_overrides_successful_execution_ledger(monkeypatch):
    """A CAS loss at mark time must not leave the execution ledger successful."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "heartbeat_fire_claim", lambda *a, **k: True)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: False)
    finished = []
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )

    ok = s.run_one_job(
        {
            "id": "stale-finalizer",
            "name": "t",
            "execution_id": "exec-stale",
            "_fire_claim_id": "old-token",
        }
    )

    assert ok is False
    assert [call[0] for call in calls] == ["run_job", "save", "deliver"]
    assert finished[-1][1]["success"] is False
    assert "ownership" in finished[-1][1]["error"].lower()


