"""The canonical delivery queue row is the cron run's send: it obeys the same fire-claim fence,
receipt and settlement rules as a direct send."""
from cron import delivery_queue, executions, jobs, scheduler


def test_claim_stolen_after_the_ownership_sample_queues_no_ghost_send(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="telegram:fixture")
    fire = jobs.claim_job_for_fire(job["id"], return_job=True, force=True)
    samples = []
    real_lost = scheduler._FireOwnership.lost

    def stolen_after_second_sample(self):
        verdict = real_lost(self)
        samples.append(verdict)
        if len(samples) == 2:  # _save_compose_deliver's pre-send sample: a replacement wins now
            jobs.update_job(job["id"], {"fire_claim": dict(fire["fire_claim"], by="replacement")})
        return verdict

    monkeypatch.setattr(scheduler._FireOwnership, "lost", stolen_after_second_sample)
    scheduler.run_one_job(fire)
    execution = executions.latest_execution(job["id"])
    assert samples[:2] == [False, False]
    assert delivery_queue.get_status(execution["id"]) is None
    assert execution["status"] == "failed"


def test_dead_drain_owner_settles_the_run_it_fenced_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="telegram:fixture")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])
    assert jobs.get_job(job["id"])["last_status"] == "delivery_queued"
    claimed = delivery_queue.claim_next()
    delivery_queue._ACTIVE_DELIVERIES.discard(claimed["execution_id"])
    with delivery_queue._transaction() as conn:  # the claiming gateway died mid-send
        conn.execute("UPDATE deliveries SET owner_process_id='gone', owner_pid=999999 "
                     "WHERE execution_id=?", (execution["id"],))
    sent = []
    delivery_queue.drain(lambda *args: sent.append(args))
    saved = jobs.get_job(job["id"])
    assert sent == [], "an uncertain send is never retried"
    assert saved["last_status"] == "delivery_failed" and "unknown" in saved["last_delivery_error"]
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "failed"


def test_origin_with_no_resolvable_target_is_not_configured_never_delivered(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="origin")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])
    assert delivery_queue.get_status(execution["id"]) is None, "no target: nothing is queued"
    assert execution["delivery_outcome"] == "not_configured"
    assert jobs.get_job(job["id"])["last_status"] == "ok"
