"""The tick-to-worker handoff must have no retirement-sized gap after can_dispatch."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading


def test_passed_gate_tick_and_queued_job_remain_busy_until_real_worker_exit(tmp_path, monkeypatch):
    from cron import jobs, scheduler
    from hermes_cli import backend_retirement

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    script = tmp_path / "job.py"
    script.write_text("print('no model')\n")
    job = jobs.create_job(prompt=None, schedule=datetime.now(timezone.utc).isoformat(), script=str(script), no_agent=True)
    for name in ("_maybe_run_worktree_maintenance", "_sweep_mcp_orphans", "_maybe_reap_dead_owners"):
        monkeypatch.setattr(scheduler, name, lambda: None)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    reached, release_tick, running, release_job = (threading.Event() for _ in range(4))
    due = scheduler.get_due_jobs
    gate_calls = []

    def get_due_jobs():
        reached.set()
        assert release_tick.wait(10)
        return due()

    def process_job(*args):
        running.set()
        assert release_job.wait(10)
        return True

    monkeypatch.setattr(scheduler, "get_due_jobs", get_due_jobs)
    monkeypatch.setattr(scheduler, "_process_due_job", process_job)
    with ThreadPoolExecutor(max_workers=1) as pool, ThreadPoolExecutor(max_workers=1) as ticker:
        monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda *args: pool)
        tick = ticker.submit(scheduler.tick, verbose=False, sync=False, can_dispatch=lambda: gate_calls.append(True) or True)
        try:
            assert reached.wait(10)
            assert gate_calls
            assert not scheduler.get_running_job_ids()
            assert fence.prepare() == {"ok": False, "idle": False}
            release_tick.set()
            assert tick.result(10) == 1
            assert running.wait(10)
            assert job["id"] in scheduler.get_running_job_ids()
            assert fence.prepare() == {"ok": False, "idle": False}
        finally:
            release_tick.set()
            release_job.set()
    assert not scheduler.get_running_job_ids()
    token = fence.prepare()["token"]
    assert scheduler.tick(verbose=False, sync=False) == 0
    assert scheduler.try_register_running_job("manual-late") is False
    assert fence.commit(token) == {"ok": True}
    assert scheduler.tick(verbose=False, sync=False) == 0


def test_detached_cron_delivery_keeps_admission_after_its_tick_returns(tmp_path, monkeypatch):
    from cron import bot_chat_delivery
    from hermes_cli import backend_retirement
    from hermes_constants import get_hermes_home

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    root = get_hermes_home() / "cron" / "bot_chat_pending"
    root.mkdir(parents=True)
    entered, release = threading.Event(), threading.Event()

    def drain(root):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(bot_chat_delivery, "_drain", drain)
    try:
        bot_chat_delivery.drain_in_background()
        assert entered.wait(10)
        assert fence.prepare() == {"ok": False, "idle": False}
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "cron-bot-chat-drain":
                thread.join(10)
    token = fence.prepare()["token"]
    entered.clear()
    bot_chat_delivery.drain_in_background()
    bot_chat_delivery.drain(root)
    assert not entered.is_set()
    assert fence.cancel(token) == {"ok": True}
