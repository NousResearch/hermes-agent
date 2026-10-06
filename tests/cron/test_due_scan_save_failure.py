"""The due scan must still hand back its due jobs when the store cannot be written.

``_get_due_jobs_locked`` repairs store-side problems mid-scan (a half-paused record self-disables,
completed one-shots are swept, records are normalized) and then persists those repairs with a
``save_jobs(...)`` on the way out. That save used to be unguarded, so any store-write failure —
ENOSPC on a full disk, a read-only mount, a permissions problem — raised out of ``get_due_jobs()``
and aborted the tick, and every job on that profile stopped firing until a write succeeded.
Observed live on a full /opt/data: repeated ``Cron tick error ... [Errno 28] No space left on
device`` from this exact call site, with every job then reporting "missed its scheduled time".

The repairs are already applied in memory — that is what the scan keys off — so the persist is a
side effect the next tick can retry. The scan still returns its due jobs; tick() skips recurring
dispatch (no durable advance) without raising or mutating the store.
"""

import errno
import json
import asyncio
import logging
import os
import sqlite3
from datetime import timedelta

import pytest
from datetime import datetime, timezone

from cron import jobs as cronjobs
from cron import store_health
from cron.jobs import get_due_jobs, load_jobs, save_jobs

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir, pin the clock, start with no degraded store."""
    monkeypatch.setattr(store_health, "_degraded", {})
    monkeypatch.setattr(store_health, "_listener", None)
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _due_job(jid="due-job"):
    """A recurring job whose next run is the pinned instant — on the schedule's grid, so the scan
    dispatches it rather than re-anchoring it."""
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "cron", "expr": "* * * * *", "display": "every minute"},
        "next_run_at": FIXED_NOW.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "active",
        "repeat": None,
        "deliver": "local",
    }


def _half_paused_job(jid="half-paused"):
    """enabled=true with pause markers: the scan repairs it in place and sets needs_save, which is
    what makes the scan persist on its way out — the call that used to abort the tick."""
    job = _due_job(jid)
    job["paused_at"] = FIXED_NOW.isoformat()
    job["paused_reason"] = "test"
    return job


def _enospc(*_args, **_kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.fixture()
def full_disk(monkeypatch):
    """Make the store unwritable the way a full disk does, without ever touching the real store."""
    monkeypatch.setattr(cronjobs, "save_jobs", _enospc)


def test_due_jobs_are_returned_when_the_store_cannot_be_saved(cron_store, full_disk, monkeypatch, caplog):
    save_jobs([_due_job(), _half_paused_job()])

    with caplog.at_level(logging.WARNING, logger="cron.jobs"):
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]
        assert [d["id"] for d in get_due_jobs()] == ["due-job"]

        with cronjobs.use_cron_store(cron_store / "other-profile"):  # a second profile's store
            cronjobs.warn_store_unwritable(OSError(errno.ENOSPC, "No space left on device"), "x", "scan")

    # One WARNING per outage per store (its degraded state), not one per 60s scan, and a
    # sibling profile on the same errno is not silenced; each names its store.
    warnings = [r.getMessage() for r in caplog.records if r.name == "cron.jobs" and r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "due-scan repairs not persisted" in warnings[0]
    assert str(cron_store / "cron") in warnings[0] and str(cron_store / "other-profile") in warnings[1]


@pytest.mark.parametrize("with_once,junk", [(True, False), (False, False), (False, True)],
                         ids=["one-shot-due", "recurring-only", "load-repair"])
def test_tick_on_unwritable_store_returns_cleanly_without_dispatch(cron_store, monkeypatch, caplog, with_once, junk):
    """Real tick(): the advance cannot be persisted, so the recurring job is NOT run (at-most-once);
    a due one-shot still reaches its fire claim, which fails closed with a ``failed`` execution row.
    The tick neither raises nor alters the store, still reaps MCP orphans (also when every due job
    was skipped), and a WARNING names the unwritable store."""
    from cron import executions, scheduler

    once = dict(_due_job("once"), schedule={"kind": "once", "run_at": FIXED_NOW.isoformat(), "display": "once"},
                repeat={"times": 1, "completed": 0})
    save_jobs([_due_job(), _half_paused_job()] + ([once] if with_once else []))
    before = load_jobs()
    if junk:  # load_jobs() drops a non-object entry and persists that repair: the save must not abort the tick
        raw = json.loads(cronjobs.JOBS_FILE.read_text())
        cronjobs.JOBS_FILE.write_text(json.dumps(dict(raw, jobs=raw["jobs"] + [42])))
    ran, sweeps = [], []
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_store / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda job, **k: ran.append(job["id"]) or True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: sweeps.append(1))
    monkeypatch.setattr(cronjobs, "_stage_jobs_payload", _enospc)
    with caplog.at_level(logging.INFO):
        assert scheduler.tick(verbose=False, sync=True) == 0

    assert ran == [] and sweeps == [1]
    assert f"Cron store {cron_store / 'cron'} is unwritable" in caplog.text
    assert "skipped 1 recurring job(s)" in caplog.text  # not silenced by the scan's earlier warning
    assert store_health.degraded_record(cron_store / "cron").skipped_runs == 1 + with_once
    assert executions.latest_execution("due-job") is None  # skipped before its fire claim
    assert load_jobs() == before
    row = executions.latest_execution("once")
    if with_once:
        assert row["status"] == "failed" and "Cron store unwritable" in row["error"]
    else:
        assert row is None


@pytest.mark.parametrize("site,exc,ledger_fails", [
    ("claim_job_for_fire", RuntimeError, False),
    ("note_cron_execution", RuntimeError, False),
    # Full disk: the executions.db write fails too; the store-unwritable skip must not raise.
    ("claim_job_for_fire", OSError, True),
])
def test_dispatch_failure_after_receipt_never_leaves_it_claimed(
        cron_store, monkeypatch, caplog, site, exc, ledger_fails):
    """A non-OSError fire-claim failure, or a failure after create_execution in _submit_with_guard,
    must settle the receipt: a ``claimed`` row never resolves and reads forever as in flight."""
    from cron import executions, scheduler

    def boom(*_a, **_k):
        raise exc("boom")

    def ledger_full(*_a, **_k):
        raise sqlite3.OperationalError("database or disk is full")

    save_jobs([_due_job()])
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_store / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda job, **k: True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler, site, boom)
    if ledger_fails:
        monkeypatch.setattr(executions, "finish_execution", ledger_full)
        monkeypatch.setattr(scheduler, "finish_execution", ledger_full)
    scheduler.tick(verbose=False, sync=True)  # a worker failure is logged, never raised
    if ledger_fails:  # the skip settles best-effort instead of raising out of the worker
        assert "failed to close execution receipt" in caplog.text
        assert "Cron job future failed" not in caplog.text
        return
    row = executions.latest_execution("due-job")
    assert row is not None and row["status"] == "failed"
    if site == "note_cron_execution":  # the receipt exists, so creation did not fail
        assert "dispatch preparation failed" in caplog.text


def test_unwritable_store_degrades_once_throttles_and_catches_up_once(cron_store, monkeypatch):
    """One degraded state per outage: entered once, advance/claim skipped while the scan's save
    keeps failing (one write attempt per tick), cleared by the first save that lands; then the recurring job and the one-shot that
    stayed due across 5 ticks each fire ONCE, and the one-shot keeps a single ``failed`` row."""
    from cron import executions, scheduler

    clock = {"now": FIXED_NOW, "mono": 0.0}
    monkeypatch.setattr(cronjobs, "_hermes_now", lambda: clock["now"])
    monkeypatch.setattr(store_health.time, "time", lambda: clock["now"].timestamp())
    monkeypatch.setattr(store_health.time, "monotonic", lambda: clock["mono"])
    once = dict(_due_job("once"), schedule={"kind": "once", "run_at": FIXED_NOW.isoformat(), "display": "once"},
                repeat={"times": 1, "completed": 0})
    save_jobs([_due_job(), _half_paused_job(), once])
    ran, events, probes, writes = [], [], [], []
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", cron_store / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda job, **k: ran.append(job["id"]) or True)
    monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)
    store_health.set_transition_listener(lambda event, record: events.append((event, record.skipped_runs)))
    real_stage = cronjobs._stage_jobs_payload

    def stage(*a, **k):
        writes.append(1)
        return _enospc() if outage else real_stage(*a, **k)

    def probe(_cron_dir):
        probes.append(1)
        return OSError(errno.ENOSPC, "No space left on device") if outage else None

    monkeypatch.setattr(cronjobs, "_stage_jobs_payload", stage)
    monkeypatch.setattr(store_health, "probe_store", probe)
    outage = True
    per_tick = []
    for minute, mono in enumerate((0, 30, 60, 90, 120)):  # 5 due ticks, re-probe at most every 60s
        clock["now"], clock["mono"] = FIXED_NOW + timedelta(minutes=minute), float(mono)
        before = len(writes)
        assert scheduler.tick(verbose=False, sync=True) == 0
        per_tick.append(len(writes) - before)
    # The scan's failing save after entry is each tick's one write attempt: no extra re-probe.
    assert ran == [] and len(probes) == 0
    assert per_tick[0] == 3 and per_tick[1:] == [1, 1, 1, 1]
    assert events == [("unwritable", 0)]
    # Distinct (job, due instant): the unpersisted fast-forward keeps both on one instant each.
    assert store_health.degraded_record(cron_store / "cron").skipped_runs == 2
    outage = False
    clock["now"], clock["mono"] = FIXED_NOW + timedelta(minutes=5), 180.0
    assert scheduler.tick(verbose=False, sync=True) == 2
    assert sorted(ran) == ["due-job", "once"] and events[-1][0] == "recovered" and len(events) == 2
    assert store_health.degraded_record(cron_store / "cron") is None
    once_rows = executions.list_executions(job_id="once")
    assert [r["status"] for r in once_rows].count("failed") == 1


@pytest.mark.platforms("posix")  # POSIX mode bits; root ignores them
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses directory modes")
def test_unwritable_store_is_shown_in_cron_status_and_announced_once(cron_store, monkeypatch, capsys, caplog):
    """`hermes cron status` probes the store itself (a real 0500 dir) and leads with the red
    headline + fix; the gateway posts ONE home-channel notice on entry and ONE on recovery,
    each naming the store, the error, since-when and the skipped runs; re-entry within the hour
    posts nothing."""
    from types import SimpleNamespace
    from gateway.cron_store_notices import install_cron_store_notices
    from hermes_cli import cron as cron_cli

    save_jobs([_due_job()])
    cron_dir = cron_store / "cron"
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "chronos")
    os.chmod(cron_dir, 0o500)
    try:
        cron_cli.cron_status()
    finally:
        os.chmod(cron_dir, 0o700)
    out = capsys.readouterr().out
    assert out.lstrip().startswith("⚠ Cron store is NOT writable — scheduled jobs are being skipped")
    assert f"{cron_dir}: EACCES" in out and "1 due run(s) not fired" in out
    assert f"fix permissions on {cron_dir}" in out

    sent = []

    async def send(platform, home, transport, message, failure_fmt):
        sent.append(message)
        return True

    home = SimpleNamespace(chat_id="c1", thread_id=None)
    runner = SimpleNamespace(_send_home_channel_message=send, _served_home_channel_transports=lambda: iter(
        [(None, "telegram", None, home, object())]))
    monkeypatch.setattr("hermes_constants.get_routing_process_hermes_home", lambda: cron_store)
    loop = asyncio.new_event_loop()
    try:
        install_cron_store_notices(runner, loop)
        enospc = OSError(errno.ENOSPC, "No space left on device")
        for site in ("scan", "advance", "claim"):  # one outage, many failing sites
            cronjobs.warn_store_unwritable(enospc, "x", site, [_due_job()])
        save_jobs([_due_job()])
        save_jobs([_due_job()])
        cronjobs.warn_store_unwritable(enospc, "x", "scan", [_due_job()])  # flaps back within the hour
        loop.run_until_complete(asyncio.sleep(0.05))
        install_cron_store_notices(SimpleNamespace(), loop)  # a send that raises is logged, not lost
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()
    assert len(sent) == 2
    assert str(cron_dir) in sent[0] and "ENOSPC: No space left on device" in sent[0] and "since " in sent[0]
    assert f"fix permissions on {cron_dir}" in sent[0]
    assert "writable again; 1 skipped run(s), catching up once per job" in sent[1]
    assert any(r.levelname == "WARNING" and "unwritable notice for" in r.getMessage() for r in caplog.records)
