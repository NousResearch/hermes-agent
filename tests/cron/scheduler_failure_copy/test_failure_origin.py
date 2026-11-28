"""Cron failure-origin: the executor stamps WHICH stage failed on the working job copy, and the
delivered notice + incident class are chosen from that origin instead of guessing at error text.

Known source stages are ``monitor_source`` (a monitor_script / monitor_url source),
``script_execution`` (a pre-run / wake-gate script, or a no_agent script) and ``agent_provider``
(agent/provider setup or execution — NOT a claim that a model necessarily received a
request). The origin is a bounded per-run stamp on the job's WORKING copy: cleared at run entry,
absent from successful runs and from the persisted job definition, and carried through the existing
completion path so committed receipts and recovery classify with the same origin.

Source/script failures render content-free notices (what to inspect, never the source's
stdout/stderr, an authenticated URL or arbitrary error text). Legacy unknown-origin, provider
auth/rate-limit/503, script-timeout and inactivity semantics are unchanged. All data is synthetic
and every path is environment-derived (tmp_path / monkeypatched homes); nothing here reaches a
model, gateway or the network.
"""

from cron import incidents, jobs, scheduler
from cron.monitor import MonitorOutcome


def _no_chain(monkeypatch):
    """No fallback chain: delivery must not read the real config or crash on its absence."""
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])


def _point_incidents(monkeypatch, tmp_path):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


# --- Delivery copy is chosen from the origin, not the error text -------------------------------

def test_monitor_source_failure_notice_is_content_free(monkeypatch):
    _no_chain(monkeypatch)
    job = {"name": "Watchdog", "id": "a1b2c3d4", "_failure_origin": "monitor_source"}
    # A synthetic source error that embeds an authenticated URL and an HTTP 503.
    error = ("monitor_url fetch failed: GET https://monitor.example.invalid/feed?token=SYNTH_TOKEN "
             "returned HTTP 503 upstream unavailable")
    msg = scheduler._summarize_cron_failure_for_delivery(job, error)
    assert "monitored source" in msg.lower(), msg
    assert "hermes cron runs a1b2c3d4" in msg, msg
    for leaked in ("SYNTH_TOKEN", "monitor.example.invalid", "503", "unavailable",
                   "monitor_url fetch failed"):
        assert leaked not in msg, (leaked, msg)


def test_script_failure_notice_is_content_free(monkeypatch):
    _no_chain(monkeypatch)
    job = {"name": "Collector", "id": "b2c3d4e5", "_failure_origin": "script_execution"}
    error = ("Traceback (most recent call last):\n  File \"/opt/x/collect.py\", line 9\n"
             "RuntimeError: connection reset by peer SYNTH_SECRET_ABC")
    msg = scheduler._summarize_cron_failure_for_delivery(job, error)
    assert "script" in msg.lower(), msg
    assert "hermes cron runs b2c3d4e5" in msg, msg
    for leaked in ("SYNTH_SECRET_ABC", "collect.py", "connection reset", "Traceback"):
        assert leaked not in msg, (leaked, msg)


def test_no_agent_script_failure_notice_is_content_free(monkeypatch):
    _no_chain(monkeypatch)
    job = {"name": "Pinger", "id": "c3d4e5f6", "no_agent": True, "_failure_origin": "script_execution"}
    error = "ping: cannot resolve host.invalid SYNTH_HOSTNAME; exit status 2"
    msg = scheduler._summarize_cron_failure_for_delivery(job, error)
    assert "script" in msg.lower(), msg
    assert "hermes cron runs c3d4e5f6" in msg, msg
    for leaked in ("SYNTH_HOSTNAME", "host.invalid", "exit status", "cannot resolve"):
        assert leaked not in msg, (leaked, msg)


def test_arbitrary_source_wording_is_content_free(monkeypatch):
    _no_chain(monkeypatch)
    wordings = (
        "HTTP 503 service unavailable",
        "unexpected token <html> from SYNTH_ENDPOINT",
        "Error code: 502 - bad gateway SYNTH_GATEWAY",
        "segmentation fault (core dumped) SYNTH_PID 4242",
    )
    for origin in ("monitor_source", "script_execution"):
        for wording in wordings:
            job = {"name": "J", "id": "d4e5f6a7", "_failure_origin": origin}
            msg = scheduler._summarize_cron_failure_for_delivery(job, wording)
            assert "SYNTH_" not in msg, (wording, msg)
            assert "502" not in msg and "503" not in msg, (wording, msg)
            assert "hermes cron runs d4e5f6a7" in msg, msg


def test_healthy_source_then_provider_error_keeps_provider_notice(monkeypatch):
    _no_chain(monkeypatch)
    # The monitored source ran fine; the agent/provider stage is what failed.
    job = {"name": "Brief", "id": "e5f6a7b8", "_failure_origin": "agent_provider"}
    msg = scheduler._summarize_cron_failure_for_delivery(job, "Error code: 503 - upstream overloaded")
    assert "monitored source" not in msg.lower(), msg
    assert "did not run successfully" not in msg, msg
    assert "hermes cron runs e5f6a7b8" in msg, msg


def test_prerun_script_then_provider_error_keeps_provider_notice(monkeypatch):
    _no_chain(monkeypatch)
    # The wake-gate script succeeded; the agent/provider stage failed -> NOT a source notice.
    job = {"name": "Poller", "id": "f6a7b8c9", "script": "/opt/x.py", "_failure_origin": "agent_provider"}
    msg = scheduler._summarize_cron_failure_for_delivery(job, "Error code: 429 - rate limited")
    assert "did not run successfully" not in msg, msg
    assert "monitored source" not in msg.lower(), msg
    assert "hermes cron runs f6a7b8c9" in msg, msg
    assert "limit" in msg.lower(), msg  # rate-limit provider diagnosis preserved


def test_script_timeout_notice_is_preserved(monkeypatch):
    _no_chain(monkeypatch)
    job = {"name": "Slow", "id": "07b8c9d0", "_failure_origin": "script_execution"}
    msg = scheduler._summarize_cron_failure_for_delivery(job, "Script timed out after 30s: /opt/x/slow.py")
    assert "timed out" in msg.lower(), msg
    assert "did not run successfully" not in msg, msg  # the dedicated timeout copy, not the generic source copy


def test_inactivity_notice_is_preserved(monkeypatch):
    _no_chain(monkeypatch)
    job = {"name": "Stuck", "id": "18c9d0e1", "_failure_origin": "agent_provider"}
    error = "TimeoutError: Cron job 'Stuck' idle for 1239s (limit 600s) — last activity: terminal"
    msg = scheduler._summarize_cron_failure_for_delivery(job, error)
    assert "stalled" in msg.lower(), msg
    assert "did not run successfully" not in msg, msg


def test_legacy_unknown_origin_notice_unchanged(monkeypatch):
    _no_chain(monkeypatch)
    # No origin stamp (legacy / unknown): the classifier still interprets the text verbatim.
    job = {"name": "Legacy", "id": "29d0e1f2"}
    msg = scheduler._summarize_cron_failure_for_delivery(job, "some opaque failure SYNTH_DETAIL")
    assert "SYNTH_DETAIL" in msg, msg


# --- The executor stamps the failing stage on the working job copy -----------------------------

def test_apply_monitor_gate_stamps_monitor_source(monkeypatch):
    from cron import monitor as monitor_mod
    monkeypatch.setattr(
        monitor_mod, "check_monitor",
        lambda job: MonitorOutcome(ok=False, error="synthetic source failure SYNTH_URL"))
    job = {"name": "M", "id": "3ae1f2a3", "monitor_url": "https://monitor.example.invalid/x"}
    early, _extra, _ctx = scheduler._apply_monitor_gate(job, job["id"], "M", None)
    assert early is not None and early[0] is False
    assert job.get("_failure_origin") == "monitor_source"


def test_run_no_agent_job_stamps_script_origin(monkeypatch, tmp_path):
    from hermes_cli import env_loader
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", lambda **k: None)
    monkeypatch.setattr(
        scheduler, "_run_job_script_with_claim_heartbeat",
        lambda *a, **k: (False, "synthetic stderr SYNTH_SECRET_XYZ"))
    job = {"name": "NoAgent", "id": "4bf2a3b4", "no_agent": True, "script": "/opt/x/ping.sh"}
    success, _doc, _resp, error = scheduler._run_no_agent_job(job, job["id"], "NoAgent", None)
    assert success is False
    assert job.get("_failure_origin") == "script_execution"
    assert error == "synthetic stderr SYNTH_SECRET_XYZ"


# --- The stamp is bounded: cleared at entry, never persisted ------------------------------------

def test_run_job_clears_stale_origin_on_successful_reused_job(monkeypatch, tmp_path):
    from hermes_cli import env_loader
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", lambda **k: None)
    monkeypatch.setattr(
        scheduler, "_run_job_script_with_claim_heartbeat",
        lambda *a, **k: (True, "wakeAgent=true\nSYNTH_OK_OUTPUT"))
    # A reused working copy still carries the previous fire's failure origin.
    job = {"name": "Reuse", "id": "5ca3b4c5", "no_agent": True, "script": "/opt/x/ok.sh",
           "_failure_origin": "script_execution"}
    success, _doc, _response, error = scheduler.run_job(job)
    assert success is True and error is None
    assert "_failure_origin" not in job  # cleared at run entry; a healthy run leaves no stamp


def test_mark_job_run_never_persists_failure_origin(tmp_path):
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="p", schedule="every 1h", deliver="local")
        stored = jobs.load_jobs()
        stored[0]["_failure_origin"] = "monitor_source"
        jobs.save_jobs(stored)
        assert jobs.get_job(job["id"]).get("_failure_origin") == "monitor_source"
        jobs.mark_job_run(job["id"], success=False, error="synthetic failure SYNTH")
        assert "_failure_origin" not in jobs.get_job(job["id"])


def test_finish_completed_run_books_and_clears_persisted_origin(tmp_path, monkeypatch):
    from cron.scheduler import _RunDelivery
    from cron.scheduler_bookkeeping import finish_completed_run
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="p", schedule="every 1h", deliver="local")
        stored = jobs.load_jobs()
        stored[0]["_failure_origin"] = "script_execution"
        stored[0]["execution_id"] = "exec-origin-1"
        jobs.save_jobs(stored)
        monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: {"id": "exec-origin-1"})
        working = jobs.get_job(job["id"])  # a reused copy that carries the stale origin
        d = _RunDelivery(job=working, success=False, error="synthetic SYNTH",
                         should_deliver=False, failure_incident_id=None)
        finish_completed_run(d, fire_owner=None, execution_id="exec-origin-1")
        assert "_failure_origin" not in jobs.get_job(job["id"])


# --- The incident class comes from the origin, not the error keywords ---------------------------

def test_source_failure_incident_is_classified_script(monkeypatch, tmp_path):
    inc = _point_incidents(monkeypatch, tmp_path)
    for i, origin in enumerate(("monitor_source", "script_execution")):
        job = {"name": "J", "id": f"src-{i}", "_failure_origin": origin}
        # Error text carries no script/no_agent keyword, so a text classifier would say "unknown".
        _withheld, inc_id = scheduler._upsert_incident_for_failure(
            job, "connection reset by peer SYNTH_PEER")
        assert inc.get_incident(inc_id)["failure_type"] == "script", origin


def test_agent_failure_incident_not_forced_to_script(monkeypatch, tmp_path):
    inc = _point_incidents(monkeypatch, tmp_path)
    job = {"name": "J", "id": "agt-429", "_failure_origin": "agent_provider"}
    _withheld, inc_id = scheduler._upsert_incident_for_failure(job, "HTTP 429 rate limited")
    # Not a source stage: the text classifier still decides (rate_limit), never a forced "script".
    assert inc.get_incident(inc_id)["failure_type"] == "rate_limit"


# --- End-to-end composition: content-free copy AND a script incident from the same origin -------

def test_composed_source_failure_is_content_free_and_script_incident(monkeypatch, tmp_path):
    inc = _point_incidents(monkeypatch, tmp_path)
    _no_chain(monkeypatch)
    job = {"name": "Watch", "id": "comp01", "_failure_origin": "monitor_source"}
    content, blocked, _silent, _acked, inc_id = scheduler._compose_run_delivery(
        job, success=False,
        error="monitor_url fetch failed: https://x.invalid?token=SYNTH_T HTTP 503",
        final_response="", output_file=None)
    assert blocked is False
    for leaked in ("SYNTH_T", "503", "x.invalid", "monitor_url fetch failed"):
        assert leaked not in content, (leaked, content)
    assert "hermes cron runs comp01" in content, content
    assert inc.get_incident(inc_id)["failure_type"] == "script"
