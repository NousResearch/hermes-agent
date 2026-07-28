"""Behavior contract for transient-failure alert suppression on recurring cron jobs.

A recurring job whose contract is "silence is the default" (ambient sense engine,
watchdog pollers) must not page the operator every time a provider stalls for a
single tick. When a recurring job fails on a *transient* infrastructure blip
(idle-kill, provider stall, rate limit, transient network error) AND its previous
run succeeded, the chat alert is suppressed — the next scheduled run recovers on
its own. Escalation: a SECOND consecutive failure pages through, so a real (non-
recovering) outage is never hidden. One-shot jobs always alert on failure.

These assert INVARIANTS (which class of failure suppresses vs pages), not frozen
values, so they survive marker-list edits.
"""
import cron.scheduler as s


# ---------------------------------------------------------------------------
# _is_transient_cron_failure — the classifier
# ---------------------------------------------------------------------------

def test_classifier_matches_idle_kill():
    assert s._is_transient_cron_failure(
        "TimeoutError: Cron job 'x' idle for 602s (limit 600s) — last activity: initializing"
    ) is True


def test_classifier_matches_provider_stalls_and_rate_limits():
    for err in (
        "Fallback chain was exhausted or unavailable",
        "Error 429 weekly usage limit exceeded",
        "503 model overloaded, please retry",
        "httpx.ReadTimeout: timed out",
        "ConnectionResetError: [Errno 54] Connection reset by peer",
        "provider timeout",
    ):
        assert s._is_transient_cron_failure(err) is True, err


def test_classifier_rejects_real_defects():
    for err in (
        "Script exited with code 1: FileNotFoundError",
        "KeyError: 'foo' in prompt template",
        "agent reported failure",
        "provider authentication error (401)",
    ):
        assert s._is_transient_cron_failure(err) is False, err


def test_classifier_handles_none_and_empty():
    assert s._is_transient_cron_failure(None) is False
    assert s._is_transient_cron_failure("") is False


# ---------------------------------------------------------------------------
# _suppress_transient_cron_failure_alert — the decision
# ---------------------------------------------------------------------------

_CFG_DEFAULT = {"cron": {}}  # suppress default ON


def _job(kind="interval", last_status: object = "ok", **extra):
    j = {"id": "j", "name": "n", "schedule": {"kind": kind}, "last_status": last_status}
    j.update(extra)
    return j


def test_recurring_transient_after_success_is_suppressed():
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", "ok"), "idle for 602s", _CFG_DEFAULT
    ) is True


def test_second_consecutive_failure_escalates():
    # last_status already "error" => not self-recovering => page through.
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", "error"), "idle for 602s", _CFG_DEFAULT
    ) is False


def test_brand_new_job_none_prev_status_is_suppressed():
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", None), "provider timeout", _CFG_DEFAULT
    ) is True


def test_one_shot_always_alerts_even_when_transient():
    assert s._suppress_transient_cron_failure_alert(
        _job("once", "ok"), "idle for 602s", _CFG_DEFAULT
    ) is False


def test_real_defect_always_alerts():
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", "ok"), "Script exited with code 1", _CFG_DEFAULT
    ) is False


def test_per_job_opt_out_forces_alert():
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", "ok", alert_on_transient_failure=True),
        "idle for 602s", _CFG_DEFAULT,
    ) is False


def test_global_knob_off_forces_alert():
    assert s._suppress_transient_cron_failure_alert(
        _job("interval", "ok"),
        "idle for 602s",
        {"cron": {"suppress_transient_failure_alerts": False}},
    ) is False


# ---------------------------------------------------------------------------
# End-to-end through run_one_job: the suppressed transient failure must still
# SAVE output and MARK the run, but must NOT deliver.
# ---------------------------------------------------------------------------

def _patch_pipeline(monkeypatch, *, success, error):
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        return (success, "out", "" if not success else "final", error)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: calls.append(("save", jid)) or f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda job, content, adapters=None, loop=None: calls.append(("deliver", job["id"])))
    monkeypatch.setattr(s, "mark_job_run", lambda jid, ok, err=None, delivery_error=None: calls.append(("mark", jid, ok)))
    return calls


def test_run_one_job_suppresses_transient_failure_delivery(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: _CFG_DEFAULT)
    calls = _patch_pipeline(monkeypatch, success=False, error="idle for 602s — last activity: initializing")

    ok = s.run_one_job(_job("interval", "ok"))

    kinds = [c[0] for c in calls]
    assert "save" in kinds          # output still saved
    assert "mark" in kinds          # run still recorded
    assert "deliver" not in kinds   # but no chat page
    assert ok is True


def test_run_one_job_delivers_transient_failure_on_second_strike(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: _CFG_DEFAULT)
    calls = _patch_pipeline(monkeypatch, success=False, error="idle for 602s")

    s.run_one_job(_job("interval", "error"))  # prev already error => escalate

    assert "deliver" in [c[0] for c in calls]


def test_run_one_job_delivers_real_defect(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: _CFG_DEFAULT)
    calls = _patch_pipeline(monkeypatch, success=False, error="Script exited with code 1")

    s.run_one_job(_job("interval", "ok"))

    assert "deliver" in [c[0] for c in calls]


def test_no_agent_watchdog_timeout_always_delivers(monkeypatch):
    monkeypatch.setattr(s, "load_config", lambda: _CFG_DEFAULT)
    calls = _patch_pipeline(monkeypatch, success=False, error="script timed out after 3600s")

    s.run_one_job(_job("interval", "ok", no_agent=True, script="watchdog.sh"))

    assert "deliver" in [c[0] for c in calls]


def test_schema_exposes_alert_on_transient_failure():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    assert "alert_on_transient_failure" in CRONJOB_SCHEMA["parameters"]["properties"]
