import pytest
from cron import scheduler as s

JOB={'id':'source-test','name':'Source test','monitor_script':'poll.py','schedule':{'kind':'interval','minutes':5}}

@pytest.mark.parametrize('error',[
 'Script exited with code 1\nstderr:\nRuntimeError: upstream HTTP 500',
 'Script exited with code 1\nstderr:\nurllib.error.URLError: [Errno -3] Temporary failure in name resolution',
 '[monitor_source_failed] monitor_url fetch failed: HTTP Error 401: Unauthorized',
])
def test_source_failure_never_blames_model_or_recommends_model_changes(monkeypatch,error):
    monkeypatch.setattr(s,'load_config',dict)
    monkeypatch.setattr(s,'get_fallback_chain',lambda cfg:[])
    notice=s._summarize_cron_failure_for_delivery(JOB,error)
    assert 'No model was invoked' in notice
    assert 'sign in again' not in notice.lower()
    assert '--provider' not in notice and '--model' not in notice

@pytest.mark.parametrize('streak,expected_silent',[(0,True),(1,True),(2,False)])
def test_short_monitor_transient_is_recorded_but_only_persistent_failure_alerts(monkeypatch,streak,expected_silent):
    captured=[]
    monkeypatch.setattr(s,'_upsert_incident_for_failure',lambda *a,**kw: (captured.append(a) or False,'incident'))
    monkeypatch.setattr(s,'_failure_streak_nudge',lambda job:'')
    text=s._compose_run_delivery(dict(JOB,failure_streak=streak),success=False,error='[monitor_source_failed] Script exited with code 1\nstderr:\nHTTP 500',final_response='',output_file='output.txt')[0]
    assert bool(text) is not expected_silent
    assert len(captured)==1

@pytest.mark.parametrize('job,error',[
 (dict(JOB,schedule={'kind':'once'}),'[monitor_source_failed] HTTP 500'),
 (dict(JOB,schedule={'kind':'interval','minutes':60}),'[monitor_source_failed] HTTP 500'),
 (JOB,'[monitor_source_failed] HTTP 401 Unauthorized'),
 (JOB,'[monitor_source_failed] disk full HTTP 500'),
 (JOB,'[monitor_source_failed] unexpected failure'),
 (JOB,'provider HTTP 500'),
])
def test_critical_unknown_rare_and_actual_model_failures_are_immediate(monkeypatch,job,error):
    monkeypatch.setattr(s,'_upsert_incident_for_failure',lambda *a,**kw:(False,'incident'))
    monkeypatch.setattr(s,'_failure_streak_nudge',lambda job:'')
    monkeypatch.setattr(s,'load_config',dict)
    monkeypatch.setattr(s,'get_fallback_chain',lambda cfg:[])
    text=s._compose_run_delivery(job,success=False,error=error,final_response='',output_file='output.txt')[0]
    assert text

@pytest.mark.parametrize('streak,expected',[(0,True),(1,True),(2,False)])
def test_reported_eai_nodata_is_a_source_transient(streak,expected):
    from cron.scheduler_failure_copy import defer_transient_source_notice
    error='[monitor_source_failed] Script exited with code 1\nstderr:\nURLError: [Errno -5] DNS lookup failed'
    assert defer_transient_source_notice(dict(JOB,failure_streak=streak),error) is expected


@pytest.mark.parametrize("schedule", [
    {"kind": "cron", "expr": "0,5 9 * * *"},
    {"kind": "cron", "expr": "*/5 9 * * *"},
])
def test_sparse_cron_never_defers_source_failure(schedule):
    from cron.scheduler_failure_copy import defer_transient_source_notice
    assert not defer_transient_source_notice(dict(JOB, schedule=schedule), "[monitor_source_failed] HTTP 503")

@pytest.mark.parametrize("completed", [0, 1])
def test_finite_monitor_alerts_before_no_retry_remains(completed):
    from cron.scheduler_failure_copy import defer_transient_source_notice
    job = dict(JOB, repeat={"times": 2, "completed": completed})
    assert defer_transient_source_notice(job, "[monitor_source_failed] HTTP 503") is (completed == 0)

@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_native_httpx_monitor_failure_is_transient(status):
    import httpx
    from cron.scheduler_failure_copy import defer_transient_source_notice
    response = httpx.Response(status, request=httpx.Request("GET", "https://example.test/monitor"))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        response.raise_for_status()
    assert defer_transient_source_notice(JOB, "[monitor_source_failed] monitor_url fetch failed: " + str(raised.value))


@pytest.mark.parametrize("expression", ["* * * * *", "*/5 * * * *"])
def test_continuous_short_cron_can_defer(expression):
    from cron.scheduler_failure_copy import defer_transient_source_notice
    assert defer_transient_source_notice(dict(JOB, schedule={"kind": "cron", "expr": expression}), "[monitor_source_failed] HTTP 503")


def test_rejected_monitor_url_keeps_credentials_out_of_delivery(monkeypatch):
    from cron.monitor import _fetch_monitor_url
    from cron.scheduler_delivery import _redact_cron_payload
    from tools import url_safety

    def no_network(*args, **kwargs):
        pytest.fail("An invalid URL must be rejected before network access")

    monkeypatch.setattr(url_safety, "create_ssrf_safe_client", no_network)
    url = "ftp://fixture-user:fixture-password@example.test/feed?access_token=fixture-query-token"
    ok, error = _fetch_monitor_url(url)
    assert not ok
    notice = s._summarize_cron_failure_for_delivery(JOB, "[monitor_source_failed] " + error)
    delivered = _redact_cron_payload(notice, "source-failure test")
    assert "fixture-query-token" not in delivered
    assert "fixture-password" not in delivered
    assert "fixture-user" not in delivered
    assert "ftp://" not in notice
    assert "Source test" in delivered
    assert "source" in delivered
    assert "No model was invoked" in delivered
    assert "hermes cron runs source-test" in delivered


@pytest.mark.parametrize("error", [
    "Script exited with code 1\nstderr:\nsubprocess.TimeoutExpired: "
    "Command '['curl', '-u', 'fixture-user:fixture-opaque-value']' timed out",
    "[monitor_source_failed] Script exited with code 1\nstderr:\n"
    "upstream returned fixture-opaque-value HTTP 401",
])
def test_source_tracebacks_stay_out_of_chat_notice(error):
    from cron.scheduler_delivery import _redact_cron_payload

    notice = s._summarize_cron_failure_for_delivery(JOB, error)
    delivered = _redact_cron_payload(notice, "source-failure test")
    assert "fixture-opaque-value" not in delivered
    assert "subprocess.TimeoutExpired" not in notice
    assert "Source test" in delivered
    assert "No model was invoked" in delivered
    assert "hermes cron runs source-test" in delivered
