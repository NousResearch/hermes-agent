"""Missing cron policy behavior, tested against actual helpers."""
import pytest
from cron import jobs, scheduler

@pytest.mark.parametrize('value', [-1, 'nan', 'inf', '-inf', None, 'bad'])
def test_invalid_timeout_defaults(monkeypatch, value):
    monkeypatch.delenv('HERMES_CRON_TIMEOUT', raising=False)
    cfg = {'cron': {'inactivity_timeout_seconds': value}}
    assert jobs.resolve_cron_inactivity_timeout(cfg) == 600
    assert scheduler._cron_inactivity_seconds(cfg) == 600
    assert jobs._oneshot_run_claim_ttl_seconds(cfg) == 1800

@pytest.mark.parametrize('value,ttl', [(0,1800), (3600,10800), (1,1800)])
def test_yaml_timeout(monkeypatch, value, ttl):
    monkeypatch.delenv('HERMES_CRON_TIMEOUT', raising=False)
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {'cron': {'inactivity_timeout_seconds': value}})
    assert scheduler._cron_inactivity_seconds() == value
    assert jobs._oneshot_run_claim_ttl_seconds() == ttl

def test_env_wins(monkeypatch):
    monkeypatch.setenv('HERMES_CRON_TIMEOUT', '1200')
    assert scheduler._cron_inactivity_seconds({'cron': {'inactivity_timeout_seconds': 3600}}) == 1200

@pytest.mark.parametrize('error', ['TimeoutError: local subprocess exceeded its deadline', 'Local subprocess timed out', 'TimeoutError: deadline exceeded'])
def test_local_deadline_not_provider(error):
    msg = scheduler._summarize_cron_failure_for_delivery({'id':'audit'}, error)
    assert 'did not respond in time' not in msg
    assert 'backup provider' not in msg.lower()
    assert error in msg

@pytest.mark.parametrize('error', ['Request timed out.', 'httpx.ReadTimeout: timed out', 'APITimeoutError: timed out', 'ConnectTimeout: timed out'])
def test_provider_timeout_retained(monkeypatch, error):
    monkeypatch.setattr(scheduler, 'load_config', dict)
    monkeypatch.setattr(scheduler, 'get_fallback_chain', lambda cfg: [])
    msg = scheduler._summarize_cron_failure_for_delivery({'id':'audit'}, error)
    assert 'did not respond in time' in msg
    assert 'No backup provider is configured' in msg

def test_stall_activity_preserves_native_actions():
    msg = scheduler._summarize_cron_failure_for_delivery({'id':'audit'}, 'TimeoutError: Cron job idle for 601s (limit 600s) — last activity: executing tool: terminal')
    assert 'stalled' in msg
    assert 'hermes cron runs audit' in msg
    assert 'Last activity: executing tool: terminal' in msg
    assert 'backup provider' not in msg

def test_per_job_messaging_remains_denied():
    assert 'messaging' in scheduler._resolve_cron_disabled_toolsets({})
    assert 'messaging' in scheduler._resolve_cron_disabled_toolsets({'cron': {'allow_agent_scheduling': True}})
