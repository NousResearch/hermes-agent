"""Opt-in content admission is separate from successful script transport."""
import json
from unittest.mock import Mock


def test_content_blocked_never_wakes_agent_and_is_not_success(monkeypatch):
    from cron import scheduler
    monkeypatch.setattr('hermes_cli.config.require_parseable_user_config', lambda: None)
    monkeypatch.setattr(scheduler, '_apply_monitor_gate', lambda *a: (None, None, None))
    gate = dict(wakeAgent=False, state='BLOCKED', contentOutcome='BLOCKED', reason='upstream-not-accepted')
    monkeypatch.setattr(scheduler, '_run_job_script_with_claim_heartbeat', lambda *a, **k: (True, json.dumps(gate)))
    build = Mock()
    monkeypatch.setattr(scheduler, '_build_job_prompt', build)
    result, prompt = scheduler._prepare_job_prompt(dict(id='fixture',prompt='fixture',script='fixture'), 'fixture', 'fixture', None, None)
    assert prompt is None
    assert result[0] is False
    assert 'CONTENT_BLOCKED' in result[3]
    assert 'upstream-not-accepted' in result[1]
    assert result[2].startswith('[CRON_FAILURE]\nCONTENT_BLOCKED:')
    build.assert_not_called()


def test_legacy_wake_false_stays_silent(monkeypatch):
    from cron import scheduler
    monkeypatch.setattr('hermes_cli.config.require_parseable_user_config', lambda: None)
    monkeypatch.setattr(scheduler, '_apply_monitor_gate', lambda *a: (None, None, None))
    monkeypatch.setattr(scheduler, '_run_job_script_with_claim_heartbeat', lambda *a, **k: (True, '{"wakeAgent":false}'))
    result, prompt = scheduler._prepare_job_prompt(dict(id='fixture',prompt='fixture',script='fixture'), 'fixture', 'fixture', None, None)
    assert prompt is None and result[0] is True
    assert result[2] == scheduler.SILENT_MARKER
