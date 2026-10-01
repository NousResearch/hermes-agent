import pytest

from cron.jobs import load_jobs
from cron.scheduler_prompt import _build_job_prompt
from responsibilities.common import get_responsibilities_root
from responsibilities.schedules import reconcile


@pytest.mark.parametrize('report', ['muted', 'local', 'telegram:123'])
@pytest.mark.parametrize('schedule', ['1h', 'every 1h'])
def test_schedule_prompt_keeps_native_rules_and_scoped_assignment(tmp_path, monkeypatch, report, schedule):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    package = get_responsibilities_root() / 'support'
    (package / 'schedules').mkdir(parents=True)
    (package / 'references').mkdir()
    charter = '---\nname: support\ntrigger: Customer support\n---\nHandle refunds and account questions.\n'
    (package / 'RESPONSIBILITY.md').write_text(charter)
    (package / 'STATE.md').write_text('Refund request A was already answered.')
    (package / 'schedules/refunds.yaml').write_text(
        f'schedule: {schedule}\nscope: Check unanswered refund requests\nreport: {report}\n'
    )
    assert reconcile() == {}
    job = load_jobs()[0]

    prompt = _build_job_prompt(job)
    native = _build_job_prompt({'id': 'native', 'prompt': 'Check inbox'})
    shared_rules = native[native.index('SILENT:'):native.index(']\n\n') + 3]
    assert shared_rules in prompt
    assert prompt.count('DELIVERY:') == 1
    assert f"Schedule '{job['name']}' (ID {job['id']})" in prompt
    assert 'Read the references for every duty in this run\'s scope.' in prompt
    assert 'Check the dates and periods in saved state' in prompt
    assert '<schedule_scope>\nCheck unanswered refund requests\n</schedule_scope>' in prompt
    assert charter in prompt
    assert 'Refund request A was already answered.' in prompt
    assert str(package) in prompt
    assert ('a one-time run of its work' in prompt) == (schedule == '1h')
    assert ('one of its recurring rhythms' in prompt) == (schedule == 'every 1h')
    if report in {'muted', 'local'}:
        assert 'recorded in the run log, not posted anywhere' in prompt
        assert 'not for a routine run report' in prompt
        assert 'automatically delivered' not in prompt
    else:
        assert report in prompt
        assert 'additional destination' in prompt
        assert 'do not duplicate the final response' in prompt

    (package / 'STATE.md').write_text('Refund request B is waiting for approval.')
    refreshed = _build_job_prompt(job)
    assert 'Refund request B is waiting for approval.' in refreshed
    assert 'Refund request A was already answered.' not in refreshed


def test_finite_schedule_preserves_completion_rules(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    package = get_responsibilities_root() / 'reminder'
    (package / 'schedules').mkdir(parents=True)
    (package / 'references').mkdir()
    (package / 'RESPONSIBILITY.md').write_text(
        '---\nname: reminder\ntrigger: Demo reminder\nlifecycle: finite\n---\nRemind Mark about the demo.\n'
    )
    (package / 'STATE.md').write_text('')
    (package / 'schedules/remind.yaml').write_text(
        'schedule: 1h\nscope: Send the demo reminder\nreport: muted\n'
    )
    assert reconcile() == {}

    prompt = _build_job_prompt(load_jobs()[0])
    assert 'This responsibility is finite' in prompt
    assert "delete this responsibility's schedule and webhook declaration files" in prompt
    assert 'Deleting the package itself is a conversational decision' in prompt
    assert '[CRON_FAILURE]' in prompt


def test_ordinary_cron_retains_native_delivery_policy():
    prompt = _build_job_prompt({'id': 'native', 'prompt': 'Check inbox'})
    assert 'do NOT use send_message or try to deliver the output yourself' in prompt
    assert prompt.endswith('Check inbox')
    assert '<responsibility_document>' not in prompt
    assert 'additional destination' not in prompt
