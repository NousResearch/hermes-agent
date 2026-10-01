import pytest

from tools.file_operations import ShellFileOperations
from tools.environments.local import LocalEnvironment
from responsibilities.common import get_responsibilities_root
from responsibilities.files import read_extras
from responsibilities.schedules import reconcile
from cron.jobs import load_jobs, save_jobs, _jobs_lock
from responsibilities.run_context import build_run_prompt

CHARTER = '---\nname: daily\ntrigger: Daily operations\n---\n# Duties\nCheck the inbox.\n'


def test_native_write_patch_read_budgets(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    root = get_responsibilities_root() / 'daily'
    charter = root / 'RESPONSIBILITY.md'
    assert not ops.write_file(str(charter), CHARTER).error
    assert (root / 'STATE.md').exists()
    state = root / 'STATE.md'
    assert not ops.write_file(str(state), 'a' * 6000).error
    assert ops.patch_replace(str(state), 'a' * 6000, 'b' * 6001).error
    assert state.read_text() == 'a' * 6000
    assert read_extras(str(state))['usage'] == '100% — 6,000/6,000 chars'
    state.write_text('a' * 6600)
    assert read_extras(str(state))['usage'].startswith('110%')
    assert 'STATE.md' in read_extras(str(charter))['linked_files']['state']


def test_native_store_reconciliation_completion_and_isolation(tmp_path, monkeypatch):
    for home in (tmp_path / 'a', tmp_path / 'b', tmp_path / 'a'):
        monkeypatch.setenv('HERMES_HOME', str(home))
        root = get_responsibilities_root() / 'daily'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'references').mkdir(exist_ok=True)
        (root / 'RESPONSIBILITY.md').write_text(CHARTER)
        (root / 'STATE.md').write_text(home.name)
        (root / 'schedules').mkdir(exist_ok=True)
        declaration = root / 'schedules/check.yaml'
        declaration.write_text('schedule: 1h\nscope: Check inbox\nreport: muted\n')
        assert reconcile() == {}
        assert len(load_jobs()) == 1
        job = load_jobs()[0]
        assert job['schedule']['kind'] == 'once'
        assert home.name in build_run_prompt(job)
        with _jobs_lock():
            job.update(state='completed', enabled=False, next_run_at=None, last_run_at='2020-01-01T00:00:00+00:00')
            job['repeat']['completed'] = 1
            save_jobs([job])
        from cron.jobs import get_due_jobs
        assert get_due_jobs() == []
        assert load_jobs()[0]['state'] == 'completed'
        declaration.write_text('schedule: 1h\nscope: Check unread inbox\nreport: muted\n')
        reconcile()
        assert load_jobs()[0]['state'] == 'completed'
        declaration.write_text('schedule: [broken')
        assert reconcile()
        assert load_jobs()[0]['prompt'] == 'Check unread inbox'
        declaration.unlink()
        reconcile()
        assert load_jobs() == []


def test_guard_executes_and_replacing_declaration_retires_old_job(tmp_path, monkeypatch):
    from cron.scheduler_script import _run_job_script, _resolve_script_path
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'scripts').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'schedules').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    script = root / 'scripts/check.py'
    script.write_text("from pathlib import Path; print(Path('../STATE.md').read_text() or 'inbox changed')\n")
    declaration = root / 'schedules/check.yaml'
    declaration.write_text('schedule: every 5m\nscope: Check inbox\nreport: muted\nrepeat: 3\nscript: scripts/check.py\n')
    assert reconcile() == {}
    old = load_jobs()[0]
    ok, output = _run_job_script(old['script'], workdir=old['workdir'])
    assert ok, output
    assert 'inbox changed' in output
    outside = tmp_path / 'other.py'
    outside.write_text("print('outside')\n")
    script.unlink()
    script.symlink_to(outside)
    assert _resolve_script_path(str(script))[0] is None
    script.unlink()
    script.write_text("from pathlib import Path; print(Path('../STATE.md').read_text() or 'inbox changed')\n")
    declaration.rename(root / 'schedules/replacement.yaml')
    assert reconcile() == {}
    rows = load_jobs()
    assert len(rows) == 1
    assert rows[0]['id'] != old['id']
    assert rows[0]['responsibility']['trigger'] == 'replacement'


def test_bad_timezone_is_local_and_bookkeeping_survives(tmp_path, monkeypatch):
    from cron.jobs import update_job, get_due_jobs
    from cron.scheduler_delivery import _record_delivery_verification
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    (root / 'schedules/good.yaml').write_text('schedule: every 15m\nscope: Check inbox\nreport: muted\n')
    (root / 'schedules/bad.yaml').write_text('schedule: in 30m\ntimezone: Europe/Berlin\nscope: Check inbox\nreport: muted\n')
    errors = reconcile()
    assert 'daily/schedules/bad.yaml' in errors
    job, = load_jobs()
    assert get_due_jobs() == []
    _record_delivery_verification(job, ['telegram:123'])
    assert load_jobs()[0]['last_delivery_unverified'] == ['telegram:123']
    update_job(job['id'], {'last_delivery_error': 'notification failed'})
    assert load_jobs()[0]['last_delivery_error'] == 'notification failed'
    import pytest
    with pytest.raises(ValueError, match='file-owned'):
        update_job(job['id'], {'prompt': 'different scope'})


def test_failures_disarm_until_declaration_is_edited(tmp_path, monkeypatch):
    from cron.jobs import mark_job_run
    from cron.scheduler import _failure_streak_nudge
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    declaration = root / 'schedules/check.yaml'
    declaration.write_text('schedule: every 15m\nscope: Check inbox\nreport: muted\n')
    assert reconcile() == {}
    for failure in range(3):
        job, = load_jobs()
        if failure == 2:
            assert 're-arm' in _failure_streak_nudge(job)
        mark_job_run(job['id'], success=False, error='failed', model_unreachable=True)
        assert reconcile() == {}
    job, = load_jobs()
    assert not job['enabled'] and job['next_run_at'] is None
    assert reconcile() == {} and not load_jobs()[0]['enabled']
    declaration.write_text('schedule: [broken')
    assert reconcile() and not load_jobs()[0]['enabled']
    declaration.write_text('schedule: every 15m\nscope: Check corrected inbox\nreport: muted\n')
    assert reconcile() == {}
    job, = load_jobs()
    assert job['enabled'] and job['next_run_at'] and job['failure_streak'] == 0


def test_patch_move_obeys_destination_budgets_and_allows_rename_at_cap(tmp_path, monkeypatch):
    from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    assert not ops.write_file(str(root / 'RESPONSIBILITY.md'), CHARTER).error
    source, dest = root / 'archive/history.md', root / 'state/new.md'
    assert not ops.write_file(str(source), 'a' * 20000).error
    patch = f'*** Begin Patch\n*** Move File: {source} -> {dest}\n*** End Patch'
    operations, error = parse_v4a_patch(patch)
    assert not error
    result = apply_v4a_operations(operations, ops)
    assert result.error and source.exists() and not dest.exists()
    (root / 'state').mkdir()
    for index in range(100):
        (root / 'state' / f'{index}.md').write_text('state')
    assert ops.move_file(str(source), str(root / 'state/overflow.md')).error
    assert not ops.move_file(str(root / 'state/0.md'), str(root / 'state/renamed.md')).error


def test_scheduled_agent_keeps_workspace_separate_from_knowledge(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools.terminal_scope import install_and_reset_profile_terminal_scope
    from agent.knowledge import render
    from cron.scheduler import _resolve_job_workdir
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (tmp_path / 'config.yaml').write_text(f'terminal:\n  cwd: {workspace}\n')
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    (root / 'schedules/check.yaml').write_text('schedule: every 15m\nscope: Check inbox\nreport: muted\n')
    with _profile_runtime_scope(tmp_path), install_and_reset_profile_terminal_scope(tmp_path):
        assert reconcile() == {}
        job, = load_jobs()
        job['workdir'] = str(root)
        with _jobs_lock():
            save_jobs([job])
        assert reconcile() == {}
        job, = load_jobs()
        assert _resolve_job_workdir(job, job['id']) is None
        assert render('{workdir}') == str(workspace)
        assert str(root) in build_run_prompt(job)


@pytest.mark.parametrize('retirement', ['expired', 'interrupted_scan', 'interrupted_claim'])
def test_retired_file_one_shots_stay_consumed(tmp_path, monkeypatch, retirement):
    from datetime import datetime, timedelta, timezone
    from cron.jobs import get_due_jobs, claim_dispatch
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    declaration = root / 'schedules/check.yaml'
    declaration.write_text('schedule: 1h\nscope: Check inbox\nreport: muted\n')
    assert reconcile() == {}
    job, = load_jobs()
    job['next_run_at'] = (datetime.now(timezone.utc) - timedelta(days=2) if retirement == 'expired'
                          else datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    if retirement != 'expired':
        job['repeat']['completed'] = job['repeat']['times']
    with _jobs_lock():
        save_jobs([job])
    if retirement == 'interrupted_claim':
        assert not claim_dispatch(job['id'])
    else:
        assert not get_due_jobs()
    assert reconcile() == {}
    retained, = load_jobs()
    assert retained['id'] == job['id'] and retained['state'] == 'completed'
    assert not retained['enabled'] and retained['next_run_at'] is None
    declaration.write_text('schedule: 2h\nscope: Check inbox\nreport: muted\n')
    assert reconcile() == {}
    assert load_jobs()[0]['enabled']


def test_initial_schedule_save_survives_reconcile_crash(tmp_path, monkeypatch):
    from cron import jobs
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    declaration = root / 'schedules/check.yaml'
    declaration.write_text('schedule: "0 9 * * *"\ntimezone: Asia/Tokyo\nscope: Check inbox\nreport: muted\n')
    create = jobs.create_job
    def interrupted_create(*args, **kwargs):
        create(*args, **kwargs)
        raise RuntimeError('process stopped after initial write')
    with monkeypatch.context() as patch:
        patch.setattr(jobs, 'create_job', interrupted_create)
        with pytest.raises(RuntimeError):
            reconcile()
    first, = load_jobs()
    assert first['responsibility']['name'] == 'daily'
    assert first['schedule']['timezone'] == 'Asia/Tokyo' and first['attach_to_session']
    assert reconcile() == {}
    assert [job['id'] for job in load_jobs()] == [first['id']]
    declaration.unlink()
    assert reconcile() == {} and load_jobs() == []


def test_bad_responsibility_root_does_not_stop_native_tick(tmp_path, monkeypatch):
    from cron import scheduler, jobs
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    get_responsibilities_root().write_text('not a directory')
    job = jobs.create_job('Native work', 'every 1h')
    scanned = []
    original = scheduler.get_due_jobs
    def scan():
        scanned.append(True)
        return original()
    monkeypatch.setattr(scheduler, 'get_due_jobs', scan)
    assert scheduler.tick(verbose=False, sync=True) == 0
    assert scanned and jobs.get_job(job['id'])


def test_patch_deletion_reconciles_declarations_before_next_scan(tmp_path, monkeypatch):
    import json
    from tools import file_tools
    from responsibilities import webhook_store
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda *args: ops)
    root = get_responsibilities_root() / 'daily'
    assert not ops.write_file(str(root / 'RESPONSIBILITY.md'), CHARTER).error
    declaration = root / 'webhooks/event.yaml'
    declaration.parent.mkdir()
    declaration.write_text('scope: Handle event\nreport: muted\n')
    webhook_store.reconcile()
    with webhook_store.database() as db:
        token = db.execute('SELECT token FROM routes').fetchone()[0]
    result = json.loads(file_tools.patch_tool(mode='patch', patch=f'*** Begin Patch\n*** Delete File: {declaration}\n*** End Patch'))
    assert not result.get('error')
    assert webhook_store.route(token) is None
    declaration.write_text('scope: Handle event with rotated URL\nreport: muted\n')
    webhook_store.reconcile()
    assert webhook_store.route(token) is None
    schedule = root / 'schedules/check.yaml'
    schedule.parent.mkdir()
    schedule.write_text('schedule: 1h\nscope: Check inbox\nreport: muted\n')
    assert reconcile() == {} and len(load_jobs()) == 1
    result = json.loads(file_tools.patch_tool(mode='patch', patch=f'*** Begin Patch\n*** Delete File: {schedule}\n*** End Patch'))
    assert not result.get('error') and load_jobs() == []


def test_quiet_guard_skips_employee_but_preserves_native_script_behavior(tmp_path, monkeypatch):
    from cron.jobs import create_job
    from cron.scheduler import _prepare_job_prompt, SILENT_MARKER
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    scripts = tmp_path / 'responsibilities/daily/scripts'
    scripts.mkdir(parents=True)
    guard = scripts / 'check.py'
    guard.write_text('print(" ")\n')
    (scripts.parent / 'RESPONSIBILITY.md').write_text(CHARTER)
    (scripts.parent / 'STATE.md').write_text('')
    (scripts.parent / 'references').mkdir()
    for owned in (True, False):
        job = create_job('Check inbox', 'every 15m', script=str(guard),
                         responsibility={'name': 'daily', 'trigger': 'check', 'report': 'muted'} if owned else None)
        early, prompt = _prepare_job_prompt(job, job['id'], job['name'], None, None)
        assert early[0] is True and early[2] == SILENT_MARKER and prompt is None


def test_schedule_replacement_fences_late_completion_and_survives_crash(tmp_path, monkeypatch):
    from cron import jobs
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = get_responsibilities_root() / 'daily'
    (root / 'schedules').mkdir(parents=True)
    (root / 'references').mkdir()
    (root / 'RESPONSIBILITY.md').write_text(CHARTER)
    (root / 'STATE.md').write_text('')
    declaration = root / 'schedules/check.yaml'
    declaration.write_text('schedule: 1h\nscope: Check inbox\nreport: muted\n')
    assert reconcile() == {}
    old, = load_jobs()
    assert jobs.claim_job_for_fire(old['id'], force=True)
    declaration.write_text('schedule: 2h\nscope: Check inbox\nreport: muted\n')
    create = jobs.create_job
    def interrupted_create(*args, **kwargs):
        create(*args, **kwargs)
        raise RuntimeError('process stopped after replacement write')
    with monkeypatch.context() as patch:
        patch.setattr(jobs, 'create_job', interrupted_create)
        with pytest.raises(RuntimeError):
            reconcile()
    new, = load_jobs()
    assert new['id'] != old['id']
    assert not jobs.mark_job_run(old['id'], success=True)
    assert reconcile() == {}
    persisted, = load_jobs()
    assert persisted['id'] == new['id']
    assert persisted['enabled'] and persisted['state'] == 'scheduled'
    assert persisted['repeat']['completed'] == 0


def test_quota_recovery_preserves_responsibility_cadence(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from cron import jobs, quota_hold
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    clock = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, '_hermes_now', lambda: clock)
    monkeypatch.setattr(quota_hold, '_hermes_now', lambda: clock)
    owned = jobs.create_job('Work', '0 9 * * *', responsibility={'name': 'daily'})
    native = jobs.create_job('Work', '0 9 * * *')
    for job in (owned, native):
        assert jobs.mark_job_run(job['id'], success=False, error='quota',
                                 quota_hold_seconds=60, recover_consumed_fire=True)
    assert jobs.get_job(owned['id'])['next_run_at'] == owned['next_run_at']
    assert jobs.get_job(native['id'])['next_run_at'] != native['next_run_at']
