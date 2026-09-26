"""Archive-only migration must survive the user-facing CLI preview gate."""
from argparse import Namespace
from pathlib import Path
import sqlite3

import pytest


@pytest.mark.parametrize('store', ['legacy', 'sqlite'])
@pytest.mark.parametrize('dry_run', [False, True])
def test_cli_applies_real_archive_only_plan(tmp_path, monkeypatch, capsys, store, dry_run):
    from hermes_cli import claw

    source = tmp_path / 'openclaw'
    if store == 'legacy':
        (source / 'cron').mkdir(parents=True)
        original = source / 'cron/jobs.json'
        original.write_text('{"jobs": []}', encoding='utf8')
        archive = 'cron-store/jobs.json'
    else:
        (source / 'state').mkdir(parents=True)
        original = source / 'state/openclaw.sqlite'
        with sqlite3.connect(original) as conn:
            conn.execute('CREATE TABLE cron_jobs (job_id TEXT, enabled INTEGER)')
            conn.execute("INSERT INTO cron_jobs VALUES ('job-1', 1)")
        archive = 'cron-sqlite-jobs.json'
    before = original.read_bytes()
    target = tmp_path / 'hermes'
    target.mkdir()
    config = target / 'config.yaml'
    config.write_text('{}', encoding='utf8')
    script = Path(__file__).resolve().parents[2] / 'optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py'
    mod = claw._load_migration_module(script)

    def load_migrator(script_path, opts):
        def run(execute):
            return mod.Migrator(source_root=source, target_root=target,
                execute=execute, workspace_target=None, overwrite=False,
                migrate_secrets=False, output_dir=target / 'report',
                selected_options={'cron-jobs'}).migrate()
        return run

    # Only host preflight and category selection are isolated. The actual CLI
    # routing, preview, confirmation, apply and file archival run unchanged.
    monkeypatch.setattr(claw, '_warn_if_openclaw_running', lambda *_: None)
    monkeypatch.setattr(claw, '_warn_if_gateway_running', lambda *_: None)
    monkeypatch.setattr(claw, 'get_hermes_home', lambda: target)
    monkeypatch.setattr(claw, 'get_config_path', lambda: config)
    monkeypatch.setattr(claw, '_find_migration_script', lambda: script)
    monkeypatch.setattr(claw, '_load_migrator', load_migrator)
    claw.claw_command(Namespace(claw_action='migrate', source=str(source),
                               yes=True, no_backup=True, dry_run=dry_run))
    output = capsys.readouterr().out
    assert (target / 'report/archive' / archive).exists() is not dry_run, output
    assert 'cron-jobs' in output
    assert 'Nothing to migrate' not in output
    assert 'archived' in output
    assert not (target / 'cron/jobs.json').exists()
    assert original.read_bytes() == before


@pytest.mark.parametrize('summary,overwrite,expected', [
    ({}, False, False),
    ({'migrated': 1}, False, True),
    ({'archived': 1, 'conflict': 1}, False, False),
    ({'archived': 1, 'conflict': 1}, True, True),
])
def test_preview_keeps_empty_normal_and_conflict_contracts(summary, overwrite, expected):
    from hermes_cli import claw
    assert claw._preview_migration(
        lambda _: {'summary': summary, 'items': []},
        Namespace(dry_run=False, overwrite=overwrite),
    ) is expected


def test_preview_exception_still_prevents_apply(capsys):
    from hermes_cli import claw
    def broken(_):
        raise ValueError('fixture failure')
    assert not claw._preview_migration(broken, Namespace(dry_run=False, overwrite=False))
    assert 'Migration preview failed' in capsys.readouterr().out
