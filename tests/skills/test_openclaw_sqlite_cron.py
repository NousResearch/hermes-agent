"""Real migration entry-point regressions for SQLite-backed OpenClaw cron."""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


@pytest.fixture
def migration(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py'
    spec = importlib.util.spec_from_file_location('openclaw_sqlite_regression', script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    source = tmp_path / 'openclaw'
    (source / 'state').mkdir(parents=True)
    (source / 'openclaw.json').write_text('{}', encoding='utf8')
    target = tmp_path / 'hermes'

    def run(execute=False, selected=None, output=True):
        migrator = mod.Migrator(source_root=source, target_root=target, execute=execute,
                               workspace_target=None, overwrite=False, migrate_secrets=False,
                               output_dir=target / 'migration-report' if output else None,
                               selected_options=selected or {'cron-jobs'})
        return migrator.migrate()
    return source, target, run


def populate(source, empty=False):
    db = source / 'state/openclaw.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cron_jobs (job_id TEXT, name TEXT, enabled INTEGER, schedule_expr TEXT, payload_message TEXT)')
        if not empty:
            conn.execute('INSERT INTO cron_jobs VALUES (?, ?, ?, ?, ?)', ('job-1', 'Morning report', 1, '0 9 * * *', 'Summarize updates'))
    return db


@pytest.mark.parametrize('output', [True, False])
def test_sqlite_cron_preview_reports_live_jobs(migration, output):
    source, target, run = migration
    db = populate(source)
    before = db.read_bytes()
    report = run(output=output)
    items = [item for item in report['items'] if item['kind'] == 'cron-jobs']
    assert not any(item['reason'] == 'No cron configuration found' for item in items), items
    assert any('sqlite' in str(item['source']) and '1 SQLite cron jobs' in item['reason'] for item in items), items
    # An explicit output_dir writes the preview report, but never the archive.
    assert not (target / 'migration-report/archive/cron-sqlite-jobs.json').exists()
    if not output:
        assert not target.exists()
    assert db.read_bytes() == before


def test_execute_exports_rows_and_notes_without_activating_jobs(migration):
    source, target, run = migration
    db = populate(source)
    before = db.read_bytes()
    run(execute=True)
    archive = target / 'migration-report/archive/cron-sqlite-jobs.json'
    jobs = json.loads(archive.read_text(encoding='utf8'))
    assert jobs[0]['payload_message'] == 'Summarize updates'
    assert jobs[0]['schedule_expr'] == '0 9 * * *'
    assert 'cron-sqlite-jobs.json' in (target / 'migration-report/MIGRATION_NOTES.md').read_text(encoding='utf8')
    assert not (target / 'cron/jobs.json').exists()
    assert db.read_bytes() == before


@pytest.mark.parametrize('store', ['empty', 'missing_table', 'corrupt'])
def test_empty_missing_and_corrupt_store_preserve_legacy_archival(migration, store):
    source, target, run = migration
    if store == 'empty':
        populate(source, empty=True)
    elif store == 'missing_table':
        with sqlite3.connect(source / 'state/openclaw.sqlite') as conn:
            conn.execute('CREATE TABLE unrelated (value TEXT)')
    else:
        (source / 'state/openclaw.sqlite').write_bytes(b'not a sqlite database')
    (source / 'cron').mkdir()
    (source / 'cron/jobs.json').write_text('{"jobs": []}', encoding='utf8')
    report = run(execute=True)
    items = [i for i in report['items'] if i['kind'] == 'cron-jobs']
    assert (target / 'migration-report/archive/cron-store/jobs.json').exists()
    assert not (target / 'migration-report/archive/cron-sqlite-jobs.json').exists()
    if store == 'empty':
        assert any('table is empty' in i['reason'] for i in items)
    elif store == 'corrupt':
        assert any(i['status'] == 'error' and 'SQLite' in i['reason'] for i in items)
    else:
        assert not any(i['status'] == 'error' for i in items)


def test_unselected_cron_is_not_exported(migration):
    source, target, run = migration
    populate(source)
    report = run(execute=True, selected={'soul'})
    assert all(i['status'] == 'skipped' for i in report['items'] if i['kind'] == 'cron-jobs')
    assert not (target / 'migration-report/archive/cron-sqlite-jobs.json').exists()
