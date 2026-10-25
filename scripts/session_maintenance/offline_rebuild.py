"""Run from the user's shell with all Hermes database holders stopped.
Never stops processes, starts chats, uses --force, or deletes conversations.
"""
import argparse

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hermes_constants import get_hermes_home
HOME = get_hermes_home()
DB = HOME / 'state.db'
ROOT = HOME / 'maintenance'
CLI = [sys.executable, '-m', 'hermes_cli.main']
SEARCH_TERMS = ()
from hermes_state_fts import fts_layout_current, fts_layout_drift, load_fts5_cjk_extension


from hermes_state_holders import foreign_state_db_holders
from verification import open_snapshot, verify_snapshots
from bounded_verification import UIDBudget, validate_budget, verify_bounded
from fts_health import verify_fts_health
from storage_safety import ensure_directory, maintenance_lock, sync_directory, sync_file


def holders(path):
    # Use Hermes' own inode-aware, deleted-WAL-aware maintenance guard.
    return sorted({pid for pid, detail in foreign_state_db_holders(path)})


def exclusive():
    live = holders(DB)
    if live:
        raise RuntimeError('Database is still open in PIDs ' + ', '.join(map(str, live)) +
                           '. Quit the CLI and stop the gateway from your shell first; no --force.')


def connect(path):
    c = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
    load_fts5_cjk_extension(c)
    return c


def fingerprint(path, *, cold_snapshot=False):
    c = open_snapshot(path) if cold_snapshot else connect(path)
    try:
        result = {}
        for table in ('sessions', 'messages'):
            digest = hashlib.sha256()
            count = 0
            columns = [r[1] for r in c.execute('pragma table_info(' + table + ')')]
            for row in c.execute('select * from ' + table + ' order by id'):
                digest.update(json.dumps(list(row), ensure_ascii=True, separators=(',', ':'),
                                         default=lambda x: {'blob_hex': x.hex()}).encode())
                digest.update(b'\n')
                count += 1
            result[table] = dict(columns=columns, rows=count, sha256=digest.hexdigest())
        return result
    finally:
        c.close()


def probes(c):
    # Both old and new layouts retain ordinary user/assistant text.
    return {q: [r[0] for r in c.execute(
        "SELECT f.rowid FROM messages_fts f JOIN messages m ON m.id=f.rowid "
        "WHERE messages_fts MATCH ? AND m.role IN ('user','assistant') "
        "ORDER BY f.rowid LIMIT 5", (q,))] for q in SEARCH_TERMS}


def check_integrity(snapshot, timeout=900):
    # SQLite's progress hook may not run promptly inside an integrity-check opcode.
    # A separate read-only child gives this expensive postcondition a hard deadline.
    done = subprocess.run([sys.executable, str(Path(__file__).with_name('quick_check.py')), str(snapshot)],
                          capture_output=True, text=True, encoding='utf-8', check=True, timeout=timeout)
    return json.loads(done.stdout)


def command(argv, log, timeout):
    with log.open('w') as output:
        subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def self_test():
    with tempfile.TemporaryDirectory(dir=HOME / 'cache/scratch') as td:
        p = Path(td) / 'synthetic.db'
        c = sqlite3.connect(p)
        c.executescript("CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT);"
                        "CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT);"
                        "INSERT INTO sessions VALUES('test','synthetic');"
                        "INSERT INTO messages VALUES(1,'synthetic');")
        c.commit()
        first = fingerprint(p)
        assert first == fingerprint(p)
        assert first['sessions']['rows'] == first['messages']['rows'] == 1
        c.execute("UPDATE messages SET content='changed' WHERE id=1")
        c.commit()
        assert first != fingerprint(p)
        c.close()
        child = subprocess.Popen([sys.executable, '-c',
            'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print("ready",flush=True);sys.stdin.read()', str(p)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        try:
            assert child.stdout is not None and child.stdout.readline().strip() == 'ready'
            assert child.pid in holders(p)
        finally:
            child.communicate('', timeout=10)
        assert child.pid not in holders(p)
    print('PASS: full-row hashes, change detection, row counts, live-holder detection and exit detection.')


def main(check_only=False, execute_offline=False, *, budget=None, lock=None):
    if check_only:
        c = connect(DB)
        try:
            print(json.dumps({'holders': holders(DB), 'layout_current': fts_layout_current(c.cursor()),
                              'layout_drift': fts_layout_drift(c.cursor()), 'db_bytes': DB.stat().st_size,
                              'free_bytes': shutil.disk_usage(HOME).free}, indent=2))
        finally:
            c.close()
        return 0
    if not execute_offline:
        raise RuntimeError('Execution disabled. Requires an approved offline window and --execute-offline.')
    if not isinstance(budget, UIDBudget):
        raise RuntimeError('An independently approved UID budget is required before any writes.')
    validate_budget(budget)
    exclusive()
    os.umask(0o077)
    with maintenance_lock(DB, lock):
        exclusive()  # No output directory, checkpoint, or DB mutation before this passes.
        ensure_directory(ROOT)
        # Backup, preserved before/after snapshots, rebuild/VACUUM headroom.
        need = DB.stat().st_size * 4 + 1024**3
        if shutil.disk_usage(HOME).free < need:
            raise RuntimeError('Insufficient room for backup, verification snapshots and full VACUUM.')
        run_dir = ROOT / ('offline-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ'))
        ensure_directory(run_dir)
        report = {'status': 'running', 'directory': str(run_dir), 'before_db_bytes': DB.stat().st_size,
                  'before_free_bytes': shutil.disk_usage(HOME).free}
        def save(phase):
            report['phase'] = phase
            report['updated_at'] = datetime.now(UTC).isoformat()
            temporary = run_dir / 'receipt.pending'
            with temporary.open('w', encoding='utf-8') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(run_dir / 'receipt.json')
            directory_fd = os.open(run_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            # All durable writes happen before optional terminal output.
            try:
                print(phase + ': ' + str(run_dir), flush=True)
            except (BrokenPipeError, OSError):
                sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        try:
            c = connect(DB)
            try:
                cursor = c.execute("SELECT value FROM state_meta WHERE key='message_uid_backfill_id'").fetchone()
                if int(cursor[0] if cursor else 0) != budget.before_cursor:
                    raise RuntimeError('Unexpected baseline UID cursor; no maintenance attempted')
                report['search_before'] = probes(c)
            finally:
                c.close()
            if len(report['search_before']) < 2 or not all(report['search_before'].values()):
                raise RuntimeError('Representative search IDs need two nonempty baseline probes')
            exclusive()
            save('cold-backup')
            c = sqlite3.connect(DB, timeout=10)
            try:
                if c.execute('pragma wal_checkpoint(TRUNCATE)').fetchone()[0]:
                    raise RuntimeError('Checkpoint busy; a new database holder appeared.')
            finally:
                c.close()
            exclusive()
            backup = run_dir / 'state-before.db.zst'
            command(['zstd', '-T2', '-3', str(DB), '-o', str(backup)], run_dir / 'backup.log', 1800)
            command(['zstd', '-t', str(backup)], run_dir / 'backup-check.log', 1800)
            sync_file(backup)
            sync_directory(run_dir)
            report['backup'] = str(backup)
            report['backup_verified'] = True
            baseline = run_dir / 'state-before.db'
            command(['zstd', '-d', str(backup), '-o', str(baseline)], run_dir / 'snapshot-before.log', 1800)
            sync_file(baseline)
            save('fingerprinting-before')
            before = fingerprint(baseline, cold_snapshot=True)
            (run_dir / 'before.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
            sync_file(run_dir / 'before.json')
            exclusive()
            save('rebuilding-index')
            env = os.environ.copy()
            env['HERMES_HOME'] = str(HOME)
            with (run_dir / 'optimize.log').open('w') as output:
                subprocess.run([*CLI, 'sessions', 'optimize-storage', '--yes'], env=env, cwd=REPO,
                               stdout=output, stderr=subprocess.STDOUT, timeout=7200, check=True)
            exclusive()
            save('verifying')
            snapshot = run_dir / 'state-after.db'
            # SQLite backup includes committed WAL without checkpointing/migrating the source.
            source = connect(DB)
            target = sqlite3.connect(snapshot)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
            exclusive()
            after = fingerprint(snapshot, cold_snapshot=True)
            sync_file(snapshot)
            (run_dir / 'after.json').write_text(json.dumps(after, indent=2) + '\n', encoding='utf-8')
            sync_file(run_dir / 'after.json')
            report['verification'] = verify_bounded(baseline, snapshot, budget)
            report['conversation_content_and_protected_fields_unchanged'] = True
            report['full_rows_unchanged'] = before == after
            report['counts'] = {key: value['rows'] for key, value in before.items()}
            c = open_snapshot(snapshot)
            load_fts5_cjk_extension(c)
            try:
                report['fts_health'] = verify_fts_health(c)
                report['layout_current'] = fts_layout_current(c.cursor())
                report['layout_drift'] = fts_layout_drift(c.cursor())
                report['pending_markers'] = c.execute("select key from state_meta where key like 'fts%rebuild%'").fetchall()
                report['search_after'] = probes(c)
            finally:
                c.close()
            report['quick_check'] = check_integrity(snapshot)
            if not report['layout_current'] or report['layout_drift'] or report['pending_markers']:
                raise RuntimeError('Index layout or pending-rebuild verification failed.')
            if (len(report['search_before']) < 2 or not all(report['search_before'].values())
                    or report['search_before'] != report['search_after']):
                raise RuntimeError('Representative search IDs changed or no baseline matches.')
            if report['quick_check'] != ['ok']:
                raise RuntimeError('SQLite integrity check failed.')
            exclusive()
            report['after_db_bytes'] = DB.stat().st_size
            report['after_free_bytes'] = shutil.disk_usage(HOME).free
            for evidence in run_dir.glob('*.log'):
                sync_file(evidence)
            report['status'] = 'complete'
            save('complete-offline-checks')
            return 0
        except BaseException as error:
            report['status'] = 'blocked'
            report['error'] = type(error).__name__ + ': ' + str(error)
            save('blocked')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--execute-offline', action='store_true')
    parser.add_argument('--uid-budget', type=Path, help='Reviewed JSON UIDBudget; never inferred from the after-state')
    parser.add_argument('--search', action='append', default=[], help='At least two distinct existing FTS queries')
    args = parser.parse_args()
    SEARCH_TERMS = tuple(dict.fromkeys(args.search))
    if args.self_test:
        self_test()
    else:
        budget = UIDBudget(**json.loads(args.uid_budget.read_text(encoding='utf-8-sig'))) if args.uid_budget else None
        sys.exit(main(args.check, args.execute_offline, budget=budget))
