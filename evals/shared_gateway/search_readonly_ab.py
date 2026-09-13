"""Fresh-process production tool A/B; all storage is disposable."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


def child(repo, case, account):
    home = account / '.hermes'
    home.mkdir()
    os.environ['HOME'] = str(account)
    os.environ['HERMES_HOME'] = str(home)
    sys.path.insert(0, str(repo))
    from hermes_state import SessionDB
    import tools.session_search_tool as search
    from tools.registry import registry

    target = home / 'profiles' / 'reader-target' if case == 'foreign' else home
    if case != 'empty':
        target.mkdir(parents=True, exist_ok=True)
        owner = SessionDB(target / 'state.db')
        owner.create_session('probe-session', source='cli')
        owner.append_message('probe-session', role='user', content='owned probe marker')
        owner.close()
    params = {} if case == 'empty' else {'session_id': 'probe-session'}
    if case == 'foreign':
        params['profile'] = 'reader-target'
    observer = sqlite3.connect(target / 'state.db') if case != 'empty' else None
    before = observer.execute('PRAGMA data_version').fetchone()[0] if observer else None
    entry = registry.get_entry('session_search')
    assert entry is not None
    result = json.loads(entry.handler(params))
    after = observer.execute('PRAGMA data_version').fetchone()[0] if observer else None
    if observer:
        observer.close()
    print(json.dumps({
        'module': search.__file__, 'case': case,
        'current_db_exists': (home / 'state.db').exists(),
        'success': result.get('success'),
        'read_marker': any(m.get('content') == 'owned probe marker' for m in result.get('messages', [])),
        'read_committed_write': before != after,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--account', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.case, args.account)
        return
    receipts = []
    for label, repo in [('baseline', args.baseline), ('fixed', args.fixed)]:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'tools/session_search_tool.py'], cwd=repo, text=True)
        import hashlib
        source_hash = hashlib.sha256((repo / 'tools/session_search_tool.py').read_bytes()).hexdigest()
        for case in ['empty', 'foreign', 'existing']:
            with tempfile.TemporaryDirectory(prefix='hermes-reader-ab-') as tmp:
                env = {'PATH': os.environ.get('PATH', ''), 'LANG': 'C.UTF-8', 'TZ': 'UTC',
                       'HOME': tmp, 'HERMES_HOME': str(Path(tmp) / '.hermes'),
                       'PYTHONDONTWRITEBYTECODE': '1'}
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(repo),
                                       '--case', case, '--account', tmp], cwd=repo, env=env,
                                      stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
                if proc.returncode:
                    raise RuntimeError(f'{label}/{case} failed: {proc.stderr}')
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                assert Path(row['module']).resolve() == (repo / 'tools/session_search_tool.py').resolve()
                row.update(label=label, head_sha=sha, source_sha256=source_hash, source_dirty=bool(diff))
                receipts.append(row)
    args.output.write_text(json.dumps({'passed': False, 'receipts': receipts}, indent=2) + '\n')
    by = {(r['label'], r['case']): r for r in receipts}
    for case in ['empty', 'foreign']:
        assert by['baseline', case]['current_db_exists'] is True
        assert by['fixed', case]['current_db_exists'] is False
    for label in ['baseline', 'fixed']:
        for case in ['foreign', 'existing']:
            assert by[label, case]['success'] is True
            assert by[label, case]['read_marker'] is True
            assert by[label, case]['read_committed_write'] is (label == 'baseline' and case == 'existing')
    assert by['fixed', 'empty']['success'] is False
    args.output.write_text(json.dumps({'passed': True, 'receipts': receipts}, indent=2) + '\n')
    print(json.dumps({'passed': True, 'process_cases': len(receipts), 'receipt': str(args.output)}))


if __name__ == '__main__':
    main()

