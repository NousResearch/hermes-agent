import json
import os
import shutil
import subprocess
import pytest

from scripts.ci import check_thread_hops as guard


def test_syntactic_calls_and_rebindings():
    source = '''\
import threading as th
from threading import Thread as T, Timer as Later
from concurrent.futures import ThreadPoolExecutor as Pool
th.Thread(target=a)
T(target=b)
Later(1, c)
anything.submit(d)
with Pool() as worker:
    worker.map(e, items)
other.map(f, items)
for th in wrappers:
    th.Thread(target=g)
'''
    assert sorted(int(hit.split(':')[1]) for hit in guard.sites('tools/a.py', source).values()) == [4, 5, 6, 7, 9, 12]


def test_wrappers_are_hits_and_only_reasoned_comments_exempt():
    source = '''\
from agent.memory_provider import ctx_bound
x.submit(ctx_bound(f))
x.submit(ctx_bound(g))  # thread-hop: ok — verified process scoped
x.submit(f, note="thread-hop: ok - spoof")
x.submit(g)  # thread-hop: ok —
# thread-hop: ok - no profile state
x.submit(h)
'''
    assert len(guard.sites('tools/a.py', source)) == 3


def test_identity_detects_replacement_and_duplicates():
    old = guard.sites('tools/a.py', 'def f():\n x.submit(a)\n x.submit(a)\n')
    new = guard.sites('tools/a.py', 'def f():\n x.submit(a)\n x.submit(b)\n')
    assert len(old) == len(new) == 2
    assert len(new.keys() - old.keys()) == 1
    assert guard.sites('tools/a.py', 'def g():\n x.submit(a)\n').keys().isdisjoint(old)


def git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, text=True, check=True, capture_output=True).stdout.strip()


def put(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def commit(repo, message):
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', message)
    return git(repo, 'rev-parse', 'HEAD')


def test_revision_tree_bootstrap_and_prune(tmp_path, monkeypatch, capsys):
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    initial = commit(tmp_path, 'initial')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'BASELINE', tmp_path / 'scripts/ci/thread_hops_baseline.json')
    fingerprints = sorted(guard.sites('tools/a.py', 'x.submit(a)\n'))
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', json.dumps(fingerprints))
    baseline = commit(tmp_path, 'baseline')
    assert guard.main(['--base', initial, '--head', baseline]) == 0
    assert guard.main(['--base', initial, '--head', 'nonexistent-revision']) == 2
    put(tmp_path, 'tools/a.py', 'x.submit(b)\n')
    assert guard.main(['--base', initial, '--head', baseline]) == 0
    assert guard.main(['--base', baseline]) == 2
    git(tmp_path, 'add', 'tools/a.py')
    staged = git(tmp_path, 'write-tree')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    assert guard.main(['--base', baseline, '--head', staged]) == 2
    put(tmp_path, 'tools/a.py', 'pass\n')
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', '[]')
    git(tmp_path, 'add', '.')
    staged_pruned = git(tmp_path, 'write-tree')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', json.dumps(fingerprints))
    assert guard.main(['--base', baseline, '--head', staged_pruned]) == 0
    with pytest.raises(SystemExit):
        guard.main(['--base', baseline, '--head', staged, '--update'])


def test_baseline_cannot_reintroduce_removed_site(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    commit(tmp_path, 'initial')
    fingerprints = sorted(guard.sites('tools/a.py', 'x.submit(a)\n'))
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', json.dumps(fingerprints))
    base = commit(tmp_path, 'baseline')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'BASELINE', tmp_path / 'scripts/ci/thread_hops_baseline.json')
    put(tmp_path, 'tools/a.py', 'pass\n')
    assert guard.main(['--base', base, '--update']) == 0
    pruned = commit(tmp_path, 'prune')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', json.dumps(fingerprints))
    assert guard.main(['--base', pruned]) == 2


def test_malformed_source_and_baseline_fail_closed(tmp_path, monkeypatch):
    put(tmp_path, 'bad.py', 'def broken(:\n    x.submit(f)\n')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    assert guard.main(['bad.py']) == 2
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', '{broken')
    assert guard.main([]) == 2


def test_worktree_add_delete_and_exact_tree(tmp_path):
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'tools/old.py', 'x.submit(old)\n')
    head = commit(tmp_path, 'initial')
    expected = guard.sites('tools/old.py', 'x.submit(old)\n')
    (tmp_path / 'tools/old.py').unlink()
    put(tmp_path, 'tools/new.py', 'x.submit(new)\n')
    put(tmp_path, '.gitignore', 'tools/ignored.py\n')
    put(tmp_path, 'tools/ignored.py', 'x.submit(ignored)\n')
    assert guard.scan(root=tmp_path) == guard.sites('tools/new.py', 'x.submit(new)\n')
    assert guard.scan(root=tmp_path, tree=head) == expected


@pytest.mark.platforms('posix')  # a newline is not a legal Windows file-name character
def test_revision_sources_use_one_batch_process(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    for name in ('a', 'b', 'has space', 'line\nbreak'):
        put(tmp_path, f'tools/{name}.py', 'x.submit(f)\n')
    head = commit(tmp_path, 'initial')
    calls = []
    original = subprocess.run

    def recording(argv, *args, **kwargs):
        calls.append(argv)
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', recording)
    assert len(guard.scan(root=tmp_path, tree=head)) == 4
    assert sum('cat-file' in command for command in calls) == 1
    assert not any('show' in command and any('tools/' in part for part in command) for command in calls)


@pytest.mark.platforms('posix')
def test_revision_sources_work_without_cat_file_z(tmp_path, monkeypatch):
    """Git before 2.38 (Ubuntu 22.04 ships 2.34) rejects `cat-file --batch -z` with exit 129."""
    shim = tmp_path / 'bin' / 'git'
    shim.parent.mkdir()
    shim.write_text(f'#!/bin/sh\n[ "$1" = cat-file ] && for a; do [ "$a" = -z ] && exit 129; done\n'
                    f'exec {shutil.which("git")} "$@"\n', encoding='utf-8')
    shim.chmod(0o755)
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    put(repo, 'tools/a.py', 'x.submit(f)\n')
    head = commit(repo, 'initial')
    monkeypatch.setenv('PATH', f'{shim.parent}{os.pathsep}{os.environ["PATH"]}')
    assert len(guard.scan(root=repo, tree=head)) == 1


def test_detached_spawns_keeps_its_trailing_comment_above_rule(tmp_path, monkeypatch):
    from scripts.ci import production_scan
    from scripts import check_detached_spawns as detached
    put(tmp_path, 'tools/a.py', 'x = 1  # detached-spawn: ok - independent preceding statement\n'
                                'subprocess.Popen(cmd, start_new_session=True)\n')
    put(tmp_path, 'tools/b.py', 'x = "detached-spawn: ok - fake"\nsubprocess.Popen(cmd, start_new_session=True)\n')
    put(tmp_path, 'tools/c.py', 'subprocess.Popen(cmd, start_new_session=True)  # detached-spawn: ok - x\nfoo = (1,\n')
    assert set(detached.scan(list((tmp_path / 'tools').glob('*.py')), root=tmp_path)) == {'tools/b.py'}
    monkeypatch.setattr(production_scan.tokenize, 'generate_tokens',
                        lambda _: pytest.fail('marker-free source must not be tokenized'))
    assert production_scan.comment_lines('x.submit(f)\n', guard.MARKER) == set()


@pytest.mark.parametrize('separator', ['-', '--', '---', '—', '—-', '--—', '— —'])
def test_separator_only_marker_reasons_are_rejected(separator):
    from scripts.ci.production_scan import comment_lines
    for marker in ('thread-hop: ok', 'detached-spawn: ok'):
        assert comment_lines(f'x.submit(f) # {marker} {separator}\n', marker) == set()
    assert len(guard.sites('tools/a.py', f'x.submit(f) # thread-hop: ok {separator}\n')) == 1


@pytest.mark.parametrize('separator', ['-', '--', '---', '—'])
def test_marker_reasons_with_explanations_remain_valid(separator):
    from scripts.ci.production_scan import comment_lines
    for marker in ('thread-hop: ok', 'detached-spawn: ok'):
        assert comment_lines(f'x.submit(f) # {marker} {separator} process-scoped work\n', marker) == {1}
    assert guard.sites('tools/a.py', f'x.submit(f) # thread-hop: ok {separator} process-scoped work\n') == {}


def test_constructors_without_a_callable_are_not_handoffs():
    source = 'import threading\nthreading.Thread()\nthreading.Thread(name="x")\nthreading.Timer(1)\n'
    assert guard.sites('tools/a.py', source) == {}


def test_initial_baseline_rejects_changed_sources_and_mismatch(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    initial = commit(tmp_path, 'initial')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', json.dumps(sorted(guard.sites('tools/a.py', 'x.submit(a)\n'))))
    put(tmp_path, 'tools/b.py', 'pass\n')
    assert guard.main(['--base', initial]) == 2
    (tmp_path / 'tools/b.py').unlink()
    put(tmp_path, 'scripts/ci/thread_hops_baseline.json', '[]')
    assert guard.main(['--base', initial]) == 2

@pytest.mark.parametrize('source', [
    'pool.submit(safe); pool.submit(unbound)  # thread-hop: ok — safe is process scoped\n',
    '# thread-hop: ok — safe is process scoped\npool.submit(safe); pool.submit(unbound)\n',
])
def test_one_reasoned_marker_only_exempts_one_handoff(source):
    assert len(guard.sites('tools/a.py', source)) == 1


def test_a_marked_handoff_still_counts_toward_its_identical_twins():
    old = guard.sites('tools/a.py', 'def f():\n    x.submit(a)\n')
    new = guard.sites('tools/a.py', 'def f():\n    x.submit(a)  # thread-hop: ok - checked\n    x.submit(a)\n')
    assert len(new) == 1 and new.keys().isdisjoint(old)


@pytest.mark.parametrize('call', ['Thread(**options)', 'Thread(*args)', 'Timer(*args)'])
def test_expanded_constructor_arguments_are_handoffs(call):
    assert len(guard.sites('tools/a.py', f'from threading import Thread, Timer\n{call}\n')) == 1


def _moved_repo(tmp_path):
    git(tmp_path, 'init', '-q')
    put(tmp_path, 'hermes_agent/compat/moves.py', 'MOVES: dict[str, str] = {"tools": "hermes_agent.tools"}\n')
    put(tmp_path, 'hermes_agent/tools/a.py', 'x.submit(a)\n')
    put(tmp_path, 'tools/a.py', 'x.submit(a)\n')
    put(tmp_path, '.gitattributes', '# BEGIN shims (generated by scripts/move/generate_shims.py; do not edit)\n'
                                    'tools/a.py linguist-generated\n# END shims\n')
    return commit(tmp_path, 'moved')


def test_moved_production_files_are_scanned_under_their_original_identity(tmp_path):
    head = _moved_repo(tmp_path)
    for found in (guard.scan(root=tmp_path), guard.scan(root=tmp_path, tree=head)):
        assert found == guard.sites('hermes_agent/tools/a.py', 'x.submit(a)\n', 'tools/a.py')
        assert found.keys() == guard.sites('tools/a.py', 'x.submit(a)\n').keys()


def test_detached_spawns_scans_a_moved_tree(tmp_path, monkeypatch):
    from scripts import check_detached_spawns as detached
    _moved_repo(tmp_path)
    put(tmp_path, 'hermes_agent/tools/a.py', 'subprocess.Popen(cmd, start_new_session=True)\n')
    assert list(detached.scan(root=tmp_path)) == ['hermes_agent/tools/a.py']


def test_compatibility_spelled_identifiers_are_not_skipped_by_the_prefilter():
    assert len(guard.sites('tools/a.py', 'ｘ.ｓｕｂｍｉｔ(f)\n')) == 1


def test_explicit_paths_outside_the_root_are_refused(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    put(root, 'tools/a.py', 'x.submit(a)\n')
    put(tmp_path, 'outside.py', 'x.submit(a)\n')
    monkeypatch.setattr(guard, 'ROOT', root)
    assert guard.main([str(root / 'tools/a.py')]) == 1
    assert guard.main([str(tmp_path / 'outside.py')]) == 2
    assert guard.main(['../outside.py']) == 2


def test_an_undeclared_old_copy_beside_its_moved_file_fails_closed(tmp_path):
    _moved_repo(tmp_path)
    put(tmp_path, '.gitattributes', '')
    head = commit(tmp_path, 'undeclared')
    for tree in (None, head):
        with pytest.raises(ValueError, match='undeclared old-path copy'):
            guard.scan(root=tmp_path, tree=tree)
