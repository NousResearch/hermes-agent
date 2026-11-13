"""Behavior contracts for the syntactic thread-handoff ratchet."""
import json
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
    assert len(guard.sites('tools/a.py', source)) == 6


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
    assert guard.main(['--base', baseline]) == 2  # stale old fingerprint
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
    put(tmp_path, 'bad.py', 'def broken(:\n')
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
    assert not any('show' in command for command in calls)


def test_shared_comment_predicate_and_detached_legacy_parity(monkeypatch):
    from scripts.ci import production_scan
    from scripts import check_detached_spawns as detached
    source = 'x = 1  # detached-spawn: ok - independent preceding statement\nsubprocess.Popen(cmd, start_new_session=True)\n'
    lines = list[str](source.splitlines())
    assert detached._marked(lines, 2)
    assert not production_scan.marked(lines, 2, detached.MARKER)
    comments = production_scan.comment_lines(source, detached.MARKER)
    assert production_scan.marked(lines, 2, detached.MARKER,
                                  comments=comments, standalone_previous=False)
    assert not detached._marked(['x = "detached-spawn: ok - fake"', 'pass'], 2)
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

@pytest.mark.parametrize('source', [
    'pool.submit(safe); pool.submit(unbound)  # thread-hop: ok — safe is process scoped\n',
    '# thread-hop: ok — safe is process scoped\npool.submit(safe); pool.submit(unbound)\n',
])
def test_one_reasoned_marker_only_exempts_one_handoff(source):
    # A single site-level exception cannot silently bless a second same-line handoff.
    assert len(guard.sites('tools/a.py', source)) == 1
