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
