"""Adversarial contracts for the site-level thread-handoff ratchet."""
import json
from pathlib import Path
from scripts.ci import check_thread_hops as guard


def test_aliases_and_arbitrary_submit_and_known_map():
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
'''
    assert len(guard.sites('tools/a.py', source)) == 5


def test_no_guessed_context_and_direct_wrappers():
    source = '''\
import threading
from contextvars import copy_context
from functools import partial
from agent.memory_provider import ctx_bound
threading.Thread(target=context.run)
threading.Thread(target=copy_context().run)
threading.Thread(target=ctx_bound(f))
x.submit(partial(ctx_bound(f), 1))
x.submit(lambda: copy_context().run(f))
x.submit(lambda: my_context.run(f))
'''
    assert len(guard.sites('tools/a.py', source)) == 2


def test_identity_detects_replacement_and_duplicates():
    old = guard.sites('tools/a.py', 'def f():\n x.submit(a)\n x.submit(a)\n')
    new = guard.sites('tools/a.py', 'def f():\n x.submit(a)\n x.submit(b)\n')
    assert len(old) == len(new) == 2
    assert len(new.keys() - old.keys()) == 1
    moved = guard.sites('tools/a.py', 'def g():\n x.submit(a)\n')
    assert moved.keys().isdisjoint(old)


def test_marker_requires_reason():
    source = '''\
x.submit(a)  # thread-hop: ok —
x.submit(b)  # thread-hop: ok — process scoped
x.submit(c)  # thread-hop: ok - why
'''
    assert len(guard.sites('a.py', source)) == 1


def test_malformed_source_and_baseline_fail_closed(tmp_path, monkeypatch, capsys):
    f = tmp_path / 'bad.py'
    f.write_text('def broken(:\n')
    assert guard.main([str(f)]) == 2
    baseline = tmp_path / 'baseline.json'
    baseline.write_text('{broken')
    monkeypatch.setattr(guard, 'BASELINE', baseline)
    monkeypatch.setattr(guard, 'scan', lambda *a: {})
    assert guard.main([]) == 2
    assert 'thread-hops:' in capsys.readouterr().err


def test_update_intersects_committed_and_existing(tmp_path, monkeypatch):
    a, b, c = ('a'*64, 'b'*64, 'c'*64)
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps([a, b]))
    monkeypatch.setattr(guard, 'BASELINE', baseline)
    monkeypatch.setattr(guard, 'scan', lambda *unused: {a: 'a', c: 'c'})
    monkeypatch.setattr(guard, 'git', lambda *args: 'base' if args[0] == 'merge-base' else json.dumps([a, b]))
    assert guard.main(['--update']) == 1
    assert json.loads(baseline.read_text()) == [a]


def test_baseline_increase_rejected(tmp_path, monkeypatch):
    a, b = 'a'*64, 'b'*64
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps([a, b]))
    monkeypatch.setattr(guard, 'BASELINE', baseline)
    monkeypatch.setattr(guard, 'scan', lambda *unused: {a: 'a', b: 'b'})
    monkeypatch.setattr(guard, 'git', lambda *args: 'base' if args[0] == 'merge-base' else json.dumps([a]))
    assert guard.main([]) == 2


def test_check_runner_dispatches_guard():
    import importlib.machinery
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    loader = importlib.machinery.SourceFileLoader('check_runner', str(root / 'scripts/check'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    # Inspecting the registered job exercises the same dispatch table as --only.
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    assert module._jobs(['thread-hops'], None, None)['thread-hops'].argv[1:] == ['scripts/ci/check_thread_hops.py']


def test_enclosing_expressions_and_method_scope():
    source = """
import threading as th
@th.Thread(target=deco)
def f(x=th.Thread(target=default)) -> th.Thread(target=annotation):
    pass
class C(th.Thread(target=base), metaclass=th.Thread(target=meta)):
    th = object()
    def method(self):
        th.Thread(target=inside)
"""
    assert len(guard.sites('a.py', source)) == 6


def test_trusted_imports_and_shadowing():
    source = """
from agent.memory_provider import ctx_bound as wrap
from contextvars import copy_context as copy
from functools import partial as part
x.submit(wrap(f))
x.submit(part(wrap(f), 1))
x.submit(copy().run)
x.submit(fake.ctx_bound(f))
x.submit(fake.copy_context().run)
def f(*wrap, **copy):
    x.submit(wrap(f))
    x.submit(copy().run)
"""
    assert len(guard.sites('a.py', source)) == 4


def test_lambda_and_comprehension_bindings_do_not_leak():
    source = """
import threading as th
(lambda th: th.Thread(target=no))(x)
[th.Thread(target=no) for th in things]
th.Thread(target=yes)
"""
    assert len(guard.sites('a.py', source)) == 1


def test_lambda_parameter_shadows_trusted_copy():
    source = """
from contextvars import copy_context
x.submit(lambda copy_context: copy_context().run(f))
x.submit(lambda: copy_context().run(f))
"""
    assert len(guard.sites('a.py', source)) == 1


def test_qualified_trusted_imports():
    source = """
import contextvars as cv
import tools.thread_context as tc
import functools as ft
x.submit(cv.copy_context().run)
x.submit(ft.partial(tc.propagate_context_to_thread(f), 1))
"""
    assert not guard.sites('a.py', source)
