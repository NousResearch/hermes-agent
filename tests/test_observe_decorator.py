import functools
import threading

import pytest


def test_observe_decorates_once_and_preserves_call(monkeypatch):
    from agent import observability as obs

    calls = {"decorate": 0, "langfuse_wrapper_calls": 0, "fn_calls": 0}

    def fake_langfuse_observe(*_args, **_kwargs):
        calls["decorate"] += 1

        def _decorator(fn):
            @functools.wraps(fn)
            def _wrapped(*a, **kw):
                calls["langfuse_wrapper_calls"] += 1
                return fn(*a, **kw)

            return _wrapped

        return _decorator

    monkeypatch.setattr(obs, "LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(obs, "_langfuse_observe", fake_langfuse_observe)

    obs.set_langfuse_enabled(True)
    try:
        @obs.observe(name="t")
        def f(a, *, b=1):
            calls["fn_calls"] += 1
            return a + b

        assert f(2, b=3) == 5
        assert f(10) == 11
    finally:
        obs.set_langfuse_enabled(None)

    assert calls["decorate"] == 1
    assert calls["langfuse_wrapper_calls"] == 2
    assert calls["fn_calls"] == 2


def test_observe_noop_when_disabled(monkeypatch):
    from agent import observability as obs

    calls = {"decorate": 0}

    def fake_langfuse_observe(*_args, **_kwargs):
        calls["decorate"] += 1

        def _decorator(fn):
            return fn

        return _decorator

    monkeypatch.setattr(obs, "LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(obs, "_langfuse_observe", fake_langfuse_observe)

    obs.set_langfuse_enabled(False)
    try:
        @obs.observe(name="t")
        def f():
            return "ok"

        assert f() == "ok"
    finally:
        obs.set_langfuse_enabled(None)

    assert calls["decorate"] == 0


def test_observe_thread_safe_single_decoration(monkeypatch):
    from agent import observability as obs

    calls = {"decorate": 0, "wrapped": 0}

    def fake_langfuse_observe(*_args, **_kwargs):
        calls["decorate"] += 1

        def _decorator(fn):
            @functools.wraps(fn)
            def _wrapped(*a, **kw):
                calls["wrapped"] += 1
                return fn(*a, **kw)

            return _wrapped

        return _decorator

    monkeypatch.setattr(obs, "LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(obs, "_langfuse_observe", fake_langfuse_observe)

    obs.set_langfuse_enabled(True)
    try:
        @obs.observe(name="t")
        def f(x):
            return x * 2

        barrier = threading.Barrier(8)
        results = []
        lock = threading.Lock()

        def _worker(i):
            barrier.wait(timeout=2)
            r = f(i)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)

        assert sorted(results) == [i * 2 for i in range(8)]
    finally:
        obs.set_langfuse_enabled(None)

    assert calls["decorate"] == 1
    assert calls["wrapped"] == 8


def test_contextvar_override_cleanup(monkeypatch):
    from agent import observability as obs

    calls = {"decorate": 0, "wrapped": 0, "fn": 0}

    def fake_langfuse_observe(*_args, **_kwargs):
        calls["decorate"] += 1

        def _decorator(fn):
            @functools.wraps(fn)
            def _wrapped(*a, **kw):
                calls["wrapped"] += 1
                return fn(*a, **kw)

            return _wrapped

        return _decorator

    monkeypatch.setattr(obs, "LANGFUSE_AVAILABLE", True)
    monkeypatch.setattr(obs, "_langfuse_observe", fake_langfuse_observe)
    monkeypatch.delenv("HERMES_LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)

    @obs.observe(name="t")
    def f():
        calls["fn"] += 1
        return "ok"

    obs.set_langfuse_enabled(True)
    try:
        assert f() == "ok"
    finally:
        obs.set_langfuse_enabled(None)

    # With override cleared and no env enablement, wrapper should run without Langfuse.
    assert f() == "ok"

    assert calls["decorate"] == 1
    assert calls["wrapped"] == 1
    assert calls["fn"] == 2
