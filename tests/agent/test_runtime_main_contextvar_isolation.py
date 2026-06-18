"""Regression tests for per-task ContextVar isolation of auxiliary_client
runtime-main credentials (PRV-F1) and the auxiliary_is_nous flag (PRV-F3).

Before the migration these were unlocked module globals, so two concurrent
in-process subagents (delegation worker threads) could read each other's
api_key/base_url or Nous flag. Each test below sets distinct values from two
threads behind a barrier (so both writes land before either read) — they pass
only because each thread reads its OWN ContextVar; a shared global would make
the last writer win and both reads return the same value.
"""
import threading

from agent import auxiliary_client as ac


def _run_two(worker):
    results = {}
    barrier = threading.Barrier(2)

    def wrap(name, *args):
        worker(name, barrier, results, *args)

    t1 = threading.Thread(target=wrap, args=("a", "KEY_A", "http://a", True))
    t2 = threading.Thread(target=wrap, args=("b", "KEY_B", "http://b", False))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results


def test_runtime_main_isolated_across_threads():
    def worker(name, barrier, results, key, base, _is_nous):
        ac.set_runtime_main(
            "custom", "model-x", base_url=base, api_key=key, api_mode="chat_completions"
        )
        barrier.wait()  # both threads have set before either reads
        results[name + ":key"] = ac._runtime_main_field("api_key", ac._RUNTIME_MAIN_API_KEY)
        results[name + ":base"] = ac._runtime_main_field("base_url", ac._RUNTIME_MAIN_BASE_URL)

    r = _run_two(worker)
    assert r["a:key"] == "KEY_A"
    assert r["b:key"] == "KEY_B"
    assert r["a:base"] == "http://a"
    assert r["b:base"] == "http://b"


def test_auxiliary_is_nous_isolated_across_threads():
    def worker(name, barrier, results, _key, _base, is_nous):
        ac._set_auxiliary_is_nous(is_nous)
        barrier.wait()
        results[name] = ac._get_auxiliary_is_nous()

    r = _run_two(worker)
    assert r["a"] is True
    assert r["b"] is False


def test_runtime_main_falls_back_to_global_default_when_unset():
    """A thread that never called set_runtime_main has an unset ContextVar, so
    the reader returns the supplied global default (backward-compatible path)."""
    out = {}

    def worker():
        out["v"] = ac._runtime_main_field("api_key", "GLOBAL_DEFAULT")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert out["v"] == "GLOBAL_DEFAULT"


def test_clear_runtime_main_resets_contextvar():
    ac.set_runtime_main("custom", "m", api_key="SECRET")
    assert ac._runtime_main_field("api_key", "") == "SECRET"
    ac.clear_runtime_main()
    assert ac._runtime_main_field("api_key", "") == ""
