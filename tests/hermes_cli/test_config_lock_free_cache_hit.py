"""A cached config read must not block behind a config WRITER.

``_load_config_impl`` served cache hits from inside ``_CONFIG_LOCK``, which
``save_config()`` holds across an atomic YAML write. A cache hit itself costs
~0.024ms-scale work, but measured on a clean tree the same cached read took
**10010ms** while another thread held the lock.

On a gateway that lands on the event loop: the per-message hook path
(``invoke_hook`` -> ``_resolve_hook_callback_timeout``) reads config, so one
background config write stalls every inbound message for its whole duration.

These tests drive the real functions against a temp HERMES_HOME -- no mocks of
the thing under test, no source reading.
"""

from __future__ import annotations

import os
import threading
import time

import pytest


# Generous: the point is 10s-vs-instant, not a tight timing assertion.
MAX_BLOCKED_READ_SECS = 2.0


@pytest.fixture()
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    lines = ["plugins:", "  hook_callback_timeout: 30", "agent:", "  max_turns: 500"]
    for i in range(100):
        lines += [f"section_{i}:", f"  key_a: value_{i}"]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import config as cfgmod

    cfgmod._LOAD_CONFIG_CACHE.clear()
    cfgmod._RAW_CONFIG_CACHE.clear()
    yield home
    cfgmod._LOAD_CONFIG_CACHE.clear()
    cfgmod._RAW_CONFIG_CACHE.clear()


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _time_cached_read_under_held_lock(reader):
    """Prime ``reader`` (lock-free), then time one more call while another thread holds
    ``_CONFIG_LOCK``. Returns ``(result, elapsed_seconds)``."""
    from hermes_cli import config as cfgmod

    reader()  # prime the cache

    holding = threading.Event()
    release = threading.Event()

    def _hold():
        with cfgmod._CONFIG_LOCK:
            holding.set()
            release.wait(timeout=30)

    thread = threading.Thread(target=_hold, daemon=True)
    thread.start()
    try:
        assert holding.wait(timeout=10), "lock holder never started"
        started = time.perf_counter()
        result = reader()
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        thread.join(timeout=10)
    return result, elapsed


def test_cached_read_completes_while_another_thread_holds_the_config_lock(config_home):
    from hermes_cli import config as cfgmod

    cfg, elapsed = _time_cached_read_under_held_lock(cfgmod.load_config_readonly)

    assert isinstance(cfg, dict) and cfg
    assert elapsed < MAX_BLOCKED_READ_SECS, (
        f"a cached load_config_readonly() blocked {elapsed:.1f}s behind a held "
        "_CONFIG_LOCK; cache hits must not serialize against config writers"
    )


def test_cached_raw_read_completes_while_another_thread_holds_the_config_lock(config_home):
    """Twin of the load_config test for the sibling ``read_raw_config`` fast path."""
    from hermes_cli import config as cfgmod

    raw, elapsed = _time_cached_read_under_held_lock(cfgmod.read_raw_config_readonly)

    assert isinstance(raw, dict) and raw.get("plugins") == {"hook_callback_timeout": 30}
    assert elapsed < MAX_BLOCKED_READ_SECS, (
        f"a cached read_raw_config_readonly() blocked {elapsed:.1f}s behind a held "
        "_CONFIG_LOCK; cache hits must not serialize against config writers"
    )


def test_hook_timeout_memo_never_pairs_a_new_sig_with_an_old_value(config_home):
    """The memo is published as ONE ``(sig, value)`` tuple: with a writer thread flipping the config
    between two timeouts, every lock-free read must see sig and value from the SAME publish."""
    from hermes_cli import plugins as pluginsmod
    from hermes_cli.config import _load_config_cache_sig, get_config_path

    def write(v):
        (config_home / "config.yaml").write_text(f"plugins:\n  hook_callback_timeout: {v}\n", encoding="utf-8")
        _bump_mtime(get_config_path())
        return pluginsmod._resolve_hook_callback_timeout(), _load_config_cache_sig(get_config_path())[1]

    sig_of = {}
    for v in (5, 7):
        value, sig = write(v)
        assert value == float(v)
        sig_of[sig] = value
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            write(5); write(7)

    t = threading.Thread(target=writer, daemon=True); t.start()
    try:
        for _ in range(2000):
            for sig, value in list(pluginsmod._HOOK_TIMEOUT_CACHE.values()):
                if sig in sig_of:
                    assert sig_of[sig] == value, "memo published a new sig with a stale value"
    finally:
        stop.set(); t.join(timeout=5)
