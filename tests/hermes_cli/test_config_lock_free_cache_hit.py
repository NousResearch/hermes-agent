"""A cached config read must not block behind a config WRITER.

``_load_config_impl`` served cache hits from inside ``_CONFIG_LOCK``, which
``save_config()`` holds across an atomic YAML write. A cache hit itself costs
~0.024us-scale work, but measured on a clean tree the same cached read took
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
    yield home
    cfgmod._LOAD_CONFIG_CACHE.clear()


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_cached_read_completes_while_another_thread_holds_the_config_lock(config_home):
    from hermes_cli import config as cfgmod

    cfgmod.load_config_readonly()  # prime the cache

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
        cfg = cfgmod.load_config_readonly()
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        thread.join(timeout=10)

    assert isinstance(cfg, dict) and cfg
    assert elapsed < MAX_BLOCKED_READ_SECS, (
        f"a cached load_config_readonly() blocked {elapsed:.1f}s behind a held "
        "_CONFIG_LOCK; cache hits must not serialize against config writers"
    )


def test_deepcopy_read_also_completes_under_a_held_lock(config_home):
    from hermes_cli import config as cfgmod

    cfgmod.load_config()

    holding = threading.Event()
    release = threading.Event()

    def _hold():
        with cfgmod._CONFIG_LOCK:
            holding.set()
            release.wait(timeout=30)

    thread = threading.Thread(target=_hold, daemon=True)
    thread.start()
    try:
        assert holding.wait(timeout=10)
        started = time.perf_counter()
        cfg = cfgmod.load_config()
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        thread.join(timeout=10)

    assert cfg["agent"]["max_turns"] == 500
    assert elapsed < MAX_BLOCKED_READ_SECS


def test_fast_path_still_sees_a_changed_config_file(config_home):
    """Freshness is not sacrificed for the lock-free read."""
    from hermes_cli import config as cfgmod

    assert cfgmod.load_config_readonly()["agent"]["max_turns"] == 500

    path = config_home / "config.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("max_turns: 500", "max_turns: 123"),
        encoding="utf-8",
    )
    _bump_mtime(path)

    assert cfgmod.load_config_readonly()["agent"]["max_turns"] == 123


def test_load_config_still_returns_an_isolated_object(config_home):
    """The deepcopy contract must survive the fast path."""
    from hermes_cli import config as cfgmod

    first = cfgmod.load_config()
    first["agent"]["max_turns"] = -1
    assert cfgmod.load_config()["agent"]["max_turns"] == 500
    assert cfgmod.load_config_readonly()["agent"]["max_turns"] == 500


def test_concurrent_readers_and_a_writer_never_see_a_torn_config(config_home):
    """The fast path reads a tuple that is replaced wholesale, never mutated."""
    from hermes_cli import config as cfgmod

    cfgmod.load_config_readonly()
    stop = threading.Event()
    errors: list[BaseException] = []
    seen: list[int] = []

    def _writer():
        try:
            while not stop.is_set():
                cfg = cfgmod.load_config()
                cfg.setdefault("agent", {})["max_turns"] = 500
                cfgmod.save_config(cfg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _reader():
        try:
            for _ in range(300):
                cfg = cfgmod.load_config_readonly()
                # Every observation must be a complete, well-formed config.
                assert isinstance(cfg, dict)
                assert isinstance(cfg.get("agent"), dict)
                seen.append(int(cfg["agent"]["max_turns"]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writer = threading.Thread(target=_writer, daemon=True)
    readers = [threading.Thread(target=_reader, daemon=True) for _ in range(4)]
    writer.start()
    for r in readers:
        r.start()
    for r in readers:
        r.join(timeout=60)
    stop.set()
    writer.join(timeout=30)

    assert not errors, f"torn/failed read under concurrency: {errors[0]!r}"
    assert seen, "readers observed nothing"


# ---------------------------------------------------------------------------
# Hook-callback timeout memoization.
# ---------------------------------------------------------------------------


def test_hook_timeout_does_not_read_config_on_every_invocation(config_home):
    from hermes_cli import config as cfgmod
    from hermes_cli import plugins as pluginsmod

    pluginsmod._reset_hook_callback_timeout_cache()
    assert pluginsmod._resolve_hook_callback_timeout() == 30.0

    calls = {"n": 0}
    real = cfgmod.load_config_readonly

    def counting():
        calls["n"] += 1
        return real()

    cfgmod.load_config_readonly = counting  # type: ignore[assignment]
    try:
        for _ in range(100):
            pluginsmod._resolve_hook_callback_timeout()
    finally:
        cfgmod.load_config_readonly = real  # type: ignore[assignment]

    assert calls["n"] == 0, (
        f"hook-timeout resolution read the config {calls['n']}x across 100 hook "
        "invocations; a gateway fires hooks per inbound message, so this is a "
        "per-message config read on the event loop"
    )


def test_hook_timeout_picks_up_a_changed_config(config_home):
    from hermes_cli import plugins as pluginsmod

    pluginsmod._reset_hook_callback_timeout_cache()
    assert pluginsmod._resolve_hook_callback_timeout() == 30.0

    path = config_home / "config.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "hook_callback_timeout: 30", "hook_callback_timeout: 45"
        ),
        encoding="utf-8",
    )
    _bump_mtime(path)

    assert pluginsmod._resolve_hook_callback_timeout() == 45.0


@pytest.mark.parametrize(
    "raw,expected_attr",
    [
        ("9999", "_MAX_HOOK_CALLBACK_TIMEOUT_SECS"),
        ("-5", "_HOOK_CALLBACK_TIMEOUT_SECS"),
        ("'not-a-number'", "_HOOK_CALLBACK_TIMEOUT_SECS"),
    ],
)
def test_hook_timeout_validation_survives_memoization(config_home, raw, expected_attr):
    """Clamping and the fallback must not be lost to the cache."""
    import re

    from hermes_cli import plugins as pluginsmod

    path = config_home / "config.yaml"
    path.write_text(
        re.sub(
            r"  hook_callback_timeout: .*",
            f"  hook_callback_timeout: {raw}",
            path.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    _bump_mtime(path)
    pluginsmod._reset_hook_callback_timeout_cache()

    assert pluginsmod._resolve_hook_callback_timeout() == getattr(
        pluginsmod, expected_attr
    )


def test_hook_timeout_zero_disables_and_is_cached(config_home):
    """``<= 0`` means 'no threaded timeout' -- a falsy value the memo must
    still return rather than treating as 'unresolved'."""
    import re

    from hermes_cli import plugins as pluginsmod

    path = config_home / "config.yaml"
    path.write_text(
        re.sub(
            r"  hook_callback_timeout: .*",
            "  hook_callback_timeout: 0",
            path.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    _bump_mtime(path)
    pluginsmod._reset_hook_callback_timeout_cache()

    assert pluginsmod._resolve_hook_callback_timeout() == 0.0
    assert pluginsmod._resolve_hook_callback_timeout() == 0.0
