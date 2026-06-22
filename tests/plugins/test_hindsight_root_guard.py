"""Root-user guard for Hindsight local_embedded mode (issue #13125).

PostgreSQL's initdb refuses to run as root, so the embedded Hindsight daemon
can never initialize under root — without a guard it crash-restart loops
forever, burning RAM/CPU with no user-visible error. initialize() must detect
root up front, skip daemon startup, disable the provider, and warn the user.
"""

import importlib
import threading

import pytest

hindsight = importlib.import_module("plugins.memory.hindsight")
HindsightMemoryProvider = hindsight.HindsightMemoryProvider


def _make_local_embedded_provider(monkeypatch):
    """Build a provider wired for local_embedded with a passing runtime probe."""
    monkeypatch.setattr(
        hindsight,
        "_load_config",
        lambda: {"mode": "local_embedded", "profile": "hermes"},
    )
    # Pretend the local runtime imports cleanly so initialize() reaches the
    # daemon-start branch instead of bailing on a missing `hindsight` package.
    monkeypatch.setattr(hindsight, "_check_local_runtime", lambda: (True, None))
    return HindsightMemoryProvider()


def _daemon_threads_alive() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name == "hindsight-daemon-start"]


def test_local_embedded_skips_daemon_as_root(monkeypatch, caplog):
    """As root, the daemon thread must NOT start and the mode is disabled."""
    provider = _make_local_embedded_provider(monkeypatch)
    monkeypatch.setattr(hindsight.os, "geteuid", lambda: 0, raising=False)

    # If the guard fails, _start_daemon would call _get_client() — make that
    # explode so a regression is loud rather than silently spawning a thread.
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: pytest.fail("daemon startup attempted while running as root"),
    )

    before = set(_daemon_threads_alive())
    with caplog.at_level("WARNING", logger="plugins.memory.hindsight"):
        provider.initialize(session_id="s1")

    assert provider._mode == "disabled"
    assert set(_daemon_threads_alive()) == before  # no new daemon thread
    # The warning is surfaced to the user via the logger AND printed to
    # stderr (E2E-verified in tests/plugins/test_hindsight_root_guard.py
    # docstring rationale); capsys can't reliably capture the module-level
    # sys.stderr write under the isolation harness, so assert on the log.
    assert any("cannot run as root" in r.message for r in caplog.records)


def test_local_embedded_starts_daemon_as_non_root(monkeypatch):
    """As a non-root user, the daemon-start thread IS spawned."""
    provider = _make_local_embedded_provider(monkeypatch)
    monkeypatch.setattr(hindsight.os, "geteuid", lambda: 1000, raising=False)

    started = threading.Event()
    monkeypatch.setattr(
        hindsight.threading,
        "Thread",
        _fake_thread_factory(started),
    )

    provider.initialize(session_id="s1")

    assert provider._mode == "local_embedded"
    assert started.is_set()


def _fake_thread_factory(started: threading.Event):
    """Return a Thread replacement that records start() without running work."""
    real_thread = threading.Thread

    def _factory(*args, **kwargs):
        if kwargs.get("name") == "hindsight-daemon-start":
            started.set()

            class _NoopThread:
                def start(self):
                    pass

            return _NoopThread()
        return real_thread(*args, **kwargs)

    return _factory
