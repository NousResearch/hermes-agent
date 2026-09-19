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


def test_root_warning_uses_gated_warning_callback_when_wired(monkeypatch, capsys):
    """CLI wiring passes agent._emit_warning: the notice goes through that gated sink, not raw stderr."""
    provider = _make_local_embedded_provider(monkeypatch)
    monkeypatch.setattr(hindsight.os, "geteuid", lambda: 0, raising=False)
    seen = []
    provider.initialize(session_id="s1", warning_callback=seen.append, platform="cli")
    assert provider._mode == "disabled"
    assert len(seen) == 1 and "cannot run as root" in seen[0]
    assert "cannot run as root" not in capsys.readouterr().err


@pytest.mark.parametrize("setting", (None, False, True))
def test_root_warning_stderr_fallback_honors_policy(tmp_path, monkeypatch, capsys, setting):
    """No warning_callback (gateway/TUI wiring): the stderr fallback follows the shared boundary;
    the logger.warning is always recorded."""
    import json
    home = tmp_path / f"home-{setting}"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(json.dumps({"display": {} if setting is None else {"suppress_warning_notifications": setting}}))
    provider = _make_local_embedded_provider(monkeypatch)
    monkeypatch.setattr(hindsight.os, "geteuid", lambda: 0, raising=False)
    provider.initialize(session_id="s1", platform="cli")
    assert provider._mode == "disabled"
    assert ("cannot run as root" in capsys.readouterr().err) is (setting is not True)
