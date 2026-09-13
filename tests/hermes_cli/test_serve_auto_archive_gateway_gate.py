"""The dashboard's auto-archive sweep must not open a writable SessionDB while a
gateway owns the store (#109727): that second connection's close-time checkpoint
tears down the WAL generation the gateway still holds.
"""
from pathlib import Path

import pytest


@pytest.fixture
def serve_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "sessions:\n  auto_archive: true\n  auto_archive_days: 3\n", encoding="utf-8")
    import hermes_cli.web_server_sessions as wss

    wss._last_auto_archive_check.clear()
    return tmp_path


def _spy_open(monkeypatch, wss):
    opened = []
    monkeypatch.setattr(
        wss, "_open_session_db_for_profile",
        lambda profile, *, read_only: opened.append((profile, read_only)) or pytest.fail(
            "auto-archive opened a SessionDB"))
    return opened


def test_stands_down_when_gateway_owns_the_profile(serve_home, monkeypatch):
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.web_server_sessions as wss

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: True)
    _spy_open(monkeypatch, wss)

    wss._maybe_auto_archive_for_profile(None)  # must not raise, must not open


def test_stands_down_when_probe_fails(serve_home, monkeypatch):
    """Unknown liveness fails closed — better a skipped sweep than a torn WAL."""
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.web_server_sessions as wss

    def _boom(home):
        raise OSError("pid file unreadable")

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", _boom)
    _spy_open(monkeypatch, wss)

    wss._maybe_auto_archive_for_profile(None)


def test_sweeps_when_no_gateway_is_running(serve_home, monkeypatch):
    """Desktop-only installs still get auto-archive: that is the whole point of the trigger."""
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.web_server_sessions as wss

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: False)

    calls = []

    class _DB:
        def maybe_auto_archive(self, **kwargs):
            calls.append(kwargs)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(
        wss, "_open_session_db_for_profile", lambda profile, *, read_only: _DB())

    wss._maybe_auto_archive_for_profile(None)

    assert calls and calls[0]["idle_days"] == 3.0
    assert calls[-1] == "closed"


def test_named_profile_gate_uses_its_own_home(serve_home, monkeypatch):
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.web_server_cron as wsc
    import hermes_cli.web_server_sessions as wss

    other = serve_home / "profiles" / "work"
    other.mkdir(parents=True)
    monkeypatch.setattr(wsc, "_cron_profile_home", lambda profile: ("work", other))

    seen = []
    monkeypatch.setattr(
        profiles_mod, "_check_gateway_running", lambda home: seen.append(Path(home)) or False)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda name: True)
    _spy_open(monkeypatch, wss)

    # Satellite profile with no gateway.pid of its own is still owned by the live multiplexer.
    wss._maybe_auto_archive_for_profile("work")

    assert seen == [other]
