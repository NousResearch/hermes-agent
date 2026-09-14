"""The dashboard's auto-archive sweep must not open a writable SessionDB while a
gateway owns the store (#109727): that second connection's close-time checkpoint
tears down the WAL generation the gateway still holds.

These drive the real liveness ladder (``gateway.status.resolve_gateway_liveness``)
rather than mocking the gate's own helper, so the ``probe_error`` / "unknown
ownership" contract is actually exercised.
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


def _forbid_open(monkeypatch, wss):
    monkeypatch.setattr(
        wss, "_open_session_db_for_profile",
        lambda profile, *, read_only: pytest.fail("auto-archive opened a SessionDB"))


class _DB:
    def __init__(self, calls):
        self._calls = calls

    def maybe_auto_archive(self, **kwargs):
        self._calls.append(kwargs)

    def close(self):
        self._calls.append("closed")


def test_stands_down_when_a_gateway_holds_the_pid(serve_home, monkeypatch):
    import gateway.status as status
    import hermes_cli.web_server_sessions as wss

    monkeypatch.setattr(status, "get_running_pid", lambda *a, **k: 4321)
    _forbid_open(monkeypatch, wss)

    wss._maybe_auto_archive_for_profile(None)


def test_unknown_ownership_stands_down(serve_home, monkeypatch):
    """The production path: a rung that RAISES leaves running=False, probe_error=True.

    Reading only ``.running`` (what ``_check_gateway_running`` exposes) would call that
    "no gateway" and open a second writer. Regression for review point 1 on #110405.
    """
    import gateway.status as status
    import hermes_cli.web_server_sessions as wss

    def _boom(*a, **k):
        raise OSError("pid file unreadable")

    monkeypatch.setattr(status, "get_running_pid", _boom)
    monkeypatch.setattr(status, "read_runtime_status", _boom)
    monkeypatch.setattr(status, "get_runtime_status_running_pid", _boom)

    liveness = status.resolve_gateway_liveness(
        profile_dir=serve_home, use_cache=False,
        pid_probe=lambda path: status.get_running_pid(path, cleanup_stale=False))
    assert liveness.running is False and liveness.probe_error is True, "probe must be unknown, not down"

    assert wss._gateway_owns_home(serve_home) is True
    _forbid_open(monkeypatch, wss)
    wss._maybe_auto_archive_for_profile(None)


def test_sweeps_when_no_gateway_is_running(serve_home, monkeypatch):
    """Desktop-only installs still get auto-archive: that is the whole point of the trigger."""
    import hermes_cli.web_server_sessions as wss

    assert wss._gateway_owns_home(serve_home) is False, "clean home must resolve as unowned"

    calls = []
    monkeypatch.setattr(
        wss, "_open_session_db_for_profile", lambda profile, *, read_only: _DB(calls))

    wss._maybe_auto_archive_for_profile(None)

    assert calls and calls[0]["idle_days"] == 3.0
    assert calls[-1] == "closed"


def test_named_satellite_profile_defers_to_the_multiplexer(serve_home, monkeypatch):
    import hermes_cli.profiles as profiles_mod
    import hermes_cli.web_server_cron as wsc
    import hermes_cli.web_server_sessions as wss

    other = serve_home / "profiles" / "work"
    other.mkdir(parents=True)
    monkeypatch.setattr(wsc, "_cron_profile_home", lambda profile: ("work", other))
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda name: True)
    _forbid_open(monkeypatch, wss)

    wss._maybe_auto_archive_for_profile("work")


def test_unresolvable_profile_fails_closed(serve_home, monkeypatch):
    import hermes_cli.web_server_cron as wsc
    import hermes_cli.web_server_sessions as wss

    def _boom(profile):
        raise RuntimeError("profile lookup exploded")

    monkeypatch.setattr(wsc, "_cron_profile_home", _boom)
    _forbid_open(monkeypatch, wss)

    assert wss._auto_archive_owned_by_gateway("work") is True
    wss._maybe_auto_archive_for_profile("work")
