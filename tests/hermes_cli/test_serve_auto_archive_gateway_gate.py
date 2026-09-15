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


def _only_default_gateway_is_live(monkeypatch, default_root):
    """Live gateway.pid for the DEFAULT root only; satellites have none of their own."""
    import gateway.status as status

    default_pid = Path(default_root) / "gateway.pid"
    monkeypatch.setattr(
        status, "get_running_pid",
        lambda path=None, *a, **k: 999 if path is not None and Path(path) == default_pid else None)


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
    import hermes_cli.gateway_multiplex_served as served
    import hermes_cli.web_server_cron as wsc
    import hermes_cli.web_server_sessions as wss

    other = serve_home / "profiles" / "work"
    other.mkdir(parents=True)
    monkeypatch.setattr(wsc, "_cron_profile_home", lambda profile: ("work", other))
    _only_default_gateway_is_live(monkeypatch, serve_home)
    monkeypatch.setattr(served, "recorded_served_profiles", lambda *a, **k: ["work"])
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


class TestSatelliteMultiplexerOwnership:
    """``_served_by_running_multiplexer`` turns every probe failure into False, so a malformed
    PID/runtime record under a LIVE default gateway reads as 'nobody serves this profile' and the
    dashboard opens a second writer into a store the multiplexer holds. Review P2 on #110405."""

    @staticmethod
    def _satellite(monkeypatch, serve_home):
        import hermes_cli.web_server_cron as wsc

        sat = serve_home / "profiles" / "work"
        sat.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(wsc, "_cron_profile_home", lambda profile: ("work", sat))
        return sat

    def test_unreadable_served_record_under_a_live_multiplexer_stands_down(
            self, serve_home, monkeypatch):
        import hermes_cli.gateway_multiplex_served as served
        import hermes_cli.web_server_sessions as wss

        self._satellite(monkeypatch, serve_home)
        # Default multiplexer is live, satellite has no gateway.pid of its own...
        _only_default_gateway_is_live(monkeypatch, serve_home)
        # ...but its served record is unreadable/malformed.
        monkeypatch.setattr(served, "recorded_served_profiles", lambda *a, **k: None)
        _forbid_open(monkeypatch, wss)

        assert wss._auto_archive_owned_by_gateway("work") is True
        wss._maybe_auto_archive_for_profile("work")

    def test_raising_probe_stands_down(self, serve_home, monkeypatch):
        import hermes_cli.gateway_multiplex_served as served
        import hermes_cli.web_server_sessions as wss

        self._satellite(monkeypatch, serve_home)
        _only_default_gateway_is_live(monkeypatch, serve_home)

        def _boom(*a, **k):
            raise OSError("gateway_state.json is malformed")

        monkeypatch.setattr(served, "recorded_served_profiles", _boom)
        _forbid_open(monkeypatch, wss)

        assert wss._auto_archive_owned_by_gateway("work") is True

    def test_live_multiplexer_that_does_not_serve_it_is_not_owner(self, serve_home, monkeypatch):
        """Fail-closed must not become fail-always: an authoritative list that omits the profile
        is a definite 'not served', and that store still needs its dashboard sweep."""
        import hermes_cli.gateway_multiplex_served as served
        import hermes_cli.web_server_sessions as wss

        self._satellite(monkeypatch, serve_home)
        _only_default_gateway_is_live(monkeypatch, serve_home)
        monkeypatch.setattr(served, "recorded_served_profiles", lambda *a, **k: ["other"])

        assert wss._auto_archive_owned_by_gateway("work") is False

    def test_served_profile_is_owned(self, serve_home, monkeypatch):
        import hermes_cli.gateway_multiplex_served as served
        import hermes_cli.web_server_sessions as wss

        self._satellite(monkeypatch, serve_home)
        _only_default_gateway_is_live(monkeypatch, serve_home)
        monkeypatch.setattr(served, "recorded_served_profiles", lambda *a, **k: ["work"])

        assert wss._auto_archive_owned_by_gateway("work") is True
