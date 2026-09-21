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
    # The gate derives the profile home from _default_db_path(), which honours a re-pointed
    # hermes_state.DEFAULT_DB_PATH. A sibling test that re-points it would otherwise send this
    # one's config lookup to another directory, so pin it to this tmp home.
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
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


class TestStrictIdentityProbe:
    """`get_running_pid()` normalises unreadable identity metadata to None, so an ACTIVE lock with
    corrupt records reported running=False, probe_error=False and the gate opened a second writer.
    Review P1 on #110405. These use REAL unreadable metadata, not a mocked helper."""

    @staticmethod
    def _plant_unreadable_lock(home: Path):
        import json
        import os

        (home / "gateway.pid").write_text(json.dumps({"pid": 4321}), encoding="utf-8")
        lock = home / "gateway.lock"
        lock.write_text("{}", encoding="utf-8")
        os.chmod(lock, 0o000)
        return lock

    @pytest.mark.skipif(hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
                        reason="root bypasses the permission bits this relies on")
    def test_unreadable_lock_metadata_is_unknown_not_absent(self, serve_home):
        import os

        from gateway.status import get_running_pid, get_running_pid_identity_strict
        import hermes_cli.web_server_sessions as wss

        lock = self._plant_unreadable_lock(serve_home)
        try:
            # The strict probe refuses to call unreadable metadata "absent"...
            with pytest.raises(RuntimeError):
                get_running_pid_identity_strict(serve_home / "gateway.pid")

            # ...and the gate follows it.
            assert wss._gateway_owns_home(serve_home) is True, \
                "unreadable identity metadata must count as owned, not absent"
        finally:
            if lock.exists():
                os.chmod(lock, 0o600)

        # The production normalisation the review pointed at, asserted LAST: the non-strict probe
        # reports plain absence for the same state — and unlinks the lock on the way past, even
        # with cleanup_stale=False, which is why this cannot run before the assertions above.
        lock = self._plant_unreadable_lock(serve_home)
        try:
            assert get_running_pid(serve_home / "gateway.pid", cleanup_stale=False) is None
        finally:
            if lock.exists():
                os.chmod(lock, 0o600)

    @pytest.mark.skipif(hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
                        reason="root bypasses the permission bits this relies on")
    def test_the_sweep_stands_down_on_unreadable_metadata(self, serve_home, monkeypatch):
        import os

        import hermes_cli.web_server_sessions as wss

        lock = self._plant_unreadable_lock(serve_home)
        try:
            _forbid_open(monkeypatch, wss)
            wss._maybe_auto_archive_for_profile(None)
        finally:
            if lock.exists():
                os.chmod(lock, 0o600)


class TestCrossContainerHealthRung:
    """In a split gateway/dashboard deployment GATEWAY_HEALTH_URL can be the ONLY evidence the
    gateway is live — local PID and runtime files are absent entirely. Review P1 on #110405."""

    def test_remote_health_alone_establishes_ownership(self, serve_home, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.web_server_gateway as wsg
        import hermes_cli.web_server_sessions as wss

        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gateway:8642", raising=False)
        monkeypatch.setattr(wsg, "_probe_gateway_health", lambda: (True, {"ok": True}))

        assert wss._gateway_owns_home(serve_home) is True, \
            "a live remote gateway owns the shared store even with no local PID files"
        _forbid_open(monkeypatch, wss)
        wss._maybe_auto_archive_for_profile(None)

    def test_a_configured_probe_that_cannot_confirm_fails_closed(self, serve_home, monkeypatch):
        """_probe_gateway_health collapses DNS/timeout/refused/non-200 into (False, None), which is
        indistinguishable from 'the gateway is down'. Configured-but-unconfirmed must not license
        a second writer."""
        import hermes_cli.web_server as ws
        import hermes_cli.web_server_gateway as wsg
        import hermes_cli.web_server_sessions as wss

        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "http://gateway:8642", raising=False)
        monkeypatch.setattr(wsg, "_probe_gateway_health", lambda: (False, None))

        assert wss._gateway_owns_home(serve_home) is True

    def test_no_health_url_configured_leaves_the_local_answer_alone(self, serve_home, monkeypatch):
        """Fail-closed must not become fail-always for the ordinary single-host install."""
        import hermes_cli.web_server as ws
        import hermes_cli.web_server_sessions as wss

        monkeypatch.setattr(ws, "_GATEWAY_HEALTH_URL", "", raising=False)

        assert wss._gateway_owns_home(serve_home) is False
