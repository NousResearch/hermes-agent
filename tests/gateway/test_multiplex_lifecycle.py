"""Phase 4: lifecycle guard + per-profile observability."""
import pytest

from gateway.config import GatewayConfig
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE


class TestServedProfilesStatus:
    def test_write_and_read_served_profiles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import gateway.status as status
        importlib.reload(status)
        try:
            status.write_runtime_status(
                gateway_state="running", served_profiles=["default", "coder"]
            )
            rec = status.read_runtime_status()
            assert rec.get("served_profiles") == ["default", "coder"]
        finally:
            importlib.reload(status)


def test_cron_profile_homes_serve_every_live_profile(tmp_path, monkeypatch):
    """The helper wired into in-process cron returns default + every live named profile;
    a tombstoned profile dir is skipped."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    for name in ("worker", "guest", "gone"):
        (default_home / "profiles" / name).mkdir(parents=True)
    from hermes_constants import mark_named_profile_deleted
    mark_named_profile_deleted(default_home / "profiles" / "gone")

    import gateway.run as gateway_run

    homes = gateway_run._multiplex_profile_homes(GatewayConfig(multiplex_profiles=True))

    assert [name for name, _home in homes] == ["default", "guest", "worker"]


def test_cron_tick_homes_include_active_named_host(tmp_path, monkeypatch):
    """A named-profile host running the multiplexer ticks its own store exactly once:
    it is part of the served set, and cron must not union it a second time."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    for name in ("host", "worker"):
        (default_home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home / "profiles" / "host"))

    import gateway.run as gateway_run

    cfg = GatewayConfig(multiplex_profiles=True)
    adapter_names = [name for name, _home in gateway_run._multiplex_profile_homes(cfg)]
    cron_homes = gateway_run._cron_tick_profile_homes(cfg)
    cron_names = [name for name, _home in cron_homes]
    cron_by_name = dict(cron_homes)

    assert adapter_names == ["default", "host", "worker"]
    assert cron_names == ["default", "host", "worker"]
    assert cron_by_name["host"] == default_home / "profiles" / "host"


class TestNamedProfileMultiplexerGuard:
    """_guard_named_profile_under_multiplexer is inert unless all conditions hold."""


    def test_force_bypasses(self, monkeypatch):
        from hermes_cli import gateway as gw
        # Even if it looks like a named profile, force returns immediately.
        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        gw._guard_named_profile_under_multiplexer(force=True)

    def test_inert_when_no_default_gateway_running(self, monkeypatch, tmp_path):
        from hermes_cli import gateway as gw
        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root", lambda: tmp_path
        )
        # No gateway.pid in tmp_path => no running default gateway => no raise.
        gw._guard_named_profile_under_multiplexer(force=False)

    def _fake_running_default_gateway(self, monkeypatch, tmp_path):
        """Make the guard believe a live default gateway exists at tmp_path."""
        from hermes_cli import gateway as gw
        import gateway.status as status

        monkeypatch.setattr(gw, "_profile_suffix", lambda: "coder")
        monkeypatch.setattr(
            "hermes_constants.get_default_hermes_root", lambda: tmp_path
        )
        (tmp_path / "gateway.pid").write_text("12345", encoding="utf-8")
        monkeypatch.setattr(status, "_read_pid_record", lambda p: {"pid": 12345})
        monkeypatch.setattr(status, "_pid_from_record", lambda rec: 12345)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)

    def test_unset_allowlist_preserves_historical_guard(self, monkeypatch, tmp_path):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        with pytest.raises(SystemExit) as excinfo:
            gw._guard_named_profile_under_multiplexer(force=False)
        assert excinfo.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE

    def test_recorded_served_profiles_win_over_config(self, monkeypatch, tmp_path):
        """The live gateway's own ``served_profiles`` record is authoritative: a named profile
        it does not list may run standalone even though the default config multiplexes."""
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n",
            encoding="utf-8",
        )
        import gateway.status as status
        monkeypatch.setattr(
            status, "read_runtime_status",
            lambda path=None: {"gateway_state": "running", "served_profiles": ["default", "worker"]},
        )

        from hermes_cli import gateway as gw

        gw._guard_named_profile_under_multiplexer(force=False)
        assert gw.named_profile_served_by_running_multiplexer("worker") is True

    def test_non_multiplexing_default_gateway_lets_named_profile_run(self, monkeypatch, tmp_path):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: false\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        gw._guard_named_profile_under_multiplexer(force=False)

    def test_named_profile_served_probe_matches_the_start_guard(self, monkeypatch, tmp_path):
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n",
            encoding="utf-8",
        )

        from hermes_cli import gateway as gw

        assert gw.named_profile_served_by_running_multiplexer() is True

        monkeypatch.setattr(gw, "_profile_suffix", lambda: "")
        assert gw.named_profile_served_by_running_multiplexer() is False


