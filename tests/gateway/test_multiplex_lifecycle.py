"""Phase 4: lifecycle guard + per-profile observability."""
import pytest


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

    def test_served_profiles_absent_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib
        import gateway.status as status
        importlib.reload(status)
        try:
            status.write_runtime_status(gateway_state="running")
            rec = status.read_runtime_status()
            assert "served_profiles" not in rec
        finally:
            importlib.reload(status)


class TestNamedProfileMultiplexerGuard:
    """_guard_named_profile_under_multiplexer is inert unless all conditions hold."""

    def test_inert_for_default_profile(self, monkeypatch):
        from hermes_cli import gateway as gw
        monkeypatch.setattr(gw, "_profile_suffix", lambda: "")
        # Should return without raising (default profile => guard N/A).
        gw._guard_named_profile_under_multiplexer(force=False)

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

    def test_env_forces_guard_even_without_config(self, monkeypatch, tmp_path):
        """GATEWAY_MULTIPLEX_PROFILES=true must trip the guard even when the
        default profile's config.yaml has no multiplex_profiles key — the hosted
        case where multiplex is forced purely by the env stamp."""
        from hermes_cli import gateway as gw
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        # No config.yaml written → the only signal is the env override.
        monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "true")
        with pytest.raises(SystemExit):
            gw._guard_named_profile_under_multiplexer(force=False)

    def test_env_false_disables_guard_over_config_true(self, monkeypatch, tmp_path):
        """GATEWAY_MULTIPLEX_PROFILES=false wins over a config.yaml opt-in, so
        the guard stays inert (symmetric with the config precedence)."""
        from hermes_cli import gateway as gw
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "multiplex_profiles: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "false")
        # Env forces OFF → guard must NOT raise.
        gw._guard_named_profile_under_multiplexer(force=False)

    def test_blank_env_falls_through_to_config_and_raises(self, monkeypatch, tmp_path):
        """A blank env value must not shadow a config.yaml opt-in: the guard
        still trips on the config value."""
        from hermes_cli import gateway as gw
        self._fake_running_default_gateway(monkeypatch, tmp_path)
        (tmp_path / "config.yaml").write_text(
            "multiplex_profiles: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "")
        with pytest.raises(SystemExit):
            gw._guard_named_profile_under_multiplexer(force=False)
