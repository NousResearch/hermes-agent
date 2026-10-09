"""Config migrations that retire or rewrite obsolete keys (v43 / v44 / v51)."""

import hermes_yaml as yaml


class TestRetiredMultiplexAllowlist:
    def test_v43_drops_multiplex_profile_allowlist_from_user_config(self, tmp_path, monkeypatch):
        """The multiplexer serves every profile; a stale allowlist must not linger in config.yaml."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 42,
            "gateway": {"multiplex_profiles": True, "multiplex_profile_allowlist": ["worker"]},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(42, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "multiplex_profile_allowlist" not in raw["gateway"]
        assert raw["gateway"]["multiplex_profiles"] is True
        assert "multiplex_profile_allowlist" not in DEFAULT_CONFIG["gateway"]


class TestCuratorFasterPrune:
    def test_v44_rewrites_old_curator_defaults_but_keeps_user_values(self, tmp_path, monkeypatch):
        """Old 30/90 defaults move to 14/30; an explicitly customized window is untouched."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 43,
            "curator": {"stale_after_days": 30, "archive_after_days": 180},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(43, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["curator"]["stale_after_days"] == DEFAULT_CONFIG["curator"]["stale_after_days"]
        assert raw["curator"]["archive_after_days"] == 180


class TestRetiredBotChatDeliveryTimeout:
    def test_v51_drops_bot_chat_delivery_timeout_with_a_note(self, tmp_path, monkeypatch):
        """The removed cron knob is dropped from existing configs with a one-time note;
        sibling cron settings and the rest of the file survive untouched."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 50,
            "cron": {"bot_chat_delivery_timeout_seconds": 900, "max_parallel_jobs": 2},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        results = {"env_added": [], "config_added": [], "warnings": []}
        run_migrations(50, results, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "bot_chat_delivery_timeout_seconds" not in raw["cron"]
        assert raw["cron"]["max_parallel_jobs"] == 2
        assert any("bot_chat_delivery_timeout_seconds" in note for note in results["config_added"])
        assert "bot_chat_delivery_timeout_seconds" not in DEFAULT_CONFIG["cron"]
