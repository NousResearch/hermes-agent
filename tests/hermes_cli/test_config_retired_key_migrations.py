"""Config migrations that retire or rewrite obsolete keys (v43 / v44 / v51, stamp-independent steps)."""

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


class TestRetiredKeysWhateverTheStamp:
    """Retired-key steps must not depend on a stamp that does not prove they ran: an unversioned
    file (seeded from an older template) and a home an unreleased branch build stamped with its
    own step numbers both still carry keys nothing reads."""

    def test_unversioned_config_drops_the_retired_bot_chat_timeout(self, tmp_path, monkeypatch):
        from hermes_cli.config import migrate_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "cron": {"bot_chat_delivery_timeout_seconds": 900, "max_parallel_jobs": 2},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["cron"] == {"max_parallel_jobs": 2}

    def test_branch_stamped_home_still_gets_main_retired_key_steps(self, tmp_path, monkeypatch):
        """A home stamped 50 by a build whose v50 was the cron-key step skipped main's v46 (MCP
        `disabled: true`) and v50 (tirith keys). Both apply, and a second pass writes nothing."""
        from hermes_cli.config import migrate_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 50,
            "mcp_servers": {"old": {"command": "x", "enabled": True, "disabled": True}},
            "security": {"redact_secrets": True, "tirith_enabled": True},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["mcp_servers"]["old"] == {"command": "x", "enabled": False}
        assert raw["security"] == {"redact_secrets": True}
        settled = config_path.read_bytes()
        assert migrate_config(interactive=False, quiet=True)["config_added"] == []
        assert config_path.read_bytes() == settled
        from hermes_cli.config_migrations import LEGACY_KEY_STEPS, STAMP_INDEPENDENT_STEPS
        assert STAMP_INDEPENDENT_STEPS <= LEGACY_KEY_STEPS
