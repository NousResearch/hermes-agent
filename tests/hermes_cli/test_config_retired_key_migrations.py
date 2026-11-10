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


#: One input per stamp-independent step that its trigger key makes it change.
_STAMP_INDEPENDENT_TRIGGERS = {
    46: {"mcp_servers": {"s": {"command": "x", "disabled": True}}},
    50: {"security": {"redact_secrets": True, "tirith_enabled": True}},
}


class TestStampIndependentStepsAreIdempotent:
    """These steps run on EVERY pass, and a step that stopped the ladder is retried on every start,
    so a second run over a step's own output must change nothing."""

    def test_every_stamp_independent_step_has_a_trigger_here(self):
        from hermes_cli.config_migrations import STAMP_INDEPENDENT_STEPS
        assert set(_STAMP_INDEPENDENT_TRIGGERS) == STAMP_INDEPENDENT_STEPS

    def test_second_run_changes_nothing(self, tmp_path, monkeypatch):
        from hermes_cli.config_migrations import MIGRATIONS, STAMP_INDEPENDENT_STEPS

        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for version, step in ((v, fn) for v, fn in MIGRATIONS if v in STAMP_INDEPENDENT_STEPS):
            config_path.write_text(yaml.safe_dump(
                {"_config_version": version - 1, **_STAMP_INDEPENDENT_TRIGGERS[version]}), encoding="utf-8")
            before = config_path.read_bytes()
            first = {"env_added": [], "config_added": [], "warnings": []}
            step(first, True)
            settled = config_path.read_bytes()
            assert settled != before and first["config_added"], f"v{version}'s trigger did not trigger it"
            second = {"env_added": [], "config_added": [], "warnings": []}
            step(second, True)
            assert config_path.read_bytes() == settled, f"v{version} changed its own output"
            assert second["config_added"] == [] and second["warnings"] == []


class TestNewerStamp:
    """A config stamped by a newer build (rollback, mixed desktop/CLI builds) is read as it is: no
    step runs on it, unknown keys stay, whole-file saves are refused, key-level writes still save."""

    def _newer(self, tmp_path, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": DEFAULT_CONFIG["_config_version"] + 7,
            "mcp_servers": {"s": {"command": "x", "disabled": True}},
            "security": {"tirith_enabled": True},
            "zz_newer_section": {"k": 1},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        return config_path

    def test_migrate_runs_no_step_on_it(self, tmp_path, monkeypatch):
        from hermes_cli.config import migrate_config

        config_path = self._newer(tmp_path, monkeypatch)
        before = config_path.read_bytes()
        results = migrate_config(interactive=False, quiet=True)
        assert config_path.read_bytes() == before
        assert "failed_step" not in results, "no step may even start on a newer file"
        assert any("newer Hermes" in w for w in results["warnings"])

    def test_whole_file_save_is_refused_and_key_level_write_is_kept(self, tmp_path, monkeypatch):
        import pytest
        from hermes_cli.config import load_config, save_config, set_config_value

        config_path = self._newer(tmp_path, monkeypatch)
        before = config_path.read_bytes()
        with pytest.raises(RuntimeError, match="newer Hermes"):
            save_config(load_config())
        assert config_path.read_bytes() == before

        set_config_value("display.personality", "pirate")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["display"]["personality"] == "pirate"
        assert raw["_config_version"] == yaml.safe_load(before)["_config_version"]
        assert raw["zz_newer_section"] == {"k": 1}
        assert raw["mcp_servers"]["s"]["disabled"] is True
