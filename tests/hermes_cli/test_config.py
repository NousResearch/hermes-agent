"""Tests for hermes_cli configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    check_config_version,
    get_hermes_home,
    ensure_hermes_home,
    get_compatible_custom_providers,
    load_config,
    load_env,
    migrate_config,
    remove_env_value,
    save_config,
    save_env_value,
    save_env_value_secure,
    sanitize_env_file,
    _sanitize_env_lines,
)


class TestGetHermesHome:
    def test_default_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HOME", None)
            home = get_hermes_home()
            assert home == Path.home() / ".hermes"

    def test_env_override(self):
        with patch.dict(os.environ, {"HERMES_HOME": "/custom/path"}):
            home = get_hermes_home()
            assert home == Path("/custom/path")


class TestEnsureHermesHome:
    def test_creates_subdirs(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            ensure_hermes_home()
            assert (tmp_path / "cron").is_dir()
            assert (tmp_path / "sessions").is_dir()
            assert (tmp_path / "logs").is_dir()
            assert (tmp_path / "memories").is_dir()

    def test_creates_default_soul_md_if_missing(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            ensure_hermes_home()
            soul_path = tmp_path / "SOUL.md"
            assert soul_path.exists()
            assert soul_path.read_text(encoding="utf-8").strip() != ""

    def test_does_not_overwrite_existing_soul_md(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            soul_path = tmp_path / "SOUL.md"
            soul_path.write_text("custom soul", encoding="utf-8")
            ensure_hermes_home()
            assert soul_path.read_text(encoding="utf-8") == "custom soul"


class TestLoadConfigDefaults:
    def test_returns_defaults_when_no_file(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            assert config["model"] == DEFAULT_CONFIG["model"]
            assert config["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
            assert "max_turns" not in config
            assert "terminal" in config
            assert config["terminal"]["backend"] == "local"
            assert config["display"]["interim_assistant_messages"] is True

    def test_budget_checkpointing_defaults_are_disabled(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            budget_cfg = config["agent"]["budget_checkpointing"]

            assert budget_cfg == {
                "enabled": False,
                "warning_ratio": 0.75,
                "checkpoint_ratio": 0.88,
                "mode": "continuation_packet",
            }
            assert DEFAULT_CONFIG["agent"]["budget_checkpointing"] == budget_cfg

    def test_legacy_root_level_max_turns_migrates_to_agent_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config_path = tmp_path / "config.yaml"
            config_path.write_text("max_turns: 42\n")

            config = load_config()
            assert config["agent"]["max_turns"] == 42
            assert "max_turns" not in config


class TestLoadConfigParseFailure:
    """A YAML parse failure must NOT silently fall back to defaults.

    Before issue #23570 this was a single ``print(...)`` that scrolled past
    on the first invocation — users saw aux-fallback misbehavior with no clue
    their config.yaml was being ignored. The helper must:
      * log at WARNING (so ``hermes logs`` surfaces it)
      * also write to stderr (so it's visible at startup even before
        ``setup_logging()`` has wired up file handlers)
      * dedup on (path, mtime_ns, size) so concurrent loads don't spam
      * re-warn after the user edits the file (different mtime)
    """

    def test_logs_and_warns_on_parse_failure(self, tmp_path, caplog, capsys):
        # Reset the dedup cache so this test isn't affected by other tests
        # that may have warned about a different broken config.
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "config.yaml").write_text("\tbroken tab indent:\n")

            import logging
            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                config = load_config()

            # Falls back to defaults — confirms the silent-fallback we're warning about
            assert config["model"] == DEFAULT_CONFIG["model"]

            # WARNING-level log was emitted with file path + reason
            assert any(
                str(tmp_path / "config.yaml") in rec.message
                and "Falling back to default config" in rec.message
                for rec in caplog.records
            ), f"expected WARNING log, got: {[r.message for r in caplog.records]}"

            # stderr also got a user-visible message (with the ⚠️ marker so it
            # stands out at hermes startup before logging is configured)
            captured = capsys.readouterr()
            assert "hermes config:" in captured.err
            assert str(tmp_path / "config.yaml") in captured.err

    def test_dedup_on_repeated_load_same_file(self, tmp_path, capsys):
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "config.yaml").write_text("\tbroken:\n")

            load_config()
            first = capsys.readouterr().err
            assert "hermes config:" in first

            load_config()
            second = capsys.readouterr().err
            assert second == "", "second load should NOT re-warn (same file, same mtime)"

    def test_rewarns_after_file_edit(self, tmp_path, capsys):
        import time
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            (tmp_path / "config.yaml").write_text("\tbroken:\n")
            load_config()
            capsys.readouterr()  # discard first warning

            # Edit the file (still broken, but different content) — mtime changes
            time.sleep(0.05)
            (tmp_path / "config.yaml").write_text("\tstill broken differently:\n")
            load_config()
            after_edit = capsys.readouterr().err
            assert "hermes config:" in after_edit, "edited file should re-warn"

    def test_corrupt_config_is_backed_up(self, tmp_path, capsys):
        """A broken config.yaml is snapshotted to a timestamped .bak so the
        user's recoverable overrides survive a later wizard/config-set rewrite.

        Ported from google-gemini/gemini-cli#21541 (policy-file TOML recovery),
        adapted: we back up but deliberately do NOT reset config.yaml.
        """
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            broken = "\tmodel: test/custom\nbroken indent:\n"
            (tmp_path / "config.yaml").write_text(broken)

            load_config()
            err = capsys.readouterr().err

            baks = list(tmp_path.glob("config.yaml.corrupt.*.bak"))
            assert len(baks) == 1, f"expected one backup, got {baks}"
            # Backup preserves the original broken content verbatim
            assert baks[0].read_text() == broken
            # Original config.yaml is left untouched (not reset to clean state)
            assert (tmp_path / "config.yaml").read_text() == broken
            # User is told where the backup landed
            assert str(baks[0]) in err

    def test_backup_skips_when_same_size_bak_exists(self, tmp_path, capsys):
        """Don't churn backups: if a corrupt backup of the same size already
        exists (same corruption already preserved), skip making another."""
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            broken = "\tbroken:\n"
            cfg = tmp_path / "config.yaml"
            cfg.write_text(broken)

            # Pre-existing backup of identical size simulates an earlier snapshot.
            (tmp_path / "config.yaml.corrupt.20260101-000000.bak").write_text(broken)

            load_config()

            baks = list(tmp_path.glob("config.yaml.corrupt.*.bak"))
            assert len(baks) == 1, f"should not add a second same-size backup, got {baks}"

    def test_corrupt_symlink_config_not_backed_up(self, tmp_path):
        """Symlinked config.yaml is not copied (mirrors Gemini #21541 lstat
        guard) — avoids clobbering whatever the symlink points at."""
        import sys as _sys
        if _sys.platform == "win32":
            pytest.skip("symlink creation requires privileges on Windows")
        from hermes_cli import config as cfg_mod
        cfg_mod._CONFIG_PARSE_WARNED.clear()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            real = tmp_path / "real_config.yaml"
            real.write_text("\tbroken:\n")
            link = tmp_path / "config.yaml"
            link.symlink_to(real)

            load_config()

            assert not list(tmp_path.glob("config.yaml.corrupt.*.bak"))


class TestSaveAndLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/custom-model"
            config["agent"]["max_turns"] = 42
            save_config(config)

            reloaded = load_config()
            assert reloaded["model"] == "test/custom-model"
            assert reloaded["agent"]["max_turns"] == 42

            saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert saved["agent"]["max_turns"] == 42
            assert "max_turns" not in saved

    def test_save_config_normalizes_legacy_root_level_max_turns(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config({"model": "test/custom-model", "max_turns": 37})

            saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert saved["agent"]["max_turns"] == 37
            assert "max_turns" not in saved

    def test_nested_values_preserved(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["terminal"]["timeout"] = 999
            save_config(config)

            reloaded = load_config()
            assert reloaded["terminal"]["timeout"] == 999


class TestSaveEnvValueSecure:
    def test_save_env_value_writes_without_stdout(self, tmp_path, capsys):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == ""

            env_values = load_env()
            assert env_values["TENOR_API_KEY"] == "sk-test-secret"

    def test_secure_save_returns_metadata_only(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = save_env_value_secure("GITHUB_TOKEN", "ghp_test_secret")
            assert result == {
                "success": True,
                "stored_as": "GITHUB_TOKEN",
                "validated": False,
            }
            assert "secret" not in str(result).lower()

    def test_save_env_value_updates_process_environment(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
            os.environ.pop("TENOR_API_KEY", None)
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            assert os.environ["TENOR_API_KEY"] == "sk-test-secret"

    def test_save_env_value_hardens_file_permissions_on_posix(self, tmp_path):
        if os.name == "nt":
            return

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            env_mode = (tmp_path / ".env").stat().st_mode & 0o777
            assert env_mode == 0o600

    def test_save_env_value_preserves_existing_file_mode_on_posix(self, tmp_path):
        """Regression for #31518: pre-existing .env mode (e.g. 0640 for a
        Docker bind-mount that the operator chose) survives subsequent
        writes. Previously _secure_file ran unconditionally after the
        mode-restore branch and re-tightened to 0600.
        """
        if os.name == "nt":
            return

        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING=value\n")
        os.chmod(env_path, 0o640)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")

        env_mode = env_path.stat().st_mode & 0o777
        assert env_mode == 0o640, f"expected 0o640, got {oct(env_mode)}"


class TestRemoveEnvValue:
    def test_removes_key_from_env_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("KEY_A=value_a\nKEY_B=value_b\nKEY_C=value_c\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "KEY_B": "value_b"}):
            result = remove_env_value("KEY_B")
            assert result is True
            content = env_path.read_text()
            assert "KEY_B" not in content
            assert "KEY_A=value_a" in content
            assert "KEY_C=value_c" in content

    def test_clears_os_environ(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("MY_KEY=my_value\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "MY_KEY": "my_value"}):
            remove_env_value("MY_KEY")
            assert "MY_KEY" not in os.environ

    def test_returns_false_when_key_not_found(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER_KEY=value\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = remove_env_value("MISSING_KEY")
            assert result is False
            # File should be untouched
            assert env_path.read_text() == "OTHER_KEY=value\n"

    def test_handles_missing_env_file(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "GHOST_KEY": "ghost"}):
            result = remove_env_value("GHOST_KEY")
            assert result is False
            # os.environ should still be cleared
            assert "GHOST_KEY" not in os.environ

    def test_clears_os_environ_even_when_not_in_file(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER=stuff\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "ORPHAN_KEY": "orphan"}):
            remove_env_value("ORPHAN_KEY")
            assert "ORPHAN_KEY" not in os.environ

    def test_remove_env_value_preserves_existing_file_mode_on_posix(self, tmp_path):
        """Regression: pre-existing .env mode (e.g. 0640 for a Docker
        bind-mount the operator chose) survives a remove just as it does a
        save. Previously _secure_file ran unconditionally after the
        mode-restore branch and re-tightened to 0600 — the same bug fixed
        in save_env_value (#33699), in the sibling remove path.
        """
        if os.name == "nt":
            return

        env_path = tmp_path / ".env"
        env_path.write_text("KEEP=value\nDROP=gone\n")
        os.chmod(env_path, 0o640)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "DROP": "gone"}):
            removed = remove_env_value("DROP")

        assert removed is True
        assert "DROP" not in env_path.read_text()
        env_mode = env_path.stat().st_mode & 0o777
        assert env_mode == 0o640, f"expected 0o640, got {oct(env_mode)}"


class TestSaveConfigAtomicity:
    """Verify save_config uses atomic writes (tempfile + os.replace)."""

    def test_no_partial_write_on_crash(self, tmp_path):
        """If save_config crashes mid-write, the previous file stays intact."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # Write an initial config
            config = load_config()
            config["model"] = "original-model"
            save_config(config)

            config_path = tmp_path / "config.yaml"
            assert config_path.exists()

            # Simulate a crash during yaml.dump by making atomic_yaml_write's
            # yaml.dump raise after the temp file is created but before replace.
            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    config["model"] = "should-not-persist"
                    save_config(config)
                except OSError:
                    pass

            # Original file must still be intact
            reloaded = load_config()
            assert reloaded["model"] == "original-model"

    def test_no_leftover_temp_files(self, tmp_path):
        """Failed writes must clean up their temp files."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            save_config(config)

            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    save_config(config)
                except OSError:
                    pass

            # No .tmp files should remain
            tmp_files = list(tmp_path.glob(".*config*.tmp"))
            assert tmp_files == []

    def test_atomic_write_creates_valid_yaml(self, tmp_path):
        """The written file must be valid YAML matching the input."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/atomic-model"
            config["agent"]["max_turns"] = 77
            save_config(config)

            # Read raw YAML to verify it's valid and correct
            config_path = tmp_path / "config.yaml"
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            assert raw["model"] == "test/atomic-model"
            assert raw["agent"]["max_turns"] == 77


class TestSanitizeEnvLines:
    """Tests for .env file corruption repair."""

    def test_splits_concatenated_keys(self):
        """Two KEY=VALUE pairs jammed on one line get split."""
        lines = ["ANTHROPIC_API_KEY=sk-ant-xxxOPENAI_BASE_URL=https://api.openai.com/v1\n"]
        result = _sanitize_env_lines(lines)
        assert result == [
            "ANTHROPIC_API_KEY=sk-ant-xxx\n",
            "OPENAI_BASE_URL=https://api.openai.com/v1\n",
        ]

    def test_preserves_clean_file(self):
        """A well-formed .env file passes through unchanged (modulo trailing newlines)."""
        lines = [
            "OPENROUTER_API_KEY=sk-or-xxx\n",
            "FIRECRAWL_API_KEY=fc-xxx\n",
            "# a comment\n",
            "\n",
        ]
        result = _sanitize_env_lines(lines)
        assert result == lines

    def test_preserves_comments_and_blanks(self):
        lines = ["# comment\n", "\n", "KEY=val\n"]
        result = _sanitize_env_lines(lines)
        assert result == lines

    def test_adds_missing_trailing_newline(self):
        """Lines missing trailing newline get one added."""
        lines = ["FOO_BAR=baz"]
        result = _sanitize_env_lines(lines)
        assert result == ["FOO_BAR=baz\n"]

    def test_three_concatenated_keys(self):
        """Three known keys on one line all get separated."""
        lines = ["FAL_KEY=111FIRECRAWL_API_KEY=222GITHUB_TOKEN=333\n"]
        result = _sanitize_env_lines(lines)
        assert result == [
            "FAL_KEY=111\n",
            "FIRECRAWL_API_KEY=222\n",
            "GITHUB_TOKEN=333\n",
        ]

    def test_value_with_equals_sign_not_split(self):
        """A value containing '=' shouldn't be falsely split (lowercase in value)."""
        lines = ["OPENAI_BASE_URL=https://api.example.com/v1?key=abc123\n"]
        result = _sanitize_env_lines(lines)
        assert result == lines

    def test_unknown_keys_not_split(self):
        """Unknown key names on one line are NOT split (avoids false positives)."""
        lines = ["CUSTOM_VAR=value123OTHER_THING=value456\n"]
        result = _sanitize_env_lines(lines)
        # Unknown keys stay on one line — no false split
        assert len(result) == 1

    def test_value_ending_with_digits_still_splits(self):
        """Concatenation is detected even when value ends with digits."""
        lines = ["OPENROUTER_API_KEY=sk-or-v1-abc123OPENAI_BASE_URL=https://api.openai.com/v1\n"]
        result = _sanitize_env_lines(lines)
        assert len(result) == 2
        assert result[0].startswith("OPENROUTER_API_KEY=")
        assert result[1].startswith("OPENAI_BASE_URL=")

    def test_glm_suffix_collision_not_split(self):
        """GLM_API_KEY / GLM_BASE_URL must not be mangled by LM_API_KEY / LM_BASE_URL suffixes (#17138)."""
        lines = [
            "GLM_API_KEY=glm-secret\n",
            "GLM_BASE_URL=https://api.z.ai/api/paas/v4\n",
        ]
        result = _sanitize_env_lines(lines)
        assert result == lines, f"GLM_* lines were corrupted by suffix collision: {result}"

    def test_suffix_collision_does_not_break_real_concatenation(self):
        """A genuine concatenation that happens to start with a suffix-superset key still splits."""
        lines = ["GLM_API_KEY=glmLM_API_KEY=lm-key\n"]
        result = _sanitize_env_lines(lines)
        assert len(result) == 2
        assert result[0].startswith("GLM_API_KEY=")
        assert result[1].startswith("LM_API_KEY=")

    def test_save_env_value_fixes_corruption_on_write(self, tmp_path):
        """save_env_value sanitizes corrupted lines when writing a new key."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ANTHROPIC_API_KEY=sk-antOPENAI_BASE_URL=https://api.openai.com/v1\n"
            "FAL_KEY=existing\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("MESSAGING_CWD", "/tmp")

            content = env_file.read_text()
            lines = content.strip().split("\n")

            # Corrupted line should be split, new key added
            assert "ANTHROPIC_API_KEY=sk-ant" in lines
            assert "OPENAI_BASE_URL=https://api.openai.com/v1" in lines
            assert "MESSAGING_CWD=/tmp" in lines

    def test_sanitize_env_file_returns_fix_count(self, tmp_path):
        """sanitize_env_file reports how many entries were fixed."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FAL_KEY=good\n"
            "OPENROUTER_API_KEY=valFIRECRAWL_API_KEY=val2\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            fixes = sanitize_env_file()
            assert fixes > 0

            # Verify file is now clean
            content = env_file.read_text()
            assert "OPENROUTER_API_KEY=val\n" in content
            assert "FIRECRAWL_API_KEY=val2\n" in content

    def test_sanitize_env_file_noop_on_clean_file(self, tmp_path):
        """No changes when file is already clean."""
        env_file = tmp_path / ".env"
        env_file.write_text("GOOD_KEY=good\nOTHER_KEY=other\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            fixes = sanitize_env_file()
            assert fixes == 0


class TestOptionalEnvVarsRegistry:
    """Verify that key env vars are registered in OPTIONAL_ENV_VARS."""

    def test_tavily_api_key_registered(self):
        """TAVILY_API_KEY is listed in OPTIONAL_ENV_VARS."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "TAVILY_API_KEY" in OPTIONAL_ENV_VARS

    def test_tavily_api_key_is_tool_category(self):
        """TAVILY_API_KEY is in the 'tool' category."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TAVILY_API_KEY"]["category"] == "tool"

    def test_tavily_api_key_is_password(self):
        """TAVILY_API_KEY is marked as password."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TAVILY_API_KEY"]["password"] is True

    def test_tavily_api_key_has_url(self):
        """TAVILY_API_KEY has a URL."""
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TAVILY_API_KEY"]["url"] == "https://app.tavily.com/home"

    def test_tavily_in_env_vars_by_version(self):
        """TAVILY_API_KEY is listed in ENV_VARS_BY_VERSION."""
        from hermes_cli.config import ENV_VARS_BY_VERSION
        all_vars = []
        for vars_list in ENV_VARS_BY_VERSION.values():
            all_vars.extend(vars_list)
        assert "TAVILY_API_KEY" in all_vars

    def test_max_iterations_not_offered_as_env_var(self):
        """HERMES_MAX_ITERATIONS must NOT be in OPTIONAL_ENV_VARS (issue #17534).

        Offering it as an editable env var (dashboard, `hermes setup`) lets a
        user write it to .env, recreating the stale ghost that shadows
        config.yaml's agent.max_turns. The iteration budget is configured ONLY
        via config.yaml; HERMES_MAX_ITERATIONS remains a read-only backward-compat
        fallback in the gateway/CLI, never a promoted write target.
        """
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "HERMES_MAX_ITERATIONS" not in OPTIONAL_ENV_VARS


class TestConfigMigrationSecretPrompts:
    def test_required_secret_env_prompt_uses_masked_prompt(self, tmp_path, monkeypatch):
        from hermes_cli import config as cfg_mod

        saved = {}

        monkeypatch.setattr(cfg_mod, "sanitize_env_file", lambda: 0)
        monkeypatch.setattr(cfg_mod, "check_config_version", lambda: (999, 999))
        monkeypatch.setattr(cfg_mod, "get_missing_config_fields", lambda: [])
        monkeypatch.setattr(cfg_mod, "get_missing_skill_config_vars", lambda: [])
        monkeypatch.setattr(
            cfg_mod,
            "get_missing_env_vars",
            lambda required_only=True: [
                {
                    "name": "TEST_API_KEY",
                    "description": "Test key",
                    "prompt": "Test API key",
                    "password": True,
                }
            ]
            if required_only
            else [],
        )
        def fake_masked_secret_prompt(prompt):
            saved["prompt"] = prompt
            return "secret"

        monkeypatch.setattr(cfg_mod, "masked_secret_prompt", fake_masked_secret_prompt)
        monkeypatch.setattr(
            cfg_mod,
            "save_env_value",
            lambda name, value: saved.update({name: value}),
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            results = cfg_mod.migrate_config(interactive=True, quiet=True)

        assert saved["prompt"] == "  Test API key: "
        assert saved["TEST_API_KEY"] == "secret"
        assert results["env_added"] == ["TEST_API_KEY"]


class TestConfigVersionDetection:
    def test_check_config_version_uses_raw_on_disk_version(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: {}\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            assert load_config()["_config_version"] == DEFAULT_CONFIG["_config_version"]
            assert check_config_version() == (0, DEFAULT_CONFIG["_config_version"])

    def test_check_config_version_treats_missing_file_as_current(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            latest = DEFAULT_CONFIG["_config_version"]
            assert check_config_version() == (latest, latest)

    def test_check_config_version_does_not_migrate_invalid_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("model: [unterminated\n", encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            latest = DEFAULT_CONFIG["_config_version"]
            assert check_config_version() == (latest, latest)


class TestAnthropicTokenMigration:
    """Test that config version 8→9 clears ANTHROPIC_TOKEN."""

    def _write_config_version(self, tmp_path, version):
        config_path = tmp_path / "config.yaml"
        import yaml
        config_path.write_text(yaml.safe_dump({"_config_version": version}))

    def test_clears_token_on_upgrade_to_v9(self, tmp_path):
        """ANTHROPIC_TOKEN is cleared unconditionally when upgrading to v9."""
        self._write_config_version(tmp_path, 8)
        (tmp_path / ".env").write_text("ANTHROPIC_TOKEN=old-token\n")
        with patch.dict(os.environ, {
            "HERMES_HOME": str(tmp_path),
            "ANTHROPIC_TOKEN": "old-token",
        }):
            migrate_config(interactive=False, quiet=True)
            assert load_env().get("ANTHROPIC_TOKEN") == ""

    def test_skips_on_version_9_or_later(self, tmp_path):
        """Already at v9 — ANTHROPIC_TOKEN is not touched."""
        self._write_config_version(tmp_path, 9)
        (tmp_path / ".env").write_text("ANTHROPIC_TOKEN=current-token\n")
        with patch.dict(os.environ, {
            "HERMES_HOME": str(tmp_path),
            "ANTHROPIC_TOKEN": "current-token",
        }):
            migrate_config(interactive=False, quiet=True)
            assert load_env().get("ANTHROPIC_TOKEN") == "current-token"


class TestCustomProviderCompatibility:
    """Custom provider compatibility across legacy and v12+ config schemas."""

    def test_v11_upgrade_moves_custom_providers_into_providers(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 11,
                    "model": {
                        "default": "openai/gpt-5.4",
                        "provider": "openrouter",
                    },
                    "custom_providers": [
                        {
                            "name": "OpenAI Direct",
                            "base_url": "https://api.openai.com/v1",
                            "api_key": "test-key",
                            "api_mode": "codex_responses",
                            "model": "gpt-5-mini",
                        }
                    ],
                    "fallback_providers": [
                        {"provider": "openai-direct", "model": "gpt-5-mini"}
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        from hermes_cli.config import DEFAULT_CONFIG
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert raw["providers"]["openai-direct"] == {
            "api": "https://api.openai.com/v1",
            "api_key": "test-key",
            "default_model": "gpt-5-mini",
            "name": "OpenAI Direct",
            "transport": "codex_responses",
        }
        # custom_providers removed by migration — runtime reads via compat layer
        assert "custom_providers" not in raw

    def test_v11_upgrade_preserves_custom_provider_model_metadata(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        model_map = {
            "kimi-k2.6": {"context_length": 262144},
            "moonshotai/Kimi-K2.6-ACED": {"context_length": 131072},
        }
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 11,
                    "custom_providers": [
                        {
                            "name": "Kimi Coding Plan",
                            "base_url": "https://api.kimi.example.com/coding",
                            "api_key_env": "KIMI_CODING_API_KEY",
                            "api_mode": "anthropic_messages",
                            "model": "kimi-k2.6",
                            "models": model_map,
                            "context_length": 262144,
                            "rate_limit_delay": 0.25,
                            "discover_models": False,
                            "extra_body": {
                                "chat_template_kwargs": {"enable_thinking": False}
                            },
                        },
                        {
                            "name": "List Models",
                            "base_url": "https://list.example.com/v1",
                            "models": ["alpha", "beta"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            compatible = get_compatible_custom_providers(raw)

        assert "custom_providers" not in raw
        provider = raw["providers"]["kimi-coding-plan"]
        assert provider["api"] == "https://api.kimi.example.com/coding"
        assert provider["key_env"] == "KIMI_CODING_API_KEY"
        assert provider["transport"] == "anthropic_messages"
        assert provider["default_model"] == "kimi-k2.6"
        assert provider["models"] == model_map
        assert provider["context_length"] == 262144
        assert provider["rate_limit_delay"] == 0.25
        assert provider["discover_models"] is False
        assert provider["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert raw["providers"]["list-models"]["models"] == {
            "alpha": {},
            "beta": {},
        }

        compatible_provider = next(
            entry for entry in compatible if entry["provider_key"] == "kimi-coding-plan"
        )
        assert compatible_provider["models"] == model_map
        assert compatible_provider["key_env"] == "KIMI_CODING_API_KEY"

    def test_providers_dict_resolves_at_runtime(self, tmp_path):
        """After migration deleted custom_providers, get_compatible_custom_providers
        still finds entries from the providers dict."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "providers": {
                        "openai-direct": {
                            "api": "https://api.openai.com/v1",
                            "api_key": "test-key",
                            "default_model": "gpt-5-mini",
                            "name": "OpenAI Direct",
                            "transport": "codex_responses",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert len(compatible) == 1
        assert compatible[0]["name"] == "OpenAI Direct"
        assert compatible[0]["base_url"] == "https://api.openai.com/v1"
        assert compatible[0]["provider_key"] == "openai-direct"
        assert compatible[0]["api_mode"] == "codex_responses"

    def test_compatible_custom_providers_prefers_base_url_then_url_then_api(self, tmp_path):
        """URL field precedence is base_url > url > api (PR #9332)."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "providers": {
                        "my-provider": {
                            "name": "My Provider",
                            "api": "https://api.example.com/v1",
                            "url": "https://url.example.com/v1",
                            "base_url": "https://base.example.com/v1",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert compatible == [
            {
                "name": "My Provider",
                "base_url": "https://base.example.com/v1",
                "provider_key": "my-provider",
            }
        ]

    def test_dedup_across_legacy_and_providers(self, tmp_path):
        """Same name+url in both schemas should not produce duplicates."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "custom_providers": [
                        {
                            "name": "OpenAI Direct",
                            "base_url": "https://api.openai.com/v1",
                            "api_key": "legacy-key",
                        }
                    ],
                    "providers": {
                        "openai-direct": {
                            "api": "https://api.openai.com/v1",
                            "api_key": "new-key",
                            "name": "OpenAI Direct",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert len(compatible) == 1
        # Legacy entry wins (read first)
        assert compatible[0]["api_key"] == "legacy-key"

    def test_dedup_preserves_entries_with_different_models(self, tmp_path):
        """Entries with same name+URL but different models must not be collapsed."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "_config_version": 17,
                    "custom_providers": [
                        {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "qwen3-coder"},
                        {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "glm-5.1"},
                        {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "kimi-k2.5"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            compatible = get_compatible_custom_providers()

        assert len(compatible) == 3
        models = [e.get("model") for e in compatible]
        assert models == ["qwen3-coder", "glm-5.1", "kimi-k2.5"]


class TestInterimAssistantMessageConfig:
    """Test the explicit gateway interim-message config gate."""

    def test_default_config_enables_interim_assistant_messages(self):
        assert DEFAULT_CONFIG["display"]["interim_assistant_messages"] is True

    def test_migrate_to_v15_adds_interim_assistant_message_gate(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"_config_version": 14, "display": {"tool_progress": "off"}}),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        from hermes_cli.config import DEFAULT_CONFIG
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert raw["display"]["tool_progress"] == "off"
        assert raw["display"]["interim_assistant_messages"] is True


class TestDiscordChannelPromptsConfig:
    def test_default_config_includes_discord_channel_prompts(self):
        assert DEFAULT_CONFIG["discord"]["channel_prompts"] == {}

    def test_migrate_adds_discord_channel_prompts_default(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"_config_version": 17, "discord": {"auto_thread": True}}),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        from hermes_cli.config import DEFAULT_CONFIG
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert raw["discord"]["auto_thread"] is True
        assert raw["discord"]["channel_prompts"] == {}


class TestUserMessagePreviewConfig:
    def test_default_config_preview_line_counts(self):
        preview = DEFAULT_CONFIG["display"]["user_message_preview"]
        assert preview["first_lines"] == 2
        assert preview["last_lines"] == 2


class TestEnvWriteDenylist:
    """``save_env_value`` refuses to persist env-var names that
    influence how subprocesses execute — ``LD_PRELOAD``, ``PYTHONPATH``,
    ``PATH``, ``EDITOR``, etc. — or any ``HERMES_*`` runtime flag.

    The dashboard exposes ``PUT /api/env`` to any authed caller (and
    the session token lives in the SPA's HTML where any future plugin
    XSS or local process could exfiltrate it). Without this gate, an
    attacker who steals the token could plant
    ``LD_PRELOAD=/tmp/evil.so`` in ``.env`` and own the next Hermes
    process on next startup via the dotenv → ``os.environ`` chain in
    ``hermes_cli/env_loader.py``.

    Regression test for the dashboard pentest finding filed alongside
    the ``web-pentest`` skill (PR #32265 / issue #32267).
    """

    @pytest.fixture(autouse=True)
    def _hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        ensure_hermes_home()

    @pytest.mark.parametrize(
        "denied_key",
        [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "LD_AUDIT",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "NODE_OPTIONS",
            "NODE_PATH",
            "PATH",
            "SHELL",
            "EDITOR",
            "VISUAL",
            "PAGER",
            "BROWSER",
            "GIT_SSH_COMMAND",
            "GIT_EXEC_PATH",
            "HERMES_HOME",
            "HERMES_PROFILE",
            "HERMES_CONFIG",
            "HERMES_ENV",
        ],
    )
    def test_denylisted_keys_rejected(self, denied_key):
        """Each denylisted name raises ``ValueError`` and never reaches
        the on-disk ``.env`` file."""
        with pytest.raises(ValueError, match="denylist"):
            save_env_value(denied_key, "anything")

        # And nothing landed on disk either.
        env = load_env()
        assert denied_key not in env

    @pytest.mark.parametrize(
        "allowed_key",
        [
            "HERMES_GEMINI_CLIENT_ID",
            "HERMES_LANGFUSE_PUBLIC_KEY",
            "HERMES_SPOTIFY_CLIENT_ID",
            "HERMES_QWEN_BASE_URL",
            "HERMES_MAX_ITERATIONS",
        ],
    )
    def test_hermes_integration_keys_still_writable(self, allowed_key):
        """``HERMES_*`` overall is NOT blocked — only the four runtime
        location names (HOME/PROFILE/CONFIG/ENV) are. Integration
        credentials following the ``HERMES_*`` convention must keep
        working or we'd regress every provider setup wizard that
        currently writes one of these (auth.py, Spotify, Langfuse, …)."""
        save_env_value(allowed_key, "test-value-123")
        env = load_env()
        assert env[allowed_key] == "test-value-123"

    def test_legitimate_provider_key_still_works(self):
        """The denylist must not regress on real provider key writes."""
        save_env_value("OPENROUTER_API_KEY", "sk-or-test-1234")
        env = load_env()
        assert env["OPENROUTER_API_KEY"] == "sk-or-test-1234"

    def test_arbitrary_user_key_still_works(self):
        """Plugin / user-defined env vars (anything outside the
        denylist and outside ``HERMES_*``) keep working. The denylist
        is narrow on purpose."""
        save_env_value("MY_PLUGIN_TOKEN", "plugin-secret-123")
        env = load_env()
        assert env["MY_PLUGIN_TOKEN"] == "plugin-secret-123"

    def test_save_env_value_secure_inherits_denylist(self):
        """The ``_secure`` variant goes through ``save_env_value`` so
        it inherits the gate — verify, don't assume."""
        with pytest.raises(ValueError, match="denylist"):
            save_env_value_secure("LD_PRELOAD", "/tmp/evil.so")

    def test_pre_existing_value_in_env_file_is_left_alone(self, tmp_path):
        """The gate is on *write*. If ``.env`` already contains
        ``LD_PRELOAD`` (set out-of-band by the operator before this
        change shipped, or hand-edited), we don't blow up — we just
        refuse to add or update it via the API."""
        env_path = tmp_path / ".env"
        env_path.write_text("LD_PRELOAD=/something/legit.so\n")

        # load_env returns it (the read path is intentionally permissive)
        env = load_env()
        assert env["LD_PRELOAD"] == "/something/legit.so"

        # But the write path still refuses to update it
        with pytest.raises(ValueError, match="denylist"):
            save_env_value("LD_PRELOAD", "/tmp/evil.so")


class TestWriteApprovalMigration:
    """Version 28→29 renames memory/skills write_mode → write_approval (bool).

    Only an explicit ``approve`` carried gating intent and maps to ``True``;
    ``on``/``off``/unset map to ``False`` (gate off). The old ``write_mode`` key
    is removed. Only a persisted key is rewritten — never invented.
    """

    def _write(self, tmp_path, body: str):
        (tmp_path / "config.yaml").write_text(body)

    def test_approve_maps_to_true(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(tmp_path,
                        "_config_version: 28\nmemory:\n  write_mode: approve\n"
                        "skills:\n  write_mode: approve\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert raw["memory"]["write_approval"] is True
            assert raw["skills"]["write_approval"] is True
            assert "write_mode" not in raw["memory"]
            assert "write_mode" not in raw["skills"]

    def test_on_and_off_map_to_false(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # YAML 1.1 parses bare on/off as bools — write_mode could be either
            # the string or the bool; both legacy "not gating" values → False.
            self._write(tmp_path,
                        "_config_version: 28\nmemory:\n  write_mode: 'on'\n"
                        "skills:\n  write_mode: 'off'\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert raw["memory"]["write_approval"] is False
            assert raw["skills"]["write_approval"] is False

    def test_unset_key_defaults_to_false(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            self._write(tmp_path, "_config_version: 28\nmemory:\n  memory_enabled: true\n")
            migrate_config(interactive=False, quiet=True)
            raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
            # No write_mode was persisted, so the rename is a no-op; the missing-
            # field pass then seeds the default (False = gate off). Either way the
            # gate ends up off and there's no leftover write_mode key.
            assert raw["memory"].get("write_approval", False) is False
            assert "write_mode" not in raw.get("memory", {})
