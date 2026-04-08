"""Tests for hermes_cli configuration management."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_hermes_home,
    ensure_hermes_home,
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

    def test_legacy_root_level_max_turns_migrates_to_agent_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config_path = tmp_path / "config.yaml"
            config_path.write_text("max_turns: 42\n")

            config = load_config()
            assert config["agent"]["max_turns"] == 42
            assert "max_turns" not in config


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
