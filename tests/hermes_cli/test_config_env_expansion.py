"""Tests for ${ENV_VAR} substitution in config.yaml values."""

import os
import pytest
from hermes_cli.config import _expand_env_vars, load_config
from unittest.mock import patch as mock_patch


class TestExpandEnvVars:
    def test_simple_substitution(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("MY_KEY", "secret123")
            assert _expand_env_vars("${MY_KEY}") == "secret123"

    def test_missing_var_kept_verbatim(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("UNDEFINED_VAR_XYZ", raising=False)
            assert _expand_env_vars("${UNDEFINED_VAR_XYZ}") == "${UNDEFINED_VAR_XYZ}"

    def test_no_placeholder_unchanged(self):
        assert _expand_env_vars("plain-value") == "plain-value"

    def test_dict_recursive(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("TOKEN", "tok-abc")
            result = _expand_env_vars({"key": "${TOKEN}", "other": "literal"})
            assert result == {"key": "tok-abc", "other": "literal"}

    def test_nested_dict(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("API_KEY", "sk-xyz")
            result = _expand_env_vars({"model": {"api_key": "${API_KEY}"}})
            assert result["model"]["api_key"] == "sk-xyz"

    def test_list_items(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("VAL", "hello")
            result = _expand_env_vars(["${VAL}", "literal", 42])
            assert result == ["hello", "literal", 42]

    def test_non_string_values_untouched(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_multiple_placeholders_in_one_string(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("HOST", "localhost")
            mp.setenv("PORT", "5432")
            assert _expand_env_vars("${HOST}:${PORT}") == "localhost:5432"

    def test_dict_keys_not_expanded(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("KEY", "value")
            result = _expand_env_vars({"${KEY}": "no-expand-key"})
            assert "${KEY}" in result


class TestLoadConfigExpansion:
    def test_load_config_expands_env_vars(self, tmp_path, monkeypatch):
        config_yaml = (
            "model:\n"
            "  api_key: ${GOOGLE_API_KEY}\n"
            "platforms:\n"
            "  telegram:\n"
            "    token: ${TELEGRAM_BOT_TOKEN}\n"
            "plain: no-substitution\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("GOOGLE_API_KEY", "gsk-test-key")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1234567:ABC-token")
        # Patch the imported function's own globals. Other tests may reload
        # hermes_cli.config, making string-target monkeypatches hit a different
        # module object than this collection-time imported load_config().
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["api_key"] == "gsk-test-key"
        assert config["platforms"]["telegram"]["token"] == "1234567:ABC-token"
        assert config["plain"] == "no-substitution"

    def test_load_config_unresolved_kept_verbatim(self, tmp_path, monkeypatch):
        config_yaml = "model:\n  api_key: ${NOT_SET_XYZ_123}\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("NOT_SET_XYZ_123", raising=False)
        monkeypatch.setitem(load_config.__globals__, "get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["api_key"] == "${NOT_SET_XYZ_123}"


class TestLoadCliConfigExpansion:
    """Verify that load_cli_config() also expands ${VAR} references."""

    def test_cli_config_expands_auxiliary_api_key(self, tmp_path, monkeypatch):
        config_yaml = (
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${TEST_VISION_KEY_XYZ}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("TEST_VISION_KEY_XYZ", "vis-key-123")
        # Patch the hermes home so load_cli_config finds our test config
        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert config["auxiliary"]["vision"]["api_key"] == "vis-key-123"

    def test_cli_config_unresolved_kept_verbatim(self, tmp_path, monkeypatch):
        config_yaml = (
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${UNSET_CLI_VAR_ABC}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("UNSET_CLI_VAR_ABC", raising=False)
        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert config["auxiliary"]["vision"]["api_key"] == "${UNSET_CLI_VAR_ABC}"
