"""Tests for config.yaml structure validation and its CLI command."""

import argparse
import os
import subprocess
import sys
from argparse import Namespace

import hermes_cli.config as config_mod
from hermes_cli.config import (
    DEFAULT_CONFIG,
    _EXTRA_KNOWN_ROOT_KEYS,
    _KNOWN_ROOT_KEYS,
    config_command,
    validate_config_structure,
    ConfigIssue,
)
from hermes_cli.subcommands.config import build_config_parser


class TestCustomProvidersValidation:
    """custom_providers must be a YAML list, not a dict."""

    def test_dict_instead_of_list(self):
        """The exact Discord user scenario — custom_providers as flat dict."""
        issues = validate_config_structure({
            "custom_providers": {
                "name": "Generativelanguage.googleapis.com",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "api_key": "xxx",
                "model": "models/gemini-2.5-flash",
                "rate_limit_delay": 2.0,
                "fallback_model": {
                    "provider": "openrouter",
                    "model": "qwen/qwen3.6-plus:free",
                },
            },
            "fallback_providers": [],
        })
        errors = [i for i in issues if i.severity == "error"]
        assert any("dict" in i.message and "list" in i.message for i in errors), (
            "Should detect custom_providers as dict instead of list"
        )

    def test_dict_detects_misplaced_fields(self):
        """When custom_providers is a dict, detect fields that look misplaced."""
        issues = validate_config_structure({
            "custom_providers": {
                "name": "test",
                "base_url": "https://example.com",
                "api_key": "xxx",
            },
        })
        warnings = [i for i in issues if i.severity == "warning"]
        # Should flag base_url, api_key as looking like custom_providers entry fields
        misplaced = [i for i in warnings if "custom_providers entry fields" in i.message]
        assert len(misplaced) == 1


    def test_list_entry_not_dict(self):
        """Non-dict list entries should warn."""
        issues = validate_config_structure({
            "custom_providers": ["not-a-dict"],
            "model": {"provider": "custom"},
        })
        assert any("not a dict" in i.message for i in issues)




class TestMissingModelSection:
    """Warn when custom_providers exists but model section is missing."""


    def test_custom_providers_with_model(self):
        issues = validate_config_structure({
            "custom_providers": [
                {"name": "test", "base_url": "https://example.com/v1"},
            ],
            "model": {"provider": "custom", "default": "test-model"},
        })
        # Should not warn about missing model section
        assert not any("no 'model' section" in i.message for i in issues)


class TestConfigIssueDataclass:
    """ConfigIssue should be a proper dataclass."""

    def test_fields(self):
        issue = ConfigIssue(severity="error", message="test msg", hint="test hint")
        assert issue.severity == "error"
        assert issue.message == "test msg"
        assert issue.hint == "test hint"

    def test_equality(self):
        a = ConfigIssue("error", "msg", "hint")
        b = ConfigIssue("error", "msg", "hint")
        assert a == b


class TestVoiceSubmitModeValidation:
    def test_default_is_direct(self):
        assert DEFAULT_CONFIG["voice"]["submit_mode"] == "direct"

    def test_direct_and_draft_are_valid(self):
        for mode in ("direct", "draft"):
            issues = validate_config_structure({"voice": {"submit_mode": mode}})
            assert not any("voice.submit_mode" in issue.message for issue in issues)

    def test_invalid_mode_is_reported(self):
        issues = validate_config_structure({"voice": {"submit_mode": "refine"}})

        assert any(
            issue.severity == "error"
            and "voice.submit_mode" in issue.message
            and "direct" in issue.hint
            and "draft" in issue.hint
            for issue in issues
        )


class TestUnknownTopLevelKeys:
    """Arbitrary top-level keys must NOT warn — they are bridged to os.environ.

    Top-level scalars in config.yaml are forwarded into the environment
    (gateway/run.py, hermes send) so users can feed skills and external apps
    env-style keys like DISCORD_HOME_CHANNEL or MY_APP_TOKEN. A closed-world
    allowlist can never enumerate those, so no "Unknown top-level config key"
    warning may exist.
    """


    def test_known_root_keys_derived_from_default_config(self):
        """_KNOWN_ROOT_KEYS must be DEFAULT_CONFIG.keys() plus extras — single source of truth."""
        assert set(DEFAULT_CONFIG.keys()).issubset(_KNOWN_ROOT_KEYS)
        assert _EXTRA_KNOWN_ROOT_KEYS.issubset(_KNOWN_ROOT_KEYS)
        assert _KNOWN_ROOT_KEYS == frozenset(DEFAULT_CONFIG.keys()) | _EXTRA_KNOWN_ROOT_KEYS

    def test_provider_like_unknown_root_keeps_misplaced_message(self):
        """Preserve existing base_url/api_key root-level guidance."""
        issues = validate_config_structure({
            "base_url": "https://example.com/v1",
            "api_key": "secret",
        })
        misplaced = [
            i for i in issues
            if i.severity == "warning" and "looks misplaced" in i.message
        ]
        assert any("base_url" in i.message for i in misplaced)
        assert any("api_key" in i.message for i in misplaced)


class TestConfigValidateParser:
    """Parser coverage for ``hermes config validate [path]``."""

    @staticmethod
    def _parse(argv):
        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")

        def handler(_args):
            return None

        build_config_parser(subparsers, cmd_config=handler)
        return parser.parse_args(argv), handler

    def test_explicit_path(self):
        args, handler = self._parse(["config", "validate", "custom.yaml"])

        assert args.func is handler
        assert args.config_command == "validate"
        assert args.path == "custom.yaml"

    def test_default_path(self):
        args, _ = self._parse(["config", "validate"])

        assert args.config_command == "validate"
        assert args.path is None


class TestConfigValidateCommand:
    """Behavior coverage for the config validation command handler."""

    def test_valid_default_config_path_returns_zero(
        self, tmp_path, monkeypatch, capsys
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  provider: openrouter\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "get_config_path", lambda: config_path)

        status = config_command(Namespace(config_command="validate", path=None))

        assert status == 0
        assert "Configuration structure is valid" in capsys.readouterr().out

    def test_explicit_path_is_validated_instead_of_default(
        self, tmp_path, monkeypatch, capsys
    ):
        default_path = tmp_path / "default.yaml"
        default_path.write_text("{}\n", encoding="utf-8")
        explicit_path = tmp_path / "custom.yaml"
        explicit_path.write_text(
            "custom_providers:\n  name: broken\n  base_url: https://example.com\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_mod, "get_config_path", lambda: default_path)

        status = config_command(
            Namespace(config_command="validate", path=str(explicit_path))
        )

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert str(explicit_path) in captured.err
        assert "custom_providers is a dict" in captured.err
        assert "Change to:" in captured.err

    def test_missing_input_returns_one(self, tmp_path, capsys):
        missing_path = tmp_path / "missing.yaml"

        status = config_command(
            Namespace(config_command="validate", path=str(missing_path))
        )

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert "Could not load config" in captured.err
        assert str(missing_path) in captured.err

    def test_unreadable_input_returns_one(self, tmp_path, capsys):
        directory_path = tmp_path / "config.yaml"
        directory_path.mkdir()

        status = config_command(
            Namespace(config_command="validate", path=str(directory_path))
        )

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert "Could not load config" in captured.err
        assert str(directory_path) in captured.err

    def test_warning_only_input_returns_zero(self, tmp_path, capsys):
        config_path = tmp_path / "warning.yaml"
        config_path.write_text(
            "custom_providers:\n  - name: example\n    base_url: https://example.com\n",
            encoding="utf-8",
        )

        status = config_command(
            Namespace(config_command="validate", path=str(config_path))
        )

        captured = capsys.readouterr()
        assert status == 0
        assert captured.out == ""
        assert "no 'model' section" in captured.err

    def test_non_mapping_root_returns_one(self, tmp_path, capsys):
        cases = [
            ("[]\n", "list"),
            ("false\n", "bool"),
            ("0\n", "int"),
            ('""\n', "str"),
            ("- model\n", "list"),
            ("plain text\n", "str"),
        ]
        config_path = tmp_path / "non-mapping.yaml"

        for contents, root_type in cases:
            config_path.write_text(contents, encoding="utf-8")

            status = config_command(
                Namespace(config_command="validate", path=str(config_path))
            )

            captured = capsys.readouterr()
            assert status == 1
            assert captured.out == ""
            assert "root must be a mapping" in captured.err
            assert root_type in captured.err

    def test_non_string_top_level_key_returns_one(self, tmp_path, capsys):
        cases = [
            ("1: value\n", "int"),
            ("true: value\n", "bool"),
            ("null: value\n", "NoneType"),
        ]
        config_path = tmp_path / "non-string-key.yaml"

        for contents, key_type in cases:
            config_path.write_text(contents, encoding="utf-8")

            status = config_command(
                Namespace(config_command="validate", path=str(config_path))
            )

            captured = capsys.readouterr()
            assert status == 1
            assert captured.out == ""
            assert "top-level keys must be strings" in captured.err
            assert key_type in captured.err

    def test_invalid_utf8_returns_one(self, tmp_path, capsys):
        config_path = tmp_path / "invalid-utf8.yaml"
        config_path.write_bytes(b"\xff")

        status = config_command(
            Namespace(config_command="validate", path=str(config_path))
        )

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert "Could not load config" in captured.err

    def test_utf8_bom_is_accepted(self, tmp_path, capsys):
        config_path = tmp_path / "bom.yaml"
        config_path.write_bytes(b"\xef\xbb\xbfmodel:\n  provider: openrouter\n")

        status = config_command(
            Namespace(config_command="validate", path=str(config_path))
        )

        captured = capsys.readouterr()
        assert status == 0
        assert "Configuration structure is valid" in captured.out
        assert captured.err == ""

    def test_malformed_yaml_returns_one(self, tmp_path, capsys):
        config_path = tmp_path / "malformed.yaml"
        config_path.write_text("model: [unterminated\n", encoding="utf-8")

        status = config_command(
            Namespace(config_command="validate", path=str(config_path))
        )

        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert "Could not load config" in captured.err

    def test_process_exit_code_is_nonzero_for_invalid_structure(self, tmp_path):
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text(
            "custom_providers:\n  name: broken\n  base_url: https://example.com\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        isolated_home = tmp_path / "profiles" / "test"
        isolated_home.mkdir(parents=True)
        env["HERMES_HOME"] = str(isolated_home)
        env.pop("HERMES_PROFILE", None)
        env.pop("HERMES_CONFIG", None)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hermes_cli.main",
                "config",
                "validate",
                str(config_path),
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            env=env,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert result.stdout == ""
        assert "custom_providers is a dict" in result.stderr
