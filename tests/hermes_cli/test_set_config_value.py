"""Tests for set_config_value — verifying secrets route to .env and config to config.yaml."""

import argparse
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import set_config_value, config_command


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text()


def _read_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    return config_path.read_text() if config_path.exists() else ""


# ---------------------------------------------------------------------------
# Explicit allowlist keys → .env
# ---------------------------------------------------------------------------

class TestExplicitAllowlist:
    """Keys in the hardcoded allowlist should always go to .env."""

    @pytest.mark.parametrize("key", [
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HONCHO_API_KEY",
        "FIRECRAWL_API_KEY",
        "BROWSERBASE_API_KEY",
        "FAL_KEY",
        "SUDO_PASSWORD",
        "GITHUB_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
    ])
    def test_explicit_key_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "test-value-123")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=test-value-123" in env_content
        # Must NOT appear in config.yaml
        assert key not in _read_config(_isolated_hermes_home)


# ---------------------------------------------------------------------------
# Catch-all patterns → .env
# ---------------------------------------------------------------------------

class TestCatchAllPatterns:
    """Any key ending in _API_KEY or _TOKEN should route to .env."""

    @pytest.mark.parametrize("key", [
        "DAYTONA_API_KEY",
        "ELEVENLABS_API_KEY",
        "SOME_FUTURE_SERVICE_API_KEY",
        "MY_CUSTOM_TOKEN",
        "WHATSAPP_BOT_TOKEN",
    ])
    def test_api_key_suffix_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "secret-456")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=secret-456" in env_content
        assert key not in _read_config(_isolated_hermes_home)

    def test_case_insensitive(self, _isolated_hermes_home):
        """Keys should be uppercased regardless of input casing."""
        set_config_value("openai_api_key", "sk-test")
        env_content = _read_env(_isolated_hermes_home)
        assert "OPENAI_API_KEY=sk-test" in env_content

    def test_terminal_ssh_prefix_routes_to_env(self, _isolated_hermes_home):
        set_config_value("TERMINAL_SSH_PORT", "2222")
        env_content = _read_env(_isolated_hermes_home)
        assert "TERMINAL_SSH_PORT=2222" in env_content


# ---------------------------------------------------------------------------
# Non-secret keys → config.yaml
# ---------------------------------------------------------------------------

class TestConfigYamlRouting:
    """Regular config keys should go to config.yaml, NOT .env."""

    def test_simple_key(self, _isolated_hermes_home):
        set_config_value("model", "gpt-4o")
        config = _read_config(_isolated_hermes_home)
        assert "gpt-4o" in config
        assert "model" not in _read_env(_isolated_hermes_home)

    def test_nested_key(self, _isolated_hermes_home):
        set_config_value("terminal.backend", "docker")
        config = _read_config(_isolated_hermes_home)
        assert "docker" in config
        assert "terminal" not in _read_env(_isolated_hermes_home)

    def test_terminal_image_goes_to_config(self, _isolated_hermes_home):
        """TERMINAL_DOCKER_IMAGE doesn't match _API_KEY or _TOKEN, so config.yaml."""
        set_config_value("terminal.docker_image", "python:3.12")
        config = _read_config(_isolated_hermes_home)
        assert "python:3.12" in config

    def test_terminal_docker_cwd_mount_flag_goes_to_config_and_env(self, _isolated_hermes_home):
        set_config_value("terminal.docker_mount_cwd_to_workspace", "true")
        config = _read_config(_isolated_hermes_home)
        env_content = _read_env(_isolated_hermes_home)
        assert "docker_mount_cwd_to_workspace: 'true'" in config or "docker_mount_cwd_to_workspace: true" in config
        assert (
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=true" in env_content
            or "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=True" in env_content
        )


# ---------------------------------------------------------------------------
# Empty / falsy values — regression tests for #4277
# ---------------------------------------------------------------------------

class TestFalsyValues:
    """config set should accept empty strings and falsy values like '0'."""

    def test_empty_string_routes_to_env(self, _isolated_hermes_home):
        """Blanking an API key should write an empty value to .env."""
        set_config_value("OPENROUTER_API_KEY", "")
        env_content = _read_env(_isolated_hermes_home)
        assert "OPENROUTER_API_KEY=" in env_content

    def test_empty_string_routes_to_config(self, _isolated_hermes_home):
        """Blanking a config key should write an empty string to config.yaml."""
        set_config_value("model", "")
        config = _read_config(_isolated_hermes_home)
        assert "model: ''" in config or "model: \"\"" in config

    def test_zero_routes_to_config(self, _isolated_hermes_home):
        """Setting a config key to '0' should write 0 to config.yaml."""
        set_config_value("verbose", "0")
        config = _read_config(_isolated_hermes_home)
        assert "verbose: 0" in config

    def test_config_command_rejects_missing_value(self):
        """config set with no value arg (None) should still exit."""
        args = argparse.Namespace(config_command="set", key="model", value=None)
        with pytest.raises(SystemExit):
            config_command(args)

    def test_config_command_accepts_empty_string(self, _isolated_hermes_home):
        """config set KEY '' should not exit — it should set the value."""
        args = argparse.Namespace(config_command="set", key="model", value="")
        config_command(args)
        config = _read_config(_isolated_hermes_home)
        assert "model" in config


# ---------------------------------------------------------------------------
# List navigation — regression tests for #17876
# ---------------------------------------------------------------------------

class TestListNavigation:
    """hermes config set must preserve YAML list fields when using numeric
    indices.  Before #17876, _set_nested would silently replace the entire
    list with a dict, destroying every sibling entry.
    """

    def _write_config(self, tmp_path, body):
        (tmp_path / "config.yaml").write_text(body)

    def test_indexed_set_preserves_sibling_list_entries(self, _isolated_hermes_home):
        """Setting custom_providers.0.api_key must not destroy entry 1."""
        self._write_config(_isolated_hermes_home, (
            "custom_providers:\n"
            "- name: provider-a\n"
            "  api_key: old-a\n"
            "  base_url: https://a.example.com\n"
            "- name: provider-b\n"
            "  api_key: old-b\n"
            "  base_url: https://b.example.com\n"
        ))

        set_config_value("custom_providers.0.api_key", "new-a")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        # The list must still be a list
        assert isinstance(reloaded["custom_providers"], list)
        assert len(reloaded["custom_providers"]) == 2
        # Entry 0 was updated
        assert reloaded["custom_providers"][0]["api_key"] == "new-a"
        assert reloaded["custom_providers"][0]["name"] == "provider-a"
        assert reloaded["custom_providers"][0]["base_url"] == "https://a.example.com"
        # Entry 1 is untouched
        assert reloaded["custom_providers"][1]["name"] == "provider-b"
        assert reloaded["custom_providers"][1]["api_key"] == "old-b"
        assert reloaded["custom_providers"][1]["base_url"] == "https://b.example.com"

    def test_indexed_set_preserves_non_targeted_fields(self, _isolated_hermes_home):
        """Setting one field in a list entry must not drop other fields."""
        self._write_config(_isolated_hermes_home, (
            "custom_providers:\n"
            "- name: provider-a\n"
            "  api_key: old\n"
            "  base_url: https://a.example.com\n"
            "  models:\n"
            "    foo: {}\n"
            "    bar: {}\n"
        ))

        set_config_value("custom_providers.0.api_key", "rotated")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        entry = reloaded["custom_providers"][0]
        assert entry["api_key"] == "rotated"
        assert entry["name"] == "provider-a"
        assert entry["base_url"] == "https://a.example.com"
        assert set(entry["models"].keys()) == {"foo", "bar"}

    def test_deeper_nesting_through_list(self, _isolated_hermes_home):
        """Navigation path mixing dict → list → dict → scalar."""
        self._write_config(_isolated_hermes_home, (
            "platforms:\n"
            "  telegram:\n"
            "    allowlist:\n"
            "    - name: alice\n"
            "      role: admin\n"
            "    - name: bob\n"
            "      role: user\n"
        ))

        set_config_value("platforms.telegram.allowlist.1.role", "admin")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        allowlist = reloaded["platforms"]["telegram"]["allowlist"]
        assert isinstance(allowlist, list)
        assert allowlist[0] == {"name": "alice", "role": "admin"}
        assert allowlist[1] == {"name": "bob", "role": "admin"}


# ---------------------------------------------------------------------------
# List-shaped string coercion — regression tests for the bug where
# `hermes config set skills.disabled '["a","b"]'` was written as a
# quoted YAML scalar instead of a real list. Before the fix at
# hermes_cli/config.py:set_config_value, downstream _normalize_string_set
# (in agent/skill_utils.py) would split a comma-shaped string but not a
# JSON-array-shaped string, so the disabled set silently contained one
# 28-char literal instead of the two intended names. Worker triage:
# see /Users/jekabs/.hermes/plans/2026-06-11-specialized-workforce-audit.md
# ---------------------------------------------------------------------------

class TestListShapedStringCoercion:
    """hermes config set must coerce JSON-array strings and plain
    comma-separated strings into real YAML lists.
    """

    def test_json_array_string_becomes_list(self, _isolated_hermes_home):
        """`set skills.disabled '["a","b"]'` → real list of 2 strings."""
        set_config_value(
            "skills.disabled", '["agency/foo", "agency/bar"]'
        )

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["skills"]["disabled"] == ["agency/foo", "agency/bar"]

    def test_json_array_on_disk_is_not_a_quoted_scalar(
        self, _isolated_hermes_home
    ):
        """The raw YAML on disk must NOT be a single-quoted string literal.
        This is the exact symptom of the original bug."""
        set_config_value(
            "skills.disabled", '["agency/foo", "agency/bar"]'
        )
        raw = _read_config(_isolated_hermes_home)

        # The bug shape: `disabled: '["agency/foo", "agency/bar"]'`
        # The fix shape: a YAML block list
        assert "'[" not in raw
        assert '"[' not in raw
        # And the list items must appear as list items, not in a string
        assert "- agency/foo" in raw
        assert "- agency/bar" in raw

    def test_comma_separated_string_becomes_list(
        self, _isolated_hermes_home
    ):
        """`set skills.disabled 'a,b'` → list of 2 strings."""
        set_config_value("skills.disabled", "agency/foo,agency/bar")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["skills"]["disabled"] == ["agency/foo", "agency/bar"]

    def test_empty_json_array_becomes_empty_list(
        self, _isolated_hermes_home
    ):
        """`set foo '[]'` → real empty list (not a string '[]')."""
        set_config_value("skills.disabled", "[]")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["skills"]["disabled"] == []

    def test_malformed_bracket_string_falls_through_safely(
        self, _isolated_hermes_home
    ):
        """A string that starts with [ but isn't valid JSON must
        stay a string. No crash, no silent split."""
        set_config_value("weird_key", "[not valid json")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["weird_key"] == "[not valid json"

    def test_terminal_list_value_round_trips_through_env_bridge(
        self, _isolated_hermes_home
    ):
        """List-typed values for terminal.* keys must still bridge to
        the matching *_ENV env var as a JSON-encoded string. This
        pins the env-bridge behavior the sibling call paths
        (terminal.docker_volumes, terminal.docker_forward_env) rely
        on.
        """
        set_config_value(
            "terminal.docker_volumes", '["/host:/workspace"]'
        )

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["terminal"]["docker_volumes"] == ["/host:/workspace"]

        env_content = _read_env(_isolated_hermes_home)
        # The env bridge uses json.dumps for lists, so the env var
        # is the JSON-encoded form.
        assert "TERMINAL_DOCKER_VOLUMES=" in env_content
        env_line = [
            l for l in env_content.splitlines()
            if l.startswith("TERMINAL_DOCKER_VOLUMES=")
        ][0]
        import json as _json
        assert _json.loads(env_line.split("=", 1)[1]) == ["/host:/workspace"]


# ---------------------------------------------------------------------------
# Bool / int / float coercion — must survive the list-shaped gate
# sitting in front of it. Regression pin.
# ---------------------------------------------------------------------------

class TestScalarCoercionStillWorks:
    """The new list-shaped gate runs before the bool/int/float
    heuristics. Make sure scalar coercion is unaffected.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("0", 0),
        ("42", 42),
        ("0.85", 0.85),
        # Note: negative numerics are NOT coerced by the original
        # code (the .isdigit() heuristic rejects the leading '-'),
        # so we don't pin that here. The list-shaped gate doesn't
        # change this behavior — the existing test set
        # TestFalsyValues already pins that '0' works.
    ])
    def test_scalar_coercion_intact(
        self, _isolated_hermes_home, raw, expected
    ):
        set_config_value("test_key", raw)
        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert reloaded["test_key"] == expected, (
            f"input {raw!r} should coerce to {expected!r}, "
            f"got {reloaded['test_key']!r}"
        )
