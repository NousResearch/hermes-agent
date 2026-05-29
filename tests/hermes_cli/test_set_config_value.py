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
        # Use a real DEFAULT_CONFIG sub-key so schema validation passes — the
        # original test used ``verbose`` which is not in the known schema.
        set_config_value("agent.gateway_timeout", "0")
        config = _read_config(_isolated_hermes_home)
        assert "gateway_timeout: 0" in config

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
            "telegram:\n"
            "  allowlist:\n"
            "    - name: alice\n"
            "      role: admin\n"
            "    - name: bob\n"
            "      role: user\n"
        ))

        # NOTE: original test path was ``platforms.telegram.allowlist.1.role``,
        # which #34067 schema validation correctly rejects (platform configs
        # live at the top level, not under a ``platforms`` namespace). Use
        # the canonical path.
        set_config_value("telegram.allowlist.1.role", "admin")

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        allowlist = reloaded["telegram"]["allowlist"]
        assert isinstance(allowlist, list)
        assert allowlist[0] == {"name": "alice", "role": "admin"}
        assert allowlist[1] == {"name": "bob", "role": "admin"}


# ---------------------------------------------------------------------------
# String-typed config values — regression tests for #47515
# ---------------------------------------------------------------------------

class TestStringTypedConfigValues:
    @pytest.mark.parametrize("value", ["off", "on", "yes", "no", "true", "false", "01"])
    def test_string_typed_values_are_not_coerced(self, _isolated_hermes_home, value):
        """Values stay strings when DEFAULT_CONFIG declares the leaf as a string."""
        set_config_value("approvals.mode", value)

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["approvals"]["mode"] == value
        assert isinstance(saved["approvals"]["mode"], str)

    @pytest.mark.parametrize("key, value, expected", [
        ("terminal.persistent_shell", "off", False),
        ("approvals.timeout", "30", 30),
    ])
    def test_non_string_defaults_keep_existing_coercion(
        self, _isolated_hermes_home, key, value, expected
    ):
        set_config_value(key, value)

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        node = saved
        for part in key.split("."):
            node = node[part]
        assert node == expected
        assert type(node) is type(expected)

    def test_unknown_keys_keep_existing_coercion(self, _isolated_hermes_home):
        set_config_value("custom.enabled", "off")

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["custom"]["enabled"] is False


# ---------------------------------------------------------------------------
# Secret redaction in display output (issue #50245)
# ---------------------------------------------------------------------------

class TestSecretRedactionInDisplay:
    """`config set`/`config show` must not echo credential values in plaintext."""

    def test_redact_config_value_masks_nested_api_key(self):
        from hermes_cli.config import redact_config_value
        secret = "cfut_SUPERSECRETTOKEN1234567890abcdef"
        model = {"default": "@cf/foo", "provider": "custom", "api_key": secret}

        out = redact_config_value(model)

        assert out["api_key"] != secret
        assert secret not in str(out)
        # Non-secret fields pass through unchanged.
        assert out["default"] == "@cf/foo"
        assert out["provider"] == "custom"

    def test_redact_config_value_walks_lists(self):
        from hermes_cli.config import redact_config_value
        secret = "sk-deadbeefdeadbeefdeadbeef"
        cfg = {"custom_providers": [{"name": "p", "api_key": secret}]}

        out = redact_config_value(cfg)

        assert secret not in str(out)
        assert out["custom_providers"][0]["name"] == "p"

    def test_redact_config_value_ignores_benign_keys(self):
        from hermes_cli.config import redact_config_value
        cfg = {"token_count": 1234, "secret_santa": "alice", "max_turns": 90}

        out = redact_config_value(cfg)

        # Exact-match only — substrings like token_count must NOT be masked.
        assert out == cfg

    def test_set_echo_masks_secret_value(self, _isolated_hermes_home, capsys):
        secret = "cfut_ANOTHERSECRET0987654321zyxwvu"
        set_config_value("model.api_key", secret)

        captured = capsys.readouterr()
        assert secret not in captured.out
        assert "Set model.api_key" in captured.out

    def test_set_echo_keeps_nonsecret_value(self, _isolated_hermes_home, capsys):
        set_config_value("model.reasoning_effort", "high")

        captured = capsys.readouterr()
        assert "Set model.reasoning_effort = high" in captured.out

# #34067: Schema validation for unknown keys
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """#34067: ``hermes config set`` must refuse to silently write unknown
    keys to config.yaml. The headline case in the issue was
    ``gateway.discord.gateway_restart_notification`` being silently accepted
    even though the correct path is ``discord.gateway_restart_notification``
    (platform configs live at the top level, not nested under ``gateway``).
    """

    def test_unknown_top_level_key_is_refused(self, _isolated_hermes_home, capsys):
        """An entirely-unknown top-level key triggers SystemExit(2) with a
        red error message."""
        with pytest.raises(SystemExit) as exc_info:
            set_config_value("totally_made_up_key", "value")
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Unknown config key" in out
        assert "totally_made_up_key" in out
        # And nothing was written to config.yaml.
        assert "totally_made_up_key" not in _read_config(_isolated_hermes_home)

    def test_unknown_subkey_is_refused(self, _isolated_hermes_home, capsys):
        """The headline #34067 bug: ``gateway`` is a valid top-level key but
        ``gateway.discord`` is not a valid sub-key (gateway only has
        strict/media_delivery_allow_dirs/trust_recent_files/...)."""
        with pytest.raises(SystemExit) as exc_info:
            set_config_value("gateway.discord.gateway_restart_notification", "false")
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Unknown config key" in out
        assert "gateway.discord.gateway_restart_notification" in out
        # Nothing was written.
        assert "discord" not in _read_config(_isolated_hermes_home).split("gateway:")[-1] if "gateway:" in _read_config(_isolated_hermes_home) else True

    def test_platforms_prefix_suggests_top_level(self, _isolated_hermes_home, capsys):
        """``platforms.discord.foo`` should suggest ``discord.foo`` since
        DEFAULT_CONFIG puts platform configs at the top level, not under
        a ``platforms:`` namespace."""
        with pytest.raises(SystemExit):
            set_config_value("platforms.discord.gateway_restart_notification", "false")
        out = capsys.readouterr().out
        assert "Did you mean" in out
        assert "discord.gateway_restart_notification" in out

    def test_close_typo_suggests_correct_key(self, _isolated_hermes_home, capsys):
        """Typo'd top-level keys should get a fuzzy-match suggestion."""
        with pytest.raises(SystemExit):
            set_config_value("disco", "false")
        out = capsys.readouterr().out
        assert "Did you mean" in out
        assert "discord" in out

    def test_typoed_subkey_suggests_sibling(self, _isolated_hermes_home, capsys):
        """``agent.max_turn`` should suggest ``agent.max_turns``."""
        with pytest.raises(SystemExit):
            set_config_value("agent.max_turn", "100")
        out = capsys.readouterr().out
        assert "agent.max_turns" in out

    def test_force_bypasses_validation(self, _isolated_hermes_home):
        """``--force`` allows writing unknown keys (forward-compat with
        config keys that a newer Hermes version adds)."""
        # Should not raise.
        set_config_value("brand_new_future_key", "value", force=True)
        # And the value WAS written.
        content = _read_config(_isolated_hermes_home)
        assert "brand_new_future_key" in content

    def test_known_top_level_key_accepted(self, _isolated_hermes_home):
        """Sanity check: real config keys still work."""
        set_config_value("terminal.backend", "docker")
        content = _read_config(_isolated_hermes_home)
        assert "backend: docker" in content

    def test_known_platform_config_accepted(self, _isolated_hermes_home):
        """Schema-defined-extensible top-level keys (platform configs) accept
        any sub-key path because PlatformConfig has dynamic ``extra`` fields."""
        # discord is a platform config — sub-keys accept anything.
        set_config_value("discord.gateway_restart_notification", "false")
        content = _read_config(_isolated_hermes_home)
        assert "gateway_restart_notification: false" in content

    def test_open_dict_mcp_servers_accepts_any_subkey(self, _isolated_hermes_home):
        """``mcp_servers.<user-named-server>.<field>`` must work for any
        user-supplied server name."""
        set_config_value("mcp_servers.my-server.command", "npx")
        content = _read_config(_isolated_hermes_home)
        assert "my-server" in content
        assert "command: npx" in content


class TestValidateConfigKey:
    """Unit tests for the validator itself."""

    @pytest.mark.parametrize("key", [
        "model",
        "terminal.backend",
        "agent.max_turns",
        "discord.gateway_restart_notification",
        "telegram.bot_token",
        "mcp_servers.foo.command",
        "providers.openrouter.api_key",
        "gateway.strict",
    ])
    def test_known_keys_pass(self, key):
        from hermes_cli.config import _validate_config_key
        is_known, _ = _validate_config_key(key)
        assert is_known, f"Expected {key!r} to validate as known"

    @pytest.mark.parametrize("key,expected_in_suggestion", [
        ("gateway.discord.gateway_restart_notification", None),  # no close suggestion
        ("platforms.discord.gateway_restart_notification", "discord.gateway_restart_notification"),
        ("disco", "discord"),
        ("agent.max_turn", "agent.max_turns"),
    ])
    def test_unknown_keys_with_suggestion(self, key, expected_in_suggestion):
        from hermes_cli.config import _validate_config_key
        is_known, suggestion = _validate_config_key(key)
        assert not is_known, f"Expected {key!r} to validate as unknown"
        if expected_in_suggestion is not None:
            assert suggestion is not None and expected_in_suggestion in suggestion, \
                f"Expected suggestion to contain {expected_in_suggestion!r}, got {suggestion!r}"
