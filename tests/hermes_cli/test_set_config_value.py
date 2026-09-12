"""Tests for set_config_value — verifying secrets route to .env and config to config.yaml."""

import argparse
import json
import logging
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import (
    config_command,
    save_env_value,
    set_config_value,
    unset_config_value,
)


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
        "API_SERVER_KEY",
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
    """Any key ending in _API_KEY, _TOKEN, or _SECRET should route to .env."""

    @pytest.mark.parametrize("key", [
        "DAYTONA_API_KEY",
        "ELEVENLABS_API_KEY",
        "SOME_FUTURE_SERVICE_API_KEY",
        "MY_CUSTOM_TOKEN",
        "WHATSAPP_BOT_TOKEN",
        "CLIENT_SECRET",
    ])
    def test_api_key_suffix_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "secret-456")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=secret-456" in env_content
        assert key not in _read_config(_isolated_hermes_home)


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


    def test_terminal_image_goes_to_config(self, _isolated_hermes_home):
        """TERMINAL_DOCKER_IMAGE doesn't match _API_KEY or _TOKEN, so config.yaml."""
        set_config_value("terminal.docker_image", "python:3.12")
        config = _read_config(_isolated_hermes_home)
        assert "python:3.12" in config

    def test_cron_script_timeout_is_recognized(self, _isolated_hermes_home, capsys):
        """The script timeout read by cron must be accepted by config set."""
        set_config_value("cron.script_timeout_seconds", "600")

        assert "not a recognized config key" not in capsys.readouterr().out
        assert "script_timeout_seconds: 600" in _read_config(_isolated_hermes_home)

    def test_memory_nudge_interval_is_recognized(self, _isolated_hermes_home, capsys):
        """The documented background-memory review interval is runtime config."""
        set_config_value("memory.nudge_interval", "0")

        assert "not a recognized config key" not in capsys.readouterr().out
        assert "nudge_interval: 0" in _read_config(_isolated_hermes_home)

    def test_terminal_docker_cwd_mount_flag_goes_to_config_and_env(self, _isolated_hermes_home):
        set_config_value("terminal.docker_mount_cwd_to_workspace", "true")
        config = _read_config(_isolated_hermes_home)
        env_content = _read_env(_isolated_hermes_home)
        assert "docker_mount_cwd_to_workspace: 'true'" in config or "docker_mount_cwd_to_workspace: true" in config
        assert (
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=true" in env_content
            or "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=True" in env_content
        )

    def test_terminal_docker_shared_key_preserves_string_values(
        self, _isolated_hermes_home, capsys
    ):
        set_config_value("terminal.docker_shared_container_key", "off")

        import yaml

        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["terminal"]["docker_shared_container_key"] == "off"
        assert "TERMINAL_DOCKER_SHARED_CONTAINER_KEY=off" in _read_env(
            _isolated_hermes_home
        )
        assert "not a recognized config key" not in capsys.readouterr().out

    def test_terminal_vercel_runtime_goes_to_config_and_env(self, _isolated_hermes_home):
        set_config_value("terminal.vercel_runtime", "python3.13")
        config = _read_config(_isolated_hermes_home)
        env_content = _read_env(_isolated_hermes_home)
        assert "vercel_runtime: python3.13" in config
        assert "TERMINAL_VERCEL_RUNTIME=python3.13" in env_content


# ---------------------------------------------------------------------------
# Empty / falsy values — regression tests for #4277
# ---------------------------------------------------------------------------

class TestFalsyValues:
    """config set should accept empty strings and falsy values like '0'."""


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


class TestConfigGetUnset:
    """config get/unset should mirror config set for scriptable workflows."""

    def test_config_get_prints_resolved_nested_value(self, _isolated_hermes_home, capsys):
        set_config_value("terminal.timeout", "120")
        capsys.readouterr()

        args = argparse.Namespace(config_command="get", key="terminal.timeout", json=False)
        config_command(args)

        assert capsys.readouterr().out.strip() == "120"


    def test_config_unset_removes_yaml_key_and_synced_env(self, _isolated_hermes_home, capsys):
        set_config_value("terminal.backend", "docker")
        assert "TERMINAL_ENV=docker" in _read_env(_isolated_hermes_home)
        capsys.readouterr()

        args = argparse.Namespace(config_command="unset", key="terminal.backend")
        config_command(args)

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home)) or {}
        assert reloaded == {}
        assert "TERMINAL_ENV=" not in _read_env(_isolated_hermes_home)
        assert "Unset terminal.backend" in capsys.readouterr().out


    def test_config_unset_removes_dotted_token_yaml_key(self, _isolated_hermes_home, capsys):
        (_isolated_hermes_home / "config.yaml").write_text(
            "platforms:\n"
            "  teams:\n"
            "    extra:\n"
            "      access_token: yaml-token\n"
            "      tenant_id: tenant\n"
        )

        args = argparse.Namespace(config_command="unset", key="platforms.teams.extra.access_token")
        config_command(args)

        import yaml
        reloaded = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert "access_token" not in reloaded["platforms"]["teams"]["extra"]
        assert reloaded["platforms"]["teams"]["extra"]["tenant_id"] == "tenant"
        assert "Unset platforms.teams.extra.access_token" in capsys.readouterr().out


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
# Unpinned-cron notice on a global model change (#59031, #44585)
# ---------------------------------------------------------------------------

def _write_cron_jobs(tmp_path, jobs):
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(
        json.dumps({"jobs": jobs}),
        encoding="utf-8",
    )


class TestCronModelChangeNotice:
    """A global model change tells the operator which unpinned jobs stay on their snapshot."""

    def test_notice_says_jobs_keep_running_and_names_the_user_owned_pin_path(
        self,
        _isolated_hermes_home,
        capsys,
    ):
        _write_cron_jobs(
            _isolated_hermes_home,
            [
                {
                    "id": "model-drift-job",
                    "enabled": True,
                    "model": None,
                    "model_snapshot": "old-model",
                }
            ],
        )

        set_config_value("model.default", "new-model")

        notice = capsys.readouterr().out
        assert "keeps running" in notice
        assert "fail closed" not in notice
        assert "hermes cron edit <job_id> --provider <provider> --model <model>" in notice
        assert "cronjob action=update" not in notice


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
        # ``custom`` is not a known top-level key, so it now requires --force
        # (schema validation, #34067); coercion behavior is unchanged.
        set_config_value("custom.enabled", "off", force=True)

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


# ---------------------------------------------------------------------------
# #34067: Schema validation for unknown keys
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """#34067: ``hermes config set`` must not report bare success for
    unrecognized keys. The key IS written (arbitrary keys are supported —
    top-level scalars bridge into os.environ for skills/external apps), but
    a post-write notice warns that Hermes may never read it and suggests the
    likely-intended path. Headline case: the plausible-but-wrong
    ``gateway.discord.gateway_restart_notification`` (correct path:
    ``discord.gateway_restart_notification``).
    """







    def test_desktop_macos_signing_identity_is_accepted(self, _isolated_hermes_home, capsys):
        """The documented TCC signing identity setting is part of the schema."""
        set_config_value("desktop.macos_signing_identity", "Hermes Local Signing")
        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["desktop"]["macos_signing_identity"] == "Hermes Local Signing"
        assert "not a recognized config key" not in capsys.readouterr().out



    def test_force_suppresses_notice(self, _isolated_hermes_home, capsys):
        """``--force`` writes unknown keys without the notice (scripted
        forward-compat writes)."""
        set_config_value("brand_new_future_key", "value", force=True)
        out = capsys.readouterr().out
        assert "not a recognized config key" not in out
        # And the value WAS written.
        content = _read_config(_isolated_hermes_home)
        assert "brand_new_future_key" in content


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
        "platforms.discord.enabled",
        "gateway.platforms.my_platform.extra.token",
        "approvals.mode",
    ])
    def test_known_keys_pass(self, key):
        from hermes_cli.config import _validate_config_key
        is_known, _ = _validate_config_key(key)
        assert is_known, f"Expected {key!r} to validate as known"

    @pytest.mark.parametrize("key,expected_in_suggestion", [
        ("gateway.discord.gateway_restart_notification", None),  # no close suggestion
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


    def test_underscore_only_first_segment_escapes(self):
        """The underscore escape only applies to the FIRST segment. A real
        typo in a sub-key (e.g. agent._max_turns) is still caught."""
        from hermes_cli.config import _validate_config_key
        is_known, suggestion = _validate_config_key("agent._max_turns")
        assert not is_known, "Sub-key typo under a known top-level key must still be flagged"


# ---------------------------------------------------------------------------
# display.skin → touch the skin file (live re-affirm broadcast)
# ---------------------------------------------------------------------------

class TestDisplaySkinTouch:
    """Setting display.skin must bump the named skin file's mtime.

    The gateway's skin watcher broadcasts ``skin.changed`` on a signature move
    of (active name, skin-file mtime). Re-affirming the already-configured skin
    (`hermes config set display.skin X` while it is already X — the recovery
    path when a surface missed the original activation) moves NEITHER part, so
    without the touch the explicit apply is invisible to every live surface.
    """

    def test_reaffirming_same_skin_moves_the_watcher_signature(self, _isolated_hermes_home):
        import os as _os
        skins = _isolated_hermes_home / "skins"
        skins.mkdir()
        skin_file = skins / "synthwave.yaml"
        skin_file.write_text("name: synthwave\ncolors:\n  background: '#1a1030'\n")
        # Age the file so an mtime bump is unambiguous even on coarse clocks.
        _os.utime(skin_file, (1_000_000_000, 1_000_000_000))

        set_config_value("display.skin", "synthwave")
        first = skin_file.stat().st_mtime
        assert first > 1_000_000_000

        _os.utime(skin_file, (1_000_000_000, 1_000_000_000))
        set_config_value("display.skin", "synthwave")  # same name, re-affirmed
        assert skin_file.stat().st_mtime > 1_000_000_000

    def test_builtin_or_missing_skin_file_is_fine(self, _isolated_hermes_home):
        """Built-ins have no user file — the set must still succeed cleanly."""
        set_config_value("display.skin", "mono")
        assert "skin: mono" in _read_config(_isolated_hermes_home)

    def test_touch_preserves_skin_file_contents(self, _isolated_hermes_home):
        skins = _isolated_hermes_home / "skins"
        skins.mkdir()
        body = "name: neon\ncolors:\n  ui_accent: '#ff33aa'\n"
        (skins / "neon.yaml").write_text(body)

        set_config_value("display.skin", "neon")
        assert (skins / "neon.yaml").read_text() == body


# ---------------------------------------------------------------------------
# Mapping guard — regression tests for #74995
# ---------------------------------------------------------------------------

class TestMappingGuard:
    """``hermes config set <section> <scalar>`` must not silently destroy an
    existing mapping.  Bare ``model`` is a documented shorthand — redirect to
    ``model.default``.  All other mapping sections are refused without --force.
    """

    def _write_config(self, tmp_path, data: dict):
        import yaml as _yaml
        (tmp_path / "config.yaml").write_text(_yaml.dump(data))

    def test_bare_model_shorthand_preserves_siblings(self, _isolated_hermes_home):
        """hermes config set model <id> → model.default, siblings survive."""
        self._write_config(_isolated_hermes_home, {
            "model": {
                "default": "gpt-4o",
                "provider": "openai-api",
                "context_length": 128_000,
                "base_url": "https://api.example.com/v1",
            }
        })
        set_config_value("model", "claude-sonnet-4-20250514")
        config_text = _read_config(_isolated_hermes_home)
        import yaml as _yaml
        parsed = _yaml.safe_load(config_text)
        assert parsed["model"]["default"] == "claude-sonnet-4-20250514"
        assert parsed["model"]["provider"] == "openai-api"
        assert parsed["model"]["context_length"] == 128_000
        assert parsed["model"]["base_url"] == "https://api.example.com/v1"

    def test_bare_model_shorthand_creates_default_when_none(self, _isolated_hermes_home):
        """Bare model shorthand still works when config is empty (legacy behaviour)."""
        set_config_value("model", "gpt-5.6-sol")
        assert "gpt-5.6-sol" in _read_config(_isolated_hermes_home)

    def test_non_model_mapping_is_refused(self, _isolated_hermes_home):
        """hermes config set terminal bash → refuse, terminal has sub-keys."""
        self._write_config(_isolated_hermes_home, {
            "terminal": {
                "backend": "docker",
                "docker_image": "python:3.12",
                "shell": "bash",
            }
        })
        with pytest.raises(SystemExit) as exc:
            set_config_value("terminal", "zsh")
        assert exc.value.code == 1

    def test_non_model_mapping_force_overwrites(self, _isolated_hermes_home):
        """hermes config set --force terminal bash → proceed, section wiped."""
        self._write_config(_isolated_hermes_home, {
            "terminal": {
                "backend": "docker",
                "shell": "bash",
            }
        })
        set_config_value("terminal", "zsh", force=True)
        import yaml as _yaml
        parsed = _yaml.safe_load(_read_config(_isolated_hermes_home))
        assert parsed["terminal"] == "zsh"

    def test_model_default_dotted_path_is_not_guarded(self, _isolated_hermes_home):
        """model.default is already a dotted path — guard must not fire."""
        self._write_config(_isolated_hermes_home, {
            "model": {
                "default": "gpt-4o",
                "provider": "openai-api",
            }
        })
        set_config_value("model.default", "claude-opus-4")
        import yaml as _yaml
        parsed = _yaml.safe_load(_read_config(_isolated_hermes_home))
        assert parsed["model"]["default"] == "claude-opus-4"
        assert parsed["model"]["provider"] == "openai-api"

    def test_model_force_overwrites_entire_section(self, _isolated_hermes_home):
        """hermes config set --force model <id> → overwrite entire section."""
        self._write_config(_isolated_hermes_home, {
            "model": {
                "default": "gpt-4o",
                "provider": "openai-api",
                "context_length": 128_000,
            }
        })
        set_config_value("model", "claude-opus-4", force=True)
        import yaml as _yaml
        parsed = _yaml.safe_load(_read_config(_isolated_hermes_home))
        assert parsed["model"] == "claude-opus-4"


class TestScalarModelSubKeyPreservation:
    """#75426: setting model.provider when model is a scalar must not lose the model id."""

    def test_scalar_model_id_preserved_after_provider_write(self, _isolated_hermes_home):
        """Seed model: gpt-4o, then set model.provider → model.default must survive."""
        import yaml

        set_config_value("model", "gpt-4o")
        set_config_value("model.provider", "openai")

        raw = _read_config(_isolated_hermes_home)
        parsed = yaml.safe_load(raw)
        model = parsed["model"]
        assert model["default"] == "gpt-4o", f"model.default lost: {model}"
        assert model["provider"] == "openai"

    def test_scalar_model_id_preserved_after_api_key_write(self, _isolated_hermes_home):
        """model.api_key must also preserve the existing scalar model id."""
        import yaml

        set_config_value("model", "claude-sonnet")
        # model.api_key is a sub-key (has a dot), so it stays in config.yaml
        set_config_value("model.api_key", "sk-test")

        raw = _read_config(_isolated_hermes_home)
        parsed = yaml.safe_load(raw)
        assert parsed["model"]["default"] == "claude-sonnet"
        assert parsed["model"]["api_key"] == "sk-test"

class TestMalformedYAMLConfigPreservation:
    """#75431: config.yaml with YAML syntax errors must not be overwritten."""

    BROKEN_CONFIG = "model: gpt-4o\nterminal:\n  backend: docker\n  broken: [this is invalid YAML"

    def _write_broken_config(self, home):
        (home / "config.yaml").write_text(self.BROKEN_CONFIG)

    def test_set_config_value_refuses_broken_yaml(self, _isolated_hermes_home, capsys):
        """set_config_value must raise, not overwrite the broken config."""
        self._write_broken_config(_isolated_hermes_home)

        with pytest.raises(RuntimeError, match="not valid YAML"):
            set_config_value("agent.max_turns", "50")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Failed to parse" in combined or "not valid YAML" in combined
        # Original config must remain intact
        raw = _read_config(_isolated_hermes_home)
        assert raw == self.BROKEN_CONFIG, f"Config was overwritten:\n{raw}"

    def test_unset_config_value_refuses_broken_yaml(self, _isolated_hermes_home, capsys):
        """unset_config_value must raise, not overwrite the broken config."""
        from hermes_cli.config import unset_config_value

        self._write_broken_config(_isolated_hermes_home)

        with pytest.raises(RuntimeError, match="not valid YAML"):
            unset_config_value("model")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Failed to parse" in combined or "not valid YAML" in combined
        raw = _read_config(_isolated_hermes_home)
        assert raw == self.BROKEN_CONFIG


# ---------------------------------------------------------------------------
# Literal dots in key paths — regression tests for #84064
# ---------------------------------------------------------------------------

class TestLiteralDotKeyEscaping:
    """``hermes config set/unset/get`` must not split a key segment on a
    literal dot.  Provider names routinely embed version numbers
    (``qwen3.5-397b-wafer``), and before the backslash-escape (#84064)
    ``providers.qwen3.5-397b-wafer.api_key`` silently created a bogus nested
    ``qwen3`` -> ``5-397b-wafer`` structure while reporting success.
    """

    def _write_config(self, tmp_path, data: dict):
        import yaml as _yaml
        (tmp_path / "config.yaml").write_text(_yaml.safe_dump(data, sort_keys=False))

    def test_split_key_path_escaped_dot(self):
        from hermes_cli.config import _split_key_path

        assert _split_key_path("providers.qwen3\\.5-397b.api_key") == [
            "providers", "qwen3.5-397b", "api_key",
        ]
        assert _split_key_path("qwen3\\.5") == ["qwen3.5"]
        assert _split_key_path("a\\.b\\.c") == ["a.b.c"]
        # Unescaped keys keep plain dot-splitting semantics.
        assert _split_key_path("terminal.backend") == ["terminal", "backend"]
        assert _split_key_path("model") == ["model"]
        # Backslash before a non-dot char is preserved verbatim.
        assert _split_key_path("win\\path.key") == ["win\\path", "key"]

    def test_set_preserves_literal_dot_in_provider_key(self, _isolated_hermes_home, capsys):
        self._write_config(_isolated_hermes_home, {
            "providers": {
                "qwen3.5-397b-wafer-non-zdr": {"api": "https://pass.wafer.ai/v1"},
                "openrouter": {"api_key": "or-keep"},
            }
        })

        set_config_value(
            "providers.qwen3\\.5-397b-wafer-non-zdr.extra_headers",
            '{"Wafer-ZDR": "required"}',
        )

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        providers = saved["providers"]
        # No bogus ``qwen3`` nesting was created; the existing entry was updated.
        assert "qwen3" not in providers
        target = providers["qwen3.5-397b-wafer-non-zdr"]
        assert target["api"] == "https://pass.wafer.ai/v1"
        # Current main coerces structured-looking values to real mappings
        # (_looks_structured_value), so the JSON string lands as a dict.
        assert target["extra_headers"] == {"Wafer-ZDR": "required"}
        # Sibling provider untouched.
        assert providers["openrouter"] == {"api_key": "or-keep"}
        # Escaped key is schema-known (providers.* is an open dict) — no warning.
        assert "not a recognized config key" not in capsys.readouterr().out

    def test_unset_removes_literal_dot_provider_key(self, _isolated_hermes_home, capsys):
        self._write_config(_isolated_hermes_home, {
            "providers": {
                "qwen3.5-397b-wafer-non-zdr": {"api": "https://pass.wafer.ai/v1"},
                "openrouter": {"api_key": "or-keep"},
            }
        })

        args = argparse.Namespace(
            config_command="unset",
            key="providers.qwen3\\.5-397b-wafer-non-zdr",
        )
        config_command(args)

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert "qwen3.5-397b-wafer-non-zdr" not in saved["providers"]
        assert saved["providers"]["openrouter"] == {"api_key": "or-keep"}
        assert "Unset providers.qwen3\\.5-397b-wafer-non-zdr" in capsys.readouterr().out

    def test_unset_nested_field_under_literal_dot_key(self, _isolated_hermes_home, capsys):
        self._write_config(_isolated_hermes_home, {
            "providers": {
                "qwen3.5-397b-wafer-non-zdr": {
                    "api": "https://pass.wafer.ai/v1",
                    "extra_headers": '{"K": "V"}',
                },
            }
        })

        args = argparse.Namespace(
            config_command="unset",
            key="providers.qwen3\\.5-397b-wafer-non-zdr.extra_headers",
        )
        config_command(args)

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        target = saved["providers"]["qwen3.5-397b-wafer-non-zdr"]
        assert "extra_headers" not in target
        assert target["api"] == "https://pass.wafer.ai/v1"

    def test_get_reads_literal_dot_provider_key(self, _isolated_hermes_home, capsys):
        self._write_config(_isolated_hermes_home, {
            "providers": {"qwen3.5-397b": {"api": "https://pass.wafer.ai/v1"}},
        })

        args = argparse.Namespace(
            config_command="get",
            key="providers.qwen3\\.5-397b.api",
            json=False,
        )
        config_command(args)

        assert capsys.readouterr().out.strip() == "https://pass.wafer.ai/v1"

    def test_unescaped_dotted_path_unchanged(self, _isolated_hermes_home):
        """Nesting semantics for plain dotted keys are untouched."""
        set_config_value("terminal.backend", "docker")

        import yaml
        saved = yaml.safe_load(_read_config(_isolated_hermes_home))
        assert saved["terminal"]["backend"] == "docker"


# ---------------------------------------------------------------------------
# Audit log: every set/unset write is recorded at INFO with key, old -> new
# ---------------------------------------------------------------------------

def _audit_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "hermes_cli.config"]


def _audit_values(line):
    """The ``<key>: <old> -> <new>`` part of an audit line, without the ``in <path> (...)`` tail
    so a pytest tmp path can never satisfy or defeat a substring assertion."""
    return line.rsplit(" in ", 1)[0]


class TestConfigWriteAuditLog:
    """A ``hermes config set``/``unset`` write logs one INFO line that names the key, what it
    replaced and what it became; credential-shaped values are masked before they reach the
    logger; a write that changes nothing never logs a change."""

    def test_write_logs_key_old_new_and_file(self, _isolated_hermes_home, caplog):
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            set_config_value("agent.max_turns", "100")
            set_config_value("agent.max_turns", "300")
            unset_config_value("agent.max_turns")
        first, rewrite, unset = _audit_lines(caplog)
        config_path = str(_isolated_hermes_home / "config.yaml")

        for line in (first, rewrite, unset):
            assert "agent.max_turns" in line and config_path in line
            assert "session=" in line and "platform=" in line
        # first write: no prior value; rewrite: old before new; unset: removed value, then nothing.
        first, rewrite, unset = (_audit_values(line) for line in (first, rewrite, unset))
        assert "<unset>" in first and first.index("<unset>") < first.index("100")
        assert rewrite.index("100") < rewrite.index("300")
        assert "<unset>" in unset and unset.index("300") < unset.index("<unset>")

    def test_noop_writes_never_log_a_fabricated_change(self, _isolated_hermes_home, caplog):
        # ``model.api_base`` is a fallback-only alias for ``model.base_url`` (issue #8919): with
        # base_url already present the write is a no-op, so old and new must both be the real,
        # unchanged base_url -- never ``<unset> -> <new>``.
        set_config_value("model.base_url", "https://old.example.com")
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            set_config_value("model.api_base", "https://new.example.com")
        (line,) = _audit_lines(caplog)
        assert "model.base_url" in line and "<unset>" not in line
        assert line.count("https://old.example.com") == 2
        assert "https://new.example.com" not in line

        # A terminal.* key present only in .env: the .env removal is logged, but there is no
        # config.yaml value to remove, so no ``config unset ... -> <unset>`` line is fabricated.
        caplog.clear()
        save_env_value("TERMINAL_ENV", "docker")
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            unset_config_value("terminal.backend")
        lines = _audit_lines(caplog)
        assert any("TERMINAL_ENV" in line and "removed" in line for line in lines)
        assert not any(line.startswith("config unset") for line in lines)

    @pytest.mark.parametrize("key, secret, raw, force", [
        ("model.api_key", "sk-live-1234567890abcdef", None, False),      # exact-match secret leaf
        ("model.api_key", "Zq7Xw2Pv9Lk4M", None, False),                 # shorter than the log floor
        ("terminal.sudo_password", "48213977261", None, False),         # numeric: coerced to int
        # a ``[``-leading value yaml-parses to a list (model.api_key has no str default to pin
        # the type): the leaf is still a secret, every element must be masked
        ("model.api_key", "sk-live-1234567890abcdef", "[sk-live-1234567890abcdef, other]", False),
        ("providers", "opaque-token-xyz-0123456789", None, True),        # --force over a nested secret
        ("OPENROUTER_API_KEY", "sk-or-supersecret-0123456789", None, False),  # .env-routed
    ])
    def test_secret_values_never_reach_the_logger(
        self, _isolated_hermes_home, caplog, key, secret, raw, force
    ):
        if force:
            # a suffix-shaped leaf (not an exact _SECRET_CONFIG_KEYS member) nested in the
            # section being replaced
            set_config_value("providers.mine.openrouter_api_key", secret)
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            set_config_value(key, "x" if force else (raw or secret), force=force)
            if key == "OPENROUTER_API_KEY":
                unset_config_value(key)
        lines = _audit_lines(caplog)
        assert lines and all(key in line for line in lines)
        for line in map(_audit_values, lines):
            assert secret not in line
            if len(secret) >= 18:
                # the log-grade mask keeps at most 6 leading + 4 trailing characters
                assert secret[6:-4] not in line
            else:
                # below the 18-character floor nothing of the secret survives, only the mask
                assert secret[:4] not in line and secret[-4:] not in line
                assert "***" in line
        if force:
            # structural masking: the nested key name survives, only its value is masked
            assert "openrouter_api_key" in caplog.text

    def test_env_line_says_cleared_when_the_slot_is_blanked(self, _isolated_hermes_home, caplog):
        # _write_anthropic_slots / use_anthropic_claude_code_credentials blank the other slot by
        # writing ""; the audit line must describe that write, not call it a "set".
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            save_env_value("ANTHROPIC_API_KEY", "sk-ant-1234567890abcdef")
            save_env_value("ANTHROPIC_API_KEY", "")
        set_line, cleared_line = _audit_lines(caplog)
        assert set_line.startswith("env ANTHROPIC_API_KEY set in ")
        assert cleared_line.startswith("env ANTHROPIC_API_KEY cleared in ")
        assert "sk-ant-1234567890abcdef" not in caplog.text

    def test_audit_line_is_single_line_and_names_the_bridged_session(
        self, _isolated_hermes_home, caplog, monkeypatch
    ):
        # Key and session fields come from the caller/environment: control characters are
        # stripped so neither can inject a second, forged log line.
        monkeypatch.setenv("HERMES_SESSION_KEY", "telegram:42\nINFO forged line")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            set_config_value("agent.max_turns\nINFO forged", "300", force=True)
        (line,) = _audit_lines(caplog)
        assert "\n" not in line and "\r" not in line
        assert "session=telegram:42" in line and "platform=telegram" in line
        assert "agent.max_turns" in line

        caplog.clear()
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
            set_config_value("agent.max_turns", "300")
        assert "session=- platform=-" in _audit_lines(caplog)[-1]

    def test_audit_line_names_the_in_process_gateway_session(
        self, _isolated_hermes_home, caplog, monkeypatch
    ):
        # gateway/pairing.py and gateway/slash_commands.py call save_env_value in-process, where
        # the session lives in gateway.session_context's ContextVars, not os.environ; the origin
        # must come from there (and the bound ContextVar beats a stale environment value).
        from gateway import session_context
        from gateway.session_context import _SESSION_ASYNC_DELIVERY, _SESSION_VARS, set_session_vars

        monkeypatch.setenv("HERMES_SESSION_KEY", "stale:1")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "stale")
        was_engaged = session_context._session_context_engaged
        tokens = set_session_vars(platform="discord", session_key="discord:77")
        try:
            with caplog.at_level(logging.INFO, logger="hermes_cli.config"):
                save_env_value("DISCORD_HOME_CHANNEL", "77")
                set_config_value("agent.max_turns", "300")
        finally:
            for var, token in zip((*_SESSION_VARS, _SESSION_ASYNC_DELIVERY), tokens):
                var.reset(token)
            session_context._session_context_engaged = was_engaged
        lines = _audit_lines(caplog)
        assert len(lines) == 2
        for line in lines:
            assert "session=discord:77 platform=discord" in line
            assert "stale" not in line
