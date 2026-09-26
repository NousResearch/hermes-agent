"""config.yaml `discord.command_sync_policy` must reach the slash-command sync mode.

Invariant: a policy configured in `config.yaml` (``discord:`` block, ``platforms.discord`` or
its ``extra``) resolves through ``PlatformConfig.extra`` even when no
``DISCORD_COMMAND_SYNC_POLICY`` env var is set. The key was silently ignored because
``_get_discord_command_sync_policy()`` resolved env only — the documented
``hermes config set discord.command_sync_policy bulk`` surface wrote a value nothing read,
so users debugging a stuck sync had to discover the env var by accident (#123629).
"""

import importlib
import sys
import types

import pytest


@pytest.fixture
def adapter_mod():
    sys.modules.pop("plugins.platforms.discord.adapter", None)
    return importlib.import_module("plugins.platforms.discord.adapter")


def _seed(adapter_mod, yaml_cfg, discord_cfg):
    return adapter_mod._apply_yaml_config(yaml_cfg, discord_cfg)


def _fake_adapter(adapter_mod, extra, env_overrides=None):
    """A DiscordAdapter shell with only the policy-resolution state populated."""
    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter.platform = __import__(
        "gateway.config", fromlist=["Platform"]
    ).Platform.DISCORD
    adapter.config = types.SimpleNamespace(extra=extra)
    snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    snapshot.update(env_overrides or {})
    adapter._gate_env_snapshot = snapshot
    return adapter


def test_yaml_policy_is_seeded_into_extra(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    seeded = _seed(adapter_mod, {}, {"command_sync_policy": "bulk"})
    assert seeded is not None
    assert seeded.get("command_sync_policy") == "bulk"


def test_yaml_policy_from_platform_extra_block(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    seeded = _seed(
        adapter_mod,
        {"platforms": {"discord": {"extra": {"command_sync_policy": "off"}}}},
        {},
    )
    assert seeded is not None
    assert seeded.get("command_sync_policy") == "off"


def test_adapter_reads_policy_from_extra_without_env(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    monkeypatch.setattr(
        adapter_mod, "_scoped_gate_env", lambda name, default="": default
    )

    adapter = _fake_adapter(adapter_mod, {"command_sync_policy": "bulk"})

    assert adapter._get_discord_command_sync_policy() == "bulk"


def test_yaml_bare_off_parses_as_bool_false(adapter_mod, monkeypatch):
    """YAML 1.1 turns a bare ``off`` into False; the policy must still resolve to ``off``."""
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    monkeypatch.setattr(
        adapter_mod, "_scoped_gate_env", lambda name, default="": default
    )

    adapter = _fake_adapter(adapter_mod, {"command_sync_policy": False})

    assert adapter._get_discord_command_sync_policy() == "off"


def test_env_value_still_wins_over_extra(adapter_mod, monkeypatch):
    """The env var keeps its legacy precedence: the .env workaround from #123629 stays authoritative."""
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)

    adapter = _fake_adapter(
        adapter_mod,
        {"command_sync_policy": "off"},
        env_overrides={"DISCORD_COMMAND_SYNC_POLICY": "bulk"},
    )

    assert adapter._get_discord_command_sync_policy() == "bulk"


def test_invalid_policy_falls_back_to_safe(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    monkeypatch.setattr(
        adapter_mod, "_scoped_gate_env", lambda name, default="": default
    )

    adapter = _fake_adapter(adapter_mod, {"command_sync_policy": "fast"})

    assert adapter._get_discord_command_sync_policy() == "safe"


def test_unset_everything_defaults_to_safe(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_COMMAND_SYNC_POLICY", raising=False)
    monkeypatch.setattr(
        adapter_mod, "_scoped_gate_env", lambda name, default="": default
    )

    adapter = _fake_adapter(adapter_mod, {})

    assert adapter._get_discord_command_sync_policy() == "safe"
