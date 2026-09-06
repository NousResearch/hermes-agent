"""config.yaml `discord.allow_bots` must reach the adapter's bot gate.

Invariant: every Discord allowlist gate configurable in `config.yaml` resolves through
`PlatformConfig.extra` even when no `DISCORD_*` env var is set. `allow_bots` regressed
because `_get_allow_bots()` read env only, so a YAML-configured value was silently
ignored and bot-to-bot handoffs stayed blocked.
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


def test_yaml_allow_bots_is_seeded_into_extra(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    seeded = _seed(adapter_mod, {}, {"allow_bots": "mentions"})
    assert seeded is not None
    assert seeded.get("allow_bots") == "mentions"


def _fake_adapter(adapter_mod, extra, env_overrides=None):
    """A DiscordAdapter shell with only the gate-resolution state populated."""
    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter.config = types.SimpleNamespace(extra=extra)
    snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    snapshot.update(env_overrides or {})
    adapter._gate_env_snapshot = snapshot
    return adapter


def test_adapter_reads_allow_bots_from_extra_without_env(adapter_mod, monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    monkeypatch.setattr(adapter_mod, "_scoped_gate_env", lambda name, default="": default)

    adapter = _fake_adapter(adapter_mod, {"allow_bots": "mentions"})

    assert adapter._get_allow_bots() == "mentions"


def test_env_still_wins_over_yaml_for_allow_bots(adapter_mod, monkeypatch):
    monkeypatch.setattr(adapter_mod, "_scoped_gate_env", lambda name, default="": default)

    adapter = _fake_adapter(
        adapter_mod, {"allow_bots": "mentions"}, {"DISCORD_ALLOW_BOTS": "all"}
    )

    assert adapter._get_allow_bots() == "all"


def test_allow_bots_defaults_to_none_when_unconfigured(adapter_mod, monkeypatch):
    monkeypatch.setattr(adapter_mod, "_scoped_gate_env", lambda name, default="": default)

    adapter = _fake_adapter(adapter_mod, {})

    assert adapter._get_allow_bots() == "none"
