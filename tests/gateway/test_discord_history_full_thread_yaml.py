"""config.yaml ``discord.history_full_thread`` must reach the adapter in both forms:
a bool flag and a per-channel list (seeded into extra as-is, env as CSV)."""

import importlib
import sys

import pytest


@pytest.fixture
def adapter_mod(monkeypatch):
    monkeypatch.delenv("DISCORD_HISTORY_FULL_THREAD", raising=False)
    sys.modules.pop("plugins.platforms.discord.adapter", None)
    return importlib.import_module("plugins.platforms.discord.adapter")


def test_bool_is_seeded_into_extra_and_env(adapter_mod):
    import os

    seeded = adapter_mod._apply_yaml_config({}, {"history_full_thread": True})
    assert seeded["history_full_thread"] is True
    assert os.environ.get("DISCORD_HISTORY_FULL_THREAD") == "true"


def test_false_is_propagated(adapter_mod):
    import os

    seeded = adapter_mod._apply_yaml_config({}, {"history_full_thread": False})
    assert seeded["history_full_thread"] is False
    assert os.environ.get("DISCORD_HISTORY_FULL_THREAD") == "false"


def test_channel_list_is_seeded_as_list_and_env_as_csv(adapter_mod):
    import os

    seeded = adapter_mod._apply_yaml_config({}, {"history_full_thread": ["111", 222]})
    assert seeded["history_full_thread"] == ["111", 222]
    assert os.environ.get("DISCORD_HISTORY_FULL_THREAD") == "111,222"


def test_existing_env_var_wins(adapter_mod, monkeypatch):
    import os

    monkeypatch.setenv("DISCORD_HISTORY_FULL_THREAD", "false")
    adapter_mod._apply_yaml_config({}, {"history_full_thread": True})
    assert os.environ.get("DISCORD_HISTORY_FULL_THREAD") == "false"
