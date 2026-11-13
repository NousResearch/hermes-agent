"""Regression for #137083: voice labels survive discord.py becoming available after plugin import.

The dependency transition is simulated; module imports and the adapter's rebinding path are real.
"""
from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.run_voice import GatewayVoiceMixin


@pytest.fixture(params=["package", "directory-plugin"])
def late_adapter(request, monkeypatch):
    from hermes_cli.plugins import PluginManager, PluginManifest

    # Exercise a real import failure, even on hosts that already have discord.py.
    original_import = builtins.__import__

    def without_discord(name, *args, **kwargs):
        if name == "discord" or name.startswith("discord."):
            raise ModuleNotFoundError("discord is unavailable before lazy installation", name="discord")
        return original_import(name, *args, **kwargs)

    prefix = "plugins.platforms.discord"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            monkeypatch.delitem(sys.modules, name)
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins/platforms/discord"
    dynamic_prefix = "hermes_plugins.discord_lazy_test"
    try:
        with monkeypatch.context() as importing:
            importing.setattr(builtins, "__import__", without_discord)
            canonical = importlib.import_module(prefix + ".adapter")
            if request.param == "directory-plugin":
                plugin = PluginManager()._load_directory_module(
                    PluginManifest(name="discord-platform", path=str(plugin_dir), kind="platform"),
                    module_name=dynamic_prefix,
                )
                module = sys.modules[plugin.__name__ + ".adapter"]
            else:
                module = canonical
        assert module.discord is None
        yield module, canonical
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + ".") or name.startswith(dynamic_prefix):
                sys.modules.pop(name, None)


def _bind_after_install(module, monkeypatch):
    import pm

    # No package manager or network: expose the installed dependency, then run the
    # real adoption code that rebinds only this adapter module's globals.
    monkeypatch.setattr(pm, "ensure_import", lambda name: importlib.import_module(name))
    assert module.check_discord_requirements()
    assert module.discord is importlib.import_module("discord")


@pytest.mark.parametrize("kind,expected_name,expected_topic", [
    ("channel", "Hermes Server / #ops", "Incident triage"),
    ("thread", "Hermes Server / #ops / original question", None),
    ("forum-thread", "Hermes Server / ops / original question", "Incident triage"),
])
def test_late_discord_binding_reaches_voice_source(
    late_adapter, monkeypatch, kind, expected_name, expected_topic,
):
    module, canonical = late_adapter
    _bind_after_install(module, monkeypatch)
    if module is not canonical:
        assert canonical.discord is None  # a fixed canonical import would read the wrong instance
    adapter = object.__new__(module.DiscordAdapter)
    adapter._semantic_thread_renames = module.SemanticThreadRenames()
    guild = SimpleNamespace(id=1, name="Hermes Server", get_member=lambda uid: None)
    parent = SimpleNamespace(id=700, name="ops", topic="Incident triage", guild=guild)
    channel = parent
    if kind != "channel":
        if kind == "forum-thread":
            parent = MagicMock(spec=module.discord.ForumChannel)
            parent.id, parent.name, parent.topic, parent.guild = 700, "ops", "Incident triage", guild
        channel = MagicMock(spec=module.discord.Thread)
        channel.id, channel.name, channel.topic = 800, "generated title", None
        channel.parent, channel.parent_id, channel.guild = parent, 700, guild
        with adapter._semantic_thread_renames.attempt("800", "original question", "generated title"):
            pass

    assert adapter._guild_channel_labels(channel) == (expected_name, expected_topic)
    adapter._voice_sources = {1: SessionSource(
        platform=Platform.DISCORD, chat_id=str(channel.id), chat_type="thread" if kind != "channel" else "group",
        user_id="42", user_name="Alice", thread_id="800" if kind != "channel" else None,
        guild_id="1", parent_chat_id="700" if kind != "channel" else None,
        message_id="123", chat_name="stale join name", chat_topic="stale topic",
    ).to_dict()}
    adapter._client = SimpleNamespace(get_guild=lambda gid: guild, get_channel=lambda cid: channel)
    source = GatewayVoiceMixin._voice_input_source(adapter, 1, 42, channel.id)
    assert (source.chat_name, source.chat_topic) == (expected_name, expected_topic)
    assert source.chat_id == str(channel.id)
    assert source.thread_id == ("800" if kind != "channel" else None)
    assert source.user_id == "42"
    assert source.user_name == "Alice"
    assert source.message_id is None


def test_labels_without_dependency_fail_explicitly(late_adapter):
    module, _ = late_adapter
    adapter = object.__new__(module.DiscordAdapter)
    channel = SimpleNamespace(id=700, name="ops", topic="Incident triage")
    with pytest.raises(RuntimeError, match="Discord dependency unavailable"):
        adapter._guild_channel_labels(channel)


def test_slash_auth_uses_owning_adapter_binding(late_adapter, monkeypatch):
    module, canonical = late_adapter
    adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    interaction = SimpleNamespace(channel=None, channel_id=None, user=SimpleNamespace(id=42, name="Alice"))
    assert adapter._evaluate_slash_authorization(interaction) == (False, "discord dependency unavailable")
    _bind_after_install(module, monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    if module is not canonical:
        assert canonical.discord is None
    assert adapter._evaluate_slash_authorization(interaction) == (True, None)
