"""Real SDK command annotations still bind after optional dependency installation."""

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig

discord = pytest.importorskip("discord")


def test_scoped_lazy_sdk_binding_registers_native_commands(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "plugins/platforms/discord"
    package = "native_discord_lazy_sdk"
    spec = importlib.util.spec_from_file_location(
        package, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    original_import = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name == "discord" or name.startswith("discord."):
            raise ImportError("SDK unavailable at first import")
        return original_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=without_sdk):
            spec.loader.exec_module(module)
        facade = importlib.import_module(f"{package}.adapter")
        assert not facade.DISCORD_AVAILABLE
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)
        assert facade.check_discord_requirements()
        adapter = facade.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
        adapter._client = facade.commands.Bot(command_prefix="!", intents=discord.Intents.none())
        adapter._register_slash_commands()
        thread = adapter._client.tree.get_command("thread")
        assert thread is not None
        assert {p.name for p in thread.parameters} == {"name", "message", "auto_archive_duration"}
    finally:
        for name in list(sys.modules):
            if name.startswith(package + "."):
                sys.modules.pop(name)
