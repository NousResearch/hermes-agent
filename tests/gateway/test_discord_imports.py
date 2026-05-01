"""Import-safety tests for the Discord gateway adapter."""

import builtins
import importlib
import sys


class TestDiscordImportSafety:
    def test_module_imports_even_when_discord_dependency_is_missing(self, monkeypatch):
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "discord" or name.startswith("discord."):
                raise ImportError("discord unavailable for test")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.delitem(sys.modules, "gateway.platforms.discord", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        module = importlib.import_module("gateway.platforms.discord")

        assert module.DISCORD_AVAILABLE is False
        assert module.discord is None
