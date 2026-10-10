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

        # Importing under simulated dependency absence mutates both sys.modules
        # and the parent package's cached attributes. MonkeyPatch restores the
        # mapping entries but not those package attributes, which made later
        # Discord tests resolve the simulated ``discord=None`` adapter. Snapshot
        # and restore both layers explicitly so this import-safety probe is
        # hermetic under the full gateway suite.
        missing = object()
        module_names = (
            "plugins.platforms.discord.adapter",
            "plugins.platforms.discord",
        )
        old_modules = {
            name: sys.modules[name] for name in module_names if name in sys.modules
        }
        platforms_package = importlib.import_module("plugins.platforms")
        old_discord_attr = getattr(platforms_package, "discord", missing)
        old_adapter_attr = (
            getattr(old_discord_attr, "adapter", missing)
            if old_discord_attr is not missing
            else missing
        )

        try:
            for name in module_names:
                sys.modules.pop(name, None)
            with monkeypatch.context() as isolated:
                isolated.setattr(builtins, "__import__", fake_import)
                module = importlib.import_module("plugins.platforms.discord.adapter")

                assert module.DISCORD_AVAILABLE is False
                assert module.discord is None
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if name in old_modules:
                    sys.modules[name] = old_modules[name]
            if old_discord_attr is missing:
                try:
                    delattr(platforms_package, "discord")
                except AttributeError:
                    pass
            else:
                setattr(platforms_package, "discord", old_discord_attr)
                if old_adapter_attr is missing:
                    try:
                        delattr(old_discord_attr, "adapter")
                    except AttributeError:
                        pass
                else:
                    setattr(old_discord_attr, "adapter", old_adapter_attr)
