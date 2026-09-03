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

        # Snapshot what is cached right now so the broken copy this test is
        # about to create can be swapped back out at the end.
        import plugins.platforms as platforms_pkg

        _MISSING = object()
        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "plugins.platforms.discord.adapter",
                "plugins.platforms.discord",
            )
        }
        saved_parent_attr = getattr(platforms_pkg, "discord", _MISSING)

        # Purge the cached module so the import below actually re-runs the
        # module body with discord.py simulated-missing.
        monkeypatch.delitem(sys.modules, "plugins.platforms.discord.adapter", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.platforms.discord", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        module = importlib.import_module("plugins.platforms.discord.adapter")

        assert module.DISCORD_AVAILABLE is False
        assert module.discord is None

        # The import above cached a discord-less copy of the adapter under its
        # real name and rebound it on the parent package. monkeypatch.delitem
        # only restores what was there before, so when the module had not been
        # imported yet that broken copy would survive this test and every later
        # one would see ``discord is None``.
        #
        # Put back the exact module objects that were cached on entry (or drop
        # them when there were none). Restoring the original objects rather
        # than re-importing matters: fixtures built from the old module object
        # keep patching the same one the code under test uses.
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        if saved_parent_attr is _MISSING:
            if hasattr(platforms_pkg, "discord"):
                delattr(platforms_pkg, "discord")
        else:
            platforms_pkg.discord = saved_parent_attr
