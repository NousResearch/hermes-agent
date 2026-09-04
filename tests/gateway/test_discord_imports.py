"""Import-safety tests for the Discord gateway adapter."""

import subprocess
import sys
import textwrap


class TestDiscordImportSafety:
    def test_module_imports_even_when_discord_dependency_is_missing(self):
        """Probe the missing-dependency import in an isolated module cache.

        Re-importing a dotted module in-process updates both ``sys.modules``
        and the parent package's ``adapter`` attribute. Restoring only the
        former leaves later Discord tests pointing at the simulated-missing
        module even though discord.py is installed.
        """
        probe = textwrap.dedent(
            """
            import builtins
            import importlib

            original_import = builtins.__import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "discord" or name.startswith("discord."):
                    raise ImportError("discord unavailable for test")
                return original_import(name, globals, locals, fromlist, level)

            builtins.__import__ = fake_import
            module = importlib.import_module("plugins.platforms.discord.adapter")
            assert module.DISCORD_AVAILABLE is False
            assert module.discord is None
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
