"""Import-safety tests for the Discord gateway adapter."""

import subprocess
import sys


class TestDiscordImportSafety:
    def test_module_imports_even_when_discord_dependency_is_missing(self):
        """Import in a subprocess so fallback-mode reloads don't poison later tests."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "class _Blocker:\n"
                    "    def find_module(self, name, path=None):\n"
                    "        if name == 'discord' or name.startswith('discord.'):\n"
                    "            return self\n"
                    "    def load_module(self, name):\n"
                    "        raise ImportError(f'blocked: {name}')\n"
                    "sys.meta_path.insert(0, _Blocker())\n"
                    "for key in list(sys.modules):\n"
                    "    if key == 'discord' or key.startswith('discord.'):\n"
                    "        del sys.modules[key]\n"
                    "    if key == 'gateway.platforms.discord':\n"
                    "        del sys.modules[key]\n"
                    "from gateway.platforms.discord import check_discord_requirements, discord\n"
                    "assert check_discord_requirements() is False\n"
                    "assert discord is None\n"
                    "print('OK')\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, (
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
