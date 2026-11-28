"""Bundled image providers must not resolve shared helpers through the user's plugin bucket."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_bundled_provider_imports_common_when_home_plugins_shadows_framework(tmp_path):
    home = tmp_path / "hermes-home"
    (home / "plugins" / "image_gen").mkdir(parents=True)
    repo_root = Path(__file__).resolve().parents[2]

    # Import the user bucket first, like a Python entry point launched from HERMES_HOME.
    # Make the framework root available only after Python has cached that namespace.
    script = """
import sys
from pathlib import Path

home = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path[:] = [str(home)] + [
    entry for entry in sys.path if entry and Path(entry).resolve() != root
]
import plugins
assert plugins.__file__ is None, plugins.__file__

sys.path.insert(0, str(root))
from hermes_cli.plugins import PluginManager, PluginManifest

manifest = PluginManifest(
    name="openrouter",
    source="bundled",
    kind="backend",
    key="image_gen/openrouter",
    path=str(root / "plugins" / "image_gen" / "openrouter"),
)
module = PluginManager()._load_directory_module(manifest)
assert callable(module.register)
from plugins.image_gen import _common
assert Path(_common.__file__).resolve() == root / "plugins" / "image_gen" / "_common.py"
"""

    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env.pop("HERMES_BUNDLED_PLUGINS", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(home), str(repo_root)],
        cwd=home, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"bundled provider failed with a shadowing user plugin bucket:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
