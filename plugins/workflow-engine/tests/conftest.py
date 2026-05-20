"""
conftest.py — make `plugins.workflow_engine` importable.

The plugin directory is named `workflow-engine` (hyphen, per Hermes convention)
but Python module names cannot contain hyphens.  This conftest registers the
package under the underscore alias before any test module is collected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Repo root = four levels up from this file
#   plugins/workflow-engine/tests/conftest.py
_PLUGIN_DIR = Path(__file__).resolve().parent.parent          # plugins/workflow-engine/
_PLUGINS_DIR = _PLUGIN_DIR.parent                             # plugins/
_REPO_ROOT = _PLUGINS_DIR.parent                              # repo root

# Ensure repo root is on sys.path so `plugins` namespace resolves.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Register `plugins.workflow_engine` → the hyphenated directory.
def _register(dotted: str, path: Path) -> None:
    if dotted in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        dotted,
        path / "__init__.py",
        submodule_search_locations=[str(path)],
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


_register("plugins.workflow_engine", _PLUGIN_DIR)
_register("plugins.workflow_engine.dashboard", _PLUGIN_DIR / "dashboard")
_register("plugins.workflow_engine.engine", _PLUGIN_DIR / "engine")

# monkeypatch.setattr resolves "plugins.workflow_engine.X" by walking
# getattr chains. Ensure the attribute is set on the plugins namespace too.
import plugins as _plugins_mod  # noqa: E402
_plugins_mod.workflow_engine = sys.modules["plugins.workflow_engine"]
