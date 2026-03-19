"""
cosmos Plugin Loader

Discovers and loads plugins from the filesystem.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import json
from loguru import logger

from .base import Plugin, PluginMetadata, PluginCapability


class PluginLoader:
    """
    Loads plugins from the filesystem.

    Supports:
    - Single-file plugins (plugin_name.py)
    - Package plugins (plugin_name/__init__.py)
    - Plugin manifests (plugin.json)
    """

    def __init__(self, plugins_dir: str = "./plugins"):
        """Initialize plugin loader."""
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_modules: Dict[str, Any] = {}

    def discover(self) -> List[Dict[str, Any]]:
        """
        Discover available plugins.

        Returns:
            List of plugin metadata dictionaries
        """
        plugins = []

        # Check for single-file plugins
        for py_file in self.plugins_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            metadata = self._get_plugin_metadata(py_file)
            if metadata:
                plugins.append({
                    "path": str(py_file),
                    "type": "file",
                    **metadata,
                })

        # Check for package plugins
        for pkg_dir in self.plugins_dir.iterdir():
            if not pkg_dir.is_dir():
                continue
            if pkg_dir.name.startswith("_"):
                continue

            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                continue

            metadata = self._get_plugin_metadata(pkg_dir)
            if metadata:
                plugins.append({
                    "path": str(pkg_dir),
                    "type": "package",
                    **metadata,
                })

        return plugins

    def _get_plugin_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from a plugin."""
        manifest_file = None

        if path.is_file():
            # Single file plugin - look for plugin.json next to it
            manifest_file = path.with_suffix(".json")
        else:
            # Package plugin - look for plugin.json inside
            manifest_file = path / "plugin.json"

        if manifest_file and manifest_file.exists():
            try:
                return json.loads(manifest_file.read_text())
            except Exception as e:
                logger.warning(f"Error reading plugin manifest {manifest_file}: {e}")

        # Try to extract from docstring
        return self._extract_metadata_from_code(path)

    def _extract_metadata_from_code(self, path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from plugin code."""
        try:
            if path.is_file():
                code = path.read_text()
            else:
                code = (path / "__init__.py").read_text()

            # Simple extraction from docstring
            name = path.stem if path.is_file() else path.name

            return {
                "name": name,
                "version": "1.0.0",
                "description": f"Plugin: {name}",
            }

        except Exception:
            return None

    def load(self, plugin_path: str) -> Optional[Plugin]:
        """
        Load a plugin from a path.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            Plugin instance if successful
        """
        path = Path(plugin_path)

        if not path.exists():
            logger.error(f"Plugin path not found: {plugin_path}")
            return None

        try:
            if path.is_file():
                return self._load_file_plugin(path)
            else:
                return self._load_package_plugin(path)

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {e}")
            return None

    def _load_file_plugin(self, path: Path) -> Optional[Plugin]:
        """Load a single-file plugin."""
        module_name = f"cosmos_plugin_{path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules[module_name] = module

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                return plugin_class()

            logger.warning(f"No Plugin class found in {path}")
            return None

        except Exception as e:
            logger.error(f"Error loading file plugin {path}: {e}")
            return None

    def _load_package_plugin(self, path: Path) -> Optional[Plugin]:
        """Load a package plugin."""
        module_name = f"cosmos_plugin_{path.name}"

        try:
            # Add to sys.path temporarily
            parent_dir = str(path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the package
            module = importlib.import_module(path.name)
            self._loaded_modules[module_name] = module

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if plugin_class:
                return plugin_class()

            logger.warning(f"No Plugin class found in {path}")
            return None

        except Exception as e:
            logger.error(f"Error loading package plugin {path}: {e}")
            return None

    def _find_plugin_class(self, module: Any) -> Optional[Type[Plugin]]:
        """Find the Plugin subclass in a module."""
        for name in dir(module):
            obj = getattr(module, name)

            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
            ):
                return obj

        return None

    def unload(self, module_name: str) -> bool:
        """Unload a plugin module."""
        if module_name in self._loaded_modules:
            del self._loaded_modules[module_name]

            if module_name in sys.modules:
                del sys.modules[module_name]

            logger.info(f"Unloaded plugin module: {module_name}")
            return True

        return False


def create_plugin_template(
    name: str,
    plugins_dir: str = "./plugins",
    capabilities: List[str] = None,
) -> str:
    """
    Create a new plugin template.

    Args:
        name: Plugin name
        plugins_dir: Directory to create plugin in
        capabilities: List of capability names

    Returns:
        Path to created plugin
    """
    from pathlib import Path

    plugins_path = Path(plugins_dir)
    plugins_path.mkdir(parents=True, exist_ok=True)

    plugin_file = plugins_path / f"{name.lower().replace(' ', '_')}.py"

    caps_list = capabilities or ["tool"]
    caps_enum = ", ".join([f"PluginCapability.{c.upper()}" for c in caps_list])

    template = f'''"""
{name} Plugin for cosmos

Description: A custom plugin that...
"""

from typing import Dict, Any
from Cosmos.plugins.base import (
    Plugin,
    PluginMetadata,
    PluginCapability,
)


class {name.replace(" ", "")}Plugin(Plugin):
    """
    {name} plugin implementation.
    """

    metadata = PluginMetadata(
        name="{name}",
        version="1.0.0",
        author="Your Name",
        description="A custom {name} plugin",
        capabilities=[{caps_enum}],
    )

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the plugin."""
        super().__init__(config)

    async def initialize(self) -> bool:
        """Initialize the plugin."""
        # Register tools
        self.register_tool("my_tool", self._my_tool_handler)

        return True

    async def _my_tool_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle my_tool action."""
        # Implement your tool logic here
        return {{"result": "success"}}
'''

    plugin_file.write_text(template)
    logger.info(f"Created plugin template: {plugin_file}")

    # Create manifest
    manifest = {
        "name": name,
        "version": "1.0.0",
        "author": "Your Name",
        "description": f"A custom {name} plugin",
        "capabilities": caps_list,
        "dependencies": [],
    }

    manifest_file = plugin_file.with_suffix(".json")
    manifest_file.write_text(json.dumps(manifest, indent=2))

    return str(plugin_file)
