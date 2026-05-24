"""Plugin SDK — developer toolkit for building Hermes plugins.

Provides a clean API for plugin authors to:
- Register tools, hooks, CLI commands
- Access config, memory, session state
- Log messages with plugin context
- Handle errors gracefully

Usage
-----
```python
# my_plugin/__init__.py
from hermes_cli.plugin_sdk import PluginContext, PluginSDK

def register(ctx: PluginContext):
    # Register a tool
    ctx.register_tool(
        name="my_tool",
        description="Does something useful",
        schema={"name": "my_tool", "parameters": {...}},
        handler=lambda args, **kw: my_tool_impl(**args),
    )

    # Register a lifecycle hook
    ctx.register_hook("post_tool_call", on_tool_complete)

    # Register a CLI command
    ctx.register_cli_command("my-plugin", setup_my_plugin_cli)
```

Plugin Template
---------------
See `plugins/example/` for a complete plugin template with:
- `plugin.yaml` — manifest with name, version, capabilities
- `__init__.py` — plugin code using this SDK
- `tests/` — plugin tests
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about the current plugin."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    capabilities: list[str] = field(default_factory=list)


@dataclass
class PluginContext:
    """Context object passed to plugin's register() function.

    Provides access to:
    - Tool registration
    - Hook registration
    - CLI command registration
    - Config access
    - Memory access
    - Logging
    """
    plugin_info: PluginInfo
    _register_tool_fn: Optional[Callable] = None
    _register_hook_fn: Optional[Callable] = None
    _register_cli_fn: Optional[Callable] = None
    _get_config_fn: Optional[Callable] = None
    _get_memory_fn: Optional[Callable] = None

    def register_tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Callable,
        check_fn: Optional[Callable] = None,
        requires_env: Optional[list[str]] = None,
    ) -> None:
        """Register a new tool.

        Parameters
        ----------
        name:
            Tool name (snake_case).
        description:
            Human-readable description.
        schema:
            JSON schema for the tool's parameters.
        handler:
            Function that executes the tool.
        check_fn:
            Optional function that returns True if tool requirements are met.
        requires_env:
            List of required environment variable names.
        """
        if self._register_tool_fn:
            self._register_tool_fn(
                name=name,
                toolset=f"plugin:{self.plugin_info.name}",
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
            )
            logger.info("Plugin '%s' registered tool: %s", self.plugin_info.name, name)

    def register_hook(self, hook_name: str, handler: Callable) -> None:
        """Register a lifecycle hook.

        Parameters
        ----------
        hook_name:
            Hook name (e.g. "pre_tool_call", "post_llm_call").
        handler:
            Function called when the hook fires.
        """
        if self._register_hook_fn:
            self._register_hook_fn(hook_name, handler)
            logger.info("Plugin '%s' registered hook: %s", self.plugin_info.name, hook_name)

    def register_cli_command(self, command_name: str, setup_fn: Callable) -> None:
        """Register a CLI subcommand.

        Parameters
        ----------
        command_name:
            CLI command name (e.g. "my-plugin").
        setup_fn:
            Function that takes an argparse subparser and adds subcommands.
        """
        if self._register_cli_fn:
            self._register_cli_fn(command_name, setup_fn)
            logger.info("Plugin '%s' registered CLI command: %s", self.plugin_info.name, command_name)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value (plugin-scoped).

        Plugin config lives under ``plugins.<plugin_name>.<key>`` in config.yaml.
        """
        if self._get_config_fn:
            return self._get_config_fn(f"plugins.{self.plugin_info.name}.{key}", default)
        return default

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get a memory value (plugin-scoped)."""
        if self._get_memory_fn:
            return self._get_memory_fn(f"{self.plugin_info.name}:{key}", default)
        return default

    def log_info(self, message: str) -> None:
        """Log an info message with plugin context."""
        logger.info("[%s] %s", self.plugin_info.name, message)

    def log_warning(self, message: str) -> None:
        """Log a warning message with plugin context."""
        logger.warning("[%s] %s", self.plugin_info.name, message)

    def log_error(self, message: str) -> None:
        """Log an error message with plugin context."""
        logger.error("[%s] %s", self.plugin_info.name, message)

    def get_plugin_home(self) -> str:
        """Get the plugin's home directory path."""
        from hermes_constants import get_hermes_home
        return str(get_hermes_home() / "plugins" / self.plugin_info.name)


class PluginSDK:
    """Plugin SDK — manages plugin registration and context creation."""

    def __init__(self):
        self._plugins: dict[str, PluginContext] = {}

    def create_context(
        self,
        plugin_info: PluginInfo,
        register_tool_fn: Optional[Callable] = None,
        register_hook_fn: Optional[Callable] = None,
        register_cli_fn: Optional[Callable] = None,
        get_config_fn: Optional[Callable] = None,
        get_memory_fn: Optional[Callable] = None,
    ) -> PluginContext:
        """Create a PluginContext for a plugin.

        Parameters
        ----------
        plugin_info:
            Plugin metadata.
        register_tool_fn:
            Function to register tools (from model_tools).
        register_hook_fn:
            Function to register hooks (from model_tools).
        register_cli_fn:
            Function to register CLI commands (from plugins.py).
        get_config_fn:
            Function to get config values.
        get_memory_fn:
            Function to get memory values.

        Returns
        -------
        PluginContext
        """
        ctx = PluginContext(
            plugin_info=plugin_info,
            _register_tool_fn=register_tool_fn,
            _register_hook_fn=register_hook_fn,
            _register_cli_fn=register_cli_fn,
            _get_config_fn=get_config_fn,
            _get_memory_fn=get_memory_fn,
        )
        self._plugins[plugin_info.name] = ctx
        return ctx

    def get_plugin_context(self, name: str) -> Optional[PluginContext]:
        """Get a plugin's context by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())


# Global SDK instance
_sdk = PluginSDK()


def get_sdk() -> PluginSDK:
    """Get the global PluginSDK instance."""
    return _sdk
