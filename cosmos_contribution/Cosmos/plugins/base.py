"""
cosmos Plugin Base Classes

"With my new invention, the plugin system, anyone can add to my genius!"

Base classes and interfaces for the cosmos plugin system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from loguru import logger


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class PluginCapability(Enum):
    """Plugin capability types."""
    AGENT = "agent"  # Provides specialist agents
    TOOL = "tool"  # Provides tools/actions
    PROVIDER = "provider"  # Data provider
    INTEGRATION = "integration"  # External service integration
    UI = "ui"  # UI components
    PROCESSOR = "processor"  # Data processor
    STORAGE = "storage"  # Storage backend
    HEALTH = "health"  # Health provider


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    capabilities: List[PluginCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    homepage: str = ""
    license: str = "MIT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
            "homepage": self.homepage,
            "license": self.license,
        }


class Plugin(ABC):
    """
    Base class for cosmos plugins.

    Plugins extend cosmos's functionality by providing:
    - New agents and tools
    - Data providers and integrations
    - UI components
    - Processing pipelines
    """

    # Plugin metadata (override in subclass)
    metadata: PluginMetadata = PluginMetadata(name="BasePlugin")

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin."""
        self.config = config or {}
        self.status = PluginStatus.UNLOADED
        self._hooks: Dict[str, List[Callable]] = {}
        self._tools: Dict[str, Callable] = {}
        self._agents: Dict[str, Type] = {}

    @property
    def id(self) -> str:
        """Plugin ID."""
        return self.metadata.id

    @property
    def name(self) -> str:
        """Plugin name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Plugin version."""
        return self.metadata.version

    # ========== Lifecycle Methods ==========

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Called when the plugin is first loaded.
        Should set up resources and validate configuration.

        Returns:
            True if initialization succeeded
        """
        pass

    async def activate(self) -> bool:
        """
        Activate the plugin.

        Called when the plugin is enabled.
        Should start background tasks and register hooks.

        Returns:
            True if activation succeeded
        """
        self.status = PluginStatus.ACTIVE
        return True

    async def deactivate(self) -> bool:
        """
        Deactivate the plugin.

        Called when the plugin is disabled.
        Should stop background tasks and unregister hooks.

        Returns:
            True if deactivation succeeded
        """
        self.status = PluginStatus.LOADED
        return True

    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.

        Called when the plugin is unloaded.
        Should clean up all resources.

        Returns:
            True if shutdown succeeded
        """
        self.status = PluginStatus.UNLOADED
        return True

    # ========== Extension Points ==========

    def get_tools(self) -> Dict[str, Callable]:
        """
        Get tools provided by this plugin.

        Returns:
            Dictionary of tool_name -> tool_function
        """
        return self._tools

    def get_agents(self) -> Dict[str, Type]:
        """
        Get agent types provided by this plugin.

        Returns:
            Dictionary of agent_name -> agent_class
        """
        return self._agents

    def get_hooks(self) -> Dict[str, List[Callable]]:
        """
        Get hooks registered by this plugin.

        Returns:
            Dictionary of hook_name -> [handler_functions]
        """
        return self._hooks

    # ========== Registration Helpers ==========

    def register_tool(self, name: str, handler: Callable):
        """Register a tool."""
        self._tools[name] = handler
        logger.debug(f"Plugin {self.name} registered tool: {name}")

    def register_agent(self, name: str, agent_class: Type):
        """Register an agent type."""
        self._agents[name] = agent_class
        logger.debug(f"Plugin {self.name} registered agent: {name}")

    def register_hook(self, hook_name: str, handler: Callable):
        """Register a hook handler."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(handler)
        logger.debug(f"Plugin {self.name} registered hook: {hook_name}")

    # ========== Configuration ==========

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value

    def validate_config(self) -> bool:
        """
        Validate plugin configuration.

        Override to implement custom validation.

        Returns:
            True if configuration is valid
        """
        return True


class PluginManager:
    """
    Manages cosmos plugins.

    Handles:
    - Plugin discovery and loading
    - Lifecycle management
    - Hook dispatching
    - Tool and agent registration
    """

    def __init__(self, plugins_dir: str = "./plugins"):
        """Initialize plugin manager."""
        from pathlib import Path

        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        self.plugins: Dict[str, Plugin] = {}
        self._global_hooks: Dict[str, List[Callable]] = {}
        self._global_tools: Dict[str, Callable] = {}
        self._global_agents: Dict[str, Type] = {}

    # ========== Plugin Management ==========

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin instance."""
        if plugin.id in self.plugins:
            logger.warning(f"Plugin already registered: {plugin.name}")
            return False

        self.plugins[plugin.id] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
        return True

    def unregister(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        if plugin_id not in self.plugins:
            return False

        plugin = self.plugins.pop(plugin_id)
        self._unregister_extensions(plugin)
        logger.info(f"Unregistered plugin: {plugin.name}")
        return True

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self.plugins.get(plugin_id)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins."""
        return [
            {
                **p.metadata.to_dict(),
                "status": p.status.value,
            }
            for p in self.plugins.values()
        ]

    # ========== Lifecycle ==========

    async def initialize_plugin(self, plugin_id: str) -> bool:
        """Initialize a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False

        try:
            plugin.status = PluginStatus.LOADING

            # Check dependencies
            for dep in plugin.metadata.dependencies:
                if dep not in self.plugins:
                    logger.error(f"Plugin {plugin.name} missing dependency: {dep}")
                    plugin.status = PluginStatus.ERROR
                    return False

            # Initialize
            if await plugin.initialize():
                plugin.status = PluginStatus.LOADED
                logger.info(f"Initialized plugin: {plugin.name}")
                return True
            else:
                plugin.status = PluginStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Error initializing plugin {plugin.name}: {e}")
            plugin.status = PluginStatus.ERROR
            return False

    async def activate_plugin(self, plugin_id: str) -> bool:
        """Activate a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin or plugin.status != PluginStatus.LOADED:
            return False

        try:
            if await plugin.activate():
                self._register_extensions(plugin)
                logger.info(f"Activated plugin: {plugin.name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error activating plugin {plugin.name}: {e}")
            return False

    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """Deactivate a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin or plugin.status != PluginStatus.ACTIVE:
            return False

        try:
            if await plugin.deactivate():
                self._unregister_extensions(plugin)
                logger.info(f"Deactivated plugin: {plugin.name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deactivating plugin {plugin.name}: {e}")
            return False

    async def shutdown_plugin(self, plugin_id: str) -> bool:
        """Shutdown a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False

        try:
            if plugin.status == PluginStatus.ACTIVE:
                await self.deactivate_plugin(plugin_id)

            if await plugin.shutdown():
                self.unregister(plugin_id)
                logger.info(f"Shutdown plugin: {plugin.name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error shutting down plugin {plugin.name}: {e}")
            return False

    async def initialize_all(self):
        """Initialize all registered plugins."""
        for plugin_id in list(self.plugins.keys()):
            await self.initialize_plugin(plugin_id)

    async def activate_all(self):
        """Activate all loaded plugins."""
        for plugin_id, plugin in list(self.plugins.items()):
            if plugin.status == PluginStatus.LOADED:
                await self.activate_plugin(plugin_id)

    async def shutdown_all(self):
        """Shutdown all plugins."""
        for plugin_id in list(self.plugins.keys()):
            await self.shutdown_plugin(plugin_id)

    # ========== Extension Registration ==========

    def _register_extensions(self, plugin: Plugin):
        """Register plugin extensions globally."""
        # Register tools
        for name, handler in plugin.get_tools().items():
            self._global_tools[f"{plugin.id}:{name}"] = handler

        # Register agents
        for name, agent_class in plugin.get_agents().items():
            self._global_agents[f"{plugin.id}:{name}"] = agent_class

        # Register hooks
        for hook_name, handlers in plugin.get_hooks().items():
            if hook_name not in self._global_hooks:
                self._global_hooks[hook_name] = []
            self._global_hooks[hook_name].extend(handlers)

    def _unregister_extensions(self, plugin: Plugin):
        """Unregister plugin extensions."""
        # Remove tools
        for name in plugin.get_tools().keys():
            self._global_tools.pop(f"{plugin.id}:{name}", None)

        # Remove agents
        for name in plugin.get_agents().keys():
            self._global_agents.pop(f"{plugin.id}:{name}", None)

        # Remove hooks
        for hook_name, handlers in plugin.get_hooks().items():
            if hook_name in self._global_hooks:
                for handler in handlers:
                    if handler in self._global_hooks[hook_name]:
                        self._global_hooks[hook_name].remove(handler)

    # ========== Hook Dispatching ==========

    async def dispatch_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Dispatch a hook to all registered handlers.

        Returns:
            List of results from all handlers
        """
        results = []
        handlers = self._global_hooks.get(hook_name, [])

        for handler in handlers:
            try:
                result = handler(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                logger.error(f"Hook handler error: {e}")

        return results

    # ========== Tool Access ==========

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a registered tool."""
        return self._global_tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._global_tools.keys())

    # ========== Agent Access ==========

    def get_agent_class(self, agent_name: str) -> Optional[Type]:
        """Get a registered agent class."""
        return self._global_agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self._global_agents.keys())
