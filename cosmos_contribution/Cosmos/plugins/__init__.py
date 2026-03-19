"""
cosmos Plugin System

Extensible plugin architecture for adding custom functionality.
"""

from .base import Plugin, PluginManager, PluginMetadata, PluginStatus
from .loader import PluginLoader
from .registry import plugin_registry

__all__ = [
    "Plugin",
    "PluginManager",
    "PluginMetadata",
    "PluginStatus",
    "PluginLoader",
    "plugin_registry",
]
