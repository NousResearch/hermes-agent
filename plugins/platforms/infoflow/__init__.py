"""Infoflow (如流) platform adapter plugin for Hermes Gateway."""

from .adapter import InfoflowAdapter, register


def setup(plugin_context):
    """Called by the plugin loader to register this platform adapter."""
    register(plugin_context)
