"""
Infoflow platform adapter plugin - delegates to the built-in gateway adapter.

The authoritative implementation is now at:
  gateway/platforms/infoflow.py
  gateway/platforms/infoflow_frame.py

This file exists only for plugin system compatibility.
"""
from gateway.platforms.infoflow import InfoflowAdapter, check_infoflow_requirements


def register(plugin_context):
    """No-op: infoflow is now a built-in platform, registered via run.py."""
    pass
