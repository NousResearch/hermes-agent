"""Deprecated warning-category import path for plugin compatibility notices.

Runtime compatibility behavior is owned by :mod:`plugin_runtime.compat`. This module
exists only because the documented warnings filter path is an external contract.
"""

from plugin_runtime.compat_warning import HermesPluginCompatWarning

__all__ = ("HermesPluginCompatWarning",)
