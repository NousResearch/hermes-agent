"""
cosmos Tools Package

"Good news, everyone! I've invented tools that do the work for you!"

Comprehensive tooling for sysadmin, security, and productivity.
"""

from typing import Dict, Any

# Lazy imports to avoid circular dependencies
_sysadmin_tools = None
_security_tools = None


def get_sysadmin_tools():
    """Get sysadmin tools module."""
    global _sysadmin_tools
    if _sysadmin_tools is None:
        from Cosmos.tools import sysadmin as _sysadmin_tools
    return _sysadmin_tools


def get_security_tools():
    """Get security tools module."""
    global _security_tools
    if _security_tools is None:
        from Cosmos.tools import security as _security_tools
    return _security_tools


__all__ = [
    "get_sysadmin_tools",
    "get_security_tools",
]
