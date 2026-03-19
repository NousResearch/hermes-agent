"""
Cosmos Optional Modules System.

Provides a registry of optional capabilities that can be installed during setup.
"""

from .registry import OPTIONAL_MODULES, get_installed_modules, is_module_available

__all__ = ['OPTIONAL_MODULES', 'get_installed_modules', 'is_module_available']
