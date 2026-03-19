"""
Cosmos Unreal Engine 5 Integration.

Provides automation for UE5 via Python API or remote connection.

Note: The 'unreal' module is only available inside the UE5 Editor.
For external control, use the RPC bridge.
"""

from .bridge import UE5Bridge
from .translator import UE5CommandTranslator

__all__ = ['UE5Bridge', 'UE5CommandTranslator']
