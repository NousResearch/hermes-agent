"""
Hermes Universal Platform Adapter Layer.
"""

from hermes_platform.factory import get_platform_adapter, detect_platform_type
from hermes_platform.common.adapter import PlatformAdapter, PlatformType

__all__ = [
    "get_platform_adapter",
    "detect_platform_type",
    "PlatformAdapter",
    "PlatformType",
]
