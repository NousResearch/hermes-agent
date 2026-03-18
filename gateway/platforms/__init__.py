"""
Platform adapters for messaging integrations.

Each adapter handles:
- Receiving messages from a platform
- Sending messages/responses back
- Platform-specific authentication
- Message formatting and media handling
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult

# Conditional imports for optional adapters
try:
    from .mission_control import MissionControlAdapter, check_mc_requirements
    _mc_available = True
except ImportError:
    _mc_available = False
    MissionControlAdapter = None  # type: ignore
    check_mc_requirements = None  # type: ignore

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
]

if _mc_available:
    __all__.extend(["MissionControlAdapter", "check_mc_requirements"])
