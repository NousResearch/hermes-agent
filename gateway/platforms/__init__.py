"""
Platform adapters for messaging integrations.

Each adapter handles:
- Receiving messages from a platform
- Sending messages/responses back
- Platform-specific authentication
- Message formatting and media handling
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult
from .qqbot import QQAdapter
from .napcat import NapCatAdapter
from .yuanbao import YuanbaoAdapter

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
    "QQAdapter",
    "NapCatAdapter",
    "YuanbaoAdapter",
]
