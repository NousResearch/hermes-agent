"""
Platform adapters for messaging integrations.

Each adapter handles:
- Receiving messages from a platform
- Sending messages/responses back
- Platform-specific authentication
- Message formatting and media handling
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
]
