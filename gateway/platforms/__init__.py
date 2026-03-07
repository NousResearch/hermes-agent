"""
Platform adapters for messaging integrations.

Each adapter handles:
- Receiving messages from a platform
- Sending messages/responses back
- Platform-specific authentication
- Message formatting and media handling
"""

from .base import BasePlatformAdapter, MessageEvent, SendResult
from .clawsuite import ClawSuiteAdapter, check_clawsuite_requirements

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
    "ClawSuiteAdapter",
    "check_clawsuite_requirements",
]
