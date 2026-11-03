"""Platform adapters for messaging integrations (receive, send, auth, media)."""

from .base import BasePlatformAdapter, SendResult
from .event import MessageEvent

__all__ = ["BasePlatformAdapter", "MessageEvent", "SendResult"]
