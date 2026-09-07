"""Platform adapters for messaging integrations (receive, send, auth, media)."""

from .base import BasePlatformAdapter, MessageEvent, SendResult

__all__ = ["BasePlatformAdapter", "MessageEvent", "SendResult"]
