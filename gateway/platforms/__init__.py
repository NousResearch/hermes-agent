"""Platform adapters for messaging integrations (receive, send, auth, media)."""

from .base import BasePlatformAdapter, SendResult, resolve_channel_project
from .event import MessageEvent

__all__ = ["BasePlatformAdapter", "MessageEvent", "SendResult", "resolve_channel_project"]
