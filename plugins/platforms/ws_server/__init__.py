"""
WebSocket Server platform adapter for Hermes Agent.
Listens for incoming WebSocket connections from frontend clients.
"""

from .adapter import register

__all__ = ["register"]