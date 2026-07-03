"""Ingress interface."""
from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable

from ..models import InboundMessage

Handler = Callable[[InboundMessage], Awaitable[None]]


class Ingress(abc.ABC):
    @abc.abstractmethod
    async def run(self, handler: Handler) -> None:
        """Start receiving messages, calling `handler` for each. Blocks."""

    @abc.abstractmethod
    async def post(self, channel_id: str, text: str, thread_id: str | None = None) -> None:
        """Send a reply."""

    async def typing(self, channel_id: str) -> None:
        """Optional 'is typing' indicator. Default no-op."""
        return None
