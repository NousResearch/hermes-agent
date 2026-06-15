from __future__ import annotations

"""Reusable Discord client runtime primitives.

This module intentionally contains only client-construction and event-binding
mechanics.  Message routing, slash-command behavior, voice behavior, and all
legacy adapter policy stay in ``adapter.py`` until Discord native multibot v2
explicitly opts into them.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiscordRuntimeEventHandlers:
    """Callbacks registered against a single ``commands.Bot`` instance."""

    on_ready: Callable[..., Any]
    on_message: Callable[..., Any]
    on_interaction: Callable[..., Any] | None = None
    on_voice_state_update: Callable[..., Any] | None = None


class DiscordClientRuntime:
    """Owns one Discord bot client's construction/start/event binding.

    The legacy ``DiscordAdapter`` still owns behavior and lifecycle policy.  A
    future native multibot adapter can create one ``DiscordClientRuntime`` per
    identity with distinct ``agent_id``/``bot_user_id``/token resolver values.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        bot_user_id: str | None = None,
        token_resolver: Callable[[str], Any] | None = None,
        bot_factory: Callable[..., Any],
        intents_factory: Any,
        allowed_mentions_factory: Callable[[], Any],
        proxy_kwargs_factory: Callable[[str | None], Mapping[str, Any]],
    ) -> None:
        self.agent_id = agent_id
        self.bot_user_id = bot_user_id
        self._token_resolver = token_resolver
        self._bot_factory = bot_factory
        self._intents_factory = intents_factory
        self._allowed_mentions_factory = allowed_mentions_factory
        self._proxy_kwargs_factory = proxy_kwargs_factory
        self.client: Any | None = None
        self.bot_task: asyncio.Task | None = None

    def resolve_token(self) -> str:
        """Resolve this runtime's token through an explicit runtime resolver."""

        if self._token_resolver is None:
            raise RuntimeError("DiscordClientRuntime has no token resolver")
        token = self._token_resolver(self.agent_id)
        reveal = getattr(token, "reveal", None)
        if callable(reveal):
            return str(reveal())
        return str(token)

    def build_intents(
        self,
        *,
        allowed_user_ids: set[str],
        allowed_role_ids: set[int],
    ) -> Any:
        """Build Discord intents for this runtime using legacy policy."""

        intents = self._intents_factory.default()
        intents.message_content = True
        intents.dm_messages = True
        intents.guild_messages = True
        intents.members = (
            any(not entry.isdigit() for entry in allowed_user_ids)
            or bool(allowed_role_ids)
        )
        intents.voice_states = True
        return intents

    async def close_existing_client(self, client: Any | None) -> None:
        """Close an existing client before reconnect to avoid zombie websockets."""

        if client is None:
            return
        try:
            is_closed = getattr(client, "is_closed", None)
            if callable(is_closed) and is_closed():
                return
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
        finally:
            if self.client is client:
                self.client = None

    def create_client(
        self,
        *,
        allowed_user_ids: set[str],
        allowed_role_ids: set[int],
        proxy_url: str | None,
    ) -> Any:
        """Create a ``commands.Bot`` configured with safe legacy defaults."""

        intents = self.build_intents(
            allowed_user_ids=allowed_user_ids,
            allowed_role_ids=allowed_role_ids,
        )
        self.client = self._bot_factory(
            command_prefix="!",
            intents=intents,
            allowed_mentions=self._allowed_mentions_factory(),
            **dict(self._proxy_kwargs_factory(proxy_url)),
        )
        return self.client

    def register_event_handlers(self, handlers: DiscordRuntimeEventHandlers) -> None:
        """Register supported Discord events on this runtime's client."""

        if self.client is None:
            raise RuntimeError("DiscordClientRuntime client has not been created")
        self.client.event(handlers.on_ready)
        self.client.event(handlers.on_message)
        if handlers.on_interaction is not None:
            self.client.event(handlers.on_interaction)
        if handlers.on_voice_state_update is not None:
            self.client.event(handlers.on_voice_state_update)

    def start(self, token: str) -> asyncio.Task:
        """Start the Discord client in a background task."""

        if self.client is None:
            raise RuntimeError("DiscordClientRuntime client has not been created")
        self.bot_task = asyncio.create_task(self.client.start(token))
        return self.bot_task
