"""Dispatch room-threading control to the receiving transport."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

from agent.i18n import t
from gateway.platforms.event import MessageEvent

if TYPE_CHECKING:
    from gateway.run import GatewayRunner


class GatewayThreadingCommandsMixin:
    async def _handle_threading_command(self: "GatewayRunner", event: MessageEvent) -> str:
        adapter = self._intake_adapter_for(event.source)
        if adapter is None or not event.source.chat_id:
            return t("threading.unavailable")
        # Thread creation is adapter policy, so persist on the bot's home, not a routed runtime.
        home = self._authorization_home_for_source(event.source)
        from gateway.run import _async_profile_runtime_scope

        scope = _async_profile_runtime_scope(home) if home is not None else nullcontext()
        async with scope:
            return await adapter.configure_room_threading(
                str(event.source.chat_id), event.get_command_args().strip().lower())
