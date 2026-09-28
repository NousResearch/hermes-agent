"""Deliver menu choices at a gateway turn boundary in the original scope."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_identity import replace_source
from gateway.wake import admit_internal_event
from tools.reaction_menu_model import MenuOption, ReactionMenu


@dataclass(frozen=True)
class MenuDelivery:
    runner: Any
    adapter: Any
    source: SessionSource
    session_key: str
    session_id: str
    profile_home: Path | None

    def _scope(self):
        if self.profile_home is None:
            return self.runner._standalone_launch_scope()
        from gateway.run import _profile_runtime_scope
        return _profile_runtime_scope(self.profile_home)

    async def selected(self, room_id: str, menu: ReactionMenu, option: MenuOption) -> None:
        with self._scope():
            entry = self.runner.session_store.lookup_by_session_key(self.session_key)
            if (room_id != self.source.chat_id or entry is None or entry.session_id != self.session_id
                    or not self.runner._is_user_authorized_for_source(self.source)):
                return
            event = MessageEvent(
                text=menu.choice_body(option), source=replace_source(self.source),
                user_id=self.source.user_id, user_name=self.source.user_name,
                internal=True, allow_gateway_control=False,
                metadata={"gateway_session_key": self.session_key, "gateway_session_id": self.session_id,
                          "gateway_session_strict": True},
            )
            await admit_internal_event(self.adapter, event)


def menu_callback(turn):
    ctx, runner = turn._ctx, turn._runner
    adapter = ctx._status_adapter
    if (ctx.source.platform != Platform.MATRIX or "reaction_menu" not in (ctx.enabled_toolsets or [])
            or adapter is None or getattr(type(adapter), "send_reaction_menu", None) is None
            or not ctx.session_key or not ctx.session_id or not ctx.source.user_id):
        return None
    source = replace_source(ctx.source)
    delivery = MenuDelivery(runner, adapter, source, ctx.session_key, ctx.session_id,
                            runner._profile_scope_key_for_source(source))
    metadata = {**(ctx._status_thread_metadata or {}), "chat_id": source.chat_id,
                "requester_user_id": source.user_id}

    async def send(menu: ReactionMenu) -> bool:
        with delivery._scope():
            if not ctx._run_still_current():
                return False
            result = await adapter.send_reaction_menu(menu, ctx.session_key, delivery.selected, metadata)
            return bool(result.success)

    def callback(menu: ReactionMenu) -> bool:
        future = asyncio.run_coroutine_threadsafe(send(menu), ctx._loop_for_step)
        try:
            return future.result(timeout=15)
        except TimeoutError:
            future.cancel()
            return False
    return callback
