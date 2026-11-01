import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("lark_oapi")
pytest.importorskip("aiohttp")

from gateway.config import PlatformConfig
from gateway.session import SessionSource
from gateway.config import Platform
from plugins.platforms.feishu import adapter as upstream
from plugins.platforms.feishu.native_cards.adapter import build_adapter_class
from plugins.platforms.feishu.native_cards.state import CardTurn, RoutingHub


def response(message_id="im_test", **data):
    return NS(success=lambda: True, data=NS(message_id=message_id, **data))


def make_adapter():
    assert upstream._load_lark_oapi()
    hub = RoutingHub()
    adapter = build_adapter_class(upstream.FeishuAdapter, hub)(PlatformConfig())
    adapter._loop = asyncio.get_running_loop()
    operations = []

    def operation(name, result=None):
        def run(request):
            operations.append((name, request))
            return result or response()

        return run

    adapter._client = NS(
        cardkit=NS(
            v1=NS(
                card=NS(
                    create=operation("create", response(card_id="card_test")),
                    update=operation("update"),
                    settings=operation("settings"),
                ),
                card_element=NS(content=operation("content")),
            )
        ),
        im=NS(v1=NS(message=NS(patch=operation("patch")))),
    )
    adapter._run_blocking = AsyncMock(side_effect=lambda fn, *args: fn(*args))
    adapter._feishu_send_with_retry = AsyncMock(return_value=response())
    return adapter, hub, operations


def add_turn(adapter, *, chat="chat_a", thread="", reply="user_a", owner="owner_a"):
    turn = CardTurn(
        chat,
        thread,
        reply,
        owner_ids=(owner,),
        session_key="session_a",
        metadata={"thread_id": thread} if thread else {},
    )
    adapter._card_turns[turn.key] = turn
    return turn


def event(chat="chat_a", thread="", message_id="user_a", owner="owner_a", text="hello"):
    return NS(
        source=SessionSource(
            platform=Platform.FEISHU, chat_id=chat, thread_id=thread, user_id=owner, chat_type="dm"
        ),
        message_id=message_id,
        text=text,
    )
