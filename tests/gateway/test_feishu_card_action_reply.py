"""Tests for Feishu card-action reply routing (#136031).

A card button click is routed as a synthetic COMMAND whose ``message_id``
becomes the gateway's reply anchor. The action token (``c-…``) is not a
message id, so Feishu rejected the reply with 99992354 and the plain-text
fallback failed the same way — the click got no response at all.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks():
    """Provide stubs for lark-oapi / aiohttp.web so the import succeeds."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = MagicMock()
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = MagicMock()
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _make_adapter() -> FeishuAdapter:
    """Create a FeishuAdapter with mocked internals."""
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    return adapter


def _card_action_data(
    *,
    open_message_id: str = "",
    token: str = "c-token-1",
) -> SimpleNamespace:
    context = SimpleNamespace(open_chat_id="oc_chat1")
    if open_message_id:
        context.open_message_id = open_message_id
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=context,
            operator=SimpleNamespace(open_id="ou_user1"),
            action=SimpleNamespace(tag="button", value={"media_pick": {"item_id": "i1"}}),
        ),
    )


def _patch_dispatch(adapter: FeishuAdapter) -> AsyncMock:
    """Patch the heavy pipeline dispatch out; keep only the routed MessageEvent kwargs."""
    adapter._is_card_action_duplicate = MagicMock(return_value=False)
    dispatch = AsyncMock()
    adapter._dispatch_synthetic_event = dispatch
    return dispatch


class TestCardActionMessageId:
    @pytest.mark.asyncio
    async def test_uses_card_message_id_for_reply(self):
        adapter = _make_adapter()
        dispatch = _patch_dispatch(adapter)
        await adapter._handle_card_action_event(
            _card_action_data(open_message_id="om_card1", token="c-token-om")
        )

        assert dispatch.call_args[1]["message_id"] == "om_card1"

    @pytest.mark.asyncio
    async def test_without_card_message_id_keeps_action_token(self):
        adapter = _make_adapter()
        dispatch = _patch_dispatch(adapter)
        await adapter._handle_card_action_event(_card_action_data(token="c-token-2"))

        assert dispatch.call_args[1]["message_id"] == "c-token-2"


def _invalid_id_response() -> SimpleNamespace:
    return SimpleNamespace(success=lambda: False, code=99992354, msg="The request you send is not a valid open_message_id")


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_new"))


class TestInvalidReplyIdFallback:
    @pytest.mark.asyncio
    async def test_invalid_reply_id_posts_new_message(self):
        adapter = _make_adapter()
        raw = AsyncMock(side_effect=lambda **kw: _ok_response() if kw["reply_to"] is None else _invalid_id_response())
        adapter._send_raw_message = raw

        response = await adapter._feishu_send_with_retry(
            chat_id="oc_chat1", msg_type="text", payload="{}", reply_to="c-token-3", metadata=None,
        )

        assert response.success() is True
        assert [c.kwargs["reply_to"] for c in raw.await_args_list] == ["c-token-3", None]

    @pytest.mark.asyncio
    async def test_invalid_reply_id_inside_thread_stays_put(self):
        adapter = _make_adapter()
        raw = AsyncMock(return_value=_invalid_id_response())
        adapter._send_raw_message = raw

        response = await adapter._feishu_send_with_retry(
            chat_id="oc_chat1", msg_type="text", payload="{}",
            reply_to="c-token-4", metadata={"thread_id": "om_thread"},
        )

        assert response.success() is False
        assert raw.await_count == 1
