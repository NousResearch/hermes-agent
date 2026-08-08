"""Regression test for issue #81169.

When a Feishu ``post`` message is routed via ``receive_id_type=thread_id``
(e.g. cron ``deliver=origin`` deliveries into a topic chat where the
origin snapshot carries ``thread_id``), the Feishu API rejects the create
call with code 99992402 ("field validation failed"). Previously ``send()``
only matched the regex ``content format of the post type is incorrect`` for
fallback, so the message was dropped. The fix retries the same ``post``
payload without ``thread_id`` so it lands in the main chat instead of
being lost.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Repo root + minimal lark-oapi / aiohttp mock so FeishuAdapter can import
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks() -> None:
    """Stub lark-oapi / aiohttp so the adapter can import without them."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        mod = type(sys)("lark_oapi")
        for name in (
            "lark_oapi", "lark_oapi.api.im.v1",
            "lark_oapi.event", "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, mod)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        aio = type(sys)("aiohttp")
        sys.modules.setdefault("aiohttp", aio)
        sys.modules.setdefault("aiohttp.web", aio.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.feishu.adapter import FeishuAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace()  # presence check; send() is mocked
    return adapter


def _failed_post_response() -> SimpleNamespace:
    """A failed response carrying the 99992402 error code from Feishu."""
    return SimpleNamespace(
        success=lambda: False,
        code=99992402,
        msg="field validation failed",
        data=None,
    )


def _success_response(message_id: str = "om_post_main_1") -> SimpleNamespace:
    return SimpleNamespace(
        success=lambda: True,
        code=0,
        msg="success",
        data=SimpleNamespace(message_id=message_id),
    )


def _other_failure_response() -> SimpleNamespace:
    """A failed response with a different error code — should NOT trigger fallback."""
    return SimpleNamespace(
        success=lambda: False,
        code=230020,  # arbitrary non-99992402 code
        msg="some other error",
        data=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPostThreadIdFallback:
    """``send()`` must fall back to chat_id when ``post`` + 99992402 + thread_id."""

    @pytest.mark.asyncio
    async def test_post_thread_id_99992402_triggers_retry_without_thread_id(self):
        adapter = _make_adapter()

        first_response = _failed_post_response()
        second_response = _success_response()

        send_mock = AsyncMock(side_effect=[first_response, second_response])
        with patch.object(adapter, "_feishu_send_with_retry", new=send_mock):
            result = await adapter.send(
                chat_id="oc_main_chat",
                content="**hello** markdown",
                reply_to=None,
                metadata={"thread_id": "omt_topic_abc"},
            )

        assert send_mock.call_count == 2, (
            "expected the 99992402 + thread_id + post combination to trigger a retry; "
            f"send() called _feishu_send_with_retry {send_mock.call_count} times"
        )

        # Second call must strip thread_id (metadata=None) so routing falls
        # back to chat_id and the message lands in the main chat.
        retry_kwargs = send_mock.call_args_list[1].kwargs
        assert retry_kwargs["metadata"] is None, (
            f"second call must retry without thread_id (metadata=None); got {retry_kwargs['metadata']!r}"
        )
        assert retry_kwargs["chat_id"] == "oc_main_chat"
        assert retry_kwargs["msg_type"] == "post", (
            "fallback should keep msg_type='post' — only routing (thread_id vs chat_id) changes"
        )
        # The original post payload should be preserved (markdown -> post).
        first_kwargs = send_mock.call_args_list[0].kwargs
        assert first_kwargs["payload"] == retry_kwargs["payload"], (
            "fallback payload should be identical to the first attempt"
        )
        assert first_kwargs["metadata"] == {"thread_id": "omt_topic_abc"}

        assert result.success is True
        assert result.message_id == "om_post_main_1"

    @pytest.mark.asyncio
    async def test_post_without_thread_id_does_not_retry_on_99992402(self):
        """No thread_id in metadata → no fallback (there's nothing to strip)."""
        adapter = _make_adapter()
        send_mock = AsyncMock(return_value=_failed_post_response())
        with patch.object(adapter, "_feishu_send_with_retry", new=send_mock):
            result = await adapter.send(
                chat_id="oc_main_chat",
                content="**hello**",
                reply_to=None,
                metadata=None,
            )
        assert send_mock.call_count == 1
        assert result.success is False
        assert "99992402" in (result.error or "")

    @pytest.mark.asyncio
    async def test_non_post_msg_type_does_not_retry_on_99992402(self):
        """``text`` messages are unaffected — only post has the 99992402 fallback."""
        adapter = _make_adapter()
        send_mock = AsyncMock(return_value=_failed_post_response())
        with patch.object(adapter, "_feishu_send_with_retry", new=send_mock):
            result = await adapter.send(
                chat_id="oc_main_chat",
                content="plain text",
                reply_to=None,
                metadata={"thread_id": "omt_topic_abc"},
            )
        assert send_mock.call_count == 1, (
            "only post messages should get the 99992402 thread_id fallback"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_other_error_codes_with_post_and_thread_id_do_not_retry(self):
        """A 99992402-specific fallback must not over-trigger on unrelated errors."""
        adapter = _make_adapter()
        send_mock = AsyncMock(return_value=_other_failure_response())
        with patch.object(adapter, "_feishu_send_with_retry", new=send_mock):
            result = await adapter.send(
                chat_id="oc_main_chat",
                content="**hello**",
                reply_to=None,
                metadata={"thread_id": "omt_topic_abc"},
            )
        assert send_mock.call_count == 1
        assert result.success is False

    @pytest.mark.asyncio
    async def test_post_send_success_path_unchanged(self):
        """Sanity check: a successful post send is not double-called."""
        adapter = _make_adapter()
        success = _success_response(message_id="om_ok_1")
        send_mock = AsyncMock(return_value=success)
        with patch.object(adapter, "_feishu_send_with_retry", new=send_mock):
            result = await adapter.send(
                chat_id="oc_main_chat",
                content="**hello**",
                reply_to=None,
                metadata={"thread_id": "omt_topic_abc"},
            )
        assert send_mock.call_count == 1
        assert result.success is True
        assert result.message_id == "om_ok_1"