"""Regression tests for the Feishu post-to-text fallback on 99992402.

Feishu's API returns the generic param-invalid code ``99992402``
(``field validation failed``) for post payloads that fail server-side
validation — e.g. empty content rows or malformed post JSON.  Before the
regex covered that code, such sends burned the full retry budget and the
message was dropped.  The adapter must instead treat it like any other
post-format rejection and degrade to a plain-text send.
"""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


async def _direct(func, *args, **kwargs):
    return func(*args, **kwargs)


def _fake_client(message_api):
    return SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(
                message=message_api,
            )
        )
    )


@patch.dict(os.environ, {}, clear=True)
class TestPostFallbackOnFieldValidationError(unittest.TestCase):
    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        return FeishuAdapter(PlatformConfig())

    def test_send_falls_back_to_text_when_post_rejected_with_99992402(self):
        adapter = self._make_adapter()
        captured = []

        class _MessageAPI:
            def create(self, request):
                captured.append(request)
                if len(captured) == 1:
                    return SimpleNamespace(
                        success=lambda: False,
                        code=99992402,
                        msg="[99992402] field validation failed",
                    )
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_fallback"),
                )

        adapter._client = _fake_client(_MessageAPI())

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(chat_id="oc_chat", content="这里有 **粗体** 文本。")
            )

        self.assertTrue(result.success)
        self.assertEqual([r.request_body.msg_type for r in captured], ["post", "text"])

    def test_send_falls_back_to_text_when_post_raises_99992402(self):
        adapter = self._make_adapter()
        captured = []

        class _MessageAPI:
            def create(self, request):
                captured.append(request)
                if len(captured) == 1:
                    raise RuntimeError("[99992402] field validation failed")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_fallback"),
                )

        adapter._client = _fake_client(_MessageAPI())

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(chat_id="oc_chat", content="这里有 **粗体** 文本。")
            )

        self.assertTrue(result.success)
        self.assertEqual([r.request_body.msg_type for r in captured], ["post", "text"])

    def test_edit_message_falls_back_to_text_when_update_rejected_with_99992402(self):
        adapter = self._make_adapter()
        captured = []

        class _MessageAPI:
            def update(self, request):
                captured.append(request)
                if len(captured) == 1:
                    return SimpleNamespace(
                        success=lambda: False,
                        code=99992402,
                        msg="[99992402] field validation failed",
                    )
                return SimpleNamespace(success=lambda: True)

        adapter._client = _fake_client(_MessageAPI())

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured[0].request_body.msg_type, "post")
        self.assertEqual(captured[1].request_body.msg_type, "text")
        self.assertEqual(
            captured[1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    def test_send_keeps_original_error_when_text_fallback_also_fails(self):
        adapter = self._make_adapter()
        captured = []

        class _MessageAPI:
            def create(self, request):
                captured.append(request)
                return SimpleNamespace(
                    success=lambda: False,
                    code=99992402,
                    msg="[99992402] field validation failed",
                )

        adapter._client = _fake_client(_MessageAPI())

        with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(chat_id="oc_chat", content="这里有 **粗体** 文本。")
            )

        self.assertFalse(result.success)
        self.assertEqual([r.request_body.msg_type for r in captured], ["post", "text"])


if __name__ == "__main__":
    unittest.main()
