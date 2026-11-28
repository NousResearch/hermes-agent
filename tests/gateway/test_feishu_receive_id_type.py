"""Feishu ``_send_raw_message`` receive_id_type inference (#7685).

Split out of ``test_feishu.py``, which is at the 2000-line target.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import lark_oapi
    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestSendRawMessageReceiveIdType(unittest.TestCase):
    """_send_raw_message infers receive_id_type from the id prefix (#7685)."""

    def _send_type(self, chat_id):
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        created = []

        class _Message:
            @staticmethod
            def create(request):
                created.append(request)
                return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_1"))

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=_Message.create)))
        )
        with patch.object(
            FeishuAdapter, "_build_create_message_body",
            staticmethod(lambda **kwargs: SimpleNamespace(**kwargs)),
        ), patch.object(
            FeishuAdapter, "_build_create_message_request",
            staticmethod(lambda receive_id_type, request_body: SimpleNamespace(
                receive_id_type=receive_id_type, request_body=request_body)),
        ):
            asyncio.run(adapter._send_raw_message(
                chat_id=chat_id, msg_type="text", payload='{"text":"ok"}',
                reply_to=None, metadata=None,
            ))
        return created[0].receive_id_type, created[0].request_body.receive_id

    def test_receive_id_type_inferred_from_prefix(self):
        for chat_id, expected in (
            ("ou_abc123", ("open_id", "ou_abc123")),
            ("on_abc123", ("union_id", "on_abc123")),
            ("oc_abc123", ("chat_id", "oc_abc123")),
        ):
            with self.subTest(chat_id=chat_id):
                self.assertEqual(self._send_type(chat_id), expected)
