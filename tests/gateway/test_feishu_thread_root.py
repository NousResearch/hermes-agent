import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.feishu import FeishuAdapter
from gateway.platforms.base import MessageType


class TestFeishuThreadRootRouting(unittest.IsolatedAsyncioTestCase):
    @patch.dict('os.environ', {'FEISHU_APP_ID': 'app', 'FEISHU_APP_SECRET': 'secret'}, clear=True)
    async def test_inbound_topic_uses_root_id_over_reply_thread_id(self):
        adapter = FeishuAdapter(PlatformConfig(extra={'require_mention': False, 'mention_policy': 'optional'}))
        captured = []
        adapter._extract_message_content = AsyncMock(return_value=('hello', MessageType.TEXT, [], [], []))
        adapter._fetch_message_text = AsyncMock(return_value='parent text')
        adapter.get_chat_info = AsyncMock(return_value={'name': 'topic group', 'type': 'group'})
        adapter._resolve_sender_profile = AsyncMock(return_value={'user_id': 'ou_user', 'user_id_alt': 'union_user', 'user_name': 'User'})
        async def capture(event):
            captured.append(event)
        adapter._dispatch_inbound_event = capture
        message = SimpleNamespace(
            message_id='om_reply',
            chat_id='oc_chat',
            chat_type='group',
            parent_id='om_parent',
            upper_message_id=None,
            root_id='om_topic_root',
            thread_id='om_reply_thread',
            content='{}',
        )
        await adapter._process_inbound_message(
            data=SimpleNamespace(event=SimpleNamespace(message=message)),
            message=message,
            sender_id=SimpleNamespace(open_id='ou_user'),
            chat_type='group',
            message_id='om_reply',
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].source.thread_id, 'om_topic_root')
        self.assertEqual(captured[0].reply_to_message_id, 'om_parent')

    def test_media_batch_compatibility_separates_different_roots(self):
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource
        from gateway.config import Platform
        from datetime import datetime
        def ev(root):
            return MessageEvent(
                text='',
                message_type=MessageType.PHOTO,
                source=SessionSource(platform=Platform.FEISHU, chat_id='oc_chat', chat_type='group', user_id='ou_user', thread_id=root),
                raw_message=None,
                message_id='m_'+root,
                media_urls=['u'],
                media_types=['image'],
                reply_to_message_id='same_parent',
                reply_to_text='same',
                timestamp=datetime.now(),
            )
        self.assertFalse(FeishuAdapter._media_batch_is_compatible(ev('root_a'), ev('root_b')))
