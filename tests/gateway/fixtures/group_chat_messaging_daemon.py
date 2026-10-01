"""The real gateway, with its Telegram transport replaced by a file-backed test double.

Inbound messages are JSON files dropped in ``$HERMES_HOME/fake-inbox``; everything the
gateway sends is appended to ``$HERMES_HOME/fake-outbox.jsonl``. The rest is unchanged:
admission, slash dispatch, the canonical authority, the room driver and the control socket.
"""
import asyncio
import json
import os
from pathlib import Path
import runpy
import uuid

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run_adapters import GatewayAdapterLifecycleMixin

HOME = Path(os.environ['HERMES_HOME'])
INBOX, OUTBOX = HOME / 'fake-inbox', HOME / 'fake-outbox.jsonl'


class FileTelegram(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config, Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        INBOX.mkdir(exist_ok=True)
        self._poller = asyncio.create_task(self._poll())
        self._mark_connected()
        return True

    async def disconnect(self):
        poller = getattr(self, '_poller', None)
        if poller is not None:
            poller.cancel()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        line = {'chat_id': str(chat_id), 'content': content, 'thread_id': (metadata or {}).get('thread_id')}
        with OUTBOX.open('a', encoding='utf-8') as outbox:
            outbox.write(json.dumps(line) + '\n')
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}

    async def _poll(self):
        while True:
            for path in sorted(INBOX.glob('*.json')):
                item = json.loads(path.read_text(encoding='utf-8-sig'))
                path.unlink()
                source = self.build_source(chat_id=item['chat_id'], chat_type=item['chat_type'],
                                           user_id=item['user_id'], user_name=item['user_name'],
                                           chat_name=item.get('chat_name'), message_id=item['message_id'])
                await self.handle_message(MessageEvent(text=item['text'], source=source,
                                                       message_id=item['message_id']))
            await asyncio.sleep(0.05)


_instantiate = GatewayAdapterLifecycleMixin._instantiate_adapter


def _instantiate_adapter(self, platform, config):
    return FileTelegram(config) if platform == Platform.TELEGRAM else _instantiate(self, platform, config)


GatewayAdapterLifecycleMixin._instantiate_adapter = _instantiate_adapter
runpy.run_module('gateway.run', run_name='__main__')
