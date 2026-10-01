"""Shared doubles for messaging control of canonical Group Chats: a receiving Bot, a runner
with one real canonical authority store, and messages from people in private or shared chats."""
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_state_runtime import begin_runtime_epoch

OWNER = 'uid:501'


class Bot:
    """The adapter that received a command; only its config and sends matter here."""
    typed_command_prefix = '/'

    def __init__(self, *, dm_admins=('alice',), group_admins=('alice', 'bob')):
        self.config = PlatformConfig(enabled=True, extra={
            'allow_admin_from': list(dm_admins), 'group_allow_admin_from': list(group_admins)})
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))


def runner_for(authority, bot):
    runner = SimpleNamespace(session_authority=authority, _primary_profile_name='default')
    runner._transport_owner = lambda source: (bot, None)
    runner._typed_command_prefix_for = lambda platform: '/'
    runner._thread_metadata_for_source = lambda source: (
        {'thread_id': source.thread_id} if source.thread_id else None)
    authority.runner = runner
    return runner


def authority_for(home, db):
    return SimpleNamespace(db=db, profile_id=str(home), instance_id='test', events={},
                           epoch=begin_runtime_epoch(db, instance_id='test'))


def message(text, *, user='alice', chat='chat-1', chat_type='dm', platform=Platform.TELEGRAM,
            thread=None, message_id='m-1', name='Alice', chat_name=None, **extra):
    source = SessionSource(platform=platform, chat_id=chat, chat_type=chat_type, user_id=user,
                           user_name=name, thread_id=thread, message_id=message_id, chat_name=chat_name)
    for key, value in extra.items():
        setattr(source, key, value)
    return MessageEvent(text=text, source=source, message_id=message_id)
