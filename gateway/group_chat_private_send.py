"""Private canonical Group inventory and exact text Send consumer.

Loaded only when the native messaging authority is composed. Legacy Group Chat
controls remain on their original handler and hosted backend.
"""
import logging
import re
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)
_GROUP_CHAT_RATE_WINDOW_SECONDS = 60.0
_GROUP_CHAT_SEND_RATE_LIMIT = 10
_GROUP_CHAT_RATE_BUCKET_CAP = 2048
SEND_UNAVAILABLE = 'Group Send is unavailable or access changed.'
INVALID_SEND = 'Use /group N send <message>.'
SEND_UNCERTAIN = ("Hermes couldn't confirm whether that message was queued. "
                  "Check /group {room_ref} before sending a new message.")


@dataclass(frozen=True)
class GroupSendRef:
    value: int
    text: str


def parse_group_args(args):
    from gateway import hosted_room_messaging_runtime as inventory
    match = re.fullmatch(r'\s*([0-9]{1,19})[ \t]+send(?:[ \t\r\n]+(.*))?\s*', args, re.DOTALL)
    if match is not None:
        room_ref = int(match.group(1))
        if not 1 <= room_ref <= 2**63 - 2:
            raise inventory.InvalidPageError
        return GroupSendRef(room_ref, match.group(2) or '')
    raise inventory.InvalidPageError


def _raw_group_args(event):
    """Extract command arguments without MessageEvent's dash normalization."""
    text = event.text if isinstance(event.text, str) else ''
    match = re.match(r'^\s*\S+(?:\s+(.*))?$', text, re.DOTALL)
    return match.group(1) if match is not None and match.group(1) is not None else ''


class PrivateGroupSendMixin:
    def _group_chat_send_rate_limit_denial(self, context):
        key, now = context.recipient_json, time.monotonic()
        buckets = getattr(self, '_group_chat_send_rate_buckets', None)
        if buckets is None:
            buckets = self._group_chat_send_rate_buckets = {}
        for stale in list(buckets):
            if not buckets[stale] or now - buckets[stale][-1] >= _GROUP_CHAT_RATE_WINDOW_SECONDS:
                del buckets[stale]
        recent = [stamp for stamp in buckets.get(key, ())
                  if now - stamp < _GROUP_CHAT_RATE_WINDOW_SECONDS]
        if (len(recent) >= _GROUP_CHAT_SEND_RATE_LIMIT
                or (key not in buckets and len(buckets) >= _GROUP_CHAT_RATE_BUCKET_CAP)):
            return 'Too many Group Send commands. Wait a moment and try again.'
        buckets[key] = [*recent, now]
        return None

    @staticmethod
    def _group_send_egress_current(runner, event, source, target, adapter, context, params):
        from gateway.group_chat_policy import private_admin_event
        try:
            context.require_current(method='groups.send', params=params)
            captured_adapter = context.room_read.inventory.adapter
            return (private_admin_event(runner, event) is adapter
                    and captured_adapter is adapter
                    and event.source is source
                    and (source.chat_id, source.thread_id) == target)
        except Exception:
            return False

    @staticmethod
    async def _group_send_handoff(adapter, target, body):
        metadata = {'_interim_send': True}
        if target[1] is not None:
            metadata['thread_id'] = target[1]
        try:
            await adapter.send(target[0], body, reply_to=None, metadata=metadata)
        except Exception:
            # Adapter failures can contain private text; never log the exception.
            logger.warning('Private Group Send acknowledgement handoff failed; not retried')

    async def _handle_group_command(self, event):
        # The Send path must not import Read. Read is independently composed,
        # while its native detail and inventory controls share this entry point.
        raw_args = _raw_group_args(event)
        if re.match(r'^\s*[0-9]+[ \t]+send(?:[ \t\r\n]|$)', raw_args) is None:
            from gateway.group_chat_private_read import handle_private_group_read
            return await handle_private_group_read(self, event)

        from gateway.group_chat_policy import PRIVATE_ADMIN_REQUIRED, private_admin_event
        from gateway import hosted_room_messaging_runtime as inventory
        from gateway.session_group_messaging_send import attest_room_send
        from hermes_state_runtime import RuntimeStoreError
        adapter = private_admin_event(self, event)
        if adapter is None:
            return PRIVATE_ADMIN_REQUIRED
        source = event.source
        # Never reselect the receiver or reply thread after an asynchronous read.
        target = (source.chat_id, source.thread_id)
        prefix = getattr(adapter, 'typed_command_prefix', '/')
        try:
            selection = parse_group_args(raw_args)
            from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES
            if (not selection.text.strip()
                    or len(selection.text.encode('utf-8')) > MAX_USER_TEXT_BYTES):
                return INVALID_SEND
            context = attest_room_send(self, event, selection.value, selection.text)
            denial = self._group_chat_send_rate_limit_denial(context)
            if denial:
                return denial
            params = {
                'room_id': context.room_id,
                'event_id': context.client_event_id,
                'payload': {'text': selection.text, 'thread_id': context.client_event_id},
            }
            from gateway.session_group_controls import dispatch_group_control
            try:
                result = await dispatch_group_control(context, 'groups.send', params)
            except RuntimeStoreError:
                return SEND_UNAVAILABLE
            except Exception:
                if self._group_send_egress_current(
                        self, event, source, target, adapter, context, params):
                    await self._group_send_handoff(
                        adapter, target, SEND_UNCERTAIN.format(room_ref=selection.value))
                return ''
            if (not result.get('accepted')
                    or not self._group_send_egress_current(
                        self, event, source, target, adapter, context, params)):
                return ''
            await self._group_send_handoff(
                adapter, target,
                f'Queued in Group {selection.value}. Check: {prefix}group {selection.value}')
            return ''
        except inventory.InvalidPageError:
            return inventory.INVALID_PAGE
        except RuntimeStoreError:
            return SEND_UNAVAILABLE
