"""Private list/help only, adapted from M73's messaging command and read buckets.

Source: dokterdok/hermes-agent 73fbc700664c56eabd9d7a55f178320662ef0c47.
Native owner consent remains owned by session_group_messaging_read.
"""
import logging
import sqlite3
import time

from gateway.group_chat_policy import PRIVATE_ADMIN_REQUIRED, private_admin_event
from gateway import hosted_room_messaging_runtime as inventory
from gateway.session_group_messaging_read import _attest_inventory
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
_GROUP_CHAT_RATE_WINDOW_SECONDS = 60.0
_GROUP_CHAT_READ_RATE_LIMIT = 30
_GROUP_CHAT_RATE_BUCKET_CAP = 2048


def parse_group_args(args):
    words = args.split()
    if words == ['help']:
        return None
    if not words or words == ['list']:
        return 1
    if (len(words) == 2 and words[0] == 'list' and len(words[1]) <= 4
            and words[1].isascii() and words[1].isdecimal() and int(words[1]) > 0):
        return int(words[1])
    raise inventory.InvalidPageError


class GroupChatSlashCommandsMixin:
    def _group_chat_rate_limit_denial(self, context):
        # Exact, bounded recipient serialization comes from the original attestation.
        key, now = context.recipient_json, time.monotonic()
        buckets = getattr(self, '_group_chat_command_rate_buckets', None)
        if buckets is None:
            buckets = self._group_chat_command_rate_buckets = {}
        for stale in list(buckets):
            if not buckets[stale] or now - buckets[stale][-1] >= _GROUP_CHAT_RATE_WINDOW_SECONDS:
                del buckets[stale]
        recent = [stamp for stamp in buckets.get(key, ()) if now - stamp < _GROUP_CHAT_RATE_WINDOW_SECONDS]
        # Do not evict live buckets: rotating recipients must not erase rate limits.
        if len(recent) >= _GROUP_CHAT_READ_RATE_LIMIT or (key not in buckets and len(buckets) >= _GROUP_CHAT_RATE_BUCKET_CAP):
            return 'Too many Group Chat commands. Wait a moment and try again.'
        buckets[key] = [*recent, now]
        return None

    async def _handle_group_command(self, event):
        adapter = private_admin_event(self, event)
        if adapter is None:
            return PRIVATE_ADMIN_REQUIRED
        source = event.source
        # Never reselect the receiver or reply thread after an asynchronous read.
        target = (source.chat_id, source.thread_id)
        prefix = getattr(adapter, 'typed_command_prefix', '/')
        try:
            context = _attest_inventory(self, event)
            denial = self._group_chat_rate_limit_denial(context)
            if denial:
                return denial
            page = parse_group_args(event.get_command_args())
            if page is None:
                body = f'Group inventory\n\n{prefix}group list [page] — names and member counts\n{prefix}group help'
            else:
                body = await inventory.render_inventory_page(context, page, prefix)
            context.require_current()
            if (private_admin_event(self, event) is not adapter
                    or context.adapter is not adapter
                    or event.source is not source
                    or (source.chat_id, source.thread_id) != target):
                return inventory.UNAVAILABLE
        except inventory.InvalidPageError:
            return inventory.INVALID_PAGE
        except inventory.InventoryLimitError:
            return inventory.ENUMERATION_LIMIT
        except (RuntimeStoreError, inventory.InventoryFormatError, sqlite3.Error):
            return inventory.UNAVAILABLE

        metadata = {'_interim_send': True}
        if target[1] is not None:
            metadata['thread_id'] = target[1]
        # No await between the original-consent/admin fence and the single handoff.
        # Already handed-off external traffic cannot be recalled. No retry/fallback.
        try:
            await adapter.send(target[0], body, reply_to=None, metadata=metadata)
        except Exception:
            # Do not log a transport exception: it can contain the private body.
            logger.warning('Private group inventory transport handoff failed; not retried')
        return ''
