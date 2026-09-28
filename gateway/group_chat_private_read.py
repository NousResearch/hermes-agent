"""Private canonical Group inventory and exact-room detail presentation.

Authority, consent and dispatch remain with the public Messaging owner.
"""
import logging
import re
import sqlite3
import time
import unicodedata

from gateway.group_chat_policy import PRIVATE_ADMIN_REQUIRED, private_admin_event
from gateway.session_group_controls import dispatch_group_control
from gateway.session_group_messaging_read import (
    MAX_INVENTORY_OFFSET, MAX_PAGE_SIZE, _attest_inventory, _attest_room_read,
)
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
DISPLAY_PAGE_SIZE = 8
UNAVAILABLE = 'Group inventory is unavailable. Private admin access and native-owner consent are required.'
ENUMERATION_LIMIT = 'Group inventory exceeds the bounded enumeration limit. No partial list is shown.'
INVALID_PAGE = 'Invalid group list page. Use a positive page number within the available list.'
NOT_AUTHORIZED = "Details for that Group Chat aren't authorized in this private chat."
STALE_REFERENCE = 'That Group Chat number is no longer available. Run /group list again.'
ACCESS_CHANGED = 'Private Group Chat access changed. Run /group list again.'
DETAIL_UNAVAILABLE = 'Group status is temporarily unavailable. Make sure its gateway is online, then try again.'
DETAIL_EVENT_WINDOW = 32
MAX_DETAIL_PREVIEWS = 5
MAX_DETAIL_ROSTER = 12
_READ_RATE_WINDOW_SECONDS = 60.0
_READ_RATE_LIMIT = 30
_RATE_BUCKET_CAP = 2048

def _read_rate_limit_denial(runner, context):
    key, now = context.recipient_json, time.monotonic()
    buckets = getattr(runner, '_group_chat_command_rate_buckets', None)
    if buckets is None:
        buckets = runner._group_chat_command_rate_buckets = {}
    for stale in list(buckets):
        if not buckets[stale] or now - buckets[stale][-1] >= _READ_RATE_WINDOW_SECONDS:
            del buckets[stale]
    recent = [stamp for stamp in buckets.get(key, ()) if now - stamp < _READ_RATE_WINDOW_SECONDS]
    if len(recent) >= _READ_RATE_LIMIT or (key not in buckets and len(buckets) >= _RATE_BUCKET_CAP):
        return 'Too many Group Chat commands. Wait a moment and try again.'
    buckets[key] = [*recent, now]
    return None

class InventoryLimitError(Exception):
    """The canonical cursor still has work beyond this consumer's bound."""

class InventoryFormatError(Exception):
    """An invalid projection or cursor cannot be treated as a complete inventory."""

class InvalidPageError(Exception):
    """The requested display page does not exist."""

class DetailFormatError(Exception):
    """Canonical state/log could not be projected without exposing internals."""

_NAME_ESCAPES = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in r'@`*_[]<>\:#&|~()!/+='})

def safe_name(name):
    name = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in name)
    return (' '.join(name.split())[:72] or 'Unnamed group').translate(_NAME_ESCAPES)

def render_inventory(rows, page, prefix):
    if type(page) is not int or page < 1:
        raise InvalidPageError
    pages = max(1, (len(rows) + DISPLAY_PAGE_SIZE - 1) // DISPLAY_PAGE_SIZE)
    if page > pages:
        raise InvalidPageError
    if not rows:
        return 'No groups are available in your authorized inventory.'
    start = (page - 1) * DISPLAY_PAGE_SIZE
    lines = [f'Groups — page {page} of {pages}', '']
    for row in rows[start:start + DISPLAY_PAGE_SIZE]:
        count = row['member_count']
        selector = f"{row['room_ref']}. " if 'room_ref' in row else ''
        lines.append(
            f"• {selector}{safe_name(row['name'])} — "
            f"{count} {'member' if count == 1 else 'members'}"
            + (f" · Open: {prefix}group {row['room_ref']}" if 'room_ref' in row else ''))
    navigation = []
    if page > 1:
        navigation.append(f'Previous: {prefix}group list {page - 1}')
    if page < pages:
        navigation.append(f'Next: {prefix}group list {page + 1}')
    if navigation:
        lines.extend(['', '\n'.join(navigation)])
    if any('room_ref' in row for row in rows[start:start + DISPLAY_PAGE_SIZE]):
        lines.extend(['', f'Open a numbered group: {prefix}group N'])
    return '\n'.join(lines)


async def render_inventory_page(context, page, prefix):
    """Enumerate under ONE attestation, including empty owner-filtered raw pages."""
    rows, offset = [], 0
    while True:
        context.require_current()
        limit = min(MAX_PAGE_SIZE, MAX_INVENTORY_OFFSET - offset)
        result = await dispatch_group_control(context, 'groups.list', {'limit': limit, 'offset': offset})
        context.require_current()
        if type(result) is not dict or set(result) != {'rooms', 'next_offset'}:
            raise InventoryFormatError
        batch = result['rooms']
        if type(batch) is not list or len(batch) > limit:
            raise InventoryFormatError
        for row in batch:
            if (type(row) is not dict or set(row) not in (
                    {'name', 'member_count'}, {'name', 'member_count', 'room_ref'})
                    or type(row['name']) is not str or type(row['member_count']) is not int
                    or row['member_count'] < 0
                    or ('room_ref' in row and (type(row['room_ref']) is not int
                                              or not 1 <= row['room_ref'] <= 2**63 - 2))):
                raise InventoryFormatError
        rows.extend(batch)
        cursor = result['next_offset']
        if cursor is None:
            return render_inventory(rows, page, prefix)
        if type(cursor) is not int or cursor <= offset:
            raise InventoryFormatError
        if cursor >= MAX_INVENTORY_OFFSET:
            raise InventoryLimitError
        offset = cursor


def _safe_text(value, maximum=180):
    if type(value) is not str:
        raise DetailFormatError
    value = re.sub(r'(?i)\bMEDIA:[^\s]*', '[media omitted]', value)
    text = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in value)
    return (' '.join(text.split())[:maximum] or 'No text').translate(_NAME_ESCAPES)


def _room_authority(state, room_id):
    if type(state) is not dict or set(state) != {'room', 'driver_status'}:
        raise DetailFormatError
    room = state['room']
    status = state['driver_status']
    if (type(room) is not dict or room.get('room_id') != room_id
            or type(room.get('authority_gateway_id')) is not str
            or type(room.get('authority_epoch')) is not int
            or type(room.get('name')) is not str
            or type(room.get('members')) is not list
            or len(room['members']) > 128
            or type(status) is not dict):
        raise DetailFormatError
    required = {'running', 'working', 'blocked', 'counts', 'pending_actions', 'peer_routes'}
    if (not required <= set(status)
            or any(type(status[field]) is not bool for field in ('running', 'working', 'blocked'))
            or status['running'] is not True
            or type(status['counts']) is not dict
            or any(type(key) is not str or type(value) is not int or value < 0
                   for key, value in status['counts'].items())
            or type(status['pending_actions']) is not list
            or type(status['peer_routes']) is not list
            or len(status['peer_routes']) > 128):
        raise DetailFormatError
    return room, status, {
        'gateway_id': room['authority_gateway_id'],
        'epoch': room['authority_epoch'],
    }


def _log_authority(log, room_id, *, maximum):
    if (type(log) is not dict
            or set(log) != {'events', 'cursor', 'latest_seq', 'has_more', 'authority'}
            or type(log['events']) is not list or len(log['events']) > maximum
            or type(log['cursor']) is not int or log['cursor'] < 0
            or type(log['latest_seq']) is not int or log['latest_seq'] < 0
            or log['cursor'] > log['latest_seq']
            or log['has_more'] is not (log['cursor'] < log['latest_seq'])
            or type(log['has_more']) is not bool
            or type(log['authority']) is not dict
            or set(log['authority']) != {'gateway_id', 'epoch'}
            or type(log['authority']['gateway_id']) is not str
            or type(log['authority']['epoch']) is not int):
        raise DetailFormatError
    previous = 0
    for event in log['events']:
        if (type(event) is not dict or event.get('room_id') != room_id
                or type(event.get('seq')) is not int or event['seq'] <= previous):
            raise DetailFormatError
        previous = event['seq']
    return log['authority']


def _member_labels(members):
    labels = {}
    visible = []
    for member in members:
        if (type(member) is not dict or type(member.get('member_id')) is not str
                or member['member_id'] in labels):
            raise DetailFormatError
        display = member.get('display_name') or member.get('handle')
        if type(display) is not str:
            raise DetailFormatError
        label = _safe_text(display, 72)
        handle = member.get('handle')
        if type(handle) is str and handle and handle != display:
            label += f" ({_safe_text(handle, 72)})"
        labels[member['member_id']] = label
        visible.append(label)
    return labels, visible


def _imported_preview(event):
    payload = event.get('payload')
    if type(payload) is not dict:
        return None
    required = {'source', 'author', 'text', 'thread_id', 'at_ms'}
    if set(payload) not in (required, required | {'attachments'}):
        return None
    source, author = payload['source'], payload['author']
    if (type(source) is not dict
            or set(source) != {'kind', 'source_id', 'source_entry_id'}
            or source.get('kind') != 'desktop_group_chat'
            or type(source.get('source_id')) is not str
            or type(source.get('source_entry_id')) is not str
            or type(author) is not dict or author.get('kind') not in {'user', 'member'}):
        return None
    allowed_authors = ({frozenset({'kind', 'name', 'member_id'})}
                       if author['kind'] == 'member'
                       else {frozenset({'kind', 'name'}), frozenset({'kind', 'name', 'member_id'})})
    if (frozenset(author) not in allowed_authors or type(author.get('name')) is not str
            or ('member_id' in author and type(author['member_id']) is not str)):
        return None
    if (type(payload['thread_id']) is not str or type(payload['at_ms']) is not int
            or ('attachments' in payload and type(payload['attachments']) is not list)):
        return None
    return _safe_text(author['name'], 72), _safe_text(payload['text'])


def _message_preview(event, member_labels):
    kind, payload = event.get('kind'), event.get('payload')
    if kind == 'history.imported':
        return _imported_preview(event)
    if kind not in {'message.user', 'message.member'} or type(payload) is not dict:
        return None
    text = payload.get('text')
    if type(text) is not str:
        return None
    if kind == 'message.user':
        return 'You', _safe_text(text)
    member_id = payload.get('member_id')
    if type(member_id) is not str:
        return None
    return member_labels.get(member_id, 'Bot'), _safe_text(text)


def render_group_detail(state, log, *, room_id, room_ref, prefix):
    if (type(room_id) is not str or type(room_ref) is not int
            or not 1 <= room_ref <= 2**63 - 2 or type(prefix) is not str):
        raise DetailFormatError
    room, status, authority = _room_authority(state, room_id)
    if _log_authority(log, room_id, maximum=DETAIL_EVENT_WINDOW) != authority:
        raise DetailFormatError
    member_labels, roster = _member_labels(room['members'])
    condition = 'blocked' if status['blocked'] else 'working' if status['working'] else 'idle'
    lines = [f"Group {room_ref} — {safe_name(room['name'])}", f'Status: {condition}']
    active = sum(status['counts'].get(key, 0) for key in ('queued', 'running', 'stopping'))
    waiting = sum(status['counts'].get(key, 0) for key in ('indeterminate', 'deferred'))
    if active or waiting:
        lines.append(f'Work: {active} active, {waiting} waiting')
    lines.extend(['', 'Bots'])
    for label in roster[:MAX_DETAIL_ROSTER]:
        lines.append(f'• {label}')
    if len(roster) > MAX_DETAIL_ROSTER:
        lines.append(f'+ {len(roster) - MAX_DETAIL_ROSTER} more Bots')
    reconnect = []
    for route in status['peer_routes']:
        if type(route) is not dict:
            raise DetailFormatError
        member_id, route_status = route.get('member_id'), route.get('status')
        if type(member_id) is not str or type(route_status) is not str:
            raise DetailFormatError
        if member_id not in member_labels:
            raise DetailFormatError
        label = member_labels[member_id]
        if route_status == 'needs_reauthorization':
            reconnect.append(f'{label} needs to reconnect.')
        elif route_status == 'unavailable':
            reconnect.append(f'{label} is unavailable.')
    if reconnect:
        lines.extend(['', *reconnect[:MAX_DETAIL_ROSTER]])
    approvals = [action for action in status['pending_actions']
                 if type(action) is dict and action.get('kind') == 'approval']
    for index, action in enumerate(approvals[:MAX_DETAIL_ROSTER], 1):
        member = action.get('member_id')
        if (type(member) is str and member in member_labels
                and type(action.get('request_id')) is str and action['request_id']
                and type(action.get('task_id')) is str and action['task_id']
                and type(action.get('execution_generation')) is int
                and action['execution_generation'] > 0
                and type(action.get('selector')) is str
                and re.fullmatch(r'pa-[0-9a-f]{64}', action['selector'])):
            lines.append(f'Approval {index}: {member_labels[member]} — '
                         f'{prefix}group {room_ref} approve {action["selector"]} once|deny')
    previews = [preview for event in log['events']
                if (preview := _message_preview(event, member_labels)) is not None]
    lines.extend(['', 'Recent messages'])
    for author, text in previews[-MAX_DETAIL_PREVIEWS:]:
        lines.append(f'• {author}: {text}')
    if not previews:
        lines.append('No recent messages in this bounded view.')
    lines.extend(['', f'Stop work: {prefix}group {room_ref} stop',
                  f'Refresh: {prefix}group {room_ref}'])
    return '\n'.join(lines)


async def read_group_detail(context, prefix):
    """Read one bounded recent tail; state and log are not claimed as one snapshot."""
    from gateway.session_group_messaging_read import _MessagingRoomRead
    if type(context) is not _MessagingRoomRead:
        raise DetailFormatError
    room_id = context.room_id
    context.require_current(method='groups.state', room_id=room_id)
    state = await dispatch_group_control(
        context, 'groups.state', {'room_id': room_id})
    context.require_current(method='groups.state', room_id=room_id)
    room, _, state_authority = _room_authority(state, room_id)
    probe = await dispatch_group_control(
        context, 'groups.log', {'room_id': room_id, 'since_seq': 0, 'limit': 1})
    context.require_current(method='groups.log', room_id=room_id)
    if _log_authority(probe, room_id, maximum=1) != state_authority:
        raise DetailFormatError
    latest = probe['latest_seq']
    if latest:
        log = await dispatch_group_control(context, 'groups.log', {
            'room_id': room_id,
            'since_seq': max(0, latest - DETAIL_EVENT_WINDOW),
            'limit': DETAIL_EVENT_WINDOW,
        })
        context.require_current(method='groups.log', room_id=room_id)
    else:
        log = probe
    if (_log_authority(log, room_id, maximum=DETAIL_EVENT_WINDOW) != state_authority
            or log['latest_seq'] < latest):
        raise DetailFormatError
    if room['room_id'] != context.room_id:
        raise DetailFormatError
    # All external awaits are finished: select operation-specific disclosure now.
    try:
        from gateway.session_group_messaging_control import pending_room_approvals
        _, approvals = pending_room_approvals(context.inventory.runner,
                                               context.inventory.event, context.room_ref)
    except RuntimeStoreError:
        approvals = []
    state = {**state, 'driver_status': {**state['driver_status'],
                                        'pending_actions': approvals}}
    return render_group_detail(
        state, log, room_id=room_id, room_ref=context.room_ref, prefix=prefix)


def _raw_group_args(event):
    # MessageEvent.get_command_args normalizes dashes, obscuring invalid refs.
    text = event.text if isinstance(event.text, str) else ''
    match = re.match(r'^\s*\S+(?:\s+(.*))?$', text, re.DOTALL)
    return match.group(1) if match is not None and match.group(1) is not None else ''

def _parse_read_args(args):
    words = args.split()
    if words == ['help']:
        return None
    if not words or words == ['list']:
        return 1
    if (len(words) == 2 and words[0] == 'list' and len(words[1]) <= 4
            and words[1].isascii() and words[1].isdecimal() and int(words[1]) > 0):
        return int(words[1])
    if (len(words) == 1 and len(words[0]) <= 19 and words[0].isascii()
            and words[0].isdecimal() and 1 <= int(words[0]) <= 2**63 - 2):
        return ('detail', int(words[0]))
    raise InvalidPageError

async def handle_private_group_read(runner, event):
    """Handle list/help/detail on the captured private receiver; return guarded handoff status."""
    adapter = private_admin_event(runner, event)
    if adapter is None:
        return PRIVATE_ADMIN_REQUIRED
    source = event.source
    target = (source.chat_id, source.thread_id)
    prefix = getattr(adapter, 'typed_command_prefix', '/')
    context = selection = None
    try:
        selection = _parse_read_args(_raw_group_args(event))
        detail = isinstance(selection, tuple)
        context = (_attest_room_read(runner, event, selection[1]) if detail
                   else _attest_inventory(runner, event))
        denial = _read_rate_limit_denial(runner, context)
        if denial:
            return denial
        if selection is None:
            body = (f'Group Chat\n\n{prefix}group list [page] — names and member counts\n'
                    f'{prefix}group N — status, Bots, and recent messages\n'
                    f'{prefix}group help')
        elif detail:
            body = await read_group_detail(context, prefix)
        else:
            body = await render_inventory_page(context, selection, prefix)
        if detail:
            context.require_current(method='groups.log', room_id=context.room_id)
            captured_adapter = context.inventory.adapter
        else:
            context.require_current()
            captured_adapter = context.adapter
        if (private_admin_event(runner, event) is not adapter
                or captured_adapter is not adapter or event.source is not source
                or (source.chat_id, source.thread_id) != target):
            return ACCESS_CHANGED if detail else UNAVAILABLE
    except InvalidPageError:
        return INVALID_PAGE
    except InventoryLimitError:
        return ENUMERATION_LIMIT
    except RuntimeStoreError as exc:
        if not isinstance(selection, tuple):
            return UNAVAILABLE
        if exc.reason == 'messaging_room_read_stale':
            return STALE_REFERENCE
        return ACCESS_CHANGED if context is not None else NOT_AUTHORIZED
    except (InventoryFormatError, DetailFormatError, sqlite3.Error):
        return DETAIL_UNAVAILABLE if isinstance(selection, tuple) else UNAVAILABLE

    metadata = {'_interim_send': True}
    if target[1] is not None:
        metadata['thread_id'] = target[1]
    # No await between the consent, receiver and original-target checks and handoff.
    try:
        await adapter.send(target[0], body, reply_to=None, metadata=metadata)
    except Exception:
        # A transport exception may contain private content. Never log it.
        logger.warning('Private Group read handoff failed; not retried')
    return ''
