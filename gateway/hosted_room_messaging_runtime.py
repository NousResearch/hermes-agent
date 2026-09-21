"""Bounded canonical inventory consumer; no classic room store or execution path.

List presentation selectively adapts M73 (dokterdok/hermes-agent,
73fbc700664c56eabd9d7a55f178320662ef0c47), without selectors or status claims.
Offset enumeration retains canonical order, not a point-in-time store snapshot.
"""
import unicodedata

from gateway.session_group_controls import dispatch_group_control
from gateway.session_group_messaging_read import MAX_INVENTORY_OFFSET, MAX_PAGE_SIZE

DISPLAY_PAGE_SIZE = 8
UNAVAILABLE = 'Group inventory is unavailable. Private admin access and native-owner consent are required.'
ENUMERATION_LIMIT = 'Group inventory exceeds the bounded enumeration limit. No partial list is shown.'
INVALID_PAGE = 'Invalid group list page. Use a positive page number within the available list.'


class InventoryLimitError(Exception):
    """The canonical cursor still has work beyond this consumer's bound."""


class InventoryFormatError(Exception):
    """An invalid projection or cursor cannot be treated as a complete inventory."""


class InvalidPageError(Exception):
    """The requested display page does not exist."""


# Use inert visual punctuation instead of backend-dependent Markdown escapes.
_NAME_ESCAPES = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in r'@`*_[]<>\\:#&|~()!/+='})


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
        lines.append(f"• {safe_name(row['name'])} — {count} {'member' if count == 1 else 'members'}")
    navigation = []
    if page > 1:
        navigation.append(f'Previous: {prefix}group list {page - 1}')
    if page < pages:
        navigation.append(f'Next: {prefix}group list {page + 1}')
    if navigation:
        lines.extend(['', '\n'.join(navigation)])
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
