"""Whether a linked conversation's content may be attached where the link was pasted."""

from __future__ import annotations

from typing import Any, Optional

from .links import PermalinkTarget, payload

_PAGE = 200
#: Membership lists above this are not verified; the link is refused instead.
_MEMBERS_CAP = 2000


async def _members(client: Any, channel_id: str) -> Optional[set[str]]:
    """Every member id of ``channel_id``; None when the list exceeds ``_MEMBERS_CAP``."""
    members: set[str] = set()
    cursor = ""
    while True:
        page = payload(await client.conversations_members(
            channel=channel_id, limit=_PAGE, cursor=cursor or None))
        members.update(str(m) for m in page.get("members") or [])
        if len(members) > _MEMBERS_CAP:
            return None
        cursor = str((page.get("response_metadata") or {}).get("next_cursor") or "")
        if not cursor:
            return members


async def access_refusal(
    client: Any, target: PermalinkTarget, *, destination_id: str, poster_id: str
) -> str:
    """Why the link's content may not be attached here (\"\" when it may).

    Same conversation: everyone here already sees it. Otherwise the bot and the poster must
    both be members of the source; a private source additionally needs every member of the
    destination to be a member of it, or posting it here would widen its audience. Direct
    messages are only read inside themselves.
    """
    if target.channel_id == destination_id:
        return ""
    info = payload(await client.conversations_info(channel=target.channel_id))
    channel = info.get("channel") or {}
    if not info.get("ok"):
        return "the bot cannot see that conversation"
    if channel.get("is_im") or channel.get("is_mpim"):
        return "links into direct messages are only read inside that conversation"
    if not channel.get("is_member"):
        return "the bot is not a member of that conversation"
    source_members = await _members(client, target.channel_id)
    if source_members is None:
        return "that conversation has too many members to verify access"
    if poster_id not in source_members:
        return "the poster is not a member of that conversation"
    if channel.get("is_private"):
        audience = await _members(client, destination_id)
        if audience is None or audience - source_members:
            return "the source is private and not everyone in this conversation can see it"
    return ""
