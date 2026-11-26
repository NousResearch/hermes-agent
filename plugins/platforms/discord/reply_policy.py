"""Reply obligation for Discord messages already admitted by channel policy."""
from typing import Any, Optional


def discord_reply_expected(
    *, bot_user_id: Optional[int], is_dm: bool, addressed: bool,
    free_response: bool, message: Any,
) -> Optional[bool]:
    """Allow opt-out only for known-unaddressed free-response chatter.

    Discord roles and no-ping replies address the bot without a user mention.
    Unknown role membership or reply authors retain the gateway fallback.
    """
    if is_dm or addressed:
        return True
    role_ids = set(getattr(message, "raw_role_mentions", ()) or ())
    role_ids.update(role.id for role in (getattr(message, "role_mentions", ()) or ()))
    if role_ids:
        member = getattr(getattr(message, "guild", None), "me", None)
        roles = getattr(member, "roles", None)
        if bot_user_id is None or getattr(member, "id", None) != bot_user_id or roles is None:
            return None
        if role_ids.intersection(role.id for role in roles):
            return True
    reference = getattr(message, "reference", None)
    if reference is not None:
        target = getattr(reference, "resolved", None) or getattr(reference, "cached_message", None)
        author_id = getattr(getattr(target, "author", None), "id", None)
        if author_id is None or bot_user_id is None:
            return None
        if str(author_id) == str(bot_user_id):
            return True
    return False if free_response else None
