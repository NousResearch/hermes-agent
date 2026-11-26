"""Reply obligation for Discord messages already admitted by channel policy."""
from typing import Any, Optional


def discord_reply_expected(
    *, bot_user_id: Optional[int], is_dm: bool, addressed: bool,
    free_response: bool, reference: Any,
) -> Optional[bool]:
    """Allow opt-out only for known-unaddressed free-response chatter.

    Discord replies may omit a ping; retain the gateway fallback when the
    referenced author is unknown rather than guessing that it was a peer.
    """
    if is_dm or addressed:
        return True
    if reference is not None:
        target = getattr(reference, "resolved", None) or getattr(reference, "cached_message", None)
        author_id = getattr(getattr(target, "author", None), "id", None)
        if author_id is None or bot_user_id is None:
            return None
        if str(author_id) == str(bot_user_id):
            return True
    return False if free_response else None
