"""Discord's ``thread_url``: the jump link the session lists open. Kept apart from adapter.py,
which is over its size cap."""

_SNOWFLAKE_CHARS = frozenset("0123456789")


def _snowflake(value) -> str:
    text = str(value or "")
    return text if text and set(text) <= _SNOWFLAKE_CHARS else ""


def discord_thread_url(source) -> str | None:
    """``https://discord.com/channels/<guild|@me>/<channel>[/<message>]``. A thread's channel id is
    the thread itself; a shared channel also names the triggering message so the reader lands on
    it instead of the channel's latest scroll position."""
    channel = _snowflake(source.chat_id)
    if not channel:
        return None
    guild = _snowflake(source.scope_id) or "@me"
    message = _snowflake(source.message_id) if source.chat_type == "group" else ""
    return f"https://discord.com/channels/{guild}/{channel}" + (f"/{message}" if message else "")
