"""Collect every MEDIA:/[[audio_as_voice]] tag emitted across one agent turn.

The gateway streams an agent turn as several assistant segments split by
tool calls. Post-stream media delivery historically scanned only the final
response segment, so when a turn emitted more than one audio clip (for
example, an answer reveal voiced just before the next question), only the
last clip was delivered and the earlier ones were silently dropped.

``collect_turn_media_text`` joins the content of every assistant segment in
the turn so all clips are delivered, falling back to the final response when
no assistant segments are available.
"""


def collect_turn_media_text(turn_messages, fallback_response=""):
    """Join every assistant segment of the turn (or fall back to the final).

    Only this turn's messages are passed in (``agent_messages`` sliced from
    ``history_offset``), never prior-turn history, so media from earlier
    turns can never be replayed.
    """
    parts = []
    for message in turn_messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    if not parts:
        return fallback_response or ""
    return "\n".join(parts)
