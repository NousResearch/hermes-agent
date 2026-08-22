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


def select_turn_messages(agent_messages, history_offset, history):
    """Return this turn's messages, or ``[]`` when the boundary is ambiguous.

    ``history_offset`` is normally a real slice point into ``agent_messages``.
    It is not trustworthy on its own: split and in-place compaction
    deliberately re-baseline it to 0 while ``agent_messages`` holds the whole
    compacted transcript, and the compressor retains copied tail messages. A 0
    offset next to a non-empty incoming ``history`` is therefore
    indistinguishable from a genuine first turn, and slicing from it would
    rescan retained prior-turn segments and replay their attachments.

    Refusing the slice in that case is deliberately conservative: the caller
    falls back to the final response alone, which is the behaviour that
    shipped before multi-segment collection existed.
    """
    if not isinstance(agent_messages, list):
        return []
    if not isinstance(history_offset, int) or isinstance(history_offset, bool):
        return []
    if not 0 <= history_offset <= len(agent_messages):
        return []
    if history_offset == 0 and history:
        return []
    return agent_messages[history_offset:]


def collect_turn_media_text(turn_messages, fallback_response=""):
    """Join every assistant segment of the turn (or fall back to the final).

    Callers pass this turn's messages (``agent_messages`` sliced from
    ``history_offset``). That slice is only turn-local when the offset is
    trustworthy: after split or in-place compaction the gateway deliberately
    reports ``history_offset=0`` while ``messages`` holds the compacted
    transcript, so the slice can still contain retained prior-turn segments.
    Replay is prevented by the caller, which passes an empty ``turn_messages``
    whenever the boundary is ambiguous (offset 0 with a non-empty incoming
    history); this function then falls back to the final response alone.
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
