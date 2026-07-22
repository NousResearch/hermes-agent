"""Collect every MEDIA:/[[audio_as_voice]] tag emitted across one agent turn.

The gateway streams an agent turn as several assistant segments split by
tool calls. Post-stream media delivery historically scanned only the final
response segment, so when a turn emitted more than one audio clip (for
example, an answer reveal voiced just before the next question), only the
last clip was delivered and the earlier ones were silently dropped.

``collect_turn_media_text`` selects the current-turn slice and joins every
assistant segment so all clips are delivered. A rebased offset of zero is
trusted only when the caller also provides the pre-turn media snapshot used
by the delivery dedup guard.
"""

import os


def normalized_media_path(path):
    """Return a stable comparison key for one media path."""
    if not isinstance(path, str) or not path.strip():
        return None
    return os.path.normcase(
        os.path.abspath(os.path.expanduser(path.strip()))
    )


def normalized_media_paths(paths):
    """Return normalized path keys, or None when the snapshot is invalid."""
    if not isinstance(paths, (list, tuple, set, frozenset)):
        return None
    normalized = set()
    for path in paths:
        key = normalized_media_path(path)
        if key is None:
            return None
        normalized.add(key)
    return frozenset(normalized)


def collect_turn_media_text(
    messages,
    fallback_response="",
    *,
    history_offset=None,
    historical_media_paths=None,
):
    """Join every assistant segment of the turn (or fall back to the final).

    When ``history_offset`` is supplied, this helper owns the source boundary.
    Offset zero means compaction may have copied old tail messages into the
    returned transcript. It is safe only when a valid pre-turn media snapshot
    is present for the delivery layer to exclude. Any uncertain boundary falls
    back to the final response, which matches the behavior before multi-clip
    collection was added.
    """
    if history_offset is None:
        turn_messages = messages
    else:
        if (
            not isinstance(messages, list)
            or not isinstance(history_offset, int)
            or isinstance(history_offset, bool)
            or not 0 <= history_offset <= len(messages)
        ):
            return fallback_response or ""
        if history_offset == 0 and normalized_media_paths(
            historical_media_paths
        ) is None:
            return fallback_response or ""
        turn_messages = messages[history_offset:]

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
