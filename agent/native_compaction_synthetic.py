"""Which pre-checkpoint user-role rows native compaction treats as Hermes scaffolding.

Real asks claim the retention budget first, so a false positive here silently demotes a human
request. Natural-language nudges (the empty-response recovery text and its peers) therefore count
only with the producer's provenance flag; bracketed producer formats are matched on their exact
prefix, since SessionDB projection drops the flags."""

import re
from typing import Any

# Background-notice formats rendered by tools/process_registry_notifications.py that the shared
# "[IMPORTANT: Background process " prefix does not cover: the batched-completion header, heartbeat
# rows, watch-disabled/overflow messages (from tools/process_registry.py) and async-delegation results.
_NOTICE_PREFIX_RE = re.compile(
    r"\[(?:IMPORTANT: (?:\d+ background processes completed\. |Watch patterns disabled for process "
    r"|Watch-pattern overflow: |Watch-pattern notifications resumed\. )"
    r"|Background process \S+ heartbeat #|ASYNC DELEGATION (?:COMPLETE|BATCH COMPLETE|TASK FAILED) — )"
)


def is_synthetic_user_row(flags: dict[str, Any], text: str) -> bool:
    """True when a user-role row with extracted *text* is runtime scaffolding.
    *flags* is the raw source row (or the item) carrying producer metadata. Textless rows
    (image-only) and unflagged prose, including prose that repeats a nudge's words, are real asks."""
    stripped = text.strip()
    if not stripped:
        return False
    from agent.context_compressor import ContextCompressor
    from agent.conversation_compression import _SYNTHETIC_USER_FLAGS, _SYNTHETIC_USER_PREFIXES
    from agent.prompt_builder import STEER_DISPLAY_KIND

    display_kind = flags.get("display_kind")
    if any(flags.get(flag) for flag in _SYNTHETIC_USER_FLAGS) or (display_kind and display_kind != STEER_DISPLAY_KIND):
        return True
    if stripped.startswith(_SYNTHETIC_USER_PREFIXES) or _NOTICE_PREFIX_RE.match(stripped):
        return True
    message = {**flags, "role": "user", "content": text}
    if ContextCompressor._is_context_summary_message(message):
        return True
    # The shared content recognizer also matches unbracketed nudge prose; trust only bracketed formats.
    return stripped.startswith("[") and ContextCompressor._is_synthetic_compression_user_turn(message)
