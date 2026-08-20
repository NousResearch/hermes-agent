"""Group activation-mode helpers for the QQ Bot adapter.

Pure, side-effect-free functions that decide whether an inbound group message
should trigger a bot reply. Kept separate from ``adapter.py`` so the
mention/mode logic is independently testable and so a future runtime
mode-switch command can reuse :func:`resolve_require_mention` without touching
adapter internals (see solution 2.2.3).

Aligns with openclaw ``extensions/qqbot/src/engine/group/`` — ``mention.ts``
(detectWasMentioned), ``message-gating.ts`` (resolveMentionGating) and
``activation.ts`` (resolveGroupActivation / requireMention fallback).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# QQ open-platform group event names.
GROUP_AT_MESSAGE_CREATE = "GROUP_AT_MESSAGE_CREATE"
GROUP_MESSAGE_CREATE = "GROUP_MESSAGE_CREATE"

# Matches an explicit @bot mention tag in raw content, e.g. ``<@!1903885637>``.
# Aligns with openclaw MENTION_TAG_RE. The captured group is the mentioned id;
# we only treat it as "the bot" when it equals our own app_id, so an @ of some
# other member never counts as addressing the bot.
_MENTION_TAG_RE = re.compile(r"<@!?(\d+)>")


def detect_mentioned(
    event_type: str,
    payload: Dict[str, Any],
    content: str,
    app_id: str = "",
) -> bool:
    """Return True when the bot was @-mentioned in a group message.

    Aligns with openclaw ``detectWasMentioned``:

    1. ``GROUP_AT_MESSAGE_CREATE`` always means the bot was @-ed (the QQ
       platform only emits it for bot-directed messages).
    2. Full-mode ``GROUP_MESSAGE_CREATE`` carries a ``mentions`` array; an
       entry with ``is_you: true`` marks the bot (authoritative).
    3. Fallback: an explicit ``<@!{app_id}>`` tag for *our* app id in the raw
       content.

    Conservative by design (see solution D1): when none of the above match we
    treat the message as NOT addressed to the bot. We intentionally do NOT use
    a generic ``^@word`` prefix here — in full-push mode that would misfire on
    messages that @ another member.
    """
    if event_type == GROUP_AT_MESSAGE_CREATE:
        return True

    mentions = payload.get("mentions")
    if isinstance(mentions, list):
        for entry in mentions:
            # Explicit ``is True`` (not truthy): keeps the conservative contract
            # if the platform ever returns a string like "false".
            if isinstance(entry, dict) and entry.get("is_you") is True:
                return True

    if app_id and content:
        target = str(app_id)
        for match in _MENTION_TAG_RE.finditer(content):
            if match.group(1) == target:
                return True

    return False


def resolve_require_mention(
    group_openid: str,
    *,
    global_default: bool,
    per_group: Optional[Dict[str, bool]] = None,
    runtime_overrides: Optional[Dict[str, bool]] = None,
) -> bool:
    """Resolve the effective ``require_mention`` for a group.

    Priority (highest first), aligned with openclaw ``resolveGroupConfig``:

    * runtime override — in-memory, reserved for a future ``/``-command
      (solution 2.2.3); empty in this release.
    * per-group config — ``groups.{group_openid}.require_mention``.
    * global default — ``group_require_mention`` (defaults to True = mention).

    Returns True for mention mode (reply only when @-ed), False for always
    mode (every group message may trigger a reply).
    """
    if runtime_overrides and group_openid in runtime_overrides:
        return bool(runtime_overrides[group_openid])
    if per_group and group_openid in per_group:
        return bool(per_group[group_openid])
    return bool(global_default)
