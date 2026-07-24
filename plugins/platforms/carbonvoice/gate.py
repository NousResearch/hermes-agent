"""Mention-aware message gate for the Carbon Voice adapter.

Decides whether an inbound message should reach the agent. DMs always
pass; group channels require an @-mention of the agent unless the
channel is on a free-response allowlist (or the global mention
requirement is disabled).

Configuration (all optional, read from env at adapter startup):

    CARBONVOICE_REQUIRE_MENTION
        ``true`` (default) — in group channels, only process messages
        that @-mention the agent.
        ``false`` — process every message in every channel (preserves
        the pre-gate "bot responds to everything" behavior, useful for
        personal-bot setups).

    CARBONVOICE_FREE_RESPONSE_CHANNELS
        Comma-separated channel_guids where the agent always responds,
        regardless of mention. Useful for channels dedicated to the
        agent (e.g., a "bot-chat" room).

    CARBONVOICE_IGNORED_CHANNELS
        Comma-separated channel_guids where the agent NEVER responds,
        even when mentioned. Hard veto — applied before every other
        rule. Useful for muting channels the bot was added to by
        accident or for maintenance windows.

DMs are never affected by these toggles. ``IGNORED_CHANNELS`` is the
one exception: it also vetoes DMs if the channel_guid is listed (so
operators can mute even a 1:1 channel without revoking the user's
allowlist entry).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .parse import is_user_mentioned

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    """Result of a single gate evaluation.

    ``process`` is the field the adapter acts on; ``reason`` is
    surfaced in debug logs so operators can audit why a particular
    message was accepted or skipped without rebuilding the gate's
    decision tree in their head.

    ``revisitable`` distinguishes rejections whose verdict can flip on
    a later re-fire of the same message_id (currently only "group
    channel without @-mention" — because cv-api's tag-resolution job
    runs async and emits a ``message:updated`` once ``tagged_user_ids``
    is populated, which can turn an earlier "no mention" into a
    "mentioned"). The adapter uses this to decide whether to mark the
    message as seen in the dedup cache or leave the door open for a
    follow-up update to re-evaluate. All other rejections are stable
    (ignored channel, allowlist) and don't benefit from re-evaluation.
    """

    process: bool
    reason: str
    revisitable: bool = False


class MentionGate:
    """Stateless gate that decides whether to dispatch a message to the agent.

    Stateless by design: all of its inputs come from the message
    payload, the resolved chat_type, and env-driven config snapshotted
    at construction. No per-channel memory yet — thread continuity is
    a follow-up (see roadmap in README).
    """

    def __init__(
        self,
        *,
        require_mention: bool,
        free_response_channels: Set[str],
        ignored_channels: Set[str],
    ):
        self._require_mention = require_mention
        self._free_response_channels = free_response_channels
        self._ignored_channels = ignored_channels

    @classmethod
    def from_env(cls) -> "MentionGate":
        return cls(
            require_mention=_env_bool("CARBONVOICE_REQUIRE_MENTION", default=True),
            free_response_channels=_env_set("CARBONVOICE_FREE_RESPONSE_CHANNELS"),
            ignored_channels=_env_set("CARBONVOICE_IGNORED_CHANNELS"),
        )

    def evaluate(
        self,
        *,
        msg: Dict[str, Any],
        chat_type: str,
        channel_id: str,
        self_user_id: Optional[str],
    ) -> GateDecision:
        """Return a GateDecision for this message.

        Order of checks (most specific veto first):
          1. Ignored channels — hard veto, applies to DMs too.
          2. DMs — always process (the canonical "talk to the bot" path).
          3. Free-response channels — explicit opt-out from the gate.
          4. ``require_mention`` disabled globally — explicit opt-out.
          5. Group channel + @-mention of the agent — process.
          6. Group channel without mention — skip.
        """
        if channel_id in self._ignored_channels:
            return GateDecision(False, f"channel {channel_id} on ignored list")
        if chat_type == "dm":
            return GateDecision(True, "dm always processes")
        if channel_id in self._free_response_channels:
            return GateDecision(True, "free-response channel")
        if not self._require_mention:
            return GateDecision(True, "require_mention disabled")
        if is_user_mentioned(msg, self_user_id):
            return GateDecision(True, "agent @-mentioned")
        return GateDecision(
            False, "group channel without @-mention", revisitable=True,
        )


def _env_bool(name: str, *, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_set(name: str) -> Set[str]:
    val = os.getenv(name)
    if not val:
        return set()
    return {c.strip() for c in val.split(",") if c.strip()}
