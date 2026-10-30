"""Branch-point resolution for ``/branch --at <message link>`` (#32105, gateway surface).

Pure helpers: the permalink names a cutoff message inside the invoking thread (the
same-thread rule is enforced by the handler), and these functions cut the transcript
at the linked message's turn. No runner state — the handler in
``slash_commands_session.py`` owns session creation and delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent.context_compressor import is_user_originated_turn

# Bot replies carry no platform_message_id (only user rows do), so a linked assistant bubble is
# matched by row timestamp. Both sides skew ~1s (a user row is written just AFTER the Slack event
# arrives; an assistant row just BEFORE its bubble posts), so the window is symmetric and
# generous, but stays far narrower than the gap between adjacent human turns.
CUTOFF_MATCH_TOLERANCE_S = 5.0


@dataclass(frozen=True)
class LinkBranchPoint:
    """Where the copied transcript ends: an inclusive index, and how the row was matched."""

    index: int
    matched: str  # "platform_message_id" | "timestamp"


class LinkCutoffUnresolved(Exception):
    """The linked message is not in the session's active history.

    ``reason`` is ``"before_history"`` when every active row postdates the message (the usual
    shape of a point behind a compaction boundary), else ``"not_found"``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _row_timestamp(message: dict) -> Optional[float]:
    value = message.get("timestamp")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _turn_end_index(history: list, index: int) -> int:
    """Inclusive last index of the user-originated turn that row *index* belongs to.

    The cut must never split a turn: a transcript ending on a tool result or on a
    tool-call-bearing assistant row cannot replay. When the containing turn's tail looks in
    flight (only the FINAL turn of a transcript can be), the cut backs up to before that turn
    started, so the copy keeps whole turns only; ``-1`` means "nothing stable to copy yet"."""

    end = index
    while end + 1 < len(history) and not is_user_originated_turn(history[end + 1]):
        end += 1
    last = history[end]
    if last.get("role") == "tool" or (last.get("role") == "assistant" and last.get("tool_calls")):
        start = index
        while start > 0 and not is_user_originated_turn(history[start]):
            start -= 1
        if is_user_originated_turn(history[start]):
            return start - 1
        return -1
    return end


def find_link_cutoff(history: list, link: Any) -> LinkBranchPoint:
    """Resolve the linked message to an inclusive cut index over *history*.

    An exact ``platform_message_id`` match wins (user rows carry the Slack ts verbatim); bot
    rows are matched by nearest timestamp inside the tolerance window. Raises
    :class:`LinkCutoffUnresolved` when the message is not in the active history."""

    target_ts = float(link.message_ts)
    for index, message in enumerate(history):
        if str(message.get("platform_message_id") or "") == str(link.message_ts):
            return LinkBranchPoint(index=_turn_end_index(history, index), matched="platform_message_id")
    nearest_index: Optional[int] = None
    nearest_distance: Optional[float] = None
    for index, message in enumerate(history):
        row_ts = _row_timestamp(message)
        if row_ts is None:
            continue
        distance = abs(row_ts - target_ts)
        if distance <= CUTOFF_MATCH_TOLERANCE_S and (nearest_distance is None or distance < nearest_distance):
            nearest_index, nearest_distance = index, distance
    if nearest_index is None:
        earliest = min((ts for ts in (_row_timestamp(m) for m in history) if ts is not None), default=None)
        if earliest is not None and target_ts < earliest - CUTOFF_MATCH_TOLERANCE_S:
            raise LinkCutoffUnresolved("before_history")
        raise LinkCutoffUnresolved("not_found")
    return LinkBranchPoint(index=_turn_end_index(history, nearest_index), matched="timestamp")