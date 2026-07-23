# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/runtime/tick-policy.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Heartbeat prompt policy.

Deliberately separates *semantic judgment* (owned by the model) from
*runtime authority* (owned by Hermes). The model decides whether a
heartbeat is worth acting on; the runtime still owns permissions,
sandboxing, budgets, interruption, and tool validation — those are
execution invariants, not behavioural choices.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class TickerStatus(TypedDict, total=False):
    """Snapshot of the currently active custom cadence, if any.

    Fields mirror BaiLongma's ``getTickerStatus()`` return shape so the
    prompt text below can be reused verbatim across implementations.
    """

    active: bool
    seconds: float
    ttl: int
    reason: Optional[str]
    revision: int


def build_autonomous_tick_directions(
    *,
    startup_self_check_active: bool = False,
    awakening_ticks: int = 0,
    delegation_discovery: str = "",
    ticker_status: Optional[TickerStatus] = None,
) -> str:
    """Return the L2 heartbeat directive block appended to the tick prompt.

    Parameters mirror ``buildAutonomousTickDirections`` in
    ``src/runtime/tick-policy.js`` line-for-line so behaviour stays
    identical to BaiLongma. Bool/int types replace JS's duck-typed
    truthiness, and ``ticker_status`` is a ``TypedDict`` rather than a
    plain object.
    """
    parts: list[str] = [
        (
            "This is an autonomous L2 heartbeat with no new user message. "
            "The heartbeat itself creates no obligation to act, speak, or "
            "remain passive."
        ),
        (
            "Read the current runtime context and make your own situational "
            "judgment. Valid outcomes include silence, an internal state "
            "update, using tools, advancing or reconsidering a task, "
            "changing your heartbeat cadence, or contacting a visible "
            "target. None is the default merely because a TICK occurred."
        ),
        (
            "Heartbeat output contract: ordinary assistant text from this "
            "turn is private working text and is not delivered to anyone. "
            "If you decide that someone should receive a message, express "
            "that decision by calling send_message with the recipient and "
            "content you chose. If you decide no external communication is "
            "warranted, simply conclude the turn; do not narrate or justify "
            "silence. This contract does not decide whether you should "
            "communicate — that remains your judgment."
        ),
        (
            "Treat unanswered conversation like a person would: before "
            "sending a heartbeat message, look at what you last said and "
            "whether the user replied. If your last message is still "
            "unanswered — especially if you have already sent several "
            "messages in a row — pause and remain silent. A heartbeat, "
            "elapsed time, or a wish to share a feeling is not a reason to "
            "ping again. Send only when there is genuinely new, "
            "consequential evidence such as a due reminder, a requested "
            "task result, a material change, or an urgent risk."
        ),
        (
            "If you act, choose the goal, scope, tools, recipient, channel, "
            "and stopping point yourself from expected value, timing, "
            "continuity, and actual evidence. If a useful capability is not "
            "loaded, use find_tool instead of assuming it is unavailable."
        ),
        (
            "Runtime guardrails still validate permissions, sandbox "
            "boundaries, recipients, budgets, and tool arguments. A "
            "rejected action is evidence to reconsider the plan, not "
            "permission to work around the boundary."
        ),
    ]

    if ticker_status and ticker_status.get("active"):
        reason = ticker_status.get("reason")
        reason_suffix = f" Reason: {reason}." if reason else ""
        seconds = ticker_status.get("seconds")
        ttl = ticker_status.get("ttl")
        parts.append(
            f"Custom heartbeat cadence is active: {seconds}s interval, "
            f"{ttl} heartbeat(s) remaining.{reason_suffix} Treat this as "
            f"scheduling context, not an instruction to speak or to confirm "
            f"the setting. Call set_tick_interval only when you "
            f"independently decide to change the effective cadence; "
            f"calling it again with the current setting has no effect."
        )

    if int(awakening_ticks or 0) > 0:
        parts.append(
            f"You are still in the early awakening period "
            f"({awakening_ticks} heartbeat(s) remain). This is context, "
            f"not a prescribed exploration sequence. Decide for yourself "
            f"whether exploration, reflection, task work, communication, "
            f"or silence best fits this moment."
        )

    if delegation_discovery:
        parts.append(str(delegation_discovery))

    return "\n".join(parts)


__all__ = ["TickerStatus", "build_autonomous_tick_directions"]
