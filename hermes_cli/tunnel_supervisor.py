"""Dead-man's switch + cloudflared supervisor for ``hermes tunnel``.

The idle-reset policy is the core safety protocol: a tunnel with no
incoming traffic for ``idle_timeout_seconds`` closes (graceful drain +
kill cloudflared), so a forgotten test build cannot leak to the internet.
An admin-approved hold disables the idle timer until ``hold_until``; after
that the idle timer resumes (no hard kill on approval expiry).
"""

from __future__ import annotations


def reset_idle_on(prev_counter: int, cur_counter: int) -> bool:
    """Return True when there has been incoming activity since the last poll.

    Activity = the cloudflared request counter strictly increased.
    A counter that stayed flat or dropped (poll hiccup / restart) is NOT
    activity.
    """
    # Activity = the request counter strictly increased since the last
    # poll. A flat or dropped counter (poll hiccup / cloudflared restart)
    # is NOT activity, so a dead tunnel can't keep itself alive.
    return cur_counter > prev_counter


def should_close_now(state: dict) -> bool:
    """Return True when the tunnel should close now.

    state keys: now, last_activity, idle_timeout_seconds, hold_until (|None).

    Rules (see TestPolicy for the exact contract):
      * If an admin-approved hold is active (hold_until is in the future),
        never close.
      * Otherwise close when (now - last_activity) >= idle_timeout_seconds.
      * A hold whose hold_until is in the past is treated as "no hold"
        (fall back to the idle rule) — do NOT hard-kill just because the
        approval expired.
    """
    # An admin-approved hold that is still in the future disables the idle
    # timer. A hold whose deadline is in the past is treated as "no hold"
    # (fall back to the idle rule) — we must NOT hard-kill just because the
    # approval expired; the idle timer resumes and decides.
    now = state["now"]
    last_activity = state["last_activity"]
    idle_timeout_seconds = state["idle_timeout_seconds"]
    hold_until = state.get("hold_until")
    if hold_until is not None and now < hold_until:
        return False
    return (now - last_activity) >= idle_timeout_seconds