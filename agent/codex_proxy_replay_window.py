"""Bounded replay window for Responses ``reasoning.encrypted_content`` on proxy routes.

``encrypted_content`` is sealed to the backend identity that minted it. A first-party issuer
(``codex_backend``, ``xai_responses``, ``github_responses``) keeps that identity stable, so those
routes replay their full reasoning history. A proxy or aggregator route (its issuer kind starts with
``other:``) can rotate that identity between turns or across a gateway restart, and the provider then
answers a replayed blob with HTTP 400 ``invalid_encrypted_content``.

This module bounds how much history such a route sends: only the ``agent.codex_proxy_replay_turns``
most recent assistant turns carrying reasoning keep their encrypted sidecar (default 2, 0 = send
none). One rejected blob then costs at most that many turns instead of the whole session. Assistant
text is never touched, so the conversation itself is unaffected.

This limits blast radius only. It does not decide *when* to stop replaying: that verdict belongs to
``AIAgent._disable_codex_reasoning_replay`` and its recovery in ``agent/turn_recovery.py``.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Replay window for proxy/aggregator issuers (``agent.codex_proxy_replay_turns``).
DEFAULT_PROXY_REPLAY_TURNS = 2


def proxy_replay_turns_from_config(raw: Any) -> int:
    """``agent.codex_proxy_replay_turns``: non-negative int (0 = replay nothing for proxy issuers);
    absent or invalid falls back to :data:`DEFAULT_PROXY_REPLAY_TURNS` with a warning."""
    if raw is None:
        return DEFAULT_PROXY_REPLAY_TURNS
    value: Optional[int] = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = raw
    elif not isinstance(raw, bool):
        with suppress(TypeError, ValueError):
            value = int(str(raw).strip())
    if value is None or value < 0:
        logger.warning(
            "agent.codex_proxy_replay_turns=%r is not a non-negative integer; using %d",
            raw, DEFAULT_PROXY_REPLAY_TURNS,
        )
        return DEFAULT_PROXY_REPLAY_TURNS
    return value


def proxy_replay_max_turns(agent: Any) -> Optional[int]:
    """Replay window (in assistant turns) for proxy issuers on ``agent``'s route.

    None when the agent carries no value, which leaves every caller's replay uncapped. An unusable
    value falls back to the default rather than silently disabling replay.
    """
    raw = getattr(agent, "codex_proxy_replay_turns", None)
    if raw is None:
        return None
    return proxy_replay_turns_from_config(raw)
