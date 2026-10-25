"""Turn leases for plugin side questions (``ctx.side_question``).

At turn finalization, when a loaded plugin holds the ``llm.side_question`` capability, the turn
thread captures the request inputs the side call will replay and hands ``post_llm_call`` an
opaque token. A token buys ONE side request shared by every plugin that receives the hook:
redemption pops the lease whether it succeeds or not. It expires after ``LEASE_TTL_SECONDS`` or
as soon as the parent agent starts its next turn. The captured ``contextvars`` context binds the
owning profile and secret scope on whatever thread redeems the token, and only a plugin context
of that same profile can redeem it. Unattended turns (cron jobs, delegated subagents) get no
token: nobody is there to read an answer bought with the main model's tokens.
"""

from __future__ import annotations

import contextvars
import copy
import logging
import secrets
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Optional

from agent.side_question_request import SideQuestionCapture, ask_text_only, runtime_key

logger = logging.getLogger(__name__)

SIDE_QUESTION_CAPABILITY = "llm.side_question"
LEASE_TTL_SECONDS = 60.0
MAX_PROMPT_CHARS = 4096
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class _Lease:
    capture: SideQuestionCapture
    expires_at: float
    home_key: str  # hermes_home_key() of the profile the turn ran under


_leases: dict[str, _Lease] = {}
_lock = threading.Lock()


def _consented_plugin_loaded() -> bool:
    """True when a loaded, enabled plugin declares ``llm.side_question`` AND it is granted."""
    from hermes_cli.plugin_capabilities import plugin_capability_granted
    from hermes_cli.plugins import get_plugin_manager

    return any(
        loaded.enabled
        and SIDE_QUESTION_CAPABILITY in loaded.manifest.capabilities
        and plugin_capability_granted(key, SIDE_QUESTION_CAPABILITY)
        for key, loaded in list(get_plugin_manager()._plugins.items())
    )


def _purge_locked(now: float, agent: Any = None) -> None:
    """Drop expired leases, leases whose parent is gone, and (on a new turn) ``agent``'s older ones."""
    for stale in [t for t, lease in _leases.items()
                  if lease.expires_at < now or (parent := lease.capture.parent_ref()) is None or parent is agent]:
        del _leases[stale]


def _unattended(agent: Any) -> bool:
    """A turn nobody is waiting on: a persistence-detached fork, a cron run, a delegated child."""
    return (getattr(agent, "_persist_disabled", False) or getattr(agent, "platform", None) == "cron"
            or getattr(agent, "_delegate_depth", 0) > 0)


def issue_side_question_token(agent: Any, messages: list[dict[str, Any]], turn_id: str) -> Optional[str]:
    """Capture the finished turn and lease it under a fresh token; None (nothing captured) for an
    unattended turn (fork, cron run, subagent) or when no consented plugin is loaded. Turn thread
    only: the captured context and profile are the turn's."""
    from hermes_constants import hermes_home_key

    if _unattended(agent) or not _consented_plugin_loaded():
        return None
    capture = SideQuestionCapture(
        parent_ref=weakref.ref(agent),
        messages=copy.deepcopy(messages),
        system_prompt=agent._cached_system_prompt,
        system_prompt_static=getattr(agent, "_cached_system_prompt_static", None),
        tools=copy.deepcopy(agent.tools or []),
        session_id=agent.session_id,
        turn_id=turn_id,
        runtime=runtime_key(agent),
        context=contextvars.copy_context(),
    )
    token = secrets.token_urlsafe(24)
    now = time.monotonic()
    with _lock:
        # Drop this agent's older leases (a new turn) and every expired or orphaned one (each holds
        # a transcript), so the table holds at most one lease per live session agent.
        _purge_locked(now, agent)
        _leases[token] = _Lease(capture, now + LEASE_TTL_SECONDS, hermes_home_key())
    return token


def _redeemable(lease: Optional[_Lease], prompt: Any) -> Optional[str]:
    """Why the popped lease cannot be redeemed with ``prompt``, or None."""
    if lease is None:
        return "unknown_token"
    if time.monotonic() > lease.expires_at:
        return "expired"
    parent = lease.capture.parent_ref()
    if parent is None:
        return "parent_gone"
    if getattr(parent, "_current_turn_id", None) != lease.capture.turn_id:
        return "turn_moved_on"
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        return "bad_prompt"
    return None


def redeem_side_question(
    token: Any, prompt: Any, *, plugin_id: str, home: Any = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Spend the lease behind ``token`` on one text-only question; the answer or None.

    ``home`` is the redeeming plugin's profile home (the active one when omitted). A lease from
    another profile is refused and left for its owner; otherwise this call consumes it whatever
    the outcome."""
    from hermes_constants import hermes_home_key

    home_key = hermes_home_key(home)
    with _lock:
        lease = _leases.get(token) if isinstance(token, str) else None
        if lease is not None and lease.home_key != home_key:
            logger.info("side_question plugin=%s outcome=rejected:other_profile", plugin_id)
            return None
        if lease is not None:
            del _leases[token]
        _purge_locked(time.monotonic())
    reason = _redeemable(lease, prompt)
    if reason is not None:
        logger.info("side_question plugin=%s outcome=rejected:%s", plugin_id, reason)
        return None
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    # Clamp to (0, 60]; NaN and non-positive values fall back to the default.
    timeout = min(timeout, MAX_TIMEOUT_SECONDS) if timeout > 0 else DEFAULT_TIMEOUT_SECONDS
    # A copy: the captured Context may already be entered elsewhere, and entering one Context
    # on two threads at once raises.
    return lease.capture.context.copy().run(
        ask_text_only, lease.capture, prompt, plugin_id=plugin_id, timeout=timeout,
    )
