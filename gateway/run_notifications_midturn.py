"""Mid-turn delivery of background process notices for ``GatewayNotificationsMixin``.

Split out of ``gateway/run_notifications.py`` (itself split out of ``gateway/run.py``); bound onto
``GatewayRunner`` through ``GatewayNotificationsMixin``, the same way
``gateway/run_notifications_ownership.py`` is.

A process-registry notice (``completion`` or ``watch_match``) that arrives while its session's turn
is still running is offered to that turn's agent at its next tool-call boundary instead of being
admitted as a separate turn. The agent queues it (``notify_between_tool_calls``) and appends it as
its own ``role:"user"`` row carrying the ``[INTERNAL NOTIFICATION ...]`` marker — never the
user-authority steer marker, because a process's stdout must not speak as the user.

Scope is deliberately narrow. An ``async_delegation`` result carries a durable delivery claim and a
full task wall, so it keeps its admit-as-a-turn path (PR #136708's carrier owns that seam). If a turn
ends without another tool batch, the agent's finalizer hands the notice back through
``result["pending_notifications"]`` and the gateway runs it as its own turn, exactly as an idle
session would have.
"""

from __future__ import annotations

import logging
from typing import Any, cast

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayMidturnNotificationMixin:
    """Offer background process notices to a running turn's next tool boundary."""

    def _midturn_notice_agent(self, source):
        """The live agent of the turn currently running for ``source``'s session, or None.

        Covers both ownership models: the session-authority claim (its adopted agent while that
        generation still executes) and the classic running-agent slot. None when the session is
        idle, still setting up, or its agent has no mid-turn notification seam.
        """
        from gateway.session_authorities import active_authority

        # The runner is the real ``self`` (this is a mixin); ``cast`` follows the file's existing
        # convention for reaching runner attributes from a split topic module.
        runner = cast(Any, self)

        # Every helper reached here is null-safe by contract (`_session_key_for_source` falls back
        # to the config path, `lookup_by_session_key`/`_receiving_agent`/`active_authority` return
        # None when there is no entry or no running claim), so a notice path never needs a
        # catch-all: an unexpected failure is a real bug and should surface, not silently swallow
        # the notice.
        session_key = runner._session_key_for_source(source)
        if not session_key:
            return None
        authority = active_authority(self)
        if authority is not None:
            entry = runner.session_store.lookup_by_session_key(session_key)
            if entry is None:
                return None
            generation = authority.executing.get(entry.session_id)
            if generation is None:
                return None
            from gateway.session_queued_controls import _receiving_agent

            return _receiving_agent(authority, entry.session_id, generation)
        from gateway.run import _AGENT_PENDING_SENTINEL

        state = runner._peek_session_state(session_key)
        agent = state.turn.agent if state is not None else None
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return None
        return agent

    def _deliver_notice_mid_turn(self, evt: dict, synth_text: str, source) -> bool:
        """Hand a process notice to the running turn's next tool boundary; True when accepted.

        Deliberately scoped to process-registry notices. An ``async_delegation`` result carries a
        durable delivery claim and a full task wall, so it keeps its admit-as-a-turn path; and a
        notice queued into a turn that then ends without another tool batch is handed back by the
        agent's finalizer (``pending_notifications``) and runs as its own turn, as before.
        """
        if evt.get("type") not in {"completion", "watch_match"}:
            return False
        agent = self._midturn_notice_agent(source)
        if agent is None:
            return False
        queue_notice = getattr(agent, "notify_between_tool_calls", None)
        if not callable(queue_notice):
            return False
        try:
            accepted = bool(queue_notice(synth_text))
        except Exception:
            logger.debug(
                "Mid-turn notification queue failed for %s", evt.get("session_id", "?"), exc_info=True
            )
            return False
        if accepted:
            logger.info(
                "Background %s delivered mid-turn for chat=%s process=%s (next tool boundary)",
                evt.get("type", "completion"), source.chat_id, evt.get("session_id", "unknown"),
            )
        return accepted
