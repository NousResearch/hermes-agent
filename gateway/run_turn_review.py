"""The gateway's foreground review ownership of one turn (``gateway/run_turn.py`` sibling).

``_handle_message_with_agent`` registers every gateway turn as live in ``agent.review_admission``
(``_GatewayReviewAdmission.begin``) before it prepares the turn, binds the turn's agent to that
admission, freezes the terminal review candidate on it, and parks ``finish`` — the completion
that releases the token and spawns the candidate — where the turn's delivery owner reads it
(``begin_gateway_review_admission``): the live session Event an adapter task completes with
its send (``platforms/base_followup_admission.py``), the event a direct-call ingress completes
itself, or the drain's handover (``session_ingress_review.py``). Ownership is released before
the spawn hop, and a dropped candidate logs a body-free, owner-tagged slug.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from contextlib import suppress
from contextvars import copy_context
from typing import Any, Optional

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


@dataclasses.dataclass
class _GatewayReviewAdmission:
    """Foreground ownership spanning gateway prework through confirmed delivery."""

    session_id: str
    profile_key: str
    token: int
    context: Any = None
    agent: Any = None
    _candidate: Optional[dict[str, Any]] = None
    _finished: bool = False
    _lock: Any = dataclasses.field(default_factory=threading.Lock)

    @classmethod
    async def begin(
        cls, owner: Any, session_id: str, profile_key: str
    ) -> _GatewayReviewAdmission:
        from agent import review_admission
        from agent.background_review import (
            cancel_background_review_for_live_turn,
            wait_for_background_review_cancellation,
        )

        with review_admission.admission_lock():
            token = review_admission.note_turn_started(session_id, profile_key)
            review_run = cancel_background_review_for_live_turn(
                owner,
                wait=False,
                session_id=session_id,
                profile_key=profile_key,
            )
        try:
            await asyncio.to_thread(wait_for_background_review_cancellation, review_run)
        except BaseException:
            review_admission.note_turn_finished(session_id, token, profile_key)
            raise
        return cls(session_id, profile_key, token)

    @property
    def owner_tag(self) -> str:
        """Hashed (profile, session) owner for log lines; never the raw identifiers."""
        from agent import review_admission

        return review_admission.owner_tag(self.profile_key, self.session_id)

    def bind_agent(self, agent: Any) -> None:
        """Route this gateway turn's automatic candidate to the delivery owner."""
        with self._lock:
            self.agent = agent
            self.context = copy_context()
            self._candidate = None
        agent._gateway_review_admission = self
        # Session hygiene in ``_hmwa_prepare_turn`` can rotate the session BEFORE this bind, when
        # neither the facade token nor this admission is on the agent for the rotation to alias.
        # Alias only (no wait on the event-loop thread): the facade's own cancel + wait for the
        # rotated id runs right after this on the executor thread.
        current = str(getattr(agent, "session_id", None) or "")
        if current and current != self.session_id:
            self.alias_session(current)

    def alias_session(self, session_id: str) -> None:
        """Keep outer gateway ownership on a compression/resume child."""
        from agent import review_admission

        with review_admission.admission_lock():
            if review_admission.alias_turn_session(
                self.token, session_id, self.profile_key
            ):
                with self._lock:
                    self.session_id = session_id

    def capture_candidate(
        self,
        agent: Any,
        messages: list[dict[str, Any]],
        *,
        review_memory: bool,
        review_skills: bool,
    ) -> None:
        """Freeze the terminal candidate on this delivery owner, never on a cached agent."""
        from agent.turn_finalizer import _clone_background_review_messages

        candidate = {
            "messages_snapshot": _clone_background_review_messages(messages),
            "review_memory": review_memory,
            "review_skills": review_skills,
            "_spawning_turn_token": self.token,
            "_review_profile_key": self.profile_key,
            "_review_session_id": str(
                getattr(agent, "session_id", None) or self.session_id
            ),
        }
        with self._lock:
            if not self._finished:
                self._candidate = candidate

    def adopt_candidate(
        self, candidate: dict[str, Any], owner: _GatewayReviewAdmission
    ) -> bool:
        """Carry a nested turn's confirmed candidate until this turn's ownership is released.

        ``/retry`` typed while this turn's reply is on the wire rewinds the transcript and nests
        the retried turn, whose delivery completes first. Spawning there is refused (this token
        is still live) and this turn's own candidate — the transcript ``/retry`` retracted —
        would be reviewed instead. The nested candidate replaces it and spawns when THIS turn
        releases, gated by this turn's delivery outcome like its own. False once this turn has
        finished: the nested ``owner`` then spawns on its own.
        """
        from agent import review_admission

        with self._lock:
            if self._finished:
                return False
            superseded = self._candidate
            self._candidate = candidate
            if self.agent is None:
                self.agent, self.context = owner.agent, owner.context
        if isinstance(superseded, dict):
            logger.info(
                "Background review candidate superseded (owner=%s): %s",
                review_admission.owner_tag(
                    self.profile_key,
                    superseded.get("_review_session_id") or self.session_id,
                ),
                review_admission.REASON_CANDIDATE_SUPERSEDED,
            )
        return True

    def drop_candidate(self, cause: str) -> bool:
        """Discard the parked candidate before delivery; ownership stays live until ``finish``.

        The compression-exhaustion auto-reset rotates the chat to a fresh, parentless session
        and re-homes the follow-ups queued behind the exhausted turn onto it. The registry keys
        that session apart from this one, so the exhausted transcript's review would decode
        beside the follow-up's turn (the overlap the follow-up fence exists to prevent), and it
        has nothing to learn: the turn answered with the timeout notice. True when a candidate
        was held; the drop logs the owner-tagged ``cause``.
        """
        from agent import review_admission

        with self._lock:
            candidate, self._candidate = self._candidate, None
        if not isinstance(candidate, dict):
            return False
        logger.info(
            "Background review skipped (owner=%s): %s",
            review_admission.owner_tag(
                self.profile_key, candidate.get("_review_session_id") or self.session_id
            ),
            cause,
        )
        return True

    def finish(
        self,
        *,
        delivery_succeeded: bool,
        cause: Optional[str] = None,
        outer: Optional[_GatewayReviewAdmission] = None,
    ) -> Optional[threading.Thread]:
        """Release once; start the latest candidate only after a successful terminal delivery.

        The spawn runs on its own thread: every caller sits on the gateway event loop, and the
        spawn pipeline (config read, replay bound, structural clone, runtime resolution) is
        O(transcript). Ownership is released HERE, before the hop, so the spawn's foreground
        probe observes the truth. A dropped candidate logs why (``cause`` names a drain
        handoff) so a starved review is as greppable as a refused one. ``outer`` is the turn
        whose delivery window this turn was nested in: while it still owns the session the
        candidate is handed to it (``adopt_candidate``) instead of being refused at spawn.
        """
        with self._lock:
            if self._finished:
                return None
            self._finished = True
        from agent import review_admission

        review_admission.note_turn_finished(
            self.session_id, self.token, self.profile_key
        )
        candidate = self._candidate
        self._candidate = None
        agent = self.agent
        if agent is not None:
            if getattr(agent, "_gateway_review_admission", None) is self:
                agent._gateway_review_admission = None
        if not isinstance(candidate, dict):
            return None
        if not delivery_succeeded:
            logger.info(
                "Background review skipped (owner=%s): %s",
                review_admission.owner_tag(
                    self.profile_key,
                    candidate.get("_review_session_id") or self.session_id,
                ),
                cause or review_admission.REASON_DELIVERY_UNCONFIRMED,
            )
            return None
        if (
            isinstance(outer, _GatewayReviewAdmission)
            and outer is not self
            and outer.adopt_candidate(candidate, self)
        ):
            return None
        if self.context is None or agent is None:
            return None
        # The Context captured at bind time carries the turn's profile/secret scope; the loop
        # thread's own context is the launch profile's, so it is entered explicitly here.
        spawner = threading.Thread(  # health: allow HX012 -- runs under the turn's bound Context, not the spawner's
            target=self.context.run,
            args=(agent._spawn_background_review,),
            kwargs=candidate,
            daemon=True,
            name="bg-review-spawn",
        )
        try:
            spawner.start()
        except Exception:  # ownership is already released: greppable, never fatal
            logger.warning(
                "Background review delivery completion failed (owner=%s, reason=%s)",
                review_admission.owner_tag(
                    self.profile_key,
                    candidate.get("_review_session_id") or self.session_id,
                ),
                review_admission.REASON_COMPLETION_ERROR,
                exc_info=True,
            )
            return None
        return spawner


async def begin_gateway_review_admission(
    runner: Any, event: Any, source: Any, session_entry: Any, quick_key: str
) -> _GatewayReviewAdmission:
    """Register the turn as live and park its completion where its delivery owner reads it:
    the live session Event for an adapter task (a drain handoff re-uses the Event), or the
    event itself for a direct-call ingress that completes ownership from its own event (the
    CLI->gateway handoff; the turn ``/retry`` nests, re-homed onto the command event its inline
    dispatch completes) — it may run beside a live adapter task on the same chat, whose Event
    carries that task's own completion, or beside a drain task that reads the guard it
    installed."""
    from agent import review_admission
    from hermes_constants import hermes_home_key

    profile_key = review_admission.current_profile_key()
    with suppress(Exception):
        profile_key = hermes_home_key(runner._resolve_profile_home_for_source(source))
    admission = await _GatewayReviewAdmission.begin(
        runner, str(session_entry.session_id), profile_key
    )
    delivery_carrier = event
    if not getattr(event, "_gateway_review_completes_on_event", False):
        with suppress(Exception):
            adapter = runner._delivery_adapter_for(source)
            delivery_carrier = (
                getattr(adapter, "_active_sessions", {}).get(quick_key) or event
            )
    delivery_carrier._gateway_review_delivery_complete = admission.finish
    return admission
