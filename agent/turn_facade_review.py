"""Foreground ownership of the background-review admission around one ``run_conversation``.

A review fork shares the turn's session_id for cache parity, so the turn publishes itself on the
session BEFORE cancelling the review: a review that has not issued its first request yet refuses
admission (``agent/review_admission.py``) instead of racing the cancel with a full-transcript
request already on the wire. The turn then fences review startup / interrupts an admitted request
and awaits its exit before opening live-turn instrumentation. That wait has no deadline: an
admitted request must acknowledge, and a run that never admitted a fork is revoked atomically by
the canceller, so no turn waits on a worker that may never start (#84423). Extracted from
``agent/turn_facade.py``; the facade owns the order of the three calls.
"""

from contextlib import suppress
from typing import Any


class ForegroundReviewOwnership:
    """One turn's registration with review admission. ``open`` is the first statement of the
    facade's ``try``; ``release_admission`` is the first statement of its ``finally`` (a leaked
    token would block every later review on the session, including when attribute assignment
    itself raised); ``finish`` runs last, after every foreground owner of the session is released."""

    def __init__(self, agent: Any, session_id: str) -> None:
        from agent import review_admission

        self.agent = agent
        self.session_id = session_id
        self.profile_key = review_admission.current_profile_key()
        self.turn_token = None
        self.review_run = None
        self.queue_started = False

    def open(self) -> None:
        from agent import review_admission
        from agent.background_review import (
            cancel_background_review_for_live_turn,
            wait_for_background_review_cancellation,
        )
        from agent.review_idle_queue import QUEUE as review_queue

        agent = self.agent
        # A cache-parity fork (the review, a /btw side question; ``build_cache_parity_fork``
        # stamps it) replays through this same facade on the session it was forked from. It is
        # not a foreground turn: registering it would label the owner ``live_turn_active`` for
        # every other spawn attempt and mask the truthful slot state, and a /btw would fence,
        # then wait on, the review it answers beside — an explicit /refine included.
        if not getattr(agent, "_foreground_exempt_fork", False):
            # Linearize registration with review request admission. If the review owns this lock
            # first it publishes its fork under the run lock, so cancellation finds it; if this
            # turn owns it first the review's foreground sample observes the live token.
            with review_admission.admission_lock():
                self.turn_token = review_admission.note_turn_started(
                    self.session_id, self.profile_key
                )
                self.review_run = cancel_background_review_for_live_turn(
                    agent,
                    wait=False,
                    session_id=self.session_id,
                    profile_key=self.profile_key,
                )
        agent._active_turn_token = self.turn_token
        agent._active_turn_profile_key = self.profile_key
        review_queue.note_turn_started()
        self.queue_started = True
        wait_for_background_review_cancellation(self.review_run)

    def release_admission(self) -> None:
        from agent import review_admission

        if self.turn_token is not None:
            review_admission.note_turn_finished(
                self.session_id, self.turn_token, self.profile_key
            )
        with suppress(Exception):
            if getattr(self.agent, "_active_turn_token", None) == self.turn_token:
                self.agent._active_turn_token = None
            if (
                getattr(self.agent, "_active_turn_profile_key", None)
                == self.profile_key
            ):
                self.agent._active_turn_profile_key = None

    def finish(self) -> None:
        from agent.review_idle_queue import QUEUE as review_queue

        if self.queue_started:
            # Balance note_turn_started so the idle queue's live-turn count cannot leak.
            with suppress(Exception):
                review_queue.note_turn_finished()
        post_turn_review = getattr(
            self.agent, "_post_turn_background_review_candidate", None
        )
        if isinstance(post_turn_review, dict):
            with suppress(AttributeError):
                del self.agent._post_turn_background_review_candidate
            # The finalizer staged this while the foreground still owned the durable session
            # row. Start it only after every foreground owner is released.
            with suppress(Exception):
                self.agent._spawn_background_review(**post_turn_review)
