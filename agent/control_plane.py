"""Frontdesk control plane — schema + classify (Phase 2 substrate).

This module sits one layer above :mod:`agent.frontdesk_policy`.  Where the
policy module returns a *routing verdict*, the control plane wraps it into a
richer :class:`ControlPlaneDecision` carrying:

* an explicit **Intent** (STOP / STATUS / STEER / NEW_TASK_MAIN /
  NEW_TASK_WORKER / ACK / DUPLICATE / NOISE) — the single-most-likely
  interpretation that drives the surface adapter's downstream branch,
* a **mode-gated recommendation** — when frontdesk mode is off, an otherwise
  worker-lane verdict is downgraded to ``MAIN`` so legacy callers keep their
  existing behaviour (design review §3.1 mode-gating invariant),
* a **fingerprint** for replay tests (INV-6 deterministic classification),
* a **transcript_render flag** so the surface knows whether to emit a
  ``control:`` line for the user (INV-2 visibility).

The :class:`WorkerTaskSpec` / :class:`WorkerTaskResult` dataclasses also live
here even though the lane manager that consumes them is Phase 4.  Putting the
schema next to the decision keeps the substrate file count small and lets
Phase 2 tests pin the exact field shapes Phase 4 will rely on.

What this module does NOT do:

* It does **not** dispatch.  Calling :func:`classify` does not run a worker,
  steer a turn, drop a queue entry, or emit a transcript line.  Those are
  surface-adapter responsibilities; the control plane just returns the
  decision object.
* It does **not** consult any runtime state.  Mode gating is supplied by the
  caller via the ``frontdesk_mode_active`` keyword; the module has no idea
  whether a worker lane is registered, a turn is in flight, or a queue is
  empty.  Verifying any of those is the surface adapter's job.
* It does **not** mutate inputs or any imported object.  Every dataclass is
  ``frozen=True``.

Hard boundaries (PRD §9.2 / design review §9.2):

* No persona changes.
* No ``/mode frontdesk`` exposure.
* No worker dispatch wiring.
* No mutation of ``_pending_input`` or the existing ``/busy integrated`` drain
  semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Literal

from agent.frontdesk_policy import (
    FrontdeskConfidence,
    FrontdeskPolicyDecision,
    FrontdeskRecommendation,
    FrontdeskSignal,
    classify_request,
    fingerprint as _fingerprint,
)

__all__ = [
    # enum re-exports (the canonical short names)
    "Signal",
    "Recommendation",
    "Confidence",
    "Intent",
    "WorkerLaneCancelMode",
    # decision schema
    "ControlPlaneDecision",
    # worker schemas (Phase 4 consumers import them from here)
    "WorkerTaskSpec",
    "WorkerTaskResult",
    # entrypoints
    "classify",
    "decide_intent_from_policy",
]


# --------------------------------------------------------------------------
# Enum re-exports (so callers can use the short names from design review §3.1)
# --------------------------------------------------------------------------
Signal = FrontdeskSignal
Recommendation = FrontdeskRecommendation
Confidence = FrontdeskConfidence


class Intent(Enum):
    """The single-most-likely interpretation of one user-input fragment.

    Closed set — each value drives a distinct downstream branch:

    * ``STOP``             — INV-1: hard purge of the queue, never re-inject.
    * ``STATUS``           — answer locally / via ``/tasks`` snapshot.
    * ``STEER``            — surface invokes ``running_agent.steer`` (after
                             verifying main is actually in flight).
    * ``NEW_TASK_MAIN``    — foreground main turn (the conservative default).
    * ``NEW_TASK_WORKER``  — worker lane dispatch (frontdesk-mode-only).
    * ``ACK``              — pure acknowledgement, drop silently with a
                             ``control:`` line.
    * ``DUPLICATE``        — covered by a recent turn, drop with a line.
    * ``NOISE``            — un-classifiable / empty, drop with a line.
    """

    STOP = "stop"
    STATUS = "status"
    STEER = "steer"
    NEW_TASK_MAIN = "new_task_main"
    NEW_TASK_WORKER = "new_task_worker"
    ACK = "ack"
    DUPLICATE = "duplicate"
    NOISE = "noise"


class WorkerLaneCancelMode(Enum):
    """PRD §7.2 three-tier cancel escalation.

    Surface adapters pass this enum into ``worker_lane_manager.cancel`` (Phase
    4).  The schema lives here so transcript-replay tests can fix the spelling
    before any lane manager exists.
    """

    GRACEFUL = "graceful"  # cancel.flag + cancel_token; escalate to HARD after 5s
    HARD = "hard"          # subprocess SIGTERM; escalate to FORCE after 5s
    FORCE = "force"        # subprocess SIGKILL; mark task CANCELLED


# --------------------------------------------------------------------------
# ControlPlaneDecision
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ControlPlaneDecision:
    """The single object the surface adapters consume.

    Hashable (frozen + ``frozenset`` for ``signals``).  ``to_dict()`` returns a
    JSON-safe view suitable for transcript logging, replay-test fixtures, or
    corpus regeneration.

    Invariants (enforced by :func:`classify` and verified by
    ``tests/agent/test_control_plane.py``):

    * ``intent == Intent.STOP`` ⇒ ``recommendation == Recommendation.CONTROL``
      and ``transcript_render is True``.
    * ``intent in {STOP, NEW_TASK_WORKER, STEER}`` ⇒ ``transcript_render
      is True``.
    * ``frontdesk_mode_active is False`` ⇒ ``recommendation in {MAIN,
      CONTROL}`` (worker/steer-shaped policy verdicts are downgraded to MAIN).
    * Same ``(text, frontdesk_mode_active)`` ⇒ same ``fingerprint``.
    """

    intent: Intent
    signals: FrozenSet[Signal]
    recommendation: Recommendation
    confidence: Confidence
    debug_label: str
    transcript_render: bool
    raw_text: str
    frontdesk_mode_active: bool
    fingerprint: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- convenience -----------------------------------------------------
    @property
    def is_stop(self) -> bool:
        return self.intent is Intent.STOP

    @property
    def is_control(self) -> bool:
        return self.recommendation is Recommendation.CONTROL

    @property
    def should_delegate(self) -> bool:
        return self.recommendation is Recommendation.WORKER_LANE

    @property
    def should_steer(self) -> bool:
        return self.recommendation is Recommendation.STEER

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "signals": sorted(s.value for s in self.signals),
            "recommendation": self.recommendation.value,
            "confidence": self.confidence.value,
            "debug_label": self.debug_label,
            "transcript_render": self.transcript_render,
            "raw_text": self.raw_text,
            "frontdesk_mode_active": self.frontdesk_mode_active,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# WorkerTaskSpec / WorkerTaskResult — schemas Phase 4 will consume
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WorkerTaskSpec:
    """One unit of work dispatched to a worker lane.

    Created from a :class:`ControlPlaneDecision` (or directly by a test) and
    handed to ``worker_lane_manager.dispatch`` in Phase 4.  Phase 2 only pins
    the schema so transcript-replay fixtures stay stable across phases.

    Field meanings:

    * ``title`` — short human-readable label ("draft frontdesk PRD review").
    * ``user_intent`` — the verbatim (or minimally trimmed) user fragment.
    * ``context`` — optional caller-supplied background prose.
    * ``requested_artifacts`` — filenames the user explicitly asked for; the
      lane MUST place them under ``$HERMES_HOME/workers/<task_id>/artifacts/``
      (design review §5.2).
    * ``source_surface`` — which surface enqueued it (logging/parity).
    * ``priority`` — lane scheduler hint; "high" can pre-empt "normal".
    * ``lane_name`` — explicit lane name; ``None`` means "default lane"
      (``"claude_code"`` once Phase 4 registers it).
    * ``allow_memory_writes`` — must default False (INV-7).  The user has to
      explicitly opt the worker into memory writes; bare dispatch does not.
    * ``decision_fingerprint`` — the
      :attr:`ControlPlaneDecision.fingerprint` that produced this spec.  Lets a
      replay test link spec → decision → fragment.
    """

    title: str
    user_intent: str
    context: str = ""
    requested_artifacts: tuple[str, ...] = ()
    source_surface: Literal["cli", "tui", "gateway"] = "cli"
    priority: Literal["normal", "high"] = "normal"
    lane_name: str | None = None
    allow_memory_writes: bool = False
    decision_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "user_intent": self.user_intent,
            "context": self.context,
            "requested_artifacts": list(self.requested_artifacts),
            "source_surface": self.source_surface,
            "priority": self.priority,
            "lane_name": self.lane_name,
            "allow_memory_writes": self.allow_memory_writes,
            "decision_fingerprint": self.decision_fingerprint,
        }

    @classmethod
    def from_decision(
        cls,
        decision: ControlPlaneDecision,
        *,
        title: str | None = None,
        context: str = "",
        requested_artifacts: tuple[str, ...] = (),
        source_surface: Literal["cli", "tui", "gateway"] = "cli",
        priority: Literal["normal", "high"] = "normal",
        lane_name: str | None = None,
        allow_memory_writes: bool = False,
    ) -> "WorkerTaskSpec":
        """Build a spec from a control-plane decision.

        Title defaults to the first 60 characters of ``raw_text``.  Memory
        writes default off (INV-7); callers MUST pass
        ``allow_memory_writes=True`` to opt in.  The decision's fingerprint is
        copied through for replay linkage.
        """
        derived_title = title if title is not None else decision.raw_text[:60].strip()
        return cls(
            title=derived_title or "(untitled)",
            user_intent=decision.raw_text,
            context=context,
            requested_artifacts=tuple(requested_artifacts),
            source_surface=source_surface,
            priority=priority,
            lane_name=lane_name,
            allow_memory_writes=allow_memory_writes,
            decision_fingerprint=decision.fingerprint,
        )


@dataclass(frozen=True, slots=True)
class WorkerTaskResult:
    """The shape returned by ``worker_lane_manager.result(task_id)``.

    INV-7 wiring (Phase 4 / Phase 5):

    * ``summary`` — one-line notification the main turn IS allowed to render
      automatically ("워커 #2 완료, artifact: PATH, /tasks import <id>로
      가져오기").
    * ``body`` — the full worker output.  MUST NOT be auto-imported into the
      main conversation.  The main agent reads it only when the user invokes
      ``/tasks import <task_id>`` or explicitly says "워커 결과 가져와줘".
    * ``gate_required`` — ``True`` iff ``body`` is non-empty AND not yet
      imported.  Surface adapters use this flag to render an ``[import gated]``
      chrome in the status bar.
    """

    task_id: str
    status: Literal["queued", "running", "done", "cancelled", "failed"]
    summary: str
    body: str = ""
    artifacts: tuple[str, ...] = ()
    log_path: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    cancel_mode: Literal["none", "graceful", "hard", "force"] = "none"
    gate_required: bool = False

    def __post_init__(self) -> None:
        # INV-7 enforcement: a non-empty body MUST come with gate_required=True.
        # We enforce this in __post_init__ rather than in factory functions so
        # any direct construction (tests, replay fixtures, future Phase 4
        # callers) cannot accidentally land an importable body.
        if self.body and not self.gate_required:
            # Using object.__setattr__ because the dataclass is frozen.
            object.__setattr__(self, "gate_required", True)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "body": self.body,
            "artifacts": list(self.artifacts),
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "cancel_mode": self.cancel_mode,
            "gate_required": self.gate_required,
        }


# --------------------------------------------------------------------------
# Intent mapping
# --------------------------------------------------------------------------
def decide_intent_from_policy(
    policy_decision: FrontdeskPolicyDecision,
    *,
    frontdesk_mode_active: bool,
) -> tuple[Intent, Recommendation]:
    """Map a policy verdict to ``(Intent, Recommendation)``.

    Pure mapping, no side effects.  The two outputs are *not* a degenerate
    pair: the recommendation can downgrade (e.g. ``WORKER_LANE`` →
    ``MAIN``) when frontdesk mode is off, while the intent stays the
    semantically-correct ``NEW_TASK_MAIN`` (matching the *effective* routing).
    Design review §3.1 mode-gating invariant.
    """
    rec = policy_decision.recommendation
    signals = policy_decision.signals

    # ------------------------------------------------------------------
    # CONTROL bucket — STOP / ACK / NOISE / STATUS-flagged-as-CONTROL.
    # ------------------------------------------------------------------
    if rec is Recommendation.CONTROL:
        if Signal.STOP in signals:
            return Intent.STOP, Recommendation.CONTROL
        if Signal.ACK in signals:
            return Intent.ACK, Recommendation.CONTROL
        if Signal.NOISE in signals:
            return Intent.NOISE, Recommendation.CONTROL
        if Signal.DUPLICATE in signals:
            return Intent.DUPLICATE, Recommendation.CONTROL
        # CONTROL with no specific signal -> NOISE as the conservative default.
        return Intent.NOISE, Recommendation.CONTROL

    # ------------------------------------------------------------------
    # MAIN bucket — STATUS hits land here; the rest become NEW_TASK_MAIN.
    # ------------------------------------------------------------------
    if rec is Recommendation.MAIN:
        if Signal.STATUS in signals:
            return Intent.STATUS, Recommendation.MAIN
        return Intent.NEW_TASK_MAIN, Recommendation.MAIN

    # ------------------------------------------------------------------
    # STEER bucket — mode-gated. Surface verifies in-flight before invoking
    # steer(), but Phase 2 must not change legacy off-mode UX; off ⇒ MAIN.
    # ------------------------------------------------------------------
    if rec is Recommendation.STEER:
        if not frontdesk_mode_active:
            return Intent.NEW_TASK_MAIN, Recommendation.MAIN
        return Intent.STEER, Recommendation.STEER

    # ------------------------------------------------------------------
    # WORKER_LANE bucket — mode-gated.  Off ⇒ downgrade to MAIN.
    # ------------------------------------------------------------------
    if rec is Recommendation.WORKER_LANE:
        if not frontdesk_mode_active:
            return Intent.NEW_TASK_MAIN, Recommendation.MAIN
        return Intent.NEW_TASK_WORKER, Recommendation.WORKER_LANE

    # ------------------------------------------------------------------
    # Defensive fallback — an unknown Recommendation value (forward-compat).
    # ------------------------------------------------------------------
    return Intent.NEW_TASK_MAIN, Recommendation.MAIN


# Intents that the surface adapter must surface to the user via a
# ``control:`` transcript line (INV-2 + design review §3.1 visibility rule).
_TRANSCRIPT_VISIBLE_INTENTS: frozenset[Intent] = frozenset(
    {Intent.STOP, Intent.NEW_TASK_WORKER, Intent.STEER, Intent.ACK, Intent.DUPLICATE, Intent.NOISE}
)


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------
def classify(
    text: str,
    *,
    lang_hint: str | None = None,
    frontdesk_mode_active: bool = False,
) -> ControlPlaneDecision:
    """Classify a single user-input fragment into a :class:`ControlPlaneDecision`.

    Pure function: same ``(text, lang_hint, frontdesk_mode_active)`` always
    returns the same value.

    Parameters
    ----------
    text:
        Raw user input.  Leading/trailing whitespace is stripped by the
        underlying policy classifier.
    lang_hint:
        Optional language hint (``"ko"`` short-circuits Korean detection).
    frontdesk_mode_active:
        ``True`` only when the surface has confirmed frontdesk mode is on for
        this session.  When ``False``, worker-lane and steer policy verdicts
        are downgraded to ``MAIN`` so legacy callers keep their existing
        behaviour — this is the central invariant that makes Phase 2 substrate
        safe to land without any UX change.

    Returns
    -------
    ControlPlaneDecision
        A frozen dataclass with the dispatcher's verdict.  Never raises for
        any string input.
    """
    policy_decision = classify_request(text, lang_hint=lang_hint)
    intent, recommendation = decide_intent_from_policy(
        policy_decision, frontdesk_mode_active=frontdesk_mode_active
    )

    transcript_render = intent in _TRANSCRIPT_VISIBLE_INTENTS
    # STATUS is rendered when frontdesk is on (the user wants to see "main
    # answering locally") but kept silent on legacy off-mode to avoid surfacing
    # a new chrome line for unchanged behaviour.
    if intent is Intent.STATUS and frontdesk_mode_active:
        transcript_render = True

    # Forward any policy-side notes; add a mode-downgrade note when the policy
    # said WORKER_LANE or STEER but mode is off.
    notes: tuple[str, ...] = tuple(policy_decision.notes)
    if not frontdesk_mode_active and policy_decision.recommendation in {
        Recommendation.WORKER_LANE,
        Recommendation.STEER,
    }:
        notes = notes + (
            f"downgraded {policy_decision.recommendation.name} -> MAIN: frontdesk mode off",
        )

    return ControlPlaneDecision(
        intent=intent,
        signals=policy_decision.signals,
        recommendation=recommendation,
        confidence=policy_decision.confidence,
        debug_label=policy_decision.debug_label,
        transcript_render=transcript_render,
        raw_text=policy_decision.raw_text,
        frontdesk_mode_active=frontdesk_mode_active,
        fingerprint=_fingerprint(
            policy_decision.raw_text, frontdesk_mode_active=frontdesk_mode_active
        ),
        notes=notes,
    )
