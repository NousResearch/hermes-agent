"""First-class control-plane authority substrate.

Where :mod:`hermes_cli.control_plane_trace` only *observes* control-plane
incidents (it emits read-only, ``dry_run`` event records), this module
provides the **authority primitives** those incidents were missing:

* an orthogonal :class:`WorkVerdict` vs :class:`DeliveryAckState`
  representation, so an ACK that is never delivered cannot mutate or
  downgrade the work verdict it was meant to carry;
* :class:`ContextContract` / :class:`LaneContract` / :class:`ResumePacket`
  / :class:`ScopedTodo` authority objects keyed on a monotonic
  ``authority_epoch``, so stale compaction summaries, stale todos and
  stale history can be *detected and demoted* before they are allowed to
  authorize a side-effecting mutation;
* :func:`validate_resume_packet_authority` and
  :func:`authorize_pre_action_mutation`, two fail-closed gates that
  return a typed :class:`AuthorizationRecord` (or raise) instead of
  merely logging the conflict.

Design constraints honoured here:

* **Generic / upstreamable.** Nothing in this module hard-codes a
  user-specific lane, board or chat. Lane names like ``#hermes-main`` or
  ``#warroom`` only ever appear in tests/fixtures/examples, never here.
* **No raw transcripts or secrets.** Every ``to_dict`` / ``project``
  helper emits only ids, enum values, epochs and status — never message
  bodies, tokens or credentials.
* **Pure.** This module performs no DB writes and no network/gateway
  calls. It is import-clean of the ``gateway`` package; the one optional
  dependency (legacy ``Origin/return_to:`` prose parsing) is imported
  lazily inside :meth:`OriginReturnContract.from_prose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional
import hashlib
import os
import time


# ---------------------------------------------------------------------------
# Typed verdict / ACK vocabulary
# ---------------------------------------------------------------------------


class WorkVerdict(str, Enum):
    """The authoritative outcome of a unit of work.

    This is orthogonal to delivery: a verdict is a fact about the *work*,
    not about whether anyone was told. See :class:`DeliveryEnvelope`.
    """

    PENDING = "PENDING"
    GO = "GO"
    BLOCK = "BLOCK"
    NEED_MORE = "NEED_MORE"
    NO_GO = "NO_GO"

    @property
    def is_terminal(self) -> bool:
        """True once the work itself has reached a final verdict."""
        return self in (WorkVerdict.GO, WorkVerdict.BLOCK, WorkVerdict.NO_GO)


class AckStatus(str, Enum):
    """Durable delivery state of an ACK — orthogonal to :class:`WorkVerdict`.

    ``SKIPPED_WITH_REASON`` is a first-class, *non-failure* outcome: e.g.
    a task finished ``GO`` but had no subscription to ACK back to. That is
    not a work problem and must never be conflated with ``NEED_MORE``.
    """

    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    SKIPPED_WITH_REASON = "SKIPPED_WITH_REASON"


class AckMode(str, Enum):
    """How a delivered ACK reached the origin.

    ``PASSIVE_SENT`` mirrors/notifies without waking an agent;
    ``ACTIVE_WAKE`` additionally injects a live trigger. These are kept
    distinct so a passive notification is never mistaken for an agent
    having been actively woken.
    """

    PASSIVE_SENT = "passive_sent"
    ACTIVE_WAKE = "active_wake"


#: ACK errors observed on the gateway notifier delivery path.
KNOWN_ACK_ERRORS = (
    "no_live_gateway_runner",
    "no_subscription",
    "no_target",
    "no_live_adapter",
)


# ---------------------------------------------------------------------------
# Orthogonal ACK state + delivery envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryAckState:
    """Durable ACK state, kept strictly orthogonal to the work verdict.

    A ``FAILED`` ack carries an ``error``; a ``SKIPPED_WITH_REASON`` ack
    carries a ``reason``; a ``SENT`` ack carries the :class:`AckMode` it
    was delivered with. None of these fields can express a work verdict —
    that separation is the whole point.
    """

    status: AckStatus = AckStatus.PENDING
    mode: Optional[AckMode] = None
    error: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status == AckStatus.SENT and self.mode is None:
            raise ValueError("a SENT ack must record an AckMode (passive/active)")
        if self.status == AckStatus.FAILED and not self.error:
            raise ValueError("a FAILED ack must record an error")
        if self.status == AckStatus.SKIPPED_WITH_REASON and not self.reason:
            raise ValueError("a SKIPPED_WITH_REASON ack must record a reason")

    @property
    def delivered(self) -> bool:
        return self.status == AckStatus.SENT

    @property
    def is_active_wake(self) -> bool:
        """True only for an ACK actually delivered as an active wake."""
        return self.status == AckStatus.SENT and self.mode == AckMode.ACTIVE_WAKE

    @property
    def is_passive_sent(self) -> bool:
        """True only for an ACK delivered passively (no agent wake)."""
        return self.status == AckStatus.SENT and self.mode == AckMode.PASSIVE_SENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "mode": self.mode.value if self.mode else None,
            "error": self.error,
            "reason": self.reason,
        }

    # -- convenience constructors -----------------------------------------

    @classmethod
    def pending(cls) -> "DeliveryAckState":
        return cls(status=AckStatus.PENDING)

    @classmethod
    def passive_sent(cls) -> "DeliveryAckState":
        return cls(status=AckStatus.SENT, mode=AckMode.PASSIVE_SENT)

    @classmethod
    def active_wake(cls) -> "DeliveryAckState":
        return cls(status=AckStatus.SENT, mode=AckMode.ACTIVE_WAKE)

    @classmethod
    def failed(cls, error: str) -> "DeliveryAckState":
        return cls(status=AckStatus.FAILED, error=error)

    @classmethod
    def skipped(cls, reason: str) -> "DeliveryAckState":
        return cls(status=AckStatus.SKIPPED_WITH_REASON, reason=reason)


@dataclass(frozen=True)
class DeliveryEnvelope:
    """Couples a :class:`WorkVerdict` with its (orthogonal) ACK state.

    The work verdict is authoritative and immutable for the lifetime of an
    envelope: :meth:`with_ack` returns a *new* envelope whose
    ``work_verdict`` is byte-for-byte preserved. An ACK that fails or is
    skipped therefore can never downgrade a ``GO``/``BLOCK`` to
    ``NEED_MORE``.
    """

    task_id: str
    work_verdict: WorkVerdict
    ack: DeliveryAckState = field(default_factory=DeliveryAckState.pending)

    def with_ack(self, ack: DeliveryAckState) -> "DeliveryEnvelope":
        """Return a new envelope with ``ack`` replaced; verdict unchanged."""
        return replace(self, ack=ack)

    @property
    def work_complete(self) -> bool:
        """Whether the *work* is done — independent of ACK delivery."""
        return self.work_verdict.is_terminal

    @property
    def ack_outstanding(self) -> bool:
        """Whether the ACK still needs attention (pending or failed)."""
        return self.ack.status in (AckStatus.PENDING, AckStatus.FAILED)

    @property
    def needs_more_work(self) -> bool:
        """True only when the *work verdict itself* is ``NEED_MORE``.

        A missing/failed/skipped ACK is explicitly NOT ``NEED_MORE``: the
        work can be fully ``GO``/``BLOCK`` while its ACK is skipped.
        """
        return self.work_verdict == WorkVerdict.NEED_MORE

    def project(self) -> dict[str, Any]:
        """Stable JSON projection that keeps work and ACK facets separate."""
        return {
            "task_id": self.task_id,
            "work": {
                "verdict": self.work_verdict.value,
                "complete": self.work_complete,
            },
            "ack": self.ack.to_dict(),
        }


# ---------------------------------------------------------------------------
# Origin / return_to structured contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginReturnContract:
    """Structured, immutable origin/return_to routing contract.

    Replaces ad-hoc body-prose parsing at the point of *use*: legacy
    ``Origin/return_to:`` prose may still appear in a task body (see
    :meth:`from_prose`), but it is materialized into this stable object
    with an opaque :attr:`return_id` rather than re-parsed on every read.
    """

    platform: str
    chat_id: str
    thread_id: Optional[str] = None
    lane: Optional[str] = None
    source: str = "structured"  # "structured" | "legacy_prose"

    @property
    def target(self) -> str:
        """Canonical ``platform:chat_id[:thread_id]`` gateway target."""
        t = f"{self.platform}:{self.chat_id}"
        if self.thread_id:
            t += f":{self.thread_id}"
        return t

    @property
    def return_id(self) -> str:
        """Stable, opaque handle for this return route.

        Derived from the canonical target only — it carries no message
        body or secret, and is byte-stable across processes.
        """
        digest = hashlib.sha1(self.target.encode("utf-8")).hexdigest()
        return f"ret_{digest[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_id": self.return_id,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "lane": self.lane,
            "source": self.source,
        }

    @classmethod
    def from_origin_dict(
        cls,
        origin: Optional[dict[str, Any]],
        *,
        lane: Optional[str] = None,
        source: str = "legacy_prose",
    ) -> Optional["OriginReturnContract"]:
        """Materialize a contract from a parsed ``{platform,chat_id,...}`` dict."""
        if not origin or not origin.get("platform") or not origin.get("chat_id"):
            return None
        return cls(
            platform=str(origin["platform"]),
            chat_id=str(origin["chat_id"]),
            thread_id=origin.get("thread_id") or None,
            lane=lane,
            source=source,
        )

    @classmethod
    def from_prose(
        cls, body: Optional[str], *, lane: Optional[str] = None
    ) -> Optional["OriginReturnContract"]:
        """Backcompat: materialize a structured contract from body prose.

        The actual ``Origin/return_to:`` prose grammar lives in
        :func:`hermes_cli.kanban_db.parse_origin_return_to` (single source
        of truth); it is imported lazily so this module stays pure and
        cheap to import. Prose parsing is retained only as a migration /
        back-compat path — new callers should construct the contract
        directly.
        """
        if not body:
            return None
        try:
            from hermes_cli.kanban_db import parse_origin_return_to
        except Exception:
            return None
        return cls.from_origin_dict(
            parse_origin_return_to(body), lane=lane, source="legacy_prose"
        )


# ---------------------------------------------------------------------------
# Lane / context authority primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneContract:
    """Authority contract for one control-plane lane.

    ``authority_epoch`` is the lane's current monotonic authority counter:
    any context/todo/compaction/history carrying an epoch *below* it is
    stale. ``allowed_mutation_lanes`` lists other lanes this lane may
    mutate without explicit approval (a pre-approved route); any other
    cross-lane mutation needs ``explicit_approval`` and
    ``allow_cross_lane_with_approval``.
    """

    lane: str
    authority_epoch: int
    allowed_mutation_lanes: frozenset[str] = frozenset()
    allow_cross_lane_with_approval: bool = True
    requires_wake: bool = False

    def permits_mutation(
        self, target_lane: str, *, explicit_approval: bool = False
    ) -> bool:
        """Whether this lane may mutate ``target_lane``.

        In-lane mutation is always permitted. A different lane is
        permitted only via a pre-approved route or an explicit approval.
        """
        if target_lane == self.lane:
            return True
        if target_lane in self.allowed_mutation_lanes:
            return True
        if explicit_approval and self.allow_cross_lane_with_approval:
            return True
        return False


@dataclass(frozen=True)
class ContextContract:
    """A context/compaction snapshot tagged with its authority epoch."""

    context_id: str
    lane: str
    authority_epoch: int
    created_at: int = field(default_factory=lambda: int(time.time()))

    def is_stale(self, lane_contract: LaneContract) -> bool:
        """Stale if it belongs to another lane or predates current authority."""
        return (
            self.lane != lane_contract.lane
            or self.authority_epoch < lane_contract.authority_epoch
        )


@dataclass(frozen=True)
class ScopedTodo:
    """A todo scoped to a lane + authority epoch.

    A todo carried over from a stale compaction/plan must not be allowed
    to authorize a fresh mutation; :meth:`is_stale` makes that detectable.
    """

    todo_id: str
    lane: str
    authority_epoch: int
    description: str = ""
    done: bool = False

    def is_stale(self, lane_contract: LaneContract) -> bool:
        return (
            self.lane != lane_contract.lane
            or self.authority_epoch < lane_contract.authority_epoch
        )


@dataclass(frozen=True)
class ResumePacket:
    """Everything an agent rehydrates with when resuming work on a lane.

    Each authority-bearing source carries its own epoch so a packet
    assembled from a *stale* compaction or *stale* history can be demoted
    wholesale — see :func:`validate_resume_packet_authority`.
    """

    packet_id: str
    lane: str
    context: ContextContract
    scoped_todos: tuple[ScopedTodo, ...] = ()
    compaction_epoch: Optional[int] = None
    history_epoch: Optional[int] = None
    origin_contract: Optional[OriginReturnContract] = None


# ---------------------------------------------------------------------------
# Authorization records + fail-closed gates
# ---------------------------------------------------------------------------


class StaleAuthorityError(RuntimeError):
    """Raised when a blocked :class:`AuthorizationRecord` is enforced."""

    def __init__(self, message: str, *, record: "AuthorizationRecord") -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class AuthorizationRecord:
    """Typed verdict of an authority check — the gate's durable output.

    ``authorized`` is the decision; ``demoted`` is set when the block was
    caused by stale/superseded authority (vs. e.g. a cross-lane route
    violation); ``blocked_sources`` names each offending source so the
    block is auditable without storing any raw transcript.
    """

    authorized: bool
    action: str
    lane: str
    reason: str
    demoted: bool = False
    blocked_sources: tuple[str, ...] = ()
    decided_at: int = field(default_factory=lambda: int(time.time()))

    def __bool__(self) -> bool:
        return self.authorized

    def raise_for_status(self) -> "AuthorizationRecord":
        """Return self if authorized, else raise :class:`StaleAuthorityError`."""
        if not self.authorized:
            raise StaleAuthorityError(self.reason, record=self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "action": self.action,
            "lane": self.lane,
            "reason": self.reason,
            "demoted": self.demoted,
            "blocked_sources": list(self.blocked_sources),
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class MutationAction:
    """A side-effecting action awaiting authorization.

    ``kind`` is a coarse category (e.g. ``"board"``, ``"cron"``,
    ``"send"``); ``target_lane`` is the lane the action would mutate.
    """

    kind: str
    target_lane: str
    description: str = ""


def validate_resume_packet_authority(
    packet: ResumePacket, lane_contract: LaneContract
) -> AuthorizationRecord:
    """Fail-closed: confirm a resume packet's authority is current.

    Every authority-bearing source in the packet — its context contract,
    any compaction epoch, any history epoch, and every scoped todo — is
    checked against ``lane_contract``. If *any* is stale or cross-lane the
    returned record is ``authorized=False, demoted=True`` and names the
    offending sources. Only a fully-current packet authorizes.
    """
    blocked: list[str] = []

    if packet.lane != lane_contract.lane:
        blocked.append(f"resume_packet:lane={packet.lane}")
    if packet.context.is_stale(lane_contract):
        blocked.append(f"context:{packet.context.context_id}")
    if (
        packet.compaction_epoch is not None
        and packet.compaction_epoch < lane_contract.authority_epoch
    ):
        blocked.append(f"compaction:epoch={packet.compaction_epoch}")
    if (
        packet.history_epoch is not None
        and packet.history_epoch < lane_contract.authority_epoch
    ):
        blocked.append(f"history:epoch={packet.history_epoch}")
    for todo in packet.scoped_todos:
        if todo.is_stale(lane_contract):
            blocked.append(f"todo:{todo.todo_id}")

    if blocked:
        return AuthorizationRecord(
            authorized=False,
            action="validate_resume_packet",
            lane=lane_contract.lane,
            reason=(
                "stale or cross-lane authority sources must be demoted "
                "before they can authorize a side-effecting mutation"
            ),
            demoted=True,
            blocked_sources=tuple(blocked),
        )
    return AuthorizationRecord(
        authorized=True,
        action="validate_resume_packet",
        lane=lane_contract.lane,
        reason="resume packet authority is current",
    )


def authorize_pre_action_mutation(
    action: MutationAction,
    lane_contract: LaneContract,
    resume_packet: ResumePacket,
    *,
    scoped_todos: tuple[ScopedTodo, ...] = (),
    origin_contract: Optional[OriginReturnContract] = None,
    explicit_approval: bool = False,
) -> AuthorizationRecord:
    """Fail-closed gate to run *before* any board/cron/send mutation.

    Authorization is granted only when **all** of the following hold:

    1. the :class:`ResumePacket` authority is current
       (:func:`validate_resume_packet_authority`) — stale compaction,
       stale history or stale todos block here and are *demoted*;
    2. no extra ``scoped_todos`` passed at the call site are stale;
    3. the action's ``target_lane`` is in-lane, on a pre-approved route,
       or carries an ``explicit_approval`` — a cross-lane mutation without
       one is blocked.

    The default outcome on any doubt is *blocked*: this gate never
    silently allows a mutation. ``origin_contract`` (when supplied) is
    recorded on the returned record's reason for auditability.
    """
    packet_auth = validate_resume_packet_authority(resume_packet, lane_contract)
    if not packet_auth.authorized:
        return AuthorizationRecord(
            authorized=False,
            action=action.kind,
            lane=lane_contract.lane,
            reason=f"blocked: {packet_auth.reason}",
            demoted=True,
            blocked_sources=packet_auth.blocked_sources,
        )

    stale_todos = tuple(
        f"todo:{t.todo_id}" for t in scoped_todos if t.is_stale(lane_contract)
    )
    if stale_todos:
        return AuthorizationRecord(
            authorized=False,
            action=action.kind,
            lane=lane_contract.lane,
            reason=(
                "blocked: stale scoped todo(s) cannot authorize a mutation; "
                "demote or refresh them first"
            ),
            demoted=True,
            blocked_sources=stale_todos,
        )

    if not lane_contract.permits_mutation(
        action.target_lane, explicit_approval=explicit_approval
    ):
        return AuthorizationRecord(
            authorized=False,
            action=action.kind,
            lane=lane_contract.lane,
            reason=(
                f"blocked: cross-lane mutation into {action.target_lane!r} "
                f"from lane {lane_contract.lane!r} requires an explicit "
                f"approval or a pre-approved route"
            ),
            blocked_sources=(f"cross_lane:{action.target_lane}",),
        )

    reason = "authorized: current authority, in-lane or approved route"
    if origin_contract is not None:
        reason += f" (return_id={origin_contract.return_id})"
    return AuthorizationRecord(
        authorized=True,
        action=action.kind,
        lane=lane_contract.lane,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Environment / session-contract bridge
# ---------------------------------------------------------------------------
#
# Real mutation entry points (send_message, Kanban board mutations) cannot
# carry a ResumePacket object across process boundaries. Instead the active
# session publishes its control-plane authority through environment
# variables, and the bridge below reconstructs the typed contracts so the
# fail-closed gate can run before any side effect.
#
# When NO authority contract is configured (``HERMES_CONTROL_PLANE_LANE``
# unset) the bridge returns an explicit *authorized* record with the reason
# ``"no authority contract configured"`` — so existing CLI / tests that
# never set these vars keep working unchanged.

#: Current lane the session holds authority for. Presence of this var is
#: what switches the bridge from no-op to enforcing.
ENV_LANE = "HERMES_CONTROL_PLANE_LANE"
#: Current monotonic authority epoch for the lane.
ENV_EPOCH = "HERMES_CONTROL_PLANE_EPOCH"
#: Comma-separated lanes this lane may mutate without explicit approval.
ENV_ALLOWED_LANES = "HERMES_CONTROL_PLANE_ALLOWED_LANES"
#: Truthy when an explicit cross-lane approval is in force.
ENV_APPROVAL = "HERMES_CONTROL_PLANE_APPROVAL"
#: Optional explicit target lane for a mutation (else: in-lane).
ENV_TARGET_LANE = "HERMES_CONTROL_PLANE_TARGET_LANE"
#: Lane the resume packet's authority belongs to (else: current lane).
ENV_RESUME_LANE = "HERMES_RESUME_LANE"
#: Authority epoch of the rehydrated context (else: current epoch).
ENV_RESUME_CONTEXT_EPOCH = "HERMES_RESUME_CONTEXT_EPOCH"
#: Authority epoch of the compaction summary the session resumed from.
ENV_RESUME_COMPACTION_EPOCH = "HERMES_RESUME_COMPACTION_EPOCH"
#: Authority epoch of the message history the session resumed from.
ENV_RESUME_HISTORY_EPOCH = "HERMES_RESUME_HISTORY_EPOCH"
#: Comma-separated authority epochs of carried-over scoped todos.
ENV_RESUME_TODO_EPOCHS = "HERMES_RESUME_TODO_EPOCHS"


def _env_bool(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def lane_contract_from_env(env: Mapping[str, str], lane: str) -> LaneContract:
    """Reconstruct the lane's :class:`LaneContract` from env vars."""
    allowed = frozenset(
        s.strip()
        for s in (env.get(ENV_ALLOWED_LANES) or "").split(",")
        if s.strip()
    )
    return LaneContract(
        lane=lane,
        authority_epoch=_env_int(env.get(ENV_EPOCH), default=0) or 0,
        allowed_mutation_lanes=allowed,
    )


def resume_packet_from_env(env: Mapping[str, str], lane: str) -> ResumePacket:
    """Reconstruct the session's :class:`ResumePacket` from env vars.

    Each authority-bearing source (context / compaction / history / todos)
    carries its own epoch so a session that resumed from a stale source
    fails the gate. Sources whose epoch var is unset are simply absent.
    """
    epoch = _env_int(env.get(ENV_EPOCH), default=0) or 0
    resume_lane = (env.get(ENV_RESUME_LANE) or lane).strip()
    ctx_epoch = _env_int(env.get(ENV_RESUME_CONTEXT_EPOCH), default=epoch)
    todos: list[ScopedTodo] = []
    raw_todos = (env.get(ENV_RESUME_TODO_EPOCHS) or "").strip()
    for idx, tok in enumerate(t for t in raw_todos.split(",") if t.strip()):
        todos.append(
            ScopedTodo(
                todo_id=f"env_todo_{idx}",
                lane=resume_lane,
                authority_epoch=_env_int(tok, default=epoch) or 0,
            )
        )
    return ResumePacket(
        packet_id="env_resume",
        lane=resume_lane,
        context=ContextContract(
            context_id="env_context",
            lane=resume_lane,
            authority_epoch=ctx_epoch if ctx_epoch is not None else epoch,
        ),
        scoped_todos=tuple(todos),
        compaction_epoch=_env_int(env.get(ENV_RESUME_COMPACTION_EPOCH)),
        history_epoch=_env_int(env.get(ENV_RESUME_HISTORY_EPOCH)),
    )


def authorize_mutation_from_env(
    action_kind: str,
    target_lane: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    explicit_approval: bool = False,
) -> AuthorizationRecord:
    """Fail-closed authority gate driven by the session's env contract.

    Call this immediately before a real side-effecting mutation
    (``send_message`` send, Kanban board mutation, ...). Returns an
    :class:`AuthorizationRecord`; the caller enforces it (``raise_for_status``
    or branch on ``.authorized``).

    * No ``HERMES_CONTROL_PLANE_LANE`` env -> explicit *authorized* record
      ("no authority contract configured"); existing callers are unaffected.
    * Stale compaction / history / context / todo -> blocked + demoted.
    * Cross-lane mutation without an approved route / explicit approval
      -> blocked.
    """
    env = os.environ if env is None else env
    lane = (env.get(ENV_LANE) or "").strip()
    if not lane:
        return AuthorizationRecord(
            authorized=True,
            action=action_kind,
            lane="",
            reason="no authority contract configured",
        )
    lane_contract = lane_contract_from_env(env, lane)
    resume_packet = resume_packet_from_env(env, lane)
    effective_target = (
        target_lane or (env.get(ENV_TARGET_LANE) or "").strip() or lane
    )
    action = MutationAction(kind=action_kind, target_lane=effective_target)
    approval = explicit_approval or _env_bool(env.get(ENV_APPROVAL))
    return authorize_pre_action_mutation(
        action,
        lane_contract,
        resume_packet,
        origin_contract=None,
        explicit_approval=approval,
    )
