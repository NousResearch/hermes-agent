"""Phase 17: Relay Decision Engine — AI-powered approval routing.

Layer 4 of the federation architecture: makes relay decisions based on
task sensitivity, peer trust, capability match, and network conditions.

Approval Modes:
  ASK     — Always ask the user before relay (most conservative)
  AUTO    — AI decides if confidence >= threshold, otherwise asks
  REVIEW  — Relay immediately, then present detailed report to user

Decision Flow:
  1. Gather: task sensitivity, peer capability, trust level, network
  2. Score: compute confidence (0.0–1.0)
  3. Decide: based on mode + thresholds

Decision Outcomes:
  AUTO_APPROVE  — confidence >= auto_threshold → proceed automatically
  REVIEW        — confidence >= review_threshold → relay, then report
  ASK           — confidence < ask_threshold → ask user
  DENY          — trust/capability mismatch → abort relay
  ABORT         — task too sensitive for this peer → abort

Integration:
  from gateway.federation.relay_decision import RelayDecisionEngine

  engine = RelayDecisionEngine(config=federation_config)
  decision = await engine.evaluate_relay(
      task_id="t-123",
      task_description="Analyze security report",
      task_sensitivity=TaskSensitivity.HIGH,
      peer={"node_id": "mac-b", "trust": "trusted", "capabilities": {...}},
  )
  # decision.outcome in {AUTO_APPROVE, REVIEW, ASK, DENY, ABORT}
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from gateway.federation.trust import (
    TrustLevel,
    TaskSensitivity,
    infer_sensitivity,
)


# === Approval modes ===

class ApprovalMode(str, enum.Enum):
    """User-configurable federation approval strategy.

    Set per-task or globally in FederationConfig.ask_mode.
    """
    ASK = "ask"       # Always ask before relay
    AUTO = "auto"     # AI decides if confidence >= threshold
    REVIEW = "review" # Relay immediately, detailed report after

    @classmethod
    def from_str(cls, s: str) -> "ApprovalMode":
        try:
            return cls(s.lower())
        except ValueError:
            return cls.ASK  # fail-closed


# === Decision outcomes ===

class DecisionOutcome(str, enum.Enum):
    """Result of relay decision evaluation."""
    # Proceed
    AUTO_APPROVE = "auto_approve"   # confidence >= threshold, proceed
    REVIEW = "review"                # relay now, report later
    ASK = "ask"                      # need user input
    # Reject
    DENY = "deny"                    # capability/trust mismatch
    ABORT = "abort"                  # task too sensitive for peer


# === Confidence scoring ===

@dataclass
class ConfidenceBreakdown:
    """Why the confidence score is what it is."""
    capability_match: float = 0.0   # 0–0.35
    trust_match: float = 0.0        # 0–0.30
    network_conditions: float = 0.0 # 0–0.20
    task_sensitivity_fit: float = 0.0 # 0–0.15
    total: float = 0.0


@dataclass
class RelayDecision:
    """Output of the relay decision engine."""
    task_id: str
    outcome: DecisionOutcome
    confidence: float            # 0.0–1.0
    breakdown: ConfidenceBreakdown
    chosen_peer: Optional[str]   # node_id of selected peer
    reason: str                  # human-readable explanation
    suggested_delay_s: float = 0.0  # how long before retry
    ask_message: str = ""        # what to ask the user (when ASK)
    created_at: float = field(default_factory=time.time)

    @property
    def can_relay(self) -> bool:
        return self.outcome in (
            DecisionOutcome.AUTO_APPROVE,
            DecisionOutcome.REVIEW,
            DecisionOutcome.ASK,  # user will decide
        )

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "confidence": round(self.confidence, 3),
            "chosen_peer": self.chosen_peer,
            "reason": self.reason,
        }


# === Thresholds ===

@dataclass
class DecisionThresholds:
    """Confidence thresholds for each approval mode.

    Defaults tuned for a typical Hermes cluster (3–5 devices,
    all on the same LAN or via Tailscale).
    """
    auto_approve: float = 0.70   # >= 0.70 → AUTO_APPROVE
    review: float = 0.30          # >= 0.30 → REVIEW (< auto_approve)
    ask: float = 0.0              # < ask_threshold → ASK

    @classmethod
    def strict(cls) -> "DecisionThresholds":
        """Stricter thresholds for sensitive environments."""
        return cls(auto_approve=0.85, review=0.50)

    @classmethod
    def permissive(cls) -> "DecisionThresholds":
        """Permissive thresholds for trusted clusters."""
        return cls(auto_approve=0.50, review=0.20)


# === Peer capability descriptor ===

@dataclass
class PeerCapability:
    """What a peer can do."""
    node_id: str
    hostname: str
    cpu_cores: int = 0
    memory_gb: float = 0.0
    load_avg: float = 0.0
    trust: TrustLevel = TrustLevel.UNKNOWN
    # Task categories this peer is good at
    specialties: Set[str] = field(default_factory=set)
    # Latency to this peer (ms); None = unknown
    latency_ms: Optional[float] = None
    # Last time this peer was responsive (unix ts)
    last_seen: float = 0.0

    @property
    def is_overloaded(self) -> bool:
        """True if load_avg > number of CPU cores (rough heuristic)."""
        return self.load_avg > self.cpu_cores * 0.8

    @property
    def is_healthy(self) -> bool:
        """Peer is responding and not overloaded."""
        if self.load_avg > 4.0:  # absolute overload threshold
            return False
        if self.last_seen > 0 and time.time() - self.last_seen > 120:
            return False  # stale
        return True


# === Task descriptor ===

@dataclass
class TaskDescriptor:
    """What a task needs for relay evaluation."""
    task_id: str
    description: str              # raw task description for sensitivity inference
    sensitivity: TaskSensitivity  # explicit sensitivity (or inferred)
    required_capabilities: Set[str] = field(default_factory=set)
    required_cpu_cores: int = 1
    required_memory_gb: float = 0.5
    max_latency_ms: Optional[float] = None  # task can't tolerate >N ms latency
    user_preference: ApprovalMode = ApprovalMode.AUTO


# === Relay Decision Engine ===

class RelayDecisionEngine:
    """AI-powered relay decision engine.

    Usage:
        engine = RelayDecisionEngine(config=federation_config)
        decision = await engine.evaluate_relay(task, candidate_peers)

    The engine is async so it can query peer capabilities in parallel
    if needed (future: HTTP health probes).
    """

    def __init__(
        self,
        thresholds: Optional[DecisionThresholds] = None,
        default_mode: ApprovalMode = ApprovalMode.AUTO,
    ) -> None:
        self._thresholds = thresholds or DecisionThresholds()
        self._default_mode = default_mode

    # --- Public API ---

    async def evaluate_relay(
        self,
        task: TaskDescriptor,
        candidate_peers: List[PeerCapability],
    ) -> RelayDecision:
        """Evaluate relay for a task across candidate peers.

        Returns the best decision for the most capable available peer.
        """
        if not candidate_peers:
            return self._deny(
                task.task_id,
                None,
                "No candidate peers available in the cluster.",
            )

        # Score each peer
        scored = []
        for peer in candidate_peers:
            score = self._score_peer(task, peer)
            scored.append((score, peer))

        # Sort by confidence descending
        scored.sort(key=lambda x: x[0].total, reverse=True)
        best_score, best_peer = scored[0]

        # Determine outcome
        outcome = self._decide_outcome(task, best_score, best_peer)

        if outcome == DecisionOutcome.DENY:
            reason = (
                f"Peer '{best_peer.node_id}' lacks capability or trust "
                f"for this task (confidence={best_score.total:.2f})."
            )
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=None,
                reason=reason,
            )

        elif outcome == DecisionOutcome.ASK:
            ask_msg = self._build_ask_message(task, best_peer, best_score)
            reason = (
                f"Confidence {best_score.total:.0%} below auto threshold "
                f"{self._thresholds.auto_approve:.0%}; asking user."
            )
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=best_peer.node_id,
                reason=reason,
                ask_message=ask_msg,
            )

        else:
            reason = self._build_approve_reason(task, best_peer, best_score)
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=best_peer.node_id,
                reason=reason,
            )

    def evaluate_relay_sync(
        self,
        task: TaskDescriptor,
        candidate_peers: List[PeerCapability],
    ) -> RelayDecision:
        """Synchronous version of evaluate_relay for non-async contexts."""
        if not candidate_peers:
            return self._deny(
                task.task_id, None, "No candidate peers available."
            )

        scored = []
        for peer in candidate_peers:
            score = self._score_peer(task, peer)
            scored.append((score, peer))

        scored.sort(key=lambda x: x[0].total, reverse=True)
        best_score, best_peer = scored[0]
        outcome = self._decide_outcome(task, best_score, best_peer)

        if outcome == DecisionOutcome.DENY:
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=None,
                reason=f"Peer '{best_peer.node_id}' lacks required capability or trust.",
            )

        elif outcome == DecisionOutcome.ASK:
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=best_peer.node_id,
                reason=f"Asking user (confidence {best_score.total:.0%} < auto {self._thresholds.auto_approve:.0%}).",
                ask_message=self._build_ask_message(task, best_peer, best_score),
            )

        else:
            return RelayDecision(
                task_id=task.task_id,
                outcome=outcome,
                confidence=best_score.total,
                breakdown=best_score,
                chosen_peer=best_peer.node_id,
                reason=self._build_approve_reason(task, best_peer, best_score),
            )

    # --- Scoring ---

    def _score_peer(
        self, task: TaskDescriptor, peer: PeerCapability
    ) -> ConfidenceBreakdown:
        cap = self._score_capability(task, peer)
        trust = self._score_trust(task, peer)
        network = self._score_network(task, peer)
        sensitivity = self._score_sensitivity_fit(task, peer)
        total = cap + trust + network + sensitivity
        return ConfidenceBreakdown(
            capability_match=cap,
            trust_match=trust,
            network_conditions=network,
            task_sensitivity_fit=sensitivity,
            total=min(total, 1.0),
        )

    def _score_capability(
        self, task: TaskDescriptor, peer: PeerCapability
    ) -> float:
        """0–0.35: does the peer have enough resources?"""
        score = 0.0

        # CPU: 0–0.15
        if peer.cpu_cores >= task.required_cpu_cores * 2:
            score += 0.15
        elif peer.cpu_cores >= task.required_cpu_cores:
            score += 0.10
        elif peer.cpu_cores >= task.required_cpu_cores * 0.5:
            score += 0.05
        # else 0.0

        # Memory: 0–0.10
        if peer.memory_gb >= task.required_memory_gb * 2:
            score += 0.10
        elif peer.memory_gb >= task.required_memory_gb:
            score += 0.07
        elif peer.memory_gb >= task.required_memory_gb * 0.5:
            score += 0.03
        # else 0.0

        # Specialties match: 0–0.10
        if task.required_capabilities:
            overlap = task.required_capabilities & peer.specialties
            score += 0.10 * (len(overlap) / len(task.required_capabilities))

        return min(score, 0.35)

    def _score_trust(
        self, task: TaskDescriptor, peer: PeerCapability
    ) -> float:
        """0–0.30: is this peer trusted enough for the task?"""
        from gateway.federation.trust import TrustPolicy
        policy = TrustPolicy()
        if policy.can_claim(peer.trust, task.sensitivity):
            # How much margin? Scale by trust rank distance
            gap = peer.trust.rank - task.sensitivity.min_trust.rank
            if gap >= 2:
                return 0.30
            elif gap == 1:
                return 0.25
            else:
                return 0.20  # exactly meets minimum
        return 0.0

    def _score_network(
        self, task: TaskDescriptor, peer: PeerCapability
    ) -> float:
        """0–0.20: is the network good enough?"""
        score = 0.0

        # Latency: 0–0.12
        if task.max_latency_ms and peer.latency_ms is not None:
            if peer.latency_ms <= task.max_latency_ms * 0.3:
                score += 0.12
            elif peer.latency_ms <= task.max_latency_ms * 0.7:
                score += 0.08
            elif peer.latency_ms <= task.max_latency_ms:
                score += 0.04
            # else 0.0 (exceeds task tolerance)

        # Peer health: 0–0.08
        if not peer.is_healthy:
            return 0.0
        if not peer.is_overloaded:
            score += 0.08

        return min(score, 0.20)

    def _score_sensitivity_fit(
        self, task: TaskDescriptor, peer: PeerCapability
    ) -> float:
        """0–0.15: does task sensitivity match peer trust?"""
        from gateway.federation.trust import TrustPolicy
        policy = TrustPolicy()
        if policy.can_claim(peer.trust, task.sensitivity):
            if task.sensitivity == TaskSensitivity.CRITICAL:
                return 0.08  # margin is thin
            return 0.15  # comfortable margin
        return 0.0

    # --- Decision logic ---

    def _decide_outcome(
        self,
        task: TaskDescriptor,
        score: ConfidenceBreakdown,
        peer: PeerCapability,
    ) -> DecisionOutcome:
        mode = task.user_preference

        # HARD SECURITY GATE: always deny if trust cannot cover sensitivity.
        # This is the primary defense; don't let thresholds override it.
        from gateway.federation.trust import TrustPolicy
        policy = TrustPolicy()
        if not policy.can_claim(peer.trust, task.sensitivity):
            return DecisionOutcome.DENY

        if mode == ApprovalMode.ASK:
            return DecisionOutcome.ASK

        if mode == ApprovalMode.REVIEW:
            if score.total >= self._thresholds.review:
                return DecisionOutcome.REVIEW
            return DecisionOutcome.ASK

        # AUTO mode
        if score.total >= self._thresholds.auto_approve:
            return DecisionOutcome.AUTO_APPROVE
        elif score.total >= self._thresholds.review:
            return DecisionOutcome.REVIEW
        elif score.total >= self._thresholds.ask:
            return DecisionOutcome.ASK
        else:
            return DecisionOutcome.DENY

    # --- Helpers ---

    def _deny(
        self, task_id: str, peer: Optional[PeerCapability], reason: str
    ) -> RelayDecision:
        return RelayDecision(
            task_id=task_id,
            outcome=DecisionOutcome.DENY,
            confidence=0.0,
            breakdown=ConfidenceBreakdown(),
            chosen_peer=peer.node_id if peer else None,
            reason=reason,
        )

    def _build_ask_message(
        self,
        task: TaskDescriptor,
        peer: PeerCapability,
        score: ConfidenceBreakdown,
    ) -> str:
        return (
            f"Node '{peer.node_id}' ({peer.hostname}) wants to relay task "
            f"'{task.task_id}' ({task.description[:60]}).\n"
            f"Confidence: {score.total:.0%}\n"
            f"  • Capability match: {score.capability_match:.0%}\n"
            f"  • Trust fit: {score.trust_match:.0%}\n"
            f"  • Network: {score.network_conditions:.0%}\n"
            f"Trust level: {peer.trust.value}\n"
            f"Approve relay? (yes/no)"
        )

    def _build_approve_reason(
        self,
        task: TaskDescriptor,
        peer: PeerCapability,
        score: ConfidenceBreakdown,
    ) -> str:
        return (
            f"Auto-approved relay to '{peer.node_id}' (confidence "
            f"{score.total:.0%}, cap={score.capability_match:.0%}, "
            f"trust={score.trust_match:.0%}, net={score.network_conditions:.0%})."
        )


# === Convenience constructors ===

def make_task_descriptor(
    task_id: str,
    description: str,
    *,
    sensitivity: Optional[TaskSensitivity] = None,
    required_capabilities: Optional[Set[str]] = None,
    required_cpu_cores: int = 1,
    required_memory_gb: float = 0.5,
    approval_mode: ApprovalMode = ApprovalMode.AUTO,
) -> TaskDescriptor:
    """Build a TaskDescriptor from common arguments.

    Sensitivity is inferred from description if not provided.
    """
    sens = sensitivity or infer_sensitivity(description)
    caps = required_capabilities or set()
    return TaskDescriptor(
        task_id=task_id,
        description=description,
        sensitivity=sens,
        required_capabilities=caps,
        required_cpu_cores=required_cpu_cores,
        required_memory_gb=required_memory_gb,
        user_preference=approval_mode,
    )


def make_peer_capability(
    node_id: str,
    hostname: str,
    *,
    cpu_cores: int = 0,
    memory_gb: float = 0.0,
    load_avg: float = 0.0,
    trust: TrustLevel = TrustLevel.UNKNOWN,
    specialties: Optional[Set[str]] = None,
    latency_ms: Optional[float] = None,
    last_seen: float = 0.0,
) -> PeerCapability:
    return PeerCapability(
        node_id=node_id,
        hostname=hostname,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        load_avg=load_avg,
        trust=trust,
        specialties=specialties or set(),
        latency_ms=latency_ms,
        last_seen=last_seen,
    )


__all__ = [
    "ApprovalMode",
    "DecisionOutcome",
    "DecisionThresholds",
    "ConfidenceBreakdown",
    "RelayDecision",
    "PeerCapability",
    "TaskDescriptor",
    "RelayDecisionEngine",
    "make_task_descriptor",
    "make_peer_capability",
]
