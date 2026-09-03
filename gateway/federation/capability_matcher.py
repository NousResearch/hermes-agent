"""Phase 18: Capability Matching — bridges PeerCapability ↔ ComputeCapability.

Integrates:
  - FederationComputePool (existing compute capability registry)
  - RelayDecisionEngine (Phase 17 relay evaluation)
  - PeerCapability / TaskDescriptor (from relay_decision.py)

Usage:
    matcher = CapabilityMatcher(compute_pool=federation_compute_pool)
    candidates = matcher.get_candidates_for_task(task_descriptor)
    decision = matcher.evaluate_relay(task_descriptor, candidate_peers)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from gateway.federation.federation_compute_pool import FederationComputePool, ComputeCapability
from gateway.federation.relay_decision import (
    PeerCapability,
    TaskDescriptor,
    RelayDecision,
    RelayDecisionEngine,
    DecisionThresholds,
    make_peer_capability,
)
from gateway.federation.trust import TrustLevel


def compute_capability_to_peer_capability(
    cap: ComputeCapability,
    trust: TrustLevel = TrustLevel.UNKNOWN,
    specialties: Optional[set] = None,
    latency_ms: Optional[float] = None,
) -> PeerCapability:
    """Convert a ComputeCapability into a PeerCapability for relay evaluation.

    ComputeCapability has: device_id, cpu_cores, memory_gb, gpu_type, load_avg, is_available.
    PeerCapability needs: all those + trust, specialties, latency_ms, last_seen.
    """
    return PeerCapability(
        node_id=cap.device_id,
        hostname=getattr(cap, "hostname", cap.device_id),
        cpu_cores=cap.cpu_cores,
        memory_gb=cap.memory_gb,
        load_avg=cap.load_avg,
        trust=trust,
        specialties=specialties or set(),
        latency_ms=latency_ms,
        last_seen=getattr(cap, "last_seen", time.time()),
    )


class CapabilityMatcher:
    """Bridges FederationComputePool and RelayDecisionEngine.

    Given a TaskDescriptor, queries the compute pool for all peers,
    converts to PeerCapabilities, and evaluates relay decisions.
    """

    def __init__(
        self,
        compute_pool: FederationComputePool,
        decision_engine: Optional[RelayDecisionEngine] = None,
        thresholds: Optional[DecisionThresholds] = None,
    ) -> None:
        self._pool = compute_pool
        self._engine = decision_engine or RelayDecisionEngine(
            thresholds=thresholds,
        )

    def get_all_peer_capabilities(
        self,
        trust_registry: Optional[Dict[str, TrustLevel]] = None,
        latency_registry: Optional[Dict[str, float]] = None,
    ) -> List[PeerCapability]:
        """Get all known peers as PeerCapabilities.

        Args:
            trust_registry: {node_id: TrustLevel} — if not provided,
                all peers default to TrustLevel.VERIFIED.
            latency_registry: {node_id: latency_ms} — if not provided,
                latency defaults to None (unknown).
        """
        trust = trust_registry or {}
        latency = latency_registry or {}
        capabilities = self._pool.get_all_capabilities()

        peers: List[PeerCapability] = []
        for node_id, cap in capabilities.items():
            if not cap.is_available:
                continue
            peers.append(
                compute_capability_to_peer_capability(
                    cap,
                    trust=trust.get(node_id, TrustLevel.VERIFIED),
                    specialties=getattr(cap, "specialties", set()),
                    latency_ms=latency.get(node_id),
                )
            )
        return peers

    def get_best_peer_for_task(
        self,
        task: TaskDescriptor,
        trust_registry: Optional[Dict[str, TrustLevel]] = None,
        latency_registry: Optional[Dict[str, float]] = None,
        top_n: int = 3,
    ) -> List[PeerCapability]:
        """Return the top-N best peers for a task, sorted by confidence."""
        all_peers = self.get_all_peer_capabilities(
            trust_registry=trust_registry,
            latency_registry=latency_registry,
        )
        if not all_peers:
            return []

        # Score all peers
        scored = []
        for peer in all_peers:
            score = self._engine._score_peer(task, peer)
            scored.append((score.total, peer))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_n]]

    def evaluate_relay(
        self,
        task: TaskDescriptor,
        trust_registry: Optional[Dict[str, TrustLevel]] = None,
        latency_registry: Optional[Dict[str, float]] = None,
    ) -> RelayDecision:
        """Evaluate relay decision for a task across all available peers.

        This is the main entry point from the relay handler.
        """
        peers = self.get_all_peer_capabilities(
            trust_registry=trust_registry,
            latency_registry=latency_registry,
        )
        return self._engine.evaluate_relay_sync(task, peers)

    async def evaluate_relay_async(
        self,
        task: TaskDescriptor,
        trust_registry: Optional[Dict[str, TrustLevel]] = None,
        latency_registry: Optional[Dict[str, float]] = None,
    ) -> RelayDecision:
        """Async version — queries peer capabilities in parallel if needed."""
        peers = self.get_all_peer_capabilities(
            trust_registry=trust_registry,
            latency_registry=latency_registry,
        )
        return await self._engine.evaluate_relay(task, peers)

    def get_task_routing_report(
        self,
        task: TaskDescriptor,
        trust_registry: Optional[Dict[str, TrustLevel]] = None,
        latency_registry: Optional[Dict[str, float]] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Get a detailed routing report for the task.

        Useful for the REVIEW mode report and for dashboard display.
        """
        peers = self.get_all_peer_capabilities(
            trust_registry=trust_registry,
            latency_registry=latency_registry,
        )

        scored = []
        for peer in peers:
            score = self._engine._score_peer(task, peer)
            outcome = self._engine._decide_outcome(task, score, peer)
            scored.append({
                "peer": {
                    "node_id": peer.node_id,
                    "hostname": peer.hostname,
                    "cpu_cores": peer.cpu_cores,
                    "memory_gb": peer.memory_gb,
                    "trust": peer.trust.value,
                    "specialties": sorted(peer.specialties),
                    "is_overloaded": peer.is_overloaded,
                    "is_healthy": peer.is_healthy,
                    "latency_ms": peer.latency_ms,
                },
                "score": {
                    "capability_match": round(score.capability_match, 3),
                    "trust_match": round(score.trust_match, 3),
                    "network_conditions": round(score.network_conditions, 3),
                    "task_sensitivity_fit": round(score.task_sensitivity_fit, 3),
                    "total": round(score.total, 3),
                },
                "outcome": outcome.value,
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"]["total"], reverse=True)

        decision = self.evaluate_relay(task, trust_registry, latency_registry)

        return {
            "task_id": task.task_id,
            "task_description": task.description,
            "task_sensitivity": task.sensitivity.value,
            "decision": decision.to_audit_dict(),
            "candidate_peers": scored[:top_n],
            "total_peers_available": len(scored),
        }


def register_default_capability_handlers(pool: FederationComputePool) -> None:
    """Register standard compute handlers on the pool.

    These are the built-in task type handlers that enable capability-based
    routing. Extend this to add custom handlers.
    """
    # Security-critical tasks — only high-trust, high-resource peers
    pool.register_handler("security.scan", lambda state: _security_scan(state))
    pool.register_handler("code.search", lambda state: _code_search(state))
    pool.register_handler("data.analysis", lambda state: _data_analysis(state))
    pool.register_handler("deployment.execute", lambda state: _deployment_execute(state))


def _security_scan(state: Any) -> Dict[str, Any]:
    """Handler for security.scan tasks."""
    return {"type": "security_scan", "handler": "default"}


def _code_search(state: Any) -> Dict[str, Any]:
    """Handler for code.search tasks."""
    return {"type": "code_search", "handler": "default"}


def _data_analysis(state: Any) -> Dict[str, Any]:
    """Handler for data.analysis tasks."""
    return {"type": "data_analysis", "handler": "default"}


def _deployment_execute(state: Any) -> Dict[str, Any]:
    """Handler for deployment.execute tasks."""
    return {"type": "deployment", "handler": "default"}


__all__ = [
    "compute_capability_to_peer_capability",
    "CapabilityMatcher",
    "register_default_capability_handlers",
]
