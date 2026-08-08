"""CRITICAL-3: 3-failure death rule.

Avoids false-positive owner death on network blips. A peer is only
considered dead after 3 consecutive failures (probe timeouts).

Pillar 5: Resilience (SECURITY-BASELINE.md)

Used by:
- ClusterHealthMonitor._probe_all_peers()
- ClusterRelayCoordinator._check_owner_dead()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional


DEFAULT_DEAD_THRESHOLD = 3  # 3 consecutive failures


@dataclass
class PeerDeathStatus:
    """Per-peer death detection state."""
    node_id: str
    failure_count: int = 0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    is_dead: bool = False
    dead_at: Optional[float] = None

    def record_success(self) -> bool:
        """Record a successful probe. Returns True if peer was previously dead."""
        was_dead = self.is_dead
        self.failure_count = 0
        self.last_success_at = time.time()
        self.is_dead = False
        self.dead_at = None
        return was_dead  # so caller can alert on revival

    def record_failure(self, threshold: int = DEFAULT_DEAD_THRESHOLD) -> bool:
        """Record a probe failure. Returns True if peer just died this turn."""
        self.failure_count += 1
        self.last_failure_at = time.time()
        if self.failure_count >= threshold and not self.is_dead:
            self.is_dead = True
            self.dead_at = time.time()
            return True
        return False


class DeathDetector:
    """Cluster-wide death detector with per-peer state.

    Thread-safe (uses internal Lock). Coordinates with ClusterHealthMonitor
    to track consecutive failures and flag confirmed-death only after N
    consecutive failures.
    """

    def __init__(self, dead_threshold: int = DEFAULT_DEAD_THRESHOLD):
        """Args:
            dead_threshold: number of consecutive failures before death confirmed.
                            Default 3 (CRITICAL-3 baseline).
        """
        if dead_threshold < 1:
            raise ValueError("dead_threshold must be >= 1")
        self._threshold = dead_threshold
        self._peers: Dict[str, PeerDeathStatus] = {}
        self._lock = Lock()

    @property
    def threshold(self) -> int:
        return self._threshold

    def record_success(self, node_id: str) -> bool:
        """Returns True if peer was previously dead (revival)."""
        with self._lock:
            if node_id not in self._peers:
                self._peers[node_id] = PeerDeathStatus(node_id=node_id)
            return self._peers[node_id].record_success()

    def record_failure(self, node_id: str) -> bool:
        """Returns True if peer just died this turn."""
        with self._lock:
            if node_id not in self._peers:
                self._peers[node_id] = PeerDeathStatus(node_id=node_id)
            return self._peers[node_id].record_failure(self._threshold)

    def is_dead(self, node_id: str) -> bool:
        with self._lock:
            peer = self._peers.get(node_id)
            return peer.is_dead if peer else False

    def get_status(self, node_id: str) -> Optional[PeerDeathStatus]:
        with self._lock:
            return self._peers.get(node_id)

    def all_dead(self) -> list[str]:
        """Return list of dead peer IDs."""
        with self._lock:
            return [p.node_id for p in self._peers.values() if p.is_dead]

    def clear(self, node_id: str) -> None:
        """Forget a peer entirely (e.g., on revoke)."""
        with self._lock:
            self._peers.pop(node_id, None)


__all__ = ["DeathDetector", "PeerDeathStatus", "DEFAULT_DEAD_THRESHOLD"]
