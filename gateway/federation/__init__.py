"""Hermes P2P Federation — multi-device coordination platform.

Provides real-time device discovery, task relay, and cross-device collaboration
without external infrastructure. All devices are peers — no primary/secondary.

Architecture:
    - ``federation_protocol.py``: Message types, serialization, auth
    - ``federation_discovery.py``: Peer discovery (manual config + mDNS)
    - ``federation_connection.py``: WebSocket connection management
    - ``federation_adapter.py``: PlatformAdapter integrating with GatewayRunner
    - ``federation_consensus.py``: Raft-lite consensus for atomic task claiming
    - ``federation_relay.py``: Task execution with checkpoint/relay support
    - ``federation_heartbeat.py``: Unified heartbeat (shared_db + lan modes)

Modes:
    - ``shared_db``: File-synced SQLite (v1, 2-3 devices, ~60s latency)
    - ``lan``: WebSocket + manual peer config (v2, N devices, <1s latency)
    - ``auto``: mDNS discovery + WebSocket (future)
"""

from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
    PeerInfo,
)
from gateway.federation.federation_consensus import FederationConsensus
from gateway.federation.federation_relay import TaskExecutorRelay, TaskCheckpoint, TaskExecutionState
from gateway.federation.federation_discovery import FederationMDNS, DiscoveredPeer

__all__ = [
    "FedMessage",
    "MessageType",
    "PeerInfo",
    "FederationConsensus",
    "TaskExecutorRelay",
    "TaskCheckpoint",
    "TaskExecutionState",
    "FederationMDNS",
    "DiscoveredPeer",
]
