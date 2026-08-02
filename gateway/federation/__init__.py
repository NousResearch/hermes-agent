"""Hermes P2P Federation — multi-device coordination platform.

Provides real-time device discovery, task relay, and cross-device collaboration
without external infrastructure. All devices are peers — no primary/secondary.

Architecture:
    - ``federation_protocol.py``: Message types, serialization, auth
    - ``federation_discovery.py``: Peer discovery (manual config + mDNS)
    - ``federation_connection.py``: WebSocket connection management
    - ``federation_adapter.py``: PlatformAdapter integrating with GatewayRunner
    - ``federation_heartbeat.py``: Legacy shared-db mode (preserved for backward compat)

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
