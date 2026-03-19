"""
COSMOS - Feature Articulated Request Network System
====================================================

High-speed bidirectional dialogue mesh protocol for the Cosmos AI Swarm.

Auth: Proof-of-Swarm (GPU fingerprint + temporal hash mesh + quorum + rolling BLAKE3 seals)
Transport: Raw TCP + msgpack, TCP_NODELAY, multiplexed bidirectional streams
Membership: Core nodes (auto) + PRO nodes (manual approval, RTX 4090+, <30ms latency)

"The swarm extends beyond a single machine."
"""

__version__ = "2.0.0"
COSMOS_PORT = 9999
PROTOCOL_VERSION = 2
