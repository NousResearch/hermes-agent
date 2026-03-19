"""
CNS Core — Backward Compatibility Shim
========================================
Re-exports the canonical classes from their real source modules so that
any code importing from `cns_core` continues to work without changes.

Original V4.0 classes are now maintained in:
    - synaptic_field.py  → EventType, CNSEvent, SynapticField, SwarmThought
    - cosmos_cns.py      → CosmosCNS (full 12D physics engine)

Author: Cory Shane Davis / Cosmos CNS
Version: 4.1 (Shim — delegates to canonical modules)
"""

# ── Re-export from synaptic_field (the single source of truth) ──
from .synaptic_field import (
    EventType,
    CNSEvent,
    SynapticField,
    SwarmThought,
)

# ── Re-export from cosmos_cns (the full 12D physics CNS) ──
from .cosmos_cns import CosmosCNS, get_cns

__all__ = [
    "EventType",
    "CNSEvent",
    "SynapticField",
    "SwarmThought",
    "CosmosCNS",
    "get_cns",
]
