"""Capability declarations for benchmark backends.

These capabilities describe which benchmark behaviors a backend can support
honestly. They are used to skip unsupported categories instead of counting
them as failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Capability profile for a benchmark backend."""

    universal_store_recall: bool = True
    time_simulation: bool = False
    access_rehearsal: bool = False
    consolidation: bool = False
    scopes: bool = False
    typed_facts: bool = False
    supersession: bool = False
    reward_learning: bool = False
    exploration: bool = False
    turn_sync: bool = False
    precompress_hook: bool = False
    session_end_hook: bool = False
    delegation_hook: bool = False

    def to_dict(self) -> dict:
        return asdict(self)