"""Mnemoria — cognitive memory plugin for hermes-agent.

Wraps the mnemoria PyPI package as a pluggable MemoryProvider.

Config env vars:
    HERMES_MEMORY_MNEMORIA_ENABLED  (bool)  Enable this provider
    HERMES_MEMORY_MNEMORIA_MODE      (str)   Profile: balanced (default)
    HERMES_MNEMORIA_DB               (path)  SQLite db path (~/.hermes/mnemoria.db)

MEMORY_SPEC notation:
    C[t]: Constraint   D[t]: Decision   V[t]: Value
    ?[t]: Unknown       ✓[t]: Done   ~[t]: Obsolete
"""

from .provider import MnemoriaMemoryProvider

__all__ = ["MnemoriaMemoryProvider"]
