"""Mnemoria — cognitive memory system plugin for hermes-agent.

Wraps the mnemoria package as a pluggable memory provider for hermes-agent.

Config env vars:
    HERMES_MEMORY_MNEMORIA_ENABLED  (bool)  Enable this provider
    HERMES_MEMORY_MNEMORIA_MODE      (str)   Profile: balanced (default)
    HERMES_MNEMORIA_DB               (path)  SQLite db path (~/.hermes/mnemoria.db)

MEMORY_SPEC notation supported:
    C[t]: Constraint   D[t]: Decision   V[t]: Value
    ?[t]: Unknown       \u2713[t]: Done   ~[t]: Obsolete
    (t = target label, e.g. C[auth]: must validate JWT)
"""

from .provider import MnemoriaMemoryProvider

__all__ = ["MnemoriaMemoryProvider"]
