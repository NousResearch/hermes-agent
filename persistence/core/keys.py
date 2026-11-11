"""The profile a gateway session key names.

A gateway session key is ``agent:<ns>:...``; the ``<ns>`` slot is the profile namespace, ``main`` for the default profile and the marked ``main~`` for a profile literally named ``main``.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["profile_from_session_key_namespace", "session_key_profile"]


def profile_from_session_key_namespace(namespace: str) -> str:
    """Inverse of ``gateway.session._session_key_namespace`` for the ``<ns>`` slot of a key:
    ``"default"`` for ``main``, ``"main"`` for the marked ``main~``, else the slot is the profile id."""
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def session_key_profile(session_key: Any) -> Optional[str]:
    """Profile encoded in an ``agent:<ns>:…`` gateway key (``default`` for ``main``), or None for a
    keyless row (CLI/subagent lineage) or a key another producer minted (hosted rooms, tests)."""
    if not isinstance(session_key, str):
        return None
    parts = session_key.split(":")
    if len(parts) < 3 or parts[0] != "agent" or not parts[1]:
        return None
    return profile_from_session_key_namespace(parts[1])
