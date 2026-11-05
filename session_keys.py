"""The profile namespace of a gateway session key, for the gateway and the storage backend."""

from persistence.core.keys import profile_from_session_key_namespace

__all__ = ["profile_from_session_key", "profile_from_session_key_namespace"]


def profile_from_session_key(session_key):
    """Profile namespace encoded in an ``agent:<ns>:...`` gateway session key, else None."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    return profile_from_session_key_namespace(parts[1] or "main")
