"""Read-only execution identity of the control process, never a remote account."""
import os


def effective_identity_hint() -> str:
    """Snapshot POSIX effective identity without probing sudo or authorizing work.

    Unlike hardware facts, credentials can change during a process lifetime;
    take this cheap snapshot when building a new session's environment block.
    Windows elevation is a different contract and is deliberately not inferred.
    """
    if not hasattr(os, "geteuid"):
        return ""
    uid = os.geteuid()
    label = "root" if uid == 0 else "non-root"
    return (f"Effective user ID: {uid} ({label}). This describes local process identity, "
            "not sudo availability, authorization, or remote privileges. A non-root "
            "identity does not imply that every network tool requires elevation.")
