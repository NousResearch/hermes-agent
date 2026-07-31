"""Hermes-owned credential store and transaction artifact classification.

The predicates in this module intentionally match the concrete filenames
created by Hermes' credential writers and restore path.  They are narrow
enough for destructive cleanup: an unrelated file beside a credential store
must never be removed merely because it has a generic ``.tmp`` suffix.
"""

from __future__ import annotations


PRIMARY_AUTH_FILENAMES: tuple[str, ...] = (
    "auth.json",
    ".anthropic_oauth.json",
)


def is_primary_auth_transient(name: str) -> bool:
    """Return whether *name* is Hermes primary/Anthropic transaction residue."""
    return (
        name
        in {
            "auth.lock",
            ".anthropic_oauth.lock",
            "auth.json.corrupt",
            ".anthropic_oauth.json.corrupt",
            # Legacy fixed temp used by docker rebootstrap before unique
            # auth.json.tmp.<pid>.<random> transactions were adopted.
            "auth.json.rebootstrap.tmp",
        }
        or name.startswith("auth.json.tmp.")
        or name.startswith("auth.json.corrupt.")
        or name.startswith(".anthropic_oauth.json.tmp.")
        or name.startswith(".anthropic_oauth.json.corrupt.")
        # Legacy Hermes Anthropic writer spelling.
        or name.startswith(".anthropic_oauth.tmp.")
        # backup._atomic_replace_credential() restore spellings.
        or (name.startswith(".auth.json.") and name.endswith(".tmp"))
        or (
            name.startswith("..anthropic_oauth.json.")
            and name.endswith(".tmp")
        )
        # utils.atomic_json_write(.anthropic_oauth.json, ...).
        or (name.startswith("..anthropic_oauth_") and name.endswith(".tmp"))
    )


def is_primary_auth_artifact(name: str) -> bool:
    """Return whether *name* is a durable store or its transaction residue."""
    return name in PRIMARY_AUTH_FILENAMES or is_primary_auth_transient(name)


def is_shared_auth_transient(name: str) -> bool:
    """Return whether *name* is Hermes shared Nous transaction residue."""
    return (
        name in {"nous_auth.lock", "nous_auth.json.corrupt"}
        or name.startswith("nous_auth.json.tmp.")
        or name.startswith("nous_auth.json.corrupt.")
        # backup._atomic_replace_credential() restore spelling.
        or (name.startswith(".nous_auth.json.") and name.endswith(".tmp"))
    )


def is_shared_auth_artifact(name: str) -> bool:
    """Return whether *name* is the shared Nous store or its residue."""
    return name == "nous_auth.json" or is_shared_auth_transient(name)
