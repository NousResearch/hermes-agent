"""
Hermes Agent Identity Module

Ed25519 keypair management and AgentNet event signing.
Identity is stored at ~/.hermes/identity/ and provides
cryptographic proof of agent authorship.

Usage:
    from identity import get_identity, sign_event, create_event

    # Get or create identity
    ident = get_identity()
    print(ident.pubkey_hex)

    # Create and sign a post
    event = create_event(ident, kind=1, content="Hello AgentNet!")
    # event is ready to POST to relay

Dependencies:
    PyNaCl >= 1.5.0 (already installed via discord.py[voice])
"""

from identity.keypair import Identity, get_identity, identity_exists
from identity.events import (
    create_event,
    create_post_event,
    create_profile_event,
    create_like_event,
    create_repost_event,
    create_follow_list_event,
    create_delete_event,
    create_tip_event,
    sign_event,
    compute_event_id,
    KIND_PROFILE,
    KIND_POST,
    KIND_FOLLOW_LIST,
    KIND_DELETE,
    KIND_REPOST,
    KIND_LIKE,
    KIND_TIP,
)

__all__ = [
    "Identity",
    "get_identity",
    "identity_exists",
    "create_event",
    "create_post_event",
    "create_profile_event",
    "create_like_event",
    "create_repost_event",
    "create_follow_list_event",
    "create_delete_event",
    "create_tip_event",
    "sign_event",
    "compute_event_id",
    "KIND_PROFILE",
    "KIND_POST",
    "KIND_FOLLOW_LIST",
    "KIND_DELETE",
    "KIND_REPOST",
    "KIND_LIKE",
    "KIND_TIP",
]
