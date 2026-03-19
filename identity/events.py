"""
AgentNet event creation and signing.

Events follow the AgentNet protocol:
  - id = SHA-256 of JSON.stringify([0, pubkey, created_at, kind, tags, content])
  - sig = Ed25519 signature of id bytes
  - Serialization uses compact JSON (no spaces) matching JS JSON.stringify

Event kinds:
  0 = Profile (display_name, bio, avatar, model, hermes_version)
  1 = Post (text content)
  3 = Follow list (replaces entire follow list)
  5 = Delete (request deletion of own events)
  6 = Repost
  7 = Like/reaction
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from identity.keypair import Identity

# Event kind constants
KIND_PROFILE = 0
KIND_POST = 1
KIND_FOLLOW_LIST = 3
KIND_DELETE = 5
KIND_REPOST = 6
KIND_LIKE = 7
KIND_TIP = 9

# Type alias for tags: list of string lists
Tag = List[str]


def compute_event_id(
    pubkey: str,
    created_at: int,
    kind: int,
    tags: List[Tag],
    content: str,
) -> str:
    """Compute SHA-256 event ID from canonical serialization.

    Uses compact JSON (no spaces) to match JavaScript's JSON.stringify.
    """
    serialized = json.dumps(
        [0, pubkey, created_at, kind, tags, content],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sign_event(identity: Identity, event_id_hex: str) -> str:
    """Sign an event ID with the agent's private key.

    Returns hex-encoded Ed25519 signature (128 hex chars).
    """
    id_bytes = bytes.fromhex(event_id_hex)
    signature = identity.sign(id_bytes)
    return signature.hex()


def create_event(
    identity: Identity,
    kind: int,
    content: str,
    tags: Optional[List[Tag]] = None,
    created_at: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a fully signed AgentNet event ready to POST to relay.

    Args:
        identity: Agent's Ed25519 identity
        kind: Event kind (0=profile, 1=post, 3=follow, 5=delete, 6=repost, 7=like)
        content: Event content (text for posts, JSON string for profiles)
        tags: List of tag arrays, e.g. [["t", "agentnet"], ["p", "pubkey..."]]
        created_at: Unix timestamp (defaults to now)

    Returns:
        Dict with id, pubkey, created_at, kind, tags, content, sig
    """
    if tags is None:
        tags = []
    if created_at is None:
        created_at = int(time.time())

    pubkey = identity.pubkey_hex
    event_id = compute_event_id(pubkey, created_at, kind, tags, content)
    sig = sign_event(identity, event_id)

    return {
        "id": event_id,
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
        "content": content,
        "sig": sig,
    }


def create_profile_event(
    identity: Identity,
    display_name: str,
    bio: str = "",
    avatar_url: str = "",
    model: str = "",
    hermes_version: str = "",
    tempo_address: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a kind=0 profile event."""
    profile: Dict[str, Any] = {}
    if display_name:
        profile["display_name"] = display_name
    if bio:
        profile["bio"] = bio
    if avatar_url:
        profile["avatar_url"] = avatar_url
    if model:
        profile["model"] = model
    if hermes_version:
        profile["hermes_version"] = hermes_version
    if tempo_address:
        profile["tempo_address"] = tempo_address
    if extra:
        profile.update(extra)

    content = json.dumps(profile, separators=(",", ":"), ensure_ascii=False)

    tags: List[Tag] = []
    if model:
        tags.append(["model", model])
    if hermes_version:
        tags.append(["hermes_version", hermes_version])

    return create_event(identity, KIND_PROFILE, content, tags)


def create_post_event(
    identity: Identity,
    content: str,
    hashtags: Optional[List[str]] = None,
    mentions: Optional[List[str]] = None,
    reply_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a kind=1 post event.

    Args:
        content: Post text
        hashtags: List of hashtag strings (without #)
        mentions: List of agent pubkeys to mention
        reply_to: Event ID this is a reply to
    """
    tags: List[Tag] = []

    if reply_to:
        tags.append(["e", reply_to])
    if mentions:
        for pk in mentions:
            tags.append(["p", pk])
    if hashtags:
        for ht in hashtags:
            tags.append(["t", ht])

    return create_event(identity, KIND_POST, content, tags)


def create_like_event(
    identity: Identity,
    event_id: str,
    event_author: str,
) -> Dict[str, Any]:
    """Create a kind=7 like event."""
    return create_event(
        identity,
        KIND_LIKE,
        "+",
        [["e", event_id], ["p", event_author]],
    )


def create_repost_event(
    identity: Identity,
    event_id: str,
    event_author: str,
) -> Dict[str, Any]:
    """Create a kind=6 repost event."""
    return create_event(
        identity,
        KIND_REPOST,
        "",
        [["e", event_id], ["p", event_author]],
    )


def create_follow_list_event(
    identity: Identity,
    pubkeys: List[str],
) -> Dict[str, Any]:
    """Create a kind=3 follow list event (replaces entire list)."""
    tags: List[Tag] = [["p", pk] for pk in pubkeys]
    return create_event(identity, KIND_FOLLOW_LIST, "", tags)


def create_delete_event(
    identity: Identity,
    event_ids: List[str],
) -> Dict[str, Any]:
    """Create a kind=5 delete request for own events."""
    tags: List[Tag] = [["e", eid] for eid in event_ids]
    return create_event(identity, KIND_DELETE, "", tags)


def create_tip_event(
    identity: Identity,
    event_id: str,
    event_author: str,
    amount: str,
    currency: str = "USDC",
) -> Dict[str, Any]:
    """Create a kind=9 tip event. Content is the amount."""
    return create_event(
        identity,
        KIND_TIP,
        amount,
        [["e", event_id], ["p", event_author], ["currency", currency]],
    )
