"""Per-turn Slack egress identity (R3-5, connector PR gateway-gateway#210).

The relay connector fills chat.startStream's recipient_user_id /
recipient_team_id (required for channel streams) from metadata.user_id /
metadata.scope_id. Before this fix the gateway only stamped slack_team_id
per-turn; user_id (and scope_id) were left for the relay adapter's
_with_scope fallback, which reads per-chat caches keyed only by chat_id —
mutable state that a CONCURRENT turn overwrites. Two users with
overlapping turns in one channel could open U1's stream with U2 as the
recipient.

These tests pin the fix: _thread_metadata_for_source stamps the authentic
per-turn identity from the turn's OWN source, and the cache fallback in
RelayAdapter._with_scope no longer decides for turns that carry it.
"""

from types import SimpleNamespace

from gateway.platforms.base import Platform
from gateway.run import GatewayRunner


def _slack_source(user_id: str, chat_id: str = "C_SHARED") -> SimpleNamespace:
    return SimpleNamespace(
        platform=Platform.SLACK,
        chat_id=chat_id,
        thread_id="1700.100",
        chat_type="channel",
        message_id="1700.100",
        scope_id="T_TEAM",
        user_id=user_id,
    )


def test_thread_metadata_stamps_per_turn_user_and_scope():
    runner = object.__new__(GatewayRunner)
    meta = runner._thread_metadata_for_source(_slack_source("U_ALICE"))
    assert meta is not None
    assert meta["user_id"] == "U_ALICE"
    assert meta["scope_id"] == "T_TEAM"
    assert meta["slack_team_id"] == "T_TEAM"


def test_concurrent_turns_carry_their_own_identity():
    """The R3-5 interleaving: U2's turn starting must not change what U1's
    already-built metadata carries — identity is per-turn source data, not
    shared per-chat state."""
    runner = object.__new__(GatewayRunner)
    meta_alice = runner._thread_metadata_for_source(_slack_source("U_ALICE"))
    # U2's turn begins in the SAME channel before U1's stream opens.
    meta_bob = runner._thread_metadata_for_source(_slack_source("U_BOB"))
    assert meta_alice is not None and meta_bob is not None
    assert meta_alice["user_id"] == "U_ALICE"
    assert meta_bob["user_id"] == "U_BOB"


def test_relay_with_scope_does_not_override_per_turn_identity():
    """_with_scope must be fill-only: a turn that already carries user_id /
    scope_id keeps them even when the per-chat cache holds another user
    (the overwrite that caused the R3-5 cross-recipient hazard)."""
    from gateway.relay.adapter import RelayAdapter

    adapter = object.__new__(RelayAdapter)
    # The mutable cache holds the OTHER user (their turn arrived later).
    adapter._scope_by_chat = {"C_SHARED": "T_CACHED"}
    adapter._dm_user_by_chat = {"C_SHARED": "U_BOB"}
    merged = adapter._with_scope(
        "C_SHARED", {"user_id": "U_ALICE", "scope_id": "T_TEAM"}
    )
    assert merged["user_id"] == "U_ALICE"
    assert merged["scope_id"] == "T_TEAM"


def test_relay_with_scope_still_fills_when_turn_carries_nothing():
    """Restart/synthetic sends (no per-turn identity) keep the cache
    fallback — the fix narrows the cache's authority, not its existence."""
    from gateway.relay.adapter import RelayAdapter

    adapter = object.__new__(RelayAdapter)
    adapter._scope_by_chat = {"C_SHARED": "T_CACHED"}
    adapter._dm_user_by_chat = {"C_SHARED": "U_CACHED"}
    merged = adapter._with_scope("C_SHARED", {"thread_id": "1700.100"})
    assert merged["user_id"] == "U_CACHED"
    assert merged["scope_id"] == "T_CACHED"
