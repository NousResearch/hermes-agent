"""In-memory routing rekey for `hermes profile rename`.

The routing index lives in ``SessionStore._entries`` and is written back periodically, so a durable
DB rewrite alone is clobbered — the live store must rekey its in-memory copy too. This is why a
renamed profile's old namespace kept resurfacing until the gateway restarted.
"""
from __future__ import annotations


def _make_store(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    store = SessionStore(
        sessions_dir,
        GatewayConfig(sessions_dir=sessions_dir, write_sessions_json=False,
                      multiplex_profiles=True),
    )
    store._ensure_loaded()
    return store


def _entry(session_key, chat_id, profile):
    from gateway.session import SessionEntry, SessionSource, Platform
    from gateway.session_lifecycle import _now
    now = _now()
    return SessionEntry(
        session_key=session_key, session_id=f"sid-{chat_id}",
        platform=Platform.FEISHU, chat_type="dm", created_at=now, updated_at=now,
        origin=SessionSource(platform=Platform.FEISHU, chat_id=chat_id, profile=profile),
    )


def test_rekeys_old_namespace_and_origin_profile(tmp_path):
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:oldname:feishu:dm:chatA"] = _entry(
            "agent:oldname:feishu:dm:chatA", "chatA", "oldname")
        store._entries["agent:keepme:feishu:dm:chatB"] = _entry(
            "agent:keepme:feishu:dm:chatB", "chatB", "keepme")

    moved = store.rekey_profile_routing("oldname", "newname")
    assert moved == 1

    assert "agent:oldname:feishu:dm:chatA" not in store._entries
    new_entry = store._entries["agent:newname:feishu:dm:chatA"]
    assert new_entry.session_key == "agent:newname:feishu:dm:chatA"
    assert new_entry.origin.profile == "newname"
    # Bystander namespace untouched.
    assert store._entries["agent:keepme:feishu:dm:chatB"].origin.profile == "keepme"


def test_does_not_overwrite_existing_new_namespace_key(tmp_path):
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:oldname:feishu:dm:chatA"] = _entry(
            "agent:oldname:feishu:dm:chatA", "chatA", "oldname")
        # A collision on the target key (should not happen in practice) is left alone.
        store._entries["agent:newname:feishu:dm:chatA"] = _entry(
            "agent:newname:feishu:dm:chatA", "chatA", "newname")

    import pytest
    with pytest.raises(ValueError, match="routing collision"):
        store.rekey_profile_routing("oldname", "newname")
    assert "agent:oldname:feishu:dm:chatA" in store._entries
    assert "agent:newname:feishu:dm:chatA" in store._entries


def test_rename_repins_the_receiving_bot_of_live_routes(tmp_path):
    """Regression for #155001: a completion on a route pinned to the renamed bot was held forever as "the
    bot of profile 'oldname' is not connected". The live index is written back over the durable rows,
    and delivery reads a route's bot through ``restore_identity`` on ``entry.origin``, which returns an
    identity already pinned there (by ingress, or by an earlier restore in this process). After the
    rename that reader names the new bot, in the old namespace and on a chat the bot received for
    another profile; a route pinned to another bot keeps it."""
    from types import SimpleNamespace

    from gateway.session_identity import identity_of, restore_identity
    store = _make_store(tmp_path)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), _primary_profile_name="default",
                             _resolve_profile_home_for_source=lambda source: tmp_path / "profiles" / source.profile)

    def receiving_bot(entry):
        restore_identity(entry.origin, runner=runner, transport_profile=entry.transport_profile)
        return identity_of(entry.origin).transport_profile

    chats = {"own": ("oldname", "oldname"), "routed": ("satellite", "oldname"), "other": ("satellite", "otherbot")}
    with store._lock:
        for chat_id, (runtime, bot) in chats.items():
            key = f"agent:{runtime}:feishu:dm:{chat_id}"
            entry = _entry(key, chat_id, runtime)
            entry.transport_profile = bot
            store._entries[key] = entry
    assert {entry.origin.chat_id: receiving_bot(entry) for entry in store._entries.values()} == {
        "own": "oldname", "routed": "oldname", "other": "otherbot"}

    assert store.rekey_profile_routing("oldname", "newname") == 1

    bots = {entry.origin.chat_id: (key, receiving_bot(entry)) for key, entry in store._entries.items()}
    assert bots == {
        "own": ("agent:newname:feishu:dm:own", "newname"),
        "routed": ("agent:satellite:feishu:dm:routed", "newname"),
        "other": ("agent:satellite:feishu:dm:other", "otherbot"),
    }
