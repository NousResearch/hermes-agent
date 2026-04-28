import json

from gateway.agent_actor import (
    build_state_packet,
    detect_public_broadcast_stop_directive,
    evaluate_send_message_policy,
    infer_platform_authority,
    maybe_record_directive_from_inbound,
    owner_user_ids_for_platform,
    record_inbound_event,
)
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB


def test_detects_public_broadcast_stop_directive():
    directive = detect_public_broadcast_stop_directive("Is this a cron? Turn this off")

    assert directive is not None
    assert directive["behavior"] == "suppress"


def test_state_packet_is_sender_scoped(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="149",
        chat_type="group",
        user_id="141",
        user_name="hands",
    )
    event_id, person_id = record_inbound_event(
        db,
        source=source,
        session_id="sid",
        session_key="agent:main:discord:group:149:141",
        text="hello",
        authority="trusted",
    )

    packet = build_state_packet(
        db,
        source=source,
        session_id="sid",
        session_key="agent:main:discord:group:149:141",
        inbound_event_id=event_id,
        person_id=person_id,
        authority="trusted",
    )

    assert "person_id: discord:141" in packet
    assert "authority: trusted" in packet
    assert "session_key: agent:main:discord:group:149:141" in packet
    db.close()


def test_owner_authority_uses_generated_soul_owner_block(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "SOUL.md").write_text(
        "## Your owner\n"
        "- Discord: `@handsdiff` (user_id `1417636184355766305`)\n"
        "\n## Peer roster\n"
        "- Someone else user_id `999999999999999999`\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("DISCORD_OWNER_USER_ID", raising=False)
    monkeypatch.delenv("DISCORD_OWNER_USER_IDS", raising=False)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1495468809216327702",
        chat_type="group",
        user_id="1417636184355766305",
        user_name="hands",
    )

    assert owner_user_ids_for_platform("discord") == {"1417636184355766305"}
    assert infer_platform_authority(source) == "owner"


def test_owner_state_packet_includes_recent_cross_session_events(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    group_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="general",
        chat_type="group",
        user_id="141",
        user_name="hands",
    )
    record_inbound_event(
        db,
        source=group_source,
        session_id="group-sid",
        session_key="agent:main:discord:group:general:141",
        text="hey from general",
        authority="owner",
    )
    dm_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm",
        chat_type="dm",
        user_id="141",
        user_name="hands",
    )
    dm_event_id, person_id = record_inbound_event(
        db,
        source=dm_source,
        session_id="dm-sid",
        session_key="agent:main:discord:dm:dm",
        text="do you see general?",
        authority="owner",
    )

    packet = build_state_packet(
        db,
        source=dm_source,
        session_id="dm-sid",
        session_key="agent:main:discord:dm:dm",
        inbound_event_id=dm_event_id,
        person_id=person_id,
        authority="owner",
    )

    assert "authority: owner" in packet
    assert "Recent Runtime Events" in packet
    assert "hey from general" in packet
    db.close()


def test_user_state_packet_surfaces_cross_session_outbounds_to_same_audience(tmp_path):
    """A non-trusted sender's state packet must reveal the agent's own
    cross-session attempts (delivered or blocked) targeting the current
    audience. Without this, the agent confabulates "must be a different
    session" when challenged about messages it actually authored from
    autonomous/cron sessions to the same chat.
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    # A blocked rebroadcast attempt to discord:general from a hub-DM session,
    # before the user's inbound. Different session_key, different person_id —
    # the existing per-session and per-person filters would both miss it.
    db.append_agent_event(
        event_type="outbound",
        event_subtype="send_message",
        status="blocked",
        session_key="agent:main:hub:dm:hub:sal",
        actor_id="main",
        actor_kind="tool",
        source="hub",
        platform="discord",
        platform_chat_id="general",
        tool_name="send_message",
        content="autonomous hub rebroadcast attempt",
        payload={"decision": "deny"},
    )

    user_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="general",
        chat_type="group",
        user_id="999",
        user_name="adam",
    )
    inbound_event_id, person_id = record_inbound_event(
        db,
        source=user_source,
        session_id="group-sid",
        session_key="agent:main:discord:group:general:999",
        text="why am I getting pinged about this stuff?",
        authority="user",
    )

    packet = build_state_packet(
        db,
        source=user_source,
        session_id="group-sid",
        session_key="agent:main:discord:group:general:999",
        inbound_event_id=inbound_event_id,
        person_id=person_id,
        authority="user",
    )

    assert "authority: user" in packet
    assert "autonomous hub rebroadcast attempt" in packet, (
        "user-authority state packet must include outbound activity targeting "
        "the same platform_chat_id, even from other sessions"
    )
    assert "blocked" in packet
    db.close()


def test_directive_blocks_public_cross_session_send(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="general",
        chat_type="group",
        user_id="141",
        user_name="hands",
    )
    event_id, person_id = record_inbound_event(
        db,
        source=source,
        session_id="sid",
        session_key="agent:main:discord:group:general:141",
        text="Is this a cron? Turn this off",
        authority="trusted",
    )
    maybe_record_directive_from_inbound(
        db,
        source=source,
        session_id="sid",
        session_key="agent:main:discord:group:general:141",
        inbound_event_id=event_id,
        person_id=person_id,
        text="Is this a cron? Turn this off",
        authority="trusted",
    )

    tokens = set_session_vars(
        platform="hub",
        chat_id="hub:speculator",
        chat_type="dm",
        user_id="speculator",
        user_name="speculator",
        session_key="agent:main:hub:dm:hub:speculator",
        agent_event_id=event_id,
        person_id="hub:speculator",
    )
    try:
        decision = evaluate_send_message_policy(
            target_platform="discord",
            target_chat_id="general",
            message="Market digest",
            db=db,
        )
    finally:
        clear_session_vars(tokens)
        db.close()

    assert decision.allowed is False
    assert decision.policy == "suppress_public_broadcasts"


def test_cron_like_hub_inbound_blocks_public_rebroadcast(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    event_id = db.append_agent_event(
        event_type="inbound",
        event_subtype="message",
        status="received",
        session_key="agent:main:hub:dm:hub:speculator",
        source="hub",
        platform="hub",
        platform_chat_id="hub:speculator",
        content="Cronjob Response: synthetic market digest",
    )
    tokens = set_session_vars(
        platform="hub",
        chat_id="hub:speculator",
        chat_type="dm",
        user_id="speculator",
        user_name="speculator",
        session_key="agent:main:hub:dm:hub:speculator",
        agent_event_id=event_id,
        person_id="hub:speculator",
    )
    try:
        decision = evaluate_send_message_policy(
            target_platform="discord",
            target_chat_id="1495468809216327702",
            message="Synthetic market digest",
            db=db,
        )
    finally:
        clear_session_vars(tokens)
        db.close()

    assert decision.allowed is False
    assert decision.policy == "autonomous_public_rebroadcast_guard"
