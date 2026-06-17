from types import SimpleNamespace

from gateway.config import HomeChannel, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context_prompt, SessionContext


def _runner(home_channel=None):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        get_home_channel=lambda platform: (
            home_channel if home_channel and platform == home_channel.platform else None
        )
    )
    return runner


def test_gateway_skips_global_user_profile_for_non_owner_source():
    runner = _runner(
        HomeChannel(platform=Platform.TELEGRAM, chat_id="owner-chat", name="Home")
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="contact-chat",
        chat_type="dm",
        user_id="contact-user",
        user_name="Jorge",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:telegram:dm:contact-chat",
        user_config={},
    ) is True


def test_gateway_keeps_global_user_profile_for_home_channel_source():
    runner = _runner(
        HomeChannel(platform=Platform.TELEGRAM, chat_id="owner-chat", name="Home")
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="owner-chat",
        chat_type="dm",
        user_id="owner-user",
        user_name="Aldo",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:telegram:dm:owner-chat",
        user_config={},
    ) is False


def test_gateway_keeps_global_user_profile_for_config_home_channel_source():
    runner = _runner()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="owner-channel",
        chat_type="dm",
        user_id="owner-user",
        user_name="Aldo",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:discord:dm:owner-channel",
        user_config={
            "platforms": {
                "discord": {
                    "home_channel": {
                        "chat_id": "owner-channel",
                    }
                }
            }
        },
    ) is False


def test_gateway_keeps_global_user_profile_for_local_source():
    runner = _runner()
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="local",
        chat_type="dm",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:local:dm",
        user_config={},
    ) is False


def _write_lid_mapping(tmp_home, lid_digits, phone_jid):
    """Create the bridge lid-mapping file the canonicalizer reads."""
    mapping_dir = tmp_home / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    import json
    (mapping_dir / f"lid-mapping-{lid_digits}.json").write_text(
        json.dumps(phone_jid), encoding="utf-8"
    )


def test_gateway_keeps_user_profile_for_owner_lid_in_whatsapp_group(tmp_path, monkeypatch):
    """Regression: the owner speaking from inside a WhatsApp GROUP arrives as a
    LID, not a phone number. The owner gate must canonicalize the LID back to
    the phone-JID home channel and recognize the owner — otherwise USER.md is
    hidden and the agent treats Aldo as a stranger in his own group.
    """
    tmp_home = tmp_path / "hermes-home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    _write_lid_mapping(tmp_home, "96370627199010", "5215514706713@s.whatsapp.net")
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215514706713@s.whatsapp.net",
            name="Home",
        )
    )
    # Owner posting in a group: user_id is the opaque LID, chat_id is the group.
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215514706713-1503324008@g.us",
        chat_type="group",
        user_id="96370627199010@lid",
        user_name="Aldo",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215514706713-1503324008@g.us",
        user_config={},
    ) is False


def test_gateway_skips_user_profile_for_non_owner_lid_in_whatsapp_group(tmp_path, monkeypatch):
    """A different group member (non-owner) whose LID maps to their OWN phone
    must still have USER.md hidden — the canonicalization must not leak owner
    status to other members, even though the group JID shares the owner's
    phone-number prefix.
    """
    tmp_home = tmp_path / "hermes-home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    # Owner mapping + a different member mapping to a different phone.
    _write_lid_mapping(tmp_home, "96370627199010", "5215514706713@s.whatsapp.net")
    _write_lid_mapping(tmp_home, "74831349469423", "5215620337474@s.whatsapp.net")
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215514706713@s.whatsapp.net",
            name="Home",
        )
    )
    # Necro (non-owner) posting in the OWNER's group.
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215514706713-1503324008@g.us",
        chat_type="group",
        user_id="74831349469423@lid",
        user_name="Necro",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215514706713-1503324008@g.us",
        user_config={},
    ) is True


def test_owner_gate_partial_mapping_only_reverse_file(tmp_path, monkeypatch):
    """Robustness: even when only the *reverse* lid-mapping file exists (the
    bridge wrote one direction), expanding the full alias set on both sides
    must still match the owner. This is why the gate uses
    expand_whatsapp_aliases (full transitive set) rather than a single
    canonical form — a partial mapping shouldn't lock the owner out.
    """
    tmp_home = tmp_path / "hermes-home"
    (tmp_home / "whatsapp" / "session").mkdir(parents=True, exist_ok=True)
    import json
    # Only the reverse mapping (lid -> phone) is present.
    (tmp_home / "whatsapp" / "session" / "lid-mapping-96370627199010_reverse.json").write_text(
        json.dumps("5215514706713@s.whatsapp.net"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215514706713@s.whatsapp.net",
            name="Home",
        )
    )
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215514706713-1503324008@g.us",
        chat_type="group",
        user_id="96370627199010@lid",
        user_name="Aldo",
    )

    assert runner._should_skip_user_profile_for_source(
        source=source,
        session_key="agent:main:whatsapp:group:5215514706713-1503324008@g.us",
        user_config={},
    ) is False


def test_owner_gate_security_cascade_protects_approve_path(tmp_path, monkeypatch):
    """SECURITY CONTRACT: the /approve owner gate (slash_commands.py) and the
    non-owner secret-redaction backstop BOTH key off this exact function.
    This test pins the contract that a non-owner group member is reported as
    'skip owner profile = True', which is what makes /approve fail-closed for
    them — while the owner (even via group LID) gets False and can approve.
    If this function ever regresses, dangerous-command authorization regresses
    with it, so the assertion is intentionally explicit.
    """
    tmp_home = tmp_path / "hermes-home"
    (tmp_home / "whatsapp" / "session").mkdir(parents=True, exist_ok=True)
    import json
    (tmp_home / "whatsapp" / "session" / "lid-mapping-96370627199010_reverse.json").write_text(
        json.dumps("5215514706713@s.whatsapp.net"), encoding="utf-8"
    )
    (tmp_home / "whatsapp" / "session" / "lid-mapping-74831349469423_reverse.json").write_text(
        json.dumps("5215620337474@s.whatsapp.net"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))

    runner = _runner(
        HomeChannel(
            platform=Platform.WHATSAPP,
            chat_id="5215514706713@s.whatsapp.net",
            name="Home",
        )
    )
    group_key = "agent:main:whatsapp:group:5215514706713-1503324008@g.us"

    owner = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215514706713-1503324008@g.us",
        chat_type="group",
        user_id="96370627199010@lid",
        user_name="Aldo",
    )
    non_owner = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="5215514706713-1503324008@g.us",
        chat_type="group",
        user_id="74831349469423@lid",
        user_name="Necro",
    )

    # Owner-in-group: recognized → may approve dangerous commands.
    assert runner._should_skip_user_profile_for_source(
        source=owner, session_key=group_key, user_config={},
    ) is False
    # Non-owner-in-group: gated → /approve fails closed for them.
    assert runner._should_skip_user_profile_for_source(
        source=non_owner, session_key=group_key, user_config={},
    ) is True


def test_session_context_marks_current_source_as_authoritative_identity():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="contact-chat",
            chat_type="dm",
            user_id="contact-user",
            user_name="Jorge",
        ),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
    )

    prompt = build_session_context_prompt(context)

    assert "Current Session Context above is authoritative" in prompt
    assert "Do not address the current speaker as the owner" in prompt
