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
