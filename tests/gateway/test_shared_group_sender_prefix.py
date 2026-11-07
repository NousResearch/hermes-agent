import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_includes_slack_author_mention_for_shared_thread():
    """Shared Slack threads expose the current author's verifiable user ID
    next to the display name so 'mention me again' requests can bind the
    mention to the CURRENT speaker (#17916)."""
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_name="team-channel",
        chat_type="group",
        user_id="U123",
        user_name="Alice",
        thread_id="171.000",
    )
    event = MessageEvent(text="mention me again", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice | Slack user <@U123>] mention me again"


@pytest.mark.asyncio
async def test_preprocess_includes_google_chat_sender_email_for_shared_space():
    """Shared Google Chat spaces show the sender's verified email next to the display name, so two
    people called 'Alice' (or a renamed account) stay distinguishable to the agent."""
    gchat = Platform("google_chat")
    runner = _make_runner(GatewayConfig(platforms={gchat: PlatformConfig(enabled=True)}))
    source = SessionSource(
        platform=gchat,
        chat_id="spaces/AAA",
        chat_name="team-space",
        chat_type="group",
        user_id="alice@example.com",
        user_name="Alice",
        user_id_alt="users/123",
        thread_id="spaces/AAA/threads/T1",
    )
    event = MessageEvent(text="what did I ask yesterday?", source=source)

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result == "[Alice | alice@example.com] what did I ask yesterday?"


@pytest.mark.asyncio
async def test_preprocess_google_chat_without_email_keeps_display_name_only():
    gchat = Platform("google_chat")
    runner = _make_runner(GatewayConfig(platforms={gchat: PlatformConfig(enabled=True)}))
    source = SessionSource(platform=gchat, chat_id="spaces/AAA", chat_name="team-space", chat_type="group",
                           user_id="users/123", user_name="Alice", thread_id="spaces/AAA/threads/T1")
    event = MessageEvent(text="hi", source=source)

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result == "[Alice] hi"
