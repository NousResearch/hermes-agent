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
async def test_sender_prefix_leads_media_notes_in_shared_group():
    """In a shared group session the sender tag must come before the image note, so the
    model knows who sent the image and not only who wrote the caption."""
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))

    async def fake_enrich(source, session_key, text, paths):
        return f"[The user sent an image]\n\n{text}"

    runner._enrich_inbound_images = fake_enrich
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="120363000000000000@g.us",
        chat_type="group",
        user_id="15550001111@s.whatsapp.net",
        user_name="Bob",
    )
    event = MessageEvent(
        text="look at this", source=source,
        media_urls=["/tmp/photo.jpg"], media_types=["image/jpeg"],
    )

    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert result == "[Bob] [The user sent an image]\n\nlook at this"
