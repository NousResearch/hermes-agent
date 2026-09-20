import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_does_not_add_slack_sender_framing_to_shared_thread():
    """Slack sender identity is carried structurally, not embedded in prompt text."""
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

    assert result == "mention me again"
    assert "Alice" not in result
    assert "<@U123>" not in result


@pytest.mark.asyncio
async def test_shared_sender_identity_is_not_added_to_model_or_persistence_text():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group",
        user_id="U1", user_name="Alice",
    )
    event = MessageEvent(text="hello everyone", source=source)

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    model_text, persisted_text, _ = runner._hmwa_apply_message_timestamp(event, model_text)

    assert model_text == "hello everyone"
    assert persisted_text == "hello everyone"


@pytest.mark.asyncio
async def test_isolated_session_has_no_sender_prefix_and_keeps_semantic_text():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="dm-1", chat_type="dm",
        user_id="U1", user_name="Alice",
    )
    event = MessageEvent(text="hello privately", source=source)

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    model_text, persisted_text, _ = runner._hmwa_apply_message_timestamp(event, model_text)

    assert model_text == "hello privately"
    assert persisted_text == "hello privately"


@pytest.mark.asyncio
async def test_user_authored_leading_brackets_are_not_treated_as_sender_prefix():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group",
        user_id="U1", user_name="Alice",
    )
    event = MessageEvent(text="[example-user] this bracketed text is mine", source=source)

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    model_text, persisted_text, _ = runner._hmwa_apply_message_timestamp(event, model_text)

    assert model_text == "[example-user] this bracketed text is mine"
    assert persisted_text == "[example-user] this bracketed text is mine"


@pytest.mark.asyncio
async def test_persistence_keeps_inbound_media_preparation_beside_clean_sender_text():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    runner._prepend_inbound_media_file_notes = (
        lambda message_text, _audio_paths, _video_paths: f"[media preparation]\n\n{message_text}"
    )
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group",
        user_id="U1", user_name="Alice",
    )
    event = MessageEvent(text="process the attachment", source=source)

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    _, persisted_text, _ = runner._hmwa_apply_message_timestamp(event, model_text)

    assert model_text == "[media preparation]\n\nprocess the attachment"
    assert persisted_text == "[media preparation]\n\nprocess the attachment"


@pytest.mark.asyncio
async def test_sender_prefix_collision_in_enrichment_keeps_both_semantic_parts():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    runner._prepend_inbound_media_file_notes = (
        lambda message_text, _audio_paths, _video_paths: f"[Alice] enrichment\n\n{message_text}"
    )
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group",
        user_id="U1", user_name="Alice",
    )
    event = MessageEvent(text="keep this request", source=source)

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[],
    )
    _, persisted_text, _ = runner._hmwa_apply_message_timestamp(event, model_text)

    assert model_text == "[Alice] enrichment\n\nkeep this request"
    assert persisted_text == "[Alice] enrichment\n\nkeep this request"


def test_gateway_forwards_source_identity_as_per_turn_author():
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    runner._consume_pending_native_image_paths = lambda _session_key: []
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="group",
        user_id="123456", user_name="Alice",
    )

    class _CapturingAgent:
        def run_conversation(self, _message, **kwargs):
            self.kwargs = kwargs
            return {"final_response": "ok"}

    agent = _CapturingAgent()
    author = runner._gateway_turn_author(source)
    context = TurnContext(
        source=source, message="hello", session_id="sid", session_key="key",
        turn_author=author,
    )
    TurnRunner(runner, context)._run_conversation_with_approval(
        agent, [], None, "hello", None,
    )

    assert agent.kwargs["turn_author"] == {
        "id": "123456", "name": "Alice", "is_bot": False, "platform": "discord",
    }
