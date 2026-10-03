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
@pytest.mark.parametrize("user_name", [None, "Same display name"])
@pytest.mark.parametrize("thread_id", [None, "omt_shared"])
async def test_feishu_shared_session_attributes_each_current_sender(user_name, thread_id):
    runner = _make_runner(GatewayConfig(group_sessions_per_user=False))
    history = []
    sender_ids = ["ou_first_member", "ou_second_member"]
    for sender_id in sender_ids:
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_team",
            chat_type="group",
            user_id=sender_id,
            user_name=user_name,
            thread_id=thread_id,
        )
        event = MessageEvent(text="show my tasks", source=source)
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=history,
        )
        assert f"Feishu sender_id={sender_id}" in result
        assert result.endswith("show my tasks")
        assert all(other not in result for other in sender_ids if other != sender_id)
        history.extend(
            [
                {"role": "user", "content": result},
                {"role": "assistant", "content": "Acknowledged."},
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_type,thread_id",
    [("dm", None), ("group", None), ("group", "omt_private")],
)
async def test_feishu_sender_prefix_does_not_change_individual_sessions(chat_type, thread_id):
    runner = _make_runner(
        GatewayConfig(group_sessions_per_user=True, thread_sessions_per_user=True)
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_individual",
        chat_type=chat_type,
        user_id="ou_member",
        user_name=None,
        thread_id=thread_id,
    )
    event = MessageEvent(text="show my tasks", source=source)
    result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    assert result == event.text
