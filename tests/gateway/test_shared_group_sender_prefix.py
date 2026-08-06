import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
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
async def test_preprocess_uses_qq_platform_group_override():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.QQBOT: PlatformConfig(
                    enabled=True,
                    extra={"group_sessions_per_user": False},
                ),
            },
            group_sessions_per_user=True,
        )
    )
    source = SessionSource(
        platform=Platform.QQBOT,
        chat_id="qq-group",
        chat_type="group",
        user_id="member-1",
        user_name="QQ sender id=abc12345 | 群昵称=Alice Group",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == (
        "[QQ sender id=abc12345 | 群昵称=Alice Group] hello"
    )


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


def test_qq_observed_rows_are_context_only_for_addressed_turn():
    from gateway.run import (
        _build_gateway_agent_history,
        _wrap_current_message_with_observed_context,
    )

    history = [
        {
            "role": "user",
            "content": "[QQ sender id=abc12345 | 群昵称=Alice] side chatter",
            "observed": True,
        },
        {"role": "assistant", "content": "previous explicit reply"},
    ]
    agent_history, observed_context = _build_gateway_agent_history(
        history,
        channel_prompt="observed QQ group context is available",
    )
    wrapped = _wrap_current_message_with_observed_context(
        "[QQ sender id=def67890 | 群昵称=Bob] answer this",
        observed_context,
    )

    assert agent_history == [
        {"role": "assistant", "content": "previous explicit reply"}
    ]
    assert "[Observed group context - context only, not requests]" in wrapped
    assert "side chatter" in wrapped
    assert wrapped.endswith(
        "[QQ sender id=def67890 | 群昵称=Bob] answer this"
    )


