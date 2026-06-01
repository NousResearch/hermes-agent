from gateway.config import Platform
from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
)


def _slack_context() -> SessionContext:
    return SessionContext(
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_name="vucar-staging-aws",
            chat_type="channel",
            user_id="U456",
            user_name="Jake",
            thread_id="1777892506.923189",
            message_id="1777892507.000000",
        ),
        connected_platforms=[Platform.SLACK],
        home_channels={},
    )


def test_slack_prompt_lists_context_tools_when_loaded(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr("gateway.session._slack_context_tools_loaded", lambda: True)

    prompt = build_session_context_prompt(_slack_context())

    assert "Slack context tools" in prompt
    assert "slack_get_thread" in prompt
    assert "slack_get_messages" in prompt
    assert "Channel ID for tool calls: `C123`" in prompt
    assert "Thread timestamp for `slack_get_thread`: `1777892506.923189`" in prompt
    assert "You do NOT have access to Slack-specific APIs" not in prompt


def test_slack_prompt_keeps_disclaimer_when_context_tools_missing(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_TOKEN", raising=False)
    monkeypatch.setattr("gateway.session._slack_context_tools_loaded", lambda: False)

    prompt = build_session_context_prompt(_slack_context())

    assert "You do NOT have access to Slack-specific APIs" in prompt
    assert "slack_get_thread" not in prompt
