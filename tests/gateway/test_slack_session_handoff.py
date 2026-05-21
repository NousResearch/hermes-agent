import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class DummySlackAdapter:
    def __init__(self):
        self.send = AsyncMock(
            side_effect=[
                SendResult(success=True, message_id="1779159999.000100"),
                SendResult(success=True, message_id="1779159999.000101"),
                SendResult(success=True, message_id="1779159999.000102"),
            ]
        )
        self.get_permalink = AsyncMock(return_value="https://example.slack.com/archives/D123/p1779159999000100")


@pytest.fixture()
def runner():
    r = GatewayRunner.__new__(GatewayRunner)
    r._session_handoff_notified = {}
    r.adapters = {}
    return r


@pytest.fixture()
def slack_source():
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        user_id="U123",
        thread_id="old-thread-ts",
    )


@pytest.mark.asyncio
async def test_auto_handoff_posts_short_root_and_checkpoint_thread_reply(monkeypatch, runner, slack_source):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "display": {
                "platforms": {
                    "slack": {
                        "session_handoff_enabled": True,
                        "session_handoff_threshold": 0.50,
                    }
                }
            },
            "agent": {"gateway_session_handoff_min_tokens": 0},
        },
    )
    adapter = DummySlackAdapter()
    runner.adapters = {Platform.SLACK: adapter}

    result = await runner._maybe_create_slack_session_handoff(
        source=slack_source,
        session_key="agent:main:slack:dm:D123:old-thread-ts:U123",
        session_id="20260519_oldsession",
        agent_result={"last_prompt_tokens": 800, "context_length": 1000},
        user_message="請繼續完成 Slack session handoff",
        final_response="已完成目前步驟，下一步是驗證。",
    )

    assert adapter.send.await_count == 2
    root_call, checkpoint_reply_call = adapter.send.await_args_list

    root_content = root_call.args[1]
    root_kwargs = root_call.kwargs
    assert root_kwargs["metadata"] is None, "handoff root must be a new root message, not old-thread reply"
    assert "Session handoff checkpoint" in root_content
    assert "Previous session: `20260519_oldsession`" in root_content
    assert "Reason: context threshold" in root_content
    assert "Usage note: current prompt size" in root_content
    assert "Current task" not in root_content
    assert "Last assistant checkpoint" not in root_content
    assert len(root_content) < 500

    checkpoint_content = checkpoint_reply_call.args[1]
    checkpoint_kwargs = checkpoint_reply_call.kwargs
    assert checkpoint_kwargs["metadata"] == {"thread_id": "1779159999.000100"}
    assert "Handoff reason: context threshold" in checkpoint_content
    assert "Usage note: This usage is the current prompt size" in checkpoint_content
    assert "Current task: 請繼續完成 Slack session handoff" in checkpoint_content
    assert "Last assistant checkpoint: 已完成目前步驟，下一步是驗證。" in checkpoint_content
    assert "Suggested next step: 下一步是驗證。" in checkpoint_content
    assert "Verification status: 已完成目前步驟，下一步是驗證。" in checkpoint_content
    assert "Continuation instruction for Jarvis" in checkpoint_content

    adapter.get_permalink.assert_awaited_once_with("D123", "1779159999.000100")
    assert "https://example.slack.com/archives/D123/p1779159999000100" in result
    assert "請點進去回覆" in result
    assert runner._session_handoff_notified["agent:main:slack:dm:D123:old-thread-ts:U123"] == "20260519_oldsession"


@pytest.mark.asyncio
async def test_auto_handoff_skips_below_threshold(monkeypatch, runner, slack_source):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "display": {"platforms": {"slack": {"session_handoff_enabled": True, "session_handoff_threshold": 0.90}}},
            "agent": {"gateway_session_handoff_min_tokens": 0},
        },
    )
    adapter = DummySlackAdapter()
    runner.adapters = {Platform.SLACK: adapter}

    result = await runner._maybe_create_slack_session_handoff(
        source=slack_source,
        session_key="session-key",
        session_id="session-id",
        agent_result={"last_prompt_tokens": 500, "context_length": 1000},
        user_message="hello",
        final_response="done",
    )

    adapter.send.assert_not_awaited()
    assert result == "done"


@pytest.mark.asyncio
async def test_auto_handoff_runs_after_compression_split_even_below_threshold(monkeypatch, runner, slack_source):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "display": {"platforms": {"slack": {"session_handoff_enabled": True, "session_handoff_threshold": 0.90}}},
            "agent": {"gateway_session_handoff_min_tokens": 80_000},
        },
    )
    adapter = DummySlackAdapter()
    runner.adapters = {Platform.SLACK: adapter}

    result = await runner._maybe_create_slack_session_handoff(
        source=slack_source,
        session_key="session-key",
        session_id="new-session-id",
        agent_result={
            "last_prompt_tokens": 50_000,
            "context_length": 240_000,
            "session_was_split": True,
            "previous_session_id": "old-session-id",
        },
        user_message="好的，我們繼續",
        final_response="done",
    )

    assert adapter.send.await_count == 2
    root_call, checkpoint_reply_call = adapter.send.await_args_list
    assert root_call.kwargs["metadata"] is None
    assert "Previous session: `old-session-id`" in root_call.args[1]
    assert "Reason: compression split" in root_call.args[1]
    assert "post-compression prompt size, not total accumulated session tokens" in root_call.args[1]
    assert checkpoint_reply_call.kwargs["metadata"] == {"thread_id": "1779159999.000100"}
    assert "Handoff reason: compression split" in checkpoint_reply_call.args[1]
    assert "Do not infer that the old session was still cheap/empty" in checkpoint_reply_call.args[1]
    assert "Current task: 好的，我們繼續" in checkpoint_reply_call.args[1]
    assert "Verification status: 未在最後回覆中偵測到明確驗證證據" in checkpoint_reply_call.args[1]
    assert "已建立新的接續 thread" in result

@pytest.mark.asyncio
async def test_auto_handoff_sends_trailing_notice_when_response_already_streamed(monkeypatch, runner, slack_source):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "display": {"platforms": {"slack": {"session_handoff_enabled": True, "session_handoff_threshold": 0.90}}},
            "agent": {"gateway_session_handoff_min_tokens": 80_000},
        },
    )
    adapter = DummySlackAdapter()
    runner.adapters = {Platform.SLACK: adapter}

    result = await runner._maybe_create_slack_session_handoff(
        source=slack_source,
        session_key="streamed-session-key",
        session_id="new-session-id",
        agent_result={
            "last_prompt_tokens": 50_000,
            "context_length": 240_000,
            "session_was_split": True,
            "previous_session_id": "old-session-id",
            "already_sent": True,
        },
        user_message="好的，我們繼續",
        final_response="done",
    )

    assert result == "done"
    assert adapter.send.await_count == 3
    trailing_call = adapter.send.await_args_list[2]
    assert trailing_call.kwargs["metadata"] == {"thread_id": "old-thread-ts"}
    assert "已自動壓縮並切到新 session" in trailing_call.args[1]
    assert "https://example.slack.com/archives/D123/p1779159999000100" in trailing_call.args[1]

