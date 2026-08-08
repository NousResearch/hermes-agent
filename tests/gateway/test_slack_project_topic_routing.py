"""Behavior contracts for opt-in Slack project-topic routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.session import build_session_key
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.project_topic_router import (
    TopicRouteDecision,
    classify_project_topic,
)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="«redacted:xox…»")
    config.extra.update({
        "reply_in_thread": True,
        "channel_reply_modes": {"C_PROJECT": "project"},
        "channel_prompts": {"C_PROJECT": "Plan the September Japan trip."},
    })
    result = SlackAdapter(config)
    result._app = MagicMock()
    result._app.client = AsyncMock()
    result._bot_user_id = "U_BOT"
    result._running = True
    result.handle_message = AsyncMock()
    result._resolve_user_name = AsyncMock(return_value="testuser")
    result._resolve_channel_name = AsyncMock(return_value="travel-japan-202609")
    return result


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


def _event(
    text: str,
    *,
    ts: str = "1700000000.000001",
    thread_ts: str | None = None,
):
    payload = {
        "channel": "C_PROJECT",
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        payload["thread_ts"] = thread_ts
    return payload


def _response(route: str, confidence: float):
    content = (
        '{"route":"%s","confidence":%s,"reason":"test"}'
        % (route, confidence)
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.parametrize(
    "text",
    [
        "题外话，这个单独聊：帮我检查服务器磁盘。",
        "开 Thread：验证项目专题分流和处理状态",
        "建一个 thread 单独处理这个问题",
    ],
)
def test_explicit_new_topic_skips_model_call(text):
    def fail_call(**_kwargs):
        raise AssertionError("explicit routing must not call the model")

    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text=text,
        call_fn=fail_call,
    )

    assert decision.use_thread is True
    assert decision.source == "directive"


def test_explicit_keep_channel_skips_model_call():
    def fail_call(**_kwargs):
        raise AssertionError("explicit routing must not call the model")

    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="继续留在当前频道聊酒店。",
        call_fn=fail_call,
    )

    assert decision.use_thread is False
    assert decision.source == "directive"


def test_high_confidence_unrelated_work_opens_thread():
    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="Can you debug my Kubernetes cluster?",
        call_fn=lambda **_kwargs: _response("thread", 0.96),
    )
    assert decision.use_thread is True


def test_high_confidence_bounded_project_subtopic_opens_thread():
    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="第三天晚上在新宿吃什么？比较三家餐厅后帮我定一家。",
        call_fn=lambda **_kwargs: _response("thread", 0.97),
    )
    assert decision.use_thread is True


@pytest.mark.parametrize(
    "text",
    [
        "日本有哪些值得吃的东西？",
        "把整个日本行程预算控制在一万元以内。",
        "东京和大阪大概要各留几天？",
    ],
)
def test_project_wide_or_broad_turns_stay_in_channel(text):
    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text=text,
        call_fn=lambda **_kwargs: _response("channel", 0.97),
    )
    assert decision.use_thread is False


def test_router_prompt_defines_branch_worthy_subtopics():
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _response("thread", 0.97)

    classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="第三天晚餐在新宿选哪家？",
        call_fn=capture,
    )

    system_prompt = captured["messages"][0]["content"]
    assert "bounded subtopic inside the project" in system_prompt
    assert "specific meal, hotel, flight, day" in system_prompt
    assert "simple one-answer questions" in system_prompt
    assert "Do not fragment the channel" in system_prompt


def test_uncertain_topic_drift_fails_safe_to_channel():
    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="What about the connection?",
        min_confidence=0.85,
        call_fn=lambda **_kwargs: _response("thread", 0.60),
    )
    assert decision.use_thread is False
    assert decision.source == "fallback"


def test_router_failure_fails_safe_to_channel():
    def unavailable(**_kwargs):
        raise RuntimeError("offline")

    decision = classify_project_topic(
        channel_name="travel-japan-202609",
        channel_prompt="Plan the Japan trip.",
        text="What about the hotel?",
        call_fn=unavailable,
    )
    assert decision.use_thread is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_thread"),
    [
        (TopicRouteDecision("channel", 0.98), None),
        (TopicRouteDecision("thread", 0.98), "1700000000.000001"),
    ],
)
async def test_adapter_routes_before_session_selection(
    adapter, decision, expected_thread
):
    with patch(
        "plugins.platforms.slack.project_topic_router.classify_project_topic",
        return_value=decision,
    ):
        await adapter._handle_slack_message(
            _event("<@U_BOT> please handle this")
        )

    assert adapter.handle_message.await_count == 1
    message = adapter.handle_message.await_args.args[0]
    assert message.source.thread_id == expected_thread
    assert message.metadata["slack_project_session_fork"] is bool(expected_thread)


@pytest.mark.asyncio
async def test_project_turns_use_shared_or_isolated_session_keys(adapter):
    decisions = [
        TopicRouteDecision("channel", 0.98),
        TopicRouteDecision("channel", 0.98),
        TopicRouteDecision("thread", 0.98),
    ]
    with patch(
        "plugins.platforms.slack.project_topic_router.classify_project_topic",
        side_effect=decisions,
    ):
        await adapter._handle_slack_message(
            _event("<@U_BOT> compare Tokyo hotels", ts="1700000000.000011")
        )
        await adapter._handle_slack_message(
            _event("<@U_BOT> add a laundry filter", ts="1700000000.000012")
        )
        await adapter._handle_slack_message(
            _event("<@U_BOT> debug my server", ts="1700000000.000013")
        )

    sources = [call.args[0].source for call in adapter.handle_message.await_args_list]
    channel_key_1 = build_session_key(sources[0], group_sessions_per_user=False)
    channel_key_2 = build_session_key(sources[1], group_sessions_per_user=False)
    thread_key = build_session_key(sources[2], group_sessions_per_user=False)
    assert channel_key_1 == channel_key_2
    assert thread_key != channel_key_1
    assert sources[2].thread_id == "1700000000.000013"
    messages = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert messages[0].metadata["slack_project_session_fork"] is False
    assert messages[1].metadata["slack_project_session_fork"] is False
    assert messages[2].metadata["slack_project_session_fork"] is True


@pytest.mark.asyncio
async def test_existing_thread_bypasses_topic_router(adapter):
    with patch(
        "plugins.platforms.slack.project_topic_router.classify_project_topic"
    ) as classify:
        await adapter._handle_slack_message(
            _event(
                "<@U_BOT> follow-up",
                ts="1700000000.000002",
                thread_ts="1700000000.000001",
            )
        )

    classify.assert_not_called()
    message = adapter.handle_message.await_args.args[0]
    assert message.source.thread_id == "1700000000.000001"
    assert message.metadata["slack_project_session_fork"] is False


@pytest.mark.asyncio
async def test_unaddressed_message_does_not_pay_for_topic_router(adapter):
    with patch(
        "plugins.platforms.slack.project_topic_router.classify_project_topic"
    ) as classify:
        await adapter._handle_slack_message(_event("ordinary channel chatter"))

    classify.assert_not_called()
    adapter.handle_message.assert_not_awaited()
