from gateway.config import GatewayConfig, Platform
from gateway.shared_group_history_runtime_service import (
    prepare_history_for_agent,
    simplify_shared_group_history_for_agent,
)
from gateway.session import SessionContext, SessionSource, build_session_context_prompt


def test_simplify_shared_group_history_drops_internal_artifacts_and_tool_traces():
    history = [
        {
            "role": "user",
            "content": "[QQ项目群合并消息，共 1 条]\n發發發: 前一个正常问题",
            "timestamp": "2026-04-10T10:00:00",
        },
        {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION] Earlier turns in this conversation were compacted...",
            "timestamp": "2026-04-10T10:00:01",
        },
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]\n- [>] t1",
            "timestamp": "2026-04-10T10:00:02",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}],
            "timestamp": "2026-04-10T10:00:03",
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_name": "terminal",
            "content": "{\"output\":\"noise\"}",
            "timestamp": "2026-04-10T10:00:04",
        },
        {
            "role": "assistant",
            "content": "旧结论，应该保留。",
            "timestamp": "2026-04-10T10:00:05",
        },
        {
            "role": "user",
            "content": "[QQ项目群合并消息，共 1 条]\nArthurX: 新问题",
            "timestamp": "2026-04-10T10:00:06",
        },
    ]

    cleaned = simplify_shared_group_history_for_agent(history)

    assert cleaned == [
        {"role": "user", "content": "[QQ项目群合并消息，共 1 条]\n發發發: 前一个正常问题"},
        {"role": "assistant", "content": "旧结论，应该保留。"},
        {"role": "user", "content": "[QQ项目群合并消息，共 1 条]\nArthurX: 新问题"},
    ]


def test_shared_group_context_prompt_instructs_model_to_focus_on_latest_batch():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.QQ_NAPCAT,
            chat_id="685403987",
            chat_type="group",
            user_id="179033731",
            user_name="發發發",
            chat_name="685403987",
        ),
        connected_platforms=[],
        home_channels={},
        admin_user_ids=[],
        shared_session=True,
        shared_session_kind="group",
    )

    prompt = build_session_context_prompt(context)

    assert "latest" in prompt.lower()
    assert "current batch" in prompt.lower()
    assert "stale topics" in prompt.lower()
    assert "real room transcript" in prompt.lower()
    assert "your own replies" in prompt.lower()
    assert "completed replies" in prompt.lower()
    assert "not drafts or pending points" in prompt.lower()
    assert "newest active prompt only" in prompt.lower()


def test_prepare_history_for_agent_leaves_non_shared_group_history_unchanged():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    prepared = prepare_history_for_agent(
        history,
        shared_session_kind=None,
    )

    assert prepared == history
