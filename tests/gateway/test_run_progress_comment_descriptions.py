"""Integration coverage for per-platform tool comment descriptions."""

import asyncio
import threading

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from tests.gateway.test_run_progress_topics import (
    CodeBlockProgressAdapter,
    ProgressCaptureAdapter,
    _run_with_agent,
)


class CommentDescriptionProgressAdapter(ProgressCaptureAdapter):
    all_progress_rendered = threading.Event()
    next_tool_ready = threading.Event()

    @classmethod
    def _mark_rendered(cls, content):
        if "Check branch and status" in content:
            # B's sender coalesces within its 1.5s edit interval. Exercise labels
            # without importing the independent full-progress/drain changes.
            asyncio.get_running_loop().call_later(1.6, cls.next_tool_ready.set)
        if "Analyze test results" in content:
            cls.all_progress_rendered.set()

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(
            chat_id, content, reply_to=reply_to, metadata=metadata
        )
        self._mark_rendered(content)
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        result = await super().edit_message(chat_id, message_id, content)
        self._mark_rendered(content)
        return result


class CommentDescriptionAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started",
            "terminal",
            "# Check branch and status #\ngit status",
            {"command": "# Check branch and status #\ngit status"},
        )
        assert CommentDescriptionProgressAdapter.next_tool_ready.wait(timeout=5.0)
        cb(
            "tool.started",
            "execute_code",
            "# Analyze test results\nprint('ok')",
            {"code": "# Analyze test results\nprint('ok')"},
        )
        CommentDescriptionProgressAdapter.all_progress_rendered.wait(timeout=3.0)
        return {"final_response": "done", "messages": [], "api_calls": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
async def test_platform_comment_description_override_reaches_real_turn_context(
    monkeypatch,
    tmp_path,
    grouping,
):
    CommentDescriptionProgressAdapter.all_progress_rendered.clear()
    CommentDescriptionProgressAdapter.next_tool_ready.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentDescriptionAgent,
        session_id=f"sess-comment-description-{grouping}",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": grouping,
                "tool_progress_comment_descriptions": False,
                "interim_assistant_messages": False,
                "platforms": {
                    "mattermost": {
                        "tool_progress_comment_descriptions": True,
                    }
                },
            }
        },
        platform=Platform.MATTERMOST,
        chat_id="channel-comment-description",
        chat_type="channel",
        thread_id="root-comment-description",
        adapter_cls=CommentDescriptionProgressAdapter,
    )

    assert result["final_response"] == "done"
    contents = [call["content"] for call in adapter.sent] + [
        call["content"] for call in adapter.edits
    ]
    combined = "\n".join(contents)
    assert "Running: Check branch and status" in combined
    assert "Running code: Analyze test results" in combined
    assert "git status" not in combined
    assert "print('ok')" not in combined


@pytest.mark.asyncio
async def test_platform_false_override_keeps_real_turn_on_legacy_preview(
    monkeypatch,
    tmp_path,
):
    CommentDescriptionProgressAdapter.all_progress_rendered.clear()
    CommentDescriptionProgressAdapter.next_tool_ready.clear()
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentDescriptionAgent,
        session_id="sess-comment-description-disabled",
        config_data={
            "display": {
                "tool_progress": "all",
                "tool_progress_grouping": "separate",
                "tool_progress_comment_descriptions": True,
                "interim_assistant_messages": False,
                "platforms": {
                    "mattermost": {
                        "tool_progress_comment_descriptions": False,
                    }
                },
            }
        },
        platform=Platform.MATTERMOST,
        chat_id="channel-comment-description-disabled",
        chat_type="channel",
        thread_id="root-comment-description-disabled",
        adapter_cls=CommentDescriptionProgressAdapter,
    )

    assert result["final_response"] == "done"
    combined = "\n".join(call["content"] for call in adapter.sent)
    assert "Running: Check branch and status" not in combined
    assert "Running code: Analyze test results" not in combined
    assert "# Check branch and status" in combined
    assert "# Analyze test results" in combined


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "source_text", "second_enabled", "expected_first", "expected_second"),
    [
        ("terminal", "# " + "x" * 40 + "\necho ok", True,
         "💻 Running: " + "x" * 17 + "．．．", '💻 terminal: "xxxxx．．．"'),
        ("execute_code", "# " + "x" * 40 + "\nprint(1)", False,
         "🐍 Running code: " + "x" * 17 + "．．．", '🐍 execute_code: "# xxx..."'),
        ("terminal", "git status --short --branch", True,
         "💻 terminal\n```\ngit status --shor...\n```", "💻 terminal\n```\ngit s...\n```"),
        ("execute_code", "print('long fallback payload')", True,
         "🐍 Running code print('long fallb...", '🐍 execute_code: "print..."'),
    ],
)
async def test_real_turn_producers_keep_interleaved_comment_settings(
    monkeypatch, tmp_path, tool_name, source_text, second_enabled,
    expected_first, expected_second,
):
    import copy
    from agent import display as agent_display
    from agent.display import (
        get_tool_preview_max_len,
        set_friendly_tool_labels, set_tool_preview_max_len,
    )

    first_ready, second_ready, first_emitted = (threading.Event() for _ in range(3))
    delivered = {p: threading.Event() for p in (Platform.TELEGRAM, Platform.MATTERMOST)}
    contexts = {}
    args = {"command" if tool_name == "terminal" else "code": source_text}
    original_args = copy.deepcopy(args)
    original_friendly, original_cap = agent_display._friendly_tool_labels, get_tool_preview_max_len()
    monkeypatch.setattr("agent.display.get_tool_emoji", lambda name, **kw: {"terminal": "💻", "execute_code": "🐍"}[name])

    class InterleavedAdapter(CodeBlockProgressAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
            delivered[self.platform].set()
            return result

    class InterleavedAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            callback = self.tool_progress_callback
            ctx = callback.__self__._ctx
            platform = ctx.source.platform
            contexts[platform] = ctx
            if platform == Platform.TELEGRAM:
                first_ready.set()
                assert second_ready.wait(5.0)
                # The other real producer has now overwritten the legacy globals.
                callback("tool.started", tool_name, source_text, args)
                set_friendly_tool_labels(True)
                set_tool_preview_max_len(40)
                first_emitted.set()
            else:
                second_ready.set()
                assert first_emitted.wait(5.0)
                callback("tool.started", tool_name, source_text, args)
            assert delivered[platform].wait(5.0)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    config = {"display": {
        "tool_progress": "all", "tool_progress_grouping": "separate",
        "tool_progress_comment_descriptions": False,
        "interim_assistant_messages": False,
        "platforms": {
            "telegram": {"tool_progress_comment_descriptions": True,
                         "friendly_tool_labels": True, "tool_preview_length": 20},
            "mattermost": {"tool_progress_comment_descriptions": second_enabled,
                           "friendly_tool_labels": False, "tool_preview_length": 8},
        },
    }}
    first_task = asyncio.create_task(_run_with_agent(
        monkeypatch, tmp_path, InterleavedAgent, session_id="comment-first",
        config_data=config, platform=Platform.TELEGRAM, chat_id="first",
        adapter_cls=InterleavedAdapter,
    ))
    second_task = None
    try:
        assert await asyncio.to_thread(first_ready.wait, 5.0)
        second_task = asyncio.create_task(_run_with_agent(
            monkeypatch, tmp_path, InterleavedAgent, session_id="comment-second",
            config_data=config, platform=Platform.MATTERMOST, chat_id="second",
            adapter_cls=InterleavedAdapter,
        ))
        first_result, second_result = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), 15.0,
        )
        for (adapter, result), expected in zip(
            (first_result, second_result), (expected_first, expected_second),
        ):
            assert result["final_response"] == "done"
            assert [entry["content"] for entry in adapter.sent] == [expected]
        for platform, enabled, cap, friendly in (
            (Platform.TELEGRAM, True, 20, True),
            (Platform.MATTERMOST, second_enabled, 8, False),
        ):
            ctx = contexts[platform]
            assert ctx.tool_progress_comment_descriptions is enabled
            assert ctx.tool_preview_max_len == cap
            assert ctx.friendly_tool_labels is friendly
        assert args == original_args
    finally:
        second_ready.set()
        first_emitted.set()
        for event in delivered.values():
            event.set()
        tasks = [t for t in (first_task, second_task) if t is not None]
        await asyncio.gather(*tasks, return_exceptions=True)
        set_friendly_tool_labels(original_friendly)
        set_tool_preview_max_len(original_cap)
