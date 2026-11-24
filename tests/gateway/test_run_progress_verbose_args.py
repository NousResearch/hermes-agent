"""Verbose tool-progress args on markdown platforms: a code block on both progress paths."""

import pytest

from agent.display import build_tool_preview
from agent.i18n import SUPPORTED_LANGUAGES, t
from gateway.stream_events import ToolCallChunk
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _run_with_agent
from tests.waiting import wait_until

CODE = "x = a_b\nprint(**kw)"


class _CodeBlockAdapter(ProgressCaptureAdapter):
    supports_code_blocks = True


class _ArgsAgent:
    """Emits one non-terminal tool call whose args look like markdown and span lines."""

    adapter = None

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        self.tool_progress_callback("tool.started", "execute_code", None, {"code": CODE})
        wait_until(lambda: _ArgsAgent.adapter.sent, what="the progress bubble was sent")
        return {"final_response": "done", "messages": [], "api_calls": 1}


async def _progress_text(monkeypatch, tmp_path, adapter_cls) -> str:
    class Capture(adapter_cls):
        def __init__(self, **kw):
            super().__init__(**kw)
            _ArgsAgent.adapter = self

    adapter, _ = await _run_with_agent(monkeypatch, tmp_path, _ArgsAgent, session_id=f"sess-{adapter_cls.__name__}",
                                       config_data={"display": {"tool_progress": "verbose"}}, adapter_cls=Capture)
    return " ".join(c["content"] for c in adapter.sent + adapter.edits)


@pytest.mark.asyncio
async def test_verbose_args_render_in_a_code_block_on_markdown_platforms(monkeypatch, tmp_path):
    """Markdown platforms get a code block with real line breaks (else a_b / **kw** render as
    markup and JSON newlines show as a literal \\n); plain platforms keep the JSON line."""
    md_text = await _progress_text(monkeypatch, tmp_path, _CodeBlockAdapter)
    plain_text = await _progress_text(monkeypatch, tmp_path, ProgressCaptureAdapter)
    assert '```\n{"code": "x = a_b\nprint(**kw)"}\n```' in md_text, md_text
    assert '{"code": "x = a_b\\nprint(**kw)"}' in plain_text and "```" not in plain_text, plain_text


def test_streamed_verbose_args_share_the_code_block_and_skip_it_for_embedded_fences():
    """format_tool_event (streamed tool chrome) renders the same block; args holding ``` anywhere
    stay plain JSON, since Slack/Discord would end the block there and render the rest as markup."""
    adapter = object.__new__(_CodeBlockAdapter)
    out = adapter.format_tool_event(ToolCallChunk("execute_code", args={"code": "a\nb"}), mode="verbose")
    assert out.endswith('```\n{"code": "a\nb"}\n```'), out
    nested = adapter.format_tool_event(ToolCallChunk("execute_code", args={"code": "see ```**k**``` x"}), mode="verbose")
    assert nested.endswith('\n{"code": "see ```**k**``` x"}'), nested


@pytest.mark.parametrize(
    ("tool_name", "args", "expected"),
    [
        ("tool_search", {"queries": ["日历"]}, "正在搜索工具 · 日历"),
        ("tool_describe", {"names": ["calendar", "tasks"]}, "正在读取工具详情 · 2 个工具"),
    ],
)
def test_streamed_bridge_progress_uses_localized_preview(monkeypatch, tool_name, args, expected):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    adapter = object.__new__(ProgressCaptureAdapter)
    preview = build_tool_preview(tool_name, args, max_len=80)
    event = ToolCallChunk(tool_name, preview=preview, args=args)
    assert expected in adapter.format_tool_event(event, mode="all")


@pytest.mark.parametrize(
    ("language", "expected_language"),
    [(lang, lang) for lang in SUPPORTED_LANGUAGES] + [pytest.param("unsupported", "en", id="fallback-en")],
)
@pytest.mark.parametrize("count", [0, 1, 2, 10])
def test_streamed_bridge_progress_keeps_full_description(monkeypatch, language, expected_language, count):
    monkeypatch.setenv("HERMES_LANGUAGE", language)
    adapter = object.__new__(ProgressCaptureAdapter)
    args = {"names": [f"tool_{index}" for index in range(count)]}
    preview = build_tool_preview("tool_describe", args)
    event = ToolCallChunk("tool_describe", preview=preview, args=args)

    title = t("display.bridge_label.describe", lang=expected_language)
    count_key = "display.bridge_label.count_one" if count == 1 else "display.bridge_label.count_many"
    count_text = t(count_key, lang=expected_language, count=count)
    assert str(count) in count_text
    assert f"{title} · {count_text}" in adapter.format_tool_event(event)
