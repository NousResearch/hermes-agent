"""T3 Code's borrowed editor MCP tools reach the client in the shape T3 renders."""

import json

from acp.schema import ToolCallProgress, ToolCallStart

from acp_adapter.tools import build_tool_complete, build_tool_start

T3_RENDER = "mcp__editor_t3_code_5e30baec__html_render"
RENDER_RESULT = {
    "htmlRender": {"attachmentId": "thread-abc-html", "title": "Memory", "height": 420},
    "message": "Shown to the reader above your reply.",
}


def _mcp_envelope(payload):
    """Hermes' MCP handler wraps an MCP text block as ``{"result": "<text>"}``."""
    return json.dumps({"result": json.dumps(payload)})


def test_t3_tool_start_is_titled_for_t3():
    start = build_tool_start("tc-1", T3_RENDER, {"title": "Memory", "height": 420, "html": "<p>x</p>"})
    assert isinstance(start, ToolCallStart)
    assert start.title == "t3-code.html_render"
    assert start.raw_input == {"title": "Memory", "height": 420, "html": "<p>x</p>"}


def test_t3_tool_complete_passes_the_tool_json_through():
    update = build_tool_complete("tc-1", T3_RENDER, result=_mcp_envelope(RENDER_RESULT))
    assert isinstance(update, ToolCallProgress)
    assert update.status == "completed"
    assert update.title == "t3-code.html_render"
    assert update.raw_output == RENDER_RESULT
    assert update.content
    assert json.loads(update.content[0].content.text) == RENDER_RESULT


def test_t3_tool_complete_prefers_structured_content():
    result = json.dumps({"result": "rendered", "structuredContent": RENDER_RESULT})
    assert build_tool_complete("tc-1", T3_RENDER, result=result).raw_output == RENDER_RESULT


def test_t3_tool_error_stays_failed():
    update = build_tool_complete("tc-1", T3_RENDER, result=json.dumps({"error": "boom"}), is_error=True)
    assert update.status == "failed"


def test_other_editor_and_mcp_tools_keep_the_generic_summary():
    for name in ("mcp__editor_github_1234abcd__create_issue", "mcp__t3_code_lookalike__html_render"):
        update = build_tool_complete("tc-1", name, result=_mcp_envelope(RENDER_RESULT))
        assert update.title is None
        assert update.raw_output is None
