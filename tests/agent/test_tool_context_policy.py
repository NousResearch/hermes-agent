import json

from agent.tool_context_policy import (
    ToolContextConfig,
    classify_tool,
    compact_tool_result,
)


def test_small_output_is_unchanged():
    content = "ok\nsmall output"
    assert compact_tool_result("terminal", {"command": "echo ok"}, content) == content


def test_html_output_is_compacted_to_page_signals():
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Example Page</title>
        <meta name="description" content="Example description">
        <style>body {{ color: red; }}</style>
        <script>console.log("noise")</script>
      </head>
      <body>
        <h1>Main Heading</h1>
        <h2>Useful Section</h2>
        <p>{body}</p>
      </body>
    </html>
    """.format(body="useful text " * 3000)
    result = compact_tool_result(
        "web_extract",
        {"url": "https://example.com"},
        json.dumps({"output": html}),
        config=ToolContextConfig(web_max_chars=4000, head_chars=1000, tail_chars=500),
    )
    assert len(result) <= 4000
    assert "tool output compacted" in result
    assert "Example Page" in result
    assert "Main Heading" in result
    assert "console.log" not in result


def test_terminal_output_keeps_error_lines_and_tail():
    output = "\n".join(
        ["start"]
        + [f"line {i}" for i in range(2000)]
        + ["ERROR: build failed", "final line"]
    )
    result = compact_tool_result(
        "terminal",
        {"command": "npm test"},
        json.dumps({"status": "error", "output": output, "exit_code": 1}),
        config=ToolContextConfig(terminal_max_chars=3500, head_chars=800, tail_chars=700),
    )
    assert len(result) <= 3500
    assert "ERROR: build failed" in result
    assert "final line" in result
    assert "tool output compacted" in result


def test_large_json_skeleton_preserves_keys():
    payload = {
        "status": "success",
        "items": [{"id": i, "value": "x" * 100} for i in range(1000)],
        "summary": "done",
    }
    result = compact_tool_result(
        "unknown_tool",
        {},
        json.dumps(payload),
        config=ToolContextConfig(json_max_chars=2500, head_chars=500, tail_chars=500),
    )
    assert len(result) <= 2500
    parsed = json.loads(result)
    assert parsed["_hermes_context_compacted"] is True
    assert parsed["summary"]["status"] == "success"
    assert parsed["summary"]["items"].startswith("[list len=")


def test_disabled_config_returns_original():
    content = "x" * 20_000
    result = compact_tool_result(
        "terminal",
        {},
        content,
        config=ToolContextConfig(enabled=False, terminal_max_chars=1000),
    )
    assert result == content


def test_classify_html_by_content():
    assert classify_tool("terminal", {}, "<html><body>hello</body></html>") == "terminal"
    assert classify_tool("unknown", {}, "<html><body>hello</body></html>") == "web"
