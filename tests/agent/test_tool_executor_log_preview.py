import json

from agent.tool_executor import _tool_error_log_preview, _tool_result_log_preview


def test_tool_error_log_preview_strips_cognitive_shock_banner_from_json_output():
    payload = {
        "output": """
========================================================================
!!! CRITICAL TOOL FAILURE DETECTED !!! DO NOT HALLUCINATE SUCCESS !!!
========================================================================
REASON: Non-zero exit code: 4
INSTRUCTION: YOU MUST ACKNOWLEDGE THIS FAILURE IN YOUR RESPONSE.
ERROR: not found: tests/foo.py::missing
""",
        "exit_code": 4,
        "error": None,
        "_COGNITIVE_SHOCK_WARNING": "huge banner duplicate",
    }

    preview = _tool_error_log_preview("terminal", json.dumps(payload), limit=500)

    assert "structured_error" in preview
    assert "tool_failure_guard=critical" in preview
    assert "exit_code=4" in preview
    assert "reason=Non-zero exit code: 4" in preview
    assert "_COGNITIVE_SHOCK_WARNING" not in preview
    assert "CRITICAL TOOL FAILURE DETECTED" not in preview
    assert "\n" not in preview


def test_tool_error_log_preview_is_single_line_and_bounded_for_plain_text():
    preview = _tool_error_log_preview(
        "terminal",
        "first line\nsecond line with useful detail\nthird line",
        limit=32,
    )

    assert preview == "first line second line with use…"
    assert "\n" not in preview
    assert len(preview) <= 32


def test_tool_error_log_preview_compacts_plain_text_critical_banner():
    preview = _tool_error_log_preview(
        "terminal",
        """
========================================================================
!!! CRITICAL TOOL FAILURE DETECTED !!! DO NOT HALLUCINATE SUCCESS !!!
========================================================================
REASON: Timeout waiting for process
INSTRUCTION: DO NOT PROCEED AS IF THE OPERATION SUCCEEDED.
""",
        limit=200,
    )

    assert preview == "tool_failure_guard=critical reason=Timeout waiting for process"
    assert "CRITICAL TOOL FAILURE DETECTED" not in preview
    assert "\n" not in preview


def test_tool_error_log_preview_does_not_mutate_dict_payload():
    payload = {
        "success": False,
        "error": "bad thing",
        "_COGNITIVE_SHOCK_WARNING": "keep me in original",
    }

    preview = _tool_error_log_preview("write_file", payload)

    assert "success=false" in preview
    assert "error=bad thing" in preview
    assert "_COGNITIVE_SHOCK_WARNING" not in preview
    assert payload["_COGNITIVE_SHOCK_WARNING"] == "keep me in original"


def test_tool_error_log_preview_drops_heavy_file_preview_fields():
    payload = {
        "success": False,
        "error": "Found 2 matches for old_string. Provide more context.",
        "file_preview": "---\nname: cliproxy-mechanics\n" + "x" * 2000,
    }

    preview = _tool_error_log_preview("skill_manage", payload, limit=500)

    assert "structured_error" in preview
    assert "Found 2 matches" in preview
    assert "file_preview" not in preview
    assert "cliproxy-mechanics" not in preview
    assert "\n" not in preview


def test_tool_error_log_preview_summarizes_nested_web_extract_errors():
    payload = {
        "results": [
            {
                "url": "https://kiro.dev/docs/cli/enterprise/governance/api-keys",
                "content": "large markdown" * 100,
                "error": "no content to process",
            }
        ],
        "raw_output": "huge raw body" * 100,
    }

    preview = _tool_error_log_preview("web_extract", payload, limit=500)

    assert "structured_error" in preview
    assert "results_error=https://kiro.dev/docs/cli/enterprise/governance/api-keys: no content to process" in preview
    assert "large markdown" not in preview
    assert "huge raw body" not in preview
    assert "\n" not in preview


def test_tool_error_log_preview_drops_memory_state_snapshots():
    payload = {
        "success": False,
        "error": "Memory at 2,198/2,200 chars. Adding this entry would exceed the limit.",
        "current_entries": [
            "DAVE require rejoin. /very/private/context",
            "another private memory entry",
        ],
    }

    preview = _tool_error_log_preview("memory", payload, limit=500)

    assert "structured_error" in preview
    assert "Memory at 2,198/2,200" in preview
    assert "current_entries" not in preview
    assert "DAVE require rejoin" not in preview
    assert "private memory" not in preview
    assert "\n" not in preview


def test_tool_result_log_preview_handles_non_string_results():
    result = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "first line\nsecond line"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }

    preview = _tool_result_log_preview(result, limit=64)

    assert isinstance(preview, str)
    assert "\n" not in preview
    assert len(preview) <= 64
