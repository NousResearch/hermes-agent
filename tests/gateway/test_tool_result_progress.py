from __future__ import annotations

import json

from gateway.run import extract_tool_result_progress_message


def test_extract_tool_result_progress_message_uses_explicit_gateway_payload() -> None:
    result = json.dumps({"gateway_progress": "╭─ H-Loops · Running ╮\n✓ validate"})

    assert extract_tool_result_progress_message("hloop_queue_tick", result) == "╭─ H-Loops · Running ╮\n✓ validate"


def test_extract_tool_result_progress_message_ignores_plain_stdout() -> None:
    result = json.dumps({"stdout": "secret or noisy implementation output"})

    assert extract_tool_result_progress_message("terminal", result) is None


def test_extract_tool_result_progress_message_accepts_structured_payload() -> None:
    result = json.dumps({"progress_message": {"content": "visible diagnostics"}})

    assert extract_tool_result_progress_message("hloop_trace_render", result) == "hloop_trace_render\nvisible diagnostics"
