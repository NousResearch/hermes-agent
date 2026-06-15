"""Tests for suppressing text-encoded provider tool-call markup."""

from __future__ import annotations

from agent.tool_call_scrubber import StreamingToolCallScrubber, strip_tool_call_markup


def _drive(deltas: list[str]) -> str:
    scrubber = StreamingToolCallScrubber()
    return "".join(scrubber.feed(delta) for delta in deltas) + scrubber.flush()


def test_strip_dsml_tool_call_from_completed_response() -> None:
    raw = (
        "I will update that.\n"
        "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"write_file\">"
        "<｜DSML｜parameter name=\"arguments\">{\"content\":\"secret code\"}"
    )
    assert strip_tool_call_markup(raw) == "I will update that.\n"


def test_streaming_dsml_tool_call_split_across_deltas() -> None:
    out = _drive(
        [
            "Working\n<｜DSM",
            "L｜tool_calls> <｜DSML｜invoke name=\"write_file\">",
            "{\"content\":\"internal\"}",
            "<｜DSML｜/tool_calls>Done",
        ]
    )
    assert out == "Working\nDone"


def test_ascii_dsml_variant_is_stripped() -> None:
    assert _drive(["ok <|DSML|tool_calls>{}", "<|DSML|/tool_calls> done"]) == "ok  done"


def test_unclosed_dsml_tool_call_dropped_at_flush() -> None:
    assert _drive(["visible <｜DSML｜tool_calls>{\"content\":\"internal\"}"]) == "visible "
