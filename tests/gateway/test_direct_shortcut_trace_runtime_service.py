"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest


from types import SimpleNamespace

from gateway.direct_shortcut_trace_runtime_service import (
    build_direct_shortcut_runtime_summary,
    record_direct_shortcut_trace,
)


def test_record_direct_shortcut_trace_appends_and_limits_recent_entries():
    runner = SimpleNamespace(_recent_direct_shortcuts=[])
    event = SimpleNamespace(
        text="往 QQ 群 192903718 发：绿帽哥！",
        source=SimpleNamespace(platform=SimpleNamespace(value="qq_napcat"), chat_type="dm", chat_id="179033731"),
    )

    for index in range(25):
        record_direct_shortcut_trace(
            runner,
            event,
            matched_handler=f"handler_{index}",
            attempted_handlers=["a", "b", f"handler_{index}"],
            response="已发送",
        )

    summary = build_direct_shortcut_runtime_summary(runner)

    assert summary["recent_count"] == 20
    assert summary["recent"][0]["matched_handler"] == "handler_24"
    assert summary["recent"][-1]["matched_handler"] == "handler_20"

