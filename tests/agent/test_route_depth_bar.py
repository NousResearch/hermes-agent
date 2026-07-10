from __future__ import annotations

from agent.route_depth_bar import apply_route_depth_bar, format_route_depth_bar
from agent.turn_receipt import TurnReceipt, apply_turn_facts, update_turn_receipt_from_result


def _completed_receipt() -> TurnReceipt:
    receipt = TurnReceipt.start(
        session_id="session-1",
        turn_id="turn-1",
        provider="openrouter",
        model="openai/gpt-5.5",
        platform="cli",
    )
    update_turn_receipt_from_result(
        receipt,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=[],
    )
    return receipt


def test_format_native_no_tool_bar_uses_runtime_unknowns_conservatively() -> None:
    receipt = _completed_receipt()

    assert format_route_depth_bar(receipt) == (
        "路径：native｜原因：runtime_default｜OpenCode unknown｜工具 none｜"
        "agents 0｜subagents 0｜人话 unknown｜用时 N/A｜证据 ✓"
    )


def test_apply_prepends_one_runtime_bar() -> None:
    receipt = _completed_receipt()

    text, changed = apply_route_depth_bar("Body", receipt)

    assert changed is True
    assert text.count("路径：") == 1
    assert text.splitlines()[0].startswith("路径：native｜原因：runtime_default")
    assert text.splitlines()[1] == "Body"


def test_existing_model_authored_bar_is_replaced() -> None:
    receipt = _completed_receipt()
    original = "路径：模型声称｜原因：模型记忆｜OpenCode 已调用\nBody"

    text, changed = apply_route_depth_bar(original, receipt)

    assert changed is True
    assert text.count("路径：") == 1
    assert "模型声称" not in text
    assert text.splitlines()[1] == "Body"


def test_runtime_facts_surface_tools_subagents_and_human_language() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "route": {"actual": "moa", "reason": "provider_moa"},
            "opencode": {"observed": False},
            "tools": {"called": True, "names": ["terminal", "read_file"], "total": 2, "succeeded": 2, "failed": 0},
            "delegation": {"observed": True, "agents": 1, "subagents": 3},
            "human_language": {"observed": True, "source": "unit"},
            "evidence": {"level": "ok", "sources": ["unit"]},
        },
    )

    bar = format_route_depth_bar(receipt)

    assert "路径：moa" in bar
    assert "原因：provider_moa" in bar
    assert "OpenCode 未调用" in bar
    assert "工具 terminal+read_file" in bar
    assert "agents 1" in bar
    assert "subagents 3" in bar
    assert "人话 ✓" in bar
    assert "证据 ✓" in bar


def test_success_tool_payloads_do_not_false_positive_as_failures() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "t2", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": '{"output":"ok", "exit_code":0, "error":null}',
        },
        {
            "role": "tool",
            "tool_call_id": "t2",
            "content": "No errors found in the file; 0 failed checks.",
        },
    ]

    update_turn_receipt_from_result(
        receipt,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=messages,
    )

    assert receipt.tool_failed == 0
    assert "failed)" not in format_route_depth_bar(receipt)


def test_structured_failed_tool_payload_is_counted() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": '{"output":"", "exit_code":1, "error":"permission denied"}',
        },
    ]

    update_turn_receipt_from_result(
        receipt,
        completed=False,
        failed=True,
        interrupted=False,
        api_calls=1,
        exit_reason="tool_error",
        messages=messages,
    )

    assert receipt.tool_failed == 1
    assert "terminal(1 failed)" in format_route_depth_bar(receipt)


def test_total_only_tool_fact_does_not_double_count() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"tools": {"total": 5}})

    bar = format_route_depth_bar(receipt)

    assert "工具 5" in bar
    assert "工具 5+5" not in bar
