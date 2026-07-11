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
    assert "工具 terminal+read_file (2 calls, 0 failed)" in bar
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
    assert "工具 terminal+read_file (2 calls, 0 failed)" in format_route_depth_bar(receipt)


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
    assert "工具 terminal (1 call, 1 failed)" in format_route_depth_bar(receipt)


def test_total_only_tool_fact_does_not_double_count() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"tools": {"total": 5}})

    bar = format_route_depth_bar(receipt)

    assert "工具 unknown (5 calls, 0 failed)" in bar
    assert "工具 5+5" not in bar


def test_all_four_same_tool_keeps_name_and_explicit_counts() -> None:
    receipt = _completed_receipt()
    receipt.tool_names = ["terminal", "terminal", "terminal", "terminal"]
    receipt.tool_total = 4
    receipt.tool_failed = 0

    bar = format_route_depth_bar(receipt)

    assert "工具 terminal (4 calls, 0 failed)" in bar
    assert "+3" not in bar


def test_duplicate_tool_names_are_deduped_with_explicit_total_and_failures() -> None:
    receipt = _completed_receipt()
    receipt.tool_names = ["skill_view", "read_file", "read_file", "read_file"]
    receipt.tool_total = 4
    receipt.tool_failed = 1

    bar = format_route_depth_bar(receipt)

    assert "工具 skill_view+read_file (4 calls, 1 failed)" in bar
    assert "read_file+read_file" not in bar
    assert "read_file+2" not in bar


def test_more_than_three_unique_tools_uses_ellipsis_and_keeps_counts() -> None:
    receipt = _completed_receipt()
    receipt.tool_names = ["skill_view", "read_file", "terminal", "patch", "python"]
    receipt.tool_total = 5
    receipt.tool_failed = 0

    bar = format_route_depth_bar(receipt)

    assert "工具 skill_view+read_file+terminal+… (5 calls, 0 failed)" in bar
    tool_field = next(field for field in bar.split("｜") if field.startswith("工具 "))
    assert "patch" not in tool_field


def test_failed_duplicate_tool_keeps_explicit_failure_count() -> None:
    receipt = _completed_receipt()
    receipt.tool_names = ["terminal", "terminal", "read_file"]
    receipt.tool_total = 3
    receipt.tool_failed = 2

    bar = format_route_depth_bar(receipt)

    assert "工具 terminal+read_file (3 calls, 2 failed)" in bar
    assert "terminal+terminal" not in bar


def test_moa_facts_render_mechanism_without_counting_references_as_subagents() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "route": {"actual": "moa", "reason": "provider_moa"},
            "moa": {
                "observed": True,
                "reference_models": ["ref-a", "ref-b", "ref-c", "ref-d"],
                "aggregator_model": "agg-model",
            },
        },
    )

    bar = format_route_depth_bar(receipt)

    assert "路径：moa" in bar
    assert "MoA 4+1" in bar
    assert "agents 0" in bar
    assert "subagents 0" in bar


def test_omo_facts_render_compact_mechanism_without_subagent_claims() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "omo": {
                "parent_session_id": "parent-1",
                "descendant_session_ids": ["child-1", "child-2"],
                "session_created_events": 2,
            },
        },
    )

    bar = format_route_depth_bar(receipt)

    assert "OMO 1+2" in bar
    assert "subagents 0" in bar


def test_deep_fact_dict_is_ignored() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "deep": {
                "observed": True,
                "protocol_key": "review",
                "child_session_ids": ["deep-a", "deep-b"],
            },
        },
    )

    bar = format_route_depth_bar(receipt)

    assert bar.startswith("路径：native｜原因：runtime_default")
    assert "Deep" not in bar
