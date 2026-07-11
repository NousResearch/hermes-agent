from __future__ import annotations

import json

import pytest

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


def _assert_single_coordination_field(bar: str, expected: str) -> None:
    fields = bar.split("｜")
    coordination_fields = [field for field in fields if field.startswith("协同 Agent ")]
    assert coordination_fields == [expected]
    assert not any(field.startswith("agents ") for field in fields)
    assert not any(field.startswith("subagents ") for field in fields)


def _apply_messages(messages: list[dict]) -> TurnReceipt:
    receipt = _completed_receipt()
    update_turn_receipt_from_result(
        receipt,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=messages,
    )
    return receipt


def _assistant_delegate_calls(*call_ids: str | None) -> dict:
    calls = []
    for call_id in call_ids:
        call = {"function": {"name": "delegate_task", "arguments": "{}"}}
        if call_id is not None:
            call["id"] = call_id
        calls.append(call)
    return {"role": "assistant", "content": "", "tool_calls": calls}


def _tool_result(
    content: dict | str,
    *,
    call_id: str | None = None,
    name: str | None = None,
    tool_name: str | None = None,
) -> dict:
    message: dict = {"role": "tool", "content": json.dumps(content) if isinstance(content, dict) else content}
    if call_id is not None:
        message["tool_call_id"] = call_id
    if name is not None:
        message["name"] = name
    if tool_name is not None:
        message["tool_name"] = tool_name
    return message


def test_format_native_no_tool_bar_uses_runtime_unknowns_conservatively() -> None:
    receipt = _completed_receipt()

    assert format_route_depth_bar(receipt) == (
        "路径：native｜原因：runtime_default｜OpenCode unknown｜工具 none｜"
        "协同 Agent 0｜人话 unknown｜用时 N/A｜证据 ✓"
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


def test_runtime_facts_surface_tools_coordination_and_human_language() -> None:
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
    _assert_single_coordination_field(bar, "协同 Agent 3")
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
    _assert_single_coordination_field(bar, "协同 Agent 5")


def test_moa_degraded_counts_successful_refs_plus_aggregator_and_marks_partial() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "moa": {
                "observed": True,
                "reference_models": ["ref-a", "ref-b"],
                "reference_count": 2,
                "reference_total": 3,
                "failed_count": 1,
                "aggregator_model": "agg-model",
                "aggregator_count": 1,
            },
        },
    )

    bar = format_route_depth_bar(receipt)

    assert "MoA 2/3+1" in bar
    _assert_single_coordination_field(bar, "协同 Agent 3")
    assert "证据 partial" in bar


def test_moa_all_refs_fail_counts_successful_aggregator_only() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "moa": {
                "observed": True,
                "reference_models": [],
                "reference_count": 0,
                "reference_total": 2,
                "failed_count": 2,
                "aggregator_model": "agg-model",
                "aggregator_count": 1,
            },
        },
    )

    bar = format_route_depth_bar(receipt)

    assert "MoA 0/2+1" in bar
    _assert_single_coordination_field(bar, "协同 Agent 1")
    assert "证据 partial" in bar


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
    _assert_single_coordination_field(bar, "协同 Agent 3")


def test_explicit_trusted_coordination_agent_count_wins_over_delegation_counts() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(
        receipt,
        {
            "coordination": {"observed": True, "agents": 6, "modes": ["deep"]},
            "delegation": {"observed": True, "agents": 2, "subagents": 5},
        },
    )

    bar = format_route_depth_bar(receipt)

    _assert_single_coordination_field(bar, "协同 Agent 6")


def test_split_call_explicit_coordination_remains_authoritative_over_generic_delegation() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"coordination": {"observed": True, "agents": 6, "modes": ["deep"]}})
    apply_turn_facts(receipt, {"delegation": {"observed": True, "agents": 2, "subagents": 5}})

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 6")
    assert receipt.coordination_breakdown == {"explicit": 6}


def test_omo_coordination_and_derived_omo_are_idempotent_per_mechanism() -> None:
    receipt = _completed_receipt()
    facts = {
        "omo": {
            "parent_session_id": "parent-1",
            "descendant_session_ids": ["child-1", "child-2"],
        },
        "coordination": {
            "observed": True,
            "modes": ["omo"],
            "breakdown": {"omo_parent": 1, "omo_descendants": 2},
        },
    }

    apply_turn_facts(receipt, facts)
    apply_turn_facts(receipt, facts)

    bar = format_route_depth_bar(receipt)

    assert bar.count("OMO 1+2") == 1
    _assert_single_coordination_field(bar, "协同 Agent 3")


def test_generic_delegation_does_not_double_count_after_omo_component() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"omo": {"observed": True, "parent_count": 1, "descendant_count": 2}})
    apply_turn_facts(receipt, {"delegation": {"observed": True, "agents": 1, "subagents": 2}})

    bar = format_route_depth_bar(receipt)

    assert "OMO 1+2" in bar
    _assert_single_coordination_field(bar, "协同 Agent 3")
    assert receipt.coordination_breakdown == {"omo": 3}


def test_generic_delegation_prefers_concrete_subagent_count_then_agents_count() -> None:
    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"delegation": {"observed": True, "agents": 4}})
    assert "协同 Agent 4" in format_route_depth_bar(receipt)

    receipt = _completed_receipt()
    apply_turn_facts(receipt, {"delegation": {"observed": True, "agents": 4, "subagents": 2}})
    assert "协同 Agent 2" in format_route_depth_bar(receipt)


def test_explicit_trusted_total_overrides_unknown_delegate_task_component() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "delegation-1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "delegation-1", "content": "not json"},
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
    apply_turn_facts(
        receipt,
        {"coordination": {"observed": True, "agents": 6, "modes": ["deep"]}},
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 6")


def test_delegate_task_tool_result_counts_results_only_from_actual_delegate_call() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "delegation-1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "delegation-1",
            "content": json.dumps({"results": [{"summary": "a"}, {"summary": "b"}]}),
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

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 2")


def test_delegate_task_background_dispatched_payload_counts_actual_top_level_delegate_result() -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls("delegation-1"),
            _tool_result(
                {"status": "dispatched", "mode": "background", "count": 3},
                call_id="delegation-1",
            ),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 3")
    assert receipt.coordination_breakdown == {"delegate_task": 3}


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "dispatched", "mode": "background"},
        {"status": "dispatched", "mode": "background", "count": None},
        {"status": "dispatched", "mode": "background", "count": "3"},
        {"status": "dispatched", "mode": "background", "count": 1.5},
        {"status": "dispatched", "mode": "background", "count": -1},
        {"status": "done", "mode": "background", "count": 3},
        {"status": "dispatched", "mode": "foreground", "count": 3},
        {"count": 3},
    ],
)
def test_delegate_task_background_dispatched_payload_invalid_count_or_shape_fails_closed_unknown(
    payload: dict,
) -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls("delegation-1"),
            _tool_result(payload, call_id="delegation-1"),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


def test_delegate_task_pool_fallback_synchronous_results_shape_remains_accepted() -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls("delegation-1"),
            _tool_result(
                {"results": [{"summary": "a"}, {"summary": "b"}, {"summary": "c"}]},
                call_id="delegation-1",
            ),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 3")
    assert receipt.coordination_breakdown == {"delegate_task": 3}


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "dispatched", "mode": "background", "count": 3},
        {"results": [{"summary": "a"}, {"summary": "b"}, {"summary": "c"}]},
    ],
)
def test_delegate_task_result_with_matching_call_id_but_explicit_terminal_name_fails_closed_unknown(
    payload: dict,
) -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls("d1"),
            _tool_result(payload, call_id="d1", name="terminal"),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


def test_delegate_task_result_with_matching_call_id_but_explicit_read_file_tool_name_fails_closed_unknown() -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls("d1"),
            _tool_result(
                {"results": [{"summary": "a"}]},
                call_id="d1",
                tool_name="read_file",
            ),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


@pytest.mark.parametrize(
    "tool_result",
    [
        _tool_result({"status": "dispatched", "mode": "background", "count": 1}, call_id="d1"),
        _tool_result({"results": [{"summary": "a"}]}, call_id="d1", name="delegate_task"),
        _tool_result({"results": [{"summary": "a"}]}, call_id="d1", name="terminal"),
    ],
)
def test_reused_assistant_call_id_between_delegate_task_and_terminal_fails_closed_unknown(
    tool_result: dict,
) -> None:
    receipt = _apply_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "d1", "function": {"name": "delegate_task", "arguments": "{}"}},
                    {"id": "d1", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            tool_result,
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


@pytest.mark.parametrize("result_message_count", [1, 2])
def test_duplicate_delegate_task_expected_call_id_fails_closed_unknown(
    result_message_count: int,
) -> None:
    messages = [_assistant_delegate_calls("d1", "d1")]
    messages.extend(
        _tool_result({"results": [{"summary": f"child-{index}"}]}, call_id="d1")
        for index in range(result_message_count)
    )

    receipt = _apply_messages(messages)

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


def test_single_anonymous_delegate_call_with_one_named_anonymous_result_may_count() -> None:
    receipt = _apply_messages(
        [
            _assistant_delegate_calls(None),
            _tool_result(
                {"results": [{"summary": "a"}, {"summary": "b"}]},
                name="delegate_task",
            ),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 2")
    assert receipt.coordination_breakdown == {"delegate_task": 2}


@pytest.mark.parametrize(
    ("anonymous_call_count", "anonymous_result_count"),
    [(2, 1), (1, 2), (2, 2)],
)
def test_multiple_anonymous_delegate_calls_or_results_fail_closed_unknown(
    anonymous_call_count: int,
    anonymous_result_count: int,
) -> None:
    messages = [_assistant_delegate_calls(*([None] * anonymous_call_count))]
    messages.extend(
        _tool_result({"results": [{"summary": f"child-{index}"}]}, name="delegate_task")
        for index in range(anonymous_result_count)
    )

    receipt = _apply_messages(messages)

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")
    assert receipt.coordination_breakdown == {"delegate_task": None}


@pytest.mark.parametrize("tool_name", ["terminal", "read_file"])
def test_generic_tool_background_count_payload_cannot_forge_coordination(tool_name: str) -> None:
    receipt = _apply_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tool-1", "function": {"name": tool_name, "arguments": "{}"}}],
            },
            _tool_result(
                {"status": "dispatched", "mode": "background", "count": 3},
                call_id="tool-1",
                name=tool_name,
            ),
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 0")


def test_assistant_prose_background_count_payload_cannot_forge_coordination() -> None:
    receipt = _apply_messages(
        [
            {
                "role": "assistant",
                "content": json.dumps({"status": "dispatched", "mode": "background", "count": 3}),
                "tool_calls": [],
            }
        ]
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 0")


def test_generic_delegation_does_not_double_count_after_delegate_task_result_component() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "delegation-1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "delegation-1",
            "content": json.dumps({"results": [{"summary": "a"}, {"summary": "b"}]}),
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
    apply_turn_facts(receipt, {"delegation": {"observed": True, "agents": 2, "subagents": 2}})

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 2")
    assert receipt.coordination_breakdown == {"delegate_task": 2}


def test_delegate_task_unparseable_result_may_render_unknown() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "delegation-1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "delegation-1", "content": "not json"},
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

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent unknown")


def test_generic_tool_output_with_results_does_not_change_coordination_facts() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tool-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "name": "terminal",
            "content": json.dumps({"results": [{"summary": "fake"}, {"summary": "fake"}]}),
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

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 0")


def test_duplicate_delegate_task_results_for_one_call_fail_closed_unknown() -> None:
    receipt = _completed_receipt()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "d1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "d1",
            "content": json.dumps({"results": [{"summary": "a"}, {"summary": "b"}]}),
        },
        {
            "role": "tool",
            "tool_call_id": "d1",
            "content": json.dumps({"results": [{"summary": "c"}, {"summary": "d"}]}),
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
    bar = format_route_depth_bar(receipt)

    _assert_single_coordination_field(bar, "协同 Agent unknown")
    assert "协同 Agent 4" not in bar
    assert receipt.coordination_breakdown == {"delegate_task": None}


def test_trusted_explicit_moa_total_is_whole_turn_and_order_independent_with_delegate_task() -> None:
    coordination_facts = {"coordination": {"observed": True, "agents": 6, "modes": ["moa"]}}
    delegate_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "d1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "d1",
            "content": json.dumps({"results": [{"summary": "a"}, {"summary": "b"}]}),
        },
    ]

    delegate_first = _completed_receipt()
    update_turn_receipt_from_result(
        delegate_first,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=delegate_messages,
    )
    apply_turn_facts(delegate_first, coordination_facts)

    explicit_first = _completed_receipt()
    apply_turn_facts(explicit_first, coordination_facts)
    update_turn_receipt_from_result(
        explicit_first,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=delegate_messages,
    )

    for receipt in (delegate_first, explicit_first):
        bar = format_route_depth_bar(receipt)
        _assert_single_coordination_field(bar, "协同 Agent 6")
        assert "协同 Agent 8" not in bar
        assert receipt.coordination_breakdown == {"explicit": 6}


def test_trusted_explicit_omo_total_is_whole_turn_and_order_independent_with_delegate_task() -> None:
    coordination_facts = {"coordination": {"observed": True, "agents": 6, "modes": ["omo"]}}
    delegate_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "d1", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "d1",
            "content": json.dumps({"results": [{"summary": "a"}, {"summary": "b"}]}),
        },
    ]

    delegate_first = _completed_receipt()
    update_turn_receipt_from_result(
        delegate_first,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=delegate_messages,
    )
    apply_turn_facts(delegate_first, coordination_facts)

    explicit_first = _completed_receipt()
    apply_turn_facts(explicit_first, coordination_facts)
    update_turn_receipt_from_result(
        explicit_first,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="text_response(finish_reason=stop)",
        messages=delegate_messages,
    )

    for receipt in (delegate_first, explicit_first):
        bar = format_route_depth_bar(receipt)
        _assert_single_coordination_field(bar, "协同 Agent 6")
        assert "协同 Agent 8" not in bar
        assert receipt.coordination_breakdown == {"explicit": 6}


def test_coordination_moa_breakdown_accepts_singular_aggregator_without_explicit_total() -> None:
    receipt = _completed_receipt()

    apply_turn_facts(
        receipt,
        {
            "coordination": {
                "modes": ["moa"],
                "breakdown": {"moa_references": 4, "moa_aggregator": 1},
            }
        },
    )

    _assert_single_coordination_field(format_route_depth_bar(receipt), "协同 Agent 5")


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
