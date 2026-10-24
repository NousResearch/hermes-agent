from unittest.mock import patch


def _compressor():
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=0,
            protect_last_n=1,
            quiet_mode=True,
        )


def _legacy_tool_turns():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "function",
            "tool_call_id": "call-1",
            "name": "lookup",
            "content": "legacy result",
        },
        {"role": "user", "content": "tail"},
    ]


def test_align_boundary_forward_skips_legacy_function_results():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "function", "name": "old_tool", "content": "legacy result"},
        {"role": "user", "content": "next user"},
    ]

    assert _compressor()._align_boundary_forward(messages, 1) == 2


def test_align_boundary_backward_keeps_legacy_function_result_with_parent_call():
    assert _compressor()._align_boundary_backward(_legacy_tool_turns(), 3) == 1


def test_turn_pair_includes_legacy_function_results():
    assert _compressor()._find_turn_pair_end(_legacy_tool_turns(), 0) == 3


def test_pending_round_includes_legacy_function_results():
    from agent.context_compressor import _pending_tool_round

    assert list(_pending_tool_round(_legacy_tool_turns()[:3])) == [2]


def test_summary_serialization_labels_legacy_function_as_tool_result():
    serialized = _compressor()._serialize_for_summary([_legacy_tool_turns()[2]])

    assert serialized == "[TOOL RESULT call-1]: legacy result"


def test_static_fallback_extracts_legacy_function_result_facts():
    with patch(
        "agent.context_compressor._summarize_tool_result",
        return_value="preserved legacy fact",
    ) as summarize:
        summary = _compressor()._build_static_fallback_summary(
            _legacy_tool_turns()[1:3]
        )

    summarize.assert_called_once_with("lookup", "{}", "legacy result")
    assert "preserved legacy fact" in summary


def test_template_visibility_skips_legacy_function_results():
    from agent.context_compressor import _template_visible_role

    assert _template_visible_role(_legacy_tool_turns()[2]) is None


def test_summary_placement_alternates_around_legacy_function_results():
    from agent.context_compressor import ContextCompressor

    head = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        _legacy_tool_turns()[2],
    ]

    summary_role, _, _, _ = ContextCompressor._summary_placement(head, [], 1)

    assert summary_role == "user"


def test_dedupe_handles_legacy_function_results():
    from agent.context_compressor import ContextCompressor

    content = "legacy duplicate " * 500
    messages = [
        {"role": "function", "tool_call_id": "old", "content": content},
        {"role": "function", "tool_call_id": "new", "content": content},
    ]

    assert ContextCompressor._dedupe_tool_results(messages) == 1
    assert messages[0]["content"].startswith("[Duplicate tool output")
    assert messages[1]["content"] == content


def test_demote_handles_legacy_function_results():
    from agent.context_compressor import ContextCompressor

    messages = [
        {"role": "function", "tool_call_id": "call-1", "content": "legacy body " * 500}
    ]

    changed = ContextCompressor._demote_tool_result_at(
        messages,
        0,
        {"call-1": ("lookup", "{}")},
        min_prune_chars=100,
    )

    assert changed is True
    assert messages[0]["role"] == "function"
    assert messages[0]["content"] != "legacy body " * 500


def test_orphan_classifier_pairs_legacy_function_results():
    from agent.agent_runtime_helpers import _classify_tool_call_orphans

    _, result_ids, orphaned_results, missing_calls = _classify_tool_call_orphans(
        _legacy_tool_turns()[:3]
    )

    assert "call-1" in result_ids
    assert orphaned_results == []
    assert missing_calls == []
