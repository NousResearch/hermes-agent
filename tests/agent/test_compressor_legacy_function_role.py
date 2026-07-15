from unittest.mock import MagicMock, patch


def _compressor():
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
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


def test_system_only_protected_head_still_forces_user_summary_role():
    compressor = _compressor()
    compressor.compression_count = 1
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "question 2"},
        {"role": "assistant", "content": "answer 2"},
        {"role": "user", "content": "question 3"},
        {"role": "assistant", "content": "answer 3"},
    ]
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "compressed middle"

    with patch("agent.context_compressor.call_llm", return_value=response):
        compressed = compressor.compress(messages)

    non_system = [msg for msg in compressed if msg.get("role") != "system"]
    assert non_system[0]["role"] == "user"
