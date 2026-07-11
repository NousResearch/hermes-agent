"""Behavior contracts for signal-aware context compaction (#57875)."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(model="test", quiet_mode=True)


def _response(content: str = "summary") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("From now on, use JSON output.", "decision"),
        ("Going forward, keep citations inline.", "decision"),
        ("We decided to use approach B.", "decision"),
        ("I prefer deterministic tests.", "decision"),
        ("Please always preserve the original quote.", "decision"),
        ("Use Postgres rather than SQLite.", "correction"),
        ("Use approach B, not approach A.", "correction"),
        ("Actually, switch to approach B.", "correction"),
        ("Do not modify generated files.", "correction"),
        ("Configure timeout to 30 seconds.", "configuration"),
        ("以后必须保留原始引用。", "decision"),
        ("我们决定采用方案 B。", "decision"),
        ("把输出格式改成 JSON。", "correction"),
    ],
)
def test_explicit_signal_matrix(text: str, kind: str):
    signal = ContextCompressor._high_signal_kind(text)
    assert signal is not None
    assert signal[0] == kind


@pytest.mark.parametrize(
    "text",
    [
        "Can you inspect the logs?",
        "Which option should we choose?",
        "Do you prefer JSON?",
        "Why is this always failing?",
        "The docs mention a default timeout.",
        "Actually, what happened here?",
        "Please compare approach A and approach B.",
        "Use the tool. This is not a decision.",
        "The command returned two rows.",
    ],
)
def test_ordinary_or_ambiguous_turns_are_not_promoted(text: str):
    assert ContextCompressor._high_signal_kind(text) is None


def test_selects_explicit_user_signals_and_excludes_tool_noise():
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "Can you inspect the logs?"},
        {"role": "assistant", "content": "I will inspect them."},
        {"role": "tool", "content": "x" * 20_000},
        {"role": "user", "content": "Use Postgres rather than SQLite."},
        {
            "role": "user",
            "content": "Actually, do not use Postgres; switch to SQLite.",
        },
        {"role": "user", "content": "以后必须保留原始引用。"},
    ]

    anchors = compressor._select_high_signal_user_anchors(turns, token_budget=800)

    assert [kind for _index, kind, _text in anchors] == [
        "correction",
        "correction",
        "decision",
    ]
    rendered = compressor._render_high_signal_user_anchors(anchors)
    assert "Can you inspect" not in rendered
    assert "x" * 100 not in rendered
    assert "Use Postgres rather than SQLite." in rendered
    assert "Actually, do not use Postgres; switch to SQLite." in rendered
    assert "以后必须保留原始引用。" in rendered


def test_anchor_budget_prefers_newer_correction_over_older_preference():
    compressor = _compressor()
    turns = [
        {
            "role": "user",
            "content": "I prefer the first approach. " + "old " * 120,
        },
        {
            "role": "user",
            "content": "Actually, switch to the second approach. " + "new " * 120,
        },
    ]
    correction = compressor._select_high_signal_user_anchors(
        [turns[1]], token_budget=800
    )
    assert len(correction) == 1
    exact_budget = estimate_messages_tokens_rough(
        [{"role": "user", "content": correction[0][2]}]
    )

    anchors = compressor._select_high_signal_user_anchors(
        turns,
        token_budget=exact_budget,
    )

    assert len(anchors) == 1
    assert anchors[0][1] == "correction"
    assert "second approach" in anchors[0][2]


def test_anchor_budget_reserves_latest_signal_before_older_higher_tier():
    compressor = _compressor()
    turns = [
        {
            "role": "user",
            "content": "Actually, do not use backend A. " + "old " * 120,
        },
        {
            "role": "user",
            "content": "Configure the backend to B. " + "new " * 120,
        },
    ]
    latest = compressor._select_high_signal_user_anchors(
        [turns[1]], token_budget=800
    )
    assert len(latest) == 1
    exact_budget = estimate_messages_tokens_rough(
        [{"role": "user", "content": latest[0][2]}]
    )

    anchors = compressor._select_high_signal_user_anchors(
        turns,
        token_budget=exact_budget,
    )

    assert len(anchors) == 1
    assert anchors[0][1] == "configuration"
    assert "backend to B" in anchors[0][2]


def test_anchor_text_is_redacted_bounded_and_json_quoted():
    compressor = _compressor()
    secret = "".join(("ghp", "_", "x" * 32))
    turns = [
        {
            "role": "user",
            "content": (
                "prefix " * 180
                + "Actually, always use the safe endpoint with token "
                + secret
                + ". "
                + "suffix " * 180
            ),
        }
    ]

    anchors = compressor._select_high_signal_user_anchors(turns, token_budget=800)
    rendered = compressor._render_high_signal_user_anchors(anchors)

    assert len(anchors) == 1
    assert len(anchors[0][2]) <= 800
    assert anchors[0][2].startswith("...[anchor truncated]...")
    assert anchors[0][2].endswith("...[anchor truncated]...")
    assert "Actually, always use the safe endpoint" in anchors[0][2]
    assert secret not in rendered
    assert "..." in rendered
    assert '"' in rendered


def test_first_compaction_prompt_prioritizes_anchors_without_extra_llm_call():
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "Please inspect the implementation."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "From now on, prefer deterministic tests."},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(),
    ) as mock_call:
        compressor._generate_summary(turns)

    assert mock_call.call_count == 1
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "HIGH-SIGNAL USER ANCHORS FROM THE COMPACTED WINDOW" in prompt
    assert "From now on, prefer deterministic tests." in prompt
    assert "later correction or decision supersedes an earlier one" in prompt


def test_default_anchor_budget_keeps_one_max_sized_correction():
    compressor = _compressor()
    turns = [
        {
            "role": "user",
            "content": (
                "context " * 70
                + "Actually, switch to approach B instead. "
                + "detail " * 70
            ),
        }
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(),
    ) as mock_call:
        compressor._generate_summary(turns)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "HIGH-SIGNAL USER ANCHORS" in prompt
    assert "Actually, switch to approach B instead." in prompt


def test_iterative_compaction_carries_previous_summary_and_new_correction():
    compressor = _compressor()
    compressor._previous_summary = "Existing decision: use approach A."
    turns = [
        {"role": "user", "content": "Actually, use approach B instead."},
        {"role": "assistant", "content": "Understood."},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("Updated summary"),
    ) as mock_call:
        compressor._generate_summary(turns)

    assert mock_call.call_count == 1
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:\nExisting decision: use approach A." in prompt
    assert "Actually, use approach B instead." in prompt
    assert "[correction]" in prompt


def test_compress_wires_middle_window_decision_into_summary_prompt():
    compressor = _compressor()
    compressor.protect_first_n = 0
    compressor.protect_last_n = 2
    compressor.tail_token_budget = 120
    turns = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "We decided to keep the SQLite backend."},
        {"role": "assistant", "content": "Decision recorded."},
    ]
    turns.extend(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"ordinary context {index} " + "detail " * 80,
        }
        for index in range(12)
    )

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("Compacted summary"),
    ) as mock_call:
        compressed = compressor.compress(
            turns,
            current_tokens=90_000,
            focus_topic="current implementation",
            force=True,
        )

    assert mock_call.call_count == 1
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "HIGH-SIGNAL USER ANCHORS" in prompt
    assert "We decided to keep the SQLite backend." in prompt
    assert len(compressed) < len(turns)


def test_ordinary_user_turns_do_not_add_anchor_block():
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "What did the command output?"},
        {"role": "assistant", "content": "It returned two rows."},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(),
    ) as mock_call:
        compressor._generate_summary(turns)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "HIGH-SIGNAL USER ANCHORS" not in prompt


def test_deterministic_fallback_preserves_bounded_user_decisions():
    compressor = _compressor()
    summary = compressor._build_static_fallback_summary(
        [
            {"role": "user", "content": "We decided to keep SQLite."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Actually, switch to Postgres instead."},
        ],
        reason="provider unavailable",
    )

    assert "## Key Decisions" in summary
    assert "[decision] We decided to keep SQLite." in summary
    assert "[correction] Actually, switch to Postgres instead." in summary
