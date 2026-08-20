"""Tests for focus_topic flowing through the compressor.

Verifies that _generate_summary and compress accept and use the focus_topic
parameter correctly.  Inspired by Claude Code's /compact <focus>.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _make_compressor():
    """Create a ContextCompressor with minimal state for testing."""
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    compressor.tail_token_budget = 20000
    compressor.context_length = 200000
    compressor.threshold_percent = 0.80
    compressor.threshold_tokens = 160000
    compressor.summary_target_ratio = 0.20
    compressor.max_summary_tokens = 10000
    compressor.quiet_mode = True
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor._previous_summary = None
    compressor._ineffective_compression_count = 0
    compressor._verify_compaction_cleared_threshold = False
    compressor._summary_failure_cooldown_until = 0.0
    compressor.summary_model = None
    compressor.model = "test-model"
    compressor.provider = "test"
    compressor.base_url = "http://localhost"
    compressor.api_key = "test-key"
    compressor.api_mode = "chat_completions"
    return compressor


def test_focus_topic_injected_into_summary_prompt():
    """When focus_topic is provided, the LLM prompt includes focus guidance."""
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Tell me about the database schema"},
        {
            "role": "assistant",
            "content": "The schema has tables: users, orders, products.",
        },
    ]

    captured_prompt = {}

    def mock_call_llm(**kwargs):
        captured_prompt["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "## Goal\nUnderstand DB schema."
        return resp

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        result = compressor._generate_summary(turns, focus_topic="database schema")

    assert result is not None
    prompt_text = captured_prompt["messages"][0]["content"]
    assert 'FOCUS TOPIC: "database schema"' in prompt_text
    assert "PRIORITISE" in prompt_text
    assert "60-70%" in prompt_text


def test_no_focus_topic_no_injection():
    """Without focus_topic, the prompt doesn't contain focus guidance."""
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    captured_prompt = {}

    def mock_call_llm(**kwargs):
        captured_prompt["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "## Goal\nGreeting."
        return resp

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        result = compressor._generate_summary(turns)

    prompt_text = captured_prompt["messages"][0]["content"]
    assert "FOCUS TOPIC" not in prompt_text


def _make_auto_compressor():
    compressor = ContextCompressor(
        model="test/model",
        provider="test",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=100_000,
    )
    compressor.tail_token_budget = 10
    return compressor


def _focus_messages(prefix: str, latest_ask: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": "system"}]
    for index in range(8):
        ask = latest_ask if index == 7 else f"{prefix}_HIST_{index}"
        messages.extend([
            {"role": "user", "content": ask + " " + ("x" * 800)},
            {
                "role": "assistant",
                "content": f"{prefix}_ANSWER_{index} " + ("y" * 800),
            },
        ])
    return messages


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_auto_focus_only_uses_summarized_window():
    compressor = _make_auto_compressor()
    latest_ask = "LATEST_PROTECTED_ASK_CANARY_7F91"
    messages = _focus_messages("WINDOW", latest_ask)
    captured = {}

    def tracking_generate(turns, focus_topic=None, memory_context=""):
        captured["turns"] = turns
        captured["focus_topic"] = focus_topic
        body = "## Goal\nWindow focus probe."
        compressor._previous_summary = body
        return compressor._with_summary_prefix(body)

    compressor._generate_summary = MagicMock(side_effect=tracking_generate)
    compressor.compress(messages, current_tokens=100_000, force=True)

    summarized_text = _messages_text(captured["turns"])
    focus_topic = captured["focus_topic"]
    summarized_users = [
        str(turn.get("content", "")).split()[0]
        for turn in captured["turns"]
        if turn.get("role") == "user"
        and not compressor._is_synthetic_compression_user_turn(turn)
    ]

    assert latest_ask not in summarized_text
    assert latest_ask not in focus_topic
    assert len(summarized_users) >= 3
    assert all(item in focus_topic for item in summarized_users[-3:])


def test_auto_focus_echo_does_not_duplicate_protected_latest_ask():
    compressor = _make_auto_compressor()
    latest_ask = "LATEST_ECHO_ASK_CANARY_84B2"
    messages = _focus_messages("ECHO", latest_ask)

    def echo_focus(turns, focus_topic=None, memory_context=""):
        body = "## Goal\n" + (focus_topic or "NO_FOCUS")
        compressor._previous_summary = body
        return compressor._with_summary_prefix(body)

    compressor._generate_summary = MagicMock(side_effect=echo_focus)
    compressed = compressor.compress(messages, current_tokens=100_000, force=True)

    assert _messages_text(compressed).count(latest_ask) == 1


def test_manual_focus_overrides_implicit_focus():
    compressor = _make_auto_compressor()
    messages = _focus_messages("MANUAL", "LATEST_MANUAL_ASK_CANARY_1C6D")
    captured = {}
    manual_focus = "manual authentication flow"

    def tracking_generate(turns, focus_topic=None, memory_context=""):
        captured["focus_topic"] = focus_topic
        body = "## Goal\nManual focus probe."
        compressor._previous_summary = body
        return compressor._with_summary_prefix(body)

    compressor._generate_summary = MagicMock(side_effect=tracking_generate)
    compressor.compress(
        messages,
        current_tokens=100_000,
        focus_topic=manual_focus,
        force=True,
    )

    assert captured["focus_topic"] == manual_focus


def test_auto_focus_userless_window_is_none():
    compressor = _make_auto_compressor()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "protected head user"},
    ]
    for index in range(8):
        call_id = f"call_{index}"
        messages.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "probe", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": f"tool result {index} " + ("r" * 800),
                "tool_call_id": call_id,
            },
        ])
    messages.extend([
        {"role": "user", "content": "TAIL_REAL_USER_CANARY " + ("z" * 800)},
        {"role": "assistant", "content": "tail reply " + ("q" * 800)},
    ])
    captured = {}

    def tracking_generate(turns, focus_topic=None, memory_context=""):
        captured["turns"] = turns
        captured["focus_topic"] = focus_topic
        body = "## Goal\nUser-less window probe."
        compressor._previous_summary = body
        return compressor._with_summary_prefix(body)

    compressor._generate_summary = MagicMock(side_effect=tracking_generate)
    compressed = compressor.compress(messages, current_tokens=100_000, force=True)

    assert all(turn.get("role") != "user" for turn in captured["turns"])
    assert captured["focus_topic"] is None
    assert len(compressed) < len(messages)


def test_iterative_auto_focus_ignores_prior_summary_and_latest_tail():
    compressor = _make_auto_compressor()
    focuses = []

    def prefixed_echo(turns, focus_topic=None, memory_context=""):
        focuses.append(focus_topic or "")
        body = f"## Goal\nROUND_{len(focuses)}_SUMMARY_SENTINEL\n" + (
            focus_topic or "NO_FOCUS"
        )
        compressor._previous_summary = body
        return compressor._with_summary_prefix(body)

    compressor._generate_summary = MagicMock(side_effect=prefixed_echo)
    first_messages = _focus_messages("ROUND1", "ROUND1_LATEST_CANARY_A91F")
    first_compressed = compressor.compress(
        first_messages,
        current_tokens=100_000,
        force=True,
    )

    latest_second_ask = "ROUND2_LATEST_CANARY_B27E"
    second_messages = list(first_compressed)
    for index in range(8):
        ask = latest_second_ask if index == 7 else f"ROUND2_HIST_{index}"
        second_messages.extend([
            {"role": "user", "content": ask + " " + ("m" * 800)},
            {
                "role": "assistant",
                "content": f"ROUND2_ANSWER_{index} " + ("n" * 800),
            },
        ])

    second_compressed = compressor.compress(
        second_messages,
        current_tokens=100_000,
        force=True,
    )

    assert len(focuses) == 2
    assert latest_second_ask not in focuses[1]
    assert "ROUND_1_SUMMARY_SENTINEL" not in focuses[1]
    assert _messages_text(second_compressed).count(latest_second_ask) == 1
