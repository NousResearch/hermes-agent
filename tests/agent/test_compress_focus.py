"""Tests for focus_topic flowing through the compressor.

Verifies that _generate_summary and compress accept and use the focus_topic
parameter correctly.  Inspired by Claude Code's /compact <focus>.
"""

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
        {"role": "assistant", "content": "The schema has tables: users, orders, products."},
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


def test_compress_passes_focus_to_generate_summary():
    """compress() passes focus_topic through to _generate_summary."""
    compressor = _make_compressor()

    # Track what _generate_summary receives
    received_kwargs = {}
    original_generate = compressor._generate_summary

    def tracking_generate(turns, **kwargs):
        received_kwargs.update(kwargs)
        return "## Goal\nTest."

    compressor._generate_summary = tracking_generate

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply2"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "reply3"},
        {"role": "user", "content": "fourth"},
        {"role": "assistant", "content": "reply4"},
    ]

    compressor.compress(messages, current_tokens=100000, focus_topic="authentication flow")

    assert received_kwargs.get("focus_topic") == "authentication flow"


def test_compress_none_focus_by_default():
    """Auto compression derives focus_topic from recent user turns by default."""
    compressor = _make_compressor()

    received_kwargs = {}

    def tracking_generate(turns, **kwargs):
        received_kwargs.update(kwargs)
        return "## Goal\nTest."

    compressor._generate_summary = tracking_generate

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply2"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "reply3"},
        {"role": "user", "content": "fourth"},
        {"role": "assistant", "content": "reply4"},
    ]

    compressor.compress(messages, current_tokens=100000)

    focus_topic = received_kwargs.get("focus_topic")
    assert focus_topic.startswith("Recent user focus:")
    assert "- second" in focus_topic
    assert "- third" in focus_topic
    assert "- fourth" in focus_topic


def test_auto_focus_skips_context_summary_handoff():
    """Persisted handoff messages should not become the inferred focus."""
    compressor = _make_compressor()
    messages = [
        {"role": "system", "content": "System prompt"},
        {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY] stale Bybit topic",
        },
        {"role": "assistant", "content": "handoff acknowledged"},
        {"role": "user", "content": "Can OpenViking support sqlite backends?"},
        {"role": "assistant", "content": "Let's inspect that."},
        {"role": "user", "content": "Compare OpenViking postgres and sqlite options."},
        {"role": "assistant", "content": "Working on it."},
        {"role": "user", "content": "Now focus on OpenViking database support."},
        {"role": "assistant", "content": "Latest tail response"},
    ]

    focus_topic = compressor._derive_auto_focus_topic(messages)

    assert "OpenViking" in focus_topic
    assert "Bybit" not in focus_topic
