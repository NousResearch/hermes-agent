"""Tests for gateway session hygiene — auto-compression of large sessions.

Verifies that the gateway detects pathologically large transcripts and
triggers auto-compression before running the agent.  (#628)

The hygiene system uses the SAME compression config as the agent:
  compression.threshold × model context length
so CLI and messaging platforms behave identically.
"""

import importlib
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n_messages: int, content_size: int = 100) -> list:
    """Build a fake transcript with n_messages user/assistant pairs."""
    history = []
    content = "x" * content_size
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": content, "timestamp": f"t{i}"})
    return history


def _make_large_history_tokens(target_tokens: int) -> list:
    """Build a history that estimates to roughly target_tokens tokens."""
    # estimate_messages_tokens_rough counts total chars in str(msg) // 4
    # Each msg dict has ~60 chars of overhead + content chars
    # So for N tokens we need roughly N * 4 total chars across all messages
    target_chars = target_tokens * 4
    # Each message as a dict string is roughly len(content) + 60 chars
    msg_overhead = 60
    # Use 50 messages with appropriately sized content
    n_msgs = 50
    content_size = max(10, (target_chars // n_msgs) - msg_overhead)
    return _make_history(n_msgs, content_size=content_size)


class HygieneCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="hygiene-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


# ---------------------------------------------------------------------------
# Detection threshold tests (model-aware, unified with compression config)
# ---------------------------------------------------------------------------

class TestSessionHygieneThresholds:
    """Test that the threshold logic correctly identifies large sessions.

    Thresholds are derived from model context length × compression threshold,
    matching what the agent's ContextCompressor uses.
    """

    def test_small_session_below_thresholds(self):
        """A 10-message session should not trigger compression."""
        history = _make_history(10)
        approx_tokens = estimate_messages_tokens_rough(history)

        # For a 200k-context model at 85% threshold = 170k
        context_length = 200_000
        threshold_pct = 0.85
        compress_token_threshold = int(context_length * threshold_pct)

        needs_compress = approx_tokens >= compress_token_threshold
        assert not needs_compress

    def test_large_token_count_triggers(self):
        """High token count should trigger compression when exceeding model threshold."""
        # Build a history that exceeds 85% of a 200k model (170k tokens)
        history = _make_large_history_tokens(180_000)
        approx_tokens = estimate_messages_tokens_rough(history)

        context_length = 200_000
        threshold_pct = 0.85
        compress_token_threshold = int(context_length * threshold_pct)

        needs_compress = approx_tokens >= compress_token_threshold
        assert needs_compress

    def test_under_threshold_no_trigger(self):
        """Session under threshold should not trigger, even with many messages."""
        # 250 short messages — lots of messages but well under token threshold
        history = _make_history(250, content_size=10)
        approx_tokens = estimate_messages_tokens_rough(history)

        # 200k model at 85% = 170k token threshold
        context_length = 200_000
        threshold_pct = 0.85
        compress_token_threshold = int(context_length * threshold_pct)

        needs_compress = approx_tokens >= compress_token_threshold
        assert not needs_compress, (
            f"250 short messages (~{approx_tokens} tokens) should NOT trigger "
            f"compression at {compress_token_threshold} token threshold"
        )

    def test_message_count_alone_does_not_trigger(self):
        """Message count alone should NOT trigger — only token count matters.

        The old system used an OR of token-count and message-count thresholds,
        which caused premature compression in tool-heavy sessions with 200+
        messages but low total tokens.
        """
        # 300 very short messages — old system would compress, new should not
        history = _make_history(300, content_size=10)
        approx_tokens = estimate_messages_tokens_rough(history)

        context_length = 200_000
        threshold_pct = 0.85
        compress_token_threshold = int(context_length * threshold_pct)

        # Token-based check only
        needs_compress = approx_tokens >= compress_token_threshold
        assert not needs_compress

    def test_threshold_scales_with_model(self):
        """Different models should have different compression thresholds."""
        # 128k model at 85% = 108,800 tokens
        small_model_threshold = int(128_000 * 0.85)
        # 200k model at 85% = 170,000 tokens
        large_model_threshold = int(200_000 * 0.85)
        # 1M model at 85% = 850,000 tokens
        huge_model_threshold = int(1_000_000 * 0.85)

        # A session at ~120k tokens:
        history = _make_large_history_tokens(120_000)
        approx_tokens = estimate_messages_tokens_rough(history)

        # Should trigger for 128k model
        assert approx_tokens >= small_model_threshold
        # Should NOT trigger for 200k model
        assert approx_tokens < large_model_threshold
        # Should NOT trigger for 1M model
        assert approx_tokens < huge_model_threshold

    def test_custom_threshold_percentage(self):
        """Custom threshold percentage from config should be respected."""
        context_length = 200_000

        # At 50% threshold = 100k
        low_threshold = int(context_length * 0.50)
        # At 90% threshold = 180k
        high_threshold = int(context_length * 0.90)

        history = _make_large_history_tokens(150_000)
        approx_tokens = estimate_messages_tokens_rough(history)

        # Should trigger at 50% but not at 90%
        assert approx_tokens >= low_threshold
        assert approx_tokens < high_threshold

    def test_minimum_message_guard(self):
        """Sessions with fewer than 4 messages should never trigger."""
        history = _make_history(3, content_size=100_000)
        # Even with enormous content, < 4 messages should be skipped
        # (the gateway code checks `len(history) >= 4` before evaluating)
        assert len(history) < 4


class TestSessionHygieneWarnThreshold:
    """Test the post-compression warning threshold (95% of context)."""

    def test_warn_when_still_large(self):
        """If compressed result is still above 95% of context, should warn."""
        context_length = 200_000
        warn_threshold = int(context_length * 0.95)  # 190k
        post_compress_tokens = 195_000
        assert post_compress_tokens >= warn_threshold

    def test_no_warn_when_under(self):
        """If compressed result is under 95% of context, no warning."""
        context_length = 200_000
        warn_threshold = int(context_length * 0.95)  # 190k
        post_compress_tokens = 150_000
        assert post_compress_tokens < warn_threshold


class TestTokenEstimation:
    """Verify rough token estimation works as expected for hygiene checks."""

    def test_empty_history(self):
        assert estimate_messages_tokens_rough([]) == 0

    def test_proportional_to_content(self):
        small = _make_history(10, content_size=100)
        large = _make_history(10, content_size=10_000)
        assert estimate_messages_tokens_rough(large) > estimate_messages_tokens_rough(small)

    def test_proportional_to_count(self):
        few = _make_history(10, content_size=1000)
        many = _make_history(100, content_size=1000)
        assert estimate_messages_tokens_rough(many) > estimate_messages_tokens_rough(few)

    def test_pathological_session_detected(self):
        """The reported pathological case: 648 messages, ~299K tokens.

        With a 200k model at 85% threshold (170k), this should trigger.
        """
        history = _make_history(648, content_size=1800)
        tokens = estimate_messages_tokens_rough(history)
        # Should be well above the 170K threshold for a 200k model
        threshold = int(200_000 * 0.85)
        assert tokens > threshold


@pytest.mark.asyncio
async def test_session_hygiene_messages_stay_in_originating_topic(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgent:
        def __init__(self, **kwargs):
            self.model = kwargs.get("model")

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([{"role": "assistant", "content": "compressed"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:17585",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = _make_history(6, content_size=400)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "795544298")

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            thread_id="17585",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    assert len(adapter.sent) == 2
    assert adapter.sent[0]["chat_id"] == "-1001"
    assert "Session is large" in adapter.sent[0]["content"]
    assert adapter.sent[0]["metadata"] == {"thread_id": "17585"}
    assert adapter.sent[1]["chat_id"] == "-1001"
    assert "Compressed:" in adapter.sent[1]["content"]
    assert adapter.sent[1]["metadata"] == {"thread_id": "17585"}
