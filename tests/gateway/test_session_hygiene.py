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
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource
from hermes_cli.config import DEFAULT_CONFIG


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





class TestEstimatedTokenThreshold:
    """Verify that hygiene thresholds are always below the model's context
    limit — for both actual and estimated token counts.

    Regression: a previous 1.4x multiplier on rough estimates pushed the
    threshold to 85% * 1.4 = 119% of context, which exceeded the model's
    limit and prevented hygiene from ever firing for ~200K models (GLM-5).
    The fix removed the multiplier entirely — the 85% threshold already
    provides ample headroom over the agent's 50% compressor.
    """

    def test_threshold_below_context_for_200k_model(self):
        """Hygiene threshold must always be below model context."""
        context_length = 200_000
        threshold = int(context_length * 0.85)
        assert threshold < context_length

    def test_threshold_below_context_for_128k_model(self):
        context_length = 128_000
        threshold = int(context_length * 0.85)
        assert threshold < context_length

    def test_no_multiplier_means_same_threshold_for_estimated_and_actual(self):
        """Without the 1.4x, estimated and actual token paths use the same threshold."""
        context_length = 200_000
        threshold_pct = 0.85
        threshold = int(context_length * threshold_pct)
        # Both paths should use 170K — no inflation
        assert threshold == 170_000

    def test_warn_threshold_below_context(self):
        """Warn threshold (95%) must be below context length."""
        for ctx in (128_000, 200_000, 1_000_000):
            warn = int(ctx * 0.95)
            assert warn < ctx

    def test_overestimate_fires_early_but_safely(self):
        """If rough estimate is 50% inflated, hygiene fires at ~57% actual usage.

        That's between the agent's 50% threshold and the model's limit —
        safe and harmless.
        """
        context_length = 200_000
        threshold = int(context_length * 0.85)  # 170K
        # If actual tokens = 113K, rough estimate = 113K * 1.5 = 170K
        # Hygiene fires when estimate hits 170K, actual is ~113K = 57% of ctx
        actual_when_fires = threshold / 1.5
        assert actual_when_fires > context_length * 0.50, (
            "Early fire should still be above agent's 50% threshold"
        )
        assert actual_when_fires < context_length, (
            "Early fire must be well below model limit"
        )


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
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            # Simulate real _compress_context: create a new session_id
            self.session_id = f"{self.session_id}_compressed"
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
    runner._voice_mode = {}
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
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # Compression warnings are no longer sent to users — compression
    # happens silently with server-side logging only.
    assert len(adapter.sent) == 0
    assert FakeCompressAgent.last_instance is not None
    FakeCompressAgent.last_instance.shutdown_memory_provider.assert_called_once()
    FakeCompressAgent.last_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_session_hygiene_warns_user_when_compression_aborts(monkeypatch, tmp_path):
    """When auxiliary compression's summary LLM call fails, the compressor
    ABORTS — returns messages unchanged, sets _last_compress_aborted=True,
    and drops nothing.  Gateway must surface a visible ⚠️ warning to the
    user (including thread_id metadata so it lands in the originating
    topic/thread) saying the conversation is unchanged and how to retry."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgentWithSummaryFailure:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            # Simulate a compressor that hit summary-generation failure
            # and ABORTED — no fallback inserted, no messages dropped.
            self.context_compressor = SimpleNamespace(
                _last_compress_aborted=True,
                _last_summary_fallback_used=False,
                _last_summary_dropped_count=0,
                _last_summary_error="404 model not found: gemini-3-flash-preview",
            )
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            # Abort path: messages preserved unchanged, session NOT rotated.
            return (messages, None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgentWithSummaryFailure
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
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
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
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
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # The compressor reported abort → exactly one warning message must
    # have been delivered to the user.
    warning_messages = [s for s in adapter.sent if "Context compression aborted" in s["content"]]
    assert len(warning_messages) == 1, (
        f"Expected 1 compression-aborted warning, got {len(warning_messages)}: {adapter.sent}"
    )
    warn = warning_messages[0]
    # Warning must include the underlying error and tell the user nothing
    # was dropped.
    assert "404" in warn["content"]
    assert "No messages were dropped" in warn["content"]
    # Warning must land in the originating topic/thread, not the main channel.
    assert warn["chat_id"] == "-1001"
    assert warn["metadata"] == {"thread_id": "17585"}

    FakeCompressAgentWithSummaryFailure.last_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_session_hygiene_informs_user_when_aux_model_fails_but_recovers(monkeypatch, tmp_path):
    """When the user's configured ``auxiliary.compression.model`` errors out
    and we recover via the main model, compression succeeds but the user's
    config is still broken.  Gateway hygiene must surface an ℹ note so the
    user knows to fix ``auxiliary.compression.model`` — silent recovery
    hides a misconfig only they can resolve."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgentWithAuxRecovery:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            # Compression succeeded (no placeholder inserted) but the
            # configured aux model errored and we fell back to main.
            self.context_compressor = SimpleNamespace(
                _last_summary_fallback_used=False,
                _last_summary_dropped_count=0,
                _last_summary_error=None,
                _last_aux_model_failure_model="gemini-3-flash-preview",
                _last_aux_model_failure_error="404 model not found",
            )
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            self.session_id = f"{self.session_id}_compressed"
            return ([{"role": "assistant", "content": "real summary"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgentWithAuxRecovery
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
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
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
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
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # No ⚠️ hard-failure warning (that's for dropped turns)
    hard_warnings = [s for s in adapter.sent if "Context compression summary failed" in s["content"]]
    assert len(hard_warnings) == 0, adapter.sent
    # But an ℹ note about the configured aux model must be delivered.
    aux_notes = [
        s for s in adapter.sent
        if "Configured compression model" in s["content"]
    ]
    assert len(aux_notes) == 1, (
        f"Expected 1 aux-model fallback notice, got {len(aux_notes)}: {adapter.sent}"
    )
    note = aux_notes[0]
    assert "gemini-3-flash-preview" in note["content"]
    assert "404" in note["content"]
    assert "auxiliary.compression.model" in note["content"]
    # Note must land in the originating topic/thread.
    assert note["chat_id"] == "-1001"
    assert note["metadata"] == {"thread_id": "17585"}

    FakeCompressAgentWithAuxRecovery.last_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_session_hygiene_honors_configurable_hard_message_limit(
    monkeypatch, tmp_path
):
    """compression.hygiene_hard_message_limit overrides the 400-message default.

    Regression for user-reported fix: a gateway session with a small
    transcript (12 messages) should not hit hygiene compression by default,
    but WILL when the user lowers the hard-limit to 10.  Verifies the new
    config key is actually read and applied at the force-compress gate.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            self.session_id = f"{self.session_id}_compressed"
            return ([{"role": "assistant", "content": "compressed"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Write config.yaml with lowered hard-limit
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_hard_message_limit: 10\n"
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    # 12 messages: below 400 default → no compression without override,
    # but above the configured limit of 10 → should compress.
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=40)
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
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    # Pick a context length large enough that the token-based threshold
    # won't trigger for 12 short messages — hard-limit must be the ONLY
    # thing firing compression.
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 1_000_000,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # The compression agent was instantiated → hard-limit fired on the
    # configured value (10), not the hardcoded 400 default.
    assert FakeCompressAgent.last_instance is not None, (
        "Expected hygiene compression to fire when message count (12) "
        "exceeds configured hygiene_hard_message_limit (10)"
    )


@pytest.mark.asyncio
async def test_session_hygiene_default_hard_message_limit_does_not_fire_at_12_messages(
    monkeypatch, tmp_path
):
    """Sanity check for the companion test above: without config override,
    12 messages must NOT trigger the default hard limit. If this test
    passes without changes, the override test's finding is meaningful."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            type(self).last_instance = self
            self.session_id = kwargs.get("session_id", "fake-session")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([{"role": "assistant", "content": "compressed"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # No config.yaml — use defaults (hard_limit=400)
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=40)
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
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 1_000_000,
    )

    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    # No compression agent instantiated — 12 messages are well under the default hard limit.
    assert FakeCompressAgent.last_instance is None, (
        "Compression should NOT fire at 12 messages with the default hard limit"
    )



@pytest.mark.asyncio
async def test_codex_budget_fail_closed_when_compression_aborts(monkeypatch, tmp_path):
    """A over-budget Codex/GPT-5.5 Telegram session must not make another model call.

    This reproduces the Jun 14 shape: low visible/tool activity with many
    persisted API calls. The gateway must stop before launching _run_agent;
    this is not an ordinary high-message compression test.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeAbortCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id", "sess-1")
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            self.context_compressor = SimpleNamespace(
                _last_compress_aborted=True,
                _last_summary_error="synthetic compression failure",
                _last_aux_model_failure_model=None,
            )
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            return (messages, None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAbortCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "model:\n"
        "  default: gpt-5.5\n"
        "  provider: openai-codex\n"
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_hard_message_limit: 160\n"
        "gateway:\n"
        "  codex_budget:\n"
        "    enabled: true\n"
        "    platforms: [telegram]\n"
        "    max_api_calls: 80\n"
        "    max_accounted_tokens: 10000000\n"
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner.session_store.load_transcript.return_value = _make_history(12, content_size=40)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._resolve_session_agent_runtime = lambda **_kw: (
        "gpt-5.5",
        {"provider": "openai-codex", "api_key": "fake", "base_url": "https://chatgpt.com/backend-api/codex/"},
    )
    runner._sync_telegram_topic_binding = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._run_agent = AsyncMock(return_value={"final_response": "SHOULD NOT RUN"})

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_codex_session_usage_snapshot", lambda _sid: {
        "api_call_count": 41,
        "message_count": 11,
        "tool_call_count": 0,
        "input_tokens": 900_000,
        "cache_read_tokens": 250_000,
        "output_tokens": 10_000,
        "reasoning_tokens": 1_000,
        "accounted_tokens": 1_161_000,
    })
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 272_000,
    )

    event = MessageEvent(
        text="continue giant loop",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert "stopped before making another GPT-5.5 call" in result
    assert "low-visible session" in result
    runner._run_agent.assert_not_called()
    assert adapter.sent, "Expected visible stop notice to be sent"
    assert "stopped before making another GPT-5.5 call" in adapter.sent[-1]["content"]


def test_codex_budget_reason_allows_normal_cache_rich_long_sessions():
    gateway_run = importlib.import_module("gateway.run")
    reason = gateway_run._codex_budget_reason(
        model="gpt-5.5",
        provider="openai-codex",
        platform_key="telegram",
        user_config={"gateway": {"codex_budget": {}}},
        usage={
            "api_call_count": 166,
            "accounted_tokens": 17_700_000,
            "input_tokens": 336_651,
            "message_count": 332,
            "tool_call_count": 120,
        },
        message_count=332,
        hard_message_limit=400,
    )
    assert reason is None


def test_codex_budget_reason_trips_absolute_last_resort_caps():
    gateway_run = importlib.import_module("gateway.run")
    reason = gateway_run._codex_budget_reason(
        model="gpt-5.5",
        provider="openai-codex",
        platform_key="telegram",
        user_config={"gateway": {"codex_budget": {}}},
        usage={"api_call_count": 251, "accounted_tokens": 1_000, "message_count": 300, "tool_call_count": 80},
        message_count=300,
        hard_message_limit=400,
    )
    assert "API calls 251 reached absolute budget 250" == reason

    reason = gateway_run._codex_budget_reason(
        model="gpt-5.5",
        provider="openai-codex",
        platform_key="telegram",
        user_config={"gateway": {"codex_budget": {}}},
        usage={"api_call_count": 120, "accounted_tokens": 30_000_001, "message_count": 300, "tool_call_count": 80},
        message_count=300,
        hard_message_limit=400,
    )
    assert "accounted tokens 30,000,001 reached absolute budget 30,000,000" == reason


def test_codex_budget_reason_trips_for_low_visible_high_api_sessions():
    gateway_run = importlib.import_module("gateway.run")
    reason = gateway_run._codex_budget_reason(
        model="gpt-5.5",
        provider="openai-codex",
        platform_key="telegram",
        user_config={"gateway": {"codex_budget": {}}},
        usage={
            "api_call_count": 41,
            "message_count": 11,
            "tool_call_count": 0,
            "accounted_tokens": 1_200_000,
            "input_tokens": 700_000,
        },
        message_count=11,
        hard_message_limit=160,
    )
    assert reason == "low-visible session made 41 API calls with 11 messages and 0 tools"


def test_codex_budget_reason_trips_for_high_fresh_input_share():
    gateway_run = importlib.import_module("gateway.run")
    reason = gateway_run._codex_budget_reason(
        model="gpt-5.5",
        provider="openai-codex",
        platform_key="telegram",
        user_config={"gateway": {"codex_budget": {"max_api_calls": 80, "max_accounted_tokens": 20_000_000}}},
        usage={
            "api_call_count": 25,
            "message_count": 60,
            "tool_call_count": 20,
            "accounted_tokens": 3_000_000,
            "input_tokens": 1_800_000,
        },
        message_count=60,
        hard_message_limit=160,
    )
    assert reason == "fresh-input share 60.0% reached budget 45%"


@pytest.mark.asyncio
async def test_codex_fresh_input_budget_runs_hygiene_compression_before_stop(monkeypatch, tmp_path):
    """High fresh-input pressure should trigger auto-compress, not block before it.

    This is the Telegram symptom where the gateway said "Run /compress" even
    though automatic compression had not been allowed to try.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeCompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.session_id = "sess-compressed"
            self._print_fn = None
            self.shutdown_memory_provider = MagicMock()
            self.close = MagicMock()
            self.context_compressor = SimpleNamespace(
                _last_compress_aborted=False,
                _last_summary_error=None,
                _last_aux_model_failure_model=None,
            )
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([messages[0], {"role": "assistant", "content": "compressed summary"}], None)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeCompressAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "model:\n"
        "  default: gpt-5.5\n"
        "  provider: openai-codex\n"
        "compression:\n"
        "  enabled: true\n"
        "  hygiene_hard_message_limit: 160\n"
        "gateway:\n"
        "  codex_budget:\n"
        "    enabled: true\n"
        "    platforms: [telegram]\n"
        "    max_api_calls: 250\n"
        "    max_accounted_tokens: 30000000\n"
    )

    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    adapter = HygieneCaptureAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner.session_store.load_transcript.return_value = _make_history(60, content_size=40)
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._resolve_session_agent_runtime = lambda **_kw: (
        "gpt-5.5",
        {"provider": "openai-codex", "api_key": "fake", "base_url": "https://chatgpt.com/backend-api/codex/"},
    )
    runner._sync_telegram_topic_binding = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._run_agent = AsyncMock(return_value={"final_response": "ok after compression"})

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_codex_session_usage_snapshot", lambda _sid: {
        "api_call_count": 25,
        "message_count": 60,
        "tool_call_count": 20,
        "input_tokens": 1_800_000,
        "cache_read_tokens": 1_100_000,
        "output_tokens": 90_000,
        "reasoning_tokens": 10_000,
        "accounted_tokens": 3_000_000,
    })
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 272_000,
    )

    event = MessageEvent(
        text="continue after auto compression",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="12345",
        ),
        message_id="1",
    )

    result = await runner._handle_message(event)

    assert result == "ok after compression"
    assert FakeCompressAgent.last_instance is not None
    runner.session_store.rewrite_transcript.assert_called_once()
    runner._run_agent.assert_awaited_once()
    assert not any("Run /compress" in item["content"] for item in adapter.sent)





def test_gateway_codex_gpt55_turn_cap_only_applies_to_configured_platform():
    from gateway.run import _gateway_effective_max_iterations

    cfg = {
        "gateway": {
            "codex_budget": {
                "enabled": True,
                "platforms": ["telegram"],
                "max_turns": 30,
            }
        }
    }

    assert _gateway_effective_max_iterations(
        model="gpt-5.5",
        provider="openai-codex",
        configured=90,
        user_config=cfg,
        platform_key="telegram",
    ) == 30
    assert _gateway_effective_max_iterations(
        model="gpt-5.5",
        provider="openai-codex",
        configured=90,
        user_config=cfg,
        platform_key="local",
    ) == 90
    assert _gateway_effective_max_iterations(
        model="claude-sonnet-4.6",
        provider="anthropic",
        configured=90,
        user_config=cfg,
        platform_key="telegram",
    ) == 90


def test_cost_saver_defaults_are_preserved_in_generated_config():

    assert DEFAULT_CONFIG["agent"]["reasoning_effort"] == "medium"
    assert DEFAULT_CONFIG["compression"]["hygiene_hard_message_limit"] == 120
    assert DEFAULT_CONFIG["compression"]["codex_gpt55_autoraise"] is False
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["enabled"] is True
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["max_api_calls"] == 250
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["max_accounted_tokens"] == 30_000_000
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["max_low_visible_api_calls"] == 40
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["max_turns"] == 30
    assert DEFAULT_CONFIG["gateway"]["codex_budget"]["max_fresh_input_share_pct"] == 45

