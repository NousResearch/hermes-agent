"""Tests for gateway /compress user-facing messaging."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/compress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_db = None
    # Merged runtime-resolution path (_resolve_session_agent_runtime) consults
    # per-session /model overrides before falling back to _resolve_gateway_model.
    # With no override present the tests' patched _resolve_gateway_model wins;
    # stub the override machinery so the real GatewayRunner instance
    # (object.__new__, no __init__) doesn't AttributeError / return a mock model.
    runner._session_model_overrides = {}
    runner._rehydrate_session_model_override = MagicMock(return_value=None)
    runner._session_key_for_source = MagicMock(
        return_value=build_session_key(_make_source())
    )
    return runner


@pytest.mark.asyncio
async def test_compress_command_reports_noop_without_success_banner():
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    def _chat_est(messages, **_kwargs):
        return 100  # chat size unchanged

    def _full_est(messages, **_kwargs):
        return 500  # full request size

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_chat_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_full_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "No changes from compression" in result
    assert "Compressed:" not in result
    # Chat line uses the unchanged form and excludes-system framing.
    assert "Chat size: ~100 tokens (unchanged" in result
    assert "excludes system, tools, tool results" in result
    # Full line still shown (estimate, no live turn yet).
    assert "Full request size: ~500 → ~500 tokens" in result
    assert "includes chat, system, tools, tool results" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_explains_when_token_estimate_rises():
    history = _make_history()
    compressed = [
        history[0],
        {"role": "hermes", "content": "Dense summary that still counts as more tokens."},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _chat_est(messages, **_kwargs):
        # chat rises despite fewer messages → denser-summary note
        return 100 if messages != compressed else 120

    def _full_est(messages, **_kwargs):
        return 500 if messages != compressed else 480

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_chat_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_full_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed: 4 → 3 messages" in result
    assert "Chat size: ~100 → ~120 tokens" in result
    assert "Full request size: ~500 → ~480 tokens" in result
    assert "denser summaries" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_full_line_uses_real_measured_before_when_available():
    """When the session has a real, provider-measured last_prompt_tokens (the
    same number /usage shows), the Full request size line must use it as the
    'before' WITHOUT a ~ prefix (it's real, not an estimate), while the
    'after' stays an estimate. The Chat size line is the separate chat-only
    figure. This keeps the two metrics distinct and labelled."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "hermes", "content": "Dense summary."},
        history[-1],
    ]
    runner = _make_runner(history)
    # Real provider-measured context from a prior live turn.
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 290_310
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor.context_length = 1_000_000
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _chat_est(messages, **_kwargs):
        return 29_521 if messages != compressed else 27_300

    def _full_est(messages, **_kwargs):
        # full-request estimate; the real before should override this number
        return 295_000 if messages != compressed else 265_000

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_chat_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_full_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Chat line = chat-only estimate.
    assert "Chat size: ~29,521 → ~27,300 tokens" in result
    assert "excludes system, tools, tool results" in result
    # Full line = REAL before (no ~) → estimate after.
    assert "Full request size: 290,310 → ~265,000 tokens" in result
    assert "before = last live request" in result
    # The char-based full-request *before* estimate must NOT be used when a
    # real count exists.
    assert "295,000" not in result


@pytest.mark.asyncio
async def test_compress_command_full_line_is_estimate_when_no_measured_value():
    """With no prior live turn (last_prompt_tokens == 0) the Full request size
    line falls back to the char-based estimate for both before and after,
    flagged with ~ and an 'estimated' note."""
    history = _make_history()
    compressed = [history[0], {"role": "hermes", "content": "s"}, history[-1]]
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 0  # no live turn yet
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor.context_length = 1_000_000
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _chat_est(messages, **_kwargs):
        return 100 if messages != compressed else 80

    def _full_est(messages, **_kwargs):
        return 500 if messages != compressed else 400

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_chat_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_full_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Chat size: ~100 → ~80 tokens" in result
    assert "Full request size: ~500 → ~400 tokens" in result
    assert "estimated — no live request yet" in result
    assert "last live request" not in result


@pytest.mark.asyncio
async def test_compress_command_appends_warning_when_compression_aborts():
    """When the auxiliary summariser fails and the compressor ABORTS (returns
    messages unchanged), /compress must append a visible ⚠️ warning to its
    reply telling the user nothing was dropped and how to retry. Otherwise
    the failure is silently logged and the user has no idea why nothing
    happened."""
    history = _make_history()
    # Abort path: compressor returns the input messages unchanged.
    compressed = list(history)
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Simulate compression aborting (force=True bypassed cooldown but the
    # aux LLM is genuinely broken).
    agent_instance.context_compressor._last_compress_aborted = True
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 100
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # A clearly-marked warning must be appended.
    assert "⚠️" in result
    assert "Compression aborted" in result
    # Underlying error must surface so users can fix their config.
    assert "404 model not found" in result
    # User must be told nothing was dropped — the whole point of the
    # new behavior is no silent data loss.
    assert "No messages were dropped" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_surfaces_aux_model_failure_even_when_recovered():
    """When the user's configured ``auxiliary.compression.model`` errors out
    but compression recovers by retrying on the main model, /compress must
    STILL inform the user.  Silent recovery hides broken config the user
    needs to fix."""
    history = _make_history()
    # Compressed transcript — normal successful compression, no placeholder.
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary via main model"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Fallback placeholder was NOT used — recovery succeeded.
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    # But the configured aux model DID fail before the retry succeeded.
    agent_instance.context_compressor._last_aux_model_failure_model = (
        "gemini-3-flash-preview"
    )
    agent_instance.context_compressor._last_aux_model_failure_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Compression succeeded
    assert "Compressed:" in result
    # No ⚠️ warning (that's reserved for dropped-turns case)
    assert "⚠️" not in result
    # But there IS an info note about the broken aux model
    assert "ℹ️" in result
    assert "gemini-3-flash-preview" in result
    assert "404" in result
    assert "auxiliary.compression.model" in result
    # The user's context is explicitly called out as intact
    assert "intact" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_passes_session_db_and_persists_rotated_session():
    """session_db must be wired into the /compress temp agent so that
    _compress_context can actually rotate the session and persist the
    compressed transcript — without it compression is a silent no-op."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent_cls,
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    mock_agent_cls.assert_called_once()
    assert mock_agent_cls.call_args.kwargs["session_db"] is runner._session_db
    runner.session_store._save.assert_called_once()
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-2", compressed
    )
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tool-heavy transcript honesty (2026-07-02: "No changes" over a 453K→32K line)
# ---------------------------------------------------------------------------


def _make_tool_heavy_history() -> list[dict]:
    """A stored transcript shaped like a real tool-heavy session: chat turns
    plus tool-result rows and a contentless assistant tool-call turn — the
    rows the gateway /compress chat-only projection excludes."""
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "BIG TOOL OUTPUT " * 50, "tool_call_id": "t1"},
        {"role": "tool", "content": "MORE TOOL OUTPUT " * 50, "tool_call_id": "t2"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _tool_heavy_chat(history: list[dict]) -> list[dict]:
    return [
        {"role": m.get("role"), "content": m.get("content")}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]


@pytest.mark.asyncio
async def test_compress_command_tool_heavy_noop_chat_reports_compaction():
    """CASE A regression: chat-only compression no-ops (chat already compact)
    but the transcript rewrite drops the stored tool/system rows. The reply
    must report the compaction — with the dropped-rows detail — instead of
    the self-contradicting 'No changes ... 453,542 → ~32,036'."""
    history = _make_tool_heavy_history()
    chat = _tool_heavy_chat(history)
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 453_542  # real provider-measured
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        # Chat no-op, but the session ROTATES → transcript rewrite happens.
        agent_instance.session_id = "sess-2"
        return list(messages), ""

    agent_instance._compress_context.side_effect = _compress

    def _chat_est(messages, **_kwargs):
        # chat-only rows small; non-chat rows big
        if all(m.get("role") in {"user", "assistant"} for m in messages):
            return 100
        return 420_000  # the non-chat (tool) rows

    def _full_est(messages, **_kwargs):
        return 453_000 if len(messages) == len(history) else 32_000

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_chat_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_full_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Headline reports the stored-transcript compaction, never "No changes".
    assert "No changes" not in result
    assert "Compacted stored transcript: 7 → 4 messages" in result
    # Chat axis honestly reported as already compact.
    assert "already compact, kept verbatim" in result
    # Dropped rows accounted: 3 non-chat rows (1 contentless + 2 tool).
    assert "Dropped: 3 stored tool/system messages" in result
    assert "420,000 tokens reclaimed" in result
    # Full line uses the REAL before (no ~) and the compressed-basis after.
    assert "Full request size: 453,542 → ~32,000 tokens" in result
    # Rewrite happened → stored token count reset.
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )


@pytest.mark.asyncio
async def test_compress_command_true_noop_preserves_measured_tokens():
    """CASE C regression: when NO rewrite happens (no rotation, in-place off)
    the reply must say the transcript was preserved, the full-request line
    must say 'unchanged' (the next request resends the same context — an
    'after' measured over the chat-only list would fabricate a shrink), and
    last_prompt_tokens must NOT be zeroed (it is still the only real
    provider-measured figure)."""
    history = _make_tool_heavy_history()
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 453_542
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"  # never rotates → no rewrite

    def _compress(messages, *_args, **_kwargs):
        return list(messages), ""

    agent_instance._compress_context.side_effect = _compress

    def _est(messages, **_kwargs):
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_est),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_est),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Honest no-op: transcript preserved, with full composition.
    assert "No changes: transcript preserved (7 messages: 4 chat + 3 tool/system)" in result
    # No dropped-rows claim, no fabricated shrink.
    assert "Dropped:" not in result
    assert "reclaimed" not in result
    assert "Full request size: 453,542 tokens (unchanged" in result
    # The real provider-measured count survives for the next /compress or /usage.
    runner.session_store.update_session.assert_not_called()
    # Transcript must not have been overwritten either.
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_compress_command_renders_granular_breakdown_on_real_compression():
    """When the rewrite happened AND the compressor actually changed the
    transcript, /compress renders the full granular CompactionStats block
    (Messages / Context / Removed-buckets with the tool sub-split) — the same
    renderer the auto-compaction announce uses — instead of the two-line form.
    """
    history = _make_tool_heavy_history()
    chat = _tool_heavy_chat(history)
    # Real compression shape: first chat row kept + 1 summary + last row kept.
    compressed = [
        dict(chat[0]),
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary of older turns"},
        dict(chat[-1]),
    ]
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 453_542
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor.name = "builtin"
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.compression_in_place = False
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        agent_instance.session_id = "sess-2"  # rotation → rewrite
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k", "provider": "test-prov"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Granular block present: Messages axis + wire Context line + removed buckets.
    assert "Messages:" in result
    # The summary row must be classified as summary, not "kept chat"
    # (built-in SUMMARY_PREFIX marker → kept 2 recent chat + 1 summary).
    assert "kept 2 recent chat + 1 summary" in result
    # Wire-first (Ace 2026-07-02): with a REAL provider-measured before-count
    # (453,542), the prominent token line is the WIRE story — measured before →
    # next-request estimate — NOT the archive estimate.
    assert "Context:   453,542 → ~" in result
    assert "before measured, after next-request estimate" in result
    # The archive totals are demoted into the Removed header, explicitly labeled
    # token-est so they can't be read as request-size savings.
    assert "Removed from stored transcript" in result
    assert "token-est reclaimed from archive" in result
    # The old stand-alone 'Stored transcript:' prominent line is gone in wire mode.
    assert "Stored transcript:" not in result
    assert "Removed from live context" not in result
    # Tool sub-split names the tool-result rows explicitly.
    assert "tool-result message" in result
    # Model line carries provider/model.
    assert "Model: test-prov/test-model" in result
    # Recovery pointer for the rotated (non-LCM) store.
    assert "previous transcript preserved: sess-1" in result
    # The duplicate Full-request line is SKIPPED — the wire story is already on
    # the Context line above (no double-reporting of 453,542).
    assert "Full request size: 453,542" not in result
    # And it must never claim "No changes".
    assert "No changes" not in result


@pytest.mark.asyncio
async def test_compress_command_granular_failure_degrades_to_two_line():
    """If the granular stats build fails, /compress must fall back to the
    two-line enhanced form — never crash, never suppress feedback."""
    history = _make_tool_heavy_history()
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 453_542
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor.name = "builtin"
    agent_instance.compression_in_place = False
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"

    chat = _tool_heavy_chat(history)
    compressed = [dict(chat[0]), {"role": "assistant", "content": "s"}, dict(chat[-1])]

    def _compress(messages, *_args, **_kwargs):
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch(
            "agent.compaction_stats.build_hygiene_stats",
            side_effect=RuntimeError("boom"),
        ),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Fell back to the enhanced two-line form.
    assert "Compressed:" in result and "stored messages" in result
    assert "Dropped: 3 stored tool/system messages" in result
    assert "Messages:" not in result  # granular block absent


@pytest.mark.asyncio
async def test_compress_command_lcm_engine_wire_first_context_line():
    """Ace's live case (2026-07-02): an LCM session whose /compress granular
    block measured the STORED transcript but labeled it 'Context:' / 'Removed
    from live context' — overstating wire savings (~689K→~37K, 'freed 651K')
    against a real 303K request. The fix: with a REAL provider-measured
    before-count, the prominent Context line becomes the WIRE story (measured
    303,201 → next-request estimate); the archive totals are demoted into the
    Removed header as token-est; and the duplicate Full-request line is skipped.
    One number story — footer and /compress can no longer disagree.
    """
    history = _make_tool_heavy_history()
    chat = _tool_heavy_chat(history)
    compressed = [
        dict(chat[0]),
        {"role": "assistant", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary of older turns"},
        dict(chat[-1]),
    ]
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    session_entry.last_prompt_tokens = 303_201  # real provider-measured (the wire truth)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor.name = "lcm"  # ← LCM engine (Ace's case)
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.compression_in_place = True
    agent_instance._last_compaction_in_place = True  # LCM compacts in place
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        return compressed, ""  # in-place: session_id unchanged, rewrite still fires

    agent_instance._compress_context.side_effect = _compress

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k", "provider": "test-prov"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
    ):
        result = await runner._handle_compress_command(_make_event())

    # E2E proof — print the full delivered message.
    print("\n───── delivered /compress message (LCM) ─────\n" + result + "\n────────────────────────────────────────────")

    # Wire-first: the prominent token line is the WIRE story with the REAL
    # measured before (303,201), not the archive estimate.
    assert "Context:   303,201 → ~" in result
    assert "before measured, after next-request estimate" in result
    # Archive totals demoted into the Removed header, labeled token-est.
    assert "Removed from stored transcript" in result
    assert "token-est reclaimed from archive" in result
    # The replacement-cost line, when present, carries the stored-basis wording.
    if "Replacement cost" in result:
        assert "kept in transcript" in result
    # The old stand-alone 'Stored transcript:' line and the misleading live-wire
    # wording must both be absent from the manual wire-first path.
    assert "Stored transcript:" not in result
    assert "Removed from live context" not in result
    assert "kept in context" not in result
    # The duplicate Full-request line is SKIPPED — the wire truth (303,201) is
    # already the Context line above; no double-reporting.
    assert "Full request size: 303,201" not in result
    # LCM recovery pointer.
    assert "lcm.db" in result
    assert "No changes" not in result
