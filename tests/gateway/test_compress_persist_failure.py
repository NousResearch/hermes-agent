"""`/compress` must not report success for a compaction that was never saved.

`compress_context()` can produce a compacted transcript in memory and then fail
to persist it — a locked/contended `state.db`, an FK error, ENOSPC. When the
rotation is rolled back internally the agent's `session_id` is left *unchanged*,
which is the same surface signature as a genuine "nothing to compress" no-op.

The gateway's `/compress` handler distinguishes rotation (`session_id` moved)
from in-place compaction (`_last_compaction_in_place`) — but a rolled-back
persist is neither, and fell through to the generic summary path, which
compares the in-memory `compressed` list against the input and cheerfully
reports `Compressed: N → M messages` for a compaction that never reached disk.
The next request resends the original context.
"""

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


def _make_runner(history):
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
    return runner


def _make_agent(history, compressed, *, persist_failed):
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Rotation was rolled back: session_id is UNCHANGED and compaction was not
    # in-place — the exact surface signature of a genuine no-op.
    agent_instance.session_id = "sess-1"
    agent_instance._last_compaction_in_place = False
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False
    # Explicit non-failure defaults: a MagicMock attribute is truthy, which
    # would otherwise fabricate an unrelated summary-failure note.
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance._last_compaction_persist_failed = persist_failed
    return agent_instance


async def _run(runner, agent_instance):
    def _estimate(messages, **_kwargs):
        return 100 if len(messages) == 4 else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        return await runner._handle_compress_command(_make_event())


@pytest.mark.asyncio
async def test_persist_failure_is_not_reported_as_a_successful_compression():
    """The headline must not claim a compaction that was never persisted."""
    history = _make_history()
    compressed = [history[0], {"role": "assistant", "content": "summary"}, history[-1]]
    runner = _make_runner(history)
    agent_instance = _make_agent(history, compressed, persist_failed=True)

    result = await _run(runner, agent_instance)

    assert "Compressed:" not in result, (
        "reported a successful compaction for a transcript that was never saved"
    )


@pytest.mark.asyncio
async def test_persist_failure_is_not_reported_as_a_benign_noop():
    """It must also not be laundered into the bland 'No changes' no-op text.

    A no-op means "there was nothing to do"; a persist failure means "there was
    something to do, we did it, and it did not save." Conflating them hides a
    retryable failure.
    """
    history = _make_history()
    compressed = [history[0], {"role": "assistant", "content": "summary"}, history[-1]]
    runner = _make_runner(history)
    agent_instance = _make_agent(history, compressed, persist_failed=True)

    result = await _run(runner, agent_instance)

    assert "No changes from compression" not in result


@pytest.mark.asyncio
async def test_persist_failure_tells_the_user_it_can_be_retried():
    """The message must name the condition and be actionable."""
    history = _make_history()
    compressed = [history[0], {"role": "assistant", "content": "summary"}, history[-1]]
    runner = _make_runner(history)
    agent_instance = _make_agent(history, compressed, persist_failed=True)

    result = await _run(runner, agent_instance)

    lowered = result.lower()
    assert "could not be saved" in lowered
    assert "/compress" in result  # actionable retry instruction
    # Reassure: nothing was lost — the original transcript is untouched.
    assert "nothing was lost" in lowered


@pytest.mark.asyncio
async def test_persist_failure_does_not_repoint_or_zero_the_stored_token_count():
    """No store mutation may follow a failed persist.

    Zeroing `last_prompt_tokens` would destroy the only tokenizer-truth figure
    for the session while the transcript is in fact unchanged.
    """
    history = _make_history()
    compressed = [history[0], {"role": "assistant", "content": "summary"}, history[-1]]
    runner = _make_runner(history)
    session_entry = runner.session_store.get_or_create_session.return_value
    agent_instance = _make_agent(history, compressed, persist_failed=True)

    await _run(runner, agent_instance)

    assert session_entry.session_id == "sess-1"
    runner.session_store.rewrite_transcript.assert_not_called()
    runner.session_store._save.assert_not_called()


@pytest.mark.asyncio
async def test_genuine_noop_still_reports_no_changes():
    """Control: a real no-op keeps its existing, correct wording.

    The flag is False and the compressor returned the input unchanged, so this
    must NOT be mistaken for a persist failure.
    """
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = _make_agent(history, list(history), persist_failed=False)

    result = await _run(runner, agent_instance)

    assert "No changes from compression" in result
    assert "could not be saved" not in result.lower()


@pytest.mark.asyncio
async def test_absent_flag_is_treated_as_no_failure():
    """Back-compat: an agent object without the attribute must not trip the path.

    Guards against a stale in-memory agent (module skew after an update) being
    reported as a persist failure on every /compress.
    """
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = _make_agent(history, list(history), persist_failed=False)
    del agent_instance._last_compaction_persist_failed

    result = await _run(runner, agent_instance)

    assert "could not be saved" not in result.lower()


def test_compressor_exposes_a_persist_failure_signal():
    """The producing half of the contract: the flag exists and defaults False.

    The gateway's report is only as honest as the signal it reads, so pin that
    `compress_context` publishes `_last_compaction_persist_failed` on the agent.
    """
    import inspect

    from agent import conversation_compression

    src = inspect.getsource(conversation_compression.compress_context)
    assert "_last_compaction_persist_failed" in src, (
        "compress_context must publish the persist-failure signal the gateway reads"
    )
