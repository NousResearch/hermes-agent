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


def _real_agent(session_db, session_id):
    """Real AIAgent on a real SessionDB; only the summariser is stubbed (no network)."""
    import os

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=session_db, session_id=session_id,
            skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = True
    agent._session_db_created = True
    agent.context_compressor.compress = lambda messages, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
        {"role": "assistant", "content": "kept reply 1"},
        {"role": "user", "content": "kept question"},
        {"role": "assistant", "content": "kept reply 2"},
    ]
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _run_real_compress(archive_raises: bool) -> bool:
    """Drive compress_context against a real SessionDB; return the published flag."""
    import tempfile
    from pathlib import Path

    from agent.conversation_compression import compress_context
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "persist.db")
        try:
            sid = "20260929_030000_persist"
            db.create_session(sid, "gateway", model="test/model")
            agent = _real_agent(db, sid)
            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)
            ]
            agent._flush_messages_to_session_db(messages)

            def _locked(*_a, **_kw):
                raise RuntimeError("database is locked (concurrent drain)")

            if archive_raises:
                with patch.object(SessionDB, "archive_and_compact", _locked):
                    compress_context(agent, messages, approx_tokens=900_000, system_message="sys")
            else:
                compress_context(agent, messages, approx_tokens=900_000, system_message="sys")
            return getattr(agent, "_last_compaction_persist_failed", None), agent
        finally:
            db.close()


def test_compressor_publishes_persist_failure_when_the_commit_raises():
    """Producer half, behaviourally: a rolled-back commit sets the flag the surfaces read."""
    flag, agent = _run_real_compress(archive_raises=True)
    assert flag is True
    assert agent._last_compaction_in_place is False


def test_compressor_clears_persist_failure_on_a_committed_compaction():
    """Control: a commit that lands must not be reported as a save failure."""
    flag, agent = _run_real_compress(archive_raises=False)
    assert flag is False
    assert agent._last_compaction_in_place is True


def test_compress_now_reports_persist_failed_and_discards_the_notification():
    """Every surface goes through compress_now: it must not call this 'compressed'."""
    from agent import conversation_compression_manual as manual

    history = _make_history()
    agent_instance = _make_agent(
        history, [history[0], {"role": "assistant", "content": "summary"}, history[-1]],
        persist_failed=True,
    )
    with patch("agent.conversation_compression.finalize_context_engine_compression_notification") as fin:
        result = manual.compress_now(agent_instance, history, manual.CompressRequest())
    assert result.status == "persist_failed"
    assert result.after_messages == history
    fin.assert_called_with(agent_instance, committed=False)
    assert "could not be saved" in "\n".join(manual.render_compress_result(result))
