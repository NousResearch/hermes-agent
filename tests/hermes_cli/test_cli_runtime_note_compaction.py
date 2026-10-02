"""CLI runtime notes must expire at real compaction boundaries (#124170).

Drive staging and the worker entry synchronously, without the terminal UI or a
provider tool loop. The worker harness calls production turn setup and compression;
only the summary LLM is stubbed. SQLite and memory-provider dispatch remain real.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _SUMMARY_END_MARKER
from agent.conversation_compression import compress_context
from agent.memory_manager import MemoryManager
from agent.turn_context import build_turn_context, substitute_api_content
from hermes_state import SessionDB
from tests.agent.test_compression_runtime_note_adoption import (
    NOTE, TASK, RecordingMemory, _tool_round,
)


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
def test_cli_note_expires_at_real_compaction(tmp_path, multimodal):
    from cli import HermesCLI, _ChatTurn
    from run_agent import AIAgent
    from agent import conversation_loop as loop

    clean = TASK if not multimodal else [
        {"type": "text", "text": TASK},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    original = copy.deepcopy(clean)
    db = SessionDB(tmp_path / "state.db")
    session_id = "cli-runtime-note"
    try:
        db.create_session(session_id, source="cli", model="test/model")
        # A user-named session needs no background title-generation LLM.
        db.set_session_title(session_id, "CLI compaction fixture")
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=session_id,
            skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        )
        agent._cached_system_prompt = "Stable fixture system."
        agent._compression_feasibility_checked = True
        agent.compression_in_place = True
        agent.context_compressor = ContextCompressor(
            "test/model", config_context_length=100000,
            protect_first_n=2, protect_last_n=2, quiet_mode=True,
        )
        agent.context_compressor.tail_token_budget = 500
        memory = RecordingMemory()
        agent._memory_manager = MemoryManager()
        agent._memory_manager.add_provider(memory)

        # No prompt-toolkit/config constructor: only the state these real CLI
        # entry methods need. Callback methods are real but never prompted.
        cli = object.__new__(HermesCLI)
        cli.agent = agent
        cli.provider = "test"
        cli.model = agent.model
        cli.session_id = session_id
        cli.conversation_history = []
        cli._pending_model_switch_note = NOTE
        cli._chat_stage_user_message(agent, clean)
        staged = agent._pending_cli_user_message
        assert staged is cli.conversation_history[-1]
        assert staged["content"] == original
        assert agent._persist_user_message_override is None
        captured = {}

        def run_to_compaction(**kwargs):
            captured.update(copy.deepcopy(kwargs))
            context = build_turn_context(
                agent, kwargs["user_message"], None, kwargs["conversation_history"],
                kwargs["task_id"], kwargs["stream_callback"], kwargs["persist_user_message"],
                restore_or_build_system_prompt=loop._restore_or_build_system_prompt,
                install_safe_stdio=loop._install_safe_stdio,
                sanitize_surrogates=loop._sanitize_surrogates,
                summarize_user_message_for_log=loop._summarize_user_message_for_log,
                set_session_context=loop.set_session_context,
                set_current_write_origin=loop.set_current_write_origin,
                ra=loop._ra,
            )
            assert context.messages[context.current_turn_user_idx] is staged
            assert agent._persist_user_message_override == original
            assert context.original_user_message == original
            assert NOTE in str(context.messages), "note must reach the live turn before compaction"
            early = db.get_messages_as_conversation(session_id)
            assert len(early) == 1, "CLI staging and turn setup must not duplicate the user row"
            assert TASK in str(early[0]["content"])
            assert NOTE not in str(early[0]["content"])

            messages = context.messages
            for boundary in range(2):
                # Recorded tool output grows the same task past its protected
                # tail; no real terminal/tool execution is needed for this seam.
                for i in range(30):
                    messages.extend(_tool_round(boundary * 100 + i))
                messages, _ = compress_context(
                    agent, messages, context.active_system_prompt,
                    approx_tokens=200000, force=True,
                )
                assert any(_SUMMARY_END_MARKER in str(m.get("content")) for m in messages)
                wire = copy.deepcopy(messages)
                for row in wire:
                    substitute_api_content(row)
                assert TASK in str(wire), "compaction must retain the actual request"
                assert NOTE not in str(wire), "renewed requests must not replay the CLI note"
                durable = db.get_messages_as_conversation(agent.session_id)
                assert durable, "a real durable compaction must have committed"
                for row in durable:
                    substitute_api_content(row)
                assert TASK in str(durable)
                assert NOTE not in str(durable)
            return {"messages": messages, "completed": True}

        # This replaces orchestration only, not the entry, prologue, compressor,
        # clean-content projection, DB transaction, or memory dispatch under test.
        agent.run_conversation = run_to_compaction
        response = MagicMock()
        response.choices[0].message.content = "## Summary\nFixture work is ongoing."
        turn = _ChatTurn()
        with patch("agent.context_compressor.call_llm", return_value=response) as llm:
            cli._chat_run_agent(turn, clean)

        assert turn.result.get("completed"), turn.result  # CLI catches worker exceptions
        assert cli._pending_model_switch_note is None
        assert captured["persist_user_message"] == original
        assert NOTE in str(captured["user_message"])
        assert clean == original, "note prepending must not mutate the caller's native parts"
        if multimodal:
            assert captured["user_message"][1] == original[1]
        assert llm.call_count >= 2
        assert TASK in str(llm.call_args_list), "the actual user turn must enter summary input"
        assert NOTE not in str(llm.call_args_list)
        assert len(memory.before) >= 2 and len(memory.ended) >= 2
        for handoff in (memory.before, memory.ended):
            assert TASK in str(handoff)
            assert NOTE not in str(handoff)
    finally:
        db.close()
