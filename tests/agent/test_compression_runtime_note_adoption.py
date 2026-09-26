"""Runtime-note boundary contract through real durable adoption (PR #124225).

Only the auxiliary LLM is stubbed. Persistence, adoption, compression, rotation,
wire substitution, and memory-provider dispatch execute their real paths.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, _SUMMARY_END_MARKER
from agent.conversation_compression import compress_context
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.turn_context import substitute_api_content
from hermes_state import SessionDB


NOTE = "[Note: model was just switched from old to new via test. Adjust your self-identification accordingly.]"
TASK = "Continue auditing the local fixture and report the result."
CONCURRENT = "Concurrent writer evidence retained by adoption."


class RecordingMemory(MemoryProvider):
    name = "adoption-test-recorder"

    def __init__(self):
        self.before = []
        self.ended = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        self.before.append(copy.deepcopy(messages))
        return ""

    def on_session_end(self, messages):
        self.ended.append(copy.deepcopy(messages))


def _tool_round(index):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{index}", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": f"c{index}",
         "content": f"Fixture evidence {index}: " + "audited local output " * 200},
    ]


@pytest.mark.parametrize("live_tool_rounds", [0, 12], ids=["live-user-in-tail", "live-user-in-summary"])
@pytest.mark.parametrize("late_append", [False, True], ids=["stable-parent", "late-writer"])
def test_adoption_keeps_runtime_note_out_of_boundary_consumers(tmp_path, live_tool_rounds, late_append):
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    parent = "runtime-note-adoption"
    try:
        db.create_session(parent, source="cli", model="test/model")
        db.append_message(parent, "user", "Audit the fixture.")
        db.append_message(parent, "assistant", "Starting the audit.")
        for i in range(20):
            for row in _tool_round(i):
                db.append_message(parent, **row)
        messages = db.get_messages_as_conversation(parent)
        live_idx = len(messages)
        messages.append({"role": "user", "content": NOTE + "\n\n" + TASK})
        for i in range(live_tool_rounds):
            messages.extend(_tool_round(100 + i))

        # Grow the durable parent past the entire in-memory snapshot, exactly
        # the preflight adoption fixture's race, without mocking DB reads.
        for i in range(live_tool_rounds + 1):
            db.append_message(parent, "user", f"{CONCURRENT} {i}")
            db.append_message(parent, "assistant", "Recorded concurrent evidence.")
        assert len(db.get_messages_as_conversation(parent)) > len(messages)

        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=parent,
            skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        )
        agent._compression_feasibility_checked = True
        agent.compression_in_place = False
        agent._cached_system_prompt = "Stable fixture system."
        agent._persist_user_message_idx = live_idx
        agent._persist_user_message_override = TASK
        agent.context_compressor = ContextCompressor(
            "test/model", config_context_length=100000,
            protect_first_n=2, protect_last_n=2, quiet_mode=True,
        )
        agent.context_compressor.tail_token_budget = 500
        memory = RecordingMemory()
        agent._memory_manager = MemoryManager()
        agent._memory_manager.add_provider(memory)
        response = MagicMock()
        response.choices[0].message.content = "## Summary\nFixture work is ongoing."

        def summarize(**kwargs):
            if late_append:
                db.append_message(parent, "user", "LATE writer must survive the summary window.")
            return response

        with patch("agent.context_compressor.call_llm", side_effect=summarize) as llm:
            result, _ = compress_context(
                agent, messages, agent._cached_system_prompt,
                approx_tokens=200000, force=True,
            )

        # Prove real adoption, successful summary/rotation, and a naturally
        # produced sidecar; don't inject an impossible DB row or mock projection.
        parent_rows = db.get_messages_as_conversation(parent, include_inactive=True)
        current = [row for row in parent_rows if row.get("content") == TASK]
        assert len(current) == 1
        assert current[0]["api_content"] == NOTE + "\n\n" + TASK
        assert agent._persist_user_message_idx == len(parent_rows) - int(late_append)
        assert agent.session_id != parent
        assert llm.called
        assert any(_SUMMARY_END_MARKER in str(row.get("content")) for row in result)
        assert memory.before and memory.ended
        assert CONCURRENT in str(memory.before)
        assert TASK in str(memory.before) and TASK in str(memory.ended)
        if live_tool_rounds:
            assert TASK in str(llm.call_args_list), "live user must actually enter the summary window"

        wire = copy.deepcopy(result)
        for row in wire:
            substitute_api_content(row)
        child = db.get_messages_as_conversation(agent.session_id)
        if late_append:
            assert sum(row.get("content") == "LATE writer must survive the summary window." for row in child) == 1
        for row in child:
            substitute_api_content(row)
        evidence = {
            "summary_llm_input": NOTE in str(llm.call_args_list),
            "precompress_memory_provider": NOTE in str(memory.before),
            "session_end_memory_provider": NOTE in str(memory.ended),
            "returned_wire": NOTE in str(wire),
            "durable_child_wire": NOTE in str(child),
        }
        assert not any(evidence.values()), f"Runtime note leaked after durable adoption: {evidence}"
    finally:
        db.close()
