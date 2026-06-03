"""Tests for acp_client.event_translator — inbound session_update normalisation."""

import acp

from acp_client.event_translator import EventTranslator


class TestMessageChunks:
    def test_agent_message_chunk_accumulates(self):
        tr = EventTranslator()
        tr.translate(acp.update_agent_message(acp.text_block("Hello ")))
        ev = tr.translate(acp.update_agent_message(acp.text_block("world")))
        assert ev["type"] == "agent_message_chunk"
        assert ev["text"] == "world"
        assert tr.message_text == "Hello world"

    def test_finalize_message_flushes_to_history(self):
        tr = EventTranslator()
        tr.translate(acp.update_agent_message(acp.text_block("done")))
        row = tr.finalize_message()
        assert row == {"role": "assistant", "content": "done"}
        assert tr.history == [{"role": "assistant", "content": "done"}]
        assert tr.message_text == ""
        # Second finalize with nothing accumulated returns None.
        assert tr.finalize_message() is None

    def test_thought_chunk_accumulates_separately(self):
        tr = EventTranslator()
        tr.translate(acp.update_agent_thought(acp.text_block("hmm")))
        assert tr.thought_text == "hmm"
        assert tr.message_text == ""

    def test_record_user_prompt(self):
        tr = EventTranslator()
        tr.record_user_prompt("do the thing")
        assert tr.history == [{"role": "user", "content": "do the thing"}]


class TestToolAndPlan:
    def test_tool_call_update_extracts_fields(self):
        tr = EventTranslator()
        update = acp.update_tool_call(
            "tc-1", title="Read a.py", kind="read", status="pending"
        )
        ev = tr.translate(update)
        assert ev["tool_call_id"] == "tc-1"
        assert ev["title"] == "Read a.py"
        assert ev["tool_kind"] == "read"
        assert ev["status"] == "pending"

    def test_plan_update_extracts_entries(self):
        tr = EventTranslator()
        update = acp.update_plan(
            [acp.plan_entry("step one", status="pending")]
        )
        ev = tr.translate(update)
        assert ev["type"] == "plan"
        assert ev["entries"][0]["content"] == "step one"
        assert ev["entries"][0]["status"] == "pending"


class TestEventSink:
    def test_on_event_sink_receives_each_event(self):
        seen = []
        tr = EventTranslator(on_event=seen.append)
        tr.translate(acp.update_agent_message(acp.text_block("a")))
        tr.translate(acp.update_tool_call("tc", title="t", kind="read"))
        assert [e["type"] for e in seen] == ["agent_message_chunk", "tool_call_update"]

    def test_sink_exception_does_not_propagate(self):
        def boom(_):
            raise RuntimeError("sink down")

        tr = EventTranslator(on_event=boom)
        # Should not raise.
        ev = tr.translate(acp.update_agent_message(acp.text_block("x")))
        assert ev["text"] == "x"
