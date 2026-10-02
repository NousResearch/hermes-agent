"""In-place compaction's positional rewind must land on the carried tail's own originals.

``archive_and_compact`` flags the newest ``tail_count`` durable rows as the carried tail's superseded originals
(``active=0, compacted=0``: hidden from display and search). The CLI and the gateway persist a turn's user row
only after the turn-start preflight, so that row rides in the tail with no durable original of its own.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import finalize_context_engine_compression_notification
from agent.conversation_compression_manual import compress_now, parse_compress_args


def _aux_llm(**kwargs):
    text = "## Goal\nNumbered steps.\n## Progress\nEarly ones done." if kwargs.get("task") == "compression" else "Title"
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="aux", usage=None)


def _reply(content, prompt_tokens):
    message = SimpleNamespace(content=content, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model")
    response.usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=100, total_tokens=prompt_tokens + 100)
    return response


@pytest.fixture
def session(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("agent.context_compressor.call_llm", _aux_llm)
    monkeypatch.setattr("agent.title_generator.call_llm", _aux_llm)
    db = SessionDB(db_path=tmp_path / "state.db")
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True, skip_memory=True)
    agent.client, agent.tool_delay, agent.save_trajectories = MagicMock(), 0, False
    assert agent.compression_in_place is True
    yield db, agent
    db.close()


def _turn(db, agent, cli, surface, n, prompt_tokens):
    """One turn exactly as the classic CLI (staged dict, history[:-1]) or the gateway (transcript reload) runs it."""
    text = f"U{n} please continue with the next step"
    agent.client.chat.completions.create.side_effect = [_reply(f"A{n} " + "lorem ipsum dolor " * 300, prompt_tokens)]
    if surface == "gateway":
        from gateway.run import _build_gateway_agent_history

        history = _build_gateway_agent_history(db.get_messages_as_conversation("sid", repair_alternation=True))[0]
        agent.run_conversation(user_message=text, conversation_history=history, task_id="sid", persist_user_message=text)
        return
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    CLIChatTurnMixin._chat_stage_user_message(cli, agent, text)
    result = agent.run_conversation(
        user_message=text, conversation_history=cli.conversation_history[:-1], task_id="sid", persist_user_message=None)
    cli.conversation_history = result.get("messages", cli.conversation_history)


def _replies_displayed(db):
    return {m["content"].split(" ")[0] for m in db.get_resume_conversations("sid")[1]
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)}


@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_turn_start_compaction_hides_no_summarized_turn(session, surface):
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 14):
        _turn(db, agent, cli, surface, n, 5_000)
    _turn(db, agent, cli, surface, 14, 200_000)  # real usage over the threshold: the next turn compacts first
    shown = _replies_displayed(db)
    assert {f"A{n}" for n in range(1, 15)} <= shown

    _turn(db, agent, cli, surface, 15, 20_000)

    assert getattr(agent, "_last_compaction_in_place", None) is True
    assert {f"A{n}" for n in range(1, 16)} <= _replies_displayed(db)


def test_manual_compress_between_turns_rewinds_every_carried_original(session):
    """Between turns the turn anchor is the last turn's, and a gateway reload is durable but unmarked: counting its
    rows as unflushed would leave carried originals at compacted=1, so each shows (and is recalled) twice."""
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 15):
        _turn(db, agent, cli, "gateway", n, 5_000)
    assert agent._persist_user_message_idx is not None

    from gateway.run import _build_gateway_agent_history

    history = _build_gateway_agent_history(db.get_messages_as_conversation("sid", repair_alternation=True))[0]
    assert compress_now(agent, history, parse_compress_args(""), system_message="").status == "compressed"

    model_history, display_history = db.get_resume_conversations("sid")
    live = [m["content"] for m in model_history if isinstance(m.get("content"), str)]
    carried = live[next(i for i, c in enumerate(live) if "Numbered steps" in c) + 1:]
    assert carried
    display = [m["content"] for m in display_history if isinstance(m.get("content"), str)]
    assert [display.count(content) for content in carried] == [1] * len(carried)


@pytest.mark.parametrize("keep", [1, 2])
def test_compress_here_n_rewinds_every_row_behind_a_merged_kept_exchange(session, monkeypatch, keep):
    """A prompt that never got its reply is merged with the next one on reload. Kept by ``/compress here N``,
    that one dict stands for two durable rows, and the rewind must take both as carried originals."""
    db, agent = session
    cli = SimpleNamespace(conversation_history=[])
    for n in range(1, 9):
        _turn(db, agent, cli, "gateway", n, 5_000)
    db.append_message("sid", "user", "U9x this prompt never got a reply")
    for n in range(9, 9 + keep):
        _turn(db, agent, cli, "gateway", n, 5_000)
    history = db.get_resume_conversations("sid")[0]  # --resume, the TUI and Desktop hold row ids

    # The second pass runs on the list the CLI installs: the kept copies still list the ids the first archived.
    for attempt in range(2):
        def _numbered_summary(**kwargs):
            response = _aux_llm(**kwargs)
            response.choices[0].message.content += f" Pass {attempt}."
            return response

        monkeypatch.setattr("agent.context_compressor.call_llm", _numbered_summary)
        result = compress_now(agent, history, parse_compress_args(f"here {keep}"), system_message="")
        assert result.status == "compressed"
        finalize_context_engine_compression_notification(agent, committed=True)  # the caller's step
        history = result.after_messages

        live = [m["content"] for m in db.get_messages_as_conversation("sid") if isinstance(m.get("content"), str)]
        carried = live[next(i for i, c in enumerate(live) if "Numbered steps" in c) + 1:]
        recalled = [row["content"] for row in db._conn.execute(
            "SELECT content FROM messages WHERE session_id = 'sid' AND (active = 1 OR compacted = 1)").fetchall()]
        assert [recalled.count(content) for content in carried] == [1] * len(carried)
        assert [c for c in recalled if "U9x" in c or c.startswith("U9 ")] == [
            "U9x this prompt never got a reply\n\nU9 please continue with the next step"]
        # A row the rewind hides from search is a carried original: its text is still in a live row.
        superseded = [row["content"] for row in db._conn.execute(
            "SELECT content FROM messages WHERE session_id = 'sid' AND active = 0 AND compacted = 0").fetchall()]
        assert [c for c in superseded if isinstance(c, str) and not any(c in held for held in live)] == []
