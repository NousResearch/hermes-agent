"""TEMP probe (not for commit): /compress here N under rotation (compression.in_place: false)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression_manual import compress_now, parse_compress_args


@pytest.fixture
def session_db(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _exchanges(n):
    history = []
    for i in range(n):
        history.append({"role": "user", "content": f"question {i} about fruit{i} " + " ".join(["filler"] * 40)})
        history.append({"role": "assistant", "content": f"answer {i} " + " ".join(["lorem"] * 400)})
    return history


@pytest.mark.parametrize("raw", ["here 2", ""])
def test_probe(session_db, raw, capsys):
    history = _exchanges(10)
    session_db.create_session("sid", "telegram", model="test/model")
    for message in history:
        session_db.append_message("sid", message["role"], message["content"])
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=session_db, session_id="sid", skip_context_files=True,
                        skip_memory=True)
    agent._compression_feasibility_checked = True
    agent.compression_in_place = False
    loaded = session_db.get_messages_as_conversation("sid")

    seen = []
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nNumbered fruit questions.\n## Progress\nEarly ones answered."

    def _llm(**kw):
        seen.append(str(kw.get("messages")))
        return response

    with patch("agent.context_compressor.call_llm", _llm):
        result = compress_now(agent, loaded, parse_compress_args(raw), system_message="", skip_without_window=True)

    def short(ms):
        return [(m.get("role"), (m.get("content") or "")[:22]) for m in ms]

    out = []
    out.append("==== raw=%r status=%s sid=%s" % (raw, result.status, agent.session_id))
    out.append("summarizer saw q8=%s q9=%s calls=%d" % (
        any("question 8 about" in s for s in seen), any("question 9 about" in s for s in seen), len(seen)))
    out.append("AFTER: %r" % short(result.after_messages))
    out.append("CHILD durable: %r" % short(session_db.get_messages_as_conversation(agent.session_id)))
    out.append("PARENT durable n: %d" % len(session_db.get_messages_as_conversation("sid")))
    nxt = [*result.after_messages, {"role": "user", "content": "question 10 next"}]
    agent._flush_messages_to_session_db(nxt, None)
    out.append("CHILD after next flush: %r" % short(session_db.get_messages_as_conversation(agent.session_id)))
    name = "119962-probe-%s.txt" % ("here" if raw else "full")
    with open("/Users/john/.claude/scheduled-tasks/p0-pr-watch/tmp/" + name, "w") as fh:
        fh.write("\n".join(out) + "\n")
