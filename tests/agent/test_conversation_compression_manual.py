"""Invariants for the shared manual-/compress core (``agent/conversation_compression_manual``)."""

from __future__ import annotations

import copy
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression_manual import compress_now, parse_compress_args


def _history():
    return [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}, {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"}, {"role": "assistant", "content": "six"},
    ]


def _agent():
    agent = MagicMock()
    agent._cached_system_prompt, agent.tools, agent.context_compressor = "sys", None, None
    agent._compression_skipped_due_to_lock = None
    agent._compress_context.return_value = ([{"role": "assistant", "content": "summary"}], "")
    return agent


@pytest.mark.parametrize("raw", ["--preview", "here 1 --preview", "--dry-run keep the tests", "--aggressive --preview"])
def test_preview_leaves_history_and_agent_byte_identical(raw):
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    result = compress_now(agent, history, parse_compress_args(raw))
    assert result.status == "preview" and result.lines
    assert history == frozen and result.after_messages == frozen
    agent._compress_context.assert_not_called()


def test_compressed_result_rejoins_verbatim_tail_and_never_mutates_input():
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    result = compress_now(agent, history, parse_compress_args("here 1"))
    assert result.status == "compressed"
    assert agent._compress_context.call_args.args[0] == frozen[:4]  # head only
    assert result.after_messages[-2:] == frozen[4:]  # last exchange verbatim
    assert history == frozen
    assert agent._compress_context.call_args.kwargs["force"] is True


@pytest.mark.parametrize("surface", ["cli", "gateway", "tui", "acp"])
def test_every_surface_honours_preview_without_compressing(surface, monkeypatch):
    """The prompt-cache-breaking mutation must be gated by the same ``--preview`` on all four surfaces."""
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    if surface == "cli":
        from hermes_cli.cli_session_mixin import CLISessionMixin
        cli = CLISessionMixin.__new__(CLISessionMixin)
        cli.agent, cli.conversation_history = agent, history
        cli._manual_compress("/compress --preview")
        assert cli.conversation_history == frozen
    elif surface == "gateway":
        import asyncio
        from gateway.run import GatewayRunner
        gw = GatewayRunner.__new__(GatewayRunner)
        gw.session_store = MagicMock()
        entry = MagicMock(session_id="sid")
        gw._async_session_store = MagicMock(
            _store=gw.session_store, get_or_create_session=_coro(entry), load_transcript=_coro(history))
        gw._run_manual_compression = MagicMock(side_effect=AssertionError("must not compress"))
        event = MagicMock(); event.get_command_args.return_value = "--preview"
        reply = asyncio.run(gw._handle_compress_command_inner(event))
        assert "Preview" in reply
    elif surface == "tui":
        from tui_gateway.server import _compress_session_history
        session = {"agent": agent, "history": history, "history_lock": threading.Lock(), "history_version": 3}
        assert _compress_session_history(session, "--preview")[0] == 0
        assert session["history"] == frozen and session["history_version"] == 3
    else:
        from acp_adapter.commands import SlashCommandsMixin
        acp = SlashCommandsMixin.__new__(SlashCommandsMixin)
        acp.session_manager = MagicMock()
        state = MagicMock(history=history, agent=agent, session_id="acp-sid")
        assert "Preview" in acp._cmd_compress("--preview", state)
        assert state.history == frozen
        acp.session_manager.save_session.assert_not_called()
    agent._compress_context.assert_not_called()


def test_windowless_gate_is_gateway_only_so_in_process_surfaces_still_reach_compress_context():
    """``has_content_to_compress`` only knows the local summary window; codex_app_server native compaction
    and the phase-1 tool-result prune inside ``_compress_context`` do useful work without one, so CLI/TUI/
    ACP (the default) must not short-circuit on it. The gateway keeps its historical early answer."""
    agent, history = _agent(), _history()
    agent.context_compressor = MagicMock()
    agent.context_compressor.has_content_to_compress.return_value = False
    agent._compress_context.return_value = (history[:-1], "")  # e.g. a pruned tool result, no summary

    default = compress_now(agent, history, parse_compress_args(""))
    assert default.status == "compressed" and default.removed == 1
    agent._compress_context.assert_called_once()

    agent._compress_context.reset_mock()
    gateway = compress_now(agent, history, parse_compress_args(""), system_message="", skip_without_window=True)
    assert gateway.status == "nothing_to_do" and gateway.after_messages == history
    agent._compress_context.assert_not_called()


def _coro(value):
    async def _inner(*_a, **_k):
        return value
    return _inner


@pytest.fixture
def session_db(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _exchanges(n, *, unanswered=()):
    history = []
    for i in range(n):
        history.append({"role": "user", "content": f"question {i} about fruit{i} " + " ".join(["filler"] * 40)})
        if i not in unanswered:
            history.append({"role": "assistant", "content": f"answer {i} " + " ".join(["lorem"] * 400)})
    return history


def _stored_agent(db, history, *, create=True):
    """A real AIAgent (default in-place mode) over ``history`` as stored rows, loaded back like the gateway does."""
    if create:
        db.create_session("sid", "telegram", model="test/model")
    for message in history:
        db.append_message("sid", message["role"], message["content"])
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True, skip_memory=True)
    agent._compression_feasibility_checked = True
    return agent, db.get_messages_as_conversation("sid")


def _compress_here(agent, history, keep):
    return _compress(agent, history, f"here {keep}")


def _compress(agent, history, raw):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nNumbered fruit questions.\n## Progress\nEarly ones answered."
    with patch("agent.context_compressor.call_llm", lambda **_kw: response):
        return compress_now(agent, history, parse_compress_args(raw), system_message="", skip_without_window=True)


def _flags(db, content):
    """``(active, compacted)`` of every stored row with this exact content."""
    rows = db._conn.execute("SELECT active, compacted FROM messages WHERE session_id = 'sid' AND content = ?",
                            (content,)).fetchall()
    return sorted(tuple(row) for row in rows)


def _live(messages):
    return [(m.get("role"), m.get("content")) for m in messages]


def test_in_place_here_n_stores_the_kept_exchanges(session_db):
    """The in-place commit archives every row under the lease watermark, the kept tail's included, so it must
    store the tail again after the head: otherwise a resume (and the gateway's next turn) loses exactly the
    exchanges the user asked to keep."""
    history = _exchanges(10)
    agent, loaded = _stored_agent(session_db, history)
    frozen = copy.deepcopy(loaded)
    assert agent.compression_in_place is True
    result = _compress_here(agent, loaded, 2)
    assert result.status == "compressed" and agent.session_id == "sid"
    assert loaded == frozen

    durable = session_db.get_messages_as_conversation("sid")
    assert _live(durable) == _live(result.after_messages)
    assert _live(durable[-4:]) == _live(history[-4:])
    model_history, display_history = session_db.get_resume_conversations("sid")
    for kept in history[-4:]:
        assert [m["content"] for m in model_history].count(kept["content"]) == 1
    # compress() carries its own recent rows after the summary, then comes the kept tail: each has one live
    # row, and its original is a superseded duplicate, not a turn summarized away that resume shows again.
    summary_at = next(i for i, m in enumerate(durable) if "Numbered fruit questions" in m["content"])
    carried = [m["content"] for m in durable[summary_at + 1:]]
    assert len(carried) > 4
    for content in carried:
        assert [m["content"] for m in display_history].count(content) == 1
        assert _flags(session_db, content) == [(0, 0), (1, 0)]

    # The kept rows are stamped as stored, so the next persist appends only the new turn.
    next_turn = [*result.after_messages, {"role": "user", "content": "question 10 about fruit10"}]
    agent._flush_messages_to_session_db(next_turn, None)
    assert _live(session_db.get_messages_as_conversation("sid")) == _live(next_turn)


def test_in_place_here_n_folds_the_seam_once(session_db):
    """A head ending on a user turn folds the tail's first message into it; the stored transcript carries the
    same fold, and the tail is not rejoined a second time."""
    history = _exchanges(10, unanswered={7})
    agent, loaded = _stored_agent(session_db, history)
    result = _compress_here(agent, loaded, 2)
    assert result.status == "compressed"

    durable = session_db.get_messages_as_conversation("sid")
    assert _live(durable) == _live(result.after_messages)
    assert f"{history[-5]['content']}\n\n{history[-4]['content']}" in [m["content"] for m in durable]
    assert _live(durable[-3:]) == _live(history[-3:])


FOREIGN_TURN = [("user", "[from Telegram] the vault code is 7741"), ("assistant", "Noted: the vault code is 7741.")]


@pytest.mark.parametrize("raw", ["", "here 2"])
def test_in_place_compress_keeps_turns_the_caller_never_held(session_db, raw):
    """A surface compacts the history it holds (Desktop/TUI and CLI /compress), and another surface may have
    appended turns to the same session since. The archive runs under state.db's newest row, so unless it stops at
    the caller's newest row it takes those turns away unseen: the summarizer never read them, and they leave every
    surface's history and search."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]  # what a resume restores: row ids included
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)

    assert _compress(agent, held, raw).status == "compressed"

    durable = session_db.get_messages_as_conversation("sid")
    assert _live(durable[-2:]) == FOREIGN_TURN
    assert len({m["content"] for m in durable}) == len(durable)  # kept (`here 2`) and carried rows once each
    model_history, display_history = session_db.get_resume_conversations("sid")
    for _role, content in FOREIGN_TURN:
        assert [m["content"] for m in model_history].count(content) == 1
        assert [m["content"] for m in display_history].count(content) == 1
    assert session_db.search_messages("vault 7741")


@pytest.mark.parametrize("stale", ["compacted elsewhere", "newest rows without ids", "trailing row without any stamp"])
def test_in_place_compress_never_leaves_two_live_copies_of_a_row(session_db, stale):
    """Stopping the archive at the caller's newest row is only exact while the held history is a live prefix of the
    session. After another surface compacted it, every live row is newer than the held (now archived) ones; a
    durable row held without its row id may sit above the stop; and a trailing row held with neither a row id nor
    the persisted marker can still be durable under the lease watermark (the TUI model-switch marker is appended
    to history and written with a bare ``append_message``). Either way the rows above the stop would be cloned
    beside their own copies in the new transcript."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    if stale == "compacted elsewhere":
        other, _ = _stored_agent(session_db, [], create=False)
        assert _compress(other, session_db.get_messages_as_conversation("sid"), "").status == "compressed"
    elif stale == "trailing row without any stamp":
        marker = "[Model switched to test/other.]"
        held.append({"role": "user", "content": marker, "display_kind": "model_switch"})
        session_db.append_message("sid", "user", marker, display_kind="model_switch")
    else:
        for message in held[-2:]:
            message.pop("_row_id")
        for role, content in FOREIGN_TURN:
            session_db.append_message("sid", role, content)

    assert _compress(agent, held, "").status == "compressed"

    contents = [m["content"] for m in session_db.get_messages_as_conversation("sid")]
    assert len(contents) == len(set(contents))


@pytest.mark.parametrize("raw", ["", "here 2"])
def test_in_place_compress_never_clones_a_row_a_merge_already_carried(session_db, raw):
    """Resume repair merges consecutive user rows into the first one's dict: that dict keeps its row id, drops the
    persisted marker, and the later row's id leaves the held history. Stopping the archive at the newest held id
    would then leave the later row above the stop, cloned as a concurrent append beside the merged row that
    already carries its content. Two strings, so the exact-duplicate check above cannot see it. With ``here 2``
    the merged row is the newest row of the verbatim tail, whose marker-swept copy must not vouch for its id."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    session_db.append_message("sid", "user", "first half of a split prompt")
    session_db.append_message("sid", "user", "second half 9931")
    held = session_db.get_resume_conversations("sid")[0]  # the resume path merges the two user rows
    assert "second half 9931" in held[-1]["content"] and held[-1]["content"] != "second half 9931"

    assert _compress(agent, held, raw).status == "compressed"

    contents = [m["content"] for m in session_db.get_messages_as_conversation("sid")]
    assert sum("second half 9931" in c for c in contents) == 1


def test_in_place_compress_keeps_a_gap_below_the_newest_held_row(session_db):
    """A surface can hold rows 1–N, miss another surface's rows, then persist its own later rows.
    The newest held id is then the lease watermark, so a cap at that id archives the unseen gap.
    Those rows were never summarized; they must stay live, once each."""
    from agent.context_compressor import _DB_PERSISTED_MARKER

    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)
    own_id = session_db.append_message("sid", "user", "continued on this surface")
    held.append({
        "role": "user", "content": "continued on this surface",
        "_row_id": own_id, _DB_PERSISTED_MARKER: True,
    })

    assert _compress(agent, held, "").status == "compressed"

    model_history, display_history = session_db.get_resume_conversations("sid")
    for _role, content in FOREIGN_TURN:
        assert any(content in (m.get("content") or "") for m in model_history)
        assert any(content in (m.get("content") or "") for m in display_history)
        assert _flags(session_db, content) == [(0, 0), (1, 0)]
    assert session_db.search_messages("vault 7741")


def test_in_place_compress_keeps_foreign_rows_above_an_unpersisted_turn(session_db):
    """Turn preflight appends the current user message before compression and persists it afterward,
    so the last dict has no row id. That must not fall back to the lease watermark and archive turns
    another surface appended since this process loaded."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)
    held.append({"role": "user", "content": "this turn is not persisted yet"})

    assert _compress(agent, held, "").status == "compressed"

    model_history, _display = session_db.get_resume_conversations("sid")
    for _role, content in FOREIGN_TURN:
        assert any(content in (m.get("content") or "") for m in model_history)
        assert _flags(session_db, content) == [(0, 0), (1, 0)]
    assert session_db.search_messages("vault 7741")
