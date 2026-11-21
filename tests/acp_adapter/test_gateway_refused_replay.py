"""A safety-refused turn stays on screen when an ACP editor reopens the session over the gateway.

#149708 takes a refused turn out of the model's context and keeps the ask visible as a
``content_refused`` display row; its review (discussion_r4486245780) made the in-process ACP
agent replay that row on ``session/load``. ``hermes acp`` runs ``GatewayACPAgent``, whose replay
used to keep only the snapshot's live rows, and a refused row is archived (shown, never live).
"""
from types import SimpleNamespace

import pytest

from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_NOTICE
from tests.acp_adapter.test_gateway_compressed_reopen import _Bridge
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import async_wait_until

_REFUSED = "Explain how to pick the lock on my neighbour's front door"
_REFUSAL_DETAIL = "I can't help with breaking into someone else's property."
_ANSWER = "Hello there."
_USER, _AGENT = "user_message_chunk", "agent_message_chunk"
_ASK = (_USER, _REFUSED, True)
_NOTICE = (_AGENT, CONTENT_REFUSED_NOTICE, False)


@pytest.fixture
def model_peer():
    """The daemon's provider: it answers the first ask and refuses the second (an HTTP-200
    ``content_filter`` finish, a provider's safety block)."""
    with FakeLLMServer([Text(_ANSWER), Text(_REFUSAL_DETAIL, finish_reason="content_filter")]) as provider:
        yield SimpleNamespace(server_port=provider.port)


def _replayed_messages(frames):
    """``(kind, text, refused)`` of every message chunk an editor received, in order."""
    shown = []
    for frame in frames:
        update = (frame.get("params") or {}).get("update") or {}
        if frame.get("method") == "session/update" and update.get("sessionUpdate") in (_USER, _AGENT):
            hermes = (update.get("_meta") or {}).get("hermes") or {}
            shown.append((update["sessionUpdate"], (update.get("content") or {}).get("text"),
                          bool(hermes.get("contentRefused"))))
    return shown


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_acp_reload_over_the_gateway_replays_the_refused_ask_and_notice(daemon, tmp_path, model_peer):
    """The real wire: a refusal through an ordinary daemon, then a fresh editor process reopening it."""
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
        created = await acp.rpc("session/new", cwd=str(tmp_path), mcpServers=[])
        assert "result" in created, created
        sid = created["result"]["sessionId"]
        reply = await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": "hello"}])
        assert "result" in reply, reply
        # The gateway ends a refused turn as failed once its notice is out (-32603 admitted_turn_failed);
        # this test is about what a reopened editor shows afterwards.
        await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": _REFUSED}])

    async with viewer(daemon) as ws:
        async def retained():
            snapshot = await ws.rpc("session.resume", session_id=sid)
            return [m for m in snapshot["messages"] if m.get("display_kind") == CONTENT_REFUSED_DISPLAY_KIND]

        rows = await async_wait_until(retained, timeout=30, what="the gateway to retain the refused turn for display")
        assert [row.get("archived") for row in rows] == [True], rows

    async with editor(daemon, tmp_path) as reopened:
        await reopened.rpc("initialize", protocolVersion=1, clientCapabilities={})
        loaded = await reopened.rpc("session/load", cwd=str(tmp_path), sessionId=sid, mcpServers=[])
        assert "result" in loaded, loaded
        shown = _replayed_messages(reopened.frames)

    assert shown == [(_USER, "hello", False), (_AGENT, _ANSWER, False), _ASK, _NOTICE], shown


class _RecordingEditor:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


def _replayed_updates(updates):
    """``(kind, text, refused)`` of every message chunk the editor got, chunks flagged ``compactionSummary``
    aside."""
    shown = []
    for update in updates:
        if update.session_update not in (_USER, _AGENT):
            continue
        hermes = (update.field_meta or {}).get("hermes") or {}
        if not hermes.get("compactionSummary"):
            shown.append((update.session_update, update.content.text, bool(hermes.get("contentRefused"))))
    return shown


def _said(*texts):
    """The chunks of a plain exchange: *texts* alternate between the user and the agent."""
    return [(kind, text, False) for kind, text in zip((_USER, _AGENT) * len(texts), texts)]


def _messages(*texts):
    return [{"role": role, "content": text} for role, text in zip(("user", "assistant") * len(texts), texts)]


def _turns(db, session_id, *texts):
    for message in _messages(*texts):
        db.append_message(session_id, message["role"], message["content"])


def _refuse(db, session_id):
    _turns(db, session_id, _REFUSED, _REFUSAL_DETAIL)
    assert db.retain_refused_turn_for_display(session_id, notice=CONTENT_REFUSED_NOTICE)


def _rotate(db, session_id, *texts):
    """Compress *session_id* into a child session, ``compressed-tip``, which holds *texts* as fresh rows."""
    assert db.try_acquire_compression_lock(session_id, "fixture")
    db.publish_compression_child(parent_session_id=session_id, child_session_id="compressed-tip", source="acp",
                                 messages=_messages(*texts), compression_lock_holder="fixture")
    db.release_compression_lock(session_id, "fixture")
    return "compressed-tip"


def _compress_in_place(db, session_id):
    """Compress *session_id* in place as the compressor commits it: a fresh copy of the first message (the protected
    head), the summary, then the last exchange carried verbatim; the rows between are summarized away."""
    from agent.context_compressor import _fresh_compaction_message_copy
    held = db.get_messages_as_conversation(session_id)
    head, tail = [_fresh_compaction_message_copy(held[0])], held[-2:]
    db.archive_and_compact(session_id, [*head, {"role": "assistant", "content": "summary"}, *tail],
                           watermark=db.get_active_message_watermark(session_id), tail_count=len(tail))


def _first_ask_refused(db, session_id):
    _refuse(db, session_id)
    _turns(db, session_id, "hello", _ANSWER)


def _first_ask_refused_and_the_answer_pruned(db, session_id):
    # A prune rewrites a row in place and keeps its old self as an archived twin: no compression.
    _first_ask_refused(db, session_id)
    answer = next(m for m in db.get_resume_conversations(session_id)[0] if m.get("role") == "assistant")
    db.rewrite_pruned_rows(session_id, [(dict(answer), {**answer, "content": "[pruned]"})])


def _refused_after_a_compression(db, session_id):
    _turns(db, session_id, "before compression")
    tip = _rotate(db, session_id, "summary", "retained")
    _turns(db, tip, "hello", _ANSWER)
    _refuse(db, tip)
    _turns(db, tip, "after", "ok")


def _refused_after_an_in_place_compression(db, session_id):
    _turns(db, session_id, "hello", _ANSWER, "earlier", "noted")
    _compress_in_place(db, session_id)
    _refuse(db, session_id)
    _turns(db, session_id, "after", "ok")


def _refused_before_a_compression(db, session_id):
    _turns(db, session_id, "hello", _ANSWER)
    _refuse(db, session_id)
    _rotate(db, session_id, "summary", "retained")


def _refused_before_an_in_place_compression(db, session_id):
    # The snapshot lists the copy of the first message first, then the rows it summarized away and the refusal,
    # though those have the lower row_id.
    _turns(db, session_id, "hello", _ANSWER)
    _refuse(db, session_id)
    _turns(db, session_id, "after", "ok")
    _compress_in_place(db, session_id)


def _refused_before_a_compression_with_a_long_answer(db, session_id):
    # The resume budget keeps the newest archived rows that fit: the refusal, not the long answer before it.
    from gateway.session_local_recovery import RESUME_HISTORY_BYTES
    _turns(db, session_id, "hello", "x" * RESUME_HISTORY_BYTES)
    _refuse(db, session_id)
    _rotate(db, session_id, "summary", "retained")


def _refused_ask_carrying_a_handoff_summary(db, session_id):
    from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
    _turns(db, session_id, "before compression")
    handoff = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}"
    tip = _rotate(db, session_id, "hello", _ANSWER, f"{handoff}\n\n{_REFUSED}")
    db.append_message(tip, "assistant", _REFUSAL_DETAIL)
    assert db.retain_refused_turn_for_display(tip, notice=CONTENT_REFUSED_NOTICE)


def _refused_ask_sent_again(db, session_id):
    # The display read drops the re-sent ask as a replay of the refused one: the snapshot lists it above the refusal.
    _turns(db, session_id, "hello", _ANSWER)
    _refuse(db, session_id)
    _turns(db, session_id, _REFUSED, "A locksmith can open it for the owner.")


@pytest.mark.asyncio
@pytest.mark.usefixtures("authority_cwd_in_tmp")  # symbolic profile ids are relative homes
@pytest.mark.parametrize("history, expected", [
    pytest.param(_first_ask_refused, [_ASK, _NOTICE, *_said("hello", _ANSWER)], id="first-ask"),
    pytest.param(_first_ask_refused_and_the_answer_pruned, [_ASK, _NOTICE, *_said("hello", "[pruned]")],
                 id="first-ask-then-a-prune"),
    pytest.param(_refused_after_a_compression,
                 [*_said("summary", "retained", "hello", _ANSWER), _ASK, _NOTICE, *_said("after", "ok")],
                 id="after-a-compression"),
    pytest.param(_refused_after_an_in_place_compression,
                 [*_said("hello", "summary", "earlier", "noted"), _ASK, _NOTICE, *_said("after", "ok")],
                 id="after-an-in-place-compression"),
    pytest.param(_refused_before_a_compression, _said("summary", "retained"), id="before-a-compression"),
    pytest.param(_refused_before_an_in_place_compression, _said("hello", "summary", "after", "ok"),
                 id="before-an-in-place-compression"),
    pytest.param(_refused_before_a_compression_with_a_long_answer, _said("summary", "retained"),
                 id="before-a-compression-over-the-resume-budget"),
    pytest.param(_refused_ask_carrying_a_handoff_summary, [*_said("hello", _ANSWER), _ASK, _NOTICE],
                 id="ask-carrying-a-handoff-summary"),
    pytest.param(_refused_ask_sent_again,
                 [*_said("hello", _ANSWER), _ASK, _NOTICE, *_said(_REFUSED, "A locksmith can open it for the owner.")],
                 id="refused-ask-sent-again"),
])
async def test_acp_reload_replays_refused_turns_since_the_latest_compression(tmp_path, monkeypatch, history, expected):
    """Of the archived rows only refused ones come back, each between the turns around it (real authority
    and SQLite, real dispatch). Rows a compression summarized away stay display-only, and so does a refusal
    from before the latest compression, into a child session or in place: the compression wrote the turns it
    kept as fresh rows, so stored order would put the refusal above them. A pruned row's archived twin is no
    compression. A refused ask that a compression merged its handoff summary into replays as the ask alone,
    and an ask sent again after its refusal replays after it."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from acp_adapter.gateway_server import GatewayACPAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"platform_toolsets": {"acp": []}})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id="fixture", instance_id="owner")
    try:
        connection = AuthorityConnection(authority, object(), {"user_id": "human", "profile_id": "fixture"})
        ref = create_local_session(authority, connection.actor, {
            "request_id": "editor", "source": "acp", "cwd": str(tmp_path), "model": "frozen", "toolsets": []})
        history(authority.db, ref.session_id)

        agent = GatewayACPAgent()
        agent._home = tmp_path
        agent._gateway, agent._conn = _Bridge(connection), _RecordingEditor()
        await agent.load_session(cwd=str(tmp_path), session_id=ref.session_id, mcp_servers=[])
        shown = _replayed_updates(agent._conn.updates)
    finally:
        store.close_all_db_handles()

    assert shown == expected, shown
