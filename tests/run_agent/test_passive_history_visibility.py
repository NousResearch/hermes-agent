"""An already-created idle agent sees externally saved history on its next turn.

These run against a real ``state.db`` with a second independent writer standing in for the
passive-ingress caller. Only ``agent.conversation_loop.run_conversation`` is replaced (by a
recorder), so admission, lease handling, tip resolution and transcript loading are the real
code paths. Fail-fast sentinels sit on the provider, tool-dispatch and approval seams to prove
that saving history starts zero execution.
"""

from __future__ import annotations

import sqlite3
from copy import deepcopy

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_state import SessionDB
from hermes_state_passive_history import PassiveHistoryBusyError
from run_agent import AIAgent

OLD_USER = {"role": "user", "content": "what is on my calendar"}
OLD_ASSISTANT = {"role": "assistant", "content": "two meetings"}
SPOKEN = {"role": "user", "content": "move the second meeting to friday"}
IDENTITY = {"producer": "talk.voice", "event_id": "evt-1", "origin_turn_id": "origin-1"}


def _agent_with_db(db, *, session_id):
    """Same façade stand-in shape as tests/run_agent/test_cross_process_turn_lease.py."""
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = "desktop"
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: session_id
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None
    agent.status_callback = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    return agent


def _recorder(monkeypatch, turns):
    """Record every loop entry; the recorded history is what the provider would replay."""

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        turns.append({"history": history, "session_id": _agent.session_id})
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    return turns


@pytest.fixture
def zero_execution(monkeypatch):
    """Fail fast if persistence ever reaches inference, provider or approval machinery."""

    def forbid(seam):
        def _boom(*_args, **_kwargs):
            raise AssertionError(f"passive history must not reach {seam}")

        return _boom

    monkeypatch.setattr("agent.agent_init._build_client", forbid("provider creation"))
    monkeypatch.setattr("model_tools.handle_function_call", forbid("tool dispatch"))
    monkeypatch.setattr("tools.approval.submit_pending", forbid("approval creation"))


@pytest.fixture
def conversation(tmp_path):
    """A durable conversation with one completed exchange, plus an external writer handle."""
    path = tmp_path / "state.db"
    host, ingress = SessionDB(path), SessionDB(path)
    host.create_session("conv", source="test")
    host.append_messages_batch("conv", [dict(OLD_USER), dict(OLD_ASSISTANT)])
    try:
        yield host, ingress
    finally:
        host.close()
        ingress.close()


def _seed(db, session_id="conv"):
    """The history an already-created agent is holding from its previous turn."""
    return db.get_messages_as_conversation(session_id, repair_alternation=True, include_row_ids=True)


def test_first_turn_of_an_idle_agent_adopts_the_external_suffix_once(
    conversation, zero_execution, monkeypatch
):
    host, ingress = conversation
    seed = _seed(host)
    agent = _agent_with_db(host, session_id="conv")
    turns = _recorder(monkeypatch, [])

    # No marker attribute at all: an already-cached agent whose first admission sees a
    # nonzero external revision must still reload.
    assert not hasattr(agent, "_passive_history_watermark")
    receipt = ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)

    result = AIAgent.run_conversation(agent, "and confirm it", conversation_history=seed)

    assert result["final_response"] == "ok"
    assert len(turns) == 1
    history = turns[0]["history"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert [m["content"] for m in history].count(SPOKEN["content"]) == 1
    # Loaded rows are born durable, so the next flush cannot re-append them.
    assert all(m.get(_DB_PERSISTED_MARKER) is True for m in history)
    assert all("_row_id" in m for m in history)
    assert agent._passive_history_watermark == ("conv", receipt.revision)
    # The short passive commit did not leave a lease behind.
    with host._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM session_turn_leases").fetchone() is None


def test_unchanged_external_history_keeps_the_caller_object_and_skips_the_reload(
    conversation, zero_execution, monkeypatch
):
    host, ingress = conversation
    ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
    agent = _agent_with_db(host, session_id="conv")
    turns = _recorder(monkeypatch, [])

    AIAgent.run_conversation(agent, "first", conversation_history=_seed(host))
    loaded = turns[0]["history"]

    reloads = []
    real_load = host.get_messages_as_conversation

    def counting_load(session_id, **kwargs):
        reloads.append(session_id)
        return real_load(session_id, **kwargs)

    monkeypatch.setattr(host, "get_messages_as_conversation", counting_load)

    AIAgent.run_conversation(agent, "second", conversation_history=loaded)

    assert reloads == []
    assert turns[1]["history"] is loaded
    # Prompt-cache-relevant bytes of the already-completed prefix are untouched.
    assert turns[1]["history"][:2] == turns[0]["history"][:2]


def test_changed_compression_tip_is_adopted_before_the_turn(
    conversation, zero_execution, monkeypatch
):
    host, ingress = conversation
    seed = _seed(host)
    host.end_session("conv", "compression")
    host.create_session("conv-2", source="test", parent_session_id="conv")
    host.append_messages_batch("conv-2", [{"role": "user", "content": "summary carried forward"}])
    receipt = ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
    assert receipt.session_id == "conv-2"

    agent = _agent_with_db(host, session_id="conv")
    turns = _recorder(monkeypatch, [])

    AIAgent.run_conversation(agent, "and confirm it", conversation_history=seed)

    assert agent.session_id == "conv-2"
    assert turns[0]["session_id"] == "conv-2"
    # Replay sees the summary carrier and the new suffix exactly once. The continuation's carrier is
    # itself a user row, so the host's existing adjacent-role repair combines them for the provider;
    # neither text is dropped and the canonical raw rows stay separate on disk.
    replayed = turns[0]["history"]
    assert [m["role"] for m in replayed] == ["user"]
    assert replayed[0]["content"].count("summary carried forward") == 1
    assert replayed[0]["content"].count(SPOKEN["content"]) == 1
    assert [m["content"] for m in host.get_messages("conv-2")] == [
        "summary carried forward", SPOKEN["content"]]
    # The marker is stored per conversation, not as a bare number.
    assert agent._passive_history_watermark == ("conv", receipt.revision)


def test_reload_failure_releases_the_lease_and_the_next_turn_retries(
    conversation, zero_execution, monkeypatch
):
    host, ingress = conversation
    seed = _seed(host)
    receipt = ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
    agent = _agent_with_db(host, session_id="conv")
    turns = _recorder(monkeypatch, [])

    real_load = host.get_messages_as_conversation
    failures = {"n": 1}

    def flaky_load(session_id, **kwargs):
        with host._read_ctx() as conn:
            assert conn.execute("SELECT 1 FROM session_turn_leases").fetchone() is not None
        if failures["n"]:
            failures["n"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_load(session_id, **kwargs)

    monkeypatch.setattr(host, "get_messages_as_conversation", flaky_load)

    with pytest.raises(sqlite3.OperationalError):
        AIAgent.run_conversation(agent, "and confirm it", conversation_history=seed)

    assert turns == []
    assert not hasattr(agent, "_passive_history_watermark")
    with host._read_ctx() as conn:
        assert conn.execute("SELECT 1 FROM session_turn_leases").fetchone() is None

    AIAgent.run_conversation(agent, "and confirm it", conversation_history=seed)

    assert len(turns) == 1
    assert [m["content"] for m in turns[0]["history"]].count(SPOKEN["content"]) == 1
    assert agent._passive_history_watermark == ("conv", receipt.revision)


@pytest.mark.parametrize("marker", ["_branched_from", "_delegate_from", "_reset_from"])
def test_compressed_child_loads_its_own_live_segment(conversation, monkeypatch, marker):
    host, ingress = conversation
    host.create_session("child", source="test", parent_session_id="conv",
                        model_config={marker: "conv"})
    host.append_messages_batch("child", [dict(OLD_USER), dict(OLD_ASSISTANT)])
    seed = _seed(host, "child")
    host.end_session("child", "compression")
    host.create_session("child-tip", source="test", parent_session_id="child",
                        model_config={marker: "conv"})
    host.append_message("child-tip", "user", "child summary")
    receipt = ingress.append_passive_messages("child", messages=[dict(SPOKEN)], **IDENTITY)
    agent = _agent_with_db(host, session_id="child")
    turns = _recorder(monkeypatch, [])

    AIAgent.run_conversation(agent, "continue", conversation_history=seed)

    assert agent.session_id == turns[0]["session_id"] == receipt.session_id == "child-tip"
    assert SPOKEN["content"] in turns[0]["history"][-1]["content"]
    assert agent._passive_history_watermark == ("child", receipt.revision)
    assert host.get_messages("conv") == ingress.get_messages("conv")


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_external_suffix_preserves_cached_payloads(conversation, zero_execution, monkeypatch, role):
    host, ingress = conversation
    # Both user and assistant payload sidecars may differ from the display text.
    host.clear_messages("conv")
    host.append_messages_batch("conv", [
        {**OLD_USER, "api_content": "  original user wire bytes\n"},
        {**OLD_ASSISTANT, "api_content": "  original assistant wire bytes\n"},
    ])
    seed = _seed(host)
    before = deepcopy(seed)
    agent = _agent_with_db(host, session_id="conv")
    agent._cached_system_prompt = "the established system prompt\n"
    turns = _recorder(monkeypatch, [])
    external = {"role": role, "content": "externally finalized text"}
    ingress.append_passive_messages("conv", messages=[external], **IDENTITY)

    AIAgent.run_conversation(agent, "continue", conversation_history=seed)

    history = turns[0]["history"]
    assert history[:len(seed)] == before
    assert seed == before
    assert agent._cached_system_prompt == "the established system prompt\n"
    assert sum(m["content"].count(external["content"]) for m in history) == 1
    assert all(a["role"] != b["role"] for a, b in zip(history, history[1:]))
    assert all(m.get(_DB_PERSISTED_MARKER) for m in history)
    assert host.get_messages("conv")[-1]["role"] == role
    assert host.get_messages("conv")[-1]["content"] == external["content"]
    # A copied replay must not flush the attributed context as another canonical utterance.
    assert agent._flush_messages_to_session_db(deepcopy(history))
    assert len(host.get_messages("conv")) == 3
    if role == "assistant":
        assert "External assistant transcript" in history[-1]["content"]
    # Loading again with a new passive event preserves the same completed prefix.
    ingress.append_passive_messages("conv", messages=[dict(SPOKEN)],
                                   **{**IDENTITY, "event_id": "evt-2"})
    AIAgent.run_conversation(agent, "continue again", conversation_history=history)
    assert turns[1]["history"][:len(seed)] == before
    assert sum(m["content"].count(external["content"]) for m in turns[1]["history"]) == 1


def test_passive_commit_during_a_live_turn_is_refused_without_touching_history(
    conversation, zero_execution, monkeypatch
):
    """Delayed persistence: the active writer wins, the caller retries later."""
    host, ingress = conversation
    agent = _agent_with_db(host, session_id="conv")
    observed = {}

    def busy_turn(_agent, _message, _system, history, *_args, **_kwargs):
        with pytest.raises(PassiveHistoryBusyError) as excinfo:
            ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
        observed["retryable"] = excinfo.value.retryable
        observed["history"] = list(history)
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", busy_turn)

    AIAgent.run_conversation(agent, "typed while speaking", conversation_history=_seed(host))

    assert observed["retryable"] is True
    assert [m["content"] for m in observed["history"]] == [
        OLD_USER["content"], OLD_ASSISTANT["content"]]
    assert ingress.get_passive_history_watermark("conv").revision == 0

    # Once the turn released the lease the same event commits, still exactly once.
    receipt = ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
    assert receipt.replayed is False
    replay = ingress.append_passive_messages("conv", messages=[dict(SPOKEN)], **IDENTITY)
    assert replay.replayed is True and replay.message_ids == receipt.message_ids


def test_persisting_history_never_enters_the_conversation_loop(
    conversation, zero_execution, monkeypatch
):
    host, ingress = conversation
    loop_calls = []

    def forbidden_loop(*_args, **_kwargs):
        loop_calls.append("unrequested")
        raise AssertionError("passive persistence must not run inference")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", forbidden_loop)

    for index in range(3):
        ingress.append_passive_messages(
            "conv", messages=[dict(SPOKEN)],
            producer="talk.voice", event_id=f"evt-{index}", origin_turn_id=f"origin-{index}")
        ingress.append_passive_messages(
            "conv", messages=[dict(SPOKEN)],
            producer="talk.voice", event_id=f"evt-{index}", origin_turn_id=f"origin-{index}")

    assert loop_calls == []
    assert host.get_session("conv")["api_call_count"] == 0
    assert host.get_passive_history_watermark("conv").revision == 3

    # The only loop entry is the turn the caller explicitly asks for.
    turns = _recorder(monkeypatch, [])
    agent = _agent_with_db(host, session_id="conv")
    AIAgent.run_conversation(agent, "now act on it", conversation_history=_seed(host))
    assert len(turns) == 1
