from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.agent_execution_service import (
    append_missing_media_tags_to_response,
    build_gateway_btw_prompt,
    collect_history_media_paths,
    execute_gateway_sync_turn,
    finalize_gateway_agent_conversation_result,
    gateway_approval_context,
    normalize_conversation_history,
    prepend_pending_model_switch_note,
    run_gateway_approved_conversation,
    run_gateway_background_conversation,
    run_gateway_btw_conversation,
    setup_gateway_stream_consumer,
    sync_gateway_execution_session_split,
)


def test_normalize_conversation_history_preserves_tool_sequences_and_reasoning():
    history = [
        {"role": "session_meta", "content": "skip"},
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hello", "timestamp": "t1"},
        {
            "role": "assistant",
            "content": "thinking",
            "timestamp": "t2",
            "reasoning": "chain",
            "reasoning_details": [{"type": "summary", "text": "r"}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function"}],
            "content": "",
            "timestamp": "t3",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": 'ok MEDIA:/tmp/a.png MEDIA:/tmp/b.mp3"}',
            "timestamp": "t4",
        },
        {
            "role": "user",
            "content": "copied",
            "mirror": True,
            "mirror_source": "weixin",
        },
        {"role": "assistant", "content": ""},
    ]

    normalized = normalize_conversation_history(history)

    assert normalized == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "thinking",
            "reasoning": "chain",
            "reasoning_details": [{"type": "summary", "text": "r"}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function"}],
            "content": "",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": 'ok MEDIA:/tmp/a.png MEDIA:/tmp/b.mp3"}',
        },
        {"role": "user", "content": "[Delivered from weixin] copied"},
    ]


def test_collect_history_media_paths_extracts_unique_paths_from_tool_messages():
    agent_history = [
        {"role": "user", "content": "ignore"},
        {"role": "tool", "content": 'foo MEDIA:/tmp/a.png bar MEDIA:/tmp/b.mp3"}'},
        {"role": "function", "content": "MEDIA:/tmp/a.png"},
    ]

    assert collect_history_media_paths(agent_history) == {"/tmp/a.png", "/tmp/b.mp3"}


def test_gateway_approval_context_sets_and_resets_context(monkeypatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "tools.approval.set_current_session_key",
        lambda value: calls.append(("set_session", value)) or "session-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_current_session_key",
        lambda token: calls.append(("reset_session", token)),
    )
    monkeypatch.setattr(
        "tools.approval.set_current_admin_policy",
        lambda ids, flag: calls.append(("set_admin", (list(ids), flag))) or "admin-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_current_admin_policy",
        lambda token: calls.append(("reset_admin", token)),
    )
    monkeypatch.setattr(
        "tools.approval.set_external_approval_backend",
        lambda backend: calls.append(("set_backend", backend)) or "backend-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_external_approval_backend",
        lambda token: calls.append(("reset_backend", token)),
    )

    backend = object()
    with gateway_approval_context(
        session_key="agent:main:test",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        external_backend=backend,
    ):
        calls.append(("inside", None))

    assert calls == [
        ("set_session", "agent:main:test"),
        ("set_admin", (["179033731"], True)),
        ("set_backend", backend),
        ("inside", None),
        ("reset_backend", "backend-token"),
        ("reset_admin", "admin-token"),
        ("reset_session", "session-token"),
    ]


def test_prepend_pending_model_switch_note_only_when_present():
    assert prepend_pending_model_switch_note("hello", None) == "hello"
    assert prepend_pending_model_switch_note("hello", "switch note") == "switch note\n\nhello"


def test_build_gateway_btw_prompt_wraps_question_in_ephemeral_instruction():
    assert build_gateway_btw_prompt("what changed?") == (
        "[Ephemeral /btw side question. Answer using the conversation "
        "context. No tools available. Be direct and concise.]\n\n"
        "what changed?"
    )


def test_append_missing_media_tags_to_response_dedupes_and_preserves_voice_directive():
    response = append_missing_media_tags_to_response(
        "done",
        messages=[
            {"role": "tool", "content": 'MEDIA:/tmp/a.png\nMEDIA:/tmp/a.png'},
            {"role": "tool", "content": "[[audio_as_voice]]\nMEDIA:/tmp/b.ogg"},
        ],
        history_media_paths={"/tmp/a.png"},
    )

    assert response == "done\n[[audio_as_voice]]\nMEDIA:/tmp/b.ogg"


def test_sync_gateway_execution_session_split_updates_store_and_resets_offset():
    entry = SimpleNamespace(session_id="sess-old")
    session_store = SimpleNamespace(
        _entries={"key-1": entry},
        _save=MagicMock(),
    )
    logger = MagicMock()
    agent = SimpleNamespace(session_id="sess-new")

    effective_session_id, effective_history_offset = sync_gateway_execution_session_split(
        agent=agent,
        session_id="sess-old",
        session_key="key-1",
        session_store=session_store,
        agent_history_len=9,
        logger=logger,
    )

    assert effective_session_id == "sess-new"
    assert effective_history_offset == 0
    assert entry.session_id == "sess-new"
    session_store._save.assert_called_once()
    logger.info.assert_called_once()


def test_finalize_gateway_agent_conversation_result_uses_empty_fallback_and_auto_title(monkeypatch):
    called = {}

    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title",
        lambda db, sid, message, final, msgs: called.update(
            db=db,
            sid=sid,
            message=message,
            final=final,
            msgs=msgs,
        ),
    )

    agent = SimpleNamespace(
        session_id="sess-1",
        model="gpt-test",
        session_prompt_tokens=11,
        session_completion_tokens=7,
        context_compressor=SimpleNamespace(last_prompt_tokens=19),
    )
    result = finalize_gateway_agent_conversation_result(
        result={
            "final_response": "(empty)",
            "messages": [{"role": "assistant", "content": "(empty)"}],
            "api_calls": 1,
            "last_reasoning": "r",
        },
        agent=agent,
        tools=[{"name": "terminal"}],
        message="hello",
        session_id="sess-1",
        session_key="key-1",
        history_media_paths=set(),
        agent_history_len=2,
        session_store=SimpleNamespace(_entries={}, _save=MagicMock()),
        session_db=object(),
        logger=MagicMock(),
        empty_response_fallback=lambda kind: "fallback text",
    )

    assert result["final_response"] == "fallback text"
    assert result["suppress_reply"] is False
    assert result["history_offset"] == 2
    assert result["last_prompt_tokens"] == 19
    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 7
    assert result["model"] == "gpt-test"
    assert called["message"] == "hello"
    assert called["final"] == "fallback text"


def test_finalize_gateway_agent_conversation_result_resets_offset_on_session_split():
    session_store = SimpleNamespace(
        _entries={"key-1": SimpleNamespace(session_id="sess-old")},
        _save=MagicMock(),
    )
    agent = SimpleNamespace(
        session_id="sess-new",
        model="gpt-test",
        session_prompt_tokens=2,
        session_completion_tokens=3,
        context_compressor=SimpleNamespace(last_prompt_tokens=5),
    )

    result = finalize_gateway_agent_conversation_result(
        result={
            "final_response": "ok",
            "messages": [{"role": "assistant", "content": "ok"}],
            "api_calls": 1,
        },
        agent=agent,
        tools=[],
        message="hello",
        session_id="sess-old",
        session_key="key-1",
        history_media_paths=set(),
        agent_history_len=10,
        session_store=session_store,
        session_db=None,
        logger=MagicMock(),
        empty_response_fallback=lambda kind: None,
    )

    assert result["session_id"] == "sess-new"
    assert result["history_offset"] == 0
    session_store._save.assert_called_once()


def test_run_gateway_approved_conversation_prepends_pending_note_and_registers_notify(monkeypatch):
    calls: list[tuple[str, object]] = []

    @contextmanager
    def _fake_approval_context(**kwargs):
        calls.append(("approval_context", kwargs))
        yield

    monkeypatch.setattr(
        "gateway.agent_execution_service.gateway_approval_context",
        _fake_approval_context,
    )
    monkeypatch.setattr(
        "tools.approval.register_gateway_notify",
        lambda session_key, callback: calls.append(("register", session_key)) or "handle-1",
    )
    monkeypatch.setattr(
        "tools.approval.unregister_gateway_notify",
        lambda session_key, handle=None: calls.append(("unregister", (session_key, handle))),
    )

    agent = SimpleNamespace(
        run_conversation=lambda message, conversation_history=None, task_id=None: {
            "message": message,
            "conversation_history": conversation_history,
            "task_id": task_id,
        }
    )

    result = run_gateway_approved_conversation(
        agent=agent,
        message="hello",
        pending_model_note="switch note",
        conversation_history=[{"role": "user", "content": "hi"}],
        task_id="sess-1",
        session_key="key-1",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        status_adapter=SimpleNamespace(pause_typing_for_chat=lambda chat_id: None),
        status_chat_id="chat-1",
        status_thread_metadata={"thread_id": "t1"},
        loop_for_step=None,
        logger=MagicMock(),
        admin_only_message_builder=lambda action: None,
    )

    assert result["message"] == "switch note\n\nhello"
    assert result["task_id"] == "sess-1"
    assert calls[0][0] == "approval_context"
    assert calls[1] == ("register", "key-1")
    assert calls[2] == ("unregister", ("key-1", "handle-1"))


def test_run_gateway_approved_conversation_passes_external_backend(monkeypatch):
    calls: list[tuple[str, object]] = []
    backend = object()

    @contextmanager
    def _fake_approval_context(**kwargs):
        calls.append(("approval_context", kwargs))
        yield

    monkeypatch.setattr(
        "gateway.agent_execution_service.gateway_approval_context",
        _fake_approval_context,
    )
    monkeypatch.setattr(
        "tools.approval.register_gateway_notify",
        lambda session_key, callback: calls.append(("register", session_key)) or "handle-1",
    )
    monkeypatch.setattr(
        "tools.approval.unregister_gateway_notify",
        lambda session_key, handle=None: calls.append(("unregister", (session_key, handle))),
    )

    agent = SimpleNamespace(
        run_conversation=lambda message, conversation_history=None, task_id=None: {
            "message": message,
            "task_id": task_id,
        }
    )

    result = run_gateway_approved_conversation(
        agent=agent,
        message="hello",
        pending_model_note=None,
        conversation_history=None,
        task_id="bg-1",
        session_key="bg-1",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        status_adapter=None,
        status_chat_id="chat-1",
        status_thread_metadata=None,
        loop_for_step=None,
        logger=MagicMock(),
        admin_only_message_builder=lambda action: None,
        external_backend=backend,
    )

    assert result["message"] == "hello"
    assert calls[0] == (
        "approval_context",
        {
            "session_key": "bg-1",
            "admin_user_ids": ["179033731"],
            "is_admin_user": True,
            "external_backend": backend,
        },
    )


def test_run_gateway_background_conversation_creates_agent_triggers_callback_and_forwards_backend(
    monkeypatch,
):
    created_calls = {}
    approved_calls = {}
    created_agents = []
    agent = SimpleNamespace(name="bg-agent")
    backend = object()
    source = SimpleNamespace(platform=None)

    def _fake_create_gateway_agent(**kwargs):
        created_calls.update(kwargs)
        return agent

    def _fake_run_gateway_approved_conversation(**kwargs):
        approved_calls.update(kwargs)
        return {"final_response": "background done"}

    monkeypatch.setattr(
        "gateway.agent_execution_service.create_gateway_agent",
        _fake_create_gateway_agent,
    )
    monkeypatch.setattr(
        "gateway.agent_execution_service.run_gateway_approved_conversation",
        _fake_run_gateway_approved_conversation,
    )

    runtime_spec = SimpleNamespace(max_iterations=14, enabled_toolsets=["core", "web"])

    result = run_gateway_background_conversation(
        runtime_spec=runtime_spec,
        session_id="sess-bg",
        source=source,
        message="hello from bg",
        conversation_history=[{"role": "user", "content": "hi"}],
        session_key="key-bg",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        status_adapter=SimpleNamespace(),
        status_chat_id="chat-bg",
        status_thread_metadata={"thread_id": "thread-bg"},
        loop_for_step="loop-bg",
        logger=MagicMock(),
        admin_only_message_builder=lambda action: None,
        session_db="db-handle",
        external_backend=backend,
        on_agent_created=created_agents.append,
    )

    assert result == {"final_response": "background done"}
    assert created_calls == {
        "runtime_spec": runtime_spec,
        "session_id": "sess-bg",
        "source": source,
        "session_db": "db-handle",
        "max_iterations": 14,
        "quiet_mode": True,
        "verbose_logging": False,
        "enabled_toolsets": ["core", "web"],
    }
    assert created_agents == [agent]
    assert approved_calls["agent"] is agent
    assert approved_calls["message"] == "hello from bg"
    assert approved_calls["pending_model_note"] is None
    assert approved_calls["conversation_history"] == [{"role": "user", "content": "hi"}]
    assert approved_calls["task_id"] == "sess-bg"
    assert approved_calls["session_key"] == "key-bg"
    assert approved_calls["admin_user_ids"] == ["179033731"]
    assert approved_calls["is_admin_user"] is True
    assert approved_calls["status_chat_id"] == "chat-bg"
    assert approved_calls["status_thread_metadata"] == {"thread_id": "thread-bg"}
    assert approved_calls["loop_for_step"] == "loop-bg"
    assert approved_calls["external_backend"] is backend


def test_run_gateway_btw_conversation_uses_ephemeral_agent_settings(monkeypatch):
    created_calls = {}
    run_calls = {}

    def _fake_create_gateway_agent(**kwargs):
        created_calls.update(kwargs)
        return SimpleNamespace(
            run_conversation=lambda **call_kwargs: run_calls.update(call_kwargs)
            or {"final_response": "btw answer"}
        )

    monkeypatch.setattr(
        "gateway.agent_execution_service.create_gateway_agent",
        _fake_create_gateway_agent,
    )

    runtime_spec = SimpleNamespace(max_iterations=25)
    conversation_history = [{"role": "assistant", "content": "earlier"}]
    source = SimpleNamespace(platform=None)

    result = run_gateway_btw_conversation(
        runtime_spec=runtime_spec,
        session_id="sess-btw",
        source=source,
        question="what changed?",
        conversation_history=conversation_history,
    )

    assert result == {"final_response": "btw answer"}
    assert created_calls == {
        "runtime_spec": runtime_spec,
        "session_id": "sess-btw",
        "source": source,
        "max_iterations": 8,
        "enabled_toolsets": [],
        "quiet_mode": True,
        "verbose_logging": False,
        "skip_memory": True,
        "skip_context_files": True,
        "persist_session": False,
    }
    assert run_calls == {
        "user_message": build_gateway_btw_prompt("what changed?"),
        "conversation_history": conversation_history,
        "task_id": "sess-btw",
    }


def test_setup_gateway_stream_consumer_builds_consumer_and_callback(monkeypatch):
    fake_consumer = MagicMock()
    fake_consumer.on_delta = object()

    monkeypatch.setattr(
        "gateway.stream_consumer.StreamConsumerConfig",
        lambda edit_interval, buffer_threshold, cursor: {
            "edit_interval": edit_interval,
            "buffer_threshold": buffer_threshold,
            "cursor": cursor,
        },
    )
    monkeypatch.setattr(
        "gateway.stream_consumer.GatewayStreamConsumer",
        lambda adapter, chat_id, config, metadata=None: fake_consumer,
    )

    holder = [None]
    stream_consumer, stream_delta_cb = setup_gateway_stream_consumer(
        streaming_config=SimpleNamespace(
            enabled=True,
            transport="edit",
            edit_interval=0.25,
            buffer_threshold=10,
            cursor="▉",
        ),
        adapter=object(),
        chat_id="chat-1",
        thread_metadata={"thread_id": "t1"},
        stream_consumer_holder=holder,
        logger=MagicMock(),
    )

    assert stream_consumer is fake_consumer
    assert stream_delta_cb is fake_consumer.on_delta
    assert holder[0] is fake_consumer


def test_setup_gateway_stream_consumer_returns_none_when_disabled():
    holder = [None]
    stream_consumer, stream_delta_cb = setup_gateway_stream_consumer(
        streaming_config=SimpleNamespace(enabled=False, transport="off"),
        adapter=object(),
        chat_id="chat-1",
        thread_metadata=None,
        stream_consumer_holder=holder,
        logger=MagicMock(),
    )

    assert stream_consumer is None
    assert stream_delta_cb is None
    assert holder[0] is None


def test_execute_gateway_sync_turn_runs_conversation_and_finalizes(monkeypatch):
    agent = SimpleNamespace(tools=[{"name": "terminal"}])
    stream_consumer = MagicMock()
    pending_notes = {"key-1": "switch note"}

    monkeypatch.setattr(
        "gateway.agent_execution_service.run_gateway_approved_conversation",
        lambda **kwargs: {
            "final_response": kwargs["message"],
            "messages": kwargs["conversation_history"],
            "api_calls": 2,
        },
    )

    finalized_calls = {}

    def _fake_finalize(**kwargs):
        finalized_calls.update(kwargs)
        return {"final_response": "finalized", "messages": kwargs["result"]["messages"]}

    monkeypatch.setattr(
        "gateway.agent_execution_service.finalize_gateway_agent_conversation_result",
        _fake_finalize,
    )

    outcome = execute_gateway_sync_turn(
        agent=agent,
        message="hello",
        history=[{"role": "user", "content": "hi"}],
        session_id="sess-1",
        session_key="key-1",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        status_adapter=SimpleNamespace(),
        status_chat_id="chat-1",
        status_thread_metadata={"thread_id": "t1"},
        loop_for_step="loop",
        logger=MagicMock(),
        admin_only_message_builder=lambda action: None,
        stream_consumer=stream_consumer,
        session_store=SimpleNamespace(_entries={}, _save=MagicMock()),
        session_db=object(),
        empty_response_fallback=lambda kind: "fallback",
        pending_model_notes=pending_notes,
    )

    assert outcome.result["api_calls"] == 2
    assert outcome.final_result["final_response"] == "finalized"
    assert outcome.tools == [{"name": "terminal"}]
    assert "key-1" not in pending_notes
    stream_consumer.finish.assert_called_once_with()
    assert finalized_calls["message"] == "hello"
    assert finalized_calls["history_media_paths"] == set()
