from __future__ import annotations

import contextlib
import dataclasses
import threading
from types import SimpleNamespace

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.execution_router import (
    ExecutionKind,
    ExecutionRouteCandidateV1,
    ExecutionRouteDecisionV1,
    ExecutionRouterProviderDescriptorV1,
    RoutedAttemptRestartRequired,
)
from hermes_cli.execution_router_runtime import (
    ExecutionRouterRegistration,
    prepare_main_turn_attempt,
    _prepare_main_turn_continuation_attempt,
)
from hermes_state import SessionDB
from tui_gateway import server


class _SurfaceProvider:
    descriptor = ExecutionRouterProviderDescriptorV1(
        plugin_id="router-plugin",
        plugin_version="1.0",
        provider_id="router-provider",
        contract_version="1.0",
        supported_execution_kinds=(ExecutionKind.MAIN_TURN,),
    )

    def __init__(self):
        self.requests = []

    def resolve_execution_route(self, request, _cancellation):
        self.requests.append(request)
        ids = tuple(candidate.candidate_id for candidate in request.eligible_candidates)
        selected = "fallback-0" if "fallback-0" in ids else request.native_candidate_id
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id=selected,
        )


def _candidates():
    return (
        ExecutionRouteCandidateV1("native", "p0", "m0", None),
        ExecutionRouteCandidateV1("fallback-0", "p1", "m1", None),
        ExecutionRouteCandidateV1("fallback-1", "p2", "m2", None),
    )


def _prepare_through_surface(surface, record, candidates, db):
    if surface == "cli":
        from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

        shell = SimpleNamespace(
            _session_db=db, _main_turn_candidates=candidates,
            model="old", requested_provider="old", provider="old", reasoning_config=None,
        )
        return CLIChatTurnMixin._prepare_main_turn_continuation(shell, record, "go")
    raise AssertionError(f"surface outside this slice: {surface}")


def _exercise_surface_continuation(tmp_path, surface, monkeypatch):
    from agent.main_turn_continuation import (
        _continue_main_turn_attempt,
        _seal_main_turn_continuation,
    )

    db = SessionDB(tmp_path / f"{surface}.db")
    sid = f"{surface}-session"
    db.create_session(sid, source="cli")
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "fixture", "arguments": "{}"},
    }]
    db.append_message(sid, role="user", content="go")
    db.append_message(sid, role="assistant", content="", tool_calls=tool_calls)
    db.append_message(sid, role="tool", content="done", tool_call_id="call-1")
    durable = db.get_messages(sid)
    for row in durable:
        row[_DB_PERSISTED_MARKER] = True

    provider = _SurfaceProvider()
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    from hermes_cli import plugins
    lifecycle = db.execution_route_lifecycle(sid)
    first = prepare_main_turn_attempt(
        raw_instruction="go",
        surface_class=surface,
        session_id=sid,
        native_candidate_id="native",
        eligible_candidates=_candidates(),
        registration=registration,
        lifecycle=lifecycle,
        revalidate=lambda _request, route: route in tuple(c.identity() for c in _candidates()),
    )
    old_agent = SimpleNamespace(
        session_id=sid,
        _last_flushed_db_idx=len(durable),
        _routed_restart_required=RoutedAttemptRestartRequired("failed", 0),
    )
    record = _seal_main_turn_continuation(
        old_agent,
        first,
        {"messages": durable, "turn_id": "turn-1", "current_turn_user_idx": 0},
    )
    continuation_provider = _SurfaceProvider()
    continuation_registration = ExecutionRouterRegistration(
        continuation_provider, 1, lambda generation: generation == 1
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(
            get_execution_router_registration=lambda: continuation_registration
        ),
    )
    second = _prepare_through_surface(surface, record, _candidates(), db)
    callback = object()

    class ContinuationAgent:
        session_id = sid
        status_callback = None

    from agent import conversation_loop
    def continue_from_transcript(agent, transcript, **_boundary):
        assert agent.status_callback is callback
        transcript.append({"role": "assistant", "content": "finished"})
        return {"messages": transcript, "final_response": "finished", "completed": True}
    monkeypatch.setattr(
        conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript
    )

    result = _continue_main_turn_attempt(
        ContinuationAgent(), record, existing_surface_callbacks={"status_callback": callback}
    )

    rows = db.get_messages(sid)
    assert sum(row["role"] == "user" and row["content"] == "go" for row in rows) == 1
    assert sum(bool(row["role"] == "assistant" and row.get("tool_calls")) for row in rows) == 1
    assert sum(row["role"] == "tool" and row.get("tool_call_id") == "call-1" for row in rows) == 1
    assert len(provider.requests) == 1
    assert len(continuation_provider.requests) == 1
    assert provider.requests[0].request_id != continuation_provider.requests[0].request_id
    assert provider.requests[0].attempt_id != continuation_provider.requests[0].attempt_id
    assert second.request == continuation_provider.requests[0]
    assert result["messages"][-1]["content"] == "finished"
    assert sum(row["role"] == "user" for row in result["messages"]) == 1
    db.close()


def _exercise_exhaustion(tmp_path, surface, monkeypatch):
    from agent.main_turn_continuation import _seal_main_turn_continuation
    from hermes_cli import plugins

    db = SessionDB(tmp_path / f"{surface}-exhaustion.db")
    db.create_session("session-a", source=surface)
    provider = _SurfaceProvider()
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: registration),
    )
    rows = [{"role": "user", "content": "go", _DB_PERSISTED_MARKER: True}]
    record = _seal_main_turn_continuation(
        SimpleNamespace(
            session_id="session-a",
            _last_flushed_db_idx=1,
            _routed_restart_required=RoutedAttemptRestartRequired("failed", 1),
        ),
        SimpleNamespace(
            request=SimpleNamespace(request_id="old-request", attempt_id="old-attempt"),
            route_signature=("p1", "m1", None, 1),
        ),
        {"messages": rows, "turn_id": "turn-1", "current_turn_user_idx": 0},
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        _prepare_through_surface(surface, record, _candidates(), db)
    assert provider.requests == []
    db.close()


def _exercise_no_router(tmp_path, surface):
    db = SessionDB(tmp_path / f"{surface}-native.db")
    sid = f"{surface}-native"
    db.create_session(sid, source="cli")
    prepared = prepare_main_turn_attempt(
        raw_instruction="native",
        surface_class=surface,
        session_id=sid,
        native_candidate_id="native",
        eligible_candidates=(_candidates()[0],),
        registration=None,
        lifecycle=db.execution_route_lifecycle(sid),
    )
    assert prepared.may_start
    assert prepared.selected_model == "m0"
    assert db.execution_route_lifecycle(sid).read_events() == ()
    db.close()


def test_cli_routed_fallback_continues_same_turn_once(tmp_path, monkeypatch):
    _exercise_surface_continuation(tmp_path, "cli", monkeypatch)
    _exercise_cli_owner_loop(tmp_path, monkeypatch)
    _assert_surface_stop_precredential("cli", tmp_path, monkeypatch)


def test_cli_routed_fallback_budget_exhaustion_is_not_reset(tmp_path, monkeypatch):
    _exercise_exhaustion(tmp_path, "cli", monkeypatch)


def test_cli_no_router_main_turn_and_native_fallback_are_unchanged(tmp_path, monkeypatch):
    _exercise_no_router(tmp_path, "cli")

    from hermes_cli import plugins
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from hermes_cli.execution_router_runtime import bind_main_turn_attempt

    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: None),
    )
    order = []
    streamed = []
    stream_callback = streamed.append

    class NativeAgent:
        def run_conversation(self, **kwargs):
            order.append("run")
            assert kwargs["stream_callback"] is stream_callback
            return {"messages": [], "final_response": "native", "completed": True}

    class NativeShell(CLIChatTurnMixin):
        def __init__(self):
            self._secret_capture_callback = None
            self._sudo_password_callback = None
            self._approval_callback = None
            self._active_agent_route_signature = ("old",)
            self.agent = NativeAgent()
            self.conversation_history = []
            self.session_id = "native-session"
            self._pending_model_switch_note = None
            self._pending_skills_reload_note = None
            self._pending_moa_config = None
            self._pending_one_turn_model_restore = None
            self._pending_moa_disable_after_turn = False
            self._pending_moa_restore_model = None

        def _ensure_runtime_credentials(self):
            order.append("credentials")
            return True

        def _resolve_turn_agent_config(self, message):
            order.append("config")
            assert message == "native"
            return {"model": "m0", "runtime": {}, "signature": ("new",)}

        def _init_agent(self, **_kwargs):
            order.append("init")
            self.agent = NativeAgent()
            return True

        def _chat_route_images(self, message, _images):
            order.append("images")
            return message

        def _chat_expand_context_references(self, message):
            order.append("normalize")
            return message, None

        def _chat_stage_user_message(self, _agent, _message):
            order.append("stage")
            self.conversation_history.append({"role": "user", "content": "native"})

        def _reset_stream_state(self):
            return None

        def _chat_setup_turn_audio(self, turn, _message, _voice_input):
            turn.stream_callback = stream_callback

        def _chat_monitor_agent_thread(self, _turn, thread):
            thread.join()
            return None

        def _chat_settle_turn(self, _turn):
            return None

        def _chat_render_turn(self, turn, _thread, _interrupt):
            return turn.result["final_response"]

        def _chat_release_turn_audio(self, _turn):
            return None

        def _flush_credit_notices(self):
            return None

    shell = NativeShell()
    assert shell.chat("native") == "native"
    assert order == ["credentials", "config", "init", "images", "normalize", "stage", "run"]
    assert not hasattr(shell, "_main_turn_prepared_attempt")
    assert not hasattr(shell.agent, "_execution_router_selected_attempt")

    untouched = SimpleNamespace(callback=stream_callback)
    before = vars(untouched).copy()
    bind_main_turn_attempt(None, untouched, ())
    assert vars(untouched) == before


def _exercise_cli_owner_loop(tmp_path, monkeypatch):
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from agent import conversation_loop

    db = SessionDB(tmp_path / "cli-owner.db")
    sid = "cli-owner"
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="go")
    durable = db.get_messages(sid)
    durable[0][_DB_PERSISTED_MARKER] = True
    provider = _SurfaceProvider()
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    candidates = _candidates()
    prepared = prepare_main_turn_attempt(
        raw_instruction="go", surface_class="cli", session_id=sid,
        native_candidate_id="native", eligible_candidates=candidates,
        registration=registration,
        revalidate=lambda _request, route: route in tuple(c.identity() for c in candidates),
    )
    observed = []
    old_stream_callback = None

    class OldAgent:
        session_id = sid
        _last_flushed_db_idx = 1
        _routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
        stream_delta_callback = None
        status_callback = None

        def run_conversation(self, **kwargs):
            nonlocal old_stream_callback
            old_stream_callback = kwargs["stream_callback"]
            old_stream_callback("before")
            return {"messages": durable, "turn_id": "turn-1", "current_turn_user_idx": 0}

        def _persist_session(self, _messages):
            return None

        def release_clients(self):
            return None

    class NewAgent:
        session_id = sid
        stream_delta_callback = None
        status_callback = None

    def continue_from_transcript(agent, transcript, **_boundary):
        old_stream_callback("late")
        agent._stream_callback("fresh")
        transcript.append({"role": "assistant", "content": "finished"})
        return {"messages": transcript, "final_response": "finished", "completed": True}
    monkeypatch.setattr(
        conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript
    )

    shell = SimpleNamespace(
        agent=OldAgent(), session_id=sid, conversation_history=list(durable),
        _main_turn_prepared_attempt=prepared, _main_turn_candidates=candidates,
        _active_agent_route_signature=prepared.route_signature,
        _sudo_password_callback=None, _approval_callback=None, _secret_capture_callback=None,
        _pending_model_switch_note=None, _pending_skills_reload_note=None,
        _pending_moa_config=None, _pending_one_turn_model_restore=None,
        _pending_moa_disable_after_turn=False, _pending_moa_restore_model=None,
        _flush_credit_notices=lambda: None,
        _ensure_runtime_credentials=lambda: True,
        _resolve_turn_agent_config=lambda _message: {
            "model": "m2", "runtime": {}, "request_overrides": None,
            "signature": ("m2",),
        },
    )
    shell._prepare_main_turn_continuation = lambda record, _message: (
        _prepare_main_turn_continuation_attempt(
            record, raw_instruction="go", surface_class="cli",
            eligible_candidates=candidates, registration=registration,
        )
    )
    def init_agent(**_kwargs):
        shell.agent = NewAgent()
        return True
    shell._init_agent = init_agent
    turn = SimpleNamespace(voice_prefix="", stream_callback=observed.append, result=None)

    CLIChatTurnMixin._chat_run_agent(shell, turn, "go")

    assert turn.result["final_response"] == "finished"
    assert sum(row["role"] == "user" for row in turn.result["messages"]) == 1
    assert observed == ["before", "fresh"]
    db.close()


def _registration(provider):
    return ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)


def _turn_state(agent):
    return server._TurnRun(agent, None, None, receipt_committed=True)


def _patch_invoke_plumbing(monkeypatch, events):
    class _Stop:
        def set(self):
            return None

    class _Thread:
        def join(self):
            return None

    monkeypatch.setattr(server, "_start_usage_ticker", lambda _sid, _agent: (_Stop(), _Thread()))
    monkeypatch.setattr(server, "_append_inflight_delta", lambda _session, text: events.append(("inflight", text)))
    monkeypatch.setattr(server, "_emit", lambda event, _sid, payload=None: events.append((event, payload)))
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: True)


def _patch_prepare_plumbing(monkeypatch):
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp")
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "_start_turn_voice", lambda: (None, False))
    monkeypatch.setattr(server, "_pending_reaction_notes", lambda _session: None)
    monkeypatch.setattr(server, "_hud_surface_note", lambda _session: None)


def test_tui_routed_fallback_continues_inside_one_prompt_submit(tmp_path, monkeypatch):
    _assert_surface_stop_precredential("tui", tmp_path, monkeypatch)
    from agent import conversation_loop
    from hermes_cli import plugins

    db = SessionDB(tmp_path / "tui.db")
    sid = "ui-session"
    session_key = "tui-session"
    db.create_session(session_key, source="tui")
    provider = _SurfaceProvider()
    registration = _registration(provider)
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: registration),
    )

    events = []
    rebuilds = []
    releases = []
    side_effects = []
    old_callbacks = {}

    class _NativeAgent:
        model = "m0"
        provider = "p0"
        requested_provider = "p0"
        reasoning_config = None
        _fallback_chain = [
            {"provider": "p1", "model": "m1"},
            {"provider": "p2", "model": "m2"},
        ]
        _session_db = db
        session_id = session_key

        def release_clients(self):
            releases.append("native")

    class _RoutedAgent:
        reasoning_config = None
        _session_db = db
        session_id = session_key
        interim_assistant_callback = None
        _on_session_title = None

        def __init__(self, provider_name, model_name):
            self.provider = provider_name
            self.requested_provider = provider_name
            self.model = model_name
            self._fallback_chain = _NativeAgent._fallback_chain
            self._last_flushed_db_idx = 0
            self._routed_restart_required = None

        def run_conversation(self, _message, **kwargs):
            side_effects.append("tool")
            kwargs["stream_callback"]("before")
            self.interim_assistant_callback("before-interim")
            self._on_session_title("before-title", "model")
            old_callbacks.update(
                stream=kwargs["stream_callback"],
                interim=self.interim_assistant_callback,
                title=self._on_session_title,
            )
            tool_calls = [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "fixture", "arguments": "{}"},
            }]
            db.append_message(session_key, role="user", content="go")
            db.append_message(session_key, role="assistant", content="", tool_calls=tool_calls)
            db.append_message(session_key, role="tool", content="done", tool_call_id="call-1")
            rows = db.get_messages(session_key)
            for row in rows:
                row[_DB_PERSISTED_MARKER] = True
            self._last_flushed_db_idx = len(rows)
            self._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
            return {"messages": rows, "turn_id": "turn-1", "current_turn_user_idx": 0}

        def _persist_session(self, _messages):
            return None

        def release_clients(self):
            releases.append(self.model)

    native = _NativeAgent()
    session = {
        "agent": native,
        "session_key": session_key,
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": True,
        "cols": 80,
    }

    def rebuild(_sid, target, **kwargs):
        rebuilds.append((kwargs["provider_override"], kwargs["model_override"], len(provider.requests)))
        replacement = _RoutedAgent(kwargs["provider_override"], kwargs["model_override"])
        target["agent"] = replacement
        return replacement

    monkeypatch.setattr(server, "_rebuild_session_agent", rebuild)
    _patch_prepare_plumbing(monkeypatch)
    _patch_invoke_plumbing(monkeypatch, events)

    def continue_from_transcript(agent, transcript, **boundary):
        old_callbacks["stream"]("late")
        old_callbacks["interim"]("late-interim")
        old_callbacks["title"]("late-title", "model")
        agent._stream_callback("fresh")
        agent.interim_assistant_callback("fresh-interim")
        agent._on_session_title("fresh-title", "model")
        assert boundary == {"turn_id": "turn-1", "current_turn_user_idx": 0}
        transcript.append({"role": "assistant", "content": "finished"})
        return {"messages": transcript, "final_response": "finished", "completed": True}

    monkeypatch.setattr(conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript)

    admissions = []
    markers = []
    commits = []
    followups = []

    class _ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_admit_prompt_turn", lambda *_args: admissions.append("admit") or ([], native))
    monkeypatch.setattr(
        server,
        "_record_turn_marker",
        lambda target, _text, **_kwargs: markers.append("marker") or target.setdefault("_active_turn_marker_key", "marker"),
    )
    monkeypatch.setattr(server, "bind_transport", lambda _transport: None)
    monkeypatch.setattr(server, "reset_transport", lambda _token: None)
    monkeypatch.setattr(server, "_routing_provenance_db", lambda _session: contextlib.nullcontext(None))
    monkeypatch.setattr(server, "_commit_turn_history", lambda target, result, *_args: commits.append(result) or target.update(history=result["messages"]))
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_complete_turn_payload", lambda _session, st, _note, _cols: ({"text": st.result["final_response"]}, st.result["final_response"], "complete"))
    monkeypatch.setattr(server, "_goal_followup_after_turn", lambda *_args: None)
    monkeypatch.setattr(server, "_after_complete_turn", lambda *_args: None)
    monkeypatch.setattr(server, "_publish_session_control_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_finish_turn", lambda *_args: None)
    monkeypatch.setattr(server, "_retire_turn_marker", lambda *_args: None)
    monkeypatch.setattr(server, "_emit_settled_session_info", lambda *_args: None)
    monkeypatch.setattr(server, "_run_post_turn_followups", lambda *_args: followups.append("final"))
    server._sessions[sid] = session
    try:
        assert server._run_prompt_submit("rid", sid, session, "go") is True
    finally:
        server._sessions.pop(sid, None)

    rows = db.get_messages(session_key)
    assert admissions == ["admit"]
    assert markers == ["marker"]
    assert [event for event, _payload in events].count("message.start") == 1
    assert len(commits) == 1
    assert followups == ["final"]
    assert sum(row["role"] == "user" and row["content"] == "go" for row in rows) == 1
    assert sum(bool(row["role"] == "assistant" and row.get("tool_calls")) for row in rows) == 1
    assert sum(row["role"] == "tool" and row.get("tool_call_id") == "call-1" for row in rows) == 1
    assert side_effects == ["tool"]
    assert len(provider.requests) == 2
    assert provider.requests[0].request_id != provider.requests[1].request_id
    assert provider.requests[0].attempt_id != provider.requests[1].attempt_id
    assert rebuilds == [("p1", "m1", 1), ("p2", "m2", 2)]
    assert releases == ["native", "m1"]
    assert ("message.delta", {"text": "late"}) not in events
    assert ("message.interim", {"text": "late-interim", "already_streamed": False}) not in events
    assert ("message.delta", {"text": "fresh"}) in events
    assert ("message.interim", {"text": "fresh-interim", "already_streamed": False}) in events
    assert not any(event == "session.title" and payload["title"] == "late-title" for event, payload in events)
    assert any(event == "session.title" and payload["title"] == "fresh-title" for event, payload in events)
    db.close()


def test_tui_routed_fallback_budget_exhaustion_is_not_a_queued_followup(tmp_path, monkeypatch):
    from hermes_cli import plugins

    db = SessionDB(tmp_path / "exhausted.db")
    session_key = "tui-exhausted"
    db.create_session(session_key, source="tui")
    provider = _SurfaceProvider()
    registration = _registration(provider)
    candidates = _candidates()[:2]
    prepared = prepare_main_turn_attempt(
        raw_instruction="go",
        surface_class="tui",
        session_id=session_key,
        native_candidate_id="native",
        eligible_candidates=candidates,
        registration=registration,
        lifecycle=db.execution_route_lifecycle(session_key),
        revalidate=lambda _request, route: route in tuple(item.identity() for item in candidates),
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: registration),
    )
    db.append_message(session_key, role="user", content="go")
    rows = db.get_messages(session_key)
    rows[0][_DB_PERSISTED_MARKER] = True

    class _Agent:
        session_id = session_key
        model = "m1"
        provider = "p1"
        requested_provider = "p1"
        reasoning_config = None
        _session_db = db
        _last_flushed_db_idx = 1
        _routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
        interim_assistant_callback = None

        def run_conversation(self, _message, **_kwargs):
            self._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
            return {"messages": rows, "turn_id": "turn-1", "current_turn_user_idx": 0}

        def _persist_session(self, _messages):
            return None

    agent = _Agent()
    session = {
        "agent": agent,
        "session_key": session_key,
        "history": list(rows),
        "history_lock": threading.RLock(),
        "_main_turn_prepared_attempt": prepared,
        "_main_turn_candidates": candidates,
    }
    st = _turn_state(agent)
    st.history = list(rows)
    events = []
    _patch_invoke_plumbing(monkeypatch, events)
    for name in ("_enqueue_prompt", "_dispatch_followup_turn", "_run_prompt_submit"):
        monkeypatch.setattr(server, name, lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(AssertionError(_name)))

    server._invoke_agent("sid", session, st, "go", "go", None, [], None, None)

    assert st.result["failed"] is True
    assert "exhausted" in st.result["error"]
    assert len(provider.requests) == 1
    assert session["agent"] is agent
    db.close()


def test_tui_no_router_main_turn_and_native_fallback_are_unchanged(tmp_path, monkeypatch):
    from agent import main_turn_continuation
    from hermes_cli import execution_router_runtime, plugins
    from tools import approval_context

    db = SessionDB(tmp_path / "native.db")
    session_key = "tui-native"
    db.create_session(session_key, source="tui")
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: None),
    )
    streamed = []

    class _NativeAgent:
        session_id = session_key
        model = "m0"
        provider = "p0"
        requested_provider = "p0"
        reasoning_config = None
        _fallback_chain = [{"provider": "p1", "model": "m1"}]
        _session_db = db
        interim_assistant_callback = None

        def run_conversation(self, _message, **kwargs):
            streamed.append(kwargs["stream_callback"])
            kwargs["stream_callback"]("native-primary")
            kwargs["stream_callback"]("native-fallback")
            return {"messages": [], "final_response": "native", "completed": True}

    agent = _NativeAgent()
    session = {
        "agent": agent,
        "session_key": session_key,
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 7,
        "_main_turn_route_signature": ("prior-routed-signature",),
        "native_marker": object(),
    }
    before = dict(session)

    prepared = server._prepare_tui_main_turn_attempt("sid", session, "native")
    assert prepared is None
    assert session == before

    _patch_prepare_plumbing(monkeypatch)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(approval_context, "set_current_session_key", lambda _key: None)
    st = _turn_state(agent)
    prepared_input = server._prepare_turn_input("sid", session, st, "native", [])
    assert prepared_input is not None
    assert prepared_input[:2] == ("native", "native")
    assert st.agent is agent
    assert st.history == []
    assert st.history_version == 7
    assert session == before

    def forbidden(*_args, **_kwargs):
        raise AssertionError("routed-only machinery reached the native branch")

    monkeypatch.setattr(main_turn_continuation, "_bind_routed_main_turn_callbacks", forbidden)
    monkeypatch.setattr(execution_router_runtime, "bind_main_turn_attempt", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_started", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_finished", forbidden)
    monkeypatch.setattr(server, "_rebuild_session_agent", forbidden)
    events = []
    append_inflight_delta = server._append_inflight_delta
    _patch_invoke_plumbing(monkeypatch, events)
    monkeypatch.setattr(server, "_append_inflight_delta", append_inflight_delta)
    server._start_inflight_turn(session, "native")

    server._invoke_agent("sid", session, st, "native", "native", None, [], None, None)

    assert st.agent is agent
    assert st.result["final_response"] == "native"
    assert len(streamed) == 1
    assert [payload["text"] for event, payload in events if event == "message.delta"] == [
        "native-primary",
        "native-fallback",
    ]
    assert session["inflight_turn"]["assistant"] == "native-primarynative-fallback"
    assert session["history_version"] == 7
    assert session["_main_turn_route_signature"] == ("prior-routed-signature",)
    assert "_main_turn_candidates" not in session
    assert "_main_turn_prepared_attempt" not in session
    assert "_main_turn_router_registration" not in session
    assert db.execution_route_lifecycle(session_key).read_events() == ()
    db.close()


def _gateway_harness(tmp_path, monkeypatch, provider, fallback_models):
    from gateway.config import Platform
    from gateway.run import SessionSource
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from hermes_cli import plugins

    db = SessionDB(tmp_path / "gateway.db")
    session_id = "gateway-session"
    db.create_session(session_id, source="local")
    registration = _registration(provider) if provider is not None else None
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(get_execution_router_registration=lambda: registration),
    )
    source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
    calls = []
    runner = SimpleNamespace(
        config=None,
        _session_db=SimpleNamespace(_db=db),
        _provider_routing={},
        _prefill_messages=None,
        _agent_cache_lock=None,
        _agent_cache={},
        _pending_model_notes={},
        _pending_skills_reload_notes={},
        session_store=SimpleNamespace(_entries={}),
        _resolve_session_agent_runtime=lambda **_kwargs: calls.append(
            ("credentials", len(provider.requests) if provider is not None else 0)
        ) or (
            "m0", {"provider": "p0"}
        ),
        _resolve_session_reasoning_config=lambda **_kwargs: None,
        _resolve_session_service_tier=lambda **_kwargs: None,
        _resolve_turn_agent_config=lambda _message, model, runtime: calls.append("config") or {
            "model": model,
            "runtime": runtime,
        },
        _refresh_fallback_model=lambda: list(fallback_models),
        _peek_session_state=lambda _key: None,
    )
    ctx = TurnContext(
        source=source,
        message="go",
        history=[],
        session_id=session_id,
        session_key=session_id,
        user_config={"model": {"default": "m0", "provider": "p0"}},
        resolve_display_setting=lambda *_args: False,
        _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )
    turn = TurnRunner(runner, ctx)
    turn._combined_ephemeral_prompt = lambda: ""
    turn._setup_stream_consumer = lambda _platform: (None, stream_callback, interim_callback, True)
    turn._load_turn_history = lambda _agent, _reused: ([], None, set())
    turn._prepare_turn_message = lambda _history: ("go", 123.0)
    turn._finish_stream_consumer = lambda *_args: calls.append("finalize")
    turn._sync_session_after_run = lambda _history: (False, session_id, 0)
    turn._append_auto_media_tags = lambda response, *_args: response

    streamed = []
    interim = []

    def stream_callback(text):
        streamed.append(text)

    def interim_callback(text, **_kwargs):
        interim.append(text)

    return turn, runner, ctx, db, calls, streamed, interim


def test_gateway_routed_fallback_continues_one_inbound_turn(tmp_path, monkeypatch):
    _assert_surface_stop_precredential("gateway", tmp_path, monkeypatch)
    from agent import conversation_loop
    from agent.execution_router import RoutedAttemptRestartRequired
    from gateway import run as gateway_run

    provider = _SurfaceProvider()
    turn, _runner, ctx, db, calls, streamed, interim = _gateway_harness(
        tmp_path,
        monkeypatch,
        provider,
        (
            {"provider": "p1", "model": "m1"},
            {"provider": "p2", "model": "m2"},
        ),
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider_name: calls.append(("route-credentials", provider_name)) or {
            "provider": provider_name
        },
    )
    agents = []
    old_callbacks = {}
    side_effects = []
    title_renames = []

    def attach_title(agent, ctx):
        agent._on_session_title = lambda title, title_source: title_renames.append(
            (agent.model, title, title_source)
        )

    turn._attach_session_title_callback = attach_title

    class Agent:
        reasoning_config = None
        context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=0)
        session_prompt_tokens = 0
        session_completion_tokens = 0

        def __init__(self, provider_name, model_name):
            self.provider = provider_name
            self.requested_provider = provider_name
            self.model = model_name
            self.session_id = ctx.session_id
            self._session_db = db
            self._fallback_index = 0
            self._last_flushed_db_idx = 0
            self._routed_restart_required = None

        def _persist_session(self, _messages):
            return None

    def resolve_agent(route, *_args):
        calls.append((
            "agent",
            route["runtime"]["provider"],
            route["model"],
            len(provider.requests),
        ))
        agent = Agent(route["runtime"]["provider"], route["model"])
        agents.append(agent)
        return agent, False

    def wire(agent, _route, _reasoning, stream, interim_cb, _want_interim):
        agent._stream_callback = stream
        agent.interim_assistant_callback = interim_cb
        turn._attach_session_title_callback(agent, ctx)
        ctx.agent_holder[0] = agent

    def first_run(agent, _history, _observed, persist_message, persist_timestamp):
        assert (persist_message, persist_timestamp) == ("go", 123.0)
        calls.append("dispatch")
        side_effects.append("tool")
        old_callbacks.update(
            stream=agent._stream_callback,
            interim=agent.interim_assistant_callback,
            title=agent._on_session_title,
        )
        tool_calls = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "fixture", "arguments": "{}"},
        }]
        db.append_message(ctx.session_id, role="user", content="go", timestamp=persist_timestamp)
        db.append_message(ctx.session_id, role="assistant", content="", tool_calls=tool_calls)
        db.append_message(ctx.session_id, role="tool", content="done", tool_call_id="call-1")
        rows = db.get_messages(ctx.session_id)
        for row in rows:
            row[_DB_PERSISTED_MARKER] = True
        agent._last_flushed_db_idx = len(rows)
        agent._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
        return {
            "messages": rows,
            "turn_id": "turn-1",
            "current_turn_user_idx": 0,
            "completed": False,
        }

    def continue_from_transcript(agent, transcript, **boundary):
        old_callbacks["stream"]("late")
        old_callbacks["interim"]("late-interim")
        old_callbacks["title"]("late-title", "llm")
        agent._stream_callback("fresh")
        agent.interim_assistant_callback("fresh-interim")
        agent._on_session_title("fresh-title", "llm")
        assert boundary == {"turn_id": "turn-1", "current_turn_user_idx": 0}
        transcript.append({"role": "assistant", "content": "finished"})
        return {"messages": transcript, "final_response": "finished", "completed": True}

    turn._resolve_turn_agent = resolve_agent
    turn._wire_turn_agent_callbacks = wire
    turn._run_conversation_with_approval = first_run
    monkeypatch.setattr(conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript)

    result = turn.run_sync()

    rows = db.get_messages(ctx.session_id)
    assert result["final_response"] == "finished"
    assert side_effects == ["tool"]
    assert sum(row["role"] == "user" and row["content"] == "go" for row in rows) == 1
    assert sum(bool(row["role"] == "assistant" and row.get("tool_calls")) for row in rows) == 1
    assert sum(row["role"] == "tool" and row.get("tool_call_id") == "call-1" for row in rows) == 1
    assert len(provider.requests) == 2
    assert provider.requests[0].request_id != provider.requests[1].request_id
    assert provider.requests[0].attempt_id != provider.requests[1].attempt_id
    assert [call for call in calls if isinstance(call, tuple) and call[0] == "agent"] == [
        ("agent", "p1", "m1", 1),
        ("agent", "p2", "m2", 2),
    ]
    lifecycle = db.execution_route_lifecycle(str(ctx.session_id))
    for request in provider.requests:
        events = lifecycle.read_events(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
        )
        assert [event.event_type.value for event in events] == [
            "route_requested",
            "route_accepted",
            "route_started",
            "route_finished",
        ]
    assert streamed == ["fresh"]
    assert interim == ["fresh-interim"]
    assert title_renames == [("m2", "fresh-title", "llm")]
    assert calls.count("dispatch") == 1
    assert calls.count("finalize") == 1
    db.close()


def test_gateway_routed_fallback_budget_exhaustion_never_queues_or_replays(
    tmp_path, monkeypatch
):
    from agent.execution_router import RoutedAttemptRestartRequired
    from gateway import run as gateway_run

    provider = _SurfaceProvider()
    turn, runner, ctx, db, calls, _streamed, _interim = _gateway_harness(
        tmp_path,
        monkeypatch,
        provider,
        ({"provider": "p1", "model": "m1"},),
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider_name: {"provider": provider_name},
    )

    class Agent:
        provider = requested_provider = "p1"
        model = "m1"
        reasoning_config = None
        session_id = ctx.session_id
        _session_db = db
        _fallback_index = 0
        _last_flushed_db_idx = 1
        _routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
        context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=0)
        session_prompt_tokens = session_completion_tokens = 0

        def _persist_session(self, _messages):
            return None

    agent = Agent()
    db.append_message(ctx.session_id, role="user", content="go")
    rows = db.get_messages(ctx.session_id)
    rows[0][_DB_PERSISTED_MARKER] = True
    turn._resolve_turn_agent = lambda *_args: (agent, False)
    turn._wire_turn_agent_callbacks = lambda target, *_args: ctx.agent_holder.__setitem__(0, target)
    def run_exhausted_attempt(*_args):
        agent._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
        return {
            "messages": rows,
            "turn_id": "turn-1",
            "current_turn_user_idx": 0,
            "completed": False,
        }

    turn._run_conversation_with_approval = run_exhausted_attempt
    for name in ("_handle_message", "_process_message_background", "_dispatch_queued_followups"):
        setattr(runner, name, lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(AssertionError(_name)))

    result = turn.run_sync()

    assert result["failed"] is True
    assert "exhausted" in result["error"]
    assert len(provider.requests) == 1
    assert sum(row["role"] == "user" and row["content"] == "go" for row in db.get_messages(ctx.session_id)) == 1
    assert calls.count("finalize") == 1
    db.close()


def test_gateway_no_router_main_turn_and_native_fallback_are_unchanged(tmp_path, monkeypatch):
    from agent import main_turn_continuation
    from gateway.turn_context import TurnContext
    from hermes_cli import execution_router_runtime

    turn, runner, ctx, db, calls, streamed, interim = _gateway_harness(
        tmp_path, monkeypatch, None, ({"provider": "p1", "model": "m1"},)
    )
    stream_object = turn._setup_stream_consumer("cli")[1]
    interim_object = turn._setup_stream_consumer("cli")[2]
    turn._setup_stream_consumer = lambda _platform: (None, stream_object, interim_object, True)
    native_agent = SimpleNamespace(
        provider="p0",
        requested_provider="p0",
        model="m0",
        reasoning_config=None,
        session_id=ctx.session_id,
        context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=0),
        session_prompt_tokens=0,
        session_completion_tokens=0,
    )
    callback_objects = []
    turn._resolve_turn_agent = lambda *_args: (native_agent, True)

    def wire(agent, _route, _reasoning, stream, interim_cb, _want_interim):
        callback_objects.append((stream, interim_cb))
        ctx.agent_holder[0] = agent

    turn._wire_turn_agent_callbacks = wire
    turn._run_conversation_with_approval = lambda *_args: {
        "messages": [],
        "final_response": "native",
        "completed": True,
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("routed-only machinery reached the disabled path")

    monkeypatch.setattr(main_turn_continuation, "_bind_routed_main_turn_callbacks", forbidden)
    monkeypatch.setattr(execution_router_runtime, "bind_main_turn_attempt", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_started", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_finished", forbidden)

    result = turn.run_sync()

    assert result["final_response"] == "native"
    assert callback_objects == [(stream_object, interim_object)]
    assert db.execution_route_lifecycle(ctx.session_id).read_events() == ()
    assert not any(field.name.startswith("_main_turn_") for field in dataclasses.fields(TurnContext))
    assert not any(name.startswith("_main_turn_") for name in vars(ctx))

    cached = object()
    evictions = []
    runner._agent_cache_lock = threading.RLock()
    runner._agent_cache = {ctx.session_key: (cached, ("old",), 0, ctx.session_id)}
    turn._pop_cached_agent_for_eviction = lambda: evictions.append("evicted")
    found = turn._lookup_cached_agent(
        ("new",), runner._agent_cache_lock, runner._agent_cache, 10,
        ctx.session_id, False, 0,
    )
    assert found.agent is None
    assert found.evicted is None
    assert evictions == []

    error_turn, error_runner, _error_ctx, error_db, *_ = _gateway_harness(
        tmp_path / "error", monkeypatch, None, ()
    )
    error_runner._resolve_session_agent_runtime = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("native credential detail")
    )
    error_result = error_turn.run_sync()
    assert error_result == {
        "final_response": "⚠️ Provider authentication failed: native credential detail",
        "messages": [],
        "api_calls": 0,
        "tools": [],
    }
    construction_turn, _construction_runner, _construction_ctx, construction_db, *_ = (
        _gateway_harness(tmp_path / "construction", monkeypatch, None, ())
    )
    construction_turn._resolve_turn_agent = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("native construction detail")
    )
    with pytest.raises(RuntimeError, match="native construction detail"):
        construction_turn.run_sync()
    assert streamed == []
    assert interim == []
    db.close()
    error_db.close()
    construction_db.close()


class _FallbackProvider:
    descriptor = ExecutionRouterProviderDescriptorV1(
        plugin_id="router-plugin",
        plugin_version="1.0",
        provider_id="router-provider",
        contract_version="1.0",
        supported_execution_kinds=(ExecutionKind.MAIN_TURN,),
    )

    def __init__(self, events, db):
        self.events = events
        self.db = db
        self.requests = []

    def resolve_execution_route(self, request, _cancellation):
        self.requests.append(request)
        self.events.append(("router", len(self.requests), request.request_id, request.attempt_id))
        candidate = next(
            item for item in request.eligible_candidates
            if item.candidate_id.startswith("fallback-")
        )
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id=candidate.candidate_id,
        )


def _patch_common(monkeypatch, oneshot, db, provider, fallback_chain, events):
    from hermes_cli import plugins

    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: db)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"default": "m0", "provider": "p0"}
    })
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: list(fallback_chain))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda _cfg, _surface: ["terminal"])
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: events.append("mcp"),
    )
    monkeypatch.setattr(oneshot, "_build_preloaded_skills_prompt", lambda _skills: "skills")
    monkeypatch.setattr(oneshot, "_linger_for_background_completions", lambda: None)
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(
            get_execution_router_registration=lambda: (
                _registration(provider) if provider is not None else None
            )
        ),
    )


def test_oneshot_routed_fallback_continues_before_single_close(tmp_path, monkeypatch):
    _assert_surface_stop_precredential("oneshot", tmp_path, monkeypatch)
    from agent import conversation_loop
    from hermes_cli import execution_router_runtime, oneshot

    events = []
    side_effects = []
    db = SessionDB(tmp_path / "state.db")
    provider = _FallbackProvider(events, db)
    fallback_chain = (
        {"provider": "p1", "model": "m1"},
        {"provider": "p2", "model": "m2"},
    )
    _patch_common(monkeypatch, oneshot, db, provider, fallback_chain, events)

    real_db_close = db.close
    db_closes = []

    def close_db():
        db_closes.append("store")
        events.append("store-close")
        real_db_close()

    monkeypatch.setattr(db, "close", close_db)

    runtime_calls = []

    def resolve_runtime_provider(**kwargs):
        runtime_calls.append(dict(kwargs))
        events.append(("credentials", kwargs["requested"], len(provider.requests)))
        return {
            "api_key": f"key-{kwargs['requested']}",
            "base_url": f"https://{kwargs['requested']}.test",
            "provider": kwargs["requested"],
            "requested_provider": kwargs["requested"],
            "api_mode": "chat_completions",
            "credential_pool": f"pool-{kwargs['requested']}",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_runtime_provider)

    agents = []

    class Agent:
        reasoning_config = None
        _session_messages = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.requested_provider = kwargs["requested_provider"]
            self.session_id = kwargs["session_id"]
            self._session_db = kwargs["session_db"]
            self._last_flushed_db_idx = 0
            self._fallback_index = 0
            self._routed_restart_required = None
            agents.append(self)
            events.append(("agent", self.provider, self.model, len(provider.requests)))

        def run_conversation(self, prompt, conversation_history=None):
            assert prompt == "go"
            assert conversation_history is None
            side_effects.append("tool")
            self._session_db.append_message(self.session_id, role="user", content=prompt)
            self._session_db.append_message(
                self.session_id,
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "fixture", "arguments": "{}"},
                }],
            )
            self._session_db.append_message(
                self.session_id, role="tool", content="done", tool_call_id="call-1"
            )
            rows = self._session_db.get_messages(self.session_id)
            for row in rows:
                row[_DB_PERSISTED_MARKER] = True
            self._last_flushed_db_idx = len(rows)
            self._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
            events.append("first-run")
            return {
                "messages": rows,
                "turn_id": "turn-1",
                "current_turn_user_idx": 0,
                "completed": False,
            }

        def _persist_session(self, messages):
            for row in messages:
                row[_DB_PERSISTED_MARKER] = True

        def release_clients(self):
            events.append(("release", self.model))

        def shutdown_memory_provider(self, *_args):
            events.append(("memory-close", self.model))

        def close(self):
            events.append(("agent-close", self.model))

    monkeypatch.setattr("run_agent.AIAgent", Agent)
    record_routed_restart = execution_router_runtime.record_main_turn_routed_restart

    def close_old_route(prepared):
        record_routed_restart(prepared)
        events.append("route-close")

    monkeypatch.setattr(
        execution_router_runtime, "record_main_turn_routed_restart", close_old_route
    )

    def continue_from_transcript(agent, transcript, **boundary):
        events.append(("continue", agent.model))
        assert boundary == {"turn_id": "turn-1", "current_turn_user_idx": 0}
        assert [row["role"] for row in transcript] == ["user", "assistant", "tool"]
        transcript.append({"role": "assistant", "content": "finished"})
        return {
            "messages": transcript,
            "final_response": "finished",
            "turn_id": "turn-1",
            "current_turn_user_idx": 0,
            "completed": True,
        }

    monkeypatch.setattr(
        conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript
    )

    text, result = oneshot._run_agent("go")

    assert text == result["final_response"] == "finished"
    assert side_effects == ["tool"]
    assert len(provider.requests) == 2
    assert provider.requests[0].request_id != provider.requests[1].request_id
    assert provider.requests[0].attempt_id != provider.requests[1].attempt_id
    assert [(agent.provider, agent.model) for agent in agents] == [("p1", "m1"), ("p2", "m2")]
    assert [(call["requested"], call["target_model"]) for call in runtime_calls] == [
        ("p1", "m1"),
        ("p2", "m2"),
    ]
    assert events.index(("router", 1, provider.requests[0].request_id, provider.requests[0].attempt_id)) < events.index(("credentials", "p1", 1))
    assert events.index("route-close") < events.index(("router", 2, provider.requests[1].request_id, provider.requests[1].attempt_id))
    assert events.index(("router", 2, provider.requests[1].request_id, provider.requests[1].attempt_id)) < events.index(("credentials", "p2", 2))
    assert events.index(("continue", "m2")) < events.index(("agent-close", "m2")) < events.index("store-close")
    assert [event for event in events if isinstance(event, tuple) and event[0] == "agent-close"] == [("agent-close", "m2")]
    assert db_closes == ["store"]

    read_db = SessionDB(tmp_path / "state.db")
    rows = read_db.get_messages(agents[0].session_id)
    assert sum(row["role"] == "user" and row["content"] == "go" for row in rows) == 1
    assert sum(bool(row["role"] == "assistant" and row.get("tool_calls")) for row in rows) == 1
    assert sum(row["role"] == "tool" and row.get("tool_call_id") == "call-1" for row in rows) == 1
    lifecycle = read_db.execution_route_lifecycle(agents[0].session_id).read_events()
    assert sorted(event.event_type.value for event in lifecycle) == sorted([
        "route_requested", "route_accepted", "route_started", "route_finished",
    ] * 2)
    read_db.close()


def test_oneshot_routed_fallback_budget_exhaustion_does_not_recurse(
    tmp_path, monkeypatch
):
    from agent import conversation_loop
    from hermes_cli import oneshot

    events = []
    db = SessionDB(tmp_path / "state.db")
    provider = _FallbackProvider(events, db)
    fallback_chain = ({"provider": "p1", "model": "m1"},)
    _patch_common(monkeypatch, oneshot, db, provider, fallback_chain, events)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs["requested"],
            "requested_provider": kwargs["requested"],
        },
    )
    runs = []
    constructions = []

    class Agent:
        reasoning_config = None

        def __init__(self, **kwargs):
            constructions.append(kwargs)
            self.model = kwargs["model"]
            self.provider = self.requested_provider = kwargs["provider"]
            self.session_id = kwargs["session_id"]
            self._session_db = kwargs["session_db"]
            self._fallback_index = 0
            self._last_flushed_db_idx = 0
            self._routed_restart_required = None

        def run_conversation(self, prompt, conversation_history=None):
            runs.append((prompt, conversation_history))
            self._session_db.append_message(self.session_id, role="user", content=prompt)
            rows = self._session_db.get_messages(self.session_id)
            rows[0][_DB_PERSISTED_MARKER] = True
            self._last_flushed_db_idx = 1
            self._routed_restart_required = RoutedAttemptRestartRequired("failed", 0)
            return {
                "messages": rows,
                "turn_id": "turn-1",
                "current_turn_user_idx": 0,
                "completed": False,
            }

        def _persist_session(self, messages):
            for row in messages:
                row[_DB_PERSISTED_MARKER] = True

        def release_clients(self):
            return None

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr(
        conversation_loop,
        "_continue_main_turn_from_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("continued")),
    )

    text, result = oneshot._run_agent("go")

    assert result["failed"] is True
    assert result["error"] == text == "main-turn fallback budget is exhausted"
    assert len(provider.requests) == 1
    assert len(constructions) == 1
    assert runs == [("go", None)]
    read_db = SessionDB(tmp_path / "state.db")
    assert sorted(
        event.event_type.value
        for event in read_db.execution_route_lifecycle(constructions[0]["session_id"]).read_events()
    ) == sorted(["route_requested", "route_accepted", "route_started", "route_finished"])
    read_db.close()


def test_oneshot_no_router_main_turn_and_native_fallback_are_unchanged(
    tmp_path, monkeypatch
):
    from agent import main_turn_continuation
    from hermes_cli import execution_router_runtime, model_switch, oneshot

    events = []
    db = SessionDB(tmp_path / "state.db")
    fallback_chain = ({"provider": "p1", "model": "m1"},)
    _patch_common(monkeypatch, oneshot, db, None, fallback_chain, events)
    real_db_close = db.close

    def close_db():
        events.append("store-close")
        real_db_close()

    monkeypatch.setattr(db, "close", close_db)
    monkeypatch.setattr(model_switch, "_ensure_direct_aliases", lambda: events.append("aliases"))
    monkeypatch.setattr(
        model_switch,
        "DIRECT_ALIASES",
        {"alias": model_switch.DirectAlias(
            "native-model", "native-provider", "https://example.test/", "secret"
        )},
    )
    monkeypatch.setattr(
        model_switch,
        "direct_alias_runtime_request",
        lambda _alias: events.append("alias-credentials") or ("custom", "alias-key"),
    )
    runtime_calls = []

    def resolve_runtime_provider(**kwargs):
        runtime_calls.append(dict(kwargs))
        events.append("runtime")
        return {
            "api_key": kwargs["explicit_api_key"],
            "base_url": kwargs["explicit_base_url"],
            "provider": "custom",
            "requested_provider": "custom",
            "api_mode": "chat_completions",
            "credential_pool": "native-pool",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_runtime_provider)

    construction = []
    run_calls = []

    class Agent:
        def __init__(self, **kwargs):
            construction.append(kwargs)
            self._session_messages = []

        def run_conversation(self, prompt, conversation_history=None):
            run_calls.append((prompt, conversation_history))
            return {"messages": [], "final_response": "native", "completed": True}

        def shutdown_memory_provider(self, messages):
            events.append(("memory-close", messages))

        def close(self):
            events.append("agent-close")

    monkeypatch.setattr("run_agent.AIAgent", Agent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("routed-only machinery reached the no-router path")

    monkeypatch.setattr(main_turn_continuation, "_continue_main_turn_attempt", forbidden)
    monkeypatch.setattr(execution_router_runtime, "bind_main_turn_attempt", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_started", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_finished", forbidden)
    monkeypatch.setattr(execution_router_runtime, "record_main_turn_not_started", forbidden)

    text, result = oneshot._run_agent("go", model="alias")

    assert text == result["final_response"] == "native"
    assert runtime_calls == [{
        "requested": "custom",
        "target_model": "native-model",
        "explicit_base_url": "https://example.test",
        "explicit_api_key": "alias-key",
    }]
    assert len(construction) == 1
    kwargs = construction[0]
    assert kwargs == {
        "api_key": "alias-key",
        "base_url": "https://example.test",
        "provider": "custom",
        "requested_provider": "custom",
        "api_mode": "chat_completions",
        "model": "native-model",
        "enabled_toolsets": ["terminal"],
        "quiet_mode": True,
        "platform": "cli",
        "session_db": db,
        "session_id": None,
        "credential_pool": "native-pool",
        "fallback_model": list(fallback_chain),
        "ephemeral_system_prompt": "skills",
        "clarify_callback": oneshot._oneshot_clarify_callback,
    }
    assert run_calls == [("go", None)]
    assert events.index("alias-credentials") < events.index("runtime")
    assert events[-3:] == [("memory-close", []), "agent-close", "store-close"]


def _assert_surface_stop_precredential(surface, tmp_path, monkeypatch):
    from hermes_cli import plugins

    class StopProvider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="router-plugin",
            plugin_version="1.0",
            provider_id="router-provider",
            contract_version="1.0",
            supported_execution_kinds=(ExecutionKind.MAIN_TURN,),
        )

        def __init__(self):
            self.calls = 0

        def resolve_execution_route(self, request, _cancellation):
            self.calls += 1
            return ExecutionRouteDecisionV1.stop(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                reason_code="blocked",
            )

    provider = StopProvider()
    registration = ExecutionRouterRegistration(
        provider, 1, lambda generation: generation == 1
    )
    monkeypatch.setattr(
        plugins,
        "get_plugin_manager",
        lambda: SimpleNamespace(
            get_execution_router_registration=lambda: registration
        ),
    )

    if surface == "cli":
        from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

        calls = []
        stopped = SimpleNamespace(
            may_start=False,
            resolution=SimpleNamespace(
                reason_text="router stopped this turn", reason_code="blocked"
            ),
        )

        def prepare(message):
            calls.append(("router", message))
            shell._main_turn_prepared_attempt = stopped
            return stopped

        shell = SimpleNamespace(
            _secret_capture_callback=None,
            _last_turn_interrupted=False,
            _chat_expand_context_references=lambda message: (
                calls.append(("normalize", message)) or (message, None)
            ),
            _prepare_main_turn_attempt=prepare,
            _ensure_runtime_credentials=lambda: calls.append("credentials") or True,
            _resolve_turn_agent_config=lambda _message: calls.append("config"),
            _init_agent=lambda **_kwargs: calls.append("agent") or True,
        )
        assert CLIChatTurnMixin.chat(shell, "hello") == "router stopped this turn"
        assert calls == [("normalize", "hello"), ("router", "hello")]
        return

    if surface == "tui":
        from tui_gateway import prompt_turn

        db = SessionDB(tmp_path / "tui-stop.db")
        db.create_session("tui-stop", source="tui")
        agent = SimpleNamespace(
            model="native-model",
            provider="native-provider",
            requested_provider="native-provider",
            reasoning_config={"effort": "medium"},
            _session_db=db,
            _fallback_chain=[],
        )
        session = {"session_key": "tui-stop", "agent": agent}
        prepared = prompt_turn._prepare_tui_main_turn_attempt("sid", session, "hello")
        assert prepared.may_start is False
        assert provider.calls == 1
        assert session["agent"] is agent
        db.close()
        return

    if surface == "gateway":
        turn, _runner, _ctx, db, calls, _streamed, _interim = _gateway_harness(
            tmp_path / "gateway-stop", monkeypatch, provider, ()
        )
        result = turn.run_sync()
        assert result["final_response"] == "blocked"
        assert provider.calls == 1
        assert not any(
            isinstance(call, tuple) and call[0] in {"route-credentials", "agent"}
            for call in calls
        )
        db.close()
        return

    if surface == "oneshot":
        from hermes_cli import oneshot

        db = SessionDB(tmp_path / "oneshot-stop.db")
        choice = SimpleNamespace(model="native-model", provider="native-provider")
        prepared, session_id = oneshot._prepare_oneshot_main_turn_attempt(
            "hello", choice, db, None, registration=registration
        )
        assert prepared.may_start is False
        assert provider.calls == 1
        assert db.get_session(session_id) is not None
        db.close()
        return

    raise AssertionError(surface)
