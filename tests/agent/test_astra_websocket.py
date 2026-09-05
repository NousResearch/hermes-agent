"""Deterministic coverage for the direct GPT-6 Astra Responses WebSocket lane."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from agent.transports.astra_websocket_session import (
    AstraDeliveryUncertainError,
    AstraPreDispatchError,
    AstraSteeringPersistenceError,
    AstraWebSocketSession,
    is_astra_websocket_eligible,
    run_astra_websocket_stream,
)


def _agent(**overrides):
    values = dict(
        api_mode="codex_responses", model="gpt-6-astra", provider="openai",
        base_url="https://api.openai.com/v1", api_key="placeholder-key", auth_mode="api_key",
        is_subagent=False, compression_checkpoint_required=False, _interrupt_requested=False,
        _pending_steer=None, _pending_steer_lock=threading.Lock(), _session_messages=[],
        _codex_streamed_text_parts=[], _session_db=FakeSessionDB(), session_id="synthetic-session",
        _session_db_created=True,
    )
    values.update(overrides)
    values["_is_codex_backend"] = lambda: False
    values["_fire_stream_delta"] = lambda text: None
    values["_fire_reasoning_delta"] = lambda text: None
    values["_touch_activity"] = lambda text: None
    return SimpleNamespace(**values)


class FakeWebSocket:
    def __init__(self, on_send=None):
        self.events = []
        self.sent = []
        self.closed = False
        self._changed = threading.Condition()
        self._on_send = on_send

    def send(self, payload):
        message = json.loads(payload)
        self.sent.append(message)
        if self._on_send:
            self._on_send(message, self)

    def recv(self):
        with self._changed:
            if not self.events:
                self._changed.wait(timeout=1)
            if not self.events:
                raise ConnectionError("fake socket exhausted")
            return json.dumps(self.events.pop(0))

    def push(self, event):
        with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    def close(self):
        self.closed = True
        with self._changed:
            self._changed.notify_all()


class FakeSessionDB:
    """Small model_config seam; the real SessionDB supplies the same atomic helpers."""

    def __init__(self, entries=None):
        self.model_config = {"_astra_steering": {"version": 1, "entries": list(entries or [])}}
        self.patches = []

    def get_session_model_config_value(self, session_id, key, default=None):
        del session_id
        return self.model_config.get(key, default)

    def patch_session_model_config(self, session_id, patch):
        del session_id
        self.patches.append(patch)
        self.model_config.update(patch)


def _connect_socket(socket):
    def connect(url, key, timeout):
        assert url == "wss://api.openai.com/v1/responses"
        assert key == "placeholder-key"
        return socket
    return connect


def test_default_connect_uses_static_key_and_responses_feature_header(monkeypatch):
    calls = {}
    client_module = types.ModuleType("websockets.sync.client")

    def connect(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return object()

    client_module.connect = connect
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client_module)
    from agent.transports.astra_websocket_session import _default_connect

    assert _default_connect("wss://api.openai.com/v1/responses", "placeholder-key", 5) is not None
    assert calls == {
        "url": "wss://api.openai.com/v1/responses",
        "kwargs": {
            "additional_headers": {
                "Authorization": "Bearer placeholder-key",
                "OpenAI-Beta": "responses_websockets=2026-02-06",
            },
            "open_timeout": 5,
        },
    }


def _created(response_id):
    return {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}, "id": f"created-{response_id}"}


def _completed(response_id, event_id, text="done"):
    return [
        {"type": "response.output_item.added", "response_id": response_id, "output_index": 0,
         "item": {"id": f"item-{response_id}", "type": "message", "role": "assistant"}, "id": f"added-{event_id}"},
        {"type": "response.output_text.delta", "response_id": response_id, "delta": text,
         "id": f"delta-{event_id}"},
        {"type": "response.output_item.done", "response_id": response_id, "output_index": 0,
         "item": {"id": f"item-{response_id}", "type": "message", "role": "assistant", "status": "completed",
                  "content": [{"type": "output_text", "text": text}]}, "id": f"done-{event_id}"},
        {"type": "response.completed", "response_id": response_id,
         "response": {"id": response_id, "status": "completed"}, "id": f"complete-{event_id}"},
    ]


def test_eligibility_is_exact_and_excludes_unsupported_routes():
    agent = _agent()
    assert is_astra_websocket_eligible(agent, {"model": "gpt-6-astra"})
    for changes in (
        {"auth_mode": "chatgpt"}, {"base_url": "https://sub.api.openai.com/v1"},
        {"base_url": "https://api.openai.com/v1/other"}, {"model": "gpt-5.6"},
        {"provider": "openai-codex"}, {"context_management": {"type": "compaction"}},
        {"previous_response_id": "resp_1"},
        {"is_subagent": True}, {"api_key": ""},
    ):
        request = {key: value for key, value in changes.items() if key in {"context_management", "previous_response_id"}}
        agent_changes = {key: value for key, value in changes.items() if key not in request}
        assert not is_astra_websocket_eligible(_agent(**agent_changes), request), changes


def test_initial_create_uses_existing_body_without_stream():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    socket.events.extend(_completed("r1", "r1"))
    agent = _agent()
    request = {"model": "gpt-6-astra", "input": [{"role": "user", "content": "hi"}], "stream": True,
               "reasoning": {"effort": "low"}}

    result = AstraWebSocketSession(agent, connect=_connect_socket(socket)).run(request)

    assert result.status == "completed"
    assert socket.sent == [{"type": "response.create", "model": "gpt-6-astra",
                            "input": [{"role": "user", "content": "hi"}], "reasoning": {"effort": "low"}}]


def test_terminal_ws_response_settles_pr2_async_executor_once():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    socket.events.extend(_completed("r1", "r1"))
    calls = []

    class Executor:
        has_admitted = True
        has_pending = False
        failed = False

        def finish_stream(self, **kwargs):
            calls.append(kwargs)
            return True

    agent = _agent(_astra_async_executor=Executor())
    result = AstraWebSocketSession(agent, connect=_connect_socket(socket)).run({"model": "gpt-6-astra"})

    assert result.status == "completed"
    assert len(calls) == 1
    assert calls[0]["assistant_content"] == "done"


def test_terminal_processing_fences_late_steer_before_assembler_feed():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    socket.push({"type": "response.completed", "response_id": "r1",
                 "response": {"id": "r1", "status": "completed"}, "id": "complete"})
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))
    feed_started = threading.Event()
    release_feed = threading.Event()

    class BlockingAssembler:
        def feed(self, event):
            feed_started.set()
            release_feed.wait(timeout=2)
            return True

        def result(self):
            return SimpleNamespace(status="completed", output=[], output_text="", model="gpt-6-astra")

    session._new_assembler = lambda _response_id: BlockingAssembler()
    outcome = []

    def run():
        outcome.append(session.run({"model": "gpt-6-astra"}))

    worker = threading.Thread(target=run)
    worker.start()
    assert feed_started.wait(timeout=2)
    assert session.request_steer("too late") is False
    assert not any(item["type"] == "response.steer" for item in socket.sent)
    release_feed.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert outcome and outcome[0].status == "completed"


def test_steer_interleaving_before_terminal_lock_waits_for_successor():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    socket.push({"type": "response.completed", "response_id": "r1",
                 "response": {"id": "r1", "status": "completed"}, "id": "complete-r1"})
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    def on_send(message, ws):
        if message["type"] == "response.steer":
            ws.push({"type": "response.steer.accepted", "id": "ack"})
            ws.push(_created("r2"))
            for frame in _completed("r2", "r2", "success"):
                ws.push(frame)

    socket._on_send = on_send
    outcome = []
    reason_started = threading.Event()
    release_reason = threading.Event()

    original_reason = session._terminal_reason

    def blocked_reason(event):
        reason_started.set()
        release_reason.wait(timeout=2)
        return original_reason(event)

    session._terminal_reason = blocked_reason
    worker = threading.Thread(target=lambda: outcome.append(
        session.run({"model": "gpt-6-astra"}),
    ))
    worker.start()
    assert reason_started.wait(timeout=2)
    release_reason.set()
    assert session.request_steer("arrived at the terminal boundary") is False
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert outcome and outcome[0].output_text == ""
    assert not any(item["type"] == "response.steer" for item in socket.sent)


def test_steer_acceptance_successor_and_predecessor_fencing():
    socket = FakeWebSocket()
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    def on_send(message, ws):
        if message["type"] == "response.create":
            ws.push(_created("r1"))
        elif message["type"] == "response.steer":
            ws.push({"type": "response.steer.accepted", "id": "ack-1"})
            ws.push({"type": "response.incomplete", "response_id": "r1", "response": {
                "id": "r1", "status": "incomplete", "incomplete_details": {"reason": "steered"},
            }, "id": "incomplete-r1"})
            ws.push(_created("r2"))
            ws.push({"type": "response.output_text.delta", "response_id": "r1", "delta": "stale", "id": "stale"})
            for frame in _completed("r2", "r2", "success"):
                ws.push(frame)

    socket._on_send = on_send

    def run():
        session.run({"model": "gpt-6-astra", "input": [{"role": "user", "content": "hi"}]})

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    assert session.request_steer("use the second plan") is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    steer = next(item for item in socket.sent if item["type"] == "response.steer")
    assert set(steer) == {"type", "previous_response_id", "input"}
    assert steer["previous_response_id"] == "r1"
    assert agent._pending_steer is None


def test_pending_required_input_continues_once_from_saved_result():
    socket = FakeWebSocket()
    agent = _agent(_session_messages=[{"role": "tool", "tool_call_id": "call-1", "content": "saved result"}])

    def on_send(message, ws):
        if message["type"] == "response.create" and "previous_response_id" not in message:
            ws.push(_created("r1"))
        elif message["type"] == "response.steer":
            ws.push({"type": "response.steer.accepted", "id": "ack"})
            ws.push({"type": "response.steer.pending", "response_id": "r1", "required_input": [
                {"type": "function_call_output", "call_id": "call-1", "output": ""},
            ], "id": "pending"})
        elif message["type"] == "response.create" and message.get("previous_response_id") == "r1":
            assert message["input"][0]["output"] == "saved result"
            ws.push(_created("r2"))
            for frame in _completed("r2", "r2", "continued"):
                ws.push(frame)

    socket._on_send = on_send
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    worker = threading.Thread(target=lambda: session.run({"model": "gpt-6-astra", "input": [{"role": "user", "content": "hi"}]}))
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    assert session.request_steer("continue") is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    creates = [item for item in socket.sent if item["type"] == "response.create"]
    assert len(creates) == 2
    assert sum(item["type"] == "response.steer" for item in socket.sent) == 1


def test_explicit_steer_failure_keeps_one_legacy_redirect():
    socket = FakeWebSocket()
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    def on_send(message, ws):
        if message["type"] == "response.create":
            ws.push(_created("r1"))
        elif message["type"] == "response.steer":
            ws.push({"type": "response.steer.failed", "id": "failed"})
            for frame in _completed("r1", "r1"):
                ws.push(frame)

    socket._on_send = on_send
    def run_interruptible():
        try:
            session.run({"model": "gpt-6-astra"})
        except InterruptedError:
            pass

    worker = threading.Thread(target=run_interruptible)
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    assert session.request_steer("legacy once") is True
    worker.join(timeout=2)
    assert agent._pending_steer is None
    assert agent._pending_redirect == "legacy once"
    assert sum(item["type"] == "response.steer" for item in socket.sent) == 1
    assert session._steering_receipts[-1]["state"] == "failed"


def test_connect_failure_falls_back_but_send_or_recv_is_uncertain():
    agent = _agent()
    with pytest.raises(AstraPreDispatchError):
        AstraWebSocketSession(agent, connect=lambda *_: (_ for _ in ()).throw(ConnectionError("offline"))).run({})

    socket = FakeWebSocket()
    socket.push(_created("r1"))
    socket._on_send = lambda message, ws: (_ for _ in ()).throw(ConnectionError("lost after send")) if message["type"] == "response.create" else None
    with pytest.raises(AstraDeliveryUncertainError):
        AstraWebSocketSession(agent, connect=_connect_socket(socket)).run({})


def test_steering_admission_is_persisted_before_socket_dispatch_and_owned_by_successor():
    socket = FakeWebSocket()
    db = FakeSessionDB()
    agent = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    observed = {}

    def on_send(message, ws):
        if message["type"] == "response.create":
            ws.push(_created("r1"))
        elif message["type"] == "response.steer":
            entry = db.model_config["_astra_steering"]["entries"][-1]
            observed["before_send_state"] = entry["state"]
            observed["input"] = entry["input"]
            ws.push({"type": "response.steer.accepted", "id": "ack"})
            ws.push({"type": "response.incomplete", "response_id": "r1", "response": {
                "id": "r1", "status": "incomplete", "incomplete_details": {"reason": "steered"},
            }, "id": "incomplete"})
            ws.push(_created("r2"))
            for frame in _completed("r2", "r2", "success"):
                ws.push(frame)

    socket._on_send = on_send
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))
    worker = threading.Thread(target=lambda: session.run({"model": "gpt-6-astra"}))
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    assert session.request_steer("synthetic correction") is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert observed["before_send_state"] == "prepared"
    assert observed["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "synthetic correction"}]}]
    receipt = db.model_config["_astra_steering"]["entries"][-1]
    assert receipt["response_id"] == "r1"
    assert receipt["generation"] == 1
    assert receipt["admission_id"]
    assert receipt["state"] == "successor_created"


def test_restart_does_not_resend_accepted_or_ambiguous_steering():
    entries = [
        {"version": 1, "admission_id": "a", "generation": 1, "response_id": "r1",
         "input": [{"role": "user", "content": [{"type": "input_text", "text": "accepted"}]}],
         "text": "accepted", "state": "accepted"},
        {"version": 1, "admission_id": "b", "generation": 2, "response_id": "r1",
         "input": [{"role": "user", "content": [{"type": "input_text", "text": "ambiguous"}]}],
         "text": "ambiguous", "state": "ambiguous"},
    ]
    db = FakeSessionDB(entries)
    agent = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    session = AstraWebSocketSession(agent)
    session._socket = FakeWebSocket()
    session._response_id = "r2"
    session._set_state("ACTIVE")
    assert agent._pending_steer is None
    assert session.request_steer("new correction") is True
    sent = session._socket.sent
    assert len(sent) == 1
    assert sent[0]["input"][0]["content"][0]["text"] == "new correction"
    assert sum(item["state"] == "ambiguous" for item in db.model_config["_astra_steering"]["entries"]) == 1

    full_db = FakeSessionDB([
        {"version": 1, "admission_id": str(index), "generation": index, "response_id": "r1",
         "input": [{"role": "user", "content": [{"type": "input_text", "text": str(index)}]}],
         "text": str(index), "state": "accepted"}
        for index in range(1, 17)
    ])
    full_agent = _agent(_session_db=full_db)
    full_session = AstraWebSocketSession(full_agent)
    full_session._socket = FakeWebSocket()
    full_session._response_id = "r2"
    full_session._set_state("ACTIVE")
    with pytest.raises(AstraSteeringPersistenceError):
        full_session.request_steer("journal is full")
    assert len(full_db.model_config["_astra_steering"]["entries"]) == 16
    assert all(item["state"] == "accepted" for item in full_db.model_config["_astra_steering"]["entries"])


def test_explicit_failure_restores_redirect_until_durable_delivery():
    entry = {"version": 1, "admission_id": "failed", "generation": 1, "response_id": "r1",
             "input": [{"role": "user", "content": [{"type": "input_text", "text": "retry once"}]}],
             "text": "retry once", "state": "failed"}
    db = FakeSessionDB([entry])
    first = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    AstraWebSocketSession(first)
    assert first._pending_redirect == "retry once"
    assert db.model_config["_astra_steering"]["entries"][0]["state"] == "failed"
    AstraWebSocketSession(first)
    assert first._pending_redirect == "retry once"
    second = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    AstraWebSocketSession(second)
    assert second._pending_redirect == "retry once"


def test_persistence_failure_before_steer_send_is_fail_closed():
    class FailingDB(FakeSessionDB):
        def patch_session_model_config(self, session_id, patch):
            raise OSError("synthetic persistence failure")

    socket = FakeWebSocket()
    db = FailingDB()
    agent = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))
    session._socket = socket
    session._response_id = "r1"
    session._set_state("ACTIVE")
    with pytest.raises(AstraSteeringPersistenceError):
        session.request_steer("must not dispatch")
    assert socket.sent == []
    assert agent._pending_steer is None


def test_missing_or_noop_session_store_is_fail_closed_before_steer_send():
    class NoOpDB(FakeSessionDB):
        def patch_session_model_config(self, session_id, patch):
            del session_id, patch

    for db in (None, NoOpDB()):
        socket = FakeWebSocket()
        agent = _agent(_session_db=db)
        session = AstraWebSocketSession(agent)
        session._socket = socket
        session._response_id = "r1"
        session._set_state("ACTIVE")
        with pytest.raises(AstraSteeringPersistenceError):
            session.request_steer("must not dispatch")
        assert socket.sent == []


def test_redacted_steer_declines_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        "agent.transports.astra_websocket_session._safe_steering_text",
        lambda text: "[steering input redacted]",
    )
    socket = FakeWebSocket()
    db = FakeSessionDB()
    agent = _agent(_session_db=db)
    session = AstraWebSocketSession(agent)
    session._socket = socket
    session._response_id = "r1"
    session._set_state("ACTIVE")
    with pytest.raises(AstraSteeringPersistenceError):
        session.request_steer("credential-like steering input")
    assert socket.sent == []
    assert db.model_config["_astra_steering"]["entries"] == []


def test_ambiguous_steer_send_retains_durable_no_resend_receipt():
    socket = FakeWebSocket()
    socket._on_send = lambda message, ws: (_ for _ in ()).throw(ConnectionError("lost after steer"))
    db = FakeSessionDB()
    agent = _agent(_session_db=db, session_id="synthetic-session", _session_db_created=True)
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))
    session._socket = socket
    session._response_id = "r1"
    session._set_state("ACTIVE")

    with pytest.raises(AstraDeliveryUncertainError):
        session.request_steer("ambiguous correction")
    receipt = db.model_config["_astra_steering"]["entries"][-1]
    assert receipt["state"] == "ambiguous"
    assert agent._pending_steer is None


def test_duplicate_frames_do_not_duplicate_output_items():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    frames = _completed("r1", "r1", "once")
    socket.events.extend(frames[:3] + [frames[2], frames[3]])
    agent = _agent()
    result = AstraWebSocketSession(agent, connect=_connect_socket(socket)).run({"model": "gpt-6-astra"})
    assert len(result.output) == 1
    assert result.output_text == "once"


def test_duplicate_sequence_number_is_fenced_before_assembler_effects():
    socket = FakeWebSocket()
    socket.push(_created("r1"))
    frames = _completed("r1", "r1", "once")
    frames[1]["sequence_number"] = 17
    duplicate = dict(frames[1])
    duplicate["id"] = "different-event-id"
    duplicate["delta"] = "duplicate"
    duplicate["sequence_number"] = 17
    socket.events.extend([frames[0], frames[1], duplicate, frames[2], frames[3]])
    agent = _agent()

    result = AstraWebSocketSession(agent, connect=_connect_socket(socket)).run({"model": "gpt-6-astra"})

    assert result.output_text == "once"


def test_successor_sequence_restart_does_not_drop_successor_frames():
    socket = FakeWebSocket()
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    def on_send(message, ws):
        # The provider can restart sequence numbering for a later response generation.
        if message["type"] == "response.create":
            if message.get("model") == "gpt-6-astra":
                ws.push({**_created("r1"), "sequence_number": 1})
        elif message["type"] == "response.steer":
            ws.push({"type": "response.steer.accepted", "id": "ack", "sequence_number": 6})
            ws.push({"type": "response.incomplete", "response_id": "r1", "response": {
                "id": "r1", "status": "incomplete", "incomplete_details": {"reason": "steered"},
            }, "id": "incomplete-r1", "sequence_number": 7})
            ws.push({**_created("r2"), "sequence_number": 1})
            successor = _completed("r2", "r2", "new")
            for offset, frame in enumerate(successor, 2):
                frame["sequence_number"] = offset
                ws.push(frame)

    socket._on_send = on_send
    # Initial response is supplied by the callback; make the first turn wait for the active state.
    worker = threading.Thread(target=lambda: session.run({"model": "gpt-6-astra"}))
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    assert session.request_steer("new generation") is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert session.response_id == "r2"


def test_interrupt_closes_lane_without_reconnect():
    socket = FakeWebSocket()
    agent = _agent()
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))
    socket.push(_created("r1"))
    def run_interruptible():
        try:
            session.run({"model": "gpt-6-astra"})
        except InterruptedError:
            pass

    worker = threading.Thread(target=run_interruptible)
    worker.start()
    deadline = time.time() + 2
    while session.state != "ACTIVE" and time.time() < deadline:
        time.sleep(0.005)
    session.request_interrupt()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert socket.closed
