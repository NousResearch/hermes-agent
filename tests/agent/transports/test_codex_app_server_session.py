"""Tests for CodexAppServerSession — drive turns through a mock client.

The session adapter has the most complex behavior of the three new modules:
notification draining, server-request handling (approvals), interrupt,
deadline timeouts. These tests pin all of that without spawning real codex.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch
from typing import Any, Optional

import pytest

import agent.transports.codex_app_server_session as session_mod
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    _ServerRequestRouting,
    _approval_choice_to_codex_decision,
    _coerce_turn_input,
)


class FakeClient:
    """Stand-in for CodexAppServerClient that records calls and lets the test
    drive the notification / server-request streams synchronously."""

    def __init__(self, *, codex_bin: str = "codex", codex_home=None) -> None:
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.requests: list[tuple[str, dict]] = []
        self.notifications_responses: list[dict] = []
        self.responses: list[tuple[Any, dict]] = []
        self.error_responses: list[tuple[Any, int, str]] = []
        self._initialized = False
        self._closed = False
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._request_handler = None  # Optional[Callable[[str, dict], dict]]

    # API matching CodexAppServerClient
    def initialize(self, **kwargs):
        self._initialized = True
        return {"userAgent": "fake/0.0.0", "codexHome": "/tmp",
                "platformOs": "linux", "platformFamily": "unix"}

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0):
        self.requests.append((method, params or {}))
        if self._request_handler is not None:
            return self._request_handler(method, params or {})
        # Sensible defaults for protocol methods used by the session
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"},
                    "activePermissionProfile": {"id": "workspace-write"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-fake-001"}}
        if method == "turn/interrupt":
            return {}
        if method == "turn/steer":
            return {"turnId": (params or {}).get("expectedTurnId")}
        return {}

    def notify(self, method: str, params=None):
        pass

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None):
        self.error_responses.append((request_id, code, message))

    def take_notification(self, timeout: float = 0.0):
        if self._notifications:
            return self._notifications.pop(0)
        # Honor a tiny sleep so the loop doesn't hot-spin; the real client
        # blocks on a queue. For tests we want determinism.
        if timeout > 0:
            time.sleep(min(timeout, 0.001))
        return None

    def take_server_request(self, timeout: float = 0.0):
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def close(self):
        self._closed = True

    def is_alive(self) -> bool:
        # Fake is "alive" until close() is called; tests that want a dead
        # subprocess can patch this attribute or call close() directly.
        return not self._closed

    def stderr_tail(self, n: int = 20):
        return list(getattr(self, "_stderr_tail", []))[-n:]

    # Test helpers
    def queue_notification(self, method: str, **params):
        # Keep legacy fixture shorthand aligned with the IDs returned by the
        # fake thread/start and turn/start responses.
        if params.get("threadId") in {"t", "th"}:
            params["threadId"] = "thread-fake-001"
        if params.get("turnId") == "tu1":
            params["turnId"] = "turn-fake-001"
        turn = params.get("turn")
        if isinstance(turn, dict) and turn.get("id") == "tu1":
            turn = dict(turn)
            turn["id"] = "turn-fake-001"
            params["turn"] = turn
        self._notifications.append({"method": method, "params": params})

    def queue_server_request(self, method: str, request_id: Any = "srv-1", **params):
        if params.get("threadId") in {"t", "th"}:
            params["threadId"] = "thread-fake-001"
        if params.get("turnId") == "tu1":
            params["turnId"] = "turn-fake-001"
        self._server_requests.append({"id": request_id, "method": method, "params": params})

    def set_stderr_tail(self, lines):
        """Test helper: seed stderr_tail() output for OAuth-refresh classifier tests."""
        self._stderr_tail = list(lines)

    def complete_after_response(self):
        """An approval-dependent turn cannot finish before its approval response."""
        original = self.respond

        def respond(request_id, result):
            original(request_id, result)
            self.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "completed"})

        self.respond = respond


def make_session(client: FakeClient, **kwargs) -> CodexAppServerSession:
    return CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **kw: client,
        **kwargs,
    )


@pytest.mark.parametrize("method", list(CodexAppServerSession._SERVER_REQUEST_HANDLERS))
@pytest.mark.parametrize("scope", [
    {}, {"threadId": "thread-fake-001"}, {"turnId": "turn-fake-001"},
    {"threadId": "", "turnId": "turn-fake-001"},
    {"threadId": "thread-fake-001", "turnId": None},
    {"threadId": "foreign", "turnId": "turn-fake-001"},
    {"threadId": "thread-fake-001", "turnId": "stale"},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "thread_id": "foreign"},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "turn": {"id": "stale"}},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "item": {"turn_id": "stale"}},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "item": {"threadId": None}},
])
def test_server_request_authority_requires_unambiguous_exact_scope(method, scope):
    client = FakeClient()
    calls = []
    session = make_session(client, approval_callback=lambda *a, **kw: calls.append(a) or "once",
                           request_routing=_ServerRequestRouting(True, True))
    session.ensure_started()
    session._active_turn_id = "turn-fake-001"
    session._handle_server_request({"id": "ambiguous", "method": method,
                                    "params": {"command": "fixture", "serverName": "hermes-tools", **scope}})
    assert not calls
    assert not client.responses
    assert client.error_responses and client.error_responses[0][:2] == ("ambiguous", -32600)


@pytest.mark.parametrize("active_turn_id", [None, "", 7])
def test_server_request_cannot_inherit_invalid_active_identity(active_turn_id):
    client = FakeClient()
    session = make_session(client, request_routing=_ServerRequestRouting(True, True))
    session.ensure_started()
    session._active_turn_id = active_turn_id
    session._handle_server_request({"id": "invalid", "method": "item/commandExecution/requestApproval",
                                    "params": {"threadId": "thread-fake-001", "turnId": active_turn_id}})
    assert not client.responses
    assert client.error_responses[0][:2] == ("invalid", -32600)


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("method", list(CodexAppServerSession._SERVER_REQUEST_HANDLERS))
def test_scoped_server_requests_reach_only_the_current_approval_route(bypass, method):
    client = FakeClient()
    calls = []
    session = make_session(client, approval_callback=lambda *a, **kw: calls.append(a) or "deny",
                           request_routing=_ServerRequestRouting(bypass, bypass))
    session.ensure_started()
    session._active_turn_id = "turn-fake-001"
    for scope in (
        {"threadId": "thread-fake-001", "turnId": "turn-fake-001"},
        {"thread_id": "thread-fake-001", "turn_id": "turn-fake-001"},
        {"turn": {"threadId": "thread-fake-001", "id": "turn-fake-001"}},
        {"item": {"thread_id": "thread-fake-001", "turn_id": "turn-fake-001"}},
    ):
        session._handle_server_request({"id": "scoped", "method": method,
                                        "params": {"command": "fixture", "serverName": "hermes-tools", **scope}})
    assert not client.error_responses
    assert len(client.responses) == 4
    if method == "mcpServer/elicitation/request":
        expected = {"action": "accept", "content": None, "_meta": None}
    else:
        expected = {"decision": "accept" if bypass and method != "item/permissions/requestApproval" else "decline"}
    assert all(response == ("scoped", expected) for response in client.responses)
    assert len(calls) == (4 if not bypass and method in {
        "item/commandExecution/requestApproval", "item/fileChange/requestApproval"} else 0)


@pytest.mark.parametrize("turn_timeout,ack_delay", [(600.0, 12.0), (20.0, 12.0), (5.0, 12.0), (600.0, 61.0)])
def test_turn_start_acknowledgement_uses_caller_capped_window(turn_timeout, ack_delay):
    client = FakeClient()
    original = client.request
    budgets = []

    def request(method, params=None, timeout=30):
        if method == "turn/start":
            budgets.append(timeout)
            if ack_delay > timeout:
                raise TimeoutError("delayed acknowledgement exceeded request budget")
            client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "completed"})
        return original(method, params, timeout)

    client.request = request
    result = make_session(client).run_turn("fixture", turn_timeout=turn_timeout)
    assert budgets == [min(turn_timeout, 60.0)]
    if ack_delay > min(turn_timeout, 60.0):
        assert result.should_retire and not result.terminal_acknowledged
        assert "timed out" in result.error
    else:
        assert result.error is None and result.terminal_acknowledged


# ---- choice mapping ----

class TestApprovalChoiceMapping:
    @pytest.mark.parametrize("choice,expected", [
        ("once", "accept"),
        ("session", "acceptForSession"),
        ("always", "acceptForSession"),
        ("deny", "decline"),
        ("anything-else", "decline"),
    ])
    def test_mapping(self, choice, expected):
        assert _approval_choice_to_codex_decision(choice) == expected


class TestTurnInputCoercion:
    def test_list_content_keeps_text_and_images(self):
        parts = _coerce_turn_input([
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert parts == [{"type": "text", "text": "caption"},
                         {"type": "image", "url": "data:image/png;base64,abc"}]


@pytest.mark.parametrize("blocked_method", ["initialize", "thread/start", "thread/resume", "turn/start", "thread/compact/start"])
def test_stop_during_submission_retires_without_admitting_later_work(blocked_method):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    client = FakeClient()
    original_request, original_initialize = client.request, client.initialize

    def block():
        entered.set()
        assert release.wait(12), "test did not release blocked RPC"

    def initialize(**kwargs):
        if blocked_method == "initialize":
            block()
            returned.set()
        return original_initialize(**kwargs)

    def request(method, params=None, timeout=30):
        if method == blocked_method:
            block()
            returned.set()
        if method == "thread/resume":
            client.requests.append((method, params))
            return {"thread": {"id": "saved-thread"}}
        return original_request(method, params, timeout)

    client.initialize, client.request = initialize, request
    session = make_session(client, thread_id="saved-thread" if blocked_method == "thread/resume" else None)
    results = []
    def target():
        if blocked_method == "thread/compact/start":
            return session.compact_thread(turn_timeout=0.1)
        return session.run_turn("no later work", turn_timeout=0.1)

    worker = threading.Thread(target=lambda: results.append(target()), daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        session.request_interrupt()
        first_deadline = session._interrupt_deadline
        session.request_interrupt()
        assert session._interrupt_deadline == first_deadline
        worker.join(timeout=5)
        assert not worker.is_alive(), "Stop did not bound the blocked startup/submission RPC"
        assert time.monotonic() - started < 5
        assert results[0].interrupted and results[0].should_retire
        assert not results[0].terminal_acknowledged
        assert client._closed
    finally:
        release.set()
        assert returned.wait(2)
        worker.join(timeout=12)
        session.close()
    if blocked_method != "turn/start":
        assert not any(method == "turn/start" for method, _ in client.requests)
    request_count = len(client.requests)
    assert session.run_turn("must remain retired").should_retire
    assert len(client.requests) == request_count


def test_interrupt_waits_for_matching_terminal_notification():
    client = FakeClient()
    session = make_session(client)

    def request(method, params):
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"}}
        if method == "turn/start":
            session.request_interrupt()
            return {"turn": {"id": "turn-fake-001"}}
        return {}

    client._request_handler = request
    client.queue_notification("turn/completed", threadId="foreign", turn={"id": "foreign", "status": "interrupted"})
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "interrupted"})
    result = session.run_turn("stop fixture", turn_timeout=0.1)
    assert client._notifications == []
    assert result.interrupted and result.terminal_acknowledged
    assert sum(method == "turn/interrupt" for method, _ in client.requests) == 1


@pytest.mark.parametrize("turn_timeout", [2.0, 30.0])
def test_interrupt_rpc_and_ack_share_one_deadline(monkeypatch, turn_timeout):
    clock = [100.0]
    monkeypatch.setattr(session_mod.time, "monotonic", lambda: clock[0])
    client = FakeClient()
    session = make_session(client)
    original_request = client.request
    budgets = []

    def request(method, params=None, timeout=30):
        if method == "turn/start":
            session.request_interrupt()
        if method == "turn/interrupt":
            budgets.append(timeout)
            clock[0] += timeout
            raise TimeoutError("missing interrupt RPC response")
        return original_request(method, params, timeout)

    def poll(timeout=0):
        clock[0] += max(timeout, 0.1)
        return None

    client.request = request
    client.take_notification = poll
    result = session.run_turn("stop", turn_timeout=turn_timeout)
    assert clock[0] <= 100.0 + min(turn_timeout, 5.0)
    assert budgets == [min(turn_timeout, 5.0)]
    assert result.interrupted and result.should_retire
    assert not result.terminal_acknowledged
    assert "unacknowledged" in result.error


@pytest.mark.parametrize("stop_during_approval", [False, True])
def test_stop_cannot_be_extended_by_approval(stop_during_approval):
    client = FakeClient()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    results = []

    def approve(*args, **kwargs):
        entered.set()
        release.wait(10)
        return "once"

    session = make_session(client, approval_callback=approve)
    original_request = client.request

    def request(method, params=None, timeout=30):
        if method == "turn/start" and not stop_during_approval:
            session.request_interrupt()
        if method == "turn/interrupt":
            client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "interrupted"})
        return original_request(method, params, timeout)

    client.request = request
    client.queue_server_request("item/commandExecution/requestApproval", command="fixture", threadId="t", turnId="tu1")

    def run():
        try:
            results.append(session.run_turn("stop", turn_timeout=30))
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        if stop_during_approval:
            assert entered.wait(2)
            session.request_interrupt()
        assert finished.wait(2), "approval blocked native cancellation"
        assert results[0].terminal_acknowledged and results[0].interrupted
        assert not any(response.get("decision") != "decline" for _, response in client.responses)
        assert len(client.responses) + len(client.error_responses) == 1
        assert entered.is_set() == stop_during_approval
    finally:
        release.set()
        worker.join(2)
    assert not any(response.get("decision") != "decline" for _, response in client.responses)


@pytest.mark.parametrize("waiting_for_lock", [False, True])
def test_stop_releases_real_cli_approval_wait(waiting_for_lock):
    from hermes_cli.cli_modal_mixin import CLIModalMixin

    entered, painted, callback_done = threading.Event(), threading.Event(), threading.Event()

    class UI(CLIModalMixin):
        _approval_state = None
        _approval_lock = threading.Lock()

        def _paint_now(self):
            if self._approval_state is not None:
                painted.set()

        def _ring_bell(self, **kwargs):
            pass

        def _persist_prompt_summary(self, *args):
            pass

    ui = UI()

    def approve(*args, **kwargs):
        entered.set()
        try:
            return ui._approval_callback(*args, **kwargs)
        finally:
            callback_done.set()

    client = FakeClient()
    session = make_session(client, approval_callback=approve)
    original_request = client.request

    def request(method, params=None, timeout=30):
        if method == "turn/interrupt":
            client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "interrupted"})
        return original_request(method, params, timeout)

    client.request = request
    client.queue_server_request("item/commandExecution/requestApproval", command="fixture", threadId="t", turnId="tu1")
    worker = threading.Thread(target=lambda: session.run_turn("stop", turn_timeout=30), daemon=True)
    if waiting_for_lock:
        ui._approval_lock.acquire()
    worker.start()
    try:
        assert (entered if waiting_for_lock else painted).wait(2)
        session.request_interrupt()
        worker.join(2)
        assert not worker.is_alive()
        assert callback_done.wait(2), "cancelled native approval still owns/waits for the CLI modal"
        assert ui._approval_state is None
        assert client.responses == [("srv-1", {"decision": "decline"})]
    finally:
        if waiting_for_lock:
            ui._approval_lock.release()
            painted.wait(2)
        if ui._approval_state is not None:
            ui._approval_state["response_queue"].put("deny")
        worker.join(2)
        callback_done.wait(2)


def test_approval_worker_preserves_turn_context():
    from contextvars import ContextVar

    scope = ContextVar("native-approval-scope", default=None)
    observed = []
    client = FakeClient()
    client.queue_server_request("item/commandExecution/requestApproval", command="fixture", threadId="t", turnId="tu1")
    client.complete_after_response()

    def approve(*args, **kwargs):
        observed.append(scope.get())
        return "once"

    session = make_session(client, approval_callback=approve)
    token = scope.set("source-turn")
    try:
        result = session.run_turn("fixture")
    finally:
        scope.reset(token)
    assert result.terminal_acknowledged
    assert observed == ["source-turn"]
    assert client.responses == [("srv-1", {"decision": "accept"})]


def test_foreign_native_approval_never_reaches_callback():
    client = FakeClient()
    decisions = []
    session = make_session(client, approval_callback=lambda *a, **kw: decisions.append(a) or "once")
    session.ensure_started()
    session._active_turn_id = "turn-fake-001"
    session._handle_server_request({"id": "foreign-approval", "method": "item/commandExecution/requestApproval",
                                    "params": {"threadId": "wrong-thread", "turnId": "turn-fake-001", "command": "echo fixture"}})
    assert decisions == []
    assert client.error_responses and not client.responses


@pytest.mark.parametrize("phase", ["before-start", "active", "after-terminal"])
def test_compaction_approval_requires_live_matching_turn(phase):
    client = FakeClient()
    decisions = []
    session = make_session(client, approval_callback=lambda *a, **kw: decisions.append(a) or "once")
    request = {"id": "stale", "method": "item/commandExecution/requestApproval",
               "params": {"threadId": "thread-fake-001", "turnId": "wrong-turn", "command": "fixture"}}
    if phase == "before-start":
        client._server_requests.append(request)

    def on_event(note):
        if note["method"] == "turn/started" and phase != "before-start":
            client._server_requests.append(request)
        if note["method"] == "turn/started" and phase == "after-terminal":
            request["params"]["turnId"] = "compact-turn"

    session._on_event = on_event
    if phase != "before-start":
        client.queue_notification("turn/started", threadId="t", turn={"id": "compact-turn"})
    if phase in {"before-start", "active"}:
        # Put completion behind the rejection, so this specifically tests the
        # wrong-turn fence rather than only the terminal-drain boundary.
        original_error = client.respond_error

        def reject(*args, **kwargs):
            original_error(*args, **kwargs)
            if phase == "before-start":
                client.queue_notification("turn/started", threadId="t", turn={"id": "compact-turn"})
            client.queue_notification("turn/completed", threadId="t", turn={"id": "compact-turn", "status": "completed"})

        client.respond_error = reject
    else:
        client.queue_notification("turn/completed", threadId="t", turn={"id": "compact-turn", "status": "completed"})
    result = session.compact_thread(turn_timeout=0.2)
    assert decisions == []
    assert client.error_responses and not client.responses
    assert result.terminal_acknowledged and not result.error
    assert not session.request_steer("stale guidance")


@pytest.mark.parametrize("mode", ["normal", "stop", "compact"])
@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("params", [
    {},
    {"threadId": "thread-fake-001", "turn": {"status": "completed"}},
    {"turn": {"id": "turn-fake-001", "status": "completed"}},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "turn": {}},
    {"threadId": "thread-fake-001", "turn": {"id": "stale", "status": "completed"}},
    {"threadId": "thread-fake-001", "turn": {"id": "turn-fake-001"}},
    {"threadId": "thread-fake-001", "turn": {"id": "turn-fake-001", "status": "inProgress"}},
    {"threadId": "thread-fake-001", "turn": {"id": "turn-fake-001", "status": []}},
    {"threadId": "thread-fake-001", "turnId": "turn-fake-001", "turn": {"id": "stale", "status": "completed"}},
    {"threadId": "thread-fake-001", "turn_id": "stale", "turn": {"id": "turn-fake-001", "status": "completed"}},
    {"threadId": "thread-fake-001", "turn": {"id": "turn-fake-001", "status": "completed"}, "item": {"turnId": "stale"}},
])
def test_terminal_authority_requires_exact_identity_and_valid_status(mode, approval, params):
    client = FakeClient()
    observed = []
    session = make_session(client)

    def on_event(note):
        observed.append(note)
        if note["method"] == "test/ready" and mode == "stop":
            session.request_interrupt()

    session._on_event = on_event
    if mode == "compact":
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
    client.queue_notification("test/ready")
    invalid = {"method": "turn/completed", "params": params}
    client._notifications.append(invalid)
    client.queue_notification("test/after-invalid")
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "interrupted" if mode == "stop" else "completed"})
    if approval:
        client.queue_server_request("item/commandExecution/requestApproval", command="obsolete", threadId="t", turnId="tu1")
    result = session.compact_thread(turn_timeout=0.2) if mode == "compact" else session.run_turn("test", turn_timeout=0.2)
    assert invalid not in observed, "malformed terminal was given projection/acknowledgement authority"
    assert any(note["method"] == "test/after-invalid" for note in observed)
    assert result.terminal_acknowledged and not result.should_retire
    assert result.interrupted == (mode == "stop")


def test_acknowledged_compaction_stop_does_not_cancel_the_next_user_turn():
    client = FakeClient()
    session = make_session(client)

    def on_event(note):
        if note["method"] == "turn/started":
            session.request_interrupt()

    session._on_event = on_event
    client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "interrupted"})
    first = session.compact_thread(turn_timeout=0.2)
    assert first.interrupted and first.terminal_acknowledged
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "completed"})
    second = session.run_turn("explicit next user turn", turn_timeout=0.2)
    assert second.terminal_acknowledged and not second.interrupted and not second.error


@pytest.mark.parametrize("compact", [False, True])
def test_failed_terminal_is_acknowledged_but_never_successful(compact):
    client = FakeClient()
    session = make_session(client)
    if compact:
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "failed"})
    result = session.compact_thread() if compact else session.run_turn("failed")
    assert result.terminal_acknowledged
    assert result.error and "failed" in result.error


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("backlog", [0, 9, 100])
def test_terminal_drain_never_prompts_obsolete_approval(compact, backlog):
    client = FakeClient()
    decisions = []
    session = make_session(client, approval_callback=lambda *a, **kw: decisions.append(a) or "once")
    client.queue_server_request("item/commandExecution/requestApproval", command="fixture", threadId="t", turnId="tu1")
    if compact:
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
    for _ in range(backlog):
        client.queue_notification("thread/status/changed", threadId="t", turnId="tu1")
    client.queue_notification("turn/completed", threadId="t", turn={"id": "tu1", "status": "completed"})
    result = session.compact_thread() if compact else session.run_turn("finished")
    assert result.terminal_acknowledged
    assert decisions == []
    assert client.error_responses and not client.responses


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("stop", [False, True])
def test_continuous_notification_drain_is_deadline_bounded(monkeypatch, compact, stop):
    client = FakeClient()
    decisions = []
    session = make_session(client, approval_callback=lambda *a, **kw: decisions.append(a) or "once")
    session.ensure_started()
    clock = [0.0]
    monkeypatch.setattr(session_mod.time, "monotonic", lambda: clock[0])

    def on_event(note):
        if stop:
            session.request_interrupt()
        clock[0] += 1
        client.queue_notification("thread/status/changed", threadId="t", turnId="tu1")

    session._on_event = on_event
    if compact:
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
    else:
        client.queue_notification("thread/status/changed", threadId="t", turnId="tu1")
    client.queue_server_request("item/commandExecution/requestApproval", command="never prompt", threadId="t", turnId="tu1")
    result = session.compact_thread(turn_timeout=100 if stop else 3) if compact else session.run_turn("flood", turn_timeout=100 if stop else 3)
    assert result.interrupted and result.should_retire and not result.terminal_acknowledged
    assert clock[0] <= (5 if stop else 3)
    assert not decisions and client.error_responses and not client.responses


# ---- lifecycle ----

class TestLifecycle:
    def test_ensure_started_is_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        tid_a = s.ensure_started()
        tid_b = s.ensure_started()
        assert tid_a == tid_b == "thread-fake-001"
        # thread/start should be called exactly once
        method_calls = [m for (m, _) in client.requests if m == "thread/start"]
        assert len(method_calls) == 1

    def test_thread_start_passes_cwd_only(self):
        """thread/start carries cwd. We intentionally do NOT pass `permissions`
        on this codex version (experimentalApi-gated + requires matching
        config.toml [permissions] table). Letting codex use its default
        (read-only unless user configures otherwise) is the documented path."""
        client = FakeClient()
        s = make_session(client, permission_profile="workspace-write")
        s.ensure_started()
        method, params = next(r for r in client.requests if r[0] == "thread/start")
        assert params["cwd"] == "/tmp"
        assert "permissions" not in params  # see session.ensure_started() comment

    def test_close_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        s.close()
        s.close()
        assert client._closed is True


# ---- turn loop ----

class TestRunTurn:
    def test_simple_text_turn_returns_final_message(self):
        client = FakeClient()
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello world"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.final_text == "hello world"
        assert r.interrupted is False
        assert r.error is None
        assert any(m["role"] == "assistant" and m.get("content") == "hello world"
                   for m in r.projected_messages)
        # turn_id propagated for downstream session-DB linkage
        assert r.turn_id == "turn-fake-001"



    def test_result_records_the_exact_submitted_input(self):
        client = FakeClient()
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        rich_input = [
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = make_session(client).run_turn(rich_input, turn_timeout=2.0)
        _, params = next(request for request in client.requests if request[0] == "turn/start")
        assert result.submitted_user_text == params["input"][0]["text"]
        assert result.submitted_user_text != rich_input

    def test_foreign_completion_in_server_request_drain_is_ignored(self):
        """Approval draining must not project a child result into the parent."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", threadId="t", turnId="tu1",
            request_id="approval-1",
            command="pwd",
            cwd="/tmp",
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="turn-child-001",
            item={
                "type": "agentMessage",
                "id": "child-message",
                "text": "child drain summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "turn-child-001",
                "status": "completed",
                "error": None,
            },
        )

        original_respond = client.respond

        def respond_and_release_parent(request_id, response):
            original_respond(request_id, response)
            client.queue_notification(
                "item/completed",
                threadId="thread-fake-001",
                turnId="turn-fake-001",
                item={
                    "type": "agentMessage",
                    "id": "parent-message",
                    "text": "parent after approval",
                },
            )
            client.queue_notification(
                "turn/completed",
                threadId="thread-fake-001",
                turn={
                    "id": "turn-fake-001",
                    "status": "completed",
                    "error": None,
                },
            )

        client.respond = respond_and_release_parent
        session = make_session(
            client,
            request_routing=_ServerRequestRouting(auto_approve_exec=True),
        )

        result = session.run_turn("delegate then continue", turn_timeout=2.0)

        assert client.responses == [("approval-1", {"decision": "accept"})]
        assert result.final_text == "parent after approval"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent after approval"}
        ]



    def test_tool_iteration_counter_ticks(self):
        client = FakeClient()
        # Two completed exec items + one final agent message
        for i, item_id in enumerate(("ex1", "ex2"), start=1):
            client.queue_notification(
                "item/completed",
                item={
                    "type": "commandExecution", "id": item_id,
                    "command": f"cmd{i}", "cwd": "/tmp",
                    "status": "completed", "aggregatedOutput": "ok",
                    "exitCode": 0, "commandActions": [],
                },
                threadId="t", turnId="tu1",
            )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("do stuff", turn_timeout=2.0)
        assert r.tool_iterations == 2
        # Each tool item produces (assistant, tool) — 2*2 + final assistant = 5 msgs
        assert len(r.projected_messages) == 5


    def test_turn_start_failure_attaches_redacted_stderr_tail(self):
        """When codex stderr has content (non-OAuth), the tail gets attached
        to the user-facing error so config/provider problems are debuggable
        instead of just 'Internal error'. Credential-shaped values in stderr
        are redacted via agent.redact(force=True); web-URL query params pass
        through (see fix(redact): pass web URLs through unchanged)."""
        client = FakeClient()
        client.set_stderr_tail([
            "ERROR: provider auth failed",
            "Authorization: Bearer sk-live-deadbeefdeadbeef",
            "url=https://api.example.com/v1?token=querysecret12345",
        ])
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32603, message="Internal error")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start failed" in r.error
        assert "Internal error" in r.error
        # Stderr tail attached
        assert "codex stderr" in r.error
        assert "provider auth failed" in r.error
        # Credential-shaped values still redacted (sk- prefix + Bearer header)
        assert "sk-live-deadbeefdeadbeef" not in r.error
        # Non-OAuth → should NOT retire (subprocess JSON-RPC is still healthy).
        assert r.should_retire is False

    def test_turn_start_timeout_attaches_redacted_stderr_tail(self):
        """A non-OAuth TimeoutError on turn/start surfaces with codex stderr
        context attached and marks the session for retirement."""
        client = FakeClient()
        client.set_stderr_tail([
            "WARN: provider request stalled",
            "Authorization: Bearer sk-stalled-secret-abc123",
        ])

        def stall(method, params):
            if method == "turn/start":
                raise TimeoutError("codex method 'turn/start' timed out after 10s")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = stall
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start timed out" in r.error
        assert "provider request stalled" in r.error
        assert "sk-stalled-secret-abc123" not in r.error
        assert r.should_retire is True




    def test_steer_appends_input_to_active_turn(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        with s._active_turn_lock:
            s._active_turn_id = "turn-live-123"

        assert s.request_steer("Use Postgres instead") is True
        method, params = client.requests[-1]
        assert method == "turn/steer"
        assert params == {
            "threadId": "thread-fake-001",
            "input": [{"type": "text", "text": "Use Postgres instead"}],
            "expectedTurnId": "turn-live-123",
        }







class TestCompactThread:
    def test_compact_thread_sends_rpc_and_waits_for_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "contextCompaction", "id": "compact-item-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "agentMessage", "id": "m1", "text": "compacted"},
        )
        client.queue_notification(
            "thread/tokenUsage/updated",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            tokenUsage={
                "last": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                "total": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                "modelContextWindow": 200000,
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1", "status": "completed", "error": None},
        )

        r = make_session(client).compact_thread(turn_timeout=2.0)

        assert ("thread/compact/start", {"threadId": "thread-fake-001"}) in client.requests
        assert r.error is None
        assert r.thread_id == "thread-fake-001"
        assert r.turn_id == "compact-turn-1"
        assert r.compacted is True
        assert r.final_text == "compacted"
        assert r.token_usage_last["totalTokens"] == 12
        assert r.model_context_window == 200000

    def test_compact_thread_ignores_foreign_child_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-child-001",
            turn={"id": "child-compact-turn"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="child-compact-turn",
            item={
                "type": "agentMessage",
                "id": "child-compact-message",
                "text": "child compact summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "child-compact-turn",
                "status": "completed",
                "error": None,
            },
        )
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={
                "type": "agentMessage",
                "id": "parent-compact-message",
                "text": "parent compacted",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={
                "id": "compact-turn-1",
                "status": "completed",
                "error": None,
            },
        )

        result = make_session(client).compact_thread(turn_timeout=2.0)

        assert result.error is None
        assert result.turn_id == "compact-turn-1"
        assert result.final_text == "parent compacted"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent compacted"}
        ]





# ---- approval bridge ----

class TestServerRequestRouting:



    def test_unknown_server_request_replied_with_error(self):
        client = FakeClient()
        client.queue_server_request("totally/unknown", threadId="t", turnId="tu1", request_id="req-3")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert any(
            rid == "req-3" and code == -32601
            for (rid, code, _msg) in client.error_responses
        )

    def test_on_event_fires_during_approval_drain(self):
        """When a server-initiated approval request arrives, the session
        drains up to 8 pending notifications first so per-turn state
        (e.g. _pending_file_changes for fileChange approvals) is current.
        Those drained notifications must also reach the on_event display
        hook — otherwise tool bubbles around approvals silently disappear.

        Regression for the issue where item/started events that landed
        in the queue alongside (or just before) an approval request got
        projected into messages but never displayed.
        """
        client = FakeClient()
        # An item/started notification is queued first, then a server
        # request — the session sees both during a single drain loop.
        client.queue_notification(
            "item/started",
            item={
                "type": "commandExecution",
                "id": "exec-1",
                "command": "echo drained",
                "cwd": "/tmp",
            },
        )
        client.queue_server_request(
            "item/commandExecution/requestApproval", threadId="t", turnId="tu1", request_id="req-d",
            command="echo drained",
            cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        events: list[dict] = []

        def cb(command, description, *, allow_permanent=True):
            return "once"

        s = make_session(
            client,
            approval_callback=cb,
            on_event=events.append,
        )
        s.run_turn("hi", turn_timeout=1.0)

        # The on_event hook must have seen the item/started even though
        # it was drained as part of the approval roundtrip — not just
        # events that arrive on the main notification path.
        item_started_events = [
            e for e in events
            if e.get("method") == "item/started"
        ]
        assert item_started_events, (
            "item/started drained alongside the approval was not "
            "forwarded to on_event — display will miss tool bubbles "
            "around approvals"
        )



    def test_routing_auto_approve_bypass(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", threadId="t", turnId="tu1", request_id="r1",
                                    command="ls", cwd="/")
        client.complete_after_response()
        # No callback, but routing says auto-approve. Should approve.
        s = make_session(client, request_routing=_ServerRequestRouting(
            auto_approve_exec=True))
        s.run_turn("hi", turn_timeout=1.0)
        assert ("r1", {"decision": "accept"}) in client.responses



# ---- enriched approval prompts ----

class TestApprovalPromptEnrichment:
    """Quirk #4: apply_patch prompt should show what's changing.
    Quirk #10: exec prompt should never show empty cwd."""

    def test_exec_falls_back_to_session_cwd(self):
        """When codex omits cwd from the approval params, the prompt shows
        the session cwd, not an empty string."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", threadId="t", turnId="tu1", request_id="r1",
            command="ls",  # no cwd
        )
        client.complete_after_response()
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Session cwd is /tmp by default in make_session()
        assert "/tmp" in captured["description"]
        assert "Codex requests exec in <unknown>" not in captured["description"]

    def test_apply_patch_prompt_summarizes_pending_changes(self):
        """When the projector has cached the fileChange item from item/started,
        the approval prompt surfaces the change summary."""
        client = FakeClient()
        # item/started fires first (carries the changes), then approval request
        client.queue_notification(
            "item/started",
            item={"type": "fileChange", "id": "fc-1",
                  "changes": [
                      {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                      {"kind": {"type": "update"}, "path": "/tmp/old.py"},
                  ]},
            threadId="t", turnId="tu1",
        )
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-1", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="add and update files",
        )
        client.complete_after_response()
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Both add and update kinds should be in the summary
        assert "1 add" in captured["command"] or "1 add" in captured["description"]
        assert "1 update" in captured["command"] or "1 update" in captured["description"]
        # And at least one of the paths
        joined = captured["command"] + " " + captured["description"]
        assert "/tmp/new.py" in joined or "/tmp/old.py" in joined

    def test_apply_patch_prompt_works_without_cached_summary(self):
        """When approval arrives before item/started (or without changes
        info), prompt falls back to whatever codex provided."""
        client = FakeClient()
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-orphan", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="apply some changes",
        )
        client.complete_after_response()
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Falls back to the reason
        assert "apply some changes" in captured["command"]


# ---- openclaw beta.8 parity: retire/wedge/oauth/abort marker ----

class TestSessionRetirement:
    """Mirrors openclaw beta.8's resilience fixes:
      - retire timed-out app-server clients (should_retire on deadline)
      - post-tool completion watchdog (don't burn the full deadline after a
        tool result if codex goes silent)
      - <turn_aborted> raw marker as terminal (don't wait for turn/completed
        that never comes)
      - OAuth refresh failure classification (suggest `codex login` instead
        of raw RPC error strings)
      - dead subprocess detection between iterations
    """



    def test_final_agent_message_without_turn_completed_remains_uncertain(self):
        """Displayable text does not prove the native tool loop has stopped."""
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "hi",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == "done"
        assert r.interrupted is True
        assert r.error is not None
        assert r.should_retire is True
        assert r.terminal_acknowledged is False
        assert any(
            msg["role"] == "assistant" and msg.get("content") == "done"
            for msg in r.projected_messages
        )
        assert any(method == "turn/interrupt" for method, _ in client.requests)


    def test_post_tool_watchdog_uses_monotonic_clock(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        clock = [1000.0]
        original_poll = client.take_notification

        def poll(timeout=0):
            clock[0] += 0.2
            return original_poll(timeout=0)

        client.take_notification = poll
        with patch.object(session_mod.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(session_mod.time, "time", return_value=999.0):
            r = s.run_turn(
                "tool then silence",
                turn_timeout=5.0,
                notification_poll_timeout=0.0,
                post_tool_quiet_timeout=0.15,
            )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "silent" in r.error

    def test_post_tool_watchdog_resets_on_further_activity(self):
        """A tool completion followed by an agent message should NOT trip
        the watchdog — further activity = codex still alive."""
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        # Non-tool activity immediately after — resets watchdog.
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "tool finished"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            "tool then talk", turn_timeout=2.0,
            notification_poll_timeout=0.01,
            post_tool_quiet_timeout=0.05,
        )
        # Tool ran, then text reset the watchdog, then turn/completed.
        # Should NOT be a retirement case.
        assert r.tool_iterations == 1
        assert r.final_text == "tool finished"
        assert r.should_retire is False
        assert r.interrupted is False







    def test_dead_subprocess_detected_between_iterations(self):
        """If codex dies (segfault, OOM, killed by its auth refresh
        thread), the inter-iteration is_alive check breaks the loop
        instead of waiting on a queue that will never fill."""
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        # Simulate subprocess death by setting _closed (FakeClient's
        # is_alive returns False when closed).
        client._closed = True
        client.set_stderr_tail([
            "thread 'tokio-runtime-worker' panicked at 'oauth: invalid_grant'",
        ])
        r = s.run_turn("x", turn_timeout=2.0,
                       notification_poll_timeout=0.01)
        assert r.should_retire is True
        # Stderr-derived auth hint takes precedence over generic message
        assert r.error and "codex login" in r.error


# ---- thread/start cross-fill ----

class TestThreadStartCrossFill:
    """Mirrors openclaw beta.8's tolerance for thread.id/sessionId aliasing."""

    def test_thread_id_under_thread_key(self):
        client = FakeClient()
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "thread-fake-001"



    def test_missing_thread_id_raises(self):
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"thread": {}, "activePermissionProfile": {"id": "x"}}
            if method == "thread/start" else
            {"turn": {"id": "tu1"}}
        )
        s = make_session(client)
        with pytest.raises(CodexAppServerError, match="no thread id"):
            s.ensure_started()


class TestHasTurnAbortedMarker:
    """Unit coverage for the marker matcher itself."""

    def test_empty_string(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("") is False
        assert _has_turn_aborted_marker(None) is False  # type: ignore[arg-type]

    def test_plain_text_no_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("normal response with no markers") is False

    def test_open_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("blah <turn_aborted> blah") is True



class TestClassifyOAuthFailure:
    """Unit coverage for the OAuth classifier; conservative on purpose."""



    def test_401_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("HTTP 401 Unauthorized")
        assert hint is not None


    def test_empty_inputs(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure() is None
        assert _classify_oauth_failure("") is None
        assert _classify_oauth_failure("", None) is None  # type: ignore[arg-type]
