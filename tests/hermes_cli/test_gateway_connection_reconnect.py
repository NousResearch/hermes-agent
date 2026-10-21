"""The classic CLI dials the gateway again after it restarts (#134889).

GatewayConnection kept its finished loop thread, so after a gateway restart every call raised
"Gateway disconnected" and the CLI never re-attached, even once another client had started a new
gateway. Each prompt then failed with copy that read as a model failure ("Run /model to switch").
"""
import asyncio
import threading
import types
from contextlib import asynccontextmanager

import pytest

from hermes_cli.cli_chat_error_copy import chat_error_response
from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection, lost_before

SID = "local-abc"


def _frame(kind, payload):
    return {"jsonrpc": "2.0", "method": "event", "params": {
        "type": kind, "session_id": SID, "admission_id": "adm-1", "execution_generation": 7, "payload": payload}}


class Life:
    """One gateway process as one client sees it: from a dial until it hangs up."""

    def __init__(self, number, *, drop_on_submit=False, resume_errors=None):
        self.number = number
        self.events = asyncio.Queue()
        self.calls = []
        self.drop_on_submit = drop_on_submit
        self.resume_errors = resume_errors if resume_errors is not None else []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.resume" and self.resume_errors:
            raise self.resume_errors.pop(0)
        if method == "session.resume":
            return {"stored_session_id": params["session_id"], "subscription_id": f"sub-{self.number}",
                    "revision": 1, "execution_generation": 7, "messages": []}
        if method == "prompt.submit":
            loop = asyncio.get_running_loop()
            if self.drop_on_submit:  # restarted mid-turn: the outcome is unknown to this client
                loop.call_soon(self.events.put_nowait, GatewayClientError("Gateway disconnected"))
            else:
                loop.call_soon(self.events.put_nowait, _frame(
                    "message.complete", {"text": f"reply {self.number}", "outcome": "completed", "status": "complete"}))
            return {"admission_id": "adm-1", "status": "queued", "execution_generation": 7}
        if method == "prompt.receipt":
            return {"result": {"final_response": f"reply {self.number}", "completed": True}, "usage": {}}
        if method in ("prompt.background", "prompt.btw"):
            return {"task_id": f"task-{self.number}"}
        return {"status": "ok", "life": self.number}


class Gateway:
    """Each dial reaches a fresh gateway life; ``refusals`` makes the next dials fail first."""

    def __init__(self):
        self.lives = []
        self.refusals = 0
        self.drop_next_submit = False
        self.resume_errors = []  # raised by the next session.resume calls, across lives
        self.dialing = threading.Event()
        self.hold = None  # an Event the next dials wait on before the socket opens

    @asynccontextmanager
    async def dial(self):
        self.dialing.set()
        if self.hold is not None:
            await asyncio.to_thread(self.hold.wait, 5)
        if self.refusals:
            self.refusals -= 1
            raise GatewayClientError("Gateway starting: deadline. Your message was not sent.")
        life = Life(len(self.lives) + 1, drop_on_submit=self.drop_next_submit, resume_errors=self.resume_errors)
        self.drop_next_submit = False
        self.lives.append(life)
        yield life

    def hang_up(self, conn):
        """The current life goes away (``gateway restart``); the loop thread ends with it."""
        thread = conn._thread
        conn.loop.call_soon_threadsafe(self.lives[-1].events.put_nowait, GatewayClientError("Gateway disconnected"))
        thread.join(5)
        assert not thread.is_alive()


@pytest.fixture
def gateway():
    return Gateway()


@pytest.fixture
def conn(gateway):
    connection = GatewayConnection(connect=gateway.dial)
    connection.start(5)
    connection.state(SID).adopt_snapshot(connection.call("session.resume", session_id=SID))
    yield connection
    connection.close()


def test_the_next_call_after_a_restart_reconnects_and_resubscribes(gateway, conn):
    assert conn.epoch == 1
    gateway.hang_up(conn)

    assert conn.call("status") == {"status": "ok", "life": 2}
    assert conn.epoch == 2
    # Resubscribed before the reconnect counts as ready, so no frame of the new life is missed.
    assert gateway.lives[1].calls[0] == ("session.resume", {"session_id": SID})
    assert conn.state(SID).subscription_id == "sub-2"
    lost = conn.events.get_nowait()
    assert isinstance(lost, GatewayClientError) and lost_before(lost, conn.epoch)


def test_close_ends_the_viewer_for_good(gateway, conn):
    conn.close()
    with pytest.raises(GatewayClientError, match="disconnected"):
        conn.call("status")
    assert len(gateway.lives) == 1


def test_a_failed_redial_is_retried_on_the_next_call(gateway, conn):
    gateway.hang_up(conn)
    gateway.refusals = 1
    with pytest.raises(GatewayClientError, match="Gateway starting"):
        conn.call("status")
    assert conn.call("status") == {"status": "ok", "life": 2}


def test_a_failed_redial_is_retried_even_while_its_loop_thread_is_still_exiting(gateway, conn):
    """The failed dial's loop thread reports ready before it returns; the next call must not read
    that still-alive thread as a dial in progress and re-raise its stale error."""
    gateway.hang_up(conn)
    gateway.refusals = 1
    released = threading.Event()
    run = conn._run

    def run_then_linger():
        run()
        released.wait(5)  # still alive after ``_ready.set()``, as on a loaded runner

    conn._run = run_then_linger
    with pytest.raises(GatewayClientError, match="Gateway starting"):
        conn.call("status")
    old = conn._thread
    assert old.is_alive()
    join = old.join

    def release_then_join(timeout=None):  # only the fixed path joins the exiting thread
        released.set()
        join(timeout)

    old.join = release_then_join
    try:
        assert conn.call("status") == {"status": "ok", "life": 2}
    finally:
        released.set()


def test_a_first_connect_failure_is_still_final(gateway):
    gateway.refusals = 2
    connection = GatewayConnection(connect=gateway.dial)
    for _ in range(2):
        with pytest.raises(GatewayClientError, match="Gateway starting"):
            connection.call("status")
    assert gateway.refusals == 1  # never redialed: nothing to come back to


def test_a_turn_lost_to_a_restart_is_reported_plainly_and_the_next_turn_runs(gateway, conn):
    agent = GatewayAgent(connection=conn, session_id=SID)
    gateway.lives[0].drop_on_submit = True

    lost = agent.run_conversation("first")
    assert lost["failed"] and lost["failure_reason"] == "gateway_connection"
    assert "turn outcome is unknown" in lost["error"] or "Gateway disconnected" in lost["error"]
    # The panel shows the transport error itself, not "The model request failed. Run /model ...".
    assert chat_error_response(lost["error"], failure_reason=lost["failure_reason"]) == lost["error"]

    conn._thread.join(5)
    result = agent.run_conversation("second")
    assert not result.get("failed") and result["final_response"] == "reply 2"
    assert [method for method, _ in gateway.lives[1].calls][:2] == ["session.resume", "prompt.submit"]


def test_the_previous_connections_loss_never_fails_a_live_turn(gateway, conn):
    """Idle when the gateway restarted: the loss is queued, and the next prompt reconnects and
    submits. Reading that loss as its own outcome would invite the user to resend accepted work."""
    agent = GatewayAgent(connection=conn, session_id=SID)
    gateway.hang_up(conn)
    result = agent.run_conversation("hi")
    assert not result.get("failed") and result["final_response"] == "reply 2"


def _resumes(life):
    return [params["session_id"] for method, params in life.calls if method == "session.resume"]


@pytest.fixture
def store(conn):
    from hermes_cli.gateway_session_store import GatewaySessionStore
    db = GatewaySessionStore(conn)
    db.attach(SID)
    return db


def test_ending_a_session_after_a_reconnect_detaches_its_current_subscription(gateway, conn, store):
    """The redial resubscribed under a new id; detaching with the first one is refused gateway-side,
    leaving a closed session attached and its close bookkeeping skipped."""
    gateway.hang_up(conn)
    conn.call("status")
    store.end_session(SID, "new_session")
    assert gateway.lives[1].calls[-1] == ("session.detach", {
        "session_id": SID, "subscription_id": "sub-2", "end_reason": "new_session"})
    # Left by this viewer: a later reconnect does not resume it again.
    gateway.hang_up(conn)
    conn.call("status")
    assert _resumes(gateway.lives[2]) == []


def test_a_deleted_session_is_not_resumed_by_later_reconnects(gateway, conn, store):
    gateway.hang_up(conn)
    conn.call("status")
    store.delete_session(SID)
    gateway.hang_up(conn)
    conn.call("status")
    assert _resumes(gateway.lives[2]) == []


def test_a_session_gone_during_the_restart_is_skipped_and_not_retried(gateway, conn):
    gateway.resume_errors.append(GatewayClientError("not_found"))
    gateway.hang_up(conn)
    assert conn.call("status") == {"status": "ok", "life": 2}
    assert conn.state(SID).subscription_id is None
    gateway.hang_up(conn)
    assert conn.call("status") == {"status": "ok", "life": 3}
    assert _resumes(gateway.lives[2]) == []


@pytest.mark.parametrize("failure", [GatewayClientError("storage_unavailable"), TimeoutError()],
                         ids=["storage_unavailable", "timeout"])
def test_a_transient_resume_failure_fails_the_reconnect_and_the_next_call_retries(gateway, conn, failure):
    """Ready without the subscription, every later prompt.submit is refused (permission_denied) on a
    live socket that never redials: the reconnect must fail instead, so the next call dials again."""
    gateway.resume_errors.append(failure)
    gateway.hang_up(conn)
    with pytest.raises(GatewayClientError):
        conn.call("status")
    assert conn.call("status") == {"status": "ok", "life": 3}
    assert _resumes(gateway.lives[2]) == [SID]
    assert conn.state(SID).subscription_id == "sub-3"


def test_close_after_a_failed_resume_is_still_final(gateway, conn):
    gateway.resume_errors.append(GatewayClientError("storage_unavailable"))
    gateway.hang_up(conn)
    with pytest.raises(GatewayClientError):
        conn.call("status")
    conn.close()
    with pytest.raises(GatewayClientError):
        conn.call("status")
    assert len(gateway.lives) == 2


class _Unanswered:
    """A socket that takes every request and never answers one (a gateway stalled mid-reconnect)."""

    def __init__(self):
        self.sent = asyncio.Event()

    async def send(self, raw):
        self.sent.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await asyncio.Future()


@pytest.mark.asyncio
async def test_a_cancelled_reconnect_cancels_its_session_resume():
    """The viewer closes while a reconnect's ``session.resume`` goes unanswered: cancelling the
    resubscribe cancels the request with it. One left pending on the client held a worker thread
    that process exit joins until the request timed out."""
    from hermes_cli.gateway_client import GatewayClient
    conn = GatewayConnection()
    conn.state(SID).subscription_id = "sub-1"
    socket = _Unanswered()
    async with GatewayClient(socket) as client:
        resubscribe = asyncio.create_task(conn._resubscribe(client))
        await asyncio.wait_for(socket.sent.wait(), 5)
        resubscribe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resubscribe
        assert client.pending == {}


def _cli(store):
    return types.SimpleNamespace(_session_db=store, _gateway_session_open=True)


@pytest.fixture
def tools(conn, monkeypatch):
    from hermes_cli import gateway_tools
    monkeypatch.setattr(gateway_tools, "_view", None)
    gateway_tools.bind(types.SimpleNamespace(connection=conn, session_id=SID))
    return gateway_tools


def test_a_hung_up_store_stays_open_and_its_next_call_redials(gateway, conn, store):
    """/sessions, /resume, /branch and /handoff gate on the store: a hang-up is not "Hermes can't
    open its session history right now"; the next call dials the restarted gateway."""
    gateway.hang_up(conn)
    assert store
    assert store.call("status") == {"status": "ok", "life": 2}
    assert store


def test_bg_and_btw_after_a_hang_up_run_on_the_gateway_not_in_this_process(gateway, conn, store):
    """None here sends /bg down the in-process AIAgent path (a local ``bg_HHMMSS_xxxxxx`` task)."""
    from hermes_cli import gateway_side
    gateway.hang_up(conn)
    assert gateway_side.connection_for(_cli(store)) is conn
    task_id, _wait = gateway_side.start(conn, SID, "background", "later")
    assert task_id == "task-2" and gateway.lives[1].calls[-1][0] == "prompt.background"


def test_a_side_answer_lost_to_a_restart_fails_instead_of_waiting_forever(gateway, conn):
    """The task died with the old gateway; another thread's redial must not make the wait endless."""
    from hermes_cli import gateway_side
    _task, wait = gateway_side.start(conn, SID, "btw", "q")
    gateway.hang_up(conn)
    conn.call("status")
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(pytest.raises(RuntimeError, wait)), daemon=True)
    waiter.start()
    waiter.join(10)
    assert not waiter.is_alive() and "outcome is unknown" in str(outcome[0].value)


def test_a_redial_between_admission_and_the_waiter_still_ends_the_wait(gateway, conn, monkeypatch):
    """Another caller redials after the old gateway admitted the task but before its waiter exists:
    the wait belongs to the admitting life, not to whatever epoch is current by then."""
    from hermes_cli import gateway_side
    register = conn.side_waiter

    def redial_first(task_id):
        gateway.hang_up(conn)
        conn.call("status")
        return register(task_id)

    monkeypatch.setattr(conn, "side_waiter", redial_first)
    task, wait = gateway_side.start(conn, SID, "btw", "q")
    assert task == "task-1" and conn.epoch == 2
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(pytest.raises(RuntimeError, wait)), daemon=True)
    waiter.start()
    waiter.join(10)
    assert not waiter.is_alive() and "outcome is unknown" in str(outcome[0].value)


def test_the_dock_and_stop_after_a_hang_up_redial(gateway, conn, tools):
    gateway.hang_up(conn)
    view = tools.process_view()
    assert view is not None and tools.process_registry() is view
    view.kill_all()
    assert [method for method, _ in gateway.lives[1].calls][:2] == ["session.resume", "process.stop"]


def test_shared_keeps_a_hung_up_connection_instead_of_opening_a_second(gateway, conn, monkeypatch):
    monkeypatch.setattr(GatewayConnection, "_shared", conn)
    gateway.hang_up(conn)
    assert GatewayConnection.shared() is conn


def test_new_after_a_hang_up_redials_and_detaches_with_the_end_reason(gateway, conn, store):
    """Dropping the subscription without ``session.detach`` leaves the owner never ending the row."""
    gateway.hang_up(conn)
    store.end_session(SID, "new_session")
    assert gateway.lives[1].calls == [("session.resume", {"session_id": SID}), ("session.detach", {
        "session_id": SID, "subscription_id": "sub-2", "end_reason": "new_session"})]
    gateway.hang_up(conn)
    conn.call("status")
    assert _resumes(gateway.lives[2]) == []


def test_after_close_every_reader_reports_the_connection_gone(gateway, conn, store, tools, monkeypatch):
    from hermes_cli import gateway_side
    monkeypatch.setattr(GatewayConnection, "_shared", conn)
    gateway.hang_up(conn)
    conn.close()
    assert not store
    assert gateway_side.connection_for(_cli(store)) is None
    assert tools.process_view() is None
    assert GatewayConnection.shared() is not conn
    store.end_session(SID, "new_session")
    with pytest.raises(GatewayClientError, match="disconnected") as refused:
        conn.call("status")
    assert "outcome is unknown" not in str(refused.value)  # nothing was sent
    assert len(gateway.lives) == 1


class _Waiters(threading.Event):
    """``_ready`` that opens ``release`` once ``count`` callers are waiting on it."""

    def __init__(self, release, count):
        super().__init__()
        self.release, self.count = release, count

    def wait(self, timeout=None):
        self.count -= 1
        if self.count <= 0:
            self.release.set()
        return super().wait(timeout)


def test_a_call_racing_a_redial_waits_for_it_and_is_sent(gateway, conn):
    """The second caller saw the dialing thread alive and no client yet: "turn outcome is unknown, do
    not resend" for a request that was never sent."""
    gateway.hang_up(conn)
    gateway.hold, gateway.dialing = threading.Event(), threading.Event()
    conn._ready = _Waiters(gateway.hold, 2)
    results = []
    first = threading.Thread(target=lambda: results.append(conn.call("status")), daemon=True)
    first.start()
    assert gateway.dialing.wait(5)
    try:
        second = conn.call("status")
    finally:
        gateway.hold.set()
    first.join(10)
    assert second == {"status": "ok", "life": 2} and results == [second]
    assert len(gateway.lives) == 2


def test_lost_before():
    error = GatewayClientError("Gateway disconnected")
    assert not lost_before(error, 2)  # untagged: a live loss
    error.connection_epoch = 1
    assert lost_before(error, 2) and not lost_before(error, 1) and not lost_before(error, None)
    assert not lost_before({"connection_epoch": 1}, 2)


def test_chat_q_says_why_a_turn_was_interrupted(capsys):
    """A restart drain settles a running `chat -q` turn interrupted with no reply text: exit 130
    as before, but no longer silently."""
    from hermes_cli.gateway_chat_view import GatewayChatView

    class Client:
        def __init__(self):
            self.events = asyncio.Queue()

        async def rpc(self, method, **params):
            if method == "prompt.submit":
                asyncio.get_running_loop().call_soon(self.events.put_nowait, _frame(
                    "message.complete", {"text": "", "outcome": "cancelled", "status": "interrupted"}))
                return {"admission_id": "adm-1", "status": "queued", "execution_generation": 7}
            if method == "prompt.receipt":
                return {"result": {"final_response": "", "interrupted": True,
                                   "interrupt_message": "Gateway shutting down"}}
            return {}

    code = asyncio.run(GatewayChatView(Client(), {"stored_session_id": SID}).run("hi", oneshot=True))
    assert code == 130
    assert "Turn interrupted (Gateway shutting down); it was not retried." in capsys.readouterr().err
