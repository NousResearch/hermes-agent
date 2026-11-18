"""A Kanban worker whose gateway socket dies mid-turn resumes the same attempt on a new connection.

A dead socket is a transport event, not a turn outcome: the owner keeps the admission, answers
``prompt.receipt`` over the next connection and returns the same admission for a repeated
``kanban.run`` (one request id per attempt). Exiting 1 instead loses work that was already running,
because ``_watch_submitter`` then ends the owner's in-flight turn (#145062). An owner's own refusal
is still an answer, and an owner that stays silent still fails the worker.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from websockets.exceptions import InvalidMessage

from hermes_cli import gateway_client, kanban_worker_client
from hermes_cli.gateway_runtime_discovery import DiscoveryError

GUE = gateway_client.GatewayUnavailableError


class _Clock:
    """Monotonic time that moves only when the client sleeps or a scripted step advances it."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class _Owner:
    """Scripted owner. ``connects`` holds one step per dial, ``receipts`` one per ``prompt.receipt``
    and ``submits`` one per ``kanban.run``; a step is None (succeed), an exception (raised) or a
    callable (run first, so it can advance the clock or raise). ``after`` is the step for every
    dial past the scripted ones (None: the dial succeeds)."""

    def __init__(self, connects=(), receipts=(), submits=(), after=None):
        self.connects, self.receipts, self.submits = list(connects), list(receipts), list(submits)
        self.after = after
        self.calls = []
        self.dials = 0

    @staticmethod
    def _step(steps, default=None):
        step = steps.pop(0) if steps else default
        if callable(step):
            step = step()
        if isinstance(step, Exception):
            raise step
        return step

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect():
            owner.dials += 1
            owner._step(owner.connects, owner.after)

            class _Client:
                def __init__(self):
                    self.events = asyncio.Queue()

                async def rpc(self, method, **params):
                    owner.calls.append(method)
                    if method == "kanban.run":
                        owner._step(owner.submits)
                        return {"session_id": "s1", "receipt": {"status": "pending", "admission_id": "a1"}}
                    assert params == {"session_id": "s1", "admission_id": "a1"}
                    return {"status": owner._step(owner.receipts, "terminal")}

            yield _Client()

        return _connect


def _disconnect():
    return gateway_client.GatewayClientError(
        "Gateway disconnected; turn outcome is unknown. Resume the printed session ID")


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(kanban_worker_client, "time", clock)

    async def _sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _sleep)
    return clock


@pytest.fixture(autouse=True)
def _worker_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result",
                        lambda *_: {"exit_code": 0, "last_output": "card closed"})
    return tmp_path


async def _run(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())
    return await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))


@pytest.mark.asyncio
@pytest.mark.parametrize("redial", [
    None,
    gateway_client.GatewayClientError("Gateway connection failed; no local fallback"),
    ConnectionRefusedError(61, "refused"),
    gateway_client.GatewayUnavailableError("Gateway starting: not_ready", "starting"),
    gateway_client.GatewayUnavailableError("Gateway inaccessible: invalid_control_peer",
                                           "inaccessible", "invalid_control_peer"),
    gateway_client.GatewayUnavailableError("Gateway inaccessible: incomplete_control_response",
                                           "inaccessible", "incomplete_control_response"),
    GUE("Gateway inaccessible: invalid_control_response", "inaccessible", "invalid_control_response"),
    InvalidMessage("did not receive a valid HTTP response"),
], ids=["reconnects", "dial-refused", "ticket-oserror", "owner-starting", "control-reset", "control-eof",
        "owner-replaced", "handshake"])
async def test_a_socket_lost_mid_turn_resumes_the_same_attempt(monkeypatch, _worker_env, clock, redial):
    owner = _Owner(connects=[None, redial, None], receipts=["pending", _disconnect(), "terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    # The claim is submitted once: every later connection re-reads the receipt.
    assert owner.calls.count("kanban.run") == 1
    assert owner.dials == (2 if redial is None else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["runtime_draining", "runtime_retired"])
async def test_a_draining_or_retired_authority_mid_turn_is_asked_again(monkeypatch, _worker_env, clock, reason):
    owner = _Owner(connects=[None, None], receipts=["pending", gateway_client.GatewayRPCError(reason), "terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    assert owner.dials == 2 and owner.calls.count("kanban.run") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", [_disconnect(), gateway_client.GatewayRPCError("runtime_draining")],
                         ids=["answer-lost", "draining"])
async def test_a_claim_with_no_answer_is_resubmitted(monkeypatch, _worker_env, clock, lost):
    # No admission id reached the worker, so it asks again; the owner answers a repeated
    # kanban.run for the same attempt with the same admission.
    owner = _Owner(connects=[None, None], submits=[lost], receipts=["terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    assert owner.calls.count("kanban.run") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["kanban.run", "prompt.receipt"])
async def test_an_owner_refusal_is_not_retried(monkeypatch, _worker_env, clock, where):
    refusal = gateway_client.GatewayRPCError("claim_mismatch")
    owner = (_Owner(submits=[refusal]) if where == "kanban.run"
             else _Owner(receipts=["pending", refusal]))

    with pytest.raises(gateway_client.GatewayRPCError, match="claim_mismatch"):
        await _run(owner, monkeypatch, _worker_env)
    assert owner.dials == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("final", [
    gateway_client.GatewayUnavailableError("Gateway inaccessible: authorization", "inaccessible", "authorization"),
    GUE("Gateway inaccessible: unsafe_control_permissions", "inaccessible", "unsafe_control_permissions"),
], ids=["refused-control", "unsafe-control"])
async def test_an_owner_that_cannot_be_trusted_fails_at_once(monkeypatch, _worker_env, clock, final):
    owner = _Owner(connects=[None, final], receipts=["pending", _disconnect()])

    with pytest.raises(type(final)):
        await _run(owner, monkeypatch, _worker_env)
    assert owner.dials == 2


@pytest.mark.asyncio
async def test_a_later_loss_gets_a_fresh_budget_from_the_owners_last_answer(monkeypatch, _worker_env, clock):
    def silent_then_lost():
        clock.now += kanban_worker_client.OWNER_WAIT_S - 10
        return _disconnect()

    # Each loss comes 170 s after the last answer; across both, far more than OWNER_WAIT_S passes.
    owner = _Owner(connects=[None, None, None],
                   receipts=["pending", silent_then_lost, "pending", silent_then_lost, "terminal"])

    assert await _run(owner, monkeypatch, _worker_env) == 0
    assert owner.dials == 3 and owner.calls.count("kanban.run") == 1


@pytest.mark.asyncio
async def test_connecting_is_not_an_answer(monkeypatch, _worker_env, clock):
    # Every redial connects, but every receipt read drops: only the owner's answers refresh the
    # budget, so this still ends at OWNER_WAIT_S instead of redialling forever.
    owner = _Owner(receipts=["pending"] + [_disconnect()] * 200)

    with pytest.raises(gateway_client.GatewayClientError, match=r"owner silent for 1[89]\ds"):
        await _run(owner, monkeypatch, _worker_env)
    assert owner.calls.count("kanban.run") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    gateway_client.GatewayClientError("Gateway connection failed; no local fallback"),
    GUE("Gateway starting: deadline", "starting", "deadline"),
    GUE("Gateway inaccessible: invalid_control_peer", "inaccessible", "invalid_control_peer"),
    gateway_client.GatewayRPCError("runtime_draining"),
], ids=["refused", "starting", "control-reset", "draining"])
async def test_an_owner_silent_for_the_whole_budget_fails_the_worker(monkeypatch, _worker_env, clock, fault):
    # Every restart shape still ends at OWNER_WAIT_S when it never clears.
    owner = _Owner(connects=[None], receipts=["pending", _disconnect()], after=fault)

    with pytest.raises(gateway_client.GatewayClientError, match=rf"owner silent for 1[89]\ds: {fault}"):
        await _run(owner, monkeypatch, _worker_env)
    assert clock.now >= kanban_worker_client.OWNER_WAIT_S
    # Jittered exponential backoff (capped at 8-16 s) between dials, not one dial a second.
    assert 3 < owner.dials < 40 and owner.calls.count("kanban.run") == 1


@pytest.mark.asyncio
async def test_giving_up_on_a_refused_handshake_names_the_status(monkeypatch, _worker_env, clock):
    from types import SimpleNamespace
    from websockets.exceptions import InvalidStatus
    refused = InvalidStatus(SimpleNamespace(status_code=403, headers={}, body=b"remote body"))
    owner = _Owner(connects=[None] + [refused] * 60, receipts=["pending", _disconnect()])

    with pytest.raises(gateway_client.GatewayClientError, match=r"InvalidStatus \(HTTP 403\)$"):
        await _run(owner, monkeypatch, _worker_env)


@pytest.mark.asyncio
async def test_discovery_hands_the_worker_its_reason(monkeypatch, _worker_env):
    """The real connect_gateway carries discovery's reason, which is what tells a restarting
    owner's reset control socket apart from a foreign one."""
    from hermes_cli import gateway_runtime
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *_, **__: gateway_runtime.GatewayDiscovery(
        "inaccessible", reason_code="incomplete_control_response"))
    with pytest.raises(gateway_client.GatewayUnavailableError) as raised:
        async with gateway_client.connect_gateway():
            pass
    assert (raised.value.state, raised.value.reason_code) == ("inaccessible", "incomplete_control_response")


@pytest.mark.asyncio
async def test_a_refused_ticket_reads_like_a_refusing_owner(monkeypatch, _worker_env):
    """The ticket fetch's DiscoveryError reaches callers as the inaccessible verdict discovery gives."""
    from types import SimpleNamespace
    from hermes_cli import gateway_runtime
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *_, **__: SimpleNamespace(endpoint=None))
    monkeypatch.setattr(gateway_runtime, "attachable", lambda result: True)

    def refuse(*_):
        raise DiscoveryError("invalid_control_response")
    monkeypatch.setattr(gateway_client, "_session_ticket", refuse)
    with pytest.raises(GUE) as raised:
        async with gateway_client.connect_gateway():
            pass
    assert (raised.value.state, raised.value.reason_code) == ("inaccessible", "invalid_control_response")
