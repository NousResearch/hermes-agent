"""Review follow-ups for the classic CLI's wait on a restarting gateway: a retry re-sends the first
attempt's exact request, Ctrl+C stops in-flight waits, local errors are not waited out, and a -q
cancel exits 130."""
import asyncio
import threading
import types
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_client import DISCONNECTED, GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_owner_restart import SuccessorWaitStopped, wait_for_successor
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_cli_owner_restart_wait import _host, _snapshot


class Owner:
    """Refuses while draining, then commits a request but loses its reply once, then answers."""

    def __init__(self):
        self.events = asyncio.Queue()
        self.seen = []
        self.on_create = None

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "runtime.describe":
            return {"session_create": {"parameters": []}}
        if method == "session.resume":
            self.seen.append((method, params))
            return _snapshot(params["session_id"])
        self.seen.append((method, params))
        attempt = sum(1 for m, _ in self.seen if m == method)
        if attempt == 1:
            raise GatewayRPCError("runtime_draining", -32000, {"reason": "runtime_draining"})
        if attempt == 2:
            raise GatewayClientError(DISCONNECTED)  # committed by the successor, reply lost
        if self.on_create is not None:
            await self.on_create()
        if method == "session.mutate":
            return {"branched_session_id": "local-child"}
        return _snapshot("local-" + params["request_id"])


@pytest.fixture
def owner(monkeypatch):
    import hermes_cli.gateway_owner_restart as restart
    monkeypatch.setattr(restart, "restart_wait_budget", lambda: 30.0)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.01)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    client = Owner()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def _sent(client, method):
    return [params for m, params in client.seen if m == method]


def test_new_session_retry_resends_the_same_request(owner):
    """A retry after a lost reply must carry the first attempt's ``request_id``: the owner then
    answers with the session that attempt committed instead of creating a second one."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    store, client = owner
    CLIInitMixin._new_session_id(_host(store), None)
    ids = {params["request_id"] for params in _sent(client, "session.create")}
    assert len(_sent(client, "session.create")) == 3 and len(ids) == 1


def test_branch_retry_resends_the_same_fenced_mutation(owner):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    store, client = owner
    host = _host(store)
    assert CLIInitMixin._new_session_id(host, None, branch_of="local-parent") == "local-child"
    mutations = _sent(client, "session.mutate")
    assert len(mutations) == 3 and all(m == mutations[0] for m in mutations)


def test_ctrl_c_during_an_in_flight_attempt_opens_nothing(owner):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    store, client = owner
    host = _host(store)
    host._gateway_command_wait = None

    never = asyncio.Event()  # the successor never answers this attempt

    async def hung_successor():
        host._gateway_command_wait.set()  # the user presses Ctrl+C while the successor is silent
        await never.wait()
    client.on_create = hung_successor
    with pytest.raises(SuccessorWaitStopped):
        CLIInitMixin._new_session_id(host, None)
    assert host.session_id == "local-old" and store.current is None


def test_a_local_error_during_the_wait_is_raised_at_once():
    calls = []

    def attempt(_remaining):
        calls.append(1)
        if len(calls) == 1:
            raise GatewayRPCError("runtime_draining")
        raise FileNotFoundError("launch directory removed")
    with pytest.raises(FileNotFoundError):
        wait_for_successor(attempt, announce=lambda: None, budget=60)
    assert len(calls) == 2


def test_quiet_query_cancel_exits_130(monkeypatch, capsys):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    def stopped():
        raise SuccessorWaitStopped("Stopped waiting for Hermes to restart; nothing was started.")
    host = types.SimpleNamespace(agent=None, finalize_preloaded_skills=lambda: None,
                                 _install_tool_callbacks=lambda: None, _ensure_runtime_credentials=lambda: True,
                                 _session_db=object(), _ensure_gateway_session=stopped, _single_query_mode=True)
    with pytest.raises(SystemExit) as exited:
        CLIAgentSetupMixin._init_agent(host)
    assert exited.value.code == 130
    assert "Stopped waiting" in capsys.readouterr().err
