"""A codex app-server that exits between turns is respawned and its thread resumed (port of centaur#1957)."""

from agent.transports.codex_app_server_session import CodexAppServerSession


class _Client:
    def __init__(self, **kwargs):
        self.alive, self.closed, self.requests = True, False, []

    def initialize(self, **kwargs):
        return {}

    def request(self, method, params=None, timeout=30.0):
        self.requests.append((method, params or {}))
        return {"thread": {"id": (params or {}).get("threadId") or "thread-1"}}

    def is_alive(self):
        return self.alive

    def stderr_tail(self, n=20):
        return ["idle exit"]

    def close(self):
        self.closed = True


def test_dead_idle_client_is_replaced_and_the_same_thread_resumed():
    clients: list[_Client] = []
    session = CodexAppServerSession(cwd="/", client_factory=lambda **kw: clients.append(_Client()) or clients[-1])
    assert session.ensure_started() == "thread-1"

    clients[0].alive = False
    assert session.ensure_started() == "thread-1"

    assert len(clients) == 2 and clients[0].closed
    assert [m for m, _ in clients[1].requests] == ["thread/resume"]
    assert clients[1].requests[0][1]["threadId"] == "thread-1"
    # A live client is reused as before: no respawn, no extra RPC.
    assert session.ensure_started() == "thread-1" and len(clients) == 2 and len(clients[1].requests) == 1
